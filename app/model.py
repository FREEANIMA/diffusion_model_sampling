import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
[Model Overview]

Input: (B, 4, 64, 64) - VAE latent

1. PatchEmbedding
   Conv2d(patch_size=2) → flatten → transpose
   (B, 4, 64, 64) → (B, 1024 tokens, 1024 d_model)

2. Condition Embedding
   ├── Sigma → SinusoidalPosEmb → MLP → sigma_emb        # noise level (timestep)
   └── Text  → Linear → MLP → text_token_emb              # tokens for cross attention
               Text  → mean pooling → MLP → pooled_text   # global condition for adaLN

   cond_emb = sigma_emb + pooled_text → adaLN modulation coefficients

3. DiT Block × num_layers
   each block receives shift/scale modulation from cond_emb (adaLN):
   ├── Self Attention + RoPE 2D  # spatial relationships between patches
   ├── Text Cross Attention       # text tokens ↔ image patches
   └── FFN                        # feature transformation

4. Final modulation + Output projection
   LayerNorm → adaLN shift/scale → Linear → unpatchify

Output: pred_velocity (B, 4, 64, 64) - direction vector from noise → clean
'''

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, sinusoid_rope_hz):
        super().__init__()
        self.sinusoid_rope_hz = sinusoid_rope_hz
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(self.sinusoid_rope_hz) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)

        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.dim = dim
        self.rope_dim_per_coord = dim // 2

        inv_freq_h = 1.0 / (base ** (torch.arange(0, self.rope_dim_per_coord, 2).float() / self.rope_dim_per_coord))
        self.register_buffer('inv_freq_h', inv_freq_h)

        inv_freq_w = 1.0 / (base ** (torch.arange(0, self.rope_dim_per_coord, 2).float() / self.rope_dim_per_coord))
        self.register_buffer('inv_freq_w', inv_freq_w)

    def forward(self, q, k, H_p, W_p):
        t_idx = torch.arange(H_p * W_p, device=q.device)
        h_idx = t_idx // W_p
        w_idx = t_idx % W_p

        freqs_h = torch.einsum('i,j->ij', h_idx.float(), self.inv_freq_h)
        freqs_w = torch.einsum('i,j->ij', w_idx.float(), self.inv_freq_w)

        freqs_h = torch.cat((freqs_h, freqs_h), dim=-1)
        freqs_w = torch.cat((freqs_w, freqs_w), dim=-1)

        cos_cached_h = freqs_h.cos().view(1, 1, H_p * W_p, self.rope_dim_per_coord)
        sin_cached_h = freqs_h.sin().view(1, 1, H_p * W_p, self.rope_dim_per_coord)
        cos_cached_w = freqs_w.cos().view(1, 1, H_p * W_p, self.rope_dim_per_coord)
        sin_cached_w = freqs_w.sin().view(1, 1, H_p * W_p, self.rope_dim_per_coord)

        q_h, q_w = q.chunk(2, dim=-1)
        k_h, k_w = k.chunk(2, dim=-1)

        q_h_rot = (q_h * cos_cached_h) + (self._rotate_half(q_h) * sin_cached_h)
        k_h_rot = (k_h * cos_cached_h) + (self._rotate_half(k_h) * sin_cached_h)

        q_w_rot = (q_w * cos_cached_w) + (self._rotate_half(q_w) * sin_cached_w)
        k_w_rot = (k_w * cos_cached_w) + (self._rotate_half(k_w) * sin_cached_w)

        q_rot = torch.cat((q_h_rot, q_w_rot), dim=-1)
        k_rot = torch.cat((k_h_rot, k_w_rot), dim=-1)

        return q_rot, k_rot

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2).contiguous()
        return x

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor, H, W):
        B = x.shape[0]
        p = self.patch_size
        C = self.in_channels
        h, w = H // p, W // p
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, h * p, w * p)
        return x


class Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rope_hz):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model

        # [수정] gate 제거 → shift/scale 6개만
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 6)
        )

        # self attention
        self.self_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding2D(d_model // nhead, rope_hz)

        # text cross
        self.text_norm = nn.LayerNorm(d_model)
        self.text_cross_q = nn.Linear(d_model, d_model)
        self.text_cross_kv = nn.Linear(d_model, d_model * 2)
        self.text_cross_out = nn.Linear(d_model, d_model)

        # ffn
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(d_model // nhead)
        self.k_norm = nn.LayerNorm(d_model // nhead)

    def self_attention(self, x_norm, H, W):
        B, T, D = x_norm.shape
        N = self.nhead
        d_k = D // N

        qkv = self.qkv(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.reshape(B, T, N, d_k).transpose(1, 2)
        K = K.reshape(B, T, N, d_k).transpose(1, 2)
        V = V.reshape(B, T, N, d_k).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q_rot, K_rot = self.rope(Q, K, H, W)

        attn_out = F.scaled_dot_product_attention(
            Q_rot, K_rot, V,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)
        return attn_out

    def _cross_attention_impl(self, x_norm, cond, q_proj, kv_proj, out_proj, H, W, attn_mask=None):
        B, T, D = x_norm.shape
        Bc, L, Dc = cond.shape

        Q = q_proj(x_norm)
        kv = kv_proj(cond)
        K, V = kv.chunk(2, dim=-1)

        N = self.nhead
        d_k = D // N

        Q = Q.reshape(B, T, N, d_k).transpose(1, 2)
        K = K.reshape(B, L, N, d_k).transpose(1, 2)
        V = V.reshape(B, L, N, d_k).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=Q.device, dtype=torch.bool)
            attn_mask = attn_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).reshape(B, T, D)
        out = out_proj(out)
        return out

    def text_cross_attention(self, x_norm, text_emb, text_mask, H, W):
        return self._cross_attention_impl(
            x_norm=x_norm,
            cond=text_emb,
            q_proj=self.text_cross_q,
            kv_proj=self.text_cross_kv,
            out_proj=self.text_cross_out,
            H=H,
            W=W,
            attn_mask=text_mask,
        )

    def forward(self, x, cond_emb, text_emb, text_mask=None, H=None, W=None, key=None):
        B, T, D = x.shape

        # [수정] shift/scale 6개만
        c = cond_emb.squeeze(1)
        chunks = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa = chunks[0], chunks[1]
        shift_cross, scale_cross = chunks[2], chunks[3]
        shift_mlp, scale_mlp = chunks[4], chunks[5]

        # 1. Self Attention (gate 제거)
        x_norm = self.self_norm(x)
        x_norm = x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        self_out = self.self_attention(x_norm=x_norm, H=H, W=W)
        x = x + self.dropout(self_out)

        # 2. Text Cross Attention (gate 제거)
        x_norm = self.text_norm(x)
        x_norm = x_norm * (1 + scale_cross[:, None, :]) + shift_cross[:, None, :]
        text_cross_out = self.text_cross_attention(
            x_norm=x_norm,
            text_emb=text_emb,
            text_mask=text_mask,
            H=H,
            W=W,
        )
        x = x + self.dropout(text_cross_out)

        # 3. FFN (gate 제거)
        x_norm = self.ffn_norm(x)
        x_norm = x_norm * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ffn = self.ff1(x_norm)
        ffn = F.gelu(ffn, approximate="tanh")
        ffn = self.ff2(ffn)
        x = x + self.dropout(ffn)

        with torch.no_grad():
            self_std = self_out.float().std().item()
            text_std = text_cross_out.float().std().item()
            ffn_std = ffn.float().std().item()

            state = {
                "key": key,
                "self_out": self_std,
                "text_out": text_std,
                "ffn_out": ffn_std,
            }

        return x, state


class Model(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, sigma_emb_hz, in_channels, patch_size, text_dim, rope_hz, **kwargs):
        super().__init__()
        dim_feedforward = 4 * d_model
        self.patch_size = patch_size
        self.in_channels = in_channels

        # patch
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model
        )
        self.patch_norm = nn.LayerNorm(d_model)

        # sigma
        self.sigma_proj = SinusoidalPosEmb(d_model, sigma_emb_hz)
        self.sigma_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigma_norm = nn.LayerNorm(d_model)

        # text token (cross attention용)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.text_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_norm = nn.LayerNorm(d_model)

        # pooled text (adaLN용)
        self.pooled_text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pooled_text_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rope_hz=rope_hz
            )
            for _ in range(num_layers)
        ])

        self.cond_norm = nn.LayerNorm(d_model)
        self.cond_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.final_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

        self.output_proj = nn.Linear(
            d_model,
            patch_size * patch_size * in_channels
        )

    def forward(self, x, sigma, text_emb, text_mask=None):
        B, C, H, W = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        sigma = sigma.to(device=x.device, dtype=torch.float32)

        # 1. 패치 임베딩
        x_emb = self.patch_norm(self.patch_embedding(x))

        # 2. sigma 임베딩
        sigma_emb = self.sigma_proj(sigma)
        sigma_emb = self.sigma_embed(sigma_emb)
        sigma_emb = self.sigma_norm(sigma_emb)

        # 3. 텍스트 토큰 임베딩 (cross attention용)
        text_token_emb = self.text_proj(text_emb)
        text_token_emb = self.text_embed(text_token_emb)
        text_token_emb = self.text_norm(text_token_emb)

        # 4. pooled text 임베딩 (adaLN용)
        if text_mask is not None:
            mask_float = text_mask.float().unsqueeze(-1)
            pooled_text = (text_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled_text = text_emb.mean(dim=1)

        pooled_text = self.pooled_text_proj(pooled_text)
        pooled_text = self.pooled_text_norm(pooled_text)

        # sigma + pooled_text → adaLN 컨디션
        cond_emb = self.cond_norm(self.cond_proj(sigma_emb + pooled_text))
        cond_emb = cond_emb[:, None, :]

        # 5. 블록 연산
        layer_states = []
        for i, block in enumerate(self.blocks):
            x_emb, state = block(
                x=x_emb,
                cond_emb=cond_emb,
                text_emb=text_token_emb,
                text_mask=text_mask,
                H=H_p,
                W=W_p,
                key=i,
            )
            layer_states.append(state)

        # 6. 최종 출력
        shift, scale = self.final_mod(cond_emb.squeeze(1)).chunk(2, dim=-1)
        x_final = self.norm(x_emb)
        x_final = x_final * (1 + scale[:, None, :]) + shift[:, None, :]

        pred_velocity = self.output_proj(x_final)
        pred_velocity = self.patch_embedding.unpatchify(pred_velocity, H, W)
        return pred_velocity, layer_states