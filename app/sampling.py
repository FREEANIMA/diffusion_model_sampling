import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

'''
[Sampling Overview]

Input: prompt (str), h, w, steps, cfg_scale

1. Text Encoding
   ├── cond_emb   = CLIP encode(prompt)       # conditional
   └── uncond_emb = CLIP encode("")           # unconditional (CFG)

2. Latent Initialization
   latent ~ N(0, 1) → shape (1, 4, latent_h, latent_w)
   latent_h = (h // 8) & ~1
   latent_w = (w // 8) & ~1

3. Sigma Schedule (Flow Matching)
   log-SNR linspace(logsnr_start → logsnr_end, steps+1)
   + resolution shift mu = log(latent_h*latent_w / 64*64)  # larger image → shift schedule
   sigmas = sigmoid(-0.5 * logsnrs)

4. Euler Sampling Loop × steps
   for each (sigma_hi → sigma_lo):
   ├── pred_cond   = model(latent, sigma_hi, cond_emb)
   ├── pred_uncond = model(latent, sigma_hi, uncond_emb)
   ├── pred_v      = pred_uncond + cfg_scale * (pred_cond - pred_uncond)  # CFG
   │
   ├── pred_clean = x_t - sigma_hi * pred_v          # estimated clean image
   ├── pred_noise = x_t + (1 - sigma_hi) * pred_v   # estimated noise
   └── latent     = (1 - sigma_lo) * pred_clean + sigma_lo * pred_noise  # euler step

5. VAE Decode
   latent / scaling_factor → VAE decode → (image + 1) / 2 → clamp(0, 1)

Output: image (1, 3, H, W), logs (List[dict]), latent_lo, pred_v
'''

class Sampling:
    def __init__(self, model, text_model, vae, text_tokenizer, device, dtype=torch.bfloat16, **kwargs):
        self.model = model.to(device)
        self.text_model = text_model.cpu().eval()  # ← 명시적으로 cpu
        self.vae = vae.to(device).eval()
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_text(self, texts, max_text_len):
        tok = self.text_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt"
        )
        input_ids = tok["input_ids"]          # CPU 유지
        attention_mask = tok["attention_mask"].bool()  # CPU 유지

        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = out.last_hidden_state.to(device=self.device, dtype=self.dtype)  # 결과만 GPU
        attention_mask = attention_mask.to(self.device)
        return text_emb, attention_mask

    @torch.no_grad()
    def predict(self, latent, sigma, cond_emb, cond_mask):
        B = latent.shape[0]
        sigma_t = torch.full((B,), float(sigma), device=self.device, dtype=torch.float32)
        x = latent.to(self.dtype)
        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=True):
            out = self.model(x=x, sigma=sigma_t, text_emb=cond_emb, text_mask=cond_mask)
            pred_delta = out[0] if isinstance(out, tuple) else out
        return pred_delta.float()

    @torch.no_grad()
    def generate_sampler(self, steps, prompt, h, w, max_text_len,
                         sigma_start=0.995, sigma_end=0.005, init_prompt="",
                         seed=None, decode=True, mode="noise", cfg_scale=7.5):
        was_training = self.model.training
        self.model.eval()

        logs = []
        image = None
        latent_lo = None
        pred_v = None

        try:
            cond_emb, cond_mask = self.encode_text([prompt], max_text_len)
            uncond_emb, uncond_mask = self.encode_text([init_prompt], max_text_len)

            latent_w = (w // 8) & ~1
            latent_h = (h // 8) & ~1
            C = getattr(self.vae.config, "latent_channels", 4)

            g = torch.Generator(device=self.device)
            if seed is not None:
                g.manual_seed(seed)

            latent = torch.randn(
                1, C, latent_h, latent_w,
                device=self.device,
                dtype=torch.float32,
                generator=g if seed is not None else None,
            )

            eps = 0.005
            sigma_start_safe = min(max(float(sigma_start), eps), 1.0 - eps)
            sigma_end_safe   = min(max(float(sigma_end),   eps), 1.0 - eps)

            logsnr_start = math.log(((1 - sigma_start_safe) ** 2) / (sigma_start_safe ** 2))
            logsnr_end   = math.log(((1 - sigma_end_safe) ** 2) / (sigma_end_safe ** 2))
            logsnrs = torch.linspace(logsnr_start, logsnr_end, steps + 1, device=self.device)

            base_sequence_len = 64 * 64
            current_sequence_len = latent_h * latent_w

            if current_sequence_len > base_sequence_len:
                mu = math.log(current_sequence_len / base_sequence_len)
                logsnrs = logsnrs + mu

            sigmas = torch.sigmoid(-0.5 * logsnrs).clamp(eps, 1.0 - eps)

            for i in tqdm(range(steps), desc="generated image"):
                sigma_hi = sigmas[i]
                sigma_lo = sigmas[i + 1]

                latent_std_before = latent.std().item()

                # CFG 분기 예측 수행
                pred_cond = self.predict(latent=latent, sigma=float(sigma_hi),
                                         cond_emb=cond_emb, cond_mask=cond_mask)
                pred_uncond = self.predict(latent=latent, sigma=float(sigma_hi),
                                           cond_emb=uncond_emb, cond_mask=uncond_mask)

                text_delta = pred_cond - pred_uncond
                pred_v = pred_uncond + cfg_scale * text_delta

                # 💡 Flow Matching 오일러 스텝 공식
                x_t = latent.float()
                pred_clean = x_t - sigma_hi * pred_v
                pred_noise = x_t + (1.0 - sigma_hi) * pred_v
                latent = (1.0 - sigma_lo) * pred_clean + sigma_lo * pred_noise

                latent_std_after = latent.std().item()
                step_delta = latent - x_t
                update_std = step_delta.std().item()

                logs.append({
                    "step": i,
                    "sigma_hi": float(sigma_hi),
                    "sigma_lo": float(sigma_lo),
                    "latent_std_before": latent_std_before,
                    "latent_std_after": latent_std_after,
                    "pred_v_std": pred_v.std().item(),
                    "text_effect": text_delta.std().item(),
                    "delta_over_latent": update_std / (latent_std_before + 1e-8),
                    "move_cos": F.cosine_similarity(
                        step_delta.flatten(1),
                        (-(x_t - latent).flatten(1)),
                        dim=1
                    ).mean().item(),
                })

            latent_lo = latent

            if decode:
                decode_latent = latent_lo / self.vae.config.scaling_factor
                self.vae.to(dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=False):
                    image = self.vae.decode(decode_latent.float()).sample
                self.vae.to(dtype=self.dtype)
                image = ((image + 1.0) / 2.0).clamp(0, 1)

        finally:
            if was_training:
                self.model.train()

        return logs, image, latent_lo, pred_v