import os
import json
from dataclasses import dataclass, field
import torch

config_path = os.path.join(os.path.dirname(__file__), 'config.json')

with open(config_path, 'r') as f:
    _cfg = json.load(f)

@dataclass
class Config:
    weight_decay: float = _cfg["weight_decay"]
    max_lr: float       = _cfg["max_lr"]
    min_lr: float       = _cfg["min_lr"]
    warmup_steps: int   = _cfg["warmup_steps"]
    d_model: int        = _cfg["d_model"]
    nhead: int          = _cfg["nhead"]
    num_layers: int     = _cfg["num_layers"]
    dropout: float      = _cfg["dropout"]
    patch_size: int     = _cfg["patch_size"]
    in_channels: int    = _cfg["in_channels"]
    sigma_emb_hz: int   = _cfg["sigma_emb_hz"]
    rope_hz: int     = _cfg["rope_hz"]
    batch_size: int     = _cfg["batch_size"]
    target_size_H: int   = _cfg["target_size_H"]
    target_size_W: int  = _cfg["target_size_W"]
    text_dim: int       = _cfg["text_dim"]
    max_text_len: int   = _cfg["max_text_len"]
    drop_prob: float    = _cfg["drop_prob"]

    device: str           = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype    = torch.bfloat16
    use_amp: bool         = True
    use_compile: bool     = False
    dynamic: bool         = False