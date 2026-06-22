import os
from app.sampling import Sampling
from app.sd_vae import download_sd_vae
from app.clip import download_clip
from app.config import Config
from app.model import Model
from dataclasses import asdict
import torch

base_local_root = os.path.dirname(os.path.abspath(__file__))
config = Config()

text_model, tokenizer = download_clip()

model = Model(**asdict(config))
ckpt_path = os.path.join(base_local_root, "weights", "image.pth")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"✅ checkpoint loaded (step {ckpt.get('global_step', '?')})")

model = model.to(config.device)
vae = download_sd_vae(device=config.device, dtype=config.dtype)

sampler = Sampling(
    model          = model,
    text_model     = text_model.cpu(),
    vae            = vae,
    text_tokenizer = tokenizer,
    device         = config.device,
    dtype          = config.dtype,
)

prompt    = input("prompt : ").strip() or "1girl, red hair, school uniform, happy, red eyes, open mouth, detailed face"
steps     = int(input("steps (default 100) : ").strip() or 100)
cfg_scale = float(input("cfg_scale (default 2.0) : ").strip() or 2.0)
seed_in   = input("seed (default 1234) : ").strip()
seed      = int(seed_in) if seed_in else 1234

logs, image, _, _ = sampler.generate_sampler(
    steps        = steps,
    prompt       = prompt,
    h            = 512,
    w            = 512,
    max_text_len = config.max_text_len,
    cfg_scale    = cfg_scale,
    sigma_start  = 0.999,
    sigma_end    = 0.001,
    seed         = seed,
)

from torchvision.utils import save_image
out_path = os.path.join(base_local_root, "output.png")
save_image(image, out_path)
print(f"✅ 저장 완료: {out_path}")