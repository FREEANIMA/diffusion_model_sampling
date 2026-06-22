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

prompt = "1girl, blue hair, school uniform, smile, blue eyes, looking at viewer, detailed face"

logs, image, _, _ = sampler.generate_sampler(
    steps        = 100,
    prompt       = prompt,
    h            = config.target_size_H,
    w            = config.target_size_W,
    max_text_len = config.max_text_len,
    cfg_scale    = 2.0,
    sigma_start  = 0.999,
    sigma_end    = 0.001,
    seed         = 1234,
)

from torchvision.utils import save_image
out_path = os.path.join(base_local_root, "output.png")
save_image(image, out_path)
print(f"✅ 저장 완료: {out_path}")