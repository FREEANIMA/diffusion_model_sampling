# sd model
from diffusers import AutoencoderKL
import torch

def download_sd_vae(model_name="madebyollin/sdxl-vae-fp16-fix", device="cuda", dtype=torch.bfloat16):
    vae = AutoencoderKL.from_pretrained(
        model_name,
        torch_dtype=dtype
    )

    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)

    return vae