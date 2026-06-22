# Diffusion Transformer

A flow matching-based diffusion transformer for anime image generation.  
This project is for **research purposes only**.

## Samples

```
prompt    = "1girl, red hair, school uniform, happy, red eyes, open mouth, detailed face"
steps     = 100
cfg_scale = 2.0
seed      = 1234
```
| 1k steps | 10k steps | 50k steps | 100k steps |
|---|---|---|---|
| ![1k](assets/1k.png) | ![10k](assets/10k.png) | ![50k](assets/50k.png) | ![100k](assets/100k.png) |

## Model Architecture

- **Backbone**: Diffusion Transformer (DiT) with adaLN modulation
- **Parameters**: ~550M
- **Framework**: Flow Matching (velocity prediction)

## Components

| Component | Model |
|---|---|
| VAE | stabilityai/sd-vae-ft-mse |
| Text Encoder | openai/clip-vit-large-patch14 |
| Tokenizer | openai/clip-vit-large-patch14 |

## Sampler Details

- **Resolution**: 512 × 512 (single bucket)
- **Noise Schedule**: Log-SNR uniform sampling with resolution-dependent shift
- **CFG**: Classifier-free guidance
- Prompts are **tag-based** (comma-separated danbooru-style tags)

## Requirements

```bash
pip install torch transformers diffusers accelerate torchvision tqdm
```

## Usage

```bash
python main.py
```

```
C:.
│  main.py
│  output.png
│  README.md
│  requirements.txt
│
├─app
│  │  clip.py
│  │  config.json
│  │  config.py
│  │  model.py
│  │  sampling.py
│  │  sd_vae.py
│  └─ __init__.py
│
├─assets
│      100k.png
│      150k.png
│      1k.png
│      50k.png
│
└─weights
       image.pth

```