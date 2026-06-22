from transformers import CLIPTextModel, CLIPTokenizer

def download_clip():
    text_model = CLIPTextModel.from_pretrained("hakurei/waifu-diffusion", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("hakurei/waifu-diffusion", subfolder="tokenizer")
    
    return text_model, tokenizer