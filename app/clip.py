from transformers import CLIPTextModel, CLIPTokenizer

def download_clip():
    text_model = CLIPTextModel.from_pretrained("hakurei/waifu-diffusion", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("hakurei/waifu-diffusion", subfolder="tokenizer")

    print(text_model.config.hidden_size)
    print("로드 성공")

    return text_model, tokenizer