from diffusers import StableDiffusionPipeline
import torch


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_id = "runwayml/stable-diffusion-v1-5"
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

if __name__ == "__main__":
    download_model()