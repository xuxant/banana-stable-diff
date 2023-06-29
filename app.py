from potassium import Potassium, Request, Response
from diffusers import StableDiffusionPipeline
from io import BytesIO
import os
import base64

from transformers import pipeline
import torch

app = Potassium("diffusion")

@app.init
def init():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = 0 if torch.cuda.is_available() else -1
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to("cuda")
    

    context = {
        "model": model
    }
    return context


@app.handler()
def handler(context:dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")

    image = model(prompt).images[0]
    buffered = BytesIO()

    image.save(buffered, format="JPEG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return Response(
        json={"output": image_b64},
        status=200,
    )

if __name__ == "__main__":
    app.serve()

    