### model_inference.py
from diffusers import StableDiffusionPipeline
import torch
import os

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    )
    pipe.to("cuda")
    return pipe

def generate_images(pipe, prompts, output_dir="outputs/"):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        path = f"{output_dir}/image_{i}.png"
        image.save(path)
        results.append((prompt, path))
    return results
