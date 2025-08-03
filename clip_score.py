### metrics/clip_score.py
import clip
from PIL import Image
import torch

model, preprocess = clip.load("ViT-B/32", device="cuda")

def clip_similarity(image_path, text_prompt):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")
    text = clip.tokenize([text_prompt]).to("cuda")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    return similarity.item()
