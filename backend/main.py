# FastAPI app for Poketrix

from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from models.generator import Generator
from models.discriminator import Discriminator
import torch
from PIL import Image
import io
from utils.preprocessing import resize_and_normalize

import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g., localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configuration and Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATOR_PATH = os.path.join(BASE_DIR, "generator.pth")
DISCRIMINATOR_PATH = os.path.join(BASE_DIR, "discriminator.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on device: {device}")

# Load models
generator = Generator(100, 10).to(device)
if os.path.exists(GENERATOR_PATH):
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=device, weights_only=True))
generator.eval()

discriminator = Discriminator(10).to(device)
if os.path.exists(DISCRIMINATOR_PATH):
    discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location=device, weights_only=True))
discriminator.eval()

class GenerateRequest(BaseModel):
    type: str
    color: str

def encode_condition(type: str, color: str):
    """Encode Pokémon type and color into a condition vector."""
    type_mapping = {"Fire": 0, "Water": 1, "Grass": 2, "Electric": 3}
    color_mapping = {"Red": 0, "Blue": 1, "Green": 2, "Yellow": 3}

    condition = torch.zeros(10)
    condition[type_mapping[type]] = 1
    condition[4 + color_mapping[color]] = 1
    return condition

@app.post("/generate")
async def generate_pokemon(request: GenerateRequest):
    """Generate a Pokémon image based on type and color."""
    condition = encode_condition(request.type, request.color).unsqueeze(0).to(device)
    noise = torch.randn(1, 100).to(device)
    with torch.no_grad():
        generated_tensor = generator(noise, condition)
        prediction = discriminator(generated_tensor, condition).item()
        generated_image = generated_tensor.squeeze(0)
    
    # Convert to PIL image
    generated_image = ((generated_image + 1) * 127.5).clamp(0, 255).byte()
    pil_image = Image.fromarray(generated_image.permute(1, 2, 0).cpu().numpy())

    # Save to buffer
    import base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image": img_b64,
        "real_probability": prediction,
        "fake_probability": 1 - prediction
    }

@app.post("/detect")
async def detect_image(file: UploadFile, type: str = Form(...), color: str = Form(...)):
    """Detect if an image is real or fake."""
    image = Image.open(file.file).convert("RGB")
    # Get shape (H, W, C) then to Tensor and permute to (C, H, W) then to (1, C, H, W)
    image_array = resize_and_normalize(image)
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).to(device)

    # Convert incoming type and color to the formal condition vector
    condition = encode_condition(type, color).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = discriminator(image_tensor, condition).item()

    return {
        "real_probability": prediction,
        "fake_probability": 1 - prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)