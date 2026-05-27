#for API
from fastapi import FastAPI, Body, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

#for MLP
import torch
import torch.nn.functional as F
import torchvision.io as io
from torchvision.transforms import v2
from PIL import Image

from model import FLAG_CNN, DEVICE, INPUT_HEIGHT, INPUT_WIDTH
from flag_mapping import flag_map

from pathlib import Path

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "https://yourusername.github.io"
]

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'flag_model_4conv.pth'
model = FLAG_CNN().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model_state_dict'])
model.eval()

async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(file.file).convert("RGB")
    transform = v2.Compose([
        v2.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        v2.CenterCrop((INPUT_HEIGHT, INPUT_WIDTH)), 
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

@app.post("/predict-image")
async def use_model_for_prediction(file: UploadFile = File(...)):

    image_tensor = await predict_image(file)

    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        probabilities = F.softmax(model(image_tensor), dim=1)
        predicted_class = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0][predicted_class].item()

    return {"country":flag_map[predicted_class+1], "confidence":confidence}

