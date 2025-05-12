import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from model import ChestXRayCNN
from utils import preprocess_image

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels
classes = ["COVID", "NORMAL", "VIRAL PNEUMONIA", "LUNG OPACITY"]

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ChestXRayCNN(num_classes=4).to(device)
model.load_state_dict(torch.load("cnn_model_1.pth", map_location=device))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    filepath = f"uploads/{file.filename}"
    
    with open(filepath, "wb") as f:
        f.write(contents)
    
    img_tensor = preprocess_image(filepath).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]
    
    os.remove(filepath)  # Clean up
    return {"prediction": prediction}
