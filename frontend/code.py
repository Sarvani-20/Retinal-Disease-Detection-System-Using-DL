from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# 1. ALLOW THE FRONTEND TO TALK TO THE BACKEND (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD YOUR MODEL (Make sure this path is 100% correct!)
# Use forward slashes / or double backslashes \\
MODEL_PATH = "C:\\EyeProject\\model\\best_combined_model.pth"

try:
    # Basic architecture setup - change if your model is different
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {e}")

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. TEST ROUTE (Health Check)
@app.get("/")
async def root():
    return {"status": "Backend is Running!"}

# 4. PREDICTION ROUTE
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            # Converting raw output to percentages
            probs = torch.sigmoid(output)[0].tolist()

        return {
            "dr": round(probs[0] * 100, 1),
            "glaucoma": round(probs[1] * 100, 1),
            "amd": round(probs[2] * 100, 1)
        }
    except Exception as e:
        return {"error": str(e)}

# Run this with: python -m uvicorn code:app --reload