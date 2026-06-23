import os
import sys
print(f"Python version: {sys.version}")

import cv2
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Import MediaPipe secara eksplisit
from mediapipe.python.solutions import face_detection as mp_face_detection
from mediapipe.python.solutions import face_mesh as mp_face_mesh

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Inisialisasi MediaPipe
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

class StrokeDetectionMLP(nn.Module):
    def __init__(self, input_dim=956, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hdim), nn.BatchNorm1d(hdim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model_path = os.getenv("MODEL_PATH", "best_model.pth")
model = None
if os.path.exists(model_path):
    try:
        model = StrokeDetectionMLP()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Model file not found at {model_path}")

def extract_landmarks_and_crop(image):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    det = face_detection.process(rgb)
    if not det.detections:
        return None, None, None, False

    bbox = det.detections[0].location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)
    margin_x = int(box_w * 0.15)
    margin_y = int(box_h * 0.15)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    box_w = min(w - x, box_w + 2 * margin_x)
    box_h = min(h - y, box_h + 2 * margin_y)
    cropped = image[y:y+box_h, x:x+box_w]
    if cropped.size == 0:
        return None, None, None, False

    crop_resized = cv2.resize(cropped, (300, 300))
    rgb_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    mesh = face_mesh.process(rgb_crop)
    if not mesh.multi_face_landmarks:
        return None, None, None, False

    landmarks = []
    for lm in mesh.multi_face_landmarks[0].landmark:
        landmarks.append(lm.x * 300)
        landmarks.append(lm.y * 300)

    features = np.array(landmarks, dtype=np.float32)
    if features.std() > 1e-6:
        features = (features - features.mean()) / (features.std() + 1e-6)

    return features.tolist(), cropped, (x, y, box_w, box_h), True

@app.get("/")
async def root():
    return {"message": "Stroke Detection API", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Model not loaded", "stroke": False})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid image"})

        features, cropped, bbox, ok = extract_landmarks_and_crop(image)
        if not ok:
            return JSONResponse(status_code=200, content={"success": False, "error": "No face detected", "stroke": False})

        input_tensor = torch.tensor([features], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            stroke_prob = probs[0][1].item()

        is_stroke = stroke_prob > 0.5
        confidence = stroke_prob if is_stroke else 1 - stroke_prob

        return {
            "success": True,
            "stroke": is_stroke,
            "confidence": round(confidence, 4),
            "probability": round(stroke_prob, 4),
            "face_detected": True
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "stroke": False})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)