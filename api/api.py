import uvicorn
import os
from fastapi import FastAPI, Form, UploadFile, File, Response
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

# --- UPDATE IMPORTS ---
# We now import the Class and the helper function we renamed/refactored
from mylib.inference import AnimalClassifier, resize_image_fn

app = FastAPI(
    title="API of the Image Predictor",
    description="API to identify the animal in a picture",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")

# --- GLOBAL MODEL INITIALIZATION ---
# This ensures we only load the heavy ONNX model once (Singleton pattern usage)
MODEL_PATH = "production_models/model.onnx"
LABELS_PATH = "production_models/class_labels.json"

# Initialize as None first
classifier = None

# Check if files exist before trying to load (avoids immediate crash if paths are wrong)
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    print("🚀 Loading model into memory...")
    classifier = AnimalClassifier(MODEL_PATH, LABELS_PATH)
else:
    print(f"⚠️ WARNING: Model files not found at {MODEL_PATH}. Prediction endpoint will fail.")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    """
    It predicts the animal in the picture uploaded by the user
    """
    if classifier is None:
        return {"error": "Model not loaded on server."}

    image_bytes = await img.read()
    
    # --- USE THE CLASS INSTANCE ---
    result = classifier.predict(image_bytes)
    return {"class_name": result}

@app.post("/resize")
async def resize(img: UploadFile = File(...), width: int = Form(...), height: int = Form(...)):
    """
    It resizes the uploaded image
    """
    image_bytes = await img.read()
    
    # --- USE THE HELPER FUNCTION ---
    # Note: We renamed 'resize_image' to 'resize_image_fn' in inference.py
    # to avoid naming conflicts with this route name
    result_bytes = resize_image_fn(image_bytes, width, height)
    
    return Response(content=result_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)