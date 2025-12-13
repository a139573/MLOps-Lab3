import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
# We assume the model and labels are in a folder named 'production_models'
# relative to where the app is running.
MODEL_PATH = "production_models/model.onnx"
LABELS_PATH = "production_models/class_labels.json"

# --- GLOBAL SESSION ---
# We load the model ONCE when the server starts, not for every request.
# This makes the API fast.
session = None
class_labels = None

def load_resources():
    """
    Loads the ONNX model and JSON labels into global variables.
    """
    global session, class_labels
    
    # 1. Load Labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            class_labels = json.load(f)
        print(f"✅ Loaded {len(class_labels)} labels.")
    else:
        print(f"❌ Error: Labels file not found at {LABELS_PATH}")
        class_labels = []

    # 2. Load ONNX Model
    if os.path.exists(MODEL_PATH):
        # Create Inference Session (CPU Execution)
        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        
        session = ort.InferenceSession(MODEL_PATH, sess_options, providers=providers)
        print("✅ ONNX Model loaded successfully.")
    else:
        print(f"❌ Error: ONNX Model not found at {MODEL_PATH}")

# Trigger loading immediately when this module is imported
load_resources()

def preprocess_image(image_bytes: bytes):
    """
    Converts raw bytes to the format MobileNetV2 expects:
    1. Resize to 224x224
    2. Normalize (ImageNet stats)
    3. Shape (1, 3, 224, 224)
    """
    with Image.open(BytesIO(image_bytes)) as img:
        # Convert to RGB (handle PNGs with alpha channel)
        img = img.convert("RGB")
        
        # 1. Resize
        img = img.resize((224, 224))
        
        # 2. Convert to Numpy & Normalize
        # MobileNet Expects: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
        img_data = np.array(img).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Apply normalization: (Image - Mean) / Std
        img_data = (img_data - mean) / std
        
        # 3. Transpose to (Channels, Height, Width) -> PyTorch format
        img_data = img_data.transpose(2, 0, 1)
        
        # 4. Add Batch Dimension (1, C, H, W)
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data

def predict_img_class(image_bytes: bytes):
    """
    Runs inference on the ONNX model.
    """
    # Safety check if model failed to load
    if session is None or not class_labels:
        return "System Error: Model not loaded."

    try:
        # 1. Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # 2. Run Inference
        # Get input name (usually 'input')
        input_name = session.get_inputs()[0].name
        
        # Run session
        # outputs is a list, the first element contains the logits
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        
        # 3. Postprocess
        # Find index of highest probability
        predicted_idx = np.argmax(logits)
        predicted_label = class_labels[predicted_idx]
        
        return predicted_label

    except Exception as e:
        print(f"Inference Error: {e}")
        return "Error during prediction"
        

def resize_image(image_bytes: bytes, width: int, height: int) -> bytes:
    """
    Resize an input image to the given width and height.
    Returns the resized image as bytes (JPEG format).
    """
    with Image.open(BytesIO(image_bytes)) as img:
        resized = img.resize((width, height))
        output = BytesIO()
        resized.save(output, format="JPEG")
        return output.getvalue()



if __name__ == "__main__":
    import sys

    # 1. Check if user provided an image path
    if len(sys.argv) < 2:
        print("Usage: uv run python mylib/inference.py <path_to_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        sys.exit(1)

    print(f"🔍 Testing inference on: {image_path}")

    # 2. Read the image as bytes (Simulating an API upload)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # 3. Run Prediction
    result = predict_img_class(image_bytes)
    
    print("-" * 30)
    print(f"🤖 Model Prediction: {result}")
    print("-" * 30)