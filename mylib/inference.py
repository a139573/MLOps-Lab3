import io
import json
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

class AnimalClassifier:
    def __init__(self, model_path: str, labels_path: str):
        """
        Load the model and labels ONCE when the class is instantiated.
        """
        print(f"Loading model from {model_path}...")
        
        # --- INSTRUCTION: Configure Session Options ---
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Specifically requested in instructions
        
        # --- INSTRUCTION: Instantiate InferenceSession ---
        # We explicitly use CPUExecutionProvider as requested
        self.session = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=["CPUExecutionProvider"]
        )
        
        # Load Labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
            
        # --- INSTRUCTION: Obtain the session name ---
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image_bytes):
        """
        INSTRUCTION: Define a function/method to preprocess the data
        """
        # Open and Resize
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        # Convert to Numpy & Normalize (ImageNet stats)
        img_data = np.array(image).astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype='float32')
        std = np.array([0.229, 0.224, 0.225], dtype='float32')
        img_data = (img_data - mean) / std
        
        # Transpose to (Channels, Height, Width) & Add Batch Dimension
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data

    def predict(self, image_bytes):
        """
        INSTRUCTION: Define a function/method to predict the class label
        """
        # 1. Preprocess
        input_data = self.preprocess(image_bytes)
        
        # 2. Create Inputs Dictionary (Session name as key)
        inputs = {self.input_name: input_data}
        
        # 3. Run Inference
        outputs = self.session.run(None, inputs)
        
        # 4. Obtain Logits (first dimension)
        logits = outputs[0]
        
        # 5. Obtain Class Label
        predicted_idx = np.argmax(logits)
        return self.labels[predicted_idx]

# Helper for resize (used by the API /resize endpoint)
def resize_image_fn(image_bytes, width, height):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((width, height))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()

# --- MAIN BLOCK FOR TESTING ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mylib/inference.py <path_to_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    MODEL_PATH = "production_models/model.onnx"
    LABELS_PATH = "production_models/class_labels.json"

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("Error: Model files not found. Run 'run_pipeline.py' first.")
        sys.exit(1)

    # Instantiate Wrapper
    classifier = AnimalClassifier(MODEL_PATH, LABELS_PATH)

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Predict
    result = classifier.predict(img_bytes)
    print(f"Prediction for {image_path}: {result}")