import gradio as gr
import requests
import os

# --- CONFIGURATION ---
# Try to get the URL from the Environment (Best Practice), otherwise fallback to default
RENDER_API_URL = os.getenv("API_URL", "https://mlops-lab2-latest-r35q.onrender.com")

def predict_animal(image_path):
    """
    Sends the image to the Render API and gets the prediction.
    """
    if image_path is None:
        return "⚠️ Please upload an image."
    
    try:
        # Prepare the image file to send
        with open(image_path, "rb") as im:
            images = {"img": im}
            response = requests.post(f"{RENDER_API_URL}/predict", files=images)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('class_name', 'Unknown')
            return f"🧠 Model Prediction: {prediction}"
        else:
            return f"❌ Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"🔌 Connection Error: {str(e)}"

# --- INTERFACE DEFINITION ---
# improved title and description for Lab 3
title = "MLOps Lab 3: Animal Image Classifier"
description = """
### 🐕 Oxford-IIIT Pet Classifier
This interface is connected to a **Deep Learning Backend** (hosted on Render). 

**Technical Details:**
* **Model:** MobileNetV2 (Transfer Learning)
* **Dataset:** Oxford-IIIT Pet Dataset (37 Categories)
* **Infrastructure:** Docker + FastAPI + ONNX Runtime

Upload an image of a cat or dog to see the model in action!
"""

# Create the Gradio Interface
iface = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="filepath", label="Upload Animal Image"),
    outputs=gr.Textbox(label="Result"),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch()
