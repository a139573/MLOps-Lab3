import gradio as gr
import requests

RENDER_API_URL = "https://mlops-lab2-latest-r35q.onrender.com"

def predict_animal(image_path):
    """
    Sends the image to the Render API and gets the prediction.
    """
    if image_path is None:
        return "Please upload an image."
    
    try:
        # Prepare the image file to send
        with open(image_path, "rb") as im:
            images = {"img": im}
            response = requests.post(f"{RENDER_API_URL}/predict", files=images)
        
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result.get('class_name', 'Unknown')}"
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Connection Error: {str(e)}"

# Create the Gradio Interface
iface = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="filepath", label="Upload Animal Image"),
    outputs="text",
    title="Animal Image Classifier",
    description="Upload an image to get a random prediction from the Render API."
)

iface.launch()