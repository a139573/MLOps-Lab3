import uvicorn
from fastapi import FastAPI, Form, UploadFile, File, Response
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from mylib.inference import predict_img_class, resize_image

app = FastAPI(
    title="API of the Image Predictor",
    description="API to identify the animal in a picture",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    """
    It predicts the animal in the picture uploaded by the user
    """
    image_bytes = await img.read()
    
    result = predict_img_class(image_bytes)
    return {"class_name": result}

@app.post("/resize")
async def resize(img: UploadFile = File(...), width: int = Form(...), height: int = Form(...)):
    """
    It resizes the uploaded image
    """
    image_bytes = await img.read()
    result_bytes = resize_image(image_bytes, width, height)
    return Response(content=result_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)