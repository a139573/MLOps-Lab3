import random
from PIL import Image
from io import BytesIO

# • Define a method to predict the class of a given image. In this first lab, the class will# 
# be randomly chosen among a set of class names (of your choice).
def predict_img_class(img):
    return random.choice(['cat', 'dog', 'fox'])

# • Define a method to resize an image to a certain size.
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

# • You can define more methods to preprocess an image (of your choice).
