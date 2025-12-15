"""
Unit tests for functions in inference.py.
"""
import pytest
from PIL import Image
import io
import os
from pathlib import Path

# --- FIX IMPORTS ---
# Import the Class and the new helper function name
from mylib.inference import (
    AnimalClassifier,
    resize_image_fn
)

# Constants
MODEL_PATH = "production_models/model.onnx"
LABELS_PATH = "production_models/class_labels.json"

# -------------------- Fixtures -------------------- #
@pytest.fixture
def sample_path_fixture():
    """Return a sample image path from the project data."""
    # Use a relative path to the data folder you already have
    path = Path("data/oxford-iiit-pet/images/Abyssinian_1.jpg")
    
    if not path.exists():
        # Fallback if that specific file isn't there, try to find ANY jpg
        try:
            path = next(Path("data/oxford-iiit-pet/images").glob("*.jpg"))
        except StopIteration:
            pytest.skip("No test images found in data/oxford-iiit-pet/images/")
            
    return path

@pytest.fixture
def sample_bytes_fixture(sample_path_fixture):
    return sample_path_fixture.read_bytes()


# -------------------- Unit Tests -------------------- #
def test_predict_animal(sample_bytes_fixture):
    """Test prediction using the AnimalClassifier class."""
    
    # Check if model exists (Skip test if pipeline hasn't run yet)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        pytest.skip("Model artifacts not found. Run 'run_pipeline.py' first.")

    # --- FIX LOGIC ---
    # Instantiate the class
    classifier = AnimalClassifier(MODEL_PATH, LABELS_PATH)
    
    # Run prediction
    prediction = classifier.predict(sample_bytes_fixture)
    
    # Assert result is a string (Breed name)
    assert isinstance(prediction, str)
    assert len(prediction) > 0


def test_resize_image(sample_bytes_fixture):
    new_width = 200
    new_height = 300

    # --- FIX FUNCTION NAME ---
    # call resize_image_fn (renamed in inference.py)
    resized_bytes = resize_image_fn(sample_bytes_fixture, new_width, new_height)

    # load the resized image from the returned bytes
    img = Image.open(io.BytesIO(resized_bytes))

    # check that size matches
    assert img.size == (new_width, new_height)