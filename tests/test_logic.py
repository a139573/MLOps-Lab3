"""
Unit tests for functions in preprocessing.py.
"""

import pytest
from PIL import Image
import io

from mylib.inference import (
    predict_img_class,
    resize_image
)

from pathlib import Path


# -------------------- Fixtures -------------------- #
@pytest.fixture
def sample_path_fixture():
    """Return a sample image path."""
    return "/home/alumno/Downloads/cat.jpg"

@pytest.fixture
def sample_bytes_fixture():
    path = Path("/home/alumno/Downloads/cat.jpg")
    return path.read_bytes()


# -------------------- Unit Tests -------------------- #
def test_predict_animal(sample_path_fixture):
    """Test removing missing values."""
    assert predict_img_class(sample_path_fixture) in ["cat", "dog", "fox"]


def test_resize_image(sample_bytes_fixture):
    new_width = 200
    new_height = 300

    # call your resize function
    resized_bytes = resize_image(sample_bytes_fixture, new_width, new_height)

    # load the resized image from the returned bytes
    img = Image.open(io.BytesIO(resized_bytes))

    # check that size matches
    assert img.size == (new_width, new_height)