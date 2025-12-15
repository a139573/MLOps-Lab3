import os
import pytest

# Define the path to the artifacts
# We assume the test is run from the root of the project
MODEL_DIR = "production_models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

def test_model_directory_exists():
    """Check if the production_models folder was created."""
    assert os.path.exists(MODEL_DIR), f"Directory '{MODEL_DIR}' does not exist. Did the training pipeline run?"

def test_onnx_model_exists():
    """Check if model.onnx exists and is not empty."""
    assert os.path.exists(MODEL_PATH), f"File '{MODEL_PATH}' missing."
    assert os.path.getsize(MODEL_PATH) > 0, f"File '{MODEL_PATH}' is empty."

def test_labels_file_exists():
    """Check if class_labels.json exists and is not empty."""
    assert os.path.exists(LABELS_PATH), f"File '{LABELS_PATH}' missing."
    assert os.path.getsize(LABELS_PATH) > 0, f"File '{LABELS_PATH}' is empty."