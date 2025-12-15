"""
Integration tests for CLI commands using Click's CliRunner.
"""
import pytest
import os
from click.testing import CliRunner
from PIL import Image
import io

# --- FIX IMPORT ---
# Your cli.py is in the 'cli' folder, not 'mylib'
from cli.cli import cli

# Fixtures
@pytest.fixture
def cli_runner():
    """Return a CliRunner instance for invoking CLI commands."""
    return CliRunner()

@pytest.fixture
def sample_path_fixture():
    """Return a sample image path."""
    path = "data/oxford-iiit-pet/images/Abyssinian_1.jpg"
    
    if not os.path.exists(path):
        # Fallback logic
        import glob
        files = glob.glob("data/oxford-iiit-pet/images/*.jpg")
        if files:
            path = files[0]
        else:
            pytest.skip(f"Test image not found at {path}")
            
    return path

@pytest.fixture
def sample_size_fixture():
    """Return a sample tuple of width and height."""
    return (224, 224)


# ------------------ Integration Tests ------------------ #
def test_resize(cli_runner, sample_path_fixture, sample_size_fixture):
    width, height = sample_size_fixture

    result = cli_runner.invoke(cli, [
        "resize",
        "--path", sample_path_fixture,
        "--width", str(width),
        "--height", str(height)
    ])
    assert result.exit_code == 0

    img_bytes = result.stdout_bytes
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (width, height)

def test_prediction(cli_runner, sample_path_fixture):
    """Test 'predict-animal' command."""
    
    # Skip if model isn't built yet
    if not os.path.exists("production_models/model.onnx"):
        pytest.skip("Model not found. Run pipeline first.")

    result = cli_runner.invoke(cli, ["predict-animal", "--path", sample_path_fixture])
    
    # check that execution was successful
    assert result.exit_code == 0
    # check that we got some text back (the class name)
    assert result.output.strip() != ""