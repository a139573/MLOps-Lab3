"""
Integration tests for CLI commands using Click's CliRunner.
"""

import pytest
from click.testing import CliRunner

from mylib.cli import cli

from PIL import Image
import io

# Fixtures
@pytest.fixture
def cli_runner():
    """Return a CliRunner instance for invoking CLI commands."""
    return CliRunner()

@pytest.fixture
def sample_path_fixture():
    """Return a sample image path."""
    return "/home/alumno/Downloads/cat.jpg"

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
    """Test 'resize' command."""
    result = cli_runner.invoke(cli, ["predict-animal", "--path", sample_path_fixture])
    # check that size matches
    assert result.exit_code == 0
    assert result.output.strip() in ["cat", "dog", "fox"]
