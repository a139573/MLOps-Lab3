"""
Command Line Interface (CLI) for data preprocessing operations.
"""
import os
import sys
import click
# --- FIX IMPORTS ---
from mylib.inference import AnimalClassifier, resize_image_fn

# Paths to models (Hardcoded for CLI usage relative to root)
MODEL_PATH = "production_models/model.onnx"
LABELS_PATH = "production_models/class_labels.json"

def sanitize_image_path(path: str) -> str:
    """Validate and normalize an input image path."""
    path = path.strip()

    if not path:
        raise click.BadParameter("The --path cannot be empty.")

    if not os.path.isfile(path):
        raise click.BadParameter(f"File not found: {path}")

    valid_ext = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    if not path.lower().endswith(valid_ext):
        raise click.BadParameter(
            f"Invalid image format. Allowed: {', '.join(valid_ext)}"
        )
    return path


@click.group(help="Main group of commands for data preprocessing.")
def cli() -> None:
    """Entry point."""


@cli.command(
    help=(
        "Predicting the class of an image between cat, dog and fox."
        "Example: uv python -m cli.cli predict-animal --path '001.png'"
    )
)
@click.option("--path", required=True, help="Path to the image.")
def predict_animal(path: str) -> None:
    """Predict class of an image."""
    path = sanitize_image_path(path)
    
    # --- FIX LOGIC: Instantiate Class ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        click.echo("Error: Model files not found. Run the pipeline first.")
        return

    classifier = AnimalClassifier(MODEL_PATH, LABELS_PATH)
    
    with open(path, "rb") as f:
        img_bytes = f.read()
        
    prediction = classifier.predict(img_bytes)
    click.echo(prediction)


@cli.command(
    help=(
        "Resize image from path. "
        "Example: python -m cli.cli resize --path '001.png' --width 100 --height 100"
    )
)
@click.option("--path", required=True, help="Path to the image.")
@click.option("--width", required=True, type=int, help="New width of the image.")
@click.option("--height", required=True, type=int, help="New height of the image.")
def resize(path: str, width: int, height: int) -> None:
    """Resize image."""
    path = sanitize_image_path(path)
    with open(path, "rb") as f:
        img_bytes = f.read()
    
    # --- FIX LOGIC: Use renamed helper function ---
    resized_img = resize_image_fn(img_bytes, width, height)
    sys.stdout.buffer.write(resized_img)


if __name__ == "__main__":
    cli()