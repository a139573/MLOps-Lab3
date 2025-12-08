"""
Command Line Interface (CLI) for data preprocessing operations.

This CLI uses Click to expose functions from preprocessing.py for:
- Cleaning missing or duplicate values
- Normalizing and transforming numeric data
- Processing text
- Manipulating data structures
"""
import os
import sys
import click
from mylib.inference import (
    predict_img_class,
    resize_image,
)

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
        "Example: uv python -m mylib.cli predict-animal --path '001.png'"
    )
)
@click.option("--path", required=True, help="Path to the image.")
def predict_animal(path: str) -> None:
    """Predict class of an image."""
    img_bytes = sanitize_image_path(path)
    prediction = predict_img_class(img_bytes)
    click.echo(prediction)


@cli.command(
    help=(
        "Resize image from path. "
        "Example: python -m mylib.cli resize --path '001.png'"
    )
)
@click.option("--path", required=True, help="Path to the image.")
@click.option("--width", required=True, type=int, help="New width of the image.")   # <-- type=int
@click.option("--height", required=True, type=int, help="New height of the image.") # <-- type=int
def resize(path: str, width: int, height: int) -> None:
    """Resize image."""
    path = sanitize_image_path(path)
    with open(path, "rb") as f:
        img_bytes = f.read()  
    resized_img = resize_image(img_bytes, width, height)
    sys.stdout.buffer.write(resized_img)


if __name__ == "__main__":
    cli()
