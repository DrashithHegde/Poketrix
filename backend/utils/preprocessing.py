# Preprocessing script for Poketrix

import os
from PIL import Image
import numpy as np

def resize_and_normalize(image_input, size=(64, 64)):
    """Resize and normalize an image to [-1, 1]."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    image = image.resize(size)
    image_array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return image_array

def merge_datasets(image_dir, metadata_file):
    """Merge image and metadata datasets."""
    # Placeholder for merging logic
    pass

def extract_dominant_color(image_path):
    """Extract the dominant color from an image using PIL."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1, 1))  # Resize to 1x1 pixel to get the dominant color
    dominant_color = image.getpixel((0, 0))
    return dominant_color