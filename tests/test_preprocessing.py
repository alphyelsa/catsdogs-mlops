import sys
import os

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipelines import preprocess_image  # noqa: E402
from config import IMAGE_SIZE  # noqa: E402


def _random_image(width=300, height=180) -> Image.Image:
    array = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_preprocess_image_shape_and_type():
    image = _random_image()
    tensor = preprocess_image(image)

    assert isinstance(tensor, torch.Tensor)
    # batch dim + 3 channels + IMAGE_SIZE x IMAGE_SIZE
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_handles_grayscale_input():
    # Grayscale / palette images are common in scraped datasets and
    # must be coerced to RGB rather than raising an error.
    grayscale = Image.fromarray(
        (np.random.rand(120, 120) * 255).astype(np.uint8), mode="L"
    )
    tensor = preprocess_image(grayscale)

    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_is_normalized():
    image = _random_image()
    tensor = preprocess_image(image)

    # After ImageNet normalization, pixel values should no longer sit
    # in the raw [0, 1] range for a typical random image.
    assert tensor.min().item() < 0 or tensor.max().item() > 1
