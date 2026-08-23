import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import build_model  # noqa: E402
from config import IMAGE_SIZE  # noqa: E402


def test_model_output_shape():
    model = build_model(num_classes=2)
    model.eval()

    batch = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (4, 2)


def test_model_inference_produces_valid_probabilities():
    model = build_model(num_classes=2)
    model.eval()

    single_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        logits = model(single_image)
        probs = torch.softmax(logits, dim=1)

    assert probs.shape == (1, 2)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_model_accepts_variable_batch_sizes():
    # AdaptiveAvgPool2d in the architecture means batch size 1 and
    # larger batches must both work without shape errors.
    model = build_model(num_classes=2)
    model.eval()

    for batch_size in (1, 3, 8):
        batch = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (batch_size, 2)
