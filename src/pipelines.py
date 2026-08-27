"""
Preprocessing pipeline for the Cats vs Dogs classifier.

Mirrors the role of a scikit-learn ColumnTransformer in a tabular
project: a single, versioned definition of "raw input -> model-ready
tensor" that is reused identically by training, evaluation, and the
inference API so there is no train/serve skew.
"""
import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    BATCH_SIZE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
)


def get_train_transforms() -> transforms.Compose:
    """
    Training-time transform: resize to a standard CNN input size and
    apply light augmentation (flip / rotation / color jitter) to
    improve generalization on a relatively small dataset.
    """
    return transforms.Compose([
        RGBTransform(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Deterministic transform used for validation, test, and live
    inference — no augmentation, since predictions must be reproducible.
    """
    return transforms.Compose([
        RGBTransform(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def preprocess_image(image: Image.Image):
    """
    Applies the evaluation transform to a single PIL image and adds a
    batch dimension. This is the exact function exercised by the
    FastAPI /predict endpoint and by the unit tests.
    """
    image = image.convert("RGB")
    tensor = get_eval_transforms()(image)
    return tensor.unsqueeze(0)  # shape: (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def get_dataloaders(batch_size: int = BATCH_SIZE):
    """Builds train/val/test DataLoaders from the processed ImageFolder layout."""
    train_ds = ImageFolder(TRAIN_DIR, transform=get_train_transforms())
    val_ds = ImageFolder(VAL_DIR, transform=get_eval_transforms())
    test_ds = ImageFolder(TEST_DIR, transform=get_eval_transforms())

    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes

class RGBTransform:
    def __call__(self, image):
        return image.convert("RGB")
