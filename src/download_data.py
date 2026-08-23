"""
Downloads the Kaggle "Cats and Dogs" classification dataset and
splits it into train/val/test folders (80/10/10) using an
ImageFolder-compatible layout:

    data/processed/train/cat/*.jpg
    data/processed/train/dog/*.jpg
    data/processed/val/cat/*.jpg
    ...

Requires Kaggle API credentials (~/.kaggle/kaggle.json or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables) — see
https://github.com/Kaggle/kaggle-api#api-credentials

Usage:
    python src/download_data.py
"""
import os
import random
import shutil
import zipfile

from config import (
    RAW_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_NAMES,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
)

KAGGLE_DATASET = "salader/dogs-vs-cats"  # canonical Kaggle slug for this dataset

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def download_from_kaggle(dest_dir: str = RAW_DATA_DIR) -> None:
    """Downloads and unzips the raw dataset via the Kaggle API."""
    os.makedirs(dest_dir, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required. Install with "
            "'pip install kaggle' and configure API credentials."
        ) from exc

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=dest_dir, unzip=False)

    zip_path = os.path.join(dest_dir, "dogs-vs-cats.zip")
    if os.path.exists(zip_path):
        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)

    print(f"Raw dataset available at: {dest_dir}")


def _collect_class_images(raw_dir: str, class_name: str) -> list:
    """
    Finds image files for a given class regardless of how the raw
    Kaggle archive happened to be laid out (some releases nest
    everything under a 'train/' subfolder, others don't).
    """
    matches = []
    for root, _dirs, files in os.walk(raw_dir):
        for fname in files:
            if not fname.lower().endswith(VALID_EXTENSIONS):
                continue
            if class_name in fname.lower() or class_name in root.lower():
                matches.append(os.path.join(root, fname))
    return matches


def split_dataset(raw_dir: str = RAW_DATA_DIR) -> None:
    """
    Splits raw images into train/val/test folders per class using the
    ratios defined in config.py. Copies (not moves) files so re-running
    this script is safe and idempotent from the raw source.
    """
    random.seed(RANDOM_SEED)

    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    for class_name in CLASS_NAMES:
        images = _collect_class_images(raw_dir, class_name)
        if not images:
            print(
                f"Warning: no images found for class '{class_name}' under "
                f"{raw_dir}. Skipping."
            )
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR: images[n_train:n_train + n_val],
            TEST_DIR: images[n_train + n_val:],
        }

        for split_dir, split_images in splits.items():
            dest_class_dir = os.path.join(split_dir, class_name)
            for src_path in split_images:
                shutil.copy2(
                    src_path,
                    os.path.join(dest_class_dir, os.path.basename(src_path)),
                )

        print(
            f"{class_name}: {n} total -> "
            f"train={len(splits[TRAIN_DIR])} "
            f"val={len(splits[VAL_DIR])} "
            f"test={len(splits[TEST_DIR])}"
        )


if __name__ == "__main__":
    if not os.path.isdir(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        download_from_kaggle()
    else:
        print(f"Raw data already present at {RAW_DATA_DIR}, skipping download.")

    split_dataset()
    print("Dataset split complete.")
