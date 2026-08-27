"""
Downloads the Kaggle Cats vs Dogs dataset and prepares an
ImageFolder-compatible train/validation/test dataset.

Raw Kaggle structure:

    data/raw/catsvsdogs/
        train/
            cats/
            dogs/
        test/
            cats/
            dogs/

Processed structure:

    data/processed/
        train/
            cat/
            dog/
        val/
            cat/
            dog/
        test/
            cat/
            dog/

The Kaggle training data is split into:
    - TRAIN_SPLIT
    - VAL_SPLIT

The original Kaggle test set is kept untouched.

Requires Kaggle API credentials.

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


KAGGLE_DATASET = "salader/dogsvscats"

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# ---------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------

def download_from_kaggle(dest_dir: str = RAW_DATA_DIR) -> None:
    """Download and extract the raw dataset from Kaggle."""

    os.makedirs(dest_dir, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required. "
            "Install with 'pip install kaggle' "
            "and configure Kaggle API credentials."
        ) from exc

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=dest_dir,
        unzip=False,
    )

    # Find ZIP file dynamically.
    zip_files = [
        filename
        for filename in os.listdir(dest_dir)
        if filename.lower().endswith(".zip")
    ]

    if not zip_files:
        raise RuntimeError(
            f"No ZIP file found after Kaggle download in {dest_dir}"
        )

    for zip_filename in zip_files:

        zip_path = os.path.join(
            dest_dir,
            zip_filename,
        )

        print(f"Extracting {zip_filename}...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        os.remove(zip_path)

    print(f"Raw dataset available at: {dest_dir}")


# ---------------------------------------------------------------------
# Raw data validation
# ---------------------------------------------------------------------

def raw_data_exists(
    raw_dir: str = RAW_DATA_DIR,
) -> bool:
    """
    Check whether the expected Kaggle dataset exists.

    Expected:

        raw/catsvsdogs/train/cats
        raw/catsvsdogs/train/dogs
        raw/catsvsdogs/test/cats
        raw/catsvsdogs/test/dogs
    """

    required_dirs = [
        os.path.join(
            raw_dir,
            "catsvsdogs",
            "train",
            "cats",
        ),
        os.path.join(
            raw_dir,
            "catsvsdogs",
            "train",
            "dogs",
        ),
        os.path.join(
            raw_dir,
            "catsvsdogs",
            "test",
            "cats",
        ),
        os.path.join(
            raw_dir,
            "catsvsdogs",
            "test",
            "dogs",
        ),
    ]

    for directory in required_dirs:

        if not os.path.isdir(directory):
            return False

        images = [
            filename
            for filename in os.listdir(directory)
            if filename.lower().endswith(
                VALID_EXTENSIONS
            )
        ]

        if not images:
            return False

    return True


# ---------------------------------------------------------------------
# Collect images
# ---------------------------------------------------------------------

def _collect_images(
    directory: str,
) -> list[str]:
    """Return all image files from a directory."""

    if not os.path.isdir(directory):
        return []

    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith(
            VALID_EXTENSIONS
        )
    ]


# ---------------------------------------------------------------------
# Clear processed data
# ---------------------------------------------------------------------

def _clear_processed_dataset() -> None:
    """
    Remove previously generated processed data.

    This ensures repeated preprocessing runs do not leave stale files.
    """

    for split_dir in (
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ):

        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    for split_dir in (
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ):

        for class_name in CLASS_NAMES:

            os.makedirs(
                os.path.join(
                    split_dir,
                    class_name,
                ),
                exist_ok=True,
            )


# ---------------------------------------------------------------------
# Copy images
# ---------------------------------------------------------------------

def _copy_images(
    images: list[str],
    destination_dir: str,
) -> None:
    """Copy images into the destination directory."""

    os.makedirs(
        destination_dir,
        exist_ok=True,
    )

    for src_path in images:

        destination = os.path.join(
            destination_dir,
            os.path.basename(src_path),
        )

        shutil.copy2(
            src_path,
            destination,
        )


# ---------------------------------------------------------------------
# Preprocess / split
# ---------------------------------------------------------------------

def split_dataset(
    raw_dir: str = RAW_DATA_DIR,
) -> None:
    """
    Create processed train/validation/test datasets.

    The Kaggle training set is split into train/validation.

    The original Kaggle test set is copied unchanged into processed/test.
    """

    random.seed(RANDOM_SEED)

    _clear_processed_dataset()

    kaggle_root = os.path.join(
        raw_dir,
        "catsvsdogs",
    )

    raw_train_dir = os.path.join(
        kaggle_root,
        "train",
    )

    raw_test_dir = os.path.join(
        kaggle_root,
        "test",
    )

    for class_name in CLASS_NAMES:

        # -------------------------------------------------------------
        # Map our class names to Kaggle directory names
        # -------------------------------------------------------------

        kaggle_class_name = (
            "cats"
            if class_name.lower() == "cat"
            else "dogs"
        )

        raw_class_train_dir = os.path.join(
            raw_train_dir,
            kaggle_class_name,
        )

        raw_class_test_dir = os.path.join(
            raw_test_dir,
            kaggle_class_name,
        )

        # -------------------------------------------------------------
        # Get training images
        # -------------------------------------------------------------

        train_images = _collect_images(
            raw_class_train_dir
        )

        if not train_images:

            print(
                f"WARNING: No training images found for "
                f"class '{class_name}'"
            )

            continue

        random.shuffle(train_images)

        n = len(train_images)

        n_train = int(
            n * TRAIN_SPLIT
        )

        n_val = int(
            n * VAL_SPLIT
        )

        train_split_images = train_images[
            :n_train
        ]

        val_split_images = train_images[
            n_train:n_train + n_val
        ]

        # -------------------------------------------------------------
        # Get original Kaggle test images
        # -------------------------------------------------------------

        test_images = _collect_images(
            raw_class_test_dir
        )

        # -------------------------------------------------------------
        # Copy processed datasets
        # -------------------------------------------------------------

        processed_train_dir = os.path.join(
            TRAIN_DIR,
            class_name,
        )

        processed_val_dir = os.path.join(
            VAL_DIR,
            class_name,
        )

        processed_test_dir = os.path.join(
            TEST_DIR,
            class_name,
        )

        _copy_images(
            train_split_images,
            processed_train_dir,
        )

        _copy_images(
            val_split_images,
            processed_val_dir,
        )

        _copy_images(
            test_images,
            processed_test_dir,
        )

        print(
            f"{class_name}: "
            f"raw_train={n:,} -> "
            f"train={len(train_split_images):,}, "
            f"val={len(val_split_images):,}; "
            f"raw_test={len(test_images):,} -> "
            f"test={len(test_images):,}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Cats vs Dogs - Data Preparation")
    print("=" * 60)

    if not raw_data_exists():

        print(
            "Raw dataset not found. "
            "Downloading from Kaggle..."
        )

        download_from_kaggle()

    else:

        print(
            f"Raw dataset already present at "
            f"{RAW_DATA_DIR}. Skipping download."
        )

    print()
    print(
        "Creating train/validation/test dataset..."
    )

    split_dataset()

    print()
    print("Dataset preparation complete.")
    print("=" * 60)