"""
Central configuration for the Cats vs Dogs MLOps pipeline.
Keeping these in one place avoids magic numbers scattered across
data prep, training, and serving code (which must all agree on
image size / normalization / class order).
"""
import os

# --- Paths -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")

MLRUNS_DIR = os.path.join(SCRIPT_DIR, "mlruns").replace("\\", "/")
TRACKING_URI = f"file:///{MLRUNS_DIR}"

# --- Data --------------------------------------------------------------
IMAGE_SIZE = 224  # standard CNN input size (e.g. ResNet/EfficientNet family)
CLASS_NAMES = ["cat", "dog"]  # index 0 = cat, index 1 = dog

# Split ratios
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ImageNet normalization stats
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- Training -----------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
