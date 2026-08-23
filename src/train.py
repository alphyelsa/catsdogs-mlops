"""
Trains the baseline CNN on the Cats vs Dogs dataset, logging
parameters/metrics/artifacts to MLflow and saving the final model
artifact for packaging into the Docker image.

Usage:
    python src/train.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless/CI-safe backend
import matplotlib.pyplot as plt

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from config import (
    TRACKING_URI,
    ARTIFACT_DIR,
    PLOTS_DIR,
    MODEL_DIR,
    MODEL_PATH,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    RANDOM_SEED,
    CLASS_NAMES,
)
from pipelines import get_dataloaders
from model import build_model

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("CatsVsDogs_Classification")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach()
            preds = torch.argmax(logits, dim=1).detach()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }
    if len(set(all_labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)

    return metrics, all_labels, all_preds, all_probs


def save_confusion_matrix(labels, preds, split_name: str) -> str:
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot()
    plt.title(f"Confusion Matrix - {split_name}")
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{split_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_roc_curve(labels, probs, split_name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax)
    plt.title(f"ROC Curve - {split_name}")
    path = os.path.join(PLOTS_DIR, f"roc_curve_{split_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_loss_curve(train_losses, val_losses) -> str:
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    path = os.path.join(PLOTS_DIR, "loss_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def train_and_log() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, classes = get_dataloaders(BATCH_SIZE)
    print(f"Classes: {classes}")
    print(f"Train/Val/Test sizes: "
          f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run(run_name="CatsDogsCNN"):
        mlflow.log_params({
            "model": "CatsDogsCNN",
            "image_size": 224,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "optimizer": "Adam",
        })

        train_losses, val_losses = [], []
        best_val_auc = -1.0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics, *_ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_metrics, val_labels, val_preds, val_probs = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            train_losses.append(train_metrics["loss"])
            val_losses.append(val_metrics["loss"])

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics.get("roc_auc", 0.0),
            }, step=epoch)

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_auc={val_metrics.get('roc_auc', float('nan')):.4f}"
            )

            if val_metrics.get("roc_auc", 0.0) > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                torch.save(model.state_dict(), MODEL_PATH)

        # Final evaluation on held-out test set using the best checkpoint
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        test_metrics, test_labels, test_preds, test_probs = run_epoch(
            model, test_loader, criterion, optimizer, device, train=False
        )

        print("\n==============================")
        print("TEST SET PERFORMANCE")
        print("==============================")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        cm_path = save_confusion_matrix(test_labels, test_preds, "test")
        roc_path = save_roc_curve(test_labels, test_probs, "test")
        loss_path = save_loss_curve(train_losses, val_losses)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(loss_path)

        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

        metrics_path = os.path.join(ARTIFACT_DIR, "model_comparison.json")
        with open(metrics_path, "w") as f:
            json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
        mlflow.log_artifact(metrics_path)

        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    train_and_log()
