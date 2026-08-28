"""
Post-deployment model performance tracking.

Sends a small batch of real (or simulated) requests, each with a known
true label, to the deployed inference service's /predict endpoint, and
reports accuracy/precision/recall/F1 plus a confusion matrix and basic
latency stats. This closes the loop between "the service is up"
(/health, smoke tests) and "the service is still making good
predictions" once it's live.

Usage:
    # Against a locally port-forwarded / docker-run service
    python src/evaluate_deployment.py --url http://localhost:8000 --n 30

    # Against a specific split, with results saved to disk
    python src/evaluate_deployment.py \
        --url http://localhost:8000 \
        --data-dir data/processed/test \
        --n 50 \
        --out src/artifacts/deployment_eval.json

Expects images laid out as:
    <data-dir>/cat/*.jpg
    <data-dir>/dog/*.jpg
(the same ImageFolder-style layout produced by download_data.py)
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import TEST_DIR, CLASS_NAMES, RANDOM_SEED, ARTIFACT_DIR

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_labeled_samples(data_dir: str, n_per_class: int, seed: int):
    """
    Pull a small labeled batch from a class-per-folder directory.
    Mirrors what a real "sample live traffic + backfill true labels"
    process would produce, but sourced from the held-out test split so
    the true label is known ahead of time.
    """
    random.seed(seed)
    samples = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: {class_dir} not found, skipping class '{class_name}'")
            continue

        images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        random.shuffle(images)
        chosen = images[:n_per_class]

        if len(chosen) < n_per_class:
            print(
                f"WARNING: requested {n_per_class} '{class_name}' images, "
                f"only found {len(chosen)}"
            )

        samples.extend((path, class_name) for path in chosen)

    random.shuffle(samples)
    return samples


def call_predict(base_url: str, image_path: str, timeout: float = 10.0):
    """POST a single image to /predict and return (label, confidence, latency_s)."""
    url = f"{base_url.rstrip('/')}/predict"

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        start = time.time()
        response = requests.post(url, files=files, timeout=timeout)
        latency = time.time() - start

    response.raise_for_status()
    body = response.json()
    return body["label"], body["confidence"], latency


def check_health(base_url: str, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def run_evaluation(base_url: str, samples: list, verbose: bool = True):
    y_true, y_pred, latencies, failures = [], [], [], []

    for image_path, true_label in samples:
        try:
            pred_label, confidence, latency = call_predict(base_url, image_path)
        except Exception as exc:  # noqa: BLE001
            failures.append({"image": image_path, "error": str(exc)})
            if verbose:
                print(f"FAILED  {os.path.basename(image_path):30s} error={exc}")
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies.append(latency)

        if verbose:
            mark = "OK " if pred_label == true_label else "MISS"
            print(
                f"{mark}  {os.path.basename(image_path):30s} "
                f"true={true_label:4s} pred={pred_label:4s} "
                f"conf={confidence:.3f} latency={latency*1000:.0f}ms"
            )

    return y_true, y_pred, latencies, failures


def summarize(y_true, y_pred, latencies, failures):
    n = len(y_true)
    if n == 0:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_requests": 0,
            "n_failures": len(failures),
            "error": "No successful predictions collected.",
        }

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES).tolist()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_requests": n,
        "n_failures": len(failures),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="dog", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="dog", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="dog", zero_division=0),
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm},
        "latency_ms": {
            "mean": (sum(latencies) / len(latencies)) * 1000,
            "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1] * 1000,
            "max": max(latencies) * 1000,
        },
        "failures": failures,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Post-deployment performance check.")
    parser.add_argument("--url", required=True, help="Base URL of the deployed service, e.g. http://localhost:8000")
    parser.add_argument("--data-dir", default=TEST_DIR, help="Labeled image directory (class-per-folder layout)")
    parser.add_argument("--n", type=int, default=15, help="Number of images to sample PER CLASS")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", default=os.path.join(ARTIFACT_DIR, "deployment_eval.json"))
    args = parser.parse_args()

    print(f"Checking service health at {args.url}/health ...")
    if not check_health(args.url):
        raise SystemExit(f"Service at {args.url} is not healthy. Aborting evaluation.")
    print("Service is healthy.\n")

    samples = collect_labeled_samples(args.data_dir, args.n, args.seed)
    if not samples:
        raise SystemExit(f"No labeled samples found under {args.data_dir}.")

    print(f"Sending {len(samples)} labeled requests to {args.url}/predict ...\n")
    y_true, y_pred, latencies, failures = run_evaluation(args.url, samples)

    summary = summarize(y_true, y_pred, latencies, failures)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("POST-DEPLOYMENT PERFORMANCE SUMMARY")
    print("=" * 50)
    for key in ("n_requests", "n_failures", "accuracy", "precision", "recall", "f1"):
        if key in summary:
            print(f"{key:>12s}: {summary[key]}")
    print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
