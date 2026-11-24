"""Evaluate clean and shifted MNIST 1-vs-7 predictions without test-set tuning.

Data: official 60,000/10,000 MNIST split from OpenML. Validation is split
only from official training examples and selects an abstention threshold.
All corruption metrics use unchanged weights and the same threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from urllib.request import urlopen
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.reliability import (
    corrupt, probability_summary, choose_abstention_threshold, selective_metrics
)


CONDITIONS = [
    "clean",
    "gaussian_noise_0.2",
    "rotation_20deg",
    "center_occlusion_8px",
]


def dataset():
    """Read the official MNIST train/test split from TensorFlow's dataset mirror.

    Caches the original archive locally. No TensorFlow dependency is needed.
    """
    path = Path("data/mnist_full.npz")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        with urlopen(url, timeout=120) as response:
            body = response.read()
        path.write_bytes(body)
    with np.load(path, allow_pickle=False) as data:
        x_train = data["x_train"]
        y_train = data["y_train"].astype(int)
        x_test = data["x_test"]
        y_test = data["y_test"].astype(int)
    if x_train.shape != (60000, 28, 28) or x_test.shape != (10000, 28, 28):
        raise ValueError("Unexpected MNIST official dataset dimensions")
    train_mask = np.isin(y_train, [1, 7])
    test_mask = np.isin(y_test, [1, 7])
    return x_train[train_mask], y_train[train_mask], x_test[test_mask], y_test[test_mask]

def overlap_count(left, right):
    """Audit exact image duplicates, including cases with conflicting labels."""
    a = {hashlib.sha256(image.tobytes()).digest() for image in left}
    b = {hashlib.sha256(image.tobytes()).digest() for image in right}
    return len(a & b)


def run():
    x_train_raw, y_train_full, x_test_raw, y_test = dataset()
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_raw, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full,
    )
    if set(np.unique(y_train)) != {1, 7} or set(np.unique(y_test)) != {1, 7}:
        raise ValueError("Expected precisely two digit classes")

    train = x_train_raw.reshape(-1, 784).astype(np.float32) / 255.0
    val = x_val_raw.reshape(-1, 784).astype(np.float32) / 255.0
    model = LogisticRegression(C=0.1, solver="liblinear", max_iter=400, random_state=42)
    model.fit(train, y_train)
    val_prob = model.predict_proba(val)
    threshold = choose_abstention_threshold(y_val, val_prob, model.classes_, maximum_risk=0.005)
    evaluated = {}
    for condition in CONDITIONS:
        input_images = corrupt(x_test_raw.astype(np.float32) / 255.0, condition, seed=42)
        prob = model.predict_proba(input_images.reshape(-1, 784))
        pred = model.classes_[prob.argmax(axis=1)]
        results = probability_summary(y_test, prob, model.classes_)
        results.update(selective_metrics(y_test, prob, model.classes_, threshold))
        results["macro_f1"] = float(f1_score(y_test, pred, average="macro", zero_division=0))
        results["confusion_matrix"] = confusion_matrix(y_test, pred, labels=[1, 7]).tolist()
        # Brier score for digit 7, not a clinical calibration guarantee.
        index7 = list(model.classes_).index(7)
        results["brier_score_digit_7"] = float(np.mean(
            (prob[:, index7] - (y_test == 7).astype(float)) ** 2
        ))
        evaluated[condition] = results

    result = {
        "title": "Handwriting robustness, calibration and selective prediction",
        "dataset": "MNIST official Keras/TensorFlow 60k/10k split; digits 1 and 7",
        "model": "LogisticRegression(C=0.1, solver=liblinear)",
        "validation_protocol": "Stratified 20% of official training split; no held-out test tuning",
        "train_rows": len(y_train), "validation_rows": len(y_val), "test_rows": len(y_test),
        "train_label_counts": {str(v): int((y_train == v).sum()) for v in [1, 7]},
        "validation_risk_target": 0.005,
        "selected_confidence_threshold": threshold,
        "validation": probability_summary(y_val, val_prob, model.classes_),
        "validation_selective": selective_metrics(y_val, val_prob, model.classes_, threshold),
        "duplicate_audit": {
            "train_validation_exact_duplicates": overlap_count(x_train_raw, x_val_raw),
            "train_test_exact_duplicates": overlap_count(x_train_raw, x_test_raw),
            "validation_test_exact_duplicates": overlap_count(x_val_raw, x_test_raw),
        },
        "conditions": evaluated,
        "python": platform.python_version(),
        "limitations": [
            "Synthetic corruptions are not independent external datasets",
            "Risk threshold is selected on validation, with no coverage guarantee under shift",
            "Only digits 1 and 7 are modelled; all other digits are out of scope",
            "No adversarial robustness or deployment reliability guarantee",
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reliability_metrics.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "train_rows": result["train_rows"],
        "validation_rows": result["validation_rows"],
        "test_rows": result["test_rows"],
        "selected_confidence_threshold": result["selected_confidence_threshold"],
        "results": result["conditions"],
    }, indent=2))


if __name__ == "__main__":
    main()
