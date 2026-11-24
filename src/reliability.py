"""Reliability metrics and synthetic input shifts for binary digit recognition.

Inputs are standardized to floating-point grayscale intensities in [0, 1].
The shifts are deterministic, not substitutes for external OOD datasets.
"""
from __future__ import annotations

import numpy as np


def probability_summary(y_true, probabilities, classes, bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if y_true.ndim != 1 or probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("probability shape does not match labels/classes")
    if len(y_true) == 0 or bins < 2:
        raise ValueError("need nonempty labels and at least two bins")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must lie within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("class probabilities must sum to one")

    pred = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if np.any(selected):
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "accuracy": float(correct.mean()),
        "expected_calibration_error": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def choose_abstention_threshold(y_validation, probabilities, classes, maximum_risk=0.05):
    """Select the largest coverage whose *observed validation* error <= risk.

    No guarantee about future or shifted inputs follows from this constraint.
    Returns >1 if all validation predictions must be rejected.
    """
    if not 0 <= maximum_risk < 1:
        raise ValueError("maximum_risk must be between 0 and 1")
    probability_summary(y_validation, probabilities, classes)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(classes)[probabilities.argmax(axis=1)] == np.asarray(y_validation)
    best = 1.01
    for threshold in np.unique(np.r_[0.0, confidence]):
        keep = confidence >= threshold
        if np.any(keep) and 1.0 - correct[keep].mean() <= maximum_risk:
            best = float(threshold)
            break
    return best


def selective_metrics(y_true, probabilities, classes, threshold):
    probability_summary(y_true, probabilities, classes)
    confidence = probabilities.max(axis=1)
    keep = confidence >= threshold
    pred = np.asarray(classes)[probabilities.argmax(axis=1)]
    return {
        "coverage": float(keep.mean()),
        "selected_accuracy": float(np.mean(pred[keep] == np.asarray(y_true)[keep])) if keep.any() else None,
        "abstained": int((~keep).sum()),
    }


def corrupt(images, condition, seed=42):
    """Apply one fixed corruption to a batch of 28x28 images in [0, 1]."""
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (28, 28):
        raise ValueError("expected batch of grayscale 28x28 images")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("pixel intensities must be finite within [0, 1]")
    if condition == "clean":
        return x.copy()
    if condition == "gaussian_noise_0.2":
        rng = np.random.default_rng(seed)
        return np.clip(x + rng.normal(0.0, 0.2, x.shape), 0, 1).astype(np.float32)
    if condition == "center_occlusion_8px":
        result = x.copy()
        result[:, 10:18, 10:18] = 0.0
        return result
    if condition == "rotation_20deg":
        from scipy.ndimage import rotate
        return np.clip(
            rotate(x, angle=20, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0),
            0, 1,
        ).astype(np.float32)
    raise ValueError(f"unknown corruption condition: {condition}")
