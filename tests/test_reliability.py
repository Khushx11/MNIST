import numpy as np
import pytest

from src.reliability import (
    corrupt, probability_summary, choose_abstention_threshold, selective_metrics
)


def test_perfect_calibration_metrics_shape_and_bounds():
    y = np.array([1, 7, 1, 7])
    p = np.array([[.9, .1], [.2, .8], [.95, .05], [.1, .9]])
    result = probability_summary(y, p, np.array([1, 7]))
    assert result["accuracy"] == 1.0
    assert 0 <= result["expected_calibration_error"] <= 1


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        probability_summary([1], [[0.5, float("nan")]], [1, 7])
    with pytest.raises(ValueError):
        probability_summary([1], [[0.2, 0.2]], [1, 7])
    with pytest.raises(ValueError):
        probability_summary([1, 7], [[1.0, 0.0]], [1, 7])


def test_abstention_threshold_selected_on_validation_predictions_only():
    y = np.array([1, 7, 1, 7])
    p = np.array([[.99, .01], [.01, .99], [.8, .2], [.55, .45]])
    threshold = choose_abstention_threshold(y, p, [1, 7], maximum_risk=0.0)
    result = selective_metrics(y, p, [1, 7], threshold)
    assert result["selected_accuracy"] == 1.0
    assert result["coverage"] == 0.75


def test_all_predictions_can_be_rejected_safely():
    threshold = choose_abstention_threshold([7], [[.99, .01]], [1, 7], maximum_risk=0)
    assert threshold > 1
    result = selective_metrics([7], [[.99, .01]], [1, 7], threshold)
    assert result["coverage"] == 0
    assert result["selected_accuracy"] is None


def test_corruptions_are_deterministic_and_bounded():
    image = np.zeros((2, 28, 28), dtype=np.float32)
    image[:, 10:18, 10:18] = 1.0
    for name in ["clean", "gaussian_noise_0.2", "center_occlusion_8px", "rotation_20deg"]:
        a = corrupt(image, name)
        b = corrupt(image, name)
        np.testing.assert_allclose(a, b)
        assert a.shape == image.shape
        assert np.isfinite(a).all()
        assert (0 <= a).all() and (a <= 1).all()


def test_corruptions_reject_unsupported_inputs():
    with pytest.raises(ValueError):
        corrupt(np.ones((28, 28)), "clean")
    with pytest.raises(ValueError):
        corrupt(np.ones((1, 28, 28)), "invented")


def test_risk_coverage_increases_when_validation_risk_constraint_relaxes():
    y = np.array([1, 7, 7, 1, 7])
    p = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
        [0.40, 0.60],
        [0.85, 0.15],
        [0.80, 0.20],
    ])
    risks = [0.0, 0.20, 0.40, 0.80]
    coverages = [
        selective_metrics(
            y, p, [1, 7],
            choose_abstention_threshold(y, p, [1, 7], maximum_risk=risk)
        )["coverage"]
        for risk in risks
    ]
    assert coverages == sorted(coverages)
    assert coverages[0] < coverages[-1]
