import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.calibration_v31 import SLOPE_LOWER_BOUND, apply_affine, fit_affine


def test_calibration_recovers_positive_slope_and_reduces_loss():
    rng = np.random.default_rng(3101)
    logits = rng.normal(size=25000)
    true_probability = 1 / (1 + np.exp(-(1.7 * logits - 0.4)))
    labels = (rng.random(len(logits)) < true_probability).astype(np.int64)
    result = fit_affine(logits, labels)
    assert result["status"] in {"optimizer_converged", "projected_gradient_converged"}
    assert result["optimizer_success"]
    assert result["calibrated_log_loss"] <= result["initial_log_loss"]
    assert result["a"] == pytest.approx(1.7, abs=0.12)
    assert result["b"] == pytest.approx(-0.4, abs=0.08)
    assert not result["slope_at_lower_boundary"]
    calibrated_logits, probabilities = apply_affine(logits, result)
    assert np.isfinite(calibrated_logits).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert roc_auc_score(labels, calibrated_logits) == pytest.approx(roc_auc_score(labels, logits))


def test_opposite_ranking_hits_positive_slope_boundary_and_reports_near_constant():
    logits = np.tile(np.asarray([-3, -2, -1, 1, 2, 3], dtype=np.float64), 50)
    labels = np.tile(np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64), 50)
    result = fit_affine(logits, labels)
    assert result["a"] >= SLOPE_LOWER_BOUND
    assert result["slope_at_lower_boundary"]
    assert result["nearly_constant_prediction"]
    assert result["projected_gradient_inf_norm"] < 1e-5
    calibrated_logits, _ = apply_affine(logits, result)
    assert roc_auc_score(labels, calibrated_logits) == pytest.approx(roc_auc_score(labels, logits))


def test_stable_sigmoid_and_input_validation():
    scores, probabilities = apply_affine(np.asarray([-1000.0, 0.0, 1000.0]), {"a": 1.0, "b": 0.0})
    assert np.array_equal(scores, [-1000.0, 0.0, 1000.0])
    assert np.isfinite(probabilities).all()
    assert probabilities[0] == 0.0 and probabilities[-1] == 1.0
    with pytest.raises(ValueError):
        fit_affine(np.asarray([1.0, 2.0]), np.asarray([1, 1]))
    with pytest.raises(ValueError):
        fit_affine(np.asarray([1.0, np.nan]), np.asarray([1, 0]))
    with pytest.raises(ValueError):
        apply_affine(np.asarray([1.0]), {"a": -1.0, "b": 0.0})
