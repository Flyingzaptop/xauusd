"""Fixed positive-slope affine logit calibration for the v3.1 calibration split."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


SLOPE_LOWER_BOUND = 1e-6
MAX_ITERATIONS = 1000
FUNCTION_TOLERANCE = 1e-12
GRADIENT_TOLERANCE = 1e-8


def _inputs(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if z.ndim != 1 or y.ndim != 1 or len(z) != len(y) or len(z) == 0:
        raise ValueError("logits and labels must be nonempty aligned one-dimensional arrays")
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("logits and labels must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be binary 0/1")
    if len(np.unique(y)) != 2:
        raise ValueError("calibration requires both binary classes")
    return z, y


def _objective(parameters: np.ndarray, z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    slope, intercept = parameters
    scores = slope * z + intercept
    loss = np.logaddexp(0.0, scores) - y * scores
    residual = expit(scores) - y
    gradient = np.asarray([np.mean(residual * z), np.mean(residual)], dtype=np.float64)
    return float(np.mean(loss)), gradient


def fit_affine(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fit `sigmoid(a*z+b)`, `a>=1e-6`, with fixed L-BFGS-B settings.

    No validation/evaluation labels are accessed here; the caller must pass
    only the preregistered calibration split. The returned dict is JSON-safe.
    """
    z, y = _inputs(logits, labels)
    initial = np.asarray([1.0, 0.0], dtype=np.float64)
    initial_loss, _ = _objective(initial, z, y)
    result = minimize(
        _objective,
        initial,
        args=(z, y),
        method="L-BFGS-B",
        jac=True,
        bounds=((SLOPE_LOWER_BOUND, None), (None, None)),
        options={"maxiter": MAX_ITERATIONS, "ftol": FUNCTION_TOLERANCE,
                 "gtol": GRADIENT_TOLERANCE},
    )
    slope, intercept = (float(value) for value in result.x)
    final_loss, gradient = _objective(result.x, z, y)
    if not np.isfinite(result.x).all() or not np.isfinite(gradient).all() \
            or not np.isfinite(final_loss) or slope < SLOPE_LOWER_BOUND:
        raise RuntimeError("Affine calibration produced nonfinite or invalid parameters")
    boundary_hit = slope <= SLOPE_LOWER_BOUND + 1e-8
    projected_gradient = gradient.copy()
    if boundary_hit and projected_gradient[0] > 0:
        projected_gradient[0] = 0.0
    projected_inf = float(np.max(np.abs(projected_gradient)))
    if result.success:
        status = "optimizer_converged"
    elif projected_inf <= max(1e-6, 10 * GRADIENT_TOLERANCE):
        status = "projected_gradient_converged"
    else:
        raise RuntimeError(
            f"Affine calibration failed: {result.message}; projected gradient {projected_inf:.3g}"
        )
    dynamic_range = float(slope * (z.max() - z.min()))
    return {
        "a": slope,
        "b": intercept,
        "status": status,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "initial_log_loss": initial_loss,
        "calibrated_log_loss": final_loss,
        "gradient": [float(value) for value in gradient],
        "projected_gradient_inf_norm": projected_inf,
        "slope_lower_bound": SLOPE_LOWER_BOUND,
        "slope_at_lower_boundary": boundary_hit,
        "calibrated_logit_dynamic_range": dynamic_range,
        "nearly_constant_prediction": bool(boundary_hit or dynamic_range <= 1e-3),
        "optimizer": {"method": "L-BFGS-B", "maxiter": MAX_ITERATIONS,
                      "ftol": FUNCTION_TOLERANCE, "gtol": GRADIENT_TOLERANCE,
                      "regularization": None},
    }


def apply_affine(logits: np.ndarray, fit: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated logits and stable probabilities without clipping.

    Use returned logits for rank/AUC calculations; clipping probabilities can
    create numerical ties even though positive-slope affine ranks are invariant.
    """
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    slope = float(fit["a"])
    intercept = float(fit["b"])
    if not np.isfinite([slope, intercept]).all() or slope < SLOPE_LOWER_BOUND:
        raise ValueError("calibration parameters must be finite with positive admissible slope")
    scores = slope * z + intercept
    if not np.isfinite(scores).all():
        raise FloatingPointError("calibrated logits became nonfinite")
    return scores, expit(scores)
