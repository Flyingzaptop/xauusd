"""Independent read-only audit of completed v3.1 fits, calibration and evaluation.

Do not run until all 15 fits, 30 calibrators and exposed evaluation are finished.
No model fitting, calibration optimization or checkpoint mutation occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.market_data_v31 import load_split


FROZEN_COMMIT = "8c45bbf0cafc8397384dde87bd8c84aceedb7d62"
V3_BASE_COMMIT = "45a4aacc2430fccb2cb5aa74b5a7923e7f16f7bc"
V2_BASE_COMMIT = "e4559e62748b919a02c841dc5d3804e9c391b4ea"
MODELS = ("a", "b", "c", "no_transfer", "no_open")
SEEDS = (7, 19, 43)
CHOICES = ("short_best", "full_best")
MODES = ("raw", "calibrated")
SELECTED_METRIC_ATOL = 1e-8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label: str, atol: float = 1e-9) -> None:
    if actual is None or expected is None:
        require(actual is None and expected is None, f"{label}: missing value mismatch")
        return
    a, b = float(actual), float(expected)
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol,
            f"{label}: {a} != {b} within {atol}")


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_preservation() -> dict:
    protected_v3 = (
        "configs/market-v3.json", "docs/market-v3-protocol.md", "docs/market-v3-budget.md",
        "docs/market-v3-reproduction.md", "src/market_data_v3.py", "src/stream_batch_v3.py",
        "src/stream_models_v3.py", "scripts/run_market_v3.py", "reports/v3",
    )
    protected_v2 = (
        "src/crossing_memory.py", "src/crossing_memory_v2.py", "src/synthetic_v2.py",
        "scripts/run_experiment.py", "scripts/run_mechanism_v2.py", "scripts/probe_memory_v2.py",
        "configs/experiment.json", "configs/mechanism-v2.json", "docs/operator.md",
        "docs/operator-v2.md", "reports/v2", "reports/market_metrics.json",
        "reports/market_metrics.csv", "reports/artifact-manifest.json",
        "reports/crossing-memory-v1.md",
    )
    for base, paths in ((V3_BASE_COMMIT, protected_v3), (V2_BASE_COMMIT, protected_v2)):
        require(git("rev-parse", base) == base, f"Protected base missing: {base}")
        require(not git("diff", "--name-only", base, "HEAD", "--", *paths),
                f"Protected committed paths changed since {base}")
        require(not git("status", "--porcelain", "--", *paths),
                f"Protected working paths changed since {base}")
    manifests = []
    for name in ("reports/artifact-manifest.json", "reports/v2/artifact-manifest.json",
                 "reports/v3/artifact-manifest.json"):
        path = ROOT / name
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            actual = relative_path(artifact["path"])
            require(actual.is_file() and actual.stat().st_size == artifact["bytes"],
                    f"Protected artifact missing or resized: {actual}")
            require(digest(actual) == artifact["sha256"], f"Protected artifact hash changed: {actual}")
        manifests.append({"manifest": name, "sha256": digest(path),
                          "artifacts_verified": len(manifest["artifacts"])})
    return {"v3_base_commit": V3_BASE_COMMIT, "v2_base_commit": V2_BASE_COMMIT,
            "protected_diffs_empty": True, "artifact_manifests": manifests}


def manual_stop(losses: list[float], cfg: dict) -> dict:
    """Replay the published rule without importing EarlyStop."""
    minimum = cfg["min_epochs"]
    maximum = cfg["max_epochs"]
    delta = cfg["min_delta"]
    patience = cfg["patience"]
    require(minimum == 4 and maximum == 20 and patience == 5 and delta == 1e-5,
            "Stopping policy differs from preregistration")
    require(9 <= len(losses) <= 20 and all(math.isfinite(value) for value in losses),
            "Invalid completed selection-loss trajectory")
    short_index = min(range(minimum), key=lambda index: losses[index])
    short_epoch = short_index + 1
    exact_index = min(range(len(losses)), key=lambda index: losses[index])
    exact_epoch = exact_index + 1
    reference = losses[short_index]
    bad = 0
    decisions = []
    running_best = math.inf
    running_best_epoch = None
    stop_epoch = None
    stop_reason = None
    for epoch, loss in enumerate(losses, 1):
        strict = loss < running_best
        if strict:
            running_best = loss
            running_best_epoch = epoch
        significant = False
        if epoch == minimum:
            reference = min(losses[:minimum])
            bad = 0
        elif epoch > minimum:
            significant = loss <= reference - delta
            if significant:
                reference = loss
                bad = 0
            else:
                bad += 1
        patience_stop = epoch > minimum and bad >= patience
        budget_stop = epoch >= maximum
        stopped = patience_stop or budget_stop
        reason = "patience" if patience_stop else "budget_cap" if budget_stop else None
        decisions.append({"epoch": epoch, "save_full_best": strict,
                          "save_short_best": epoch == minimum,
                          "best_epoch": running_best_epoch, "best_loss": running_best,
                          "short_best_epoch": short_epoch if epoch >= minimum else None,
                          "short_best_loss": losses[short_index] if epoch >= minimum else None,
                          "reference_loss": reference if epoch >= minimum else None,
                          "significant_improvement": significant,
                          "bad_epochs": bad, "patience_exhausted": patience_stop,
                          "budget_cap_reached": budget_stop, "stop": stopped,
                          "stop_reason": reason})
        if stopped:
            stop_epoch = epoch
            stop_reason = reason
            break
    require(stop_epoch == len(losses),
            f"Recorded {len(losses)} epochs, manual rule stops at {stop_epoch}")
    return {"short_best_epoch": short_epoch, "full_best_epoch": exact_epoch,
            "stop_reason": stop_reason, "decisions": decisions,
            "significant_reference_loss": reference, "bad_epochs": bad,
            "last_five_loss_change": losses[-1] - losses[-5]}


def tensor_tree_finite(value, label: str) -> int:
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all().item(), f"Nonfinite tensor in {label}")
        return value.numel()
    if isinstance(value, dict):
        return sum(tensor_tree_finite(child, f"{label}/{key}") for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_finite(child, f"{label}/{index}") for index, child in enumerate(value))
    return 0


def state_tensors_equal(left: dict, right: dict, label: str) -> None:
    require(left.keys() == right.keys(), f"{label}: model key mismatch")
    for key in left:
        require(torch.equal(left[key], right[key]), f"{label}: weight changed: {key}")


def nested_equal(left, right) -> bool:
    """Exact equality for trusted checkpoint trees including tensors and RNG arrays."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() \
            and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) \
            and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def verify_checkpoint(record: dict, role: str, expected_epoch: int,
                      expected_loss: float, provenance: dict) -> tuple[dict, int]:
    entry = record["checkpoints"][role]
    path = relative_path(entry["path"])
    require(path.is_file() and digest(path) == entry["sha256"], f"Checkpoint hash mismatch: {path}")
    require(entry["epoch"] == expected_epoch, f"Checkpoint epoch mismatch: {path}")
    close(entry["selection_log_loss"], expected_loss, f"Checkpoint score {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload["schema_version"] == 1 and payload["provenance"] == provenance,
            f"Checkpoint schema/provenance mismatch: {path}")
    require((payload["model"], payload["seed"], payload["epoch"]) ==
            (record["model"], record["seed"], expected_epoch), f"Checkpoint identity mismatch: {path}")
    close(payload["selection_log_loss"], expected_loss, f"Checkpoint payload loss {path}")
    require(payload["total_optimizer_updates"] == 294 * expected_epoch,
            f"Checkpoint update count mismatch: {path}")
    require(len(payload["epoch_curve"]) == expected_epoch and
            len(payload["batch_curve"]) == 294 * expected_epoch,
            f"Checkpoint curve extent mismatch: {path}")
    require(set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"},
            f"Checkpoint RNG state incomplete: {path}")
    require(isinstance(payload["rng_state"]["torch_cpu"], torch.Tensor) and
            payload["rng_state"]["torch_cpu"].numel() > 0,
            f"Checkpoint Torch RNG state invalid: {path}")
    require("param_groups" in payload["optimizer_state_dict"] and
            "state" in payload["optimizer_state_dict"], f"Optimizer state absent: {path}")
    for parameter_state in payload["optimizer_state_dict"]["state"].values():
        if "step" in parameter_state:
            step = float(parameter_state["step"])
            require(0 < step <= payload["total_optimizer_updates"],
                    f"Optimizer step out of range: {path}")
    weights = tensor_tree_finite(payload["model_state_dict"], str(path))
    optimizer_values = tensor_tree_finite(payload["optimizer_state_dict"], str(path))
    require(weights > 0 and optimizer_values > 0, f"Empty weights/optimizer state: {path}")
    require(payload["control_state"]["last_epoch"] == expected_epoch,
            f"Checkpoint controller epoch mismatch: {path}")
    return payload, weights


def verify_fits(cfg: dict, budget: dict, artifact_dir: Path, report_dir: Path) -> tuple[dict, list[dict]]:
    require(tuple(cfg["models"]) == MODELS and tuple(cfg["seeds"]) == SEEDS,
            "Fit matrix differs from preregistered five models/three seeds")
    require(budget["splits"]["train"]["supervised_rows"] == 491102 and
            budget["splits"]["train"]["chunks_with_valid_labels"] == 294,
            "Frozen train counts differ")
    common_provenance = None
    findings = []
    for model in MODELS:
        for seed in SEEDS:
            path = report_dir / f"fit-{model}-{seed}.json"
            require(path.is_file(), f"Missing fit report: {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            require(record.get("complete") is True and record["config"] == cfg and
                    (record["model"], record["seed"]) == (model, seed),
                    f"Incomplete or wrong fit: {path}")
            provenance = record["provenance"]
            if common_provenance is None:
                common_provenance = provenance
            require(provenance == common_provenance and provenance["code_commit"] == FROZEN_COMMIT,
                    f"Fit source/config provenance mismatch: {path}")
            epochs = record["epochs_completed"]
            curve = record["epoch_curve"]
            batches = record["batch_curve"]
            require(len(curve) == epochs and len(batches) == 294 * epochs and
                    record["total_optimizer_updates"] == 294 * epochs,
                    f"Epoch/batch/update count mismatch: {path}")
            require(record["supervised_examples_per_epoch"] == 491102 and
                    record["all_gradients_finite"] is True,
                    f"Supervised or gradient report mismatch: {path}")
            require(math.isfinite(record["maximum_gradient_norm_before_clip"]),
                    f"Nonfinite reported gradient maximum: {path}")
            for name, value in record["maximum_state_abs"].items():
                require(math.isfinite(value) and value >= 0, f"Nonfinite state maximum: {path}/{name}")
            losses = [row["selection"]["log_loss"] for row in curve]
            manual = manual_stop(losses, cfg)
            require(record["stop_reason"] == manual["stop_reason"], f"Wrong stopping reason: {path}")
            for epoch, (row, expected) in enumerate(zip(curve, manual["decisions"]), 1):
                require(row["epoch"] == epoch and row["supervised"] == 491102 and
                        row["optimizer_updates"] == 294, f"Epoch budget mismatch: {path}/e{epoch}")
                published = row["control"]
                for field, value in expected.items():
                    if isinstance(value, float):
                        close(published[field], value, f"Manual stop {path}/e{epoch}/{field}")
                    else:
                        require(published[field] == value,
                                f"Manual stop {path}/e{epoch}/{field} differs")
                chunk = batches[(epoch - 1) * 294:epoch * 294]
                require([item["epoch"] for item in chunk] == [epoch] * 294 and
                        [item["chunk"] for item in chunk] == list(range(1, 295)) and
                        [item["update"] for item in chunk] == list(range((epoch - 1) * 294 + 1,
                                                                          epoch * 294 + 1)),
                        f"Batch numbering mismatch: {path}/e{epoch}")
                require(sum(item["n"] for item in chunk) == 491102,
                        f"Batch supervised population mismatch: {path}/e{epoch}")
                require(all(math.isfinite(item["loss"]) and
                            math.isfinite(item["gradient_norm_before_clip"]) for item in chunk),
                        f"Nonfinite batch record: {path}/e{epoch}")
                weighted_loss = sum(item["loss"] * item["n"] for item in chunk) / 491102
                close(row["train_log_loss"], weighted_loss, f"Weighted train loss {path}/e{epoch}")
            require(record["control_state"]["last_epoch"] == epochs and
                    record["control_state"]["best_epoch"] == manual["full_best_epoch"],
                    f"Final control state mismatch: {path}")
            short, short_weights = verify_checkpoint(record, "short_best", manual["short_best_epoch"],
                                                    losses[manual["short_best_epoch"] - 1], provenance)
            full, full_weights = verify_checkpoint(record, "full_best", manual["full_best_epoch"],
                                                  losses[manual["full_best_epoch"] - 1], provenance)
            last, last_weights = verify_checkpoint(record, "last", epochs, losses[-1], provenance)
            require(short_weights == full_weights == last_weights,
                    f"Checkpoint model shapes differ: {path}")
            require(short_weights == record["parameters"],
                    f"Checkpoint parameter count differs: {path}")
            require(last["control_state"] == record["control_state"] and
                    last["epoch_curve"] == curve and last["batch_curve"] == batches,
                    f"Recovery payload differs from final report: {path}")
            state_tensors_equal(last["short_candidate_payload"]["model_state_dict"],
                                short["model_state_dict"], f"Short recovery candidate {path}")
            state_tensors_equal(last["full_best_payload"]["model_state_dict"],
                                full["model_state_dict"], f"Full recovery candidate {path}")
            require(nested_equal(last["short_candidate_payload"], short) and
                    nested_equal(last["full_best_payload"], full),
                    f"Recovery embedded candidates differ from selected model/optimizer/RNG checkpoints: {path}")
            v3 = json.loads((ROOT / f"reports/v3/fit-{model}-{seed}.json").read_text(encoding="utf-8"))["runs"][0]
            require(len(v3["batch_curve"]) >= 1176 and len(v3["epoch_curve"]) >= 4,
                    f"v3 four-epoch reference incomplete: {model}/{seed}")
            for index in range(1176):
                old, new = v3["batch_curve"][index], batches[index]
                for field in ("epoch", "chunk", "update", "n"):
                    require(old[field] == new[field], f"v3/v3.1 first-four batch identity differs: {model}/{seed}/{index}")
                for field in ("loss", "gradient_norm_before_clip"):
                    close(old[field], new[field], f"v3/v3.1 first-four {field}: {model}/{seed}/{index}", atol=1e-7)
            for index in range(4):
                close(v3["epoch_curve"][index]["train_log_loss"], curve[index]["train_log_loss"],
                      f"v3/v3.1 first-four train loss: {model}/{seed}/{index}", atol=1e-7)
            findings.append({"model": model, "seed": seed, "epochs": epochs,
                             "stop_reason": manual["stop_reason"],
                             "short_best_epoch": manual["short_best_epoch"],
                             "full_best_epoch": manual["full_best_epoch"],
                             "last_five_loss_change": manual["last_five_loss_change"],
                             "updates": 294 * epochs,
                             "first_four_train_matches_v3": True,
                             "checkpoint_sha256": {role: record["checkpoints"][role]["sha256"]
                                                   for role in ("short_best", "full_best", "last")}})
    require(len(findings) == 15, "Not all 15 fits were verified")
    require(common_provenance["config_sha256"] == digest(ROOT / "configs/market-v31.json"),
            "Frozen configuration hash mismatch")
    for source, expected in common_provenance["source_sha256"].items():
        require(digest(ROOT / source) == expected, f"Frozen source hash mismatch: {source}")
    for split, expected in common_provenance["cache_sha256"].items():
        require(digest(artifact_dir / f"{split}.npz") == expected,
                f"Prepared split cache hash mismatch: {split}")
    require(digest(artifact_dir / "audit.json") == common_provenance["preprocessing_audit_sha256"],
            "Preprocessing audit hash mismatch")
    audit = json.loads((artifact_dir / "audit.json").read_text(encoding="utf-8"))
    require(audit["provenance"]["config_sha256"] == common_provenance["config_sha256"] and
            audit["provenance"]["cache_sha256"] == common_provenance["cache_sha256"],
            "Cache audit provenance differs")
    total_updates = sum(item["updates"] for item in findings)
    require(39690 <= total_updates <= 88200, "Actual optimizer updates outside preregistered bounds")
    return common_provenance, findings


def metric(y: np.ndarray, logits: np.ndarray, ranking: np.ndarray | None = None) -> dict:
    """Independent implementation of the published raw-logit score definitions."""
    labels = np.asarray(y, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    require(labels.shape == z.shape and labels.ndim == 1 and len(labels) > 0 and
            np.isfinite(z).all(), "Invalid metric population/logits")
    probability = expit(z)
    ranking_scores = z if ranking is None else np.asarray(ranking, dtype=np.float64)
    require(ranking_scores.shape == z.shape and np.isfinite(ranking_scores).all(),
            "Invalid ranking scores")
    auc = float(roc_auc_score(labels, ranking_scores)) if len(np.unique(labels)) == 2 else None
    bins = np.minimum((probability * 10).astype(np.int64), 9)
    calibration = []
    ece = 0.0
    for number in range(10):
        mask = bins == number
        if mask.any():
            predicted = float(probability[mask].mean())
            observed = float(labels[mask].mean())
            calibration.append({"bin": number, "count": int(mask.sum()),
                                "predicted": predicted, "observed": observed})
            ece += float(mask.mean()) * abs(predicted - observed)
    return {"n": len(labels),
            "log_loss": float(np.mean(np.logaddexp(0, z) - labels * z)),
            "auc": auc, "brier": float(np.square(probability - labels).mean()),
            "ece": ece, "calibration": calibration}


def compare_metric(published: dict, computed: dict, label: str) -> None:
    require(published["n"] == computed["n"], f"{label}: metric population differs")
    for field in ("log_loss", "auc", "brier", "ece"):
        close(published[field], computed[field], f"{label}/{field}", atol=SELECTED_METRIC_ATOL)
    require(len(published["calibration"]) == len(computed["calibration"]),
            f"{label}: calibration table length differs")
    for old, new in zip(published["calibration"], computed["calibration"]):
        require(old["bin"] == new["bin"] and old["count"] == new["count"],
                f"{label}: calibration bin/count differs")
        close(old["predicted"], new["predicted"], f"{label}/predicted")
        close(old["observed"], new["observed"], f"{label}/observed")
    require(sum(item["count"] for item in computed["calibration"]) == computed["n"],
            f"{label}: calibration bins fail to cover all observations")


def verify_calibrators(cfg: dict, provenance: dict, fit_findings: list[dict],
                       artifact_dir: Path, report_dir: Path) -> tuple[dict, dict]:
    calibration_path = report_dir / "calibrators.json"
    freeze_path = report_dir / "calibration-freeze.json"
    require(calibration_path.is_file() and freeze_path.is_file(), "Calibration barrier not complete")
    published = json.loads(calibration_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(published["complete"] is True and published["provenance"] == provenance and
            published["config"] == cfg and
            published["phase"] == "calibration only; evaluation not loaded",
            "Calibration report provenance or phase differs")
    require(freeze["complete"] is True and freeze["entry_count"] == 30 and
            freeze["provenance"] == provenance and
            freeze["calibrators_sha256"] == digest(calibration_path),
            "Calibration freeze does not match 30 frozen identities")
    split = load_split(artifact_dir, "calibration")
    valid = split["valid"]
    labels = np.asarray(split["y"][valid], dtype=np.float64)
    timestamps = split["timestamp"][valid]
    require(len(labels) == 82081 and np.isin(labels, (0, 1)).all() and
            np.all(np.diff(timestamps) > 0), "Calibration population/sort differs")
    require(len(published["entries"]) == 30, "Calibration identity count differs")
    identities = [(e["model"], e["seed"], e["choice"]) for e in published["entries"]]
    expected = {(model, seed, choice) for model in MODELS for seed in SEEDS for choice in CHOICES}
    require(len(set(identities)) == 30 and set(identities) == expected,
            "Calibration identities missing or duplicated")
    fit_map = {(item["model"], item["seed"]): item for item in fit_findings}
    entries = {}
    logits_by_identity = {}
    near_constant = []
    for entry in published["entries"]:
        model, seed, choice = entry["model"], entry["seed"], entry["choice"]
        label = f"{model}/{seed}/{choice}"
        checkpoint = entry["checkpoint"]
        require(checkpoint["sha256"] == fit_map[(model, seed)]["checkpoint_sha256"][choice],
                f"Calibrator checkpoint does not match fit: {label}")
        source_checkpoint = relative_path(checkpoint["path"])
        require(digest(source_checkpoint) == checkpoint["sha256"],
                f"Calibration checkpoint altered: {label}")
        payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        weight_hasher = hashlib.sha256()
        for name, value in sorted(payload["model_state_dict"].items()):
            weight_hasher.update(name.encode())
            weight_hasher.update(value.detach().cpu().numpy().tobytes())
        require(entry["model_weights_sha256"] == weight_hasher.hexdigest(),
                f"Calibrator weight identity differs: {label}")
        logits_path = relative_path(entry["logits_path"])
        require(logits_path.is_file() and digest(logits_path) == entry["logits_sha256"],
                f"Calibration logits hash mismatch: {label}")
        with np.load(logits_path, allow_pickle=False) as file:
            require(set(file.files) == {"logits", "timestamp"},
                    f"Unexpected fields in calibration-only logits: {label}")
            logits = file["logits"].copy()
            np.testing.assert_array_equal(file["timestamp"], timestamps)
        require(logits.shape == (82081,) and np.isfinite(logits).all(),
                f"Calibration logits not finite/aligned: {label}")
        params = entry["parameters"]
        slope, intercept = float(params["a"]), float(params["b"])
        require(math.isfinite(slope) and math.isfinite(intercept) and slope >= 1e-6,
                f"Calibration violates positive-slope bound: {label}")
        require(params["status"] in {"optimizer_converged", "projected_gradient_converged"} and
                params["optimizer"] == {"method": "L-BFGS-B", "maxiter": 1000,
                                        "ftol": 1e-12, "gtol": 1e-8, "regularization": None},
                f"Calibration optimizer differs: {label}")
        if params["status"] == "projected_gradient_converged":
            require(params["optimizer_success"] is False and
                    params["projected_gradient_inf_norm"] <= 1e-6,
                    f"Projected-gradient acceptance differs: {label}")
        else:
            require(params["optimizer_success"] is True,
                    f"Converged calibrator lacks successful solver: {label}")
        require(params["n"] == 82081 and params["positive_count"] == int(labels.sum()) and
                params["negative_count"] == 82081 - int(labels.sum()) and
                params["iterations"] <= 1000,
                f"Calibration sample/iteration count differs: {label}")
        calibrated = slope * logits + intercept
        require(np.isfinite(calibrated).all(), f"Calibrated logits nonfinite: {label}")
        initial_loss = float(np.mean(np.logaddexp(0, logits) - labels * logits))
        final_loss = float(np.mean(np.logaddexp(0, calibrated) - labels * calibrated))
        residual = expit(calibrated) - labels
        gradient = np.asarray([np.mean(residual * logits), np.mean(residual)])
        projected = gradient.copy()
        boundary = slope <= 1e-6 + 1e-8
        if boundary and projected[0] > 0:
            projected[0] = 0.0
        close(params["initial_log_loss"], initial_loss, f"Calibration initial objective {label}")
        close(params["calibrated_log_loss"], final_loss, f"Calibration fitted objective {label}")
        for original, recalculated in zip(params["gradient"], gradient):
            close(original, recalculated, f"Calibration objective gradient {label}")
        close(params["projected_gradient_inf_norm"], float(np.max(np.abs(projected))),
              f"Calibration projected gradient {label}")
        require(params["slope_at_lower_boundary"] == boundary,
                f"Slope boundary flag differs: {label}")
        dynamic_range = float(slope * (logits.max() - logits.min()))
        close(params["calibrated_logit_dynamic_range"], dynamic_range,
              f"Calibration dynamic range {label}")
        require(params["nearly_constant_prediction"] == (boundary or dynamic_range <= 1e-3),
                f"Near-constant calibration flag differs: {label}")
        if params["nearly_constant_prediction"]:
            near_constant.append(label)
        compare_metric(entry["raw"], metric(labels, logits), f"Calibration raw {label}")
        compare_metric(entry["calibrated"], metric(labels, calibrated, logits),
                       f"Calibration corrected {label}")
        close(entry["raw"]["auc"], entry["calibrated"]["auc"],
              f"Calibration AUC invariance {label}")
        entries[(model, seed, choice)] = entry
        logits_by_identity[(model, seed, choice)] = logits
    for model in MODELS:
        for seed in SEEDS:
            short = entries[(model, seed, "short_best")]
            full = entries[(model, seed, "full_best")]
            same_weights = short["model_weights_sha256"] == full["model_weights_sha256"]
            require(short["reused_same_weights"] is False and
                    full["reused_same_weights"] == same_weights,
                    f"Same-weight reuse flag differs: {model}/{seed}")
            if same_weights:
                np.testing.assert_array_equal(logits_by_identity[(model, seed, "short_best")],
                                              logits_by_identity[(model, seed, "full_best")])
                require(short["parameters"] == full["parameters"],
                        f"Identical weights have different calibrators: {model}/{seed}")
    return {"entries": 30, "calibration_rows": 82081,
            "calibrators_sha256": digest(calibration_path),
            "calibration_freeze_sha256": digest(freeze_path),
            "near_constant_identities": near_constant,
            "calibration_only_logit_arrays_verified": 30}, entries


def expected_comparisons() -> list[tuple[tuple, tuple, str]]:
    specs = []
    for choice in CHOICES:
        for mode in MODES:
            for left, right in (("c", "a"), ("c", "b"), ("b", "a"),
                                ("c", "no_transfer"), ("c", "no_open")):
                specs.append(((left, choice, mode), (right, choice, mode), "between_models"))
            for model in MODELS:
                specs.append(((model, choice, mode),
                              ("constant", "train_frequency", "raw"), "constant_control"))
    for model in MODELS:
        for mode in MODES:
            specs.append(((model, "full_best", mode), (model, "short_best", mode),
                          "training_extension"))
        for choice in CHOICES:
            specs.append(((model, choice, "calibrated"), (model, choice, "raw"),
                          "calibration_effect"))
    require(len(specs) == 60 and len(set(specs)) == 60, "Comparison specification must have 60 identities")
    return specs


def verify_evaluation(cfg: dict, provenance: dict, calibration: dict, entries: dict,
                      artifact_dir: Path, report_dir: Path) -> dict:
    evaluation_path = report_dir / "evaluation.json"
    logits_path = artifact_dir / "evaluation-logits.npz"
    require(evaluation_path.is_file() and logits_path.is_file(), "Evaluation not complete")
    published = json.loads(evaluation_path.read_text(encoding="utf-8"))
    require(published["provenance"] == provenance and published["config"] == cfg and
            published["evaluation_status"] == "exposed exploratory research, not independent confirmation",
            "Evaluation provenance/scope differs")
    require(published["calibration_freeze_sha256"] == calibration["calibration_freeze_sha256"],
            "Evaluation did not use frozen calibrators")
    require(published["evaluated_selected_checkpoints"] == 30 and
            published["evaluated_variant_identities"] == 60,
            "Evaluation identity count differs")
    with np.load(logits_path, allow_pickle=False) as file:
        expected_keys = {"y", "timestamp"} | {
            f"{model}-{seed}-{choice}" for model in MODELS for seed in SEEDS for choice in CHOICES
        }
        require(set(file.files) == expected_keys, "Raw-logit NPZ lacks 30 exact identities")
        y = file["y"].copy()
        stamp = file["timestamp"].copy()
        raw_logits = {key: file[key].copy() for key in expected_keys - {"y", "timestamp"}}
    require(len(y) == len(stamp) == 221343 and np.isin(y, (0, 1)).all() and
            np.issubdtype(stamp.dtype, np.integer) and np.all(np.diff(stamp) > 0),
            "Exposed 221,343-row population/sort differs")
    for key, values in raw_logits.items():
        require(values.shape == (221343,) and np.isfinite(values).all(),
                f"Raw evaluation logits missing/nonfinite: {key}")
    prepared = load_split(artifact_dir, "evaluation")
    keep = prepared["valid"]
    np.testing.assert_array_equal(y, prepared["y"][keep])
    np.testing.assert_array_equal(stamp, prepared["timestamp"][keep])
    months = np.asarray(pd.to_datetime(stamp, unit="s", utc=True).strftime("%Y-%m"))
    month_names = sorted(set(months))
    require(month_names == [f"2026-{month:02d}" for month in range(1, 9)],
            "Exposed evaluation months differ")
    periods = ("exposed_evaluation", *month_names)
    train = load_split(artifact_dir, "train")
    train_frequency = float(train["y"][train["valid"]].mean())
    close(published["constant_train_probability"], train_frequency, "Train-frequency constant")
    require(0 < train_frequency < 1, "Degenerate train frequency")
    constant_logits = np.full(len(y), np.log(train_frequency / (1 - train_frequency)))
    score_vectors = {}
    for model in MODELS:
        for seed in SEEDS:
            for choice in CHOICES:
                key = f"{model}-{seed}-{choice}"
                original = raw_logits[key]
                parameters = entries[(model, seed, choice)]["parameters"]
                calibrated = parameters["a"] * original + parameters["b"]
                require(np.isfinite(calibrated).all(), f"Calibrated evaluation logits nonfinite: {key}")
                sorted_scores = calibrated[np.argsort(original, kind="stable")]
                require(not np.any(np.diff(sorted_scores) < 0), f"Positive calibration changed ranking: {key}")
                score_vectors[(model, seed, choice, "raw")] = original
                score_vectors[(model, seed, choice, "calibrated")] = calibrated
    score_vectors[("constant", "train_frequency", "raw")] = constant_logits
    row_lookup = {}
    for row in published["metrics"]:
        identity = (row["model"], row["seed"], row["choice"], row["mode"], row["period"])
        require(identity not in row_lookup, f"Duplicate metric identity: {identity}")
        row_lookup[identity] = row
    expected_metric_identities = {
        (model, seed, choice, mode, period)
        for model in MODELS for seed in SEEDS for choice in CHOICES
        for mode in MODES for period in periods
    } | {("constant", None, "train_frequency", "raw", period) for period in periods}
    require(len(row_lookup) == 549 and set(row_lookup) == expected_metric_identities,
            "Exposed metric matrix is not 60×9 plus constant×9")
    for identity, row in row_lookup.items():
        model, seed, choice, mode, period = identity
        logits = score_vectors[(model, choice, mode)] if model == "constant" else \
            score_vectors[(model, seed, choice, mode)]
        ranking = logits if model == "constant" else raw_logits[f"{model}-{seed}-{choice}"]
        mask = np.ones(len(y), dtype=np.bool_) if period == "exposed_evaluation" else months == period
        compare_metric(row, metric(y[mask], logits[mask], ranking[mask]),
                       f"Evaluation {model}/{seed}/{choice}/{mode}/{period}")
    for model in MODELS:
        for seed in SEEDS:
            for choice in CHOICES:
                for period in periods:
                    raw_auc = row_lookup[(model, seed, choice, "raw", period)]["auc"]
                    calibrated_auc = row_lookup[(model, seed, choice, "calibrated", period)]["auc"]
                    close(raw_auc, calibrated_auc,
                          f"Ranking invariance {model}/{seed}/{choice}/{period}")
    csv_path = report_dir / "metrics.csv"
    require(csv_path.is_file() and len(pd.read_csv(csv_path)) == 549,
            "Metrics CSV does not contain all 549 rows")

    def losses(identity: tuple) -> np.ndarray:
        model, choice, mode = identity
        if model == "constant":
            base = np.logaddexp(0, constant_logits) - y * constant_logits
            return np.repeat(base[None, :], 3, axis=0)
        return np.stack([
            np.logaddexp(0, score_vectors[(model, seed, choice, mode)]) -
            y * score_vectors[(model, seed, choice, mode)] for seed in SEEDS
        ])

    all_losses = {identity: losses(identity) for spec in expected_comparisons()
                  for identity in spec[:2]}
    week = (stamp + 3 * 86400) // (7 * 86400)
    unique_weeks, week_index = np.unique(week, return_inverse=True)
    week_counts = np.bincount(week_index, minlength=len(unique_weeks))
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        0, len(unique_weeks), size=(cfg["bootstrap_replicates"], len(unique_weeks))
    )
    specs = expected_comparisons()
    comparisons = published["comparisons"]
    require(len(comparisons) == 60, "Expected exactly 60 paired comparisons")
    primary_count = 0
    for index, ((left, right, family), row) in enumerate(zip(specs, comparisons)):
        label = f"comparison[{index}] {left}-{right}"
        require(tuple(row["left"]) == left and tuple(row["right"]) == right and
                row["family"] == family, f"Paired contrast order/identity differs: {label}")
        primary = left == ("c", "full_best", "calibrated") and right == ("a", "full_best", "calibrated")
        require(row["primary"] == primary, f"Primary contrast flag differs: {label}")
        primary_count += int(primary)
        per_seed = all_losses[left] - all_losses[right]
        minute_mean = per_seed.mean(axis=0)
        weekly_sums = np.bincount(week_index, weights=minute_mean, minlength=len(unique_weeks))
        bootstrap = weekly_sums[draws].sum(axis=1) / week_counts[draws].sum(axis=1)
        close(row["difference"], float(minute_mean.mean()), f"{label}/point")
        require(row["weeks"] == len(unique_weeks), f"{label}: week count differs")
        for original, recalculated in zip(row["ci95"], np.quantile(bootstrap, [0.025, 0.975])):
            close(original, float(recalculated), f"{label}/interval")
        require(set(row["per_seed"]) == {str(seed) for seed in SEEDS},
                f"{label}: seed list differs")
        for position, seed in enumerate(SEEDS):
            close(row["per_seed"][str(seed)], float(per_seed[position].mean()),
                  f"{label}/seed{seed}")
    require(primary_count == 1, "Primary full calibrated C-A contrast not unique")
    return {"evaluation_rows": 221343, "raw_logit_arrays": 30,
            "raw_logit_npz_sha256": digest(logits_path), "metric_rows_recomputed": 549,
            "comparisons_recomputed": 60, "primary": "full_best calibrated C - full_best calibrated A",
            "utc_month_counts": {month: int((months == month).sum()) for month in month_names},
            "calibration_bins_recomputed": True,
            "auc_preservation_verified_for_all_variants": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market-v31.json")
    parser.add_argument("--output", default="reports/v31/verification.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    artifact_dir = ROOT / cfg["output"]
    report_dir = ROOT / "reports/v31"
    budget = json.loads((report_dir / "budget.json").read_text(encoding="utf-8"))
    provenance, fits = verify_fits(cfg, budget, artifact_dir, report_dir)
    calibration, entries = verify_calibrators(cfg, provenance, fits, artifact_dir, report_dir)
    evaluation = verify_evaluation(cfg, provenance, calibration, entries,
                                   artifact_dir, report_dir)
    preserved = verify_preservation()
    result = {"status": "passed",
              "scope": "exposed exploratory evaluation; not independent confirmation",
              "frozen_source_commit": FROZEN_COMMIT,
              "config_sha256": digest(ROOT / args.config),
              "fit_count": 15,
              "total_optimizer_updates": sum(item["updates"] for item in fits),
              "training": fits,
              "calibration": calibration,
              "evaluation": evaluation,
              "preservation": preserved,
              "verifier_sha256": digest(Path(__file__))}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": "passed", "fits": 15,
                      "calibrators": 30, "evaluation_rows": evaluation["evaluation_rows"],
                      "output": str(output)}))


if __name__ == "__main__":
    main()
