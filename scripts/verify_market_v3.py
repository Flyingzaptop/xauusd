"""Read-only completion audit for the frozen, exposed v3 market experiment.

Run only after all 15 fits and exposed evaluation have completed. This script
never fits a model or changes checkpoints. It reads evaluation labels solely
to verify already-published predictions and metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.market_data_v3 import load_split


FROZEN_SOURCE_COMMIT = "2c4da15e33bcd2c8527fae68230f325bc953499f"
PROTECTED_BASE = "e4559e62748b919a02c841dc5d3804e9c391b4ea"
PROTECTED_PATHS = (
    "src/crossing_memory.py", "src/crossing_memory_v2.py", "src/synthetic_v2.py",
    "scripts/run_experiment.py", "scripts/run_mechanism_v2.py", "scripts/probe_memory_v2.py",
    "configs/experiment.json", "configs/mechanism-v2.json", "docs/operator.md",
    "docs/operator-v2.md", "reports/v2", "reports/market_metrics.json",
    "reports/market_metrics.csv", "reports/artifact-manifest.json",
    "reports/crossing-memory-v1.md",
)
EXPECTED_SEEDS = (7, 19, 43)
EXPECTED_MODELS = ("a", "b", "c", "no_transfer", "no_open")
EXPECTED_EVALUATION_ROWS = 221343


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float | None, expected: float | None, label: str, atol: float = 1e-9) -> None:
    if actual is None or expected is None:
        require(actual is None and expected is None, f"{label}: missing value mismatch")
    else:
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f"{label}: nonfinite")
        require(abs(float(actual) - float(expected)) <= atol, f"{label}: {actual} != {expected}")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_protected_history() -> dict:
    require(git("rev-parse", PROTECTED_BASE) == PROTECTED_BASE, "Protected base commit is unavailable")
    committed = git("diff", "--name-only", PROTECTED_BASE, "HEAD", "--", *PROTECTED_PATHS)
    working = git("status", "--porcelain", "--", *PROTECTED_PATHS)
    require(not committed, f"Protected v1/v2 tracked files changed since base: {committed}")
    require(not working, f"Protected v1/v2 working files changed: {working}")
    manifests = []
    for relative in ("reports/artifact-manifest.json", "reports/v2/artifact-manifest.json"):
        path = ROOT / relative
        manifest = json.loads(path.read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        for entry in artifacts:
            artifact = ROOT / entry["path"]
            require(artifact.is_file(), f"Protected artifact missing: {artifact}")
            require(artifact.stat().st_size == entry["bytes"], f"Protected artifact size changed: {artifact}")
            require(sha256(artifact) == entry["sha256"], f"Protected artifact hash changed: {artifact}")
        manifests.append({"path": relative, "sha256": sha256(path), "artifacts_verified": len(artifacts)})
    return {"base_commit": PROTECTED_BASE, "tracked_diff_empty": True, "working_diff_empty": True,
            "manifests": manifests}


def verify_fits(cfg: dict, budget: dict, artifact_dir: Path, report_dir: Path) -> tuple[dict, list[dict]]:
    require(tuple(cfg["seeds"]) == EXPECTED_SEEDS and tuple(cfg["models"]) == EXPECTED_MODELS,
            "Frozen seed/model matrix differs from expected 15 fits")
    train_budget = budget["splits"]["train"]
    require(train_budget["optimizer_updates_per_epoch"] == 294, "Frozen updates/epoch differs")
    require(train_budget["supervised_rows"] == 491102, "Frozen supervised rows differ")
    require(budget["configuration"]["epochs"] == 4, "Frozen epoch count differs")
    require(budget["configuration"]["seeds"] == cfg["seeds"], "Budget seeds differ")
    require(budget["configuration"]["models"] == cfg["models"], "Budget models differ")
    require(train_budget["cache_sha256"] == sha256(artifact_dir / "train.npz"), "Train cache differs from budget")

    common_provenance = None
    records = []
    for model in cfg["models"]:
        for seed in cfg["seeds"]:
            path = report_dir / f"fit-{model}-{seed}.json"
            require(path.is_file(), f"Fit report missing: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            require(report.get("complete") is True and report["config"] == cfg, f"Incomplete/mismatched fit: {path}")
            require(len(report["runs"]) == 1, f"Fit report must contain one run: {path}")
            run = report["runs"][0]
            require((run["model"], run["seed"]) == (model, seed), f"Wrong fit identity: {path}")
            provenance = report["provenance"]
            if common_provenance is None:
                common_provenance = provenance
            require(provenance == common_provenance, f"Inconsistent provenance: {path}")
            require(provenance["code_commit"] == FROZEN_SOURCE_COMMIT, f"Unfrozen source commit: {path}")
            require(run["parameters"] == budget["model_parameters"][model], f"Parameter budget mismatch: {path}")
            require(run["total_optimizer_updates"] == 1176, f"Expected 4×294 updates: {path}")
            require(run["supervised_examples_per_epoch"] == 491102, f"Supervised count mismatch: {path}")
            require(run["all_gradients_finite"] is True, f"Nonfinite gradients reported: {path}")
            require(math.isfinite(run["maximum_gradient_norm_before_clip"]), f"Nonfinite max gradient: {path}")
            for value in run["maximum_state_abs"].values():
                require(math.isfinite(value) and value >= 0, f"Nonfinite/negative state maximum: {path}")
            curve = run["epoch_curve"]
            batch_curve = run["batch_curve"]
            require(len(curve) == 4 and len(batch_curve) == 1176, f"Curve length mismatch: {path}")
            require([row["epoch"] for row in curve] == [1, 2, 3, 4], f"Epoch numbering mismatch: {path}")
            require([row["update"] for row in batch_curve] == list(range(1, 1177)),
                    f"Update numbering mismatch: {path}")
            for epoch, row in enumerate(curve, 1):
                chunks = [batch for batch in batch_curve if batch["epoch"] == epoch]
                require(len(chunks) == 294 and [item["chunk"] for item in chunks] == list(range(1, 295)),
                        f"Chunk numbering mismatch: {path}/epoch{epoch}")
                require(sum(item["n"] for item in chunks) == 491102, f"Seen count mismatch: {path}/epoch{epoch}")
                require(row["supervised"] == 491102 and row["optimizer_updates"] == 294,
                        f"Epoch budget mismatch: {path}/epoch{epoch}")
                weighted = sum(item["loss"] * item["n"] for item in chunks) / 491102
                close(weighted, row["train_log_loss"], f"Weighted train loss {path}/epoch{epoch}")
                for item in chunks:
                    require(math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm_before_clip"]),
                            f"Nonfinite batch record: {path}/epoch{epoch}")
                require(math.isfinite(row["validation"]["log_loss"]), f"Nonfinite validation: {path}")
            best = min(range(4), key=lambda index: curve[index]["validation"]["log_loss"]) + 1
            require(run["best_epoch"] == best, f"Best epoch not selected by validation loss: {path}")
            stem = f"{model}-{seed}"
            checkpoint = artifact_dir / f"{stem}-best.pt"
            saved_val = artifact_dir / f"{stem}-validation.npz"
            require(checkpoint.is_file() and saved_val.is_file(), f"Missing selected artifact: {stem}")
            require(sha256(checkpoint) == run["checkpoint_sha256"], f"Checkpoint hash mismatch: {stem}")
            require(sha256(saved_val) == run["validation_predictions_sha256"],
                    f"Validation predictions hash mismatch: {stem}")
            records.append({"model": model, "seed": seed, "best_epoch": best,
                            "checkpoint_sha256": run["checkpoint_sha256"]})
    require(len(records) == 15, "Fit matrix is incomplete")
    require(common_provenance["config_sha256"] == sha256(ROOT / "configs/market-v3.json"),
            "Config hash mismatch")
    for relative, expected in common_provenance["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Frozen source hash mismatch: {relative}")
    for name, expected in common_provenance["cache_sha256"].items():
        require(sha256(artifact_dir / f"{name}.npz") == expected, f"Prepared cache hash mismatch: {name}")
    audit = json.loads((artifact_dir / "audit.json").read_text(encoding="utf-8"))
    require(audit["provenance"]["config_sha256"] == common_provenance["config_sha256"],
            "Prepared audit config hash mismatch")
    require(audit["provenance"]["data_sha256"] == common_provenance["data_sha256"],
            "Prepared audit data hash mismatch")
    require(sum(run["total_optimizer_updates"] for run in
                (json.loads((report_dir / f"fit-{m}-{s}.json").read_text())["runs"][0]
                 for m in cfg["models"] for s in cfg["seeds"])) == 17640,
            "Total optimizer updates differ from frozen 17,640")
    return common_provenance, records


def score(y: np.ndarray, probability: np.ndarray) -> dict:
    """Independent recalculation of the frozen metric definitions."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    require(len(y) == len(p) and len(y) > 0, "Empty/misaligned score population")
    loss = -y * np.log(p) - (1 - y) * np.log1p(-p)
    buckets = np.minimum((p * 10).astype(np.int64), 9)
    calibration = []
    ece = 0.0
    for number in range(10):
        mask = buckets == number
        if mask.any():
            predicted = float(p[mask].mean())
            observed = float(y[mask].mean())
            ece += float(mask.mean()) * abs(predicted - observed)
            calibration.append({"bin": number, "count": int(mask.sum()),
                                "predicted": predicted, "observed": observed})
    return {"n": len(y), "log_loss": float(loss.mean()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(np.square(p - y).mean()), "ece": ece,
            "calibration": calibration}


def compare_score(recorded: dict, computed: dict, label: str) -> None:
    require(recorded["n"] == computed["n"], f"{label}: population mismatch")
    for field in ("log_loss", "auc", "brier", "ece"):
        close(recorded[field], computed[field], f"{label}/{field}")
    require(len(recorded["calibration"]) == len(computed["calibration"]),
            f"{label}: calibration bin count mismatch")
    for expected, actual in zip(recorded["calibration"], computed["calibration"]):
        require(expected["bin"] == actual["bin"] and expected["count"] == actual["count"],
                f"{label}: calibration count mismatch")
        close(expected["predicted"], actual["predicted"], f"{label}/bin predicted")
        close(expected["observed"], actual["observed"], f"{label}/bin observed")
    require(sum(item["count"] for item in computed["calibration"]) == computed["n"],
            f"{label}: calibration does not cover every observation")


def verify_evaluation(cfg: dict, provenance: dict, artifact_dir: Path, report_dir: Path) -> dict:
    path = report_dir / "evaluation.json"
    predictions_path = artifact_dir / "evaluation-predictions.npz"
    require(path.is_file() and predictions_path.is_file(), "Evaluation artifacts are incomplete")
    published = json.loads(path.read_text(encoding="utf-8"))
    require(published["provenance"] == provenance and published["config"] == cfg,
            "Evaluation provenance/config differs from fits")
    require("exposed" in published["evaluation_status"] and "not independent" in published["evaluation_status"],
            "Evaluation scope must state exposed exploratory status")
    with np.load(predictions_path, allow_pickle=False) as file:
        expected_keys = {"timestamp", "y"} | {f"{model}-{seed}" for model in cfg["models"] for seed in cfg["seeds"]}
        require(set(file.files) == expected_keys, "Prediction NPZ columns differ from complete matrix")
        ts = file["timestamp"].copy()
        y = file["y"].copy()
        probabilities = {key: file[key].copy() for key in expected_keys - {"timestamp", "y"}}
    require(len(ts) == EXPECTED_EVALUATION_ROWS and len(y) == len(ts),
            "Prediction population is not the frozen 221,343 rows")
    require(np.issubdtype(ts.dtype, np.integer) and np.all(np.diff(ts) > 0),
            "Prediction timestamps are not strictly sorted UTC seconds")
    require(np.isin(y, (0, 1)).all(), "Prediction labels are not binary")
    for key, values in probabilities.items():
        require(len(values) == len(ts) and np.isfinite(values).all(), f"Missing/nonfinite probabilities: {key}")
        require(np.all((values >= 0) & (values <= 1)), f"Probabilities outside [0,1]: {key}")
    prepared = load_split(artifact_dir, "evaluation")
    keep = prepared["valid"]
    np.testing.assert_array_equal(ts, prepared["timestamp"][keep])
    np.testing.assert_array_equal(y, prepared["y"][keep])
    train = load_split(artifact_dir, "train")
    train_probability = float(train["y"][train["valid"]].mean())
    close(published["constant_train_probability"], train_probability, "Constant train prevalence")
    probabilities["constant-None"] = np.full(len(y), train_probability, dtype=np.float64)
    months = np.asarray(pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m"))
    month_names = sorted(set(months))
    expected_identities = {(model, seed) for model in cfg["models"] for seed in cfg["seeds"]} | {("constant", None)}
    expected_periods = {"exposed_evaluation", *month_names}
    rows = published["metrics"]
    identities = [(row["model"], row["seed"], row["period"]) for row in rows]
    require(len(rows) == len(expected_identities) * len(expected_periods)
            and len(set(identities)) == len(identities), "Missing/duplicate metric rows")
    require(set(identities) == {(model, seed, period) for model, seed in expected_identities for period in expected_periods},
            "Metric matrix does not cover all models/seeds/months")
    for row in rows:
        key = "constant-None" if row["model"] == "constant" else f"{row['model']}-{row['seed']}"
        mask = np.ones(len(y), dtype=np.bool_) if row["period"] == "exposed_evaluation" else months == row["period"]
        computed = score(y[mask], probabilities[key][mask])
        compare_score(row, computed, f"{key}/{row['period']}")

    def log_losses(values: np.ndarray) -> np.ndarray:
        p = np.clip(values, 1e-7, 1 - 1e-7)
        return -y * np.log(p) - (1 - y) * np.log1p(-p)

    losses = {
        model: np.stack([log_losses(probabilities[f"{model}-{seed}"]) for seed in cfg["seeds"]])
        for model in cfg["models"]
    }
    losses["constant"] = np.repeat(log_losses(probabilities["constant-None"])[None, :], len(cfg["seeds"]), axis=0)
    week_ids = (ts + 3 * 86400) // (7 * 86400)
    weeks = np.unique(week_ids)
    week_counts = np.asarray([(week_ids == week).sum() for week in weeks])
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    sampled = rng.integers(0, len(weeks), size=(cfg["bootstrap_replicates"], len(weeks)))
    contrasts = [("c", "b"), ("b", "a"), ("c", "a"), ("c", "no_transfer"),
                 ("c", "no_open")] + [(model, "constant") for model in cfg["models"]]
    published_contrasts = {item["comparison"]: item for item in published["paired_weekly_intervals"]}
    require(len(published_contrasts) == len(contrasts), "Paired contrast matrix incomplete/duplicated")
    for left, right in contrasts:
        name = f"{left} - {right}"
        require(name in published_contrasts, f"Missing paired contrast: {name}")
        recorded = published_contrasts[name]
        per_seed = losses[left] - losses[right]
        difference = per_seed.mean(axis=0)
        week_sums = np.asarray([difference[week_ids == week].sum() for week in weeks])
        bootstrap = week_sums[sampled].sum(axis=1) / week_counts[sampled].sum(axis=1)
        interval = np.quantile(bootstrap, [0.025, 0.975])
        close(recorded["difference"], float(difference.mean()), f"{name}/point")
        require(recorded["weeks"] == len(weeks), f"{name}: week count differs")
        for endpoint, actual in zip(recorded["ci95"], interval):
            close(endpoint, float(actual), f"{name}/interval")
        for index, seed in enumerate(cfg["seeds"]):
            close(recorded["per_seed"][str(seed)], float(per_seed[index].mean()), f"{name}/seed{seed}")
    csv_path = report_dir / "metrics.csv"
    require(csv_path.is_file(), "Metrics CSV missing")
    csv = pd.read_csv(csv_path)
    require(len(csv) == len(rows), "Metrics CSV row count differs from JSON")
    return {"prediction_npz_sha256": sha256(predictions_path), "rows": len(ts),
            "sorted_unique_timestamps": True, "model_probability_arrays": len(probabilities) - 1,
            "months": month_names, "month_counts": {month: int((months == month).sum()) for month in month_names},
            "metric_rows_recomputed": len(rows), "calibration_counts_recomputed": True,
            "paired_weekly_contrasts_recomputed": len(contrasts),
            "constant_train_probability": train_probability}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market-v3.json")
    parser.add_argument("--output", default="reports/v3/verification.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_dir = ROOT / cfg["output"]
    report_dir = ROOT / "reports/v3"
    budget = json.loads((report_dir / "frozen-budget.json").read_text(encoding="utf-8"))
    provenance, fit_records = verify_fits(cfg, budget, artifact_dir, report_dir)
    evaluated = verify_evaluation(cfg, provenance, artifact_dir, report_dir)
    protected = verify_protected_history()
    result = {
        "status": "passed",
        "scope": "exposed exploratory evaluation; not independent confirmation",
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "config_sha256": sha256(config_path),
        "fit_count": len(fit_records),
        "optimizer_updates_verified": 17640,
        "supervised_train_rows_per_epoch_per_fit": 491102,
        "fit_records": fit_records,
        "evaluation": evaluated,
        "protected_v1_v2": protected,
        "verifier_sha256": sha256(Path(__file__)),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "passed", "fits": 15, "rows": evaluated["rows"], "output": str(output)}))


if __name__ == "__main__":
    main()
