"""Render frozen market-v3 learning and exposed-evaluation figures.

Reads only completed fit reports and the exposed evaluation report. It never
loads model checkpoints or market source data. Batch curves use a causal,
label-count-weighted rolling window. Seed traces remain visible behind the
three-seed mean, with fixed colors across every panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("a", "b", "c", "no_transfer", "no_open")
COLORS = {"a": "#404b54", "b": "#7953a6", "c": "#087eaa",
          "no_transfer": "#d2691e", "no_open": "#168255", "constant": "#8f969d"}
NAMES = {"a": "A · candle GRU", "b": "B · routed GRU", "c": "C · tree memory",
         "no_transfer": "C without transfer", "no_open": "C without open readout",
         "constant": "Train-prior constant"}
SEED_COLUMNS = (
    "model", "seed", "gru_hidden", "parameters", "training_seconds", "best_epoch",
    "total_optimizer_updates", "supervised_examples_per_epoch", "maximum_gradient_norm_before_clip",
    "all_gradients_finite", "last_train_log_loss", "best_validation_log_loss",
    "last_validation_log_loss", "evaluation_log_loss", "evaluation_auc", "evaluation_brier",
    "evaluation_ece", "code_commit", "runner_sha256", "checkpoint_sha256",
    "fit_report_sha256", "evaluation_report_sha256", "plot_script_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(report_dir: Path, config_path: Path) -> tuple[dict, dict, dict]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    fits = {}
    provenance = None
    for model in cfg["models"]:
        for seed in cfg["seeds"]:
            path = report_dir / f"fit-{model}-{seed}.json"
            if not path.exists():
                raise RuntimeError(f"Missing fit report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report.get("complete") or report.get("config") != cfg or len(report.get("runs", [])) != 1:
                raise RuntimeError(f"Incomplete or mismatched fit report: {path}")
            run = report["runs"][0]
            if (run["model"], run["seed"]) != (model, seed):
                raise RuntimeError(f"Wrong model/seed in {path}")
            if provenance is None:
                provenance = report["provenance"]
            elif report["provenance"] != provenance:
                raise RuntimeError("Fit reports have inconsistent source/data provenance")
            if not run.get("all_gradients_finite"):
                raise RuntimeError(f"Nonfinite-gradient fit: {path}")
            fits[(model, seed)] = (run, sha256(path))
    if len(fits) != len(MODELS) * len(cfg["seeds"]):
        raise RuntimeError("Unexpected number of model/seed fits")
    evaluation_path = report_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise RuntimeError("Exposed evaluation report is not available yet")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("provenance") != provenance or evaluation.get("config") != cfg:
        raise RuntimeError("Evaluation provenance/config differs from fits")
    evaluation["_sha256"] = sha256(evaluation_path)
    return cfg, fits, evaluation


def rolling_weighted_loss(curve: list[dict], window: int) -> tuple[np.ndarray, np.ndarray]:
    updates = np.asarray([point["update"] for point in curve], dtype=np.int64)
    if len(updates) == 0 or not np.array_equal(updates, np.arange(1, len(updates) + 1)):
        raise ValueError("Batch update indices must be complete and consecutive")
    counts = np.asarray([point["n"] for point in curve], dtype=np.float64)
    losses = np.asarray([point["loss"] for point in curve], dtype=np.float64)
    if np.any(counts <= 0) or not np.isfinite(losses).all():
        raise ValueError("Invalid batch loss/count")
    total_count = np.r_[0.0, np.cumsum(counts)]
    total_loss = np.r_[0.0, np.cumsum(counts * losses)]
    ends = np.arange(1, len(curve) + 1)
    starts = np.maximum(0, ends - window)
    means = (total_loss[ends] - total_loss[starts]) / (total_count[ends] - total_count[starts])
    return updates, means


def _legend(fig, models: tuple[str, ...] = MODELS, *, y: float = 0.06) -> None:
    handles = [Line2D([0], [0], color=COLORS[name], linewidth=2.6, label=NAMES[name])
               for name in models]
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)),
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=9)


def _save(fig, output_dir: Path, stem: str) -> None:
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_batch_learning(cfg: dict, fits: dict, output_dir: Path, window: int) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.23)
    all_values = []
    for model in MODELS:
        curves = [rolling_weighted_loss(fits[(model, seed)][0]["batch_curve"], window)
                  for seed in cfg["seeds"]]
        updates = curves[0][0]
        if any(not np.array_equal(x, updates) for x, _ in curves[1:]):
            raise ValueError(f"Seed update axes differ for {model}")
        values = np.stack([y for _, y in curves])
        all_values.append(values)
        for seed_values in values:
            ax.plot(updates, seed_values, color=COLORS[model], alpha=0.20, linewidth=0.85)
        ax.plot(updates, values.mean(axis=0), color=COLORS[model], linewidth=2.2)
    values = np.concatenate([value.ravel() for value in all_values])
    spread = max(float(values.max() - values.min()), 0.002)
    ax.set_ylim(float(values.min()) - 0.08 * spread, float(values.max()) + 0.08 * spread)
    ax.set_xlim(1, int(updates[-1]))
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Training binary cross-entropy")
    ax.grid(True, color="#e7eaed", linewidth=0.7)
    fig.suptitle(f"Market v3 learning · {window}-update weighted rolling loss", y=0.96,
                 fontsize=16, fontweight="bold")
    _legend(fig, y=0.07)
    fig.text(0.5, 0.025, "Faint: each seed · Thick: mean of three seeds · Continuous state across chunks",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "training_batch_loss")


def plot_epoch_learning(cfg: dict, fits: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.23, wspace=0.07)
    values_all = []
    for model in MODELS:
        runs = [fits[(model, seed)][0] for seed in cfg["seeds"]]
        epochs = np.asarray([point["epoch"] for point in runs[0]["epoch_curve"]], dtype=int)
        if not np.array_equal(epochs, np.arange(1, cfg["epochs"] + 1)):
            raise ValueError(f"Unexpected epoch count for {model}")
        for panel, extract in ((0, lambda row: row["train_log_loss"]),
                               (1, lambda row: row["validation"]["log_loss"])):
            values = np.asarray([[extract(point) for point in run["epoch_curve"]]
                                 for run in runs], dtype=float)
            values_all.append(values)
            for trace in values:
                axes[panel].plot(epochs, trace, marker="o", markersize=3.5,
                                 color=COLORS[model], alpha=0.22, linewidth=1)
            axes[panel].plot(epochs, values.mean(axis=0), marker="o", markersize=5,
                             color=COLORS[model], linewidth=2.2)
    all_values = np.concatenate([value.ravel() for value in values_all])
    spread = max(float(all_values.max() - all_values.min()), 0.002)
    axes[0].set_ylim(float(all_values.min()) - 0.12 * spread,
                     float(all_values.max()) + 0.12 * spread)
    for axis, title in zip(axes, ("Training", "Chronological validation")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(np.arange(1, cfg["epochs"] + 1))
        axis.grid(True, color="#e7eaed", linewidth=0.7)
    axes[0].set_ylabel("Binary cross-entropy / log loss")
    fig.suptitle("Market v3 · four fixed epochs", y=0.96, fontsize=16, fontweight="bold")
    _legend(fig, y=0.07)
    fig.text(0.5, 0.025, "Faint: each seed · Thick: mean of three seeds · Best checkpoint selected by validation log loss",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "epoch_train_validation_loss")


def _evaluation_rows(evaluation: dict) -> dict[tuple[str, int | None, str], dict]:
    rows = {}
    for row in evaluation["metrics"]:
        key = (row["model"], row["seed"], row["period"])
        if key in rows:
            raise ValueError(f"Duplicate evaluation row {key}")
        rows[key] = row
    return rows


def plot_months(cfg: dict, evaluation: dict, output_dir: Path) -> None:
    rows = _evaluation_rows(evaluation)
    months = sorted({period for _, _, period in rows if period != "exposed_evaluation"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.9))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.29, wspace=0.16)
    x = np.arange(len(months))
    for model in MODELS:
        for panel, field in enumerate(("log_loss", "auc")):
            values = np.asarray([[rows[(model, seed, month)][field] for month in months]
                                 for seed in cfg["seeds"]], dtype=float)
            for trace in values:
                axes[panel].plot(x, trace, color=COLORS[model], alpha=0.18, linewidth=0.9)
            axes[panel].plot(x, values.mean(axis=0), color=COLORS[model], linewidth=2.1)
    for panel, field in enumerate(("log_loss", "auc")):
        constant = [rows[("constant", None, month)][field] for month in months]
        axes[panel].plot(x, constant, color=COLORS["constant"], linestyle="--", linewidth=2.0)
        axes[panel].set_xticks(x, months, rotation=45, ha="right")
        axes[panel].set_ylabel("Log loss" if panel == 0 else "ROC-AUC")
        axes[panel].grid(True, color="#e7eaed", linewidth=0.7)
    fig.suptitle("Exposed evaluation by UTC month · exploratory", y=0.96,
                 fontsize=16, fontweight="bold")
    _legend(fig, MODELS + ("constant",), y=0.09)
    fig.text(0.5, 0.022, "Faint: each seed · Thick: seed mean · Same chronological evaluation period for all models",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "evaluation_monthly")


def plot_contrasts(evaluation: dict, output_dir: Path) -> None:
    comparisons = evaluation["paired_weekly_intervals"]
    labels = [row["comparison"] for row in comparisons]
    means = np.asarray([row["difference"] for row in comparisons], dtype=float)
    intervals = np.asarray([row["ci95"] for row in comparisons], dtype=float)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    fig.subplots_adjust(left=0.25, right=0.96, top=0.88, bottom=0.15)
    ax.axvline(0, color="#5d6570", linewidth=1)
    ax.hlines(y, intervals[:, 0], intervals[:, 1], color="#087eaa", linewidth=2)
    ax.plot(means, y, "o", color="#087eaa", markersize=6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired log-loss difference (negative favors first model)")
    ax.grid(axis="x", color="#e7eaed", linewidth=0.7)
    fig.suptitle("Exposed evaluation · weekly block bootstrap intervals", y=0.97,
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.045, "95% intervals resample whole UTC weeks; exploratory association, not a trading claim",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "evaluation_paired_contrasts")


def plot_calibration(cfg: dict, evaluation: dict, output_dir: Path) -> None:
    rows = _evaluation_rows(evaluation)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.18)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9da5ad", linewidth=1.2)
    for model in MODELS:
        totals = defaultdict(lambda: [0.0, 0.0, 0])
        for seed in cfg["seeds"]:
            for point in rows[(model, seed, "exposed_evaluation")]["calibration"]:
                count = int(point["count"])
                cell = totals[int(point["bin"])]
                cell[0] += count * float(point["predicted"])
                cell[1] += count * float(point["observed"])
                cell[2] += count
        supported = [b for b in sorted(totals) if totals[b][2] >= 100]
        sparse = [b for b in sorted(totals) if 0 < totals[b][2] < 100]
        predicted = [totals[b][0] / totals[b][2] for b in supported]
        observed = [totals[b][1] / totals[b][2] for b in supported]
        ax.plot(predicted, observed, marker="o", linewidth=1.8, markersize=4.5,
                color=COLORS[model], label=NAMES[model])
        if sparse:
            sparse_predicted = [totals[b][0] / totals[b][2] for b in sparse]
            sparse_observed = [totals[b][1] / totals[b][2] for b in sparse]
            ax.scatter(sparse_predicted, sparse_observed, s=24, facecolors="none",
                       edgecolors=COLORS[model], linewidths=1.3, zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted up probability",
           ylabel="Observed up fraction")
    ax.grid(True, color="#e7eaed", linewidth=0.7)
    fig.suptitle("Exposed evaluation · pooled seed calibration bins", y=0.96,
                 fontsize=15, fontweight="bold")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.text(0.5, 0.045, "All nonempty bins shown; small hollow markers have pooled n < 100 and are not line-connected",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "evaluation_calibration")


def write_seed_csv(cfg: dict, fits: dict, evaluation: dict, output_dir: Path) -> int:
    eval_rows = _evaluation_rows(evaluation)
    plot_sha = sha256(Path(__file__))
    rows = []
    for model in MODELS:
        for seed in cfg["seeds"]:
            fit, fit_sha = fits[(model, seed)]
            eval_row = eval_rows[(model, seed, "exposed_evaluation")]
            epochs = fit["epoch_curve"]
            row = {
                "model": model, "seed": seed, "gru_hidden": fit["gru_hidden"],
                "parameters": fit["parameters"], "training_seconds": fit["training_seconds"],
                "best_epoch": fit["best_epoch"], "total_optimizer_updates": fit["total_optimizer_updates"],
                "supervised_examples_per_epoch": fit["supervised_examples_per_epoch"],
                "maximum_gradient_norm_before_clip": fit["maximum_gradient_norm_before_clip"],
                "all_gradients_finite": fit["all_gradients_finite"],
                "last_train_log_loss": epochs[-1]["train_log_loss"],
                "best_validation_log_loss": min(point["validation"]["log_loss"] for point in epochs),
                "last_validation_log_loss": epochs[-1]["validation"]["log_loss"],
                "evaluation_log_loss": eval_row["log_loss"], "evaluation_auc": eval_row["auc"],
                "evaluation_brier": eval_row["brier"], "evaluation_ece": eval_row["ece"],
                "code_commit": evaluation["provenance"]["code_commit"],
                "runner_sha256": evaluation["provenance"]["source_sha256"]["scripts/run_market_v3.py"],
                "checkpoint_sha256": fit["checkpoint_sha256"], "fit_report_sha256": fit_sha,
                "evaluation_report_sha256": evaluation["_sha256"],
                "plot_script_sha256": plot_sha,
            }
            rows.append(row)
    with (output_dir / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/market-v3.json")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/v3/plots")
    parser.add_argument("--rolling-updates", type=int, default=50)
    args = parser.parse_args()
    if args.rolling_updates < 1:
        raise ValueError("rolling-updates must be positive")
    cfg, fits, evaluation = load_reports(args.report_dir, args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_batch_learning(cfg, fits, args.output_dir, args.rolling_updates)
    plot_epoch_learning(cfg, fits, args.output_dir)
    plot_months(cfg, evaluation, args.output_dir)
    plot_contrasts(evaluation, args.output_dir)
    plot_calibration(cfg, evaluation, args.output_dir)
    rows = write_seed_csv(cfg, fits, evaluation, args.output_dir)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fit_count": len(fits), "seed_metric_rows": rows,
        "evaluation_status": evaluation["evaluation_status"],
        "provenance": evaluation["provenance"],
        "evaluation_report_sha256": evaluation["_sha256"],
        "plot_script_sha256": sha256(Path(__file__)),
        "rolling_updates": args.rolling_updates,
        "files": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote 5 PNG/SVG figures and {rows} seed metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
