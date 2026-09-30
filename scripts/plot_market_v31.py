"""Plot all completed v3.1 fit, calibration, and exposed-evaluation reports.

No cache, checkpoint, or market labels are read. Means at each epoch/update
include only seeds that actually reached that point; the active-seed panel
shows the changing denominator. Every nonempty reliability bin is displayed,
with sparse bins marked hollow rather than hidden or joined to the curve.
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
CHOICES = ("short_best", "full_best")
MODES = ("raw", "calibrated")
COLORS = {"a": "#404b54", "b": "#7953a6", "c": "#087eaa",
          "no_transfer": "#d2691e", "no_open": "#168255", "constant": "#9299a1"}
NAMES = {"a": "A · candle GRU", "b": "B · routed GRU", "c": "C · tree memory",
         "no_transfer": "C without transfer", "no_open": "C without open readout",
         "constant": "Train-prior constant"}
SEED_COLUMNS = (
    "model", "seed", "choice", "mode", "selected_epoch", "selection_log_loss",
    "stop_epoch", "stop_reason", "best_epoch_all", "parameters", "gru_hidden",
    "total_optimizer_updates", "supervised_examples_per_epoch", "training_seconds",
    "evaluation_n", "evaluation_log_loss", "evaluation_auc", "evaluation_brier", "evaluation_ece",
    "raw_auc", "calibrated_auc", "auc_equal_raw_calibrated",
    "calibration_a", "calibration_b", "calibration_status", "checkpoint_sha256",
    "fit_report_sha256", "calibrators_sha256", "evaluation_report_sha256",
    "plot_script_sha256", "code_commit",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(report_dir: Path, config_path: Path) -> tuple[dict, dict, dict, dict]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    fits: dict[tuple[str, int], dict] = {}
    provenance = None
    for model in cfg["models"]:
        for seed in cfg["seeds"]:
            path = report_dir / f"fit-{model}-{seed}.json"
            if not path.exists():
                raise RuntimeError(f"Missing fit report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if (not report.get("complete") or report.get("model") != model
                    or report.get("seed") != seed or report.get("config") != cfg):
                raise RuntimeError(f"Incomplete or mismatched fit report: {path}")
            if provenance is None:
                provenance = report["provenance"]
            elif report["provenance"] != provenance:
                raise RuntimeError("Fit provenance differs across model/seed runs")
            if len(report["epoch_curve"]) != report["epochs_completed"]:
                raise RuntimeError(f"Epoch curve length differs from stop epoch: {path}")
            if len(report["batch_curve"]) != report["total_optimizer_updates"]:
                raise RuntimeError(f"Batch curve length differs from update count: {path}")
            if report["stop_reason"] not in ("patience", "budget_cap"):
                raise RuntimeError(f"Fit has no completed stop reason: {path}")
            if report["stop_reason"] == "budget_cap" and report["epochs_completed"] != cfg["max_epochs"]:
                raise RuntimeError(f"Budget cap reported before max epoch: {path}")
            for choice in CHOICES:
                if report["checkpoints"].get(choice) is None:
                    raise RuntimeError(f"Missing selected {choice}: {path}")
            report["_sha256"] = sha256(path)
            fits[(model, seed)] = report
    if len(fits) != len(MODELS) * len(cfg["seeds"]):
        raise RuntimeError("Expected fifteen complete fits")

    calibrators_path = report_dir / "calibrators.json"
    freeze_path = report_dir / "calibration-freeze.json"
    evaluation_path = report_dir / "evaluation.json"
    if not (calibrators_path.exists() and freeze_path.exists() and evaluation_path.exists()):
        raise RuntimeError("Calibration and exposed evaluation must be complete before plotting")
    calibrators = json.loads(calibrators_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (not calibrators.get("complete") or not freeze.get("complete")
            or freeze.get("calibrators_sha256") != sha256(calibrators_path)
            or evaluation.get("calibration_freeze_sha256") != sha256(freeze_path)):
        raise RuntimeError("Calibration freeze chain is incomplete or mismatched")
    if any(item.get("provenance") != provenance or item.get("config") != cfg
           for item in (calibrators, evaluation)) or freeze.get("provenance") != provenance:
        raise RuntimeError("Fit/calibration/evaluation provenance or config differs")
    expected = {(model, seed, choice) for model in MODELS for seed in cfg["seeds"] for choice in CHOICES}
    keys = [(row["model"], row["seed"], row["choice"]) for row in calibrators["entries"]]
    if len(keys) != 30 or set(keys) != expected:
        raise RuntimeError("Expected thirty distinct calibrators")
    overall = [(row["model"], row["seed"], row["choice"], row["mode"])
               for row in evaluation["metrics"] if row["period"] == "exposed_evaluation"
               and row["model"] != "constant"]
    expected60 = {(model, seed, choice, mode) for model in MODELS for seed in cfg["seeds"]
                  for choice in CHOICES for mode in MODES}
    if len(overall) != 60 or set(overall) != expected60:
        raise RuntimeError("Expected sixty distinct exposed evaluation identities")
    calibrators["_sha256"] = sha256(calibrators_path)
    evaluation["_sha256"] = sha256(evaluation_path)
    return cfg, fits, calibrators, evaluation


def _legend(fig, *, y: float = 0.06, include_constant: bool = False) -> None:
    models = MODELS + (("constant",) if include_constant else ())
    handles = [Line2D([0], [0], color=COLORS[model], linewidth=2.5, label=NAMES[model])
               for model in models]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, y),
               frameon=False, fontsize=9)


def _save(fig, output_dir: Path, stem: str) -> None:
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _available_mean(traces: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = max(len(x) for x, _ in traces)
    values = np.full((len(traces), maximum), np.nan, dtype=float)
    for row, (_, y) in enumerate(traces):
        values[row, :len(y)] = y
    counts = np.isfinite(values).sum(axis=0)
    means = np.nansum(values, axis=0) / counts
    return np.arange(1, maximum + 1), means, counts


def _rolling_loss(curve: list[dict], window: int) -> tuple[np.ndarray, np.ndarray]:
    updates = np.asarray([row["update"] for row in curve], dtype=int)
    if not np.array_equal(updates, np.arange(1, len(curve) + 1)):
        raise RuntimeError("Batch update sequence is incomplete")
    weights = np.asarray([row["n"] for row in curve], dtype=float)
    losses = np.asarray([row["loss"] for row in curve], dtype=float)
    if np.any(weights <= 0) or not np.isfinite(losses).all():
        raise RuntimeError("Invalid batch loss/count")
    numerator = np.r_[0.0, np.cumsum(weights * losses)]
    denominator = np.r_[0.0, np.cumsum(weights)]
    ends = np.arange(1, len(curve) + 1)
    starts = np.maximum(0, ends - window)
    return updates, (numerator[ends] - numerator[starts]) / (denominator[ends] - denominator[starts])


def _active_count_panel(axis, counts_by_model: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for model in MODELS:
        x, counts = counts_by_model[model]
        axis.step(x, counts, color=COLORS[model], linewidth=1.7, where="post")
    axis.set_ylim(0.8, 3.2)
    axis.set_yticks([1, 2, 3])
    axis.set_ylabel("Active seeds")
    axis.grid(True, color="#e7eaed", linewidth=0.7)


def plot_training(cfg: dict, fits: dict, output_dir: Path, rolling_updates: int) -> None:
    fig = plt.figure(figsize=(13.5, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(4.2, 1), left=0.08, right=0.985,
                           top=0.88, bottom=0.19, hspace=0.12, wspace=0.09)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    count_axes = [fig.add_subplot(grid[1, 0], sharex=axes[0]),
                  fig.add_subplot(grid[1, 1], sharex=axes[1])]
    counts_by_panel = [{}, {}]
    for model in MODELS:
        runs = [fits[(model, seed)] for seed in cfg["seeds"]]
        for panel, extract in ((0, lambda run: (np.arange(1, len(run["epoch_curve"]) + 1),
                                                      np.asarray([row["train_log_loss"] for row in run["epoch_curve"]]))),
                               (1, lambda run: (np.arange(1, len(run["epoch_curve"]) + 1),
                                                      np.asarray([row["selection"]["log_loss"] for row in run["epoch_curve"]])))):
            traces = [extract(run) for run in runs]
            for x, y in traces:
                axes[panel].plot(x, y, color=COLORS[model], alpha=0.19, linewidth=0.9)
            x, mean, counts = _available_mean(traces)
            axes[panel].plot(x, mean, color=COLORS[model], linewidth=2.2)
            counts_by_panel[panel][model] = (x, counts)
        for run in runs:
            for choice, marker, size in (("short_best", "s", 55), ("full_best", "*", 90)):
                selected = run["checkpoints"][choice]
                epoch = selected["epoch"]
                loss = run["epoch_curve"][epoch - 1]["selection"]["log_loss"]
                axes[1].scatter(epoch, loss, marker=marker, s=size,
                                facecolors="none" if choice == "short_best" else COLORS[model],
                                edgecolors=COLORS[model], linewidths=1.2, zorder=5)
    for panel, title in enumerate(("Training", "Selection · selected epochs marked")):
        axes[panel].set_title(title)
        axes[panel].set_ylabel("Log loss" if panel == 0 else "Selection log loss")
        axes[panel].grid(True, color="#e7eaed", linewidth=0.7)
        axes[panel].set_xlim(1, cfg["max_epochs"])
        count_axes[panel].set_xlabel("Epoch")
        _active_count_panel(count_axes[panel], counts_by_panel[panel])
    fig.suptitle("Market v3.1 · all observed training and selection epochs", y=0.97,
                 fontsize=16, fontweight="bold")
    _legend(fig, y=0.065)
    fig.text(0.5, 0.027,
             "Faint: individual seeds · Thick: mean over active seeds only (count below) · □ short best · ★ full best · cap at 20 is a budget stop",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "epoch_learning_selection")

    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.3, 1), left=0.08, right=0.985,
                           top=0.86, bottom=0.22, hspace=0.10)
    ax = fig.add_subplot(grid[0])
    count_ax = fig.add_subplot(grid[1], sharex=ax)
    active = {}
    for model in MODELS:
        traces = [_rolling_loss(fits[(model, seed)]["batch_curve"], rolling_updates)
                  for seed in cfg["seeds"]]
        for x, y in traces:
            ax.plot(x, y, color=COLORS[model], alpha=0.18, linewidth=0.75)
        x, mean, counts = _available_mean(traces)
        ax.plot(x, mean, color=COLORS[model], linewidth=2.0)
        active[model] = (x, counts)
    ax.set(xlabel="", ylabel="Training binary cross-entropy")
    ax.grid(True, color="#e7eaed", linewidth=0.7)
    ax.set_xlim(left=1)
    _active_count_panel(count_ax, active)
    count_ax.set_xlabel("Optimizer update")
    fig.suptitle(f"Market v3.1 · {rolling_updates}-update weighted rolling training loss", y=0.96,
                 fontsize=16, fontweight="bold")
    _legend(fig, y=0.075)
    fig.text(0.5, 0.025, "Faint: individual seeds · Thick: available-seed mean · No values imputed after early stop",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "batch_training_loss")


def _overall_rows(evaluation: dict) -> dict[tuple, dict]:
    rows = {}
    for row in evaluation["metrics"]:
        key = (row["model"], row["seed"], row["choice"], row["mode"], row["period"])
        if key in rows:
            raise RuntimeError(f"Duplicate evaluation row: {key}")
        rows[key] = row
    return rows


def plot_selected_comparison(cfg: dict, evaluation: dict, output_dir: Path) -> None:
    rows = _overall_rows(evaluation)
    constant = rows[("constant", None, "train_frequency", "raw", "exposed_evaluation")]["log_loss"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.13,
                        hspace=0.13, wspace=0.07)
    x = np.arange(len(MODELS))
    for r, choice in enumerate(CHOICES):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            ax.axhline(constant, color=COLORS["constant"], linestyle="--", linewidth=1.4)
            for position, model in enumerate(MODELS):
                values = np.asarray([rows[(model, seed, choice, mode, "exposed_evaluation")]["log_loss"]
                                     for seed in cfg["seeds"]], dtype=float)
                ax.scatter(position + np.array([-0.09, 0.0, 0.09]), values, s=17,
                           color=COLORS[model], alpha=0.35, zorder=3)
                ax.scatter(position, values.mean(), s=65, marker="D", color=COLORS[model], zorder=4)
            ax.set_title(f"{'Short' if choice == 'short_best' else 'Full'} selection · {mode}")
            ax.grid(axis="y", color="#e7eaed", linewidth=0.7)
            ax.set_xticks(x, ["A", "B", "C", "No transfer", "No open"])
            ax.tick_params(axis="x", rotation=20)
            if c == 0:
                ax.set_ylabel("Exposed evaluation log loss")
    primary = next((row for row in evaluation["comparisons"] if row.get("primary")), None)
    if primary:
        fig.text(0.5, 0.065,
                 f"Primary C−A (full/calibrated): {primary['difference']:+.6f}, weekly 95% CI "
                 f"[{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}] · negative favors C",
                 ha="center", fontsize=9, color="#33383d")
    fig.suptitle("All 60 selected raw/calibrated identities · exposed evaluation", y=0.965,
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.025, "Small circles: each seed · Diamonds: three-seed mean · Dashed: train-prior constant",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "selected_logloss_all_identities")


def plot_calibration_coefficients(cfg: dict, calibrators: dict, output_dir: Path) -> None:
    entries = {(row["model"], row["seed"], row["choice"]): row
               for row in calibrators["entries"]}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.23, wspace=0.15)
    for panel, field in enumerate(("a", "b")):
        ax = axes[panel]
        for index, model in enumerate(MODELS):
            for choice, shift, marker in (("short_best", -0.16, "s"), ("full_best", 0.16, "o")):
                values = np.asarray([entries[(model, seed, choice)]["parameters"][field]
                                     for seed in cfg["seeds"]], dtype=float)
                ax.scatter(np.full(len(values), index + shift) + np.array([-0.035, 0.0, 0.035]),
                           values, s=20, color=COLORS[model], alpha=0.32)
                ax.scatter(index + shift, values.mean(), s=66, marker=marker,
                           facecolors="none" if choice == "short_best" else COLORS[model],
                           edgecolors=COLORS[model], linewidths=1.5)
        ax.set_xticks(np.arange(len(MODELS)), ["A", "B", "C", "No transfer", "No open"])
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("Positive slope a" if field == "a" else "Intercept b")
        ax.grid(axis="y", color="#e7eaed", linewidth=0.7)
    fig.suptitle("Thirty frozen positive-affine calibration fits", y=0.96,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.075, "Squares: short best · Circles: full best · Faint: individual seeds · Large: mean",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "calibration_coefficients")


def plot_reliability(cfg: dict, evaluation: dict, output_dir: Path) -> None:
    rows = _overall_rows(evaluation)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.16,
                        hspace=0.16, wspace=0.10)
    for r, choice in enumerate(CHOICES):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], linestyle="--", color="#a5adb4", linewidth=1)
            for model in MODELS:
                totals = defaultdict(lambda: [0.0, 0.0, 0])
                for seed in cfg["seeds"]:
                    points = rows[(model, seed, choice, mode, "exposed_evaluation")]["calibration"]
                    for point in points:
                        count = int(point["count"])
                        cell = totals[int(point["bin"])]
                        cell[0] += count * float(point["predicted"])
                        cell[1] += count * float(point["observed"])
                        cell[2] += count
                supported = [b for b in sorted(totals) if totals[b][2] >= 100]
                sparse = [b for b in sorted(totals) if 0 < totals[b][2] < 100]
                ax.plot([totals[b][0] / totals[b][2] for b in supported],
                        [totals[b][1] / totals[b][2] for b in supported],
                        marker="o", markersize=3.3, linewidth=1.4, color=COLORS[model])
                if sparse:
                    ax.scatter([totals[b][0] / totals[b][2] for b in sparse],
                               [totals[b][1] / totals[b][2] for b in sparse],
                               s=22, facecolors="none", edgecolors=COLORS[model],
                               linewidths=1.2, zorder=4)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.set_title(f"{'Short' if choice == 'short_best' else 'Full'} · {mode}")
            ax.grid(True, color="#e7eaed", linewidth=0.7)
            if c == 0:
                ax.set_ylabel("Observed up fraction")
            if r == 1:
                ax.set_xlabel("Mean predicted up probability")
    fig.suptitle("Exposed evaluation reliability · every nonempty bin", y=0.97,
                 fontsize=16, fontweight="bold")
    _legend(fig, y=0.065)
    fig.text(0.5, 0.025, "Seed-pooled ten-bin tables · Hollow markers: pooled n < 100, shown but not line-connected",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "reliability_all_bins")


def plot_months(cfg: dict, evaluation: dict, output_dir: Path) -> None:
    rows = _overall_rows(evaluation)
    months = sorted({row["period"] for row in evaluation["metrics"]
                     if row["period"] != "exposed_evaluation"})
    x = np.arange(len(months))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.89, bottom=0.21,
                        hspace=0.13, wspace=0.07)
    for r, choice in enumerate(CHOICES):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            constant = [rows[("constant", None, "train_frequency", "raw", month)]["log_loss"]
                        for month in months]
            ax.plot(x, constant, color=COLORS["constant"], linestyle="--", linewidth=1.5)
            for model in MODELS:
                traces = np.asarray([[rows[(model, seed, choice, mode, month)]["log_loss"]
                                      for month in months] for seed in cfg["seeds"]], dtype=float)
                for trace in traces:
                    ax.plot(x, trace, color=COLORS[model], alpha=0.17, linewidth=0.8)
                ax.plot(x, traces.mean(axis=0), color=COLORS[model], linewidth=1.8)
            ax.set_title(f"{'Short' if choice == 'short_best' else 'Full'} · {mode}")
            ax.grid(True, color="#e7eaed", linewidth=0.7)
            ax.set_xticks(x, months, rotation=45, ha="right")
            if c == 0:
                ax.set_ylabel("Monthly exposed log loss")
    fig.suptitle("Exposed evaluation by UTC month · all 60 identities", y=0.965,
                 fontsize=15, fontweight="bold")
    _legend(fig, y=0.07, include_constant=True)
    fig.text(0.5, 0.02, "Faint: each seed · Thick: three-seed mean · Dashed: train-prior constant · Exploratory",
             ha="center", fontsize=9, color="#555b63")
    _save(fig, output_dir, "monthly_all_identities")


def write_csv(cfg: dict, fits: dict, calibrators: dict, evaluation: dict, output_dir: Path) -> int:
    cal = {(row["model"], row["seed"], row["choice"]): row
           for row in calibrators["entries"]}
    scores = _overall_rows(evaluation)
    script_sha = sha256(Path(__file__))
    rows = []
    for model in MODELS:
        for seed in cfg["seeds"]:
            fit = fits[(model, seed)]
            for choice in CHOICES:
                checkpoint = fit["checkpoints"][choice]
                parameters = cal[(model, seed, choice)]["parameters"]
                raw = scores[(model, seed, choice, "raw", "exposed_evaluation")]
                calibrated = scores[(model, seed, choice, "calibrated", "exposed_evaluation")]
                auc_equal = raw["auc"] == calibrated["auc"]
                if not auc_equal:
                    raise RuntimeError(f"Positive-affine calibration changed ranking AUC: {model}/{seed}/{choice}")
                for mode, score in (("raw", raw), ("calibrated", calibrated)):
                    rows.append({
                        "model": model, "seed": seed, "choice": choice, "mode": mode,
                        "selected_epoch": checkpoint["epoch"],
                        "selection_log_loss": checkpoint["selection_log_loss"],
                        "stop_epoch": fit["epochs_completed"], "stop_reason": fit["stop_reason"],
                        "best_epoch_all": fit["control_state"]["best_epoch"],
                        "parameters": fit["parameters"], "gru_hidden": fit["gru_hidden"],
                        "total_optimizer_updates": fit["total_optimizer_updates"],
                        "supervised_examples_per_epoch": fit["supervised_examples_per_epoch"],
                        "training_seconds": fit["training_seconds"],
                        "evaluation_n": score["n"], "evaluation_log_loss": score["log_loss"],
                        "evaluation_auc": score["auc"], "evaluation_brier": score["brier"],
                        "evaluation_ece": score["ece"], "raw_auc": raw["auc"],
                        "calibrated_auc": calibrated["auc"], "auc_equal_raw_calibrated": auc_equal,
                        "calibration_a": parameters["a"], "calibration_b": parameters["b"],
                        "calibration_status": parameters["status"],
                        "checkpoint_sha256": checkpoint["sha256"],
                        "fit_report_sha256": fit["_sha256"],
                        "calibrators_sha256": calibrators["_sha256"],
                        "evaluation_report_sha256": evaluation["_sha256"],
                        "plot_script_sha256": script_sha,
                        "code_commit": evaluation["provenance"]["code_commit"],
                    })
    with (output_dir / "identity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/market-v31.json")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/v31")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/v31/plots")
    parser.add_argument("--rolling-updates", type=int, default=50)
    args = parser.parse_args()
    if args.rolling_updates < 1:
        raise ValueError("rolling-updates must be positive")
    cfg, fits, calibrators, evaluation = load_reports(args.report_dir, args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training(cfg, fits, args.output_dir, args.rolling_updates)
    plot_selected_comparison(cfg, evaluation, args.output_dir)
    plot_calibration_coefficients(cfg, calibrators, args.output_dir)
    plot_reliability(cfg, evaluation, args.output_dir)
    plot_months(cfg, evaluation, args.output_dir)
    rows = write_csv(cfg, fits, calibrators, evaluation, args.output_dir)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fit_count": len(fits), "selected_checkpoint_count": 30,
        "identity_metrics_rows": rows, "evaluation_status": evaluation["evaluation_status"],
        "provenance": evaluation["provenance"],
        "calibrators_sha256": calibrators["_sha256"],
        "evaluation_report_sha256": evaluation["_sha256"],
        "plot_script_sha256": sha256(Path(__file__)),
        "rolling_updates": args.rolling_updates,
        "figures": sorted(path.name for path in args.output_dir.iterdir()
                          if path.suffix in (".png", ".svg")),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote 6 PNG/SVG figures and {rows} identity rows to {args.output_dir}")


if __name__ == "__main__":
    main()
