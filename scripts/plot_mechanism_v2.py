"""Plot frozen mechanism-v2 reports and export per-seed/aggregate CSVs.

By default this refuses missing or incomplete task reports. Pass
``--allow-incomplete`` only for visibly marked progress plots; final artifacts
must be regenerated after all six reports have ``complete: true``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_ORDER = ("tiny_hard", "easy_closed", "medium_closed", "hard_closed", "hard_open", "random_hard")
VARIANT_ORDER = ("v1", "tree", "no_transfer", "no_open", "gru")
COLORS = {
    "v1": "#7651a8",
    "tree": "#087eaa",
    "no_transfer": "#d2691e",
    "no_open": "#168255",
    "gru": "#3f4850",
}
LABELS = {
    "v1": "V1 Crossing Memory",
    "tree": "V2 tree",
    "no_transfer": "V2 without transfer",
    "no_open": "V2 without open readout",
    "gru": "GRU",
}
TASK_LABELS = {
    "tiny_hard": "Tiny hard · memorization, same examples",
    "easy_closed": "Easy closed",
    "medium_closed": "Medium closed",
    "hard_closed": "Hard closed",
    "hard_open": "Hard open",
    "random_hard": "Independent random labels",
}
METRIC_FIELDS = ("loss", "accuracy", "balanced_accuracy")
SEED_COLUMNS = [
    "task", "variant", "seed", "test_kind", "source_commit", "runner_sha256", "plot_script_sha256",
    "report_sha256", "parameters", "training_seconds", "best_epoch", "epochs", "optimizer_updates",
    "finite_gradients", "train_n", "validation_n", "test_n",
    *[f"final_train_{field}" for field in METRIC_FIELDS],
    *[f"selected_test_{field}" for field in METRIC_FIELDS],
    *[f"final_test_{field}" for field in METRIC_FIELDS],
]
AGG_COLUMNS = [
    "task", "variant", "test_kind", "seed_count", "source_commit", "runner_sha256", "plot_script_sha256",
    "parameters", "train_n", "validation_n", "test_n",
    *[f"mean_{name}" for name in ("training_seconds", "best_epoch", "final_train_loss",
                                      "final_train_accuracy", "final_train_balanced_accuracy",
                                      "selected_test_loss", "selected_test_accuracy",
                                      "selected_test_balanced_accuracy", "final_test_loss",
                                      "final_test_accuracy", "final_test_balanced_accuracy")],
    *[f"sd_{name}" for name in ("training_seconds", "best_epoch", "final_train_loss",
                                    "final_train_accuracy", "final_train_balanced_accuracy",
                                    "selected_test_loss", "selected_test_accuracy",
                                    "selected_test_balanced_accuracy", "final_test_loss",
                                    "final_test_accuracy", "final_test_balanced_accuracy")],
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(report_dir: Path, allow_incomplete: bool) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    missing = []
    incomplete = []
    for task in TASK_ORDER:
        path = report_dir / f"{task}.json"
        if not path.exists():
            missing.append(task)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("task") != task:
            raise ValueError(f"Task mismatch in {path}")
        if not report.get("complete", False):
            incomplete.append(task)
        report["_sha256"] = sha256(path)
        reports[task] = report
    if not allow_incomplete and (missing or incomplete):
        raise RuntimeError(f"Final plots require all complete reports; missing={missing}, incomplete={incomplete}")
    if not reports:
        raise RuntimeError("No task reports found")
    provenance = [r["provenance"] for r in reports.values()]
    first = provenance[0]
    for item in provenance[1:]:
        if item["commit"] != first["commit"] or item["files_sha256"] != first["files_sha256"]:
            raise ValueError("Task reports have inconsistent frozen source provenance")
    for task, report in reports.items():
        expected_seeds = set(report["config"]["seeds"])
        expected_variants = set(report["config"]["tasks"][task])
        keys = [(run["seed"], run["variant"]) for run in report["runs"]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate seed/variant in {task}")
        if any(seed not in expected_seeds or variant not in expected_variants for seed, variant in keys):
            raise ValueError(f"Unexpected seed/variant in {task}")
        expected = {(seed, variant) for seed in expected_seeds for variant in expected_variants}
        if report.get("complete") and set(keys) != expected:
            raise ValueError(f"Completed report {task} lacks expected runs")
    return reports


def flatten_seed_rows(reports: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    plot_sha = sha256(Path(__file__))
    for task in TASK_ORDER:
        if task not in reports:
            continue
        report = reports[task]
        dataset = {int(item["seed"]): item["splits"] for item in report["dataset"]}
        for run in report["runs"]:
            seed = int(run["seed"])
            sizes = dataset[seed]
            row = {
                "task": task, "variant": run["variant"], "seed": seed,
                "test_kind": "memorization_same_examples" if task == "tiny_hard" else "independent_held_out",
                "source_commit": report["provenance"]["commit"],
                "runner_sha256": report["provenance"]["files_sha256"]["scripts/run_mechanism_v2.py"],
                "plot_script_sha256": plot_sha, "report_sha256": report["_sha256"],
                "parameters": run["parameters"], "training_seconds": run["training_seconds"],
                "best_epoch": run["best_epoch"], "epochs": run["epochs"],
                "optimizer_updates": run["optimizer_updates"], "finite_gradients": run["finite_gradients"],
                "train_n": sizes["train"]["n"], "validation_n": sizes["validation"]["n"],
                "test_n": sizes["test"]["n"],
            }
            for metric_set in ("final_train", "selected_test", "final_test"):
                for field in METRIC_FIELDS:
                    row[f"{metric_set}_{field}"] = run[metric_set][field]
            rows.append(row)
    rows.sort(key=lambda row: (TASK_ORDER.index(row["task"]), VARIANT_ORDER.index(row["variant"]), row["seed"]))
    return rows


def aggregate_rows(seed_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["task"], row["variant"])].append(row)
    aggregate: list[dict] = []
    numeric = ("training_seconds", "best_epoch", "final_train_loss", "final_train_accuracy",
               "final_train_balanced_accuracy", "selected_test_loss", "selected_test_accuracy",
               "selected_test_balanced_accuracy", "final_test_loss", "final_test_accuracy",
               "final_test_balanced_accuracy")
    for task in TASK_ORDER:
        for variant in VARIANT_ORDER:
            items = grouped.get((task, variant))
            if not items:
                continue
            first = items[0]
            row = {key: first[key] for key in ("task", "variant", "test_kind", "source_commit",
                                                "runner_sha256", "plot_script_sha256", "parameters",
                                                "train_n", "validation_n", "test_n")}
            row["seed_count"] = len(items)
            for name in numeric:
                values = np.asarray([float(item[name]) for item in items], dtype=np.float64)
                row[f"mean_{name}"] = float(values.mean())
                row[f"sd_{name}"] = float(values.std(ddof=1)) if len(values) > 1 else ""
            aggregate.append(row)
    return aggregate


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _metric_limits(reports: dict[str, dict]) -> tuple[float, tuple[float, float]]:
    losses = [point[split]["loss"]
              for report in reports.values() for run in report["runs"]
              for point in run["curve"] for split in ("train", "validation")]
    # Shared loss and balanced-accuracy scales across all six task figures.
    loss_top = max(0.8, math.ceil(max(losses, default=0.8) * 10) / 10 + 0.1)
    return loss_top, (0.0, 1.0)


def plot_task(task: str, report: dict, output_dir: Path, *, incomplete: bool,
              loss_top: float, ba_limits: tuple[float, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.89, bottom=0.18,
                        hspace=0.11, wspace=0.06)
    variants = [variant for variant in VARIANT_ORDER if variant in report["config"]["tasks"][task]]
    for variant in variants:
        runs = sorted((run for run in report["runs"] if run["variant"] == variant), key=lambda r: r["seed"])
        if not runs:
            continue
        color = COLORS[variant]
        for col, split in enumerate(("train", "validation")):
            for row_idx, field in enumerate(("loss", "balanced_accuracy")):
                axis = axes[row_idx, col]
                lengths = {len(run["curve"]) for run in runs}
                if len(lengths) != 1:
                    raise ValueError(f"Unequal epoch counts for {task}/{variant}")
                epochs = np.asarray([item["epoch"] for item in runs[0]["curve"]], dtype=int)
                traces = np.asarray([[point[split][field] for point in run["curve"]] for run in runs])
                for trace in traces:
                    axis.plot(epochs, trace, color=color, alpha=0.22, linewidth=0.85)
                axis.plot(epochs, traces.mean(axis=0), color=color, linewidth=2.2)
    axes[0, 0].set_title("Training")
    axes[0, 1].set_title("Validation" if task != "tiny_hard" else "Same training examples (duplicated)")
    for axis in axes[0]:
        axis.set_ylim(0, loss_top)
    for axis in axes[1]:
        axis.set_ylim(*ba_limits)
    for axis in axes.flat:
        axis.grid(True, color="#e6e9ed", linewidth=0.7)
        axis.set_xlim(left=1)
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    note = "Faint lines: individual seeds · Thick lines: seed mean"
    if task == "tiny_hard":
        note += " · Tiny validation/test repeat training examples; no held-out estimate"
    if incomplete:
        note += " · INCOMPLETE REPORT"
        fig.text(0.5, 0.5, "INCOMPLETE", ha="center", va="center", rotation=30,
                 fontsize=48, color="#a92727", alpha=0.13)
    fig.suptitle(TASK_LABELS[task], y=0.965, fontsize=16, fontweight="bold")
    handles = [Line2D([0], [0], color=COLORS[variant], linewidth=2.5, label=LABELS[variant])
               for variant in variants if any(run["variant"] == variant for run in report["runs"])]
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 0.064),
               frameon=False)
    fig.text(0.5, 0.025, note, ha="center", fontsize=9, color="#555b63")
    stem = f"{task}{'-INCOMPLETE' if incomplete else ''}"
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "v2" / "plots")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    reports = load_reports(args.report_dir, args.allow_incomplete)
    all_complete = len(reports) == len(TASK_ORDER) and all(r.get("complete") for r in reports.values())
    suffix = "" if all_complete else "-INCOMPLETE"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = flatten_seed_rows(reports)
    agg_rows = aggregate_rows(seed_rows)
    write_csv(args.output_dir / f"seed_metrics{suffix}.csv", SEED_COLUMNS, seed_rows)
    write_csv(args.output_dir / f"aggregate_summary{suffix}.csv", AGG_COLUMNS, agg_rows)
    loss_top, ba_limits = _metric_limits(reports)
    for task in TASK_ORDER:
        if task in reports:
            plot_task(task, reports[task], args.output_dir, incomplete=not all_complete,
                      loss_top=loss_top, ba_limits=ba_limits)
    print(f"Wrote {len(reports)} task figures (PNG and SVG), {len(seed_rows)} seed rows, "
          f"{len(agg_rows)} aggregate rows to {args.output_dir}; complete={all_complete}")


if __name__ == "__main__":
    main()
