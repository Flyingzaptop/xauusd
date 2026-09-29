"""Freeze exact market-v3 stream/update volume before any model fitting.

Reads prepared split arrays and preflight throughput only. The cached target
arrays are deliberately not loaded: placeholder zeros satisfy PackedStream's
shape contract while `valid` alone determines supervised counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stream_batch_v3 import PackedStream, TREE_KEYS  # noqa: E402
from src.stream_models_v3 import StreamForecaster, matched_baseline_widths, parameter_count  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_without_targets(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as file:
        x = file["x"].copy()
        valid = file["valid"].copy()
        reset = file["reset"].copy()
        timestamp = file["timestamp"].copy()
        tree = {key: file[f"tree_{key}"].copy() for key in TREE_KEYS}
    return {"x": x, "tree": tree, "y": np.zeros(len(valid), dtype=np.float32),
            "valid": valid, "reset": reset, "timestamp": timestamp}


def audit_split(path: Path, batch_size: int, chunk_length: int) -> dict:
    split = _load_without_targets(path)
    packer = PackedStream(split, batch_size, chunk_length)
    seen = np.zeros(packer.n_observations, dtype=np.bool_)
    encountered = valid_encountered = padding_slots = valid_chunks = 0
    for batch in packer:
        ids = batch["indices"]
        real = ids >= 0
        current = ids[real]
        if len(np.unique(current)) != len(current) or seen[current].any():
            raise AssertionError(f"Duplicate observation in {path.name}")
        seen[current] = True
        encountered += len(current)
        valid_encountered += int(batch["valid"].sum())
        padding_slots += int((~real).sum())
        valid_chunks += int(batch["valid"].any())
        if batch["valid"][~real].any() or not batch["reset"][~real].all():
            raise AssertionError(f"Unsafe padding in {path.name}")
    if encountered != packer.n_observations or not seen.all():
        raise AssertionError(f"Missing observation in {path.name}")
    if valid_encountered != packer.n_valid_labels:
        raise AssertionError(f"Supervised label count differs in {path.name}")
    if padding_slots + encountered != len(packer) * batch_size * chunk_length:
        raise AssertionError(f"Packed slots differ in {path.name}")
    lengths = [len(lane) for lane in packer.lane_indices]
    return {
        "rows": packer.n_observations,
        "segments": len(packer.segments),
        "supervised_rows": packer.n_valid_labels,
        "unsupervised_input_rows": packer.n_observations - packer.n_valid_labels,
        "total_chunks": len(packer),
        "chunks_with_valid_labels": valid_chunks,
        "chunks_without_valid_labels": len(packer) - valid_chunks,
        "optimizer_updates_per_epoch": valid_chunks,
        "padded_slots": padding_slots,
        "lane_observation_counts": lengths,
        "exactly_once_coverage_verified": True,
        "valid_labels_exactly_once_verified": True,
        "cache_sha256": sha256(path),
    }


def freeze(config_path: Path, speed_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    speed = json.loads(speed_path.read_text(encoding="utf-8"))
    cache = Path(cfg["output"])
    audit = json.loads((cache / "audit.json").read_text(encoding="utf-8"))
    config_sha = sha256(config_path)
    if audit["provenance"]["config_sha256"] != config_sha or speed["config_sha256"] != config_sha:
        raise RuntimeError("Prepared cache and speed preflight must match current config")
    if speed["kind"] != "throwaway_synthetic_preflight_not_market_training":
        raise ValueError("Unexpected speed preflight kind")

    widths = matched_baseline_widths(cfg["c_gru_hidden"], cfg["memory_hidden"], 4)
    if (widths["a"], widths["b"], widths["c"]) != (51, 35, 16):
        raise RuntimeError("Matched model widths differ from the preregistered A51/B35/C16")
    model_parameters = {
        name: parameter_count(StreamForecaster(name, widths.get(name, cfg["c_gru_hidden"]),
                                               cfg["memory_hidden"], 4))
        for name in cfg["models"]
    }
    for name, count in model_parameters.items():
        if speed["benchmark"][name]["parameters"] != count:
            raise RuntimeError(f"Model {name} parameter count differs from speed preflight")
        if (speed["benchmark"][name]["batch_size"] != cfg["batch_size"] or
                speed["benchmark"][name]["chunk_length"] != cfg["chunk_length"]):
            raise RuntimeError(f"Model {name} speed preflight used different B/T")

    splits = {name: audit_split(cache / f"{name}.npz", cfg["batch_size"], cfg["chunk_length"])
              for name in ("train", "validation", "evaluation")}
    for name, row in splits.items():
        source = audit["splits"][name]
        if (row["rows"] != source["rows"] or row["segments"] != source["segments"] or
                row["supervised_rows"] != source["supervised_rows"]):
            raise AssertionError(f"Packed {name} counts differ from data preflight")

    epochs = int(cfg["epochs"])
    seeds = tuple(int(seed) for seed in cfg["seeds"])
    models = tuple(cfg["models"])
    fits = len(seeds) * len(models)
    train = splits["train"]
    volume = {
        "epochs_per_fit": epochs,
        "seeds": seeds,
        "models": models,
        "fits": fits,
        "optimizer_updates_per_fit": train["optimizer_updates_per_epoch"] * epochs,
        "total_optimizer_updates": train["optimizer_updates_per_epoch"] * epochs * fits,
        "training_chunks_processed_per_fit": train["total_chunks"] * epochs,
        "total_training_chunks_processed": train["total_chunks"] * epochs * fits,
        "raw_training_observation_visits": train["rows"] * epochs * fits,
        "supervised_label_visits": train["supervised_rows"] * epochs * fits,
        "validation_forward_chunks_per_fit": splits["validation"]["total_chunks"] * epochs,
        "total_validation_forward_chunks": splits["validation"]["total_chunks"] * epochs * fits,
        "evaluation_forward_chunks_per_fit": splits["evaluation"]["total_chunks"],
        "total_evaluation_forward_chunks": splits["evaluation"]["total_chunks"] * fits,
    }

    runtime_by_model = {}
    for name in models:
        step_seconds = float(speed["benchmark"][name]["median_seconds_per_chunk"])
        forward_seconds = float(speed["benchmark"][name]["eval_forward_only_median_seconds_per_chunk"])
        train_seconds = train["total_chunks"] * epochs * len(seeds) * step_seconds
        validation_proxy = splits["validation"]["total_chunks"] * epochs * len(seeds) * forward_seconds
        evaluation_proxy = splits["evaluation"]["total_chunks"] * len(seeds) * forward_seconds
        runtime_by_model[name] = {
            "median_measured_training_seconds_per_chunk": step_seconds,
            "median_measured_forward_seconds_per_chunk": forward_seconds,
            "estimated_training_seconds": train_seconds,
            "validation_forward_seconds_at_training_chunk_rate": validation_proxy,
            "evaluation_forward_seconds_at_training_chunk_rate": evaluation_proxy,
            "estimated_total_seconds_at_training_chunk_rate": train_seconds + validation_proxy + evaluation_proxy,
        }
    estimated_train = sum(item["estimated_training_seconds"] for item in runtime_by_model.values())
    estimated_validation = sum(item["validation_forward_seconds_at_training_chunk_rate"]
                               for item in runtime_by_model.values())
    estimated_evaluation = sum(item["evaluation_forward_seconds_at_training_chunk_rate"]
                               for item in runtime_by_model.values())
    raw_total = estimated_train + estimated_validation + estimated_evaluation
    runtime = {
        "method": "measured synthetic training and separate forward-only chunk rates; legacy proxy key names retained",
        "limitation": "Cache/metric/checkpoint overhead and concurrent load vary; estimates are planning proxies, not guaranteed upper bounds or deadlines.",
        "by_model": runtime_by_model,
        "training_seconds": estimated_train,
        "validation_forward_seconds_proxy": estimated_validation,
        "evaluation_forward_seconds_proxy": estimated_evaluation,
        "sum_seconds_proxy": raw_total,
        "sum_hours_proxy": raw_total / 3600,
        "sum_plus_20_percent_overhead_seconds": raw_total * 1.2,
        "sum_plus_20_percent_overhead_hours": raw_total * 1.2 / 3600,
    }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_market_fitting",
        "configuration": {key: cfg[key] for key in ("batch_size", "chunk_length", "epochs", "seeds", "models")},
        "matched_widths": widths,
        "model_parameters": model_parameters,
        "splits": splits,
        "volume": volume,
        "runtime_estimate": runtime,
        "provenance": {
            "git_commit_at_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": config_sha,
            "prepared_data_sha256": audit["provenance"]["data_sha256"],
            "data_preflight_sha256": sha256(cache / "audit.json"),
            "speed_preflight_sha256": sha256(speed_path),
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                              for path in (ROOT / "src/stream_batch_v3.py", ROOT / "src/stream_models_v3.py",
                                           ROOT / "src/crossing_memory_v2.py", ROOT / "scripts/run_market_v3.py",
                                           Path(__file__).resolve())},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/market-v3.json")
    parser.add_argument("--speed", type=Path, default=ROOT / "reports/v3/speed-stability.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/v3/frozen-budget.json")
    args = parser.parse_args()
    report = freeze(args.config, args.speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"splits": {name: {key: row[key] for key in
                                       ("rows", "supervised_rows", "total_chunks", "optimizer_updates_per_epoch")}
                                  for name, row in report["splits"].items()},
                      "volume": report["volume"],
                      "runtime_estimate_hours_with_20_percent_overhead":
                      report["runtime_estimate"]["sum_plus_20_percent_overhead_hours"]}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
