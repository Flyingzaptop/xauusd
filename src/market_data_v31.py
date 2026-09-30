"""Four-way v3.1 M1 split using frozen v3 train parameters and source caches.

Selection and calibration are routed as separate streams. The v3 train and
exposed evaluation caches are reused byte for byte; evaluation outcomes are
never summarized by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.market_data_v3 import TREE_KEYS, _boundary, _raw_split, _validate_frame


SPLITS = ("train", "selection", "calibration", "evaluation")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_split(cache_dir: str | Path, name: str) -> dict:
    """Load a v3.1 split without pickle; retain full warmup and invalid tails."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split: {name}")
    with np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=False) as file:
        split = {key: file[key].copy() for key in ("x", "y", "valid", "reset", "timestamp")}
        split["tree"] = {key: file[f"tree_{key}"].copy() for key in TREE_KEYS}
    n = len(split["timestamp"])
    if any(len(split[key]) != n for key in ("x", "y", "valid", "reset")) \
            or any(len(value) != n for value in split["tree"].values()):
        raise ValueError("Cache arrays have inconsistent row counts")
    return split


def _population(split: dict, warmup: int) -> dict:
    stamp = split["timestamp"]
    reset = split["reset"]
    starts = np.flatnonzero(reset)
    lengths = np.diff(np.r_[starts, len(stamp)])
    age = np.arange(len(stamp)) - np.maximum.accumulate(np.where(reset, np.arange(len(stamp)), 0))
    warmup_rows = int(np.count_nonzero(age < warmup))
    return {
        "rows": len(stamp), "segments": int(reset.sum()),
        "supervised_rows": int(split["valid"].sum()),
        "warmup_rows": warmup_rows,
        "horizon_or_gap_excluded_after_warmup_rows": int((~split["valid"]).sum()) - warmup_rows,
        "segment_length_min_median_max": [int(lengths.min()), float(np.median(lengths)), int(lengths.max())],
        "first_timestamp_utc": pd.Timestamp(int(stamp[0]), unit="s", tz="UTC").isoformat(),
        "last_timestamp_utc": pd.Timestamp(int(stamp[-1]), unit="s", tz="UTC").isoformat(),
    }


def prepare(config: Mapping, frame: pd.DataFrame | None = None,
            progress: Callable[[str, int, int], None] | None = None) -> tuple[dict, dict]:
    """Return four splits and a population-only audit, with frozen v3 train stats."""
    cfg = dict(config)
    bounds = [_boundary(cfg[key]) for key in ("train_start", "selection_start",
                                               "calibration_start", "evaluation_start", "end")]
    if not all(a < b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("v3.1 split boundaries must increase strictly")
    source_cache = Path(cfg.get("source_cache", "artifacts/market-v3"))
    source_audit_path = Path(cfg.get("source_audit", source_cache / "audit.json"))
    source_config_path = Path(cfg.get("source_config", "configs/market-v3.json"))
    source_budget_path = Path(cfg.get("source_budget", "reports/v3/frozen-budget.json"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_budget = json.loads(source_budget_path.read_text(encoding="utf-8"))
    if source_audit["provenance"]["config_sha256"] != sha256(source_config_path):
        raise ValueError("Frozen v3 source config hash differs from its audit")
    for key in ("data", "train_start", "evaluation_start", "end", "horizon", "warmup",
                "threshold_multipliers", "feature_clip", "metadata_clip"):
        if cfg[key] != source_config[key]:
            raise ValueError(f"v3.1 must preserve frozen v3 {key}")
    if cfg["selection_start"] != source_config["validation_start"]:
        raise ValueError("Selection must start at the frozen v3 validation boundary")
    if not bounds[2] < bounds[3]:
        raise ValueError("Calibration must precede evaluation")
    hashes = {}
    for name in ("train", "evaluation"):
        path = source_cache / f"{name}.npz"
        hashes[name] = sha256(path)
        if hashes[name] != source_budget["splits"][name]["cache_sha256"]:
            raise ValueError(f"Frozen v3 {name} cache hash differs from budget")
    if frame is None:
        if sha256(cfg["data"]) != source_audit["provenance"]["data_sha256"]:
            raise ValueError("Raw M1 bytes differ from frozen v3 input")
        frame = pd.read_parquet(cfg["data"], columns=["timestamp", "open", "high", "low", "close"])
    ts, ohlc = _validate_frame(frame)
    thresholds = np.asarray(source_audit["thresholds_log_price"], dtype=np.float64)
    splits = {name: load_split(source_cache, name) for name in ("train", "evaluation")}
    for name, start, end in (("selection", bounds[1], bounds[2]),
                             ("calibration", bounds[2], bounds[3])):
        callback = (lambda done, total, name=name: progress(name, done, total)) if progress else None
        split = _raw_split(ts, ohlc, start, end, int(cfg["horizon"]), int(cfg["warmup"]),
                           thresholds, callback)
        feature_mean = np.asarray(source_audit["feature_mean_train"], dtype=np.float64)
        feature_std = np.asarray(source_audit["feature_std_train"], dtype=np.float64)
        metadata_mean = np.asarray(source_audit["metadata_mean_train_by_depth"], dtype=np.float64)
        metadata_std = np.asarray(source_audit["metadata_std_train_by_depth"], dtype=np.float64)
        split["x"] = np.clip((split["x"] - feature_mean) / feature_std,
                             -cfg["feature_clip"], cfg["feature_clip"]).astype(np.float32)
        split["tree"]["metadata"] = np.clip((split["tree"]["metadata"] - metadata_mean)
                                                   / metadata_std, -cfg["metadata_clip"],
                                                   cfg["metadata_clip"]).astype(np.float32)
        splits[name] = split
    periods = {name: {"start": cfg[start_key], "end": cfg[end_key]}
               for name, start_key, end_key in zip(SPLITS,
                   ("train_start", "selection_start", "calibration_start", "evaluation_start"),
                   ("selection_start", "calibration_start", "evaluation_start", "end"))}
    audit = {
        "periods_utc_exclusive_end": periods,
        "thresholds_log_price_frozen_train": source_audit["thresholds_log_price"],
        "threshold_basis_frozen_train": source_audit["threshold_basis_train_median_abs_contiguous_log_return"],
        "feature_mean_frozen_train": source_audit["feature_mean_train"],
        "feature_std_frozen_train": source_audit["feature_std_train"],
        "metadata_mean_frozen_train_by_depth": source_audit["metadata_mean_train_by_depth"],
        "metadata_std_frozen_train_by_depth": source_audit["metadata_std_train_by_depth"],
        "source_v3": {
            "audit_sha256": sha256(source_audit_path),
            "config_sha256": sha256(source_config_path),
            "raw_data_sha256": source_audit["provenance"]["data_sha256"],
            "train_cache_sha256": hashes["train"],
            "evaluation_cache_sha256": hashes["evaluation"],
        },
        "splits": {name: _population(splits[name], int(cfg["warmup"])) for name in SPLITS},
    }
    return splits, audit
