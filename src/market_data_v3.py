"""Continuous, chronological M1 preparation for the v3 research experiment.

Only the train split fits thresholds and normalization. No model is trained here.
The evaluation interval is already exposed and is not an untouched holdout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.crossing_memory_v2 import TemporalEventRouter


TREE_KEYS = ("metadata", "births", "closures", "observer", "active", "ids",
             "parent_ids", "birth_ids", "closure_ids", "closure_parent_ids",
             "closure_reasons")
FEATURE_NAMES = ("log_close_delta", "log_close_over_open", "log_high_over_close",
                 "log_low_over_close")
METADATA_NAMES = ("direction", "log_age", "signed_log_progress", "log_max_excursion",
                  "log_retreats", "signed_log_latest_move")


def _boundary(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp())


def _validate_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    expected = ["timestamp", "open", "high", "low", "close"]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamp.isna().any():
        raise ValueError("Null timestamp")
    # pandas 3 may store datetimes in microseconds; force nanoseconds explicitly.
    ts = timestamp.dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000_000
    if np.any(np.diff(ts) <= 0):
        raise ValueError("Timestamps must be strictly increasing and unique")
    if np.any(ts % 60):
        raise ValueError("Timestamps must align to UTC minutes")
    ohlc = frame[expected[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(ohlc).all() or np.any(ohlc <= 0):
        raise ValueError("OHLC must be finite and positive")
    o, h, l, c = ohlc.T
    if np.any((h < l) | (o < l) | (o > h) | (c < l) | (c > h)):
        raise ValueError("Invalid OHLC envelope")
    return ts, ohlc


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _route(log_close: np.ndarray, reset: np.ndarray, thresholds: np.ndarray,
           progress: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
    sample = TemporalEventRouter(thresholds)._empty_tick()
    n = len(log_close)
    result = {key: np.empty((n,) + np.shape(sample[key]), dtype=np.asarray(sample[key]).dtype)
              for key in TREE_KEYS}
    router = None
    for i, value in enumerate(log_close):
        if reset[i]:
            router = TemporalEventRouter(thresholds)
        tick = router.step(value)
        for key in TREE_KEYS:
            result[key][i] = tick[key]
        if progress is not None and (i + 1) % 100_000 == 0:
            progress(i + 1, n)
    return result


def _raw_split(ts: np.ndarray, ohlc: np.ndarray, start: int, end: int,
               horizon: int, warmup: int, thresholds: np.ndarray,
               progress: Callable[[int, int], None] | None = None) -> dict:
    select = (ts >= start) & (ts < end)
    stamp = ts[select]
    candle = ohlc[select]
    if not len(stamp):
        raise ValueError(f"Empty split [{start}, {end})")
    reset = np.empty(len(stamp), dtype=np.bool_)
    reset[0] = True
    reset[1:] = np.diff(stamp) != 60
    logp = np.log(candle)
    log_close = logp[:, 3]
    delta = np.zeros(len(stamp), dtype=np.float64)
    delta[1:] = np.diff(log_close)
    delta[reset] = 0.0
    x = np.stack((delta, log_close-logp[:, 0], logp[:, 1]-log_close,
                  logp[:, 2]-log_close), axis=-1).astype(np.float32)
    tree = _route(log_close, reset, thresholds, progress)
    metadata = tree["metadata"]
    metadata[:, :, 2] = _signed_log1p(metadata[:, :, 2])
    metadata[:, :, 3] = np.log1p(metadata[:, :, 3])
    metadata[:, :, 5] = _signed_log1p(metadata[:, :, 5])
    segments = np.cumsum(reset, dtype=np.int64)
    index = np.arange(len(stamp))
    target = index + horizon
    in_range = target < len(stamp)
    # Equal segment IDs prove every intervening timestamp is one minute apart.
    same_session = np.zeros(len(stamp), dtype=np.bool_)
    same_session[in_range] = segments[target[in_range]] == segments[in_range]
    age_in_session = index - np.maximum.accumulate(np.where(reset, index, 0))
    valid = (age_in_session >= warmup) & same_session
    y = np.zeros(len(stamp), dtype=np.float32)
    y[valid] = (log_close[target[valid]] > log_close[valid]).astype(np.float32)
    return {"x": x, "tree": tree, "y": y, "valid": valid, "reset": reset,
            "timestamp": stamp.astype(np.int64)}


def _abs_quantiles(array: np.ndarray, axis: int) -> list:
    return np.quantile(np.abs(array), [0.5, 0.95, 0.99, 1.0], axis=axis).tolist()


def load_split(cache_dir: str | Path, name: str) -> dict:
    """Load one prepared split without pickle or changes to stored arrays."""
    if name not in {"train", "validation", "evaluation"}:
        raise ValueError(f"Unknown split: {name}")
    with np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=False) as file:
        plain = {key: file[key].copy() for key in ("x", "y", "valid", "reset", "timestamp")}
        plain["tree"] = {key: file[f"tree_{key}"].copy() for key in TREE_KEYS}
    n = len(plain["timestamp"])
    if any(len(value) != n for key, value in plain.items() if key != "tree") \
            or any(len(value) != n for value in plain["tree"].values()):
        raise ValueError("Cache arrays have inconsistent row counts")
    return plain


def prepare(config: Mapping, frame: pd.DataFrame | None = None,
            progress: Callable[[str, int, int], None] | None = None) -> tuple[dict, dict]:
    """Prepare all rows in train, validation and exposed research evaluation.

    `valid` is the only supervised-loss mask. Invalid `y` entries are placeholders.
    The first `warmup` closes of each segment and the final `horizon` before a
    gap or split are kept as sequence context but have no supervised label.
    """
    cfg = dict(config)
    horizon, warmup = int(cfg["horizon"]), int(cfg["warmup"])
    if horizon <= 0 or warmup < 0:
        raise ValueError("horizon must be positive and warmup nonnegative")
    bounds = [_boundary(cfg[k]) for k in ("train_start", "validation_start",
                                          "evaluation_start", "end")]
    if not all(a < b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Split boundaries must be strictly increasing")
    if frame is None:
        frame = pd.read_parquet(cfg["data"], columns=["timestamp", "open", "high", "low", "close"])
    ts, ohlc = _validate_frame(frame)
    train_mask = (ts >= bounds[0]) & (ts < bounds[1])
    if not train_mask.any():
        raise ValueError("Empty training interval")
    train_ts = ts[train_mask]
    train_close = np.log(ohlc[train_mask, 3])
    contiguous = np.diff(train_ts) == 60
    changes = np.abs(np.diff(train_close)[contiguous])
    if not len(changes):
        raise ValueError("No consecutive training minutes for threshold basis")
    basis = float(np.median(changes))
    if not np.isfinite(basis) or basis <= 0:
        raise ValueError("Degenerate training threshold basis")
    multipliers = np.asarray(cfg["threshold_multipliers"], dtype=np.float64)
    if multipliers.ndim != 1 or len(multipliers) == 0 or not np.all(np.isfinite(multipliers)) \
            or np.any(multipliers <= 0) or np.any(np.diff(multipliers) >= 0):
        raise ValueError("Threshold multipliers must be positive, finite, coarse to fine")
    thresholds = basis * multipliers
    names = ("train", "validation", "evaluation")
    splits = {}
    for k, name in enumerate(names):
        callback = (lambda done, total, name=name: progress(name, done, total)) if progress else None
        splits[name] = _raw_split(ts, ohlc, bounds[k], bounds[k+1], horizon, warmup,
                                  thresholds, callback)
    train_x = splits["train"]["x"]
    train_meta = splits["train"]["tree"]["metadata"]
    feature_mean = train_x.mean(axis=0, dtype=np.float64)
    feature_std = train_x.std(axis=0, dtype=np.float64).clip(1e-8)
    metadata_mean = train_meta.mean(axis=0, dtype=np.float64)
    metadata_std = train_meta.std(axis=0, dtype=np.float64).clip(1e-8)
    feature_clip = float(cfg["feature_clip"])
    metadata_clip = float(cfg["metadata_clip"])
    if feature_clip <= 0 or metadata_clip <= 0:
        raise ValueError("Clip bounds must be positive")
    train_feature_normalized = (train_x - feature_mean) / feature_std
    train_metadata_normalized = (train_meta - metadata_mean) / metadata_std
    audit = {
        "periods_utc_exclusive_end": {name: {"start": cfg[key], "end": cfg[next_key]}
                                      for name, key, next_key in zip(names,
                                          ("train_start", "validation_start", "evaluation_start"),
                                          ("validation_start", "evaluation_start", "end"))},
        "threshold_basis_train_median_abs_contiguous_log_return": basis,
        "threshold_multipliers_coarse_to_fine": multipliers.tolist(),
        "thresholds_log_price": thresholds.tolist(),
        "feature_names": FEATURE_NAMES,
        "metadata_names": METADATA_NAMES,
        "feature_mean_train": feature_mean.tolist(),
        "feature_std_train": feature_std.tolist(),
        "metadata_mean_train_by_depth": metadata_mean.tolist(),
        "metadata_std_train_by_depth": metadata_std.tolist(),
        "train_feature_abs_quantiles_p50_p95_p99_max": _abs_quantiles(train_x, axis=0),
        "train_metadata_abs_quantiles_p50_p95_p99_max_by_depth": _abs_quantiles(train_meta, axis=0),
        "train_feature_preclip_abs_max": np.max(np.abs(train_feature_normalized), axis=0).tolist(),
        "train_metadata_preclip_abs_max_by_depth": np.max(np.abs(train_metadata_normalized), axis=0).tolist(),
        "train_feature_clipped_entries": int(np.count_nonzero(np.abs(train_feature_normalized) > feature_clip)),
        "train_metadata_clipped_entries": int(np.count_nonzero(np.abs(train_metadata_normalized) > metadata_clip)),
        "train_feature_clipped_fraction": float(np.mean(np.abs(train_feature_normalized) > feature_clip)),
        "train_metadata_clipped_fraction": float(np.mean(np.abs(train_metadata_normalized) > metadata_clip)),
        "splits": {},
    }
    for name, split in splits.items():
        split["x"] = np.clip((split["x"] - feature_mean) / feature_std,
                             -feature_clip, feature_clip).astype(np.float32)
        split["tree"]["metadata"] = np.clip((split["tree"]["metadata"] - metadata_mean)
                                                   / metadata_std, -metadata_clip, metadata_clip).astype(np.float32)
        stamp = split["timestamp"]
        boundaries = np.flatnonzero(split["reset"])
        segment_lengths = np.diff(np.r_[boundaries, len(stamp)])
        age_in_session = np.arange(len(stamp)) - np.maximum.accumulate(
            np.where(split["reset"], np.arange(len(stamp)), 0))
        warmup_rows = int(np.count_nonzero(age_in_session < warmup))
        audit["splits"][name] = {
            "rows": len(stamp), "segments": int(split["reset"].sum()),
            "supervised_rows": int(split["valid"].sum()),
            "warmup_rows": warmup_rows,
            "horizon_or_gap_excluded_after_warmup_rows": int((~split["valid"]).sum()) - warmup_rows,
            "segment_length_min_median_max": [int(segment_lengths.min()),
                                               float(np.median(segment_lengths)),
                                               int(segment_lengths.max())],
            "max_available_history_observations": int(age_in_session.max() + 1),
            "first_timestamp_utc": pd.Timestamp(int(stamp[0]), unit="s", tz="UTC").isoformat(),
            "last_timestamp_utc": pd.Timestamp(int(stamp[-1]), unit="s", tz="UTC").isoformat(),
        }
    return splits, audit
