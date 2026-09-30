"""Synthetic v3.1 split checks; no real evaluation outcomes are read."""

import json

import numpy as np
import pandas as pd

from src.market_data_v3 import prepare as prepare_v3
from src.market_data_v31 import prepare, sha256


def _environment(tmp_path):
    parts = []
    for start, n in (("2024-01-01", 120), ("2025-09-30 22:00", 120),
                     ("2025-10-01 00:00", 120), ("2026-01-01", 120)):
        ts = pd.date_range(start, periods=n, freq="min", tz="UTC")
        close = 2000 + 0.1 * np.arange(n)
        parts.append(pd.DataFrame({"timestamp": ts, "open": close-0.02,
                                   "high": close+0.05, "low": close-0.07, "close": close}))
    frame = pd.concat(parts, ignore_index=True)
    source_cfg = {"data": "synthetic.parquet", "train_start": "2024-01-01",
                  "validation_start": "2025-07-01", "evaluation_start": "2026-01-01",
                  "end": "2026-01-02", "horizon": 2, "warmup": 4,
                  "threshold_multipliers": [8, 4, 2], "feature_clip": 10.0,
                  "metadata_clip": 8.0}
    source_cache = tmp_path / "source"
    source_cache.mkdir()
    source_config_path = tmp_path / "source.json"
    source_config_path.write_text(json.dumps(source_cfg))
    old_splits, old_audit = prepare_v3(source_cfg, frame=frame)
    old_audit["provenance"] = {"config_sha256": sha256(source_config_path),
                               "data_sha256": "synthetic-only"}
    (source_cache / "audit.json").write_text(json.dumps(old_audit))
    hashes = {}
    for name, split in old_splits.items():
        arrays = {key: value for key, value in split.items() if key != "tree"}
        arrays.update({f"tree_{key}": value for key, value in split["tree"].items()})
        path = source_cache / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        hashes[name] = sha256(path)
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps({"splits": {name: {"cache_sha256": value}
                                                  for name, value in hashes.items()}}))
    cfg = {key: value for key, value in source_cfg.items() if key != "validation_start"}
    cfg.update(selection_start="2025-07-01", calibration_start="2025-10-01",
               source_cache=str(source_cache), source_config=str(source_config_path),
               source_budget=str(budget_path))
    return frame, cfg, old_splits


def test_outer_splits_exact_and_calibration_restarts_at_boundary(tmp_path):
    frame, cfg, old_splits = _environment(tmp_path)
    splits, audit = prepare(cfg, frame=frame)
    for name in ("train", "evaluation"):
        for key in ("x", "y", "valid", "reset", "timestamp"):
            np.testing.assert_array_equal(splits[name][key], old_splits[name][key])
        for key in old_splits[name]["tree"]:
            np.testing.assert_array_equal(splits[name]["tree"][key], old_splits[name]["tree"][key])
    selection, calibration = splits["selection"], splits["calibration"]
    assert len(selection["timestamp"]) == len(calibration["timestamp"]) == 120
    assert selection["timestamp"][-1] + 60 == calibration["timestamp"][0]
    assert selection["reset"][0] and calibration["reset"][0]
    assert calibration["tree"]["ids"][0].tolist() == [0, -1, -1, -1]
    assert not selection["valid"][-2:].any()  # horizon purged before calibration
    assert not calibration["valid"][:4].any()  # independent warmup after boundary
    assert audit["splits"]["selection"]["supervised_rows"] == 114
    assert audit["splits"]["calibration"]["supervised_rows"] == 114
    assert audit["source_v3"]["train_cache_sha256"]


def test_frozen_normalization_and_hash_mismatch_guard(tmp_path):
    frame, cfg, _ = _environment(tmp_path)
    _, audit = prepare(cfg, frame=frame)
    source = json.loads((tmp_path / "source/audit.json").read_text())
    assert audit["feature_mean_frozen_train"] == source["feature_mean_train"]
    assert audit["metadata_std_frozen_train_by_depth"] == source["metadata_std_train_by_depth"]
    source_cache = tmp_path / "source/train.npz"
    with source_cache.open("ab") as stream:
        stream.write(b"changed")
    import pytest
    with pytest.raises(ValueError, match="cache hash"):
        prepare(cfg, frame=frame)
