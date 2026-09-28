"""Small synthetic checks for the frozen market protocol; no real test labels read."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_experiment import metrics, prepare


def fixture_data(tmp_path):
    pieces = []
    for start in ["2024-01-01", "2025-07-01", "2026-01-01"]:
        times = pd.date_range(start, periods=100, freq="min", tz="UTC")
        close = 2000 + np.arange(100) * 0.1
        close[70:74] = close[69]  # a true zero return for a two-minute horizon
        piece = pd.DataFrame({
            "timestamp": times,
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.07,
            "close": close,
            "volume": np.ones(100, dtype=np.int64),
        })
        pieces.append(piece.drop(index=50))  # one missing minute per period
    frame = pd.concat(pieces, ignore_index=True)
    path = tmp_path / "tiny.parquet"
    frame.to_parquet(path)
    cfg = {
        "data": str(path), "train_start": "2024-01-01",
        "validation_start": "2025-07-01", "test_start": "2026-01-01",
        "end": "2026-01-02", "window": 4, "horizon": 2, "stride": 1,
        "max_train": 1000, "max_validation": 1000, "max_test": 1000,
        "threshold_multipliers": [2, 4, 8],
    }
    return frame, cfg


def test_prepare_uses_real_utc_seconds_and_keeps_windows_and_targets_in_split(tmp_path):
    frame, cfg = fixture_data(tmp_path)
    splits, info = prepare(cfg)
    present = set(frame.timestamp.dt.as_unit("ns").astype("int64").to_numpy() // 10**9)
    bounds = [int(pd.Timestamp(cfg[key], tz="UTC").timestamp()) for key in
              ["train_start", "validation_start", "test_start", "end"]]
    for k, name in enumerate(["train", "validation", "test"]):
        split = splits[name]
        assert len(split["y"]) > 0
        assert info["splits"][name]["samples"] == len(split["y"])
        assert np.all(split["timestamp"] >= bounds[k] + (cfg["window"] - 1) * 60)
        assert np.all(split["timestamp"] + cfg["horizon"] * 60 < bounds[k + 1])
        for end in split["timestamp"]:
            assert set(range(end - (cfg["window"] - 1) * 60,
                             end + (cfg["horizon"] + 1) * 60, 60)).issubset(present)
    assert 0 in splits["train"]["y"]  # zero target return maps to non-up


def test_future_target_changes_label_but_not_past_features(tmp_path):
    frame, cfg = fixture_data(tmp_path)
    before, _ = prepare(cfg)
    target = pd.Timestamp("2026-01-01 00:12:00", tz="UTC")
    altered = frame.copy()
    row = altered.index[altered.timestamp == target][0]
    altered.loc[row, ["open", "high", "low", "close"]] = [1900.0, 1900.05, 1899.95, 1900.0]
    altered.to_parquet(cfg["data"])
    after, _ = prepare(cfg)
    end = int(pd.Timestamp("2026-01-01 00:10:00", tz="UTC").timestamp())
    before_i = np.flatnonzero(before["test"]["timestamp"] == end).item()
    after_i = np.flatnonzero(after["test"]["timestamp"] == end).item()
    np.testing.assert_array_equal(before["test"]["x"][before_i], after["test"]["x"][after_i])
    np.testing.assert_array_equal(before["test"]["stats"][before_i], after["test"]["stats"][after_i])
    assert before["test"]["y"][before_i] == 1
    assert after["test"]["y"][after_i] == 0


def test_metrics_known_values_and_single_class_auc():
    result = metrics(np.array([0, 1]), np.array([0.0, 0.1]),
                     np.array([0.2, 0.8]), np.array([0.0, 0.05]))
    assert result["log_loss"] == pytest.approx(-np.log(0.8))
    assert result["roc_auc"] == 1.0
    assert result["brier"] == pytest.approx(0.04)
    assert result["ece"] == pytest.approx(0.2)
    assert result["mae"] == pytest.approx(0.025)
    assert sum(row["count"] for row in result["calibration"]) == 2
    one_class = metrics(np.array([0, 0]), np.array([0.0, 0.0]),
                        np.array([0.2, 0.3]), np.array([0.0, 0.0]))
    assert one_class["roc_auc"] is None


def test_zero_horizon_is_rejected_before_label_construction(tmp_path):
    _, cfg = fixture_data(tmp_path)
    cfg["horizon"] = 0
    with pytest.raises(ValueError, match="horizon"):
        prepare(cfg)
