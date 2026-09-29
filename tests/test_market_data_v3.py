"""Small chronological streams for v3 preprocessing; no market labels inspected."""

import numpy as np
import pandas as pd
import pytest

from src.market_data_v3 import load_split, prepare


def _frame_and_config():
    pieces = []
    for start in ("2024-01-01", "2025-07-01", "2026-01-01"):
        timestamp = pd.date_range(start, periods=100, freq="min", tz="UTC")
        close = 2000 + 0.1 * np.arange(100) + 0.02 * np.sin(np.arange(100))
        piece = pd.DataFrame({"timestamp": timestamp, "open": close - 0.02,
                              "high": close + 0.05, "low": close - 0.07,
                              "close": close})
        pieces.append(piece.drop(index=50))
    frame = pd.concat(pieces, ignore_index=True)
    cfg = {"train_start": "2024-01-01", "validation_start": "2025-07-01",
           "evaluation_start": "2026-01-01", "end": "2026-01-02",
           "horizon": 2, "warmup": 4, "threshold_multipliers": [8, 4, 2],
           "feature_clip": 10.0, "metadata_clip": 8.0}
    return frame, cfg


def test_every_gap_and_split_resets_and_labels_stay_within_segment():
    frame, cfg = _frame_and_config()
    splits, audit = prepare(cfg, frame=frame)
    for name in ("train", "validation", "evaluation"):
        split = splits[name]
        ts = split["timestamp"]
        assert split["reset"][0]
        np.testing.assert_array_equal(split["reset"][1:], np.diff(ts) != 60)
        assert split["reset"].sum() == 2
        assert len(ts) == 99
        assert audit["splits"][name]["rows"] == 99
        assert audit["splits"][name]["segments"] == 2
        assert split["x"].shape == (99, 4)
        assert split["tree"]["metadata"].shape == (99, 4, 6)
        assert split["tree"]["active"].shape == (99, 4)
        assert split["tree"]["ids"][0, 0] == 0
        assert split["tree"]["ids"][50, 0] == 0
        for i, is_valid in enumerate(split["valid"]):
            segment_start = 0 if i < 50 else 50
            segment_end = 49 if i < 50 else 98
            expected = i - segment_start >= 4 and i + 2 <= segment_end
            assert bool(is_valid) == expected
            if is_valid:
                source_row = frame.loc[frame.timestamp == pd.Timestamp(int(ts[i]), unit="s", tz="UTC")].iloc[0]
                target_row = frame.loc[frame.timestamp == pd.Timestamp(int(ts[i + 2]), unit="s", tz="UTC")].iloc[0]
                assert split["y"][i] == float(target_row.close > source_row.close)
            else:
                assert split["y"][i] == 0  # placeholder, never a supervised label


def test_train_only_threshold_and_normalization_ignore_later_prices():
    frame, cfg = _frame_and_config()
    original, first_audit = prepare(cfg, frame=frame)
    changed = frame.copy()
    later = changed.timestamp >= pd.Timestamp("2025-07-01", tz="UTC")
    offset = np.linspace(0, 50, int(later.sum()))
    for column in ("open", "high", "low", "close"):
        changed.loc[later, column] += offset
    revised, second_audit = prepare(cfg, frame=changed)
    for key in ("thresholds_log_price", "feature_mean_train", "feature_std_train",
                "metadata_mean_train_by_depth", "metadata_std_train_by_depth"):
        assert first_audit[key] == second_audit[key]
    np.testing.assert_array_equal(original["train"]["x"], revised["train"]["x"])
    np.testing.assert_array_equal(original["train"]["tree"]["metadata"],
                                  revised["train"]["tree"]["metadata"])
    assert not np.array_equal(original["evaluation"]["x"], revised["evaluation"]["x"])


def test_bad_ohlc_or_unordered_time_rejected():
    frame, cfg = _frame_and_config()
    broken = frame.copy()
    broken.loc[5, "high"] = broken.loc[5, "low"] - 1
    with pytest.raises(ValueError, match="OHLC"):
        prepare(cfg, frame=broken)
    unordered = frame.copy()
    unordered.loc[5, "timestamp"] = unordered.loc[4, "timestamp"]
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare(cfg, frame=unordered)


def test_cache_round_trip(tmp_path):
    frame, cfg = _frame_and_config()
    splits, _ = prepare(cfg, frame=frame)
    split = splits["train"]
    arrays = {key: value for key, value in split.items() if key != "tree"}
    arrays.update({f"tree_{key}": value for key, value in split["tree"].items()})
    np.savez_compressed(tmp_path / "train.npz", **arrays)
    loaded = load_split(tmp_path, "train")
    for key in arrays:
        actual = loaded["tree"][key[5:]] if key.startswith("tree_") else loaded[key]
        np.testing.assert_array_equal(arrays[key], actual)
