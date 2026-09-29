"""Read-only gates and prediction mapping for the v3 runner; never call fit."""

import json

import numpy as np
import pytest
import torch
from scipy.special import expit

from scripts import run_market_v3 as runner
from src.stream_models_v3 import StreamForecaster, matched_baseline_widths, parameter_count


def test_incomplete_fit_reports_block_evaluation_before_loading_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/v3").mkdir(parents=True)
    cfg = {"seeds": [7, 19, 43], "models": ["a", "b", "c", "no_transfer", "no_open"],
           "output": str(tmp_path / "cache")}
    provenance = {"fixed": "hash"}
    for seed in cfg["seeds"]:
        for name in cfg["models"]:
            (tmp_path / f"reports/v3/fit-{name}-{seed}.json").write_text(json.dumps({
                "complete": seed != 43, "provenance": provenance, "config": cfg,
                "runs": [{"model": name, "seed": seed}],
            }))
    requested = []
    monkeypatch.setattr(runner, "load_split", lambda *args: requested.append(args))
    with pytest.raises(RuntimeError, match="Incomplete or mismatched"):
        runner.evaluate_all(cfg, provenance)
    assert requested == []


def test_predict_restores_original_row_order_and_carries_chunk_state():
    n = 8
    timestamp = np.array([0, 60, 120, 180, 360, 420, 480, 540], dtype=np.int64)
    reset = np.array([True, False, False, False, True, False, False, False])
    x = np.zeros((n, 4), dtype=np.float32)
    x[:, 0] = np.arange(n, dtype=np.float32) / 10
    split = {"x": x, "y": np.zeros(n, dtype=np.float32),
             "valid": np.ones(n, dtype=np.bool_), "reset": reset, "timestamp": timestamp,
             "tree": {"metadata": np.zeros((n, 4, 6), dtype=np.float32),
                      "births": np.zeros((n, 4), dtype=np.bool_),
                      "closures": np.zeros((n, 4), dtype=np.bool_),
                      "observer": np.zeros(n, dtype=np.int64),
                      "active": np.tile(np.array([True, False, False, False]), (n, 1))}}

    class Echo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states_seen = []

        def forward(self, x, tree, reset, state):
            self.states_seen.append(state)
            return x[:, :, 0], {"chunk": len(self.states_seen)}

    model = Echo()
    probability = runner.predict(model, split, {"batch_size": 2, "chunk_length": 3})
    np.testing.assert_allclose(probability, expit(x[:, 0]), rtol=1e-6)
    assert model.states_seen[0] is None
    assert model.states_seen[1] == {"chunk": 1}


def test_matched_parameters_and_metric_definitions():
    widths = matched_baseline_widths(16, 8, 4)
    counts = {kind: parameter_count(StreamForecaster(kind, widths.get(kind, 16), 8, 4))
              for kind in ("a", "b", "c", "no_transfer", "no_open")}
    assert counts["c"] == counts["no_transfer"] == counts["no_open"]
    assert abs(counts["a"] - counts["c"]) / counts["c"] < 0.02
    assert abs(counts["b"] - counts["c"]) / counts["c"] < 0.02
    outcome = runner.metrics(np.array([0., 1.]), np.array([0.2, 0.8]))
    assert outcome["log_loss"] == pytest.approx(-np.log(0.8))
    assert outcome["auc"] == pytest.approx(1.0)
    assert outcome["brier"] == pytest.approx(0.04)
    assert outcome["ece"] == pytest.approx(0.2)
