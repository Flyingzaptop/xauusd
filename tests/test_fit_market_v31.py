import json
from pathlib import Path

import numpy as np
import torch

from scripts.fit_market_v31 import fit_one
from src.crossing_memory_v2 import route_windows_v2
from src.stream_batch_v3 import PackedStream, TREE_KEYS


def _split(offset: int) -> dict:
    paths = ([0, 1, 2, 1, 0, -1], [0, -1, -2, -1, 0, 1])
    pieces = []
    for segment, values in enumerate(paths):
        z = np.asarray(values, dtype=np.float32) + offset * 0.1
        changes = np.diff(z, prepend=z[:1])
        x = np.stack((changes, z / 5, np.sin(z), np.cos(z)), axis=-1).astype(np.float32)
        valid = np.ones(len(z), dtype=np.bool_)
        valid[0] = False
        y = (np.arange(len(z)) % 2).astype(np.float32)
        y[~valid] = 0
        reset = np.zeros(len(z), dtype=np.bool_)
        reset[0] = True
        timestamp = 1_700_000_000 + segment * 1_000 + np.arange(len(z)) * 60
        routed = route_windows_v2(z[None], [3, 2, 1])
        pieces.append((x, y, valid, reset, timestamp.astype(np.int64),
                       {key: routed[key][0] for key in TREE_KEYS}))
    return {
        "x": np.concatenate([part[0] for part in pieces]),
        "y": np.concatenate([part[1] for part in pieces]),
        "valid": np.concatenate([part[2] for part in pieces]),
        "reset": np.concatenate([part[3] for part in pieces]),
        "timestamp": np.concatenate([part[4] for part in pieces]),
        "tree": {key: np.concatenate([part[5][key] for part in pieces]) for key in TREE_KEYS},
    }


def _cfg(tmp_path: Path) -> dict:
    return {
        "output": str(tmp_path / "cache"), "models": ["a"], "seeds": [7],
        "threads": 1, "batch_size": 2, "chunk_length": 4,
        "c_gru_hidden": 8, "memory_hidden": 4,
        "learning_rate": 0.001, "gradient_clip": 1.0,
        "min_epochs": 4, "max_epochs": 5, "patience": 2, "min_delta": 1e-5,
    }


def _tensors_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_fit_checkpoint_roles_and_deterministic_epoch_resume(tmp_path):
    cfg = _cfg(tmp_path)
    train, selection = _split(0), _split(1)
    packer = PackedStream(train, cfg["batch_size"], cfg["chunk_length"])
    expected_supervised = int(train["valid"].sum())
    expected_updates = sum(batch["valid"].any() for batch in packer)
    provenance = {"code_commit": "unit-test", "cache_sha256": {"train": "fixture",
                                                                  "selection": "fixture"}}

    def loader(_cache, name):
        return train if name == "train" else selection

    common = dict(split_loader=loader, expected_supervised=expected_supervised,
                  expected_updates=expected_updates)
    continuous_root = tmp_path / "continuous"
    uninterrupted = fit_one(cfg, "a", 7, provenance,
                            checkpoint_root=continuous_root / "checkpoints",
                            report_dir=continuous_root / "reports", **common)
    assert uninterrupted["complete"] is True
    assert uninterrupted["stop_reason"] == "budget_cap"
    assert uninterrupted["epochs_completed"] == 5
    assert uninterrupted["total_optimizer_updates"] == expected_updates * 5
    assert len(uninterrupted["epoch_curve"]) == 5

    interrupted_root = tmp_path / "interrupted"
    partial = fit_one(cfg, "a", 7, provenance, interrupt_after_epoch=2,
                      checkpoint_root=interrupted_root / "checkpoints",
                      report_dir=interrupted_root / "reports", **common)
    assert partial["complete"] is False
    assert partial["epochs_completed"] == 2
    assert partial["checkpoints"]["short_best"] is None
    full_path = interrupted_root / "checkpoints" / "a-7" / "full-best.pt"
    full_path.unlink()  # simulate crash after last.pt and before best reconciliation
    resumed = fit_one(cfg, "a", 7, provenance, resume=True,
                      checkpoint_root=interrupted_root / "checkpoints",
                      report_dir=interrupted_root / "reports", **common)
    assert resumed["complete"] is True
    assert resumed["total_optimizer_updates"] == uninterrupted["total_optimizer_updates"]
    assert [(row["train_log_loss"], row["selection"]["log_loss"])
            for row in resumed["epoch_curve"]] == [
                (row["train_log_loss"], row["selection"]["log_loss"])
                for row in uninterrupted["epoch_curve"]]

    for role in ("short_best", "full_best", "last"):
        left = torch.load(uninterrupted["checkpoints"][role]["path"], weights_only=False)
        right = torch.load(resumed["checkpoints"][role]["path"], weights_only=False)
        _tensors_equal(left["model_state_dict"], right["model_state_dict"])
    assert 1 <= resumed["checkpoints"]["short_best"]["epoch"] <= 4
    assert resumed["checkpoints"]["full_best"]["epoch"] == resumed["control_state"]["best_epoch"]
    assert resumed["checkpoints"]["short_best"]["epoch"] == resumed["control_state"]["short_best_epoch"]
    checkpoint_files = sorted(path.name for path in (interrupted_root / "checkpoints" / "a-7").iterdir())
    assert checkpoint_files == ["full-best.pt", "last.pt", "short-best.pt"]
    saved_report = json.loads((interrupted_root / "reports" / "fit-a-7.json").read_text())
    assert saved_report["complete"] and saved_report["checkpoints"] == resumed["checkpoints"]
    hashes_before = {role: resumed["checkpoints"][role]["sha256"]
                     for role in ("short_best", "full_best", "last")}
    completed_again = fit_one(cfg, "a", 7, provenance, resume=True,
                              checkpoint_root=interrupted_root / "checkpoints",
                              report_dir=interrupted_root / "reports", **common)
    assert {role: completed_again["checkpoints"][role]["sha256"]
            for role in hashes_before} == hashes_before


def test_fit_refuses_accidental_overwrite_and_wrong_resume(tmp_path):
    cfg = _cfg(tmp_path)
    train, selection = _split(0), _split(1)
    provenance = {"code_commit": "unit-test", "cache_sha256": {"train": "fixture",
                                                                  "selection": "fixture"}}

    def loader(_cache, name):
        return train if name == "train" else selection

    paths = dict(checkpoint_root=tmp_path / "checkpoints", report_dir=tmp_path / "reports")
    from pytest import raises
    with raises(FileNotFoundError):
        fit_one(cfg, "a", 7, provenance, resume=True, split_loader=loader,
                expected_supervised=10, expected_updates=2, **paths)
    fit_one(cfg, "a", 7, provenance, interrupt_after_epoch=1, split_loader=loader,
            expected_supervised=10, expected_updates=2, **paths)
    with raises(FileExistsError):
        fit_one(cfg, "a", 7, provenance, split_loader=loader,
                expected_supervised=10, expected_updates=2, **paths)
    with raises(RuntimeError, match="provenance"):
        fit_one(cfg, "a", 7, {"code_commit": "different", "cache_sha256": provenance["cache_sha256"]},
                resume=True, split_loader=loader, expected_supervised=10,
                expected_updates=2, **paths)
