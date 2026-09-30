"""Fake-artifact checks of v3.1 scoring barriers; no real splits or fits."""

import json

import pytest

from scripts import score_market_v31 as score


def _fake_reports(tmp_path, cfg, provenance):
    report_dir = tmp_path / "reports/v31"
    report_dir.mkdir(parents=True)
    for model in cfg["models"]:
        for seed in cfg["seeds"]:
            checkpoint = tmp_path / f"{model}-{seed}.pt"
            checkpoint.write_bytes(f"fake-{model}-{seed}".encode())
            selected = {"path": str(checkpoint), "sha256": score.digest(checkpoint), "epoch": 4}
            report = {"complete": True, "provenance": provenance, "config": cfg,
                      "model": model, "seed": seed,
                      "checkpoints": {"short_best": selected, "full_best": selected}}
            (report_dir / f"fit-{model}-{seed}.json").write_text(json.dumps(report))
    return report_dir


def test_all_fifteen_fit_identities_and_checkpoint_hashes_gate_calibration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"models": ["a", "b", "c", "no_transfer", "no_open"],
           "seeds": [7, 19, 43], "output": str(tmp_path / "cache")}
    provenance = {"frozen": "source-and-cache-hashes"}
    reports = _fake_reports(tmp_path, cfg, provenance)
    assert len(score.read_fits(cfg, provenance)) == 15
    monkeypatch.setattr(score, "load_split", lambda *args: pytest.fail("Calibration labels loaded early"))

    missing = reports / "fit-no_open-43.json"
    missing.unlink()
    with pytest.raises(RuntimeError, match="All15"):
        score.calibrate(cfg, provenance)

    _fake = {"complete": True, "provenance": provenance, "config": cfg,
             "model": "no_open", "seed": 43,
             "checkpoints": {"short_best": {"path": str(tmp_path / "no_open-43.pt"),
                                            "sha256": "wrong", "epoch": 4},
                             "full_best": {"path": str(tmp_path / "no_open-43.pt"),
                                           "sha256": "wrong", "epoch": 4}}}
    missing.write_text(json.dumps(_fake))
    with pytest.raises(RuntimeError, match="Checkpoint hash mismatch"):
        score.calibrate(cfg, provenance)


def test_calibrator_freeze_hash_blocks_evaluation_before_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/v31").mkdir(parents=True)
    provenance = {"frozen": "source-and-cache-hashes"}
    (tmp_path / "reports/v31/calibrators.json").write_text("{}")
    (tmp_path / "reports/v31/calibration-freeze.json").write_text(json.dumps({
        "complete": True, "provenance": provenance, "calibrators_sha256": "wrong"}))
    monkeypatch.setattr(score, "read_fits", lambda *args: [])
    monkeypatch.setattr(score, "load_split", lambda *args: pytest.fail("Evaluation labels loaded early"))
    with pytest.raises(RuntimeError, match="Calibrator freeze mismatch"):
        score.evaluate({"output": str(tmp_path / "cache")}, provenance)
