"""Train one frozen v3.1 model/seed with resumable epoch-boundary checkpoints.

Only train and selection caches are opened. Exactly three checkpoint names are
used per fit: `short-best.pt` (best of epochs 1..4, finalized at epoch 4),
`full-best.pt` (strict minimum selection log loss), and `last.pt` (recovery
only). The report is written atomically after checkpoint files each epoch.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_market_v3 import make_model, tensors  # noqa: E402
from src.inference_v31 import binary_metrics, predict_logits  # noqa: E402
from src.market_data_v31 import load_split  # noqa: E402
from src.provenance_v31 import digest, get_provenance  # noqa: E402
from src.stream_batch_v3 import PackedStream  # noqa: E402
from src.stream_models_v3 import matched_baseline_widths, parameter_count  # noqa: E402
from src.training_control_v31 import EarlyStop  # noqa: E402


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def _base_checkpoint(
    *, model: nn.Module, optimizer: torch.optim.Optimizer, control: EarlyStop,
    kind: str, seed: int, epoch: int, selection_loss: float, updates_total: int,
    epoch_curve: list, batch_curve: list, elapsed_seconds: float,
    maximum_state_abs: dict, maximum_gradient_norm_before_clip: float,
    provenance: dict,
) -> dict:
    # Deep copies are essential: state_dict() otherwise aliases live parameter
    # and optimizer tensors, causing earlier best weights to mutate later.
    return {
        "schema_version": 1, "model": kind, "seed": seed, "epoch": epoch,
        "selection_log_loss": selection_loss,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "control_state": copy.deepcopy(control.state_dict()),
        "rng_state": _rng_state(),
        "total_optimizer_updates": updates_total,
        "epoch_curve": copy.deepcopy(epoch_curve),
        "batch_curve": copy.deepcopy(batch_curve),
        "elapsed_seconds": elapsed_seconds,
        "maximum_state_abs": dict(maximum_state_abs),
        "maximum_gradient_norm_before_clip": maximum_gradient_norm_before_clip,
        "provenance": provenance,
    }


def _checkpoint_record(path: Path, payload: dict | None) -> dict | None:
    if payload is None:
        return None
    if not path.exists():
        raise RuntimeError(f"Expected checkpoint missing: {path}")
    return {"path": path.as_posix(), "sha256": digest(path),
            "epoch": payload["epoch"], "selection_log_loss": payload["selection_log_loss"]}


def _report(
    *, cfg: dict, provenance: dict, kind: str, seed: int, model: nn.Module,
    control: EarlyStop, epoch_curve: list, batch_curve: list, updates_total: int,
    supervised_per_epoch: int, elapsed_seconds: float, maximum_state_abs: dict,
    maximum_gradient_norm_before_clip: float, checkpoint_dir: Path,
    short_candidate: dict | None, full_best: dict | None, last: dict | None,
) -> dict:
    checkpoints = {
        "short_best": _checkpoint_record(checkpoint_dir / "short-best.pt",
                                         short_candidate if control.last_epoch >= cfg["min_epochs"] else None),
        "full_best": _checkpoint_record(checkpoint_dir / "full-best.pt", full_best),
        "last": _checkpoint_record(checkpoint_dir / "last.pt", last),
    }
    return {
        "provenance": provenance, "config": cfg, "model": kind, "seed": seed,
        "complete": control.should_stop, "epochs_completed": control.last_epoch,
        "stop_reason": control.stop_reason,
        "parameters": parameter_count(model), "gru_hidden": model.gru_hidden,
        "split_sha256": provenance["cache_sha256"],
        "epoch_curve": epoch_curve, "batch_curve": batch_curve,
        "total_optimizer_updates": updates_total,
        "supervised_examples_per_epoch": supervised_per_epoch,
        "training_seconds": elapsed_seconds,
        "maximum_state_abs": maximum_state_abs,
        "maximum_gradient_norm_before_clip": maximum_gradient_norm_before_clip,
        "all_gradients_finite": True,
        "control_state": control.state_dict(),
        "checkpoints": checkpoints,
    }


def fit_one(
    cfg: dict, kind: str, seed: int, provenance: dict, *, resume: bool = False,
    split_loader=load_split, inference=predict_logits,
    checkpoint_root: Path | None = None, report_dir: Path | None = None,
    expected_supervised: int = 491_102, expected_updates: int = 294,
    interrupt_after_epoch: int | None = None,
) -> dict:
    """Fit a single model. `interrupt_after_epoch` is an internal test hook."""
    if kind not in cfg["models"] or seed not in cfg["seeds"]:
        raise ValueError("Model and seed must be preregistered")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    cache = Path(cfg["output"])
    train = split_loader(cache, "train")
    selection = split_loader(cache, "selection")
    packer = PackedStream(train, cfg["batch_size"], cfg["chunk_length"])
    if packer.n_valid_labels != expected_supervised or sum(batch["valid"].any() for batch in packer) != expected_updates:
        raise RuntimeError("Train label/update counts differ from frozen v3.1 budget")
    widths = matched_baseline_widths(cfg["c_gru_hidden"], cfg["memory_hidden"], 4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_model(kind, cfg, widths)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    control = EarlyStop(cfg["min_epochs"], cfg["max_epochs"], cfg["min_delta"], cfg["patience"])
    checkpoint_dir = (checkpoint_root or cache / "checkpoints") / f"{kind}-{seed}"
    report_path = (report_dir or ROOT / "reports/v31") / f"fit-{kind}-{seed}.json"
    last_path = checkpoint_dir / "last.pt"
    short_path = checkpoint_dir / "short-best.pt"
    full_path = checkpoint_dir / "full-best.pt"
    epoch_curve: list[dict] = []
    batch_curve: list[dict] = []
    updates_total = 0
    elapsed_seconds = 0.0
    maximum_state_abs = {"gru": 0.0, "tree": 0.0}
    maximum_gradient_norm_before_clip = 0.0
    short_candidate: dict | None = None
    full_best: dict | None = None
    last_payload: dict | None = None

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume without {last_path}")
        last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if (last_payload["provenance"] != provenance or last_payload["model"] != kind
                or last_payload["seed"] != seed):
            raise RuntimeError("Recovery checkpoint provenance/model/seed differs")
        if report_path.exists():
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            if prior.get("complete"):
                if (prior.get("provenance") != provenance or prior.get("config") != cfg
                        or (prior.get("model"), prior.get("seed")) != (kind, seed)
                        or not last_payload["control_state"]["should_stop"]
                        or prior.get("epochs_completed") != last_payload["epoch"]):
                    raise RuntimeError("Completed report differs from recovery checkpoint")
                for role in ("short_best", "full_best", "last"):
                    selected = prior["checkpoints"][role]
                    if selected is None or digest(selected["path"]) != selected["sha256"]:
                        raise RuntimeError(f"Completed {role} checkpoint hash differs")
                return prior
        model.load_state_dict(last_payload["model_state_dict"])
        optimizer.load_state_dict(last_payload["optimizer_state_dict"])
        control.load_state_dict(last_payload["control_state"])
        _restore_rng(last_payload["rng_state"])
        epoch_curve = copy.deepcopy(last_payload["epoch_curve"])
        batch_curve = copy.deepcopy(last_payload["batch_curve"])
        updates_total = int(last_payload["total_optimizer_updates"])
        elapsed_seconds = float(last_payload["elapsed_seconds"])
        maximum_state_abs = dict(last_payload["maximum_state_abs"])
        maximum_gradient_norm_before_clip = float(last_payload["maximum_gradient_norm_before_clip"])
        short_candidate = last_payload.get("short_candidate_payload")
        full_best = last_payload.get("full_best_payload")
        if full_best is None or (control.last_epoch >= cfg["min_epochs"] and short_candidate is None):
            raise RuntimeError("Recovery checkpoint lacks selected candidate weights")
        # Repair a crash between best-checkpoint write, last.pt, and report.
        _atomic_torch_save(full_best, full_path)
        if control.last_epoch >= cfg["min_epochs"]:
            _atomic_torch_save(short_candidate, short_path)
        if report_path.exists():
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            if prior.get("epochs_completed", 0) > control.last_epoch:
                raise RuntimeError("Fit report is ahead of recovery checkpoint")
        _atomic_json_save(_report(cfg=cfg, provenance=provenance, kind=kind, seed=seed,
                                  model=model, control=control, epoch_curve=epoch_curve,
                                  batch_curve=batch_curve, updates_total=updates_total,
                                  supervised_per_epoch=expected_supervised,
                                  elapsed_seconds=elapsed_seconds,
                                  maximum_state_abs=maximum_state_abs,
                                  maximum_gradient_norm_before_clip=maximum_gradient_norm_before_clip,
                                  checkpoint_dir=checkpoint_dir, short_candidate=short_candidate,
                                  full_best=full_best, last=last_payload), report_path)
        if control.should_stop:
            return json.loads(report_path.read_text(encoding="utf-8"))
    else:
        if report_path.exists() or any(path.exists() for path in (last_path, short_path, full_path)):
            raise FileExistsError("Existing v3.1 fit found; use --resume to continue it")

    for epoch in range(control.last_epoch + 1, cfg["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        state = None
        loss_sum = 0.0
        seen = updates = 0
        for chunk_number, raw in enumerate(packer, 1):
            batch = tensors(raw)
            if state is not None:
                state = model.detach_state(state)
            logits, state = model(batch["x"], batch["tree"], batch["reset"], state)
            for name, value in state.items():
                if value is not None:
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite carried {name} state")
                    maximum_state_abs[name] = max(maximum_state_abs[name], float(value.detach().abs().max()))
            keep = batch["valid"]
            count = int(keep.sum())
            if count:
                loss = nn.functional.binary_cross_entropy_with_logits(logits[keep], batch["y"][keep])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite train loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                maximum_gradient_norm_before_clip = max(maximum_gradient_norm_before_clip, float(norm))
                optimizer.step()
                updates += 1
                updates_total += 1
                value = float(loss.detach())
                loss_sum += value * count
                seen += count
                batch_curve.append({"epoch": epoch, "chunk": chunk_number, "update": updates_total,
                                    "n": count, "loss": value,
                                    "gradient_norm_before_clip": float(norm)})
            if chunk_number % 100 == 0:
                print(json.dumps({"model": kind, "seed": seed, "epoch": epoch,
                                  "chunk": chunk_number, "chunks": len(packer),
                                  "supervised": seen, "train_log_loss_so_far": loss_sum / max(seen, 1)}), flush=True)
        if seen != expected_supervised or updates != expected_updates:
            raise RuntimeError("Actual train label/update count differs from frozen budget")
        raw_selection = inference(model, selection, cfg)
        selection_metrics = binary_metrics(selection["y"][selection["valid"]],
                                           raw_selection[selection["valid"]])
        selection_loss = float(selection_metrics["log_loss"])
        decision = control.update(selection_loss, epoch)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_seconds += epoch_seconds
        epoch_curve.append({"epoch": epoch, "train_log_loss": loss_sum / seen,
                            "supervised": seen, "optimizer_updates": updates,
                            "selection": selection_metrics, "elapsed_seconds": epoch_seconds,
                            "control": decision})
        snapshot = _base_checkpoint(
            model=model, optimizer=optimizer, control=control, kind=kind, seed=seed,
            epoch=epoch, selection_loss=selection_loss, updates_total=updates_total,
            epoch_curve=epoch_curve, batch_curve=batch_curve, elapsed_seconds=elapsed_seconds,
            maximum_state_abs=maximum_state_abs,
            maximum_gradient_norm_before_clip=maximum_gradient_norm_before_clip,
            provenance=provenance,
        )
        if decision["save_full_best"]:
            full_best = snapshot
            _atomic_torch_save(full_best, full_path)
        if epoch <= cfg["min_epochs"] and (short_candidate is None or
                                             selection_loss < short_candidate["selection_log_loss"]):
            short_candidate = snapshot
        if decision["save_short_best"]:
            if short_candidate is None or short_candidate["epoch"] != decision["short_best_epoch"]:
                raise AssertionError("Short checkpoint candidate disagrees with early-stop controller")
            _atomic_torch_save(short_candidate, short_path)
        if full_best is None:
            raise AssertionError("First epoch must establish a full best checkpoint")
        last_payload = dict(snapshot)
        last_payload["short_candidate_payload"] = short_candidate
        last_payload["full_best_payload"] = full_best
        _atomic_torch_save(last_payload, last_path)
        report = _report(cfg=cfg, provenance=provenance, kind=kind, seed=seed,
                         model=model, control=control, epoch_curve=epoch_curve,
                         batch_curve=batch_curve, updates_total=updates_total,
                         supervised_per_epoch=expected_supervised,
                         elapsed_seconds=elapsed_seconds,
                         maximum_state_abs=maximum_state_abs,
                         maximum_gradient_norm_before_clip=maximum_gradient_norm_before_clip,
                         checkpoint_dir=checkpoint_dir, short_candidate=short_candidate,
                         full_best=full_best, last=last_payload)
        _atomic_json_save(report, report_path)
        print(json.dumps({"model": kind, "seed": seed, "epoch": epoch,
                          "train_log_loss": epoch_curve[-1]["train_log_loss"],
                          "selection_log_loss": selection_loss,
                          "updates": updates, "best_epoch": control.best_epoch,
                          "stop": decision["stop"], "stop_reason": decision["stop_reason"]}), flush=True)
        if decision["stop"] or (interrupt_after_epoch is not None and epoch >= interrupt_after_epoch):
            break
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("a", "b", "c", "no_transfer", "no_open"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/market-v31.json").read_text(encoding="utf-8"))
    provenance = get_provenance(cfg)
    report = fit_one(cfg, args.model, args.seed, provenance, resume=args.resume)
    print(json.dumps({"finished": report["complete"], "model": args.model,
                      "seed": args.seed, "epochs": report["epochs_completed"],
                      "stop_reason": report["stop_reason"]}), flush=True)


if __name__ == "__main__":
    main()
