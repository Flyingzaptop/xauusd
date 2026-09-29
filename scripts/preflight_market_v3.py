"""Throwaway synthetic speed and long-state preflight for market v3.

No market labels or checkpoints are read. These random-label updates are not
research fits and their weights are discarded after this process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crossing_memory_v2 import route_windows_v2
from src.market_data_v3 import load_split
from src.stream_batch_v3 import PackedStream
from src.stream_models_v3 import StreamForecaster, matched_baseline_widths, parameter_count


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_inputs(batch: int, length: int, seed: int, metadata_clip: float):
    """Independent labels; transformed, fixed-scale metadata like v3 inputs."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 0.65, size=(batch, length))
    z = np.cumsum(increments, axis=1)
    z -= z[:, :1]
    routing = route_windows_v2(z, [6.0, 3.0, 1.0])
    metadata = routing["metadata"].astype(np.float32, copy=True)
    for feature in (2, 5):
        values = metadata[..., feature]
        metadata[..., feature] = np.sign(values) * np.log1p(np.abs(values))
    metadata[..., 3] = np.log1p(np.maximum(metadata[..., 3], 0))
    # Fixed synthetic centering/scales; no fitted market statistics are used.
    center = np.asarray([0, 2, 0, 0.8, 0.3, 0], dtype=np.float32)
    scale = np.asarray([1, 1.5, 1.5, 1, 1, 0.8], dtype=np.float32)
    metadata = np.clip((metadata - center) / scale, -metadata_clip, metadata_clip)
    routing["metadata"] = metadata
    x = np.clip(rng.normal(0, 1, size=(batch, length, 4)), -10, 10).astype(np.float32)
    labels = rng.integers(0, 2, size=(batch, length), dtype=np.int64).astype(np.float32)
    reset = np.zeros((batch, length), dtype=np.bool_)
    reset[:, 0] = True
    return (
        torch.from_numpy(x),
        {key: torch.from_numpy(value) for key, value in routing.items()
         if key in ("metadata", "births", "closures", "active", "observer")},
        torch.from_numpy(reset),
        torch.from_numpy(labels),
    )


def check_finite_gradients(model: nn.Module) -> float:
    norms = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            norms.append(parameter.grad.square().sum())
    if not norms:
        raise RuntimeError("No gradients from random-label diagnostic loss")
    norm = torch.stack(norms).sum().sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite total gradient norm")
    return float(norm.detach())


def benchmark(cfg: dict, widths: dict, warmup: int = 2, measured: int = 5) -> dict:
    batch = cfg["batch_size"]
    length = cfg["chunk_length"]
    x, routing, reset, labels = synthetic_inputs(
        batch, length * (warmup + measured), 2026092901, cfg["metadata_clip"]
    )
    results = {}
    for variant in cfg["models"]:
        torch.manual_seed(100 + cfg["models"].index(variant))
        width = widths[variant] if variant in ("a", "b") else cfg["c_gru_hidden"]
        model = StreamForecaster(variant, width, cfg["memory_hidden"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        state = model.initial_state(batch, x.device, x.dtype)
        times = []
        grad_norms = []
        carried = []
        for chunk in range(warmup + measured):
            start, stop = chunk * length, (chunk + 1) * length
            if chunk:
                state = model.detach_state(state)
                carried.append(float(state["gru"].abs().max().detach()))
            route_chunk = {key: value[:, start:stop] for key, value in routing.items()}
            optimizer.zero_grad(set_to_none=True)
            began = time.perf_counter()
            logits, state = model(x[:, start:stop], None if variant == "a" else route_chunk,
                                  reset[:, start:stop], state)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[:, start:stop])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite diagnostic loss: {variant}/{chunk}")
            loss.backward()
            gradient_norm = check_finite_gradients(model)
            clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            if not torch.isfinite(clipped_norm):
                raise FloatingPointError(f"Nonfinite clipped norm: {variant}/{chunk}")
            optimizer.step()
            elapsed = time.perf_counter() - began
            if chunk >= warmup:
                times.append(elapsed)
                grad_norms.append(gradient_norm)
        model.eval()
        eval_state = model.initial_state(batch, x.device, x.dtype)
        eval_times = []
        with torch.no_grad():
            for chunk in range(warmup + measured):
                start, stop = chunk * length, (chunk + 1) * length
                route_chunk = {key: value[:, start:stop] for key, value in routing.items()}
                began = time.perf_counter()
                logits, eval_state = model(
                    x[:, start:stop], None if variant == "a" else route_chunk,
                    reset[:, start:stop], eval_state,
                )
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"Nonfinite evaluation logits: {variant}/{chunk}")
                if chunk >= warmup:
                    eval_times.append(time.perf_counter() - began)
        result = {
            "parameters": parameter_count(model),
            "warmup_chunks": warmup,
            "measured_chunks": measured,
            "batch_size": batch,
            "chunk_length": length,
            "seconds_per_chunk": times,
            "mean_seconds_per_chunk": statistics.mean(times),
            "median_seconds_per_chunk": statistics.median(times),
            "observations_per_second": batch * length / statistics.mean(times),
            "eval_forward_only_seconds_per_chunk": eval_times,
            "eval_forward_only_median_seconds_per_chunk": statistics.median(eval_times),
            "preclip_gradient_norm_range": [min(grad_norms), max(grad_norms)],
            "state_nonzero_on_all_chunk_boundaries": all(value > 0 for value in carried),
        }
        results[variant] = result
        print(f"benchmark {variant}: {result['mean_seconds_per_chunk']:.3f}s/chunk, "
              f"{result['observations_per_second']:.0f} observations/s", flush=True)
    return results


def long_stability(cfg: dict, ticks: int = 16384) -> dict:
    """One uninterrupted stream, 128-tick TBPTT, fixed random C weights."""
    length = cfg["chunk_length"]
    if ticks < 16384 or ticks % length:
        raise ValueError("long stability must cover >=16384 ticks in complete chunks")
    x, routing, reset, labels = synthetic_inputs(1, ticks, 2026092902, cfg["metadata_clip"])
    torch.manual_seed(20260929)
    model = StreamForecaster("c", cfg["c_gru_hidden"], cfg["memory_hidden"])
    state = model.initial_state(1, x.device, x.dtype)
    max_gru = 0.0
    max_tree = 0.0
    max_grad = 0.0
    min_grad = float("inf")
    carried = 0
    times = []
    for chunk, start in enumerate(range(0, ticks, length)):
        stop = start + length
        if chunk:
            state = model.detach_state(state)
            if float(state["gru"].abs().max().detach()) > 0:
                carried += 1
        model.zero_grad(set_to_none=True)
        began = time.perf_counter()
        logits, state = model(x[:, start:stop],
                              {key: value[:, start:stop] for key, value in routing.items()},
                              reset[:, start:stop], state)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[:, start:stop])
        if not torch.isfinite(loss) or not torch.isfinite(state["gru"]).all() or not torch.isfinite(state["tree"]).all():
            raise FloatingPointError(f"Nonfinite loss or hidden state at long chunk {chunk}")
        loss.backward()
        grad_norm = check_finite_gradients(model)
        max_grad = max(max_grad, grad_norm)
        min_grad = min(min_grad, grad_norm)
        max_gru = max(max_gru, float(state["gru"].abs().max().detach()))
        max_tree = max(max_tree, float(state["tree"].abs().max().detach()))
        times.append(time.perf_counter() - began)
        if (chunk + 1) % 16 == 0:
            print(f"long C: {stop}/{ticks} ticks, max hidden {max(max_gru,max_tree):.3f}", flush=True)
    if carried != ticks // length - 1:
        raise RuntimeError("Continuous state was zero at a TBPTT boundary")
    return {
        "variant": "c",
        "ticks": ticks,
        "batch_size": 1,
        "chunk_length": length,
        "chunks": ticks // length,
        "real_gap_resets": 0,
        "first_tick_reset_only": True,
        "optimizer_steps": 0,
        "weights_fixed": True,
        "carried_nonzero_boundaries": carried,
        "max_abs_gru_hidden": max_gru,
        "max_abs_tree_hidden": max_tree,
        "log10_1p_max_abs_hidden": float(np.log10(1 + max(max_gru, max_tree))),
        "preclip_gradient_norm_range": [min_grad, max_grad],
        "all_states_and_gradients_finite": True,
        "mean_seconds_per_chunk": statistics.mean(times),
        "total_seconds": sum(times),
    }


def real_train_stability(cfg: dict, ticks: int = 16384) -> tuple[dict, dict]:
    """Use normalized real train inputs and resets with independent dummy labels.

    This never reads evaluation data or uses actual train labels in the loss.
    It does not update weights. Actual cache arrays are loaded to derive exact
    deterministic packer chunk counts for the frozen resource estimate.
    """
    cache_dir = ROOT / cfg["output"]
    split = load_split(cache_dir, "train")
    if len(split["x"]) < ticks:
        raise RuntimeError("Prepared train cache shorter than requested stability stream")
    packed = PackedStream(split, cfg["batch_size"], cfg["chunk_length"])
    x = torch.from_numpy(np.asarray(split["x"][:ticks], dtype=np.float32))[None]
    routing = {
        key: torch.from_numpy(np.asarray(value[:ticks]))[None]
        for key, value in split["tree"].items()
        if key in ("metadata", "births", "closures", "active", "observer")
    }
    reset = torch.from_numpy(np.asarray(split["reset"][:ticks]))[None]
    rng = np.random.default_rng(2026092903)
    labels = torch.from_numpy(rng.integers(0, 2, size=(1, ticks)).astype(np.float32))
    length = cfg["chunk_length"]
    torch.manual_seed(20260930)
    model = StreamForecaster("c", cfg["c_gru_hidden"], cfg["memory_hidden"])
    state = model.initial_state(1, x.device, x.dtype)
    max_gru = 0.0
    max_tree = 0.0
    gradient_norms = []
    carried_nonzero = 0
    times = []
    for chunk, start in enumerate(range(0, ticks, length)):
        stop = min(start + length, ticks)
        if chunk:
            state = model.detach_state(state)
            carried_nonzero += int(float(state["gru"].abs().max().detach()) > 0)
        model.zero_grad(set_to_none=True)
        began = time.perf_counter()
        logits, state = model(x[:, start:stop],
                              {key: value[:, start:stop] for key, value in routing.items()},
                              reset[:, start:stop], state)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[:, start:stop])
        if not torch.isfinite(loss) or not torch.isfinite(state["gru"]).all() or not torch.isfinite(state["tree"]).all():
            raise FloatingPointError(f"Nonfinite real-train-input state/loss at chunk {chunk}")
        loss.backward()
        gradient_norms.append(check_finite_gradients(model))
        max_gru = max(max_gru, float(state["gru"].abs().max().detach()))
        max_tree = max(max_tree, float(state["tree"].abs().max().detach()))
        times.append(time.perf_counter() - began)
        if (chunk + 1) % 32 == 0:
            print(f"real train C: {stop}/{ticks} rows, max hidden {max(max_gru,max_tree):.3f}", flush=True)
    report = {
        "source": "prepared train cache only",
        "cache_train_sha256": sha256(cache_dir / "train.npz"),
        "rows": ticks,
        "first_timestamp_utc_seconds": int(split["timestamp"][0]),
        "last_timestamp_utc_seconds": int(split["timestamp"][ticks - 1]),
        "actual_reset_count": int(reset.sum()),
        "dummy_labels": "independent Bernoulli(0.5); stored train labels unused",
        "optimizer_steps": 0,
        "weights_fixed": True,
        "chunks": len(times),
        "carried_nonzero_boundaries": carried_nonzero,
        "max_abs_gru_hidden": max_gru,
        "max_abs_tree_hidden": max_tree,
        "log10_1p_max_abs_hidden": float(np.log10(1 + max(max_gru, max_tree))),
        "preclip_gradient_norm_range": [min(gradient_norms), max(gradient_norms)],
        "all_states_and_gradients_finite": True,
        "mean_seconds_per_chunk": statistics.mean(times),
        "total_seconds": sum(times),
    }
    packer = {
        "train_rows": packed.n_observations,
        "train_valid_labels": packed.n_valid_labels,
        "train_segments": len(packed.segments),
        "train_chunks_per_epoch": len(packed),
        "epochs": cfg["epochs"],
        "seeds": len(cfg["seeds"]),
        "models": len(cfg["models"]),
        "total_expected_optimizer_updates": len(packed) * cfg["epochs"] * len(cfg["seeds"]) * len(cfg["models"]),
    }
    return report, packer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market-v3.json")
    parser.add_argument("--output", default="reports/v3/speed-stability.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["batch_size"] != 16 or cfg["chunk_length"] != 128:
        raise ValueError("This preflight requires the frozen B16,T128 configuration")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    widths = matched_baseline_widths(cfg["c_gru_hidden"], cfg["memory_hidden"])
    start = time.perf_counter()
    speed = benchmark(cfg, widths)
    stability = long_stability(cfg)
    actual_stability, packer = real_train_stability(cfg)
    rough_train_seconds = {
        variant: result["mean_seconds_per_chunk"] * packer["train_chunks_per_epoch"]
                 * cfg["epochs"] * len(cfg["seeds"])
        for variant, result in speed.items()
    }
    report = {
        "kind": "throwaway_synthetic_preflight_not_market_training",
        "config_sha256": sha256(config_path),
        "source_sha256": {
            "stream_models_v3": sha256(ROOT / "src/stream_models_v3.py"),
            "crossing_memory_v2": sha256(ROOT / "src/crossing_memory_v2.py"),
            "preflight_script": sha256(Path(__file__)),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "synthetic_input": {
            "log_close_step_std": 0.65,
            "thresholds": [6, 3, 1],
            "metadata_signed_log1p_channels": [2, 5],
            "metadata_log1p_channels": [3],
            "metadata_center": [0, 2, 0, 0.8, 0.3, 0],
            "metadata_scale": [1, 1.5, 1.5, 1, 1, 0.8],
            "metadata_clip": cfg["metadata_clip"],
            "labels": "independent Bernoulli(0.5)",
        },
        "matched_widths": widths,
        "benchmark": speed,
        "long_stability": stability,
        "real_train_input_stability": actual_stability,
        "train_packer": packer,
        "rough_serial_train_seconds_by_variant_from_synthetic_timing": rough_train_seconds,
        "rough_serial_train_seconds_all_variants": sum(rough_train_seconds.values()),
        "wall_seconds": time.perf_counter() - start,
        "limitation": "Synthetic routing and fixed-scale normalization measure throughput and numeric stability only. Timing may include contention from other local work; no market accuracy is measured.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
