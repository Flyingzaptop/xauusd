"""Reproducible nested-history learnability and random-label controls.

Run from the repository root after ``src/crossing_memory.py`` is available::

    python scripts/synthetic_checks.py

The target is the order of two fine excursions inside the *last* of two
completed coarse excursions. Both classes have the same start, finish, total
variation, number of positive/negative steps, and excursion sizes. A random
first excursion and variable neutral padding prevent a fixed-index shortcut.
Labels are assigned by the path generator, independent of the routing code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crossing_memory import CrossingMemory, route_windows  # noqa: E402


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (7, 19, 43)
    train_size: int = 320
    test_size: int = 160
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.003
    base_log_step: float = 0.0025
    threshold_bins: tuple[int, ...] = (1, 3, 6)
    hidden_dim: int = 12
    max_seconds: float = 600.0


def _episode(sign: int, order: int) -> list[int]:
    """Closed coarse excursion with two ordered fine excursions inside it."""
    return (
        [sign] * 7
        + [sign * order] * 3
        + [-sign * order] * 6
        + [sign * order] * 3
        + [-sign] * 7
    )


def make_dataset(size: int, seed: int, base_log_step: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = np.empty((size, 73), dtype=np.float32)
    labels = np.empty(size, dtype=np.int64)
    for i in range(size):
        label = int(rng.integers(0, 2))
        signs = rng.choice(np.array([-1, 1]), size=2)
        distractor_order = int(rng.choice(np.array([-1, 1])))
        target_order = 1 if label else -1
        # Ten zero-net two-step loops, split across the three gaps. Their
        # random locations obscure the start of the relevant excursion.
        gaps = rng.multinomial(10, [1 / 3] * 3)

        def padding(pairs: int) -> list[int]:
            steps: list[int] = []
            for _ in range(int(pairs)):
                direction = int(rng.choice(np.array([-1, 1])))
                steps.extend((direction, -direction))
            return steps

        steps = (
            padding(gaps[0])
            + _episode(int(signs[0]), distractor_order)
            + padding(gaps[1])
            + _episode(int(signs[1]), target_order)
            + padding(gaps[2])
        )
        assert len(steps) == 72 and sum(steps) == 0
        log_close = np.concatenate(([0.0], np.cumsum(steps, dtype=np.float64)))
        close[i] = (100.0 * np.exp(log_close * base_log_step)).astype(np.float32)
        labels[i] = label
    return close, labels


def make_features(close: np.ndarray, base_log_step: float) -> torch.Tensor:
    log_close = np.log(close.astype(np.float64))
    level = (log_close - log_close[:, :1]) / (8 * base_log_step)
    returns = np.diff(log_close, axis=1, prepend=log_close[:, :1]) / base_log_step
    return torch.from_numpy(np.stack((returns, level), axis=-1).astype(np.float32))


class RoutedClassifier(nn.Module):
    def __init__(self, config: Config, *, transfer: bool, read_open: bool):
        super().__init__()
        self.memory = CrossingMemory(
            input_dim=2,
            hidden_dim=config.hidden_dim,
            scales=len(config.threshold_bins),
            transfer=transfer,
            read_open=read_open,
        )
        self.head = nn.Sequential(nn.Linear(self.memory.output_dim, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor, stats: torch.Tensor, closures: torch.Tensor) -> torch.Tensor:
        return self.head(self.memory(x, stats, closures)[:, -1])


class GRUClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        width = config.hidden_dim * len(config.threshold_bins)
        self.gru = nn.GRU(2, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor, stats: torch.Tensor, closures: torch.Tensor) -> torch.Tensor:
        del stats, closures
        return self.head(self.gru(x)[0][:, -1])


def _routed(close: np.ndarray, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    thresholds = [config.base_log_step * k for k in config.threshold_bins]
    stats, closures = route_windows(close, thresholds)
    expected = (len(close), close.shape[1], len(thresholds))
    if stats.shape != (*expected, 6) or closures.shape != expected:
        raise ValueError(f"Unexpected routing shape: {stats.shape}, {closures.shape}")
    return torch.as_tensor(stats, dtype=torch.float32), torch.as_tensor(closures, dtype=torch.bool)


def _evaluate(model: nn.Module, tensors: tuple[torch.Tensor, ...], batch_size: int) -> dict[str, float]:
    x, stats, closures, y = tensors
    model.eval()
    correct = total_loss = 0.0
    by_class = [[0, 0], [0, 0]]
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            sl = slice(start, start + batch_size)
            logits = model(x[sl], stats[sl], closures[sl])
            loss = F.cross_entropy(logits, y[sl], reduction="sum")
            prediction = logits.argmax(-1)
            correct += float((prediction == y[sl]).sum())
            total_loss += float(loss)
            for class_id in (0, 1):
                mask = y[sl] == class_id
                by_class[class_id][0] += int((prediction[mask] == class_id).sum())
                by_class[class_id][1] += int(mask.sum())
    return {
        "accuracy": correct / len(y),
        "balanced_accuracy": sum(hit / count for hit, count in by_class) / 2,
        "loss": total_loss / len(y),
    }


def _train_one(
    variant: str,
    seed: int,
    config: Config,
    train: tuple[torch.Tensor, ...],
    test: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if variant == "full":
        model = RoutedClassifier(config, transfer=True, read_open=True)
    elif variant == "no_transfer":
        model = RoutedClassifier(config, transfer=False, read_open=True)
    elif variant == "no_open":
        model = RoutedClassifier(config, transfer=True, read_open=False)
    elif variant == "gru":
        model = GRUClassifier(config)
    else:
        raise ValueError(variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x, stats, closures, y = train
    shuffle = torch.Generator().manual_seed(seed + 10_000)
    curve = []
    max_gradient = 0.0
    for epoch in range(config.epochs):
        model.train()
        for indexes in torch.randperm(len(y), generator=shuffle).split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indexes], stats[indexes], closures[indexes])
            loss = F.cross_entropy(logits, y[indexes])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {variant} seed={seed} epoch={epoch}")
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient: {variant} seed={seed} epoch={epoch}")
                    max_gradient = max(max_gradient, float(p.grad.abs().max()))
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        curve.append({"epoch": epoch + 1, "train": _evaluate(model, train, config.batch_size)})
    return {
        "variant": variant,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "max_abs_gradient": max_gradient,
        "curve": curve,
        "final_train": curve[-1]["train"],
        "final_test": _evaluate(model, test, config.batch_size),
    }


def _git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    code_sha256 = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "scripts" / "synthetic_checks.py", ROOT / "src" / "crossing_memory.py")
    }
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short"),
            "code_sha256": code_sha256}


def _digest(close: np.ndarray, labels: np.ndarray) -> str:
    return hashlib.sha256(close.tobytes() + labels.tobytes()).hexdigest()


def run(config: Config) -> dict[str, Any]:
    torch.set_num_threads(1)
    begun = time.monotonic()
    results: list[dict[str, Any]] = []
    dataset_records = []
    for seed in config.seeds:
        train_close, train_y = make_dataset(config.train_size, seed * 100 + 1, config.base_log_step)
        test_close, test_y = make_dataset(config.test_size, seed * 100 + 2, config.base_log_step)
        if np.array_equal(train_close[: min(len(train_close), len(test_close))], test_close[: min(len(train_close), len(test_close))]):
            raise AssertionError("Train and test paths unexpectedly coincide")
        train = (make_features(train_close, config.base_log_step), *_routed(train_close, config), torch.from_numpy(train_y))
        test = (make_features(test_close, config.base_log_step), *_routed(test_close, config), torch.from_numpy(test_y))
        dataset_records.append({
            "seed": seed,
            "train_sha256": _digest(train_close, train_y),
            "test_sha256": _digest(test_close, test_y),
            "train_positive_rate": float(train_y.mean()),
            "test_positive_rate": float(test_y.mean()),
        })
        for variant in ("full", "no_transfer", "no_open", "gru"):
            if time.monotonic() - begun > config.max_seconds:
                raise TimeoutError(f"Time budget exhausted before {variant} seed={seed}; increase --max-seconds")
            result = _train_one(variant, seed, config, train, test)
            results.append({"task": "ordered_nested_excursions", **result})
            print(f"seed={seed} {variant}: train={result['final_train']['balanced_accuracy']:.3f} "
                  f"test={result['final_test']['balanced_accuracy']:.3f}", flush=True)

        # Independent labels: generated after the paths, from distinct streams.
        # The same architecture and schedule must not generalize these labels.
        random_train_y = torch.from_numpy(np.random.default_rng(seed * 100 + 51).integers(0, 2, len(train_y), dtype=np.int64))
        random_test_y = torch.from_numpy(np.random.default_rng(seed * 100 + 52).integers(0, 2, len(test_y), dtype=np.int64))
        random_result = _train_one("full", seed + 50_000, config,
                                   (*train[:3], random_train_y), (*test[:3], random_test_y))
        results.append({"task": "independent_random_labels", **random_result})
        dataset_records[-1]["random_train_positive_rate"] = float(random_train_y.float().mean())
        dataset_records[-1]["random_test_positive_rate"] = float(random_test_y.float().mean())
        print(f"seed={seed} random-label control: train={random_result['final_train']['balanced_accuracy']:.3f} "
              f"test={random_result['final_test']['balanced_accuracy']:.3f}", flush=True)

    summary: dict[str, Any] = {}
    for task in ("ordered_nested_excursions", "independent_random_labels"):
        for variant in ("full", "no_transfer", "no_open", "gru"):
            vals = [r["final_test"]["balanced_accuracy"] for r in results
                    if r["task"] == task and r["variant"] == variant]
            if vals:
                summary[f"{task}/{variant}"] = {"mean_balanced_accuracy": float(np.mean(vals)),
                                                "std_balanced_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                                "per_seed": vals}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "config": asdict(config),
        "dataset": dataset_records,
        "results": results,
        "summary": summary,
        "elapsed_seconds": time.monotonic() - begun,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "synthetic_metrics.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=320)
    parser.add_argument("--test-size", type=int, default=160)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    args = parser.parse_args()
    config = Config(seeds=tuple(args.seeds), epochs=args.epochs, train_size=args.train_size,
                    test_size=args.test_size, max_seconds=args.max_seconds)
    report = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
