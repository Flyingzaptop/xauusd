"""Audit identifiability of the frozen v1 synthetic target without training.

The decoder reads only rounded log-close increments. It neither reads latent
generator choices nor calls the crossing router. Exact grammar enumeration also
computes the Bayes error under the v1 generator's multinomial gap distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.synthetic_checks import make_dataset  # noqa: E402

BASE_STEP = 0.0025
NEG, POS = 0, 1
PAIR = (bytes((NEG, POS)), bytes((POS, NEG)))
MASK32 = (1 << 32) - 1


def episode(sign: int, order: int) -> bytes:
    def run(direction: int, length: int) -> bytes:
        return bytes((POS if direction > 0 else NEG,)) * length

    return (
        run(sign, 7)
        + run(sign * order, 3)
        + run(-sign * order, 6)
        + run(sign * order, 3)
        + run(-sign, 7)
    )


EPISODES = {(sign, order): episode(sign, order) for sign in (-1, 1) for order in (-1, 1)}
EPISODE_LOOKUP = {value: key for key, value in EPISODES.items()}
OPEN_LOOKUP = {value[:19]: key for key, value in EPISODES.items()}


def close_to_steps(close: np.ndarray, log_step: float = BASE_STEP) -> tuple[bytes, float]:
    """Decode observed prices; reject increments that are not close to +/-1."""
    if close.ndim != 1 or len(close) < 2 or np.any(close <= 0):
        raise ValueError("Expected a one-dimensional positive close path")
    increments = np.diff(np.log(close.astype(np.float64))) / log_step
    rounded = np.rint(increments).astype(np.int8)
    error = float(np.max(np.abs(increments - rounded)))
    if error > 1e-3 or np.any(np.abs(rounded) != 1):
        raise ValueError(f"Close path does not match +/-1-step alphabet: error={error}")
    return bytes((POS if step > 0 else NEG) for step in rounded), error


def is_padding(chunk: bytes) -> bool:
    return len(chunk) % 2 == 0 and all(chunk[i] != chunk[i + 1] for i in range(0, len(chunk), 2))


def parse_full(steps: bytes) -> list[tuple[int, int, int, int, int]]:
    """Return all (p0,p1,p2,label,open_cut_index) grammar parses."""
    if len(steps) != 72:
        return []
    matches = []
    for p0 in range(11):
        start1 = 2 * p0
        if not is_padding(steps[:start1]):
            continue
        if steps[start1 : start1 + 26] not in EPISODE_LOOKUP:
            continue
        for p1 in range(11 - p0):
            p2 = 10 - p0 - p1
            start2 = start1 + 26 + 2 * p1
            if not is_padding(steps[start1 + 26 : start2]):
                continue
            signature = EPISODE_LOOKUP.get(steps[start2 : start2 + 26])
            if signature is None or not is_padding(steps[start2 + 26 :]):
                continue
            if len(steps[start2 + 26 :]) != 2 * p2:
                continue
            label = int(signature[1] == 1)
            matches.append((p0, p1, p2, label, start2 + 19))
    return matches


def parse_open(steps: bytes) -> list[tuple[int, int, int]]:
    """Return all (p0,p1,label) parses of a prefix before final outer return."""
    if len(steps) < 45 or len(steps) > 65 or (len(steps) - 45) % 2:
        return []
    pairs = (len(steps) - 45) // 2
    matches = []
    for p0 in range(pairs + 1):
        p1 = pairs - p0
        start1 = 2 * p0
        start2 = start1 + 26 + 2 * p1
        if not is_padding(steps[:start1]) or not is_padding(steps[start1 + 26 : start2]):
            continue
        if steps[start1 : start1 + 26] not in EPISODE_LOOKUP:
            continue
        signature = OPEN_LOOKUP.get(steps[start2:])
        if signature is not None:
            matches.append((p0, p1, int(signature[1] == 1)))
    return matches


def _mask(parses: list[tuple[int, ...]], label_index: int) -> int:
    return sum(1 << label for label in set(row[label_index] for row in parses))


def sample_audit(seed_count: int, paths_per_seed: int) -> dict:
    full_counter: Counter[str] = Counter()
    open_counter: Counter[str] = Counter()
    max_rounding_error = 0.0
    examples = []
    sampled = 0
    for seed in range(seed_count):
        close, labels = make_dataset(paths_per_seed, 90_000 + seed, BASE_STEP)
        for path, label in zip(close, labels):
            steps, error = close_to_steps(path)
            max_rounding_error = max(max_rounding_error, error)
            full = parse_full(steps)
            mask = _mask(full, 3)
            full_counter[{0: "unparseable", 1: "label_0_only", 2: "label_1_only", 3: "ambiguous"}[mask]] += 1
            if not mask & (1 << int(label)):
                full_counter["excludes_generator_label"] += 1
            if full:
                cuts = {row[4] for row in full}
                if len(cuts) != 1:
                    open_counter["multiple_cut_positions"] += 1
                # Every full parse has a corresponding open prefix. Keep all
                # cut candidates to avoid silently using latent gap positions.
                for cut in cuts:
                    open_matches = parse_open(steps[:cut])
                    open_mask = _mask(open_matches, 2)
                    open_counter[{0: "unparseable", 1: "label_0_only", 2: "label_1_only", 3: "ambiguous"}[open_mask]] += 1
                    if not open_mask & (1 << int(label)):
                        open_counter["excludes_generator_label"] += 1
            if (mask == 3 or not mask & (1 << int(label))) and len(examples) < 5:
                examples.append({"seed": seed, "label": int(label), "step_bits": steps.hex(),
                                 "parses": full[:10]})
            sampled += 1
    return {"seed_count": seed_count, "paths_per_seed": paths_per_seed,
            "paths": sampled, "full": dict(full_counter), "open": dict(open_counter),
            "maximum_rounding_error_steps": max_rounding_error, "examples": examples}


def original_run_audit() -> dict:
    """Replay the exact six datasets listed in the frozen v1 metrics file."""
    original = json.loads((ROOT / "reports" / "synthetic_metrics.json").read_text(encoding="utf-8"))
    config = original["config"]
    rows = []
    for record in original["dataset"]:
        seed = int(record["seed"])
        for split, size, data_seed in (("train", int(config["train_size"]), seed * 100 + 1),
                                       ("test", int(config["test_size"]), seed * 100 + 2)):
            close, labels = make_dataset(size, data_seed, float(config["base_log_step"]))
            digest = hashlib.sha256(close.tobytes() + labels.tobytes()).hexdigest()
            if digest != record[f"{split}_sha256"]:
                raise AssertionError(f"Frozen v1 {split} data hash mismatch, seed={seed}")
            recovered = 0
            for path, label in zip(close, labels):
                steps, _ = close_to_steps(path)
                parses = parse_full(steps)
                recovered += int(_mask(parses, 3) == (1 << int(label)))
            rows.append({"seed": seed, "split": split, "paths": size,
                         "sha256": digest, "unique_correct_decodes": recovered})
    return {"datasets": rows, "total_paths": sum(row["paths"] for row in rows),
            "total_unique_correct_decodes": sum(row["unique_correct_decodes"] for row in rows)}


def _add_mass(table: dict[bytes, int], path: bytes, label: int, weight: int) -> None:
    table[path] = table.get(path, 0) + (weight << (32 * label))


def _summarize_mass(table: dict[bytes, int]) -> dict:
    ambiguous = 0
    mass0 = mass1 = error_mass = 0
    for value in table.values():
        weight0 = value & MASK32
        weight1 = value >> 32
        mass0 += weight0
        mass1 += weight1
        error_mass += min(weight0, weight1)
        ambiguous += int(weight0 > 0 and weight1 > 0)
    total = mass0 + mass1
    return {"distinct_paths": len(table), "ambiguous_label_paths": ambiguous,
            "weighted_label_0_mass": mass0, "weighted_label_1_mass": mass1,
            "weighted_total_mass": total, "bayes_error": error_mass / total,
            "bayes_accuracy": 1 - error_mass / total}


def exhaustive_audit() -> dict:
    """Enumerate all 16 episode choices, 66 gap splits, 1024 padding bits."""
    full_mass: dict[bytes, int] = {}
    open_mass: dict[bytes, int] = {}
    derivations = 0
    for bits in range(1 << 10):
        loops = tuple(PAIR[(bits >> i) & 1] for i in range(10))
        for p0 in range(11):
            left = b"".join(loops[:p0])
            for p1 in range(11 - p0):
                p2 = 10 - p0 - p1
                middle = b"".join(loops[p0 : p0 + p1])
                right = b"".join(loops[p0 + p1 :])
                weight = math.comb(10, p0) * math.comb(10 - p0, p1)
                for e1 in EPISODES.values():
                    prefix = left + e1 + middle
                    for (sign2, order2), e2 in EPISODES.items():
                        del sign2
                        label = int(order2 == 1)
                        _add_mass(full_mass, prefix + e2 + right, label, weight)
                        # The original generator still draws trailing loops,
                        # which are unobserved after cutting this prefix.
                        _add_mass(open_mass, prefix + e2[:19], label, weight)
                        derivations += 1
    expected_mass = (3**10) * (2**10) * 16
    full = _summarize_mass(full_mass)
    opened = _summarize_mass(open_mass)
    if full["weighted_total_mass"] != expected_mass or opened["weighted_total_mass"] != expected_mass:
        raise AssertionError("Exhaustive grammar probability mass is inconsistent")
    return {"derivations": derivations, "expected_derivations": 16 * math.comb(12, 2) * (2**10),
            "weighted_expected_mass": expected_mass, "full": full, "open_prefix": opened}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--paths-per-seed", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "v2" / "v1-diagnosis.json")
    args = parser.parse_args()
    sampled = sample_audit(args.seed_count, args.paths_per_seed)
    original_run = original_run_audit()
    exact = exhaustive_audit()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "v1_generator_sha256": sha(ROOT / "scripts" / "synthetic_checks.py"),
        "diagnostic_sha256": sha(Path(__file__)),
        "method": "independent integer-log-increment parser; exact derivation enumeration",
        "sampled": sampled,
        "original_run": original_run,
        "exhaustive": exact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sampled": sampled, "original_run": original_run, "exhaustive": exact}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
