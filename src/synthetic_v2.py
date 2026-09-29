"""Frozen-mechanism synthetic datasets and an independent path oracle.

All paths are integer log-price positions. `oracle_labels` infers labels from
observed increments by parsing the grammar; it never uses generator choices.
The hard tasks reuse v1's unchanged price generator and its original labels.
No model training or test-based selection occurs in this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection

import numpy as np

LOG_STEP = 0.0025
TASKS = ("tiny", "easy_closed", "medium_closed", "hard_closed", "hard_open")
LENGTHS = {"tiny": 73, "easy_closed": 39, "medium_closed": 61,
           "hard_closed": 73, "hard_open": 69}
PADDING_PAIRS = {"easy_closed": 6, "medium_closed": 4,
                 "hard_closed": 10, "hard_open": 10, "tiny": 10}


def _generated_episode(sign: int, order: int) -> list[int]:
    return ([sign] * 7 + [sign * order] * 3 + [-sign * order] * 6
            + [sign * order] * 3 + [-sign] * 7)


def _padding(rng: np.random.Generator, pairs: int) -> list[int]:
    directions = rng.choice(np.array([-1, 1]), size=pairs)
    return [step for direction in directions for step in (int(direction), -int(direction))]


def _generate_new(task: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    label = int(rng.integers(0, 2))
    order = 1 if label else -1
    if task == "easy_closed":
        sign = int(rng.choice(np.array([-1, 1])))
        before = int(rng.integers(0, 7))
        steps = (_padding(rng, before) + _generated_episode(sign, order)
                 + _padding(rng, 6 - before))
    elif task == "medium_closed":
        signs = rng.choice(np.array([-1, 1]), size=2)
        distractor_order = int(rng.choice(np.array([-1, 1])))
        gaps = rng.multinomial(4, [1 / 3] * 3)
        steps = (_padding(rng, int(gaps[0]))
                 + _generated_episode(int(signs[0]), distractor_order)
                 + _padding(rng, int(gaps[1]))
                 + _generated_episode(int(signs[1]), order)
                 + _padding(rng, int(gaps[2])))
    else:
        raise ValueError(task)
    z = np.concatenate(([0], np.cumsum(steps, dtype=np.int16)))
    return z.astype(np.int16), label


def _v1_close_to_z(close: np.ndarray) -> np.ndarray:
    log_relative = np.log(close.astype(np.float64)) - np.log(float(close[0]))
    scaled = log_relative / LOG_STEP
    rounded = np.rint(scaled)
    if np.max(np.abs(scaled - rounded)) > 1e-3:
        raise ValueError("v1 close cannot be decoded onto the integer log grid")
    z = rounded.astype(np.int16)
    if np.any(np.abs(np.diff(z)) != 1):
        raise ValueError("v1 close has an increment outside the +/-1 grammar")
    return z


def _valid_pairs(steps: np.ndarray) -> bool:
    return len(steps) % 2 == 0 and bool(np.all(steps[::2] == -steps[1::2]))


def _parse_episode(steps: np.ndarray, opened: bool) -> int | None:
    """Read run relationships directly, separate from generator construction."""
    if len(steps) != (22 if opened else 26):
        return None
    sign = int(steps[0])
    inner = int(steps[7])
    if sign not in (-1, 1) or inner not in (-1, 1):
        return None
    if not (np.all(steps[:7] == sign) and np.all(steps[7:10] == inner)
            and np.all(steps[10:16] == -inner) and np.all(steps[16:19] == inner)):
        return None
    if not np.all(steps[19:22] == -sign):
        return None
    if not opened and not np.all(steps[22:26] == -sign):
        return None
    return int(inner == sign)


def _labels_from_steps(task: str, steps: np.ndarray) -> set[int]:
    matches: set[int] = set()
    if task == "easy_closed":
        if len(steps) != 38:
            return matches
        for before in range(7):
            start = 2 * before
            if _valid_pairs(steps[:start]) and _valid_pairs(steps[start + 26:]):
                label = _parse_episode(steps[start:start + 26], opened=False)
                if label is not None:
                    matches.add(label)
        return matches

    if task in ("medium_closed", "hard_closed", "tiny"):
        pairs = PADDING_PAIRS[task]
        if len(steps) != 52 + 2 * pairs:
            return matches
        for p0 in range(pairs + 1):
            start1 = 2 * p0
            if not _valid_pairs(steps[:start1]):
                continue
            if _parse_episode(steps[start1:start1 + 26], opened=False) is None:
                continue
            for p1 in range(pairs - p0 + 1):
                start2 = start1 + 26 + 2 * p1
                if not (_valid_pairs(steps[start1 + 26:start2])
                        and _valid_pairs(steps[start2 + 26:])):
                    continue
                label = _parse_episode(steps[start2:start2 + 26], opened=False)
                if label is not None:
                    matches.add(label)
        return matches

    if task == "hard_open":
        if len(steps) < 48 or len(steps) > 68 or (len(steps) - 48) % 2:
            return matches
        pairs = (len(steps) - 48) // 2
        for p0 in range(pairs + 1):
            start1 = 2 * p0
            start2 = start1 + 26 + 2 * (pairs - p0)
            if not (_valid_pairs(steps[:start1]) and
                    _valid_pairs(steps[start1 + 26:start2])):
                continue
            if _parse_episode(steps[start1:start1 + 26], opened=False) is None:
                continue
            label = _parse_episode(steps[start2:], opened=True)
            if label is not None:
                matches.add(label)
        return matches
    raise ValueError(f"Unknown task: {task}")


def oracle_labels(task: str, z: np.ndarray) -> np.ndarray:
    """Decode labels solely from observed integer positions [N,T]."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    paths = np.asarray(z)
    if paths.ndim != 2 or paths.shape[1] != LENGTHS[task]:
        raise ValueError(f"Expected z [N,{LENGTHS[task]}] for {task}")
    result = np.empty(len(paths), dtype=np.int64)
    for i, path in enumerate(paths):
        if not np.issubdtype(path.dtype, np.integer):
            raise ValueError("z must use integer positions")
        if task == "hard_open":
            # The only artificial observations are leading repeated closes.
            first_change = int(np.flatnonzero(path != path[0])[0]) if np.any(path != path[0]) else len(path)
            path = path[first_change - 1:]
        steps = np.diff(path)
        labels = _labels_from_steps(task, steps)
        if len(labels) != 1:
            raise ValueError(f"Path {i} has {len(labels)} possible oracle labels: {labels}")
        result[i] = labels.pop()
    return result


def _open_prefix(z: np.ndarray) -> np.ndarray:
    """Keep three outer-return steps, stop while the outer event remains open."""
    steps = np.diff(z)
    cuts = set()
    for p0 in range(11):
        start1 = 2 * p0
        if not _valid_pairs(steps[:start1]):
            continue
        if _parse_episode(steps[start1:start1 + 26], opened=False) is None:
            continue
        for p1 in range(11 - p0):
            start2 = start1 + 26 + 2 * p1
            if (_valid_pairs(steps[start1 + 26:start2])
                    and _parse_episode(steps[start2:start2 + 26], opened=False) is not None
                    and _valid_pairs(steps[start2 + 26:])):
                cuts.add(start2 + 22)
    if len(cuts) != 1:
        raise ValueError(f"v1 path has {len(cuts)} possible open cut positions")
    prefix = z[:cuts.pop() + 1]
    return np.pad(prefix, (69 - len(prefix), 0), mode="constant", constant_values=int(z[0]))


def path_hash(z: np.ndarray) -> str:
    """Hash integer observations, including their fixed-window length."""
    path = np.asarray(z, dtype="<i2")
    return hashlib.sha256(path.tobytes()).hexdigest()


def make_dataset(
    task: str,
    size: int,
    seed: int,
    exclude_hashes: Collection[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate unique paths, rejecting hashes already used by another split.

    Returned labels are checked against `oracle_labels` before delivery.
    For `tiny`, request 16 examples and evaluate training examples only.
    """
    if task not in TASKS or size < 1:
        raise ValueError(f"Invalid task or size: {task}, {size}")
    if task == "tiny" and size != 16:
        raise ValueError("tiny is fixed at 16 examples")
    rng = np.random.default_rng(seed)
    excluded = set(exclude_hashes or ())
    collected_z: list[np.ndarray] = []
    collected_y: list[int] = []
    hashes: list[str] = []
    attempts = 0
    while len(collected_z) < size:
        attempts += 1
        if attempts > max(10_000, 100 * size):
            raise RuntimeError(f"Cannot find {size} unique {task} paths outside exclusions")
        if task in ("tiny", "hard_closed", "hard_open"):
            # Each candidate is drawn from the unchanged v1 generator. Derive
            # independent deterministic seeds from this dataset's RNG stream.
            from scripts.synthetic_checks import make_dataset as make_v1_dataset

            candidate_seed = int(rng.integers(0, 2**31 - 1))
            close, labels = make_v1_dataset(1, candidate_seed, LOG_STEP)
            path = _v1_close_to_z(close[0])
            if task == "hard_open":
                path = _open_prefix(path)
            label = int(labels[0])
        else:
            path, label = _generate_new(task, rng)
        digest = path_hash(path)
        if digest in excluded:
            continue
        excluded.add(digest)
        collected_z.append(path)
        collected_y.append(label)
        hashes.append(digest)
    z = np.stack(collected_z).astype(np.int16)
    y = np.asarray(collected_y, dtype=np.int64)
    decoded = oracle_labels(task, z)
    if not np.array_equal(y, decoded):
        raise AssertionError(f"Generator labels disagree with independent oracle for {task}")
    return z, y, hashes


def to_close(z: np.ndarray, initial_close: float = 100.0) -> np.ndarray:
    """Convert integer positions to positive close arrays for route_windows."""
    positions = np.asarray(z)
    if positions.ndim != 2 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("z must have integer shape [N,T]")
    return (initial_close * np.exp(positions.astype(np.float64) * LOG_STEP)).astype(np.float32)
