"""Deterministic, stateful TBPTT packing of complete minute-bar segments.

`PackedStream` assigns whole chronological segments round-robin to independent
lanes. It preserves every input minute, including observations with invalid
labels, and emits fixed-size chunks in lane order. `reset[b,t]` must be applied
to model state *before* processing observation `[b,t]`. Padding has
`indices=-1`, `valid=False`, `reset=True`, and zero data. A model must never
use padded positions for loss; no padded position precedes later real data in
the same lane. The iterator is deterministic and reusable across epochs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np


TREE_KEYS = ("metadata", "births", "closures", "observer", "active")


class PackedStream:
    """Pack a prepared chronological split into `batch_size` stateful lanes."""

    def __init__(self, split: Mapping, batch_size: int, chunk_length: int) -> None:
        if batch_size <= 0 or chunk_length <= 0:
            raise ValueError("batch_size and chunk_length must be positive")
        self.batch_size = int(batch_size)
        self.chunk_length = int(chunk_length)

        required = ("x", "tree", "y", "valid", "reset", "timestamp")
        missing = [name for name in required if name not in split]
        if missing:
            raise ValueError(f"Prepared split missing keys: {missing}")
        self.x = np.asarray(split["x"])
        self.y = np.asarray(split["y"])
        self.valid = np.asarray(split["valid"])
        self.reset = np.asarray(split["reset"])
        self.timestamp = np.asarray(split["timestamp"])
        self.tree = {key: np.asarray(split["tree"][key]) for key in TREE_KEYS}
        n = len(self.x)
        if self.x.shape != (n, 4):
            raise ValueError("x must have shape [N,4]")
        if self.y.shape != (n,) or self.valid.shape != (n,) or self.reset.shape != (n,):
            raise ValueError("y, valid, and reset must each have shape [N]")
        if self.timestamp.shape != (n,) or not np.issubdtype(self.timestamp.dtype, np.integer):
            raise ValueError("timestamp must have integer UTC seconds with shape [N]")
        if self.valid.dtype != np.bool_ or self.reset.dtype != np.bool_:
            raise ValueError("valid and reset must be boolean arrays")
        if n and not self.reset[0]:
            raise ValueError("First observation of every split must reset state")
        if n > 1:
            deltas = np.diff(self.timestamp.astype(np.int64))
            if np.any(deltas <= 0):
                raise ValueError("timestamp must increase strictly")
            unmarked = np.flatnonzero((deltas != 60) & ~self.reset[1:])
            if len(unmarked):
                raise ValueError(f"Minute gap before index {int(unmarked[0] + 1)} lacks reset")
        if not np.isfinite(self.x).all():
            raise ValueError("x contains nonfinite values")

        metadata = self.tree["metadata"]
        if metadata.ndim != 3 or metadata.shape[0] != n or metadata.shape[2] != 6:
            raise ValueError("tree.metadata must have shape [N,D,6]")
        depth = metadata.shape[1]
        if depth < 1:
            raise ValueError("tree must have at least one depth")
        for key in ("births", "closures", "active"):
            if self.tree[key].shape != (n, depth) or self.tree[key].dtype != np.bool_:
                raise ValueError(f"tree.{key} must be boolean [N,D]")
        if self.tree["observer"].shape != (n,):
            raise ValueError("tree.observer must have shape [N]")
        if not np.isfinite(metadata).all():
            raise ValueError("tree.metadata contains nonfinite values")

        starts = np.flatnonzero(self.reset)
        stops = np.concatenate((starts[1:], np.asarray([n], dtype=np.int64)))
        self.segments = tuple((int(start), int(stop)) for start, stop in zip(starts, stops))
        lane_parts: list[list[np.ndarray]] = [[] for _ in range(self.batch_size)]
        for segment_number, (start, stop) in enumerate(self.segments):
            lane_parts[segment_number % self.batch_size].append(np.arange(start, stop, dtype=np.int64))
        self.lane_indices = tuple(
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            for parts in lane_parts
        )
        self.n_observations = n
        self.n_valid_labels = int(self.valid.sum())
        self.total_chunks = (max((len(lane) for lane in self.lane_indices), default=0)
                             + self.chunk_length - 1) // self.chunk_length

    def __len__(self) -> int:
        return self.total_chunks

    def __iter__(self) -> Iterator[dict[str, np.ndarray | dict[str, np.ndarray]]]:
        b, t = self.batch_size, self.chunk_length
        for chunk in range(self.total_chunks):
            indices = np.full((b, t), -1, dtype=np.int64)
            for lane_number, lane in enumerate(self.lane_indices):
                selected = lane[chunk * t : (chunk + 1) * t]
                indices[lane_number, : len(selected)] = selected
            real = indices >= 0
            x = np.zeros((b, t, 4), dtype=self.x.dtype)
            y = np.zeros((b, t), dtype=self.y.dtype)
            valid = np.zeros((b, t), dtype=np.bool_)
            reset = np.ones((b, t), dtype=np.bool_)
            x[real] = self.x[indices[real]]
            y[real] = self.y[indices[real]]
            valid[real] = self.valid[indices[real]]
            reset[real] = self.reset[indices[real]]
            tree = {}
            for key, values in self.tree.items():
                packed = np.zeros((b, t) + values.shape[1:], dtype=values.dtype)
                packed[real] = values[indices[real]]
                tree[key] = packed
            yield {"indices": indices, "reset": reset, "valid": valid,
                   "x": x, "y": y, "tree": tree}
