"""Temporal crossing-event tree and its trainable, fixed-capacity memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROUTE_DIM_V2 = 6
ROUTE_FEATURES_V2 = (
    "direction",
    "log1p_age",
    "signed_progress_over_threshold",
    "max_directional_excursion_over_threshold",
    "log1p_internal_retreats",
    "latest_signed_move_over_threshold",
)
NATURAL = 1
ANCESTOR_FINALIZATION = 2
# Dimensionless tolerance factor; each router scales it by its finest threshold.
THRESHOLD_ATOL = 1e-10


@dataclass
class _Node:
    id: int
    parent_id: int
    origin: float
    anchor: float
    direction: int
    birth_time: int
    age: int = 0
    max_excursion: float = 0.0
    retreats: int = 0
    previous_aligned_progress: float = 0.0


class TemporalEventRouter:
    """One-path, bounded temporal event router for a single observed log-price stream."""

    def __init__(self, thresholds: Sequence[float], record_events: bool = False) -> None:
        levels = np.asarray(thresholds, dtype=np.float64)
        if levels.ndim != 1 or len(levels) == 0:
            raise ValueError("thresholds must be a nonempty coarse-to-fine sequence")
        if not np.all(np.isfinite(levels)) or np.any(levels <= 0) or np.any(np.diff(levels) >= 0):
            raise ValueError("thresholds must be positive, finite, and strictly decreasing")
        self.thresholds = levels
        self.tolerance = THRESHOLD_ATOL * float(levels[-1])
        self.depths = len(levels) + 1
        self.nodes: list[_Node] = []
        self.previous_close: float | None = None
        self.time_index = -1
        self.next_event_id = 1
        self.born_count = 0
        self.terminated_count = 0
        self.transfer_count = 0
        self.record_events = record_events
        self.records: list[dict] = []

    @property
    def closed_count(self) -> int:
        return self.terminated_count

    @property
    def delivered_count(self) -> int:
        return self.transfer_count

    def _scale(self, depth: int) -> float:
        return float(self.thresholds[max(0, depth - 1)])

    def _metadata(self, node: _Node, depth: int, close: float, latest_move: float) -> np.ndarray:
        scale = self._scale(depth)
        signed_progress = (close - node.origin) / scale
        return np.asarray(
            [
                node.direction,
                np.log1p(node.age),
                signed_progress,
                node.max_excursion / scale,
                np.log1p(node.retreats),
                latest_move / scale,
            ],
            dtype=np.float32,
        )

    def step(self, log_close: float) -> dict[str, np.ndarray | np.int64]:
        """Observe one actual close; return routing decisions for that tick."""
        close = float(log_close)
        if not np.isfinite(close):
            raise ValueError("log_close must be finite")
        self.time_index += 1
        first = self.previous_close is None
        latest_move = 0.0 if first else close - self.previous_close
        if first:
            self.nodes = [_Node(0, -1, close, close, 0, self.time_index)]
        old_deepest = len(self.nodes) - 1
        metadata = np.zeros((self.depths, ROUTE_DIM_V2), dtype=np.float32)
        births = np.zeros(self.depths, dtype=np.bool_)
        closures = np.zeros(self.depths, dtype=np.bool_)
        birth_ids = np.full(self.depths, -1, dtype=np.int64)
        closure_ids = np.full(self.depths, -1, dtype=np.int64)
        closure_parent_ids = np.full(self.depths, -1, dtype=np.int64)
        closure_reasons = np.zeros(self.depths, dtype=np.int8)

        # Metadata belongs to the preexisting chain before any birth/closure.
        for depth, node in enumerate(self.nodes):
            node.age += 1
            aligned = node.direction * (close - node.origin)
            if depth == 0:
                node.max_excursion = max(node.max_excursion, abs(close - node.origin))
            else:
                if aligned > self.tolerance and aligned < node.previous_aligned_progress - self.tolerance:
                    node.retreats += 1
                node.max_excursion = max(node.max_excursion, aligned)
                node.previous_aligned_progress = aligned
            metadata[depth] = self._metadata(node, depth, close, latest_move)

        shallowest_close: int | None = None
        if not first:
            for depth in range(1, len(self.nodes)):
                node = self.nodes[depth]
                if node.direction * (close - node.origin) <= self.tolerance:
                    shallowest_close = depth
                    break

        if shallowest_close is not None:
            observer = old_deepest
            for depth in range(old_deepest, shallowest_close - 1, -1):
                node = self.nodes[depth]
                parent = self.nodes[depth - 1]
                natural = node.direction * (close - node.origin) <= self.tolerance
                reason = NATURAL if natural else ANCESTOR_FINALIZATION
                closures[depth] = True
                closure_ids[depth] = node.id
                closure_parent_ids[depth] = node.parent_id
                closure_reasons[depth] = reason
                self.terminated_count += 1
                self.transfer_count += 1
                if self.record_events:
                    self.records.append(
                        {
                            "id": node.id,
                            "parent_id": node.parent_id,
                            "birth_time": node.birth_time,
                            "end_time": self.time_index,
                            "reason": int(reason),
                        }
                    )
                parent.anchor = close
            del self.nodes[shallowest_close:]
        else:
            observer = old_deepest
            if not first and old_deepest < len(self.thresholds):
                parent = self.nodes[-1]
                displacement = close - parent.anchor
                threshold = float(self.thresholds[old_deepest])
                if abs(displacement) + self.tolerance >= threshold:
                    child = _Node(
                        id=self.next_event_id,
                        parent_id=parent.id,
                        origin=parent.anchor,
                        anchor=close,
                        direction=1 if displacement > 0 else -1,
                        birth_time=self.time_index,
                        age=1,
                        max_excursion=abs(displacement),
                        previous_aligned_progress=abs(displacement),
                    )
                    self.nodes.append(child)
                    observer = old_deepest + 1
                    births[observer] = True
                    birth_ids[observer] = child.id
                    metadata[observer] = self._metadata(child, observer, close, latest_move)
                    self.next_event_id += 1
                    self.born_count += 1

        active = np.zeros(self.depths, dtype=np.bool_)
        ids = np.full(self.depths, -1, dtype=np.int64)
        parent_ids = np.full(self.depths, -1, dtype=np.int64)
        for depth, node in enumerate(self.nodes):
            active[depth] = True
            ids[depth] = node.id
            parent_ids[depth] = node.parent_id
        self.previous_close = close
        return {
            "metadata": metadata,
            "births": births,
            "closures": closures,
            "observer": np.int64(observer),
            "active": active,
            "ids": ids,
            "parent_ids": parent_ids,
            "birth_ids": birth_ids,
            "closure_ids": closure_ids,
            "closure_parent_ids": closure_parent_ids,
            "closure_reasons": closure_reasons,
        }

    def process(self, log_closes: Sequence[float]) -> dict[str, np.ndarray]:
        """Continue this stream for any chunk length; no boundary flush."""
        ticks = [self.step(value) for value in log_closes]
        if ticks:
            return {key: np.stack([tick[key] for tick in ticks], axis=0) for key in ticks[0]}
        sample = self._empty_tick()
        return {key: np.empty((0,) + np.shape(value), dtype=np.asarray(value).dtype) for key, value in sample.items()}

    def _empty_tick(self) -> dict:
        d = self.depths
        return {
            "metadata": np.zeros((d, ROUTE_DIM_V2), dtype=np.float32),
            "births": np.zeros(d, dtype=np.bool_),
            "closures": np.zeros(d, dtype=np.bool_),
            "observer": np.int64(0),
            "active": np.zeros(d, dtype=np.bool_),
            "ids": np.full(d, -1, dtype=np.int64),
            "parent_ids": np.full(d, -1, dtype=np.int64),
            "birth_ids": np.full(d, -1, dtype=np.int64),
            "closure_ids": np.full(d, -1, dtype=np.int64),
            "closure_parent_ids": np.full(d, -1, dtype=np.int64),
            "closure_reasons": np.zeros(d, dtype=np.int8),
        }

    def state_dict(self) -> dict:
        """Bounded routing state; optional debug records are excluded."""
        return {
            "thresholds": self.thresholds.tolist(),
            "nodes": [asdict(node) for node in self.nodes],
            "previous_close": self.previous_close,
            "time_index": self.time_index,
            "next_event_id": self.next_event_id,
            "born_count": self.born_count,
            "terminated_count": self.terminated_count,
            "transfer_count": self.transfer_count,
        }

    def load_state_dict(self, state: dict) -> None:
        if not np.array_equal(np.asarray(state["thresholds"], dtype=np.float64), self.thresholds):
            raise ValueError("state thresholds differ from router thresholds")
        self.nodes = [_Node(**node) for node in state["nodes"]]
        self.previous_close = state["previous_close"]
        self.time_index = int(state["time_index"])
        self.next_event_id = int(state["next_event_id"])
        self.born_count = int(state["born_count"])
        self.terminated_count = int(state["terminated_count"])
        self.transfer_count = int(state["transfer_count"])
        self.records.clear()


def route_windows_v2(log_prices: np.ndarray, thresholds: Sequence[float]) -> dict[str, np.ndarray]:
    """Route independent `[N,T]` observed log-close windows."""
    prices = np.asarray(log_prices, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] == 0 or not np.all(np.isfinite(prices)):
        raise ValueError("log_prices must be finite [N,T] with T > 0")
    if prices.shape[0] == 0:
        sample = TemporalEventRouter(thresholds).process(prices.reshape(0))
        return {key: np.empty((0, prices.shape[1]) + value.shape[1:], dtype=value.dtype) for key, value in sample.items()}
    routed = [TemporalEventRouter(thresholds).process(row) for row in prices]
    return {key: np.stack([sample[key] for sample in routed], axis=0) for key in routed[0]}


class TreeCrossingMemory(nn.Module):
    """Learned observation and exactly-once postorder merge over a routed tree."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depths: int,
        transfer: bool = True,
        read_open: bool = True,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, depths) < 1:
            raise ValueError("input_dim, hidden_dim, and depths must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depths = depths
        self.output_dim = depths * hidden_dim
        self.transfer = transfer
        self.read_open = read_open
        self.observe = nn.ModuleList(
            nn.GRUCell(input_dim + ROUTE_DIM_V2, hidden_dim) for _ in range(depths)
        )
        self.merge = nn.ModuleList(
            nn.GRUCell(hidden_dim + ROUTE_DIM_V2, hidden_dim) for _ in range(depths - 1)
        )
        self.readout = nn.ModuleList(
            nn.Linear(hidden_dim + ROUTE_DIM_V2, hidden_dim) for _ in range(depths)
        )

    def initial_state(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        return torch.zeros((batch_size, self.depths, self.hidden_dim), device=device, dtype=dtype)

    def step(self, x_t: Tensor, routing_t: dict[str, Tensor], state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x_t.ndim != 2 or x_t.shape[1] != self.input_dim:
            raise ValueError("x_t must be [B,input_dim]")
        batch = x_t.shape[0]
        metadata = routing_t["metadata"]
        births = routing_t["births"]
        closures = routing_t["closures"]
        observer = routing_t["observer"]
        active = routing_t["active"]
        if metadata.shape != (batch, self.depths, ROUTE_DIM_V2):
            raise ValueError("metadata must be [B,D,ROUTE_DIM_V2]")
        if births.shape != (batch, self.depths) or closures.shape != births.shape or active.shape != births.shape:
            raise ValueError("births, closures, active must be [B,D]")
        if observer.shape != (batch,):
            raise ValueError("observer must be [B]")
        if state is None:
            state = self.initial_state(batch, x_t.device, x_t.dtype)
        if state.shape != (batch, self.depths, self.hidden_dim):
            raise ValueError("state must be [B,D,H]")
        slots = [state[:, depth, :] for depth in range(self.depths)]
        for depth in range(self.depths):
            born = births[:, depth].bool().unsqueeze(-1)
            slots[depth] = torch.where(born, torch.zeros_like(slots[depth]), slots[depth])
            observes = (observer == depth).unsqueeze(-1)
            candidate = self.observe[depth](torch.cat((x_t, metadata[:, depth, :]), dim=-1), slots[depth])
            slots[depth] = torch.where(observes, candidate, slots[depth])
        for depth in range(self.depths - 1, 0, -1):
            closing = closures[:, depth].bool().unsqueeze(-1)
            child_content = slots[depth] if self.transfer else torch.zeros_like(slots[depth])
            merge_input = torch.cat((child_content, metadata[:, depth - 1, :]), dim=-1)
            merged = self.merge[depth - 1](merge_input, slots[depth - 1])
            slots[depth - 1] = torch.where(closing, merged, slots[depth - 1])
            slots[depth] = torch.where(closing, torch.zeros_like(slots[depth]), slots[depth])
        next_state = torch.stack(slots, dim=1)
        outputs = []
        for depth in range(self.depths):
            visible = torch.tanh(
                self.readout[depth](torch.cat((slots[depth], metadata[:, depth, :]), dim=-1))
            )
            allowed = active[:, depth].bool().unsqueeze(-1)
            if depth > 0 and not self.read_open:
                allowed = torch.zeros_like(allowed)
            outputs.append(torch.where(allowed, visible, torch.zeros_like(visible)))
        return torch.cat(outputs, dim=-1), next_state

    def forward(
        self, x: Tensor, routing: dict[str, Tensor], state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError("x must be [B,T,input_dim]")
        batch, time, _ = x.shape
        if state is None:
            state = self.initial_state(batch, x.device, x.dtype)
        outputs = []
        for t in range(time):
            tick = {key: value[:, t] for key, value in routing.items() if key in {"metadata", "births", "closures", "observer", "active"}}
            out, state = self.step(x[:, t], tick, state)
            outputs.append(out)
        if outputs:
            return torch.stack(outputs, dim=1), state
        return x.new_zeros((batch, 0, self.output_dim)), state
