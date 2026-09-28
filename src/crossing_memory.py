"""Causal fixed-grid routing and trainable crossing memory."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROUTE_DIM = 6


def route_windows(
    close: np.ndarray, thresholds: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Route independent close windows into causal crossing summaries.

    The shared log-price grid is anchored at each window's first close. Each
    observed close causes at most one event update per scale, even on a jump.
    """
    prices = np.asarray(close, dtype=np.float64)
    levels = np.asarray(thresholds, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] == 0:
        raise ValueError("close must have shape [N,T] with T > 0")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("close must contain positive finite prices")
    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("thresholds must be a nonempty sequence")
    if not np.all(np.isfinite(levels)) or np.any(levels <= 0) or np.any(np.diff(levels) <= 0):
        raise ValueError("thresholds must be positive, finite, and strictly increasing")
    adjacent_ratios = levels[1:] / levels[:-1]
    if not np.allclose(adjacent_ratios, np.round(adjacent_ratios), rtol=1e-8, atol=1e-8):
        raise ValueError("each threshold must be an integer multiple of the preceding scale")

    n, t = prices.shape
    s = len(levels)
    stats = np.zeros((n, t, s, ROUTE_DIM), dtype=np.float32)
    closures = np.zeros((n, t, s), dtype=np.bool_)
    if n == 0 or t == 1:
        return stats, closures

    log_relative = np.log(prices) - np.log(prices[:, :1])
    index = np.floor(log_relative[:, :, None] / levels[None, None, :]).astype(np.int64)
    index[:, 0, :] = 0
    direction = np.zeros((n, s), dtype=np.int8)
    origin = np.zeros((n, s), dtype=np.int64)
    age = np.zeros((n, s), dtype=np.int64)
    maximum = np.zeros((n, s), dtype=np.int64)
    returns = np.zeros((n, s), dtype=np.int64)
    previous_progress = np.zeros((n, s), dtype=np.int64)

    for k in range(1, t):
        before = index[:, k - 1, :]
        after = index[:, k, :]
        changed = after != before
        idle = direction == 0
        starting = idle & changed
        direction[starting] = np.sign(after[starting] - before[starting]).astype(np.int8)
        origin[starting] = before[starting]
        maximum[starting] = np.abs(after[starting] - before[starting])
        previous_progress[starting] = maximum[starting]

        active = direction != 0
        age[active] += 1
        progress = direction.astype(np.int64) * (after - origin)
        maximum[active] = np.maximum(maximum[active], progress[active])
        closing = active & ~starting & (progress <= 0)
        internal_return = active & ~starting & ~closing & (progress < previous_progress)
        returns[internal_return] += 1

        out = stats[:, k, :, :]
        out[:, :, 0] = direction
        out[:, :, 1] = np.log1p(age)
        out[:, :, 2] = np.where(active, progress, 0)
        out[:, :, 3] = maximum
        out[:, :, 4] = np.log1p(returns)
        out[:, :, 5] = (log_relative[:, k] - log_relative[:, k - 1])[:, None] / levels[None, :]
        closures[:, k, :] = closing
        previous_progress[active] = progress[active]

        direction[closing] = 0
        origin[closing] = after[closing]
        age[closing] = 0
        maximum[closing] = 0
        returns[closing] = 0
        previous_progress[closing] = 0

    return stats, closures


class CrossingMemory(nn.Module):
    """Fixed-capacity differentiable memory for routed close crossings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 12,
        scales: int = 3,
        transfer: bool = True,
        read_open: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or scales < 1:
            raise ValueError("input_dim, hidden_dim, and scales must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scales = scales
        self.transfer = transfer
        self.read_open = read_open
        self.output_dim = scales * hidden_dim
        self.open_update = nn.ModuleList(
            nn.GRUCell(input_dim + ROUTE_DIM + hidden_dim, hidden_dim)
            for _ in range(scales)
        )
        self.close_update = nn.ModuleList(
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(scales)
        )
        self.readout = nn.ModuleList(
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(scales)
        )

    def initial_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> tuple[Tensor, Tensor]:
        shape = (batch_size, self.scales, self.hidden_dim)
        return (torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))

    def reset_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> tuple[Tensor, Tensor]:
        return self.initial_state(batch_size, device, dtype)

    def step(
        self,
        x_t: Tensor,
        stats_t: Tensor,
        closures_t: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if x_t.ndim != 2 or x_t.shape[1] != self.input_dim:
            raise ValueError("x_t must have shape [B,input_dim]")
        batch = x_t.shape[0]
        if stats_t.shape != (batch, self.scales, ROUTE_DIM):
            raise ValueError("stats_t must have shape [B,scales,ROUTE_DIM]")
        if closures_t.shape != (batch, self.scales):
            raise ValueError("closures_t must have shape [B,scales]")
        if state is None:
            state = self.initial_state(batch, x_t.device, x_t.dtype)
        open_prev, closed_prev = state
        if open_prev.shape != (batch, self.scales, self.hidden_dim) or closed_prev.shape != open_prev.shape:
            raise ValueError("state has wrong shape")

        open_next: list[Tensor] = []
        closed_next: list[Tensor] = []
        read: list[Tensor] = []
        child_message = torch.zeros((batch, self.hidden_dim), device=x_t.device, dtype=x_t.dtype)
        for scale in range(self.scales):
            transferred = child_message if self.transfer else torch.zeros_like(child_message)
            update_input = torch.cat((x_t, stats_t[:, scale, :], transferred), dim=-1)
            open_candidate = self.open_update[scale](update_input, open_prev[:, scale, :])
            is_active = (stats_t[:, scale, 0] != 0).unsqueeze(-1)
            updated_open = torch.where(is_active, open_candidate, torch.zeros_like(open_candidate))
            closed_candidate = self.close_update[scale](updated_open, closed_prev[:, scale, :])
            is_closed = closures_t[:, scale].bool().unsqueeze(-1)
            new_closed = torch.where(is_closed, closed_candidate, closed_prev[:, scale, :])
            new_open = torch.where(is_closed, torch.zeros_like(updated_open), updated_open)
            open_next.append(new_open)
            closed_next.append(new_closed)
            visible_open = new_open if self.read_open else torch.zeros_like(new_open)
            read.append(torch.tanh(self.readout[scale](torch.cat((new_closed, visible_open), dim=-1))))
            child_message = torch.where(is_closed, updated_open, torch.zeros_like(updated_open))
        return torch.cat(read, dim=-1), (torch.stack(open_next, dim=1), torch.stack(closed_next, dim=1))

    def forward(self, x: Tensor, stats: Tensor, closures: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("x must have shape [B,T,input_dim]")
        batch, length, _ = x.shape
        if stats.shape != (batch, length, self.scales, ROUTE_DIM):
            raise ValueError("stats must have shape [B,T,scales,ROUTE_DIM]")
        if closures.shape != (batch, length, self.scales):
            raise ValueError("closures must have shape [B,T,scales]")
        state = self.initial_state(batch, x.device, x.dtype)
        outputs: list[Tensor] = []
        for t in range(length):
            output, state = self.step(x[:, t], stats[:, t], closures[:, t], state)
            outputs.append(output)
        if not outputs:
            return x.new_zeros((batch, 0, self.output_dim))
        return torch.stack(outputs, dim=1)
