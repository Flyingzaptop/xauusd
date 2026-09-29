"""Comparable causal stream models for the v3 M1 close-time experiment."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from src.crossing_memory_v2 import ROUTE_DIM_V2, TreeCrossingMemory


INPUT_DIM = 4
VARIANTS = ("a", "b", "c", "no_transfer", "no_open")


class StreamForecaster(nn.Module):
    """One-observation-at-a-time GRU with optional routed metadata and tree memory.

    A receives four candle features. B receives those plus all causal v2
    routing fields: metadata, birth/closure/active masks and observer one-hot.
    C and its ablations also receive the trainable tree readout. All
    variants share the same GRUCell/head structure and reset semantics.
    """

    def __init__(
        self,
        variant: str,
        gru_hidden: int,
        memory_hidden: int = 8,
        depths: int = 4,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if min(gru_hidden, memory_hidden, depths) < 1:
            raise ValueError("hidden widths and depths must be positive")
        self.variant = variant
        self.gru_hidden = gru_hidden
        self.memory_hidden = memory_hidden
        self.depths = depths
        self.metadata_dim = depths * ROUTE_DIM_V2
        self.routing_dim = depths * (ROUTE_DIM_V2 + 4)
        self.uses_tree = variant in ("c", "no_transfer", "no_open")
        if self.uses_tree:
            self.tree = TreeCrossingMemory(
                input_dim=INPUT_DIM,
                hidden_dim=memory_hidden,
                depths=depths,
                transfer=variant != "no_transfer",
                read_open=variant != "no_open",
            )
        else:
            self.tree = None
        gru_input_dim = INPUT_DIM
        if variant != "a":
            gru_input_dim += self.routing_dim
        if self.uses_tree:
            gru_input_dim += self.tree.output_dim
        self.gru = nn.GRUCell(gru_input_dim, gru_hidden)
        self.head = nn.Linear(gru_hidden, 1)

    def initial_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor | None]:
        return {
            "gru": torch.zeros((batch_size, self.gru_hidden), device=device, dtype=dtype),
            "tree": self.tree.initial_state(batch_size, device, dtype) if self.uses_tree else None,
        }

    @staticmethod
    def detach_state(state: Mapping[str, Tensor | None]) -> dict[str, Tensor | None]:
        """Explicitly truncate history at a TBPTT boundary without resetting values."""
        return {key: value.detach() if value is not None else None for key, value in state.items()}

    def step(
        self,
        x_t: Tensor,
        routing_t: Mapping[str, Tensor] | None,
        reset_t: Tensor,
        state: Mapping[str, Tensor | None] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        if x_t.ndim != 2 or x_t.shape[-1] != INPUT_DIM:
            raise ValueError("x_t must be [B,4]")
        batch = x_t.shape[0]
        if reset_t.shape != (batch,):
            raise ValueError("reset_t must be [B]")
        if state is None:
            state = self.initial_state(batch, x_t.device, x_t.dtype)
        gru_state = state["gru"]
        tree_state = state["tree"]
        if gru_state is None or gru_state.shape != (batch, self.gru_hidden):
            raise ValueError("GRU state has wrong shape")
        reset = reset_t.bool().unsqueeze(-1)
        gru_state = torch.where(reset, torch.zeros_like(gru_state), gru_state)
        inputs = [x_t]
        if self.variant != "a":
            required = ("metadata", "births", "closures", "active", "observer")
            if routing_t is None or any(key not in routing_t for key in required):
                raise ValueError("B/C variants require all causal routing fields")
            metadata = routing_t["metadata"]
            if metadata.shape != (batch, self.depths, ROUTE_DIM_V2):
                raise ValueError("metadata must be [B,depths,ROUTE_DIM_V2]")
            for key in ("births", "closures", "active"):
                if routing_t[key].shape != (batch, self.depths):
                    raise ValueError(f"{key} must be [B,depths]")
            if routing_t["observer"].shape != (batch,):
                raise ValueError("observer must be [B]")
            observer_one_hot = nn.functional.one_hot(
                routing_t["observer"].long(), num_classes=self.depths
            ).to(dtype=x_t.dtype)
            inputs.extend(
                (
                    metadata.reshape(batch, self.metadata_dim),
                    routing_t["births"].to(dtype=x_t.dtype),
                    routing_t["closures"].to(dtype=x_t.dtype),
                    routing_t["active"].to(dtype=x_t.dtype),
                    observer_one_hot,
                )
            )
        if self.uses_tree:
            if tree_state is None or tree_state.shape != (batch, self.depths, self.memory_hidden):
                raise ValueError("tree state has wrong shape")
            tree_state = torch.where(
                reset.unsqueeze(-1), torch.zeros_like(tree_state), tree_state
            )
            tree_features, tree_state = self.tree.step(x_t, routing_t, tree_state)
            inputs.append(tree_features)
        next_gru = self.gru(torch.cat(inputs, dim=-1), gru_state)
        logit = self.head(next_gru).squeeze(-1)
        return logit, {"gru": next_gru, "tree": tree_state}

    def forward(
        self,
        x: Tensor,
        routing: Mapping[str, Tensor] | None,
        reset: Tensor,
        state: Mapping[str, Tensor | None] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        if x.ndim != 3 or x.shape[-1] != INPUT_DIM:
            raise ValueError("x must be [B,T,4]")
        batch, length, _ = x.shape
        if reset.shape != (batch, length):
            raise ValueError("reset must be [B,T]")
        if state is None:
            state = self.initial_state(batch, x.device, x.dtype)
        outputs = []
        for t in range(length):
            routing_t = None if routing is None else {key: value[:, t] for key, value in routing.items()}
            output, state = self.step(x[:, t], routing_t, reset[:, t], state)
            outputs.append(output)
        return (torch.stack(outputs, dim=1) if outputs else x.new_zeros((batch, 0))), dict(state)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_baseline_widths(
    c_gru_hidden: int = 16,
    memory_hidden: int = 8,
    depths: int = 4,
    max_width: int = 256,
) -> dict[str, int]:
    """Choose A/B GRU widths with total trainable counts nearest fixed C."""
    target = parameter_count(StreamForecaster("c", c_gru_hidden, memory_hidden, depths))

    def baseline_count(input_dim: int, width: int) -> int:
        # GRUCell: input/hidden weights plus two biases; linear binary head.
        return 3 * width * input_dim + 3 * width * width + 6 * width + width + 1

    def closest(input_dim: int) -> int:
        return min(range(1, max_width + 1), key=lambda width: (abs(baseline_count(input_dim, width) - target), width))

    a_width = closest(INPUT_DIM)
    b_width = closest(INPUT_DIM + depths * (ROUTE_DIM_V2 + 4))
    return {
        "a": a_width,
        "b": b_width,
        "c": c_gru_hidden,
        "a_parameters": baseline_count(INPUT_DIM, a_width),
        "b_parameters": baseline_count(INPUT_DIM + depths * (ROUTE_DIM_V2 + 4), b_width),
        "c_parameters": target,
    }
