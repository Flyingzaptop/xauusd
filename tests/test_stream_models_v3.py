import numpy as np
import pytest
import torch

from src.crossing_memory_v2 import TemporalEventRouter
from src.stream_models_v3 import StreamForecaster, matched_baseline_widths, parameter_count


def _routing_with_resets(paths, resets):
    samples = []
    for path, lane_reset in zip(paths, resets):
        router = None
        ticks = []
        for close, reset in zip(path, lane_reset):
            if reset:
                router = TemporalEventRouter([6, 3, 1])
            ticks.append(router.step(float(close)))
        samples.append({key: np.stack([tick[key] for tick in ticks]) for key in ticks[0]})
    return {
        key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
        for key in samples[0]
    }


@pytest.mark.parametrize("variant", ["a", "b", "c", "no_transfer", "no_open"])
def test_full_step_chunk_and_lane_reset_equivalence(variant):
    torch.manual_seed(31)
    paths = np.asarray(
        [[0, 6, 9, 10, 0, 6, 9, 0], [0, -6, -9, -10, -9, -6, 0, 6]],
        dtype=np.float64,
    )
    resets = torch.tensor(
        [[True, False, False, False, True, False, False, False],
         [True, False, False, False, False, False, True, False]]
    )
    routing = _routing_with_resets(paths, resets.numpy())
    x = torch.randn(2, 8, 4)
    model = StreamForecaster(variant, gru_hidden=7, memory_hidden=4)
    route_arg = None if variant == "a" else routing
    full, final_state = model(x, route_arg, resets)
    assert full.shape == (2, 8)

    state = model.initial_state(2, x.device, x.dtype)
    input_state = {key: value.clone() if value is not None else None for key, value in state.items()}
    parts = []
    for start, stop in [(0, 3), (3, 5), (5, 8)]:
        route_chunk = None if route_arg is None else {key: value[:, start:stop] for key, value in route_arg.items()}
        out, state = model(x[:, start:stop], route_chunk, resets[:, start:stop], state)
        parts.append(out)
    torch.testing.assert_close(torch.cat(parts, dim=1), full)
    torch.testing.assert_close(state["gru"], final_state["gru"])
    if model.uses_tree:
        torch.testing.assert_close(state["tree"], final_state["tree"])
    for key, value in input_state.items():
        if value is not None:
            torch.testing.assert_close(value, torch.zeros_like(value))

    step_state = model.initial_state(2, x.device, x.dtype)
    step_outputs = []
    for t in range(8):
        route_t = None if route_arg is None else {key: value[:, t] for key, value in route_arg.items()}
        out, step_state = model.step(x[:, t], route_t, resets[:, t], step_state)
        step_outputs.append(out)
    torch.testing.assert_close(torch.stack(step_outputs, dim=1), full)

    # The first lane's real gap at t=4 must erase all dependence on t<4.
    suffix_route = None if route_arg is None else {key: value[:1, 4:] for key, value in route_arg.items()}
    suffix, _ = model(x[:1, 4:], suffix_route, resets[:1, 4:])
    torch.testing.assert_close(suffix[0], full[0, 4:])


@pytest.mark.parametrize("variant", ["a", "b", "c", "no_transfer", "no_open"])
def test_future_change_does_not_change_prefix(variant):
    torch.manual_seed(37)
    paths = np.asarray([[0, 6, 9, 10, 9, 6, 0], [0, -6, -9, -10, -9, -6, 0]], dtype=np.float64)
    changed_paths = paths.copy()
    changed_paths[:, -1] = [20, -20]
    reset = torch.zeros((2, 7), dtype=torch.bool)
    reset[:, 0] = True
    routing = _routing_with_resets(paths, reset.numpy())
    changed_routing = _routing_with_resets(changed_paths, reset.numpy())
    x = torch.randn(2, 7, 4)
    changed_x = x.clone()
    changed_x[:, -1] *= 30
    model = StreamForecaster(variant, gru_hidden=7, memory_hidden=4)
    original, _ = model(x, None if variant == "a" else routing, reset)
    altered, _ = model(changed_x, None if variant == "a" else changed_routing, reset)
    torch.testing.assert_close(original[:, :-1], altered[:, :-1])


def test_explicit_detach_and_tree_gradients():
    torch.manual_seed(41)
    path = np.asarray([[0, 6, 9, 10, 9, 6, 0]], dtype=np.float64)
    reset = torch.tensor([[True, False, False, False, False, False, False]])
    route = _routing_with_resets(path, reset.numpy())
    x = torch.randn(1, 7, 4)
    model = StreamForecaster("c", gru_hidden=7, memory_hidden=4)
    first, state = model(x[:, :4], {key: value[:, :4] for key, value in route.items()}, reset[:, :4])
    assert state["gru"].grad_fn is not None
    assert state["tree"].grad_fn is not None
    detached = model.detach_state(state)
    assert detached["gru"].grad_fn is None
    assert detached["tree"].grad_fn is None
    torch.testing.assert_close(detached["gru"], state["gru"])
    second, _ = model(x[:, 4:], {key: value[:, 4:] for key, value in route.items()}, reset[:, 4:], state)
    (first.square().sum() + second.square().sum()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.tree.observe[3].weight_ih.grad.abs().sum() > 0
    assert model.tree.merge[0].weight_ih.grad.abs().sum() > 0


def test_parameter_matching_is_nearest_and_ablations_keep_budget():
    matched = matched_baseline_widths(c_gru_hidden=16, memory_hidden=8, depths=4)
    target = matched["c_parameters"]
    assert StreamForecaster("a", matched["a"]).gru.input_size == 4
    assert StreamForecaster("b", matched["b"]).gru.input_size == 44
    assert StreamForecaster("c", 16).gru.input_size == 76
    for variant in ("a", "b"):
        width = matched[variant]
        assert parameter_count(StreamForecaster(variant, width)) == matched[f"{variant}_parameters"]
        for neighbor in (width - 1, width + 1):
            if neighbor >= 1:
                competitor = parameter_count(StreamForecaster(variant, neighbor))
                assert abs(matched[f"{variant}_parameters"] - target) <= abs(competitor - target)
    for variant in ("no_transfer", "no_open"):
        assert parameter_count(StreamForecaster(variant, 16)) == target
