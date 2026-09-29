import numpy as np
import torch

from src.crossing_memory_v2 import TreeCrossingMemory, TemporalEventRouter, route_windows_v2


def _tensor_route(route):
    return {key: torch.from_numpy(value) for key, value in route.items()}


def test_neural_chunk_and_step_equivalence_with_state_and_gradients():
    torch.manual_seed(17)
    paths = np.asarray(
        [[0, 6, 9, 10, 9, 6, 0, 6], [0, -6, -9, -10, -9, -6, 0, -6]],
        dtype=np.float64,
    )
    full_route = _tensor_route(route_windows_v2(paths, [6, 3, 1]))
    x = torch.randn(2, 8, 2)
    model = TreeCrossingMemory(2, hidden_dim=5, depths=4)
    full_out, full_state = model(x, full_route)
    assert full_out.shape == (2, 8, 20)

    router_a = TemporalEventRouter([6, 3, 1])
    router_b = TemporalEventRouter([6, 3, 1])
    chunks_a = [router_a.process(paths[0, :4]), router_a.process(paths[0, 4:])]
    chunks_b = [router_b.process(paths[1, :4]), router_b.process(paths[1, 4:])]
    state = model.initial_state(2, x.device, x.dtype)
    original_state = state.clone()
    parts = []
    for start, stop, samples in [(0, 4, [chunks_a[0], chunks_b[0]]), (4, 8, [chunks_a[1], chunks_b[1]])]:
        route = _tensor_route({key: np.stack([sample[key] for sample in samples]) for key in samples[0]})
        out, state = model(x[:, start:stop], route, state)
        parts.append(out)
    torch.testing.assert_close(torch.cat(parts, dim=1), full_out)
    torch.testing.assert_close(state, full_state)
    torch.testing.assert_close(original_state, torch.zeros_like(original_state))

    step_state = model.initial_state(2, x.device, x.dtype)
    step_outputs = []
    for t in range(8):
        route_t = {key: value[:, t] for key, value in full_route.items()}
        step_out, step_state = model.step(x[:, t], route_t, step_state)
        step_outputs.append(step_out)
    torch.testing.assert_close(torch.stack(step_outputs, dim=1), full_out)
    torch.testing.assert_close(step_state, full_state)

    model.zero_grad()
    full_out.square().mean().backward()
    full_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    model.zero_grad()
    torch.cat(parts, dim=1).square().mean().backward()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, full_grads[name])


def test_postorder_merge_open_readout_and_finite_gradients():
    torch.manual_seed(19)
    paths = np.asarray([[0, 6, 9, 10, -2], [0, -6, -9, -10, 2]], dtype=np.float64)
    route = _tensor_route(route_windows_v2(paths, [6, 3, 1]))
    x = torch.randn(2, 5, 3)
    model = TreeCrossingMemory(3, hidden_dim=6, depths=4)
    outputs, state = model(x, route)
    assert torch.all(state[:, 1:] == 0)
    assert torch.any(state[:, 0] != 0)
    loss = outputs[:, 3].square().sum() + outputs[:, 4, :6].square().sum()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.observe[3].weight_ih.grad.abs().sum() > 0
    assert model.merge[2].weight_ih.grad.abs().sum() > 0
    assert model.merge[0].weight_ih.grad.abs().sum() > 0

    no_transfer = TreeCrossingMemory(3, 6, 4, transfer=False)
    no_transfer.load_state_dict(model.state_dict())
    no_transfer_outputs, _ = no_transfer(x, route)
    assert not torch.allclose(outputs[:, 4, :6], no_transfer_outputs[:, 4, :6])

    no_open = TreeCrossingMemory(3, 6, 4, read_open=False)
    no_open.load_state_dict(model.state_dict())
    no_open_outputs, _ = no_open(x, route)
    assert torch.all(no_open_outputs[:, 3, 6:] == 0)
    assert torch.any(outputs[:, 3, 6:] != 0)


def test_future_change_preserves_neural_prefix():
    torch.manual_seed(23)
    path = np.asarray([[0, 6, 9, 10, 9, 6, 0]], dtype=np.float64)
    changed = path.copy()
    changed[:, -1] = -20
    route = _tensor_route(route_windows_v2(path, [6, 3, 1]))
    changed_route = _tensor_route(route_windows_v2(changed, [6, 3, 1]))
    x = torch.randn(1, 7, 2)
    altered_x = x.clone()
    altered_x[:, -1] *= 7
    model = TreeCrossingMemory(2, 4, 4)
    base_out, _ = model(x, route)
    altered_out, _ = model(altered_x, changed_route)
    torch.testing.assert_close(base_out[:, :-1], altered_out[:, :-1])


def test_tiny_threshold_has_no_zero_birth_and_mirrors():
    thresholds = [3e-14, 1e-14, 1e-15]
    positive = np.asarray([0.0, 0.0, 3e-14, 4e-14, 0.0])
    negative = -positive
    forward = TemporalEventRouter(thresholds).process(positive)
    mirror = TemporalEventRouter(thresholds).process(negative)
    assert not forward["births"][1].any()
    assert forward["births"][2, 1]
    assert forward["births"][3, 2]
    np.testing.assert_array_equal(forward["birth_ids"], mirror["birth_ids"])
    np.testing.assert_array_equal(forward["closure_ids"], mirror["closure_ids"])
    np.testing.assert_array_equal(forward["closure_reasons"], mirror["closure_reasons"])
    np.testing.assert_array_equal(forward["metadata"][..., 0], -mirror["metadata"][..., 0])
