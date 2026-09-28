import numpy as np
import pytest
import torch

from src.crossing_memory import ROUTE_DIM, CrossingMemory, route_windows


def _prices(log_paths):
    return np.exp(np.asarray(log_paths, dtype=np.float64))


def test_hand_routes_returns_closure_and_atomic_jump():
    close = _prices([[0, 0.012, 0.023, 0.013, 0.002, -0.012]])
    stats, closures = route_windows(close, [0.01, 0.02])
    assert stats.shape == (1, 6, 2, ROUTE_DIM)
    assert closures.shape == (1, 6, 2)
    assert stats[0, 2, 0, 2] == 2  # open progress
    assert stats[0, 3, 0, 4] == pytest.approx(np.log(2))  # one internal return
    assert closures[0, 3, 1]  # coarse event closes first
    assert closures[0, 4, 0]  # fine event closes at its origin
    assert stats[0, 4, 0, 0] == 1  # closing direction retained
    assert stats[0, 5, 0, 0] == -1  # new opposite event on next close

    jump = _prices([[0, 0.035, -0.015]])
    j_stats, j_closed = route_windows(jump, [0.01, 0.02])
    assert j_stats[0, 1, 0, 2] == 3
    assert j_stats[0, 1, 1, 2] == 1
    assert j_closed[0, 2].tolist() == [True, True]
    assert j_stats[0, 2, 0, 1] == pytest.approx(np.log(3))  # only two observed updates


def test_route_is_causal_and_windows_independent():
    base = _prices([[0, 0.015, 0.027, 0.009, 0.004], [0, -0.015, -0.023, -0.01, 0.001]])
    changed = base.copy()
    changed[:, 4] *= 1.5
    a, ac = route_windows(base, [0.01, 0.02])
    b, bc = route_windows(changed, [0.01, 0.02])
    np.testing.assert_array_equal(a[:, :4], b[:, :4])
    np.testing.assert_array_equal(ac[:, :4], bc[:, :4])
    one, one_c = route_windows(base[:1], [0.01, 0.02])
    np.testing.assert_array_equal(a[:1], one)
    np.testing.assert_array_equal(ac[:1], one_c)
    for length in range(1, base.shape[1] + 1):
        prefix_stats, prefix_closures = route_windows(base[:, :length], [0.01, 0.02])
        np.testing.assert_array_equal(a[:, :length], prefix_stats)
        np.testing.assert_array_equal(ac[:, :length], prefix_closures)


def test_route_rejects_invalid_grids_and_prices():
    with pytest.raises(ValueError):
        route_windows(np.ones((1, 2)), [0.01, 0.015])
    with pytest.raises(ValueError):
        route_windows(np.ones((1, 2)), [0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        route_windows(np.array([[1.0, 0.0]]), [0.01])


def test_stream_batch_equivalence_and_neural_causality():
    torch.manual_seed(4)
    close = _prices([[0, 0.015, 0.027, 0.009, 0.004], [0, -0.015, -0.023, -0.01, 0.001]])
    stats, closures = route_windows(close, [0.01, 0.02])
    st = torch.from_numpy(stats)
    cl = torch.from_numpy(closures)
    x = torch.randn(2, 5, 3)
    model = CrossingMemory(3, hidden_dim=5, scales=2)
    batch = model(x, st, cl)
    state = model.initial_state(2, x.device, x.dtype)
    streamed = []
    for t in range(5):
        out, state = model.step(x[:, t], st[:, t], cl[:, t], state)
        streamed.append(out)
    torch.testing.assert_close(batch, torch.stack(streamed, dim=1))

    future_x = x.clone()
    future_x[:, 4] *= 100
    future_close = close.copy()
    future_close[:, 4] *= 1.5
    future_stats, future_closures = route_windows(future_close, [0.01, 0.02])
    future_out = model(future_x, torch.from_numpy(future_stats), torch.from_numpy(future_closures))
    torch.testing.assert_close(batch[:, :4], future_out[:, :4])


def test_transfer_open_readout_and_finite_gradients():
    torch.manual_seed(5)
    close = _prices([[0, 0.012, 0.025, 0], [0, -0.012, -0.025, 0]])
    stats, closures = route_windows(close, [0.01, 0.02])
    st, cl = torch.from_numpy(stats), torch.from_numpy(closures)
    x = torch.randn(2, 4, 2)
    model = CrossingMemory(2, hidden_dim=6, scales=2)
    out = model(x, st, cl)
    assert out.shape == (2, 4, 12)
    loss = out[:, -1, 6:].square().sum()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.open_update[0].weight_ih.grad.abs().sum() > 0  # child-to-parent path

    no_transfer = CrossingMemory(2, hidden_dim=6, scales=2, transfer=False)
    no_transfer.load_state_dict(model.state_dict())
    assert not torch.allclose(model(x, st, cl)[:, -1, 6:], no_transfer(x, st, cl)[:, -1, 6:])
    no_open = CrossingMemory(2, hidden_dim=6, scales=2, read_open=False)
    no_open.load_state_dict(model.state_dict())
    assert not torch.allclose(model(x, st, cl)[:, 1], no_open(x, st, cl)[:, 1])


def test_memory_parameters_learn_nested_history():
    torch.manual_seed(8)
    paths = [[0, 0.012, 0.025, 0], [0, -0.012, -0.025, 0]] * 8
    close = _prices(paths)
    stats, closures = route_windows(close, [0.01, 0.02])
    st, cl = torch.from_numpy(stats), torch.from_numpy(closures)
    x = torch.zeros((len(paths), 4, 1))
    y = torch.tensor([1, 0] * 8, dtype=torch.float32)
    model = CrossingMemory(1, hidden_dim=6, scales=2)
    head = torch.nn.Linear(model.output_dim, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=0.03)
    initial = None
    for step in range(60):
        optimizer.zero_grad()
        logits = head(model(x, st, cl)[:, -1]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.1
    assert float(loss.detach()) < initial / 3
