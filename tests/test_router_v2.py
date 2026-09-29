"""Contract checks for the causal v2 event tree; generated log-close paths only."""

import numpy as np
import pytest

from src.crossing_memory_v2 import TemporalEventRouter, route_windows_v2


THRESHOLDS = [6.0, 3.0, 1.0]


def _route(values):
    return TemporalEventRouter(THRESHOLDS).process(np.asarray(values, dtype=np.float64))


def _event_ids(output, field):
    return [int(value) for value in output[field].ravel() if value > 0]


def test_manual_nested_natural_closures_and_observation_owner():
    output = _route([0, 6, 9, 10, 9, 6, 0])
    assert output["birth_ids"].shape == (7, 4)
    assert output["observer"].tolist() == [0, 1, 2, 3, 3, 2, 1]
    assert output["birth_ids"][1, 1] == 1
    assert output["birth_ids"][2, 2] == 2
    assert output["birth_ids"][3, 3] == 3
    assert output["closure_ids"][4, 3] == 3
    assert output["closure_ids"][5, 2] == 2
    assert output["closure_ids"][6, 1] == 1
    assert output["closure_parent_ids"][4, 3] == 2
    assert output["closure_parent_ids"][5, 2] == 1
    assert output["closure_parent_ids"][6, 1] == 0
    assert output["closure_reasons"][4, 3] == 1
    assert output["closure_reasons"][5, 2] == 1
    assert output["closure_reasons"][6, 1] == 1
    assert output["active"][-1].tolist() == [True, False, False, False]


def test_ancestor_finalization_is_deepest_first_and_does_not_reopen():
    output = _route([0, 6, 3, 2, 0])
    assert output["observer"][-1] == 3
    assert output["closure_ids"][-1].tolist() == [-1, 1, 2, 3]
    assert output["closure_parent_ids"][-1].tolist() == [-1, 0, 1, 2]
    assert output["closure_reasons"][-1].tolist() == [0, 1, 2, 2]
    assert not output["births"][-1].any()
    assert output["active"][-1].tolist() == [True, False, False, False]

    simultaneous = _route([0, 6, 9, 10, -2])
    assert simultaneous["closure_reasons"][-1].tolist() == [0, 1, 1, 1]
    assert simultaneous["closure_ids"][-1].tolist() == [-1, 1, 2, 3]
    assert not simultaneous["births"][-1].any()


def test_random_paths_have_true_parent_containment_and_exactly_once_accounting():
    rng = np.random.default_rng(20260929)
    increments = rng.choice([-15.0, -9.0, -3.0, -1.0, 0.0, 1.0, 3.0, 9.0, 15.0], size=1200)
    z = np.r_[0.0, np.cumsum(increments)]
    output = _route(z)
    births = {}
    closures = {}
    for t in range(len(z)):
        live = output["ids"][t]
        parents = output["parent_ids"][t]
        assert output["active"][t, 0]
        assert np.array_equal(output["active"][t], np.arange(4) <= output["active"][t].sum() - 1)
        for depth in range(1, 4):
            if output["active"][t, depth]:
                assert output["active"][t, depth - 1]
                assert parents[depth] == live[depth - 1]
            birth_id = int(output["birth_ids"][t, depth])
            if birth_id > 0:
                assert birth_id not in births
                births[birth_id] = (t, int(parents[depth]))
            closure_id = int(output["closure_ids"][t, depth])
            if closure_id > 0:
                assert closure_id not in closures
                closures[closure_id] = (t, int(output["closure_parent_ids"][t, depth]))
    active_ids = {int(v) for v in output["ids"][-1, 1:] if v > 0}
    assert set(births) == set(closures) | active_ids
    assert not (set(closures) & active_ids)
    assert len(births) > 50  # exercise repeated slot reuse
    for event_id, (close_t, parent_id) in closures.items():
        assert births[event_id][1] == parent_id
        if parent_id:
            assert births[parent_id][0] < births[event_id][0]
            assert close_t <= closures[parent_id][0] if parent_id in closures else parent_id in active_ids


def test_mirror_invariance_on_random_jumps_and_atomic_births():
    rng = np.random.default_rng(41)
    z = np.r_[0.0, np.cumsum(rng.choice([-20., -7., -1., 0., 1., 7., 20.], 400))]
    positive = _route(z)
    negative = _route(-z)
    for key in ["births", "closures", "active", "observer", "ids", "parent_ids",
                "birth_ids", "closure_ids", "closure_parent_ids", "closure_reasons"]:
        np.testing.assert_array_equal(positive[key], negative[key])
    assert np.all(positive["births"].sum(axis=1) <= 1)
    np.testing.assert_allclose(positive["metadata"][:, :, [0, 2, 5]],
                               -negative["metadata"][:, :, [0, 2, 5]], atol=1e-5)
    np.testing.assert_allclose(positive["metadata"][:, :, [1, 3, 4]],
                               negative["metadata"][:, :, [1, 3, 4]], atol=1e-5)
    jump = _route([0, 20, 20, 20])
    assert _event_ids(jump, "birth_ids") == [1]
    assert not jump["closures"].any()


def test_chunking_matches_full_run_and_does_not_flush_open_events():
    z = np.array([0, 6, 9, 10, 9, 6, 0, 6, 3, 2, 0], dtype=float)
    full_router = TemporalEventRouter(THRESHOLDS)
    full = full_router.process(z)
    chunks = [z[:4], z[4:7], z[7:8], z[8:]]
    streamed = TemporalEventRouter(THRESHOLDS)
    parts = []
    for chunk in chunks:
        parts.append(streamed.process(chunk))
        if len(parts) == 1:
            assert parts[0]["active"][-1].tolist() == [True, True, True, True]
            assert not parts[0]["closures"][-1].any()
    for key in full:
        np.testing.assert_array_equal(full[key], np.concatenate([p[key] for p in parts], axis=0))
    assert streamed.state_dict() == full_router.state_dict()

    resumed = TemporalEventRouter(THRESHOLDS)
    first = resumed.process(z[:4])
    snapshot = resumed.state_dict()
    resumed = TemporalEventRouter(THRESHOLDS)
    resumed.load_state_dict(snapshot)
    rest = resumed.process(z[4:])
    for key in full:
        np.testing.assert_array_equal(full[key], np.concatenate([first[key], rest[key]], axis=0))
    assert resumed.state_dict() == full_router.state_dict()

    matrix = route_windows_v2(z[None, :], THRESHOLDS)
    for key in full:
        np.testing.assert_array_equal(matrix[key][0], full[key])


def test_bad_thresholds_rejected():
    with pytest.raises(ValueError):
        TemporalEventRouter([1, 3, 6])  # API takes coarse to fine
    with pytest.raises(ValueError):
        TemporalEventRouter([6, 3, 0])


def test_operational_state_stays_bounded_without_event_ledger():
    router = TemporalEventRouter(THRESHOLDS)
    path = np.tile([0.0, 6.0], 500)
    router.process(path)
    assert router.born_count > 100
    assert router.terminated_count + len(router.nodes) - 1 == router.born_count
    assert router.transfer_count == router.terminated_count
    assert len(router.nodes) <= 4
    assert router.records == []
    saved = router.state_dict()
    assert "records" not in saved
    assert len(saved["nodes"]) <= 4
