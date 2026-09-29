import numpy as np
import pytest
import torch

from src.crossing_memory_v2 import TreeCrossingMemory, route_windows_v2
from src.stream_batch_v3 import PackedStream, TREE_KEYS


def _prepared_split():
    paths = [
        [0, 1, 2, 3, 2, 1, 0],
        [0, -1, -2, -3, -2],
        [0, 1, 2, 3, 4, 3, 2, 1, 0],
        [0, -1, -2],
        [0, 1, 2, 1],
    ]
    pieces = []
    clock = 1_700_000_000
    for path in paths:
        z = np.asarray(path, dtype=np.float32)
        change = np.diff(z, prepend=z[:1])
        x = np.stack((change, z, z * z / 10, np.ones_like(z)), axis=-1)
        valid = np.ones(len(z), dtype=np.bool_)
        valid[:2] = False
        valid[-1] = False
        y = (z > 0).astype(np.float32)
        y[~valid] = 0
        reset = np.zeros(len(z), dtype=np.bool_)
        reset[0] = True
        timestamp = clock + np.arange(len(z), dtype=np.int64) * 60
        clock = int(timestamp[-1]) + 180
        routed = route_windows_v2(z[None, :], [3, 2, 1])
        pieces.append(dict(x=x, y=y, valid=valid, reset=reset, timestamp=timestamp,
                           tree={key: value[0] for key, value in routed.items()}))
    split = {key: np.concatenate([piece[key] for piece in pieces])
             for key in ("x", "y", "valid", "reset", "timestamp")}
    split["tree"] = {key: np.concatenate([piece["tree"][key] for piece in pieces])
                     for key in pieces[0]["tree"]}
    return split


def test_all_minutes_once_with_valid_labels_padding_and_resets():
    split = _prepared_split()
    stream = PackedStream(split, batch_size=2, chunk_length=4)
    batches = list(stream)
    assert len(batches) == len(stream)
    assert stream.n_observations == len(split["x"])
    assert stream.n_valid_labels == int(split["valid"].sum())
    assert stream.segments == ((0, 7), (7, 12), (12, 21), (21, 24), (24, 28))
    assert np.array_equal(stream.lane_indices[0], np.r_[0:7, 12:21, 24:28])
    assert np.array_equal(stream.lane_indices[1], np.r_[7:12, 21:24])

    observed = []
    valid_observed = []
    for batch in batches:
        ids = batch["indices"]
        real = ids >= 0
        observed.extend(ids[real].tolist())
        valid_observed.extend(ids[batch["valid"]].tolist())
        np.testing.assert_array_equal(batch["reset"][real], split["reset"][ids[real]])
        np.testing.assert_array_equal(batch["valid"][real], split["valid"][ids[real]])
        np.testing.assert_array_equal(batch["x"][real], split["x"][ids[real]])
        np.testing.assert_array_equal(batch["y"][real], split["y"][ids[real]])
        assert np.all(batch["reset"][~real])
        assert not np.any(batch["valid"][~real])
        assert not np.any(batch["x"][~real])
        assert not np.any(batch["y"][~real])
        for key in TREE_KEYS:
            np.testing.assert_array_equal(batch["tree"][key][real], split["tree"][key][ids[real]])
            assert not np.any(batch["tree"][key][~real])
    np.testing.assert_array_equal(sorted(observed), np.arange(len(split["x"])))
    np.testing.assert_array_equal(sorted(valid_observed), np.flatnonzero(split["valid"]))
    assert batches[0]["reset"][:, 0].all()
    for lane in range(2):
        lane_ids = np.concatenate([batch["indices"][lane] for batch in batches])
        first_pad = np.flatnonzero(lane_ids == -1)
        if len(first_pad):
            assert np.all(lane_ids[first_pad[0]:] == -1)


def test_chunked_state_matches_serial_segments_without_cross_segment_leakage():
    split = _prepared_split()
    stream = PackedStream(split, batch_size=2, chunk_length=4)
    torch.manual_seed(23)
    model = TreeCrossingMemory(input_dim=4, hidden_dim=5, depths=4)
    model.eval()
    serial = {}
    with torch.no_grad():
        for start, stop in stream.segments:
            x = torch.from_numpy(split["x"][start:stop][None])
            tree = {key: torch.from_numpy(split["tree"][key][start:stop][None]) for key in TREE_KEYS}
            output, _ = model(x, tree)
            for offset, index in enumerate(range(start, stop)):
                serial[index] = output[0, offset].clone()

        state = model.initial_state(stream.batch_size, torch.device("cpu"), torch.float32)
        packed = {}
        for batch in stream:
            tree = {key: torch.from_numpy(value) for key, value in batch["tree"].items()}
            x = torch.from_numpy(batch["x"])
            for time in range(stream.chunk_length):
                reset = torch.from_numpy(batch["reset"][:, time])
                state = torch.where(reset[:, None, None], torch.zeros_like(state), state)
                tick = {key: value[:, time] for key, value in tree.items()}
                output, state = model.step(x[:, time], tick, state)
                for lane, index in enumerate(batch["indices"][:, time]):
                    if index >= 0:
                        packed[int(index)] = output[lane].clone()
    assert packed.keys() == serial.keys()
    for index in serial:
        torch.testing.assert_close(packed[index], serial[index], rtol=1e-5, atol=1e-6)


def test_missing_gap_reset_and_nonmonotonic_time_are_rejected():
    split = _prepared_split()
    missing = dict(split, reset=split["reset"].copy())
    missing["reset"][7] = False
    with pytest.raises(ValueError, match="lacks reset"):
        PackedStream(missing, 2, 4)
    reversed_time = dict(split, timestamp=split["timestamp"].copy())
    reversed_time["timestamp"][7] = reversed_time["timestamp"][6]
    with pytest.raises(ValueError, match="increase strictly"):
        PackedStream(reversed_time, 2, 4)


def test_future_values_do_not_change_packed_prefix():
    split = _prepared_split()
    changed = dict(split, x=split["x"].copy(), tree={k: v.copy() for k, v in split["tree"].items()})
    changed["x"][20:] *= 7
    changed["tree"]["metadata"][20:] *= 3
    before = list(PackedStream(split, 2, 4))
    after = list(PackedStream(changed, 2, 4))
    for first, second in zip(before, after):
        prefix = (first["indices"] >= 0) & (first["indices"] < 20)
        np.testing.assert_array_equal(first["indices"], second["indices"])
        np.testing.assert_array_equal(first["x"][prefix], second["x"][prefix])
        np.testing.assert_array_equal(first["tree"]["metadata"][prefix],
                                      second["tree"]["metadata"][prefix])
