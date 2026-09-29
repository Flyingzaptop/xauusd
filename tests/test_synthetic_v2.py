import numpy as np
import pytest
from collections import defaultdict
from itertools import product

from src.synthetic_v2 import LENGTHS, make_dataset, oracle_labels, to_close


@pytest.mark.parametrize("task", ["tiny", "easy_closed", "medium_closed", "hard_closed", "hard_open"])
def test_generator_agrees_with_independent_path_oracle(task):
    size = 16 if task == "tiny" else 48
    z, labels, hashes = make_dataset(task, size, seed=121)
    assert z.shape == (size, LENGTHS[task])
    assert z.dtype == np.int16
    assert labels.dtype == np.int64
    assert len(set(hashes)) == size
    np.testing.assert_array_equal(oracle_labels(task, z), labels)
    assert set(np.unique(labels)) == {0, 1}
    assert np.all(z[:, 0] == 0)
    assert np.all(z[:, -1] == 0) if task != "hard_open" else np.all(z[:, -1] != 0)
    assert np.all(to_close(z) > 0)
    if task == "hard_open":
        assert np.all(np.diff(z, axis=1)[:, -1] != 0)  # no end padding
        assert np.all(np.abs(z[:, -1]) == 4)
        np.testing.assert_array_equal(z[:, -1] - z[:, -2], -np.sign(z[:, -1]))


@pytest.mark.parametrize("task", ["easy_closed", "medium_closed", "hard_closed", "hard_open"])
def test_requested_split_sizes_have_disjoint_observed_paths(task):
    train_z, _, train_hashes = make_dataset(task, 128, seed=31)
    val_z, _, val_hashes = make_dataset(task, 64, seed=32, exclude_hashes=train_hashes)
    test_z, _, test_hashes = make_dataset(task, 128, seed=33,
                                         exclude_hashes=[*train_hashes, *val_hashes])
    assert len(set(train_hashes + val_hashes + test_hashes)) == 320
    np.testing.assert_array_equal(oracle_labels(task, train_z), oracle_labels(task, train_z.copy()))
    assert train_z.shape[1] == val_z.shape[1] == test_z.shape[1]


def test_oracle_rejects_broken_or_ambiguous_path():
    z, _, _ = make_dataset("easy_closed", 1, seed=9)
    broken = z.copy()
    broken[0, 10:] += 2  # introduces a three-step jump inconsistent with the grammar
    with pytest.raises(ValueError, match="possible oracle labels"):
        oracle_labels("easy_closed", broken)


def test_hard_open_terminal_pair_is_label_balanced_over_exhaustive_support():
    """All 327,696 possible observable prefixes have a nonleaking endpoint."""
    episodes = {
        (sign, order): tuple([sign] * 7 + [sign * order] * 3
                             + [-sign * order] * 6 + [sign * order] * 3
                             + [-sign] * 7)
        for sign, order in product((-1, 1), repeat=2)
    }
    terminal_counts = defaultdict(lambda: [0, 0])
    distinct = set()
    for p0 in range(11):
        for p1 in range(11 - p0):
            for directions in product((-1, 1), repeat=p0 + p1):
                left = tuple(step for direction in directions[:p0]
                             for step in (direction, -direction))
                middle = tuple(step for direction in directions[p0:]
                               for step in (direction, -direction))
                for first in episodes.values():
                    prefix = left + first + middle
                    for (sign, order), second in episodes.items():
                        path = prefix + second[:22]
                        label = int(order == 1)
                        terminal = (sum(path), path[-1])
                        assert terminal == (4 * sign, -sign)
                        terminal_counts[terminal][label] += 1
                        distinct.add(path)
    assert len(distinct) == 327_696
    assert dict(terminal_counts) == {(4, -1): [81_924, 81_924],
                                     (-4, 1): [81_924, 81_924]}
