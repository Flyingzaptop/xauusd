"""Premarket contract checks for the frozen synthetic runner; never call fit()."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.run_mechanism_v2 import Classifier, datasets, evaluate, matched_gru_width, parameter_count, subset
from src.synthetic_v2 import oracle_labels


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/mechanism-v2.json").read_text())


def _packed_path_keys(split):
    return {row.numpy().tobytes() for row in split["x"]}


def test_runner_datasets_are_reproducible_disjoint_and_oracle_labeled():
    first, counts = datasets("hard_closed", 7, CONFIG)
    second, repeated_counts = datasets("hard_closed", 7, CONFIG)
    assert counts == repeated_counts
    assert [counts[k]["n"] for k in ["train", "validation", "test"]] == [128, 64, 128]
    keys = [_packed_path_keys(first[name]) for name in ["train", "validation", "test"]]
    assert [len(group) for group in keys] == [128, 64, 128]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    for name in ["train", "validation", "test"]:
        split = first[name]
        assert torch.equal(split["x"], second[name]["x"])
        assert torch.equal(split["y"], second[name]["y"])
        assert 0 < counts[name]["positive"] < counts[name]["n"]
        recovered_z = np.cumsum(np.rint(split["x"][:, :, 0].numpy()).astype(np.int16), axis=1)
        np.testing.assert_array_equal(split["y"].numpy(), oracle_labels("hard_closed", recovered_z))


def test_random_labels_are_deterministic_and_tiny_is_explicit_memorization():
    hard, _ = datasets("hard_closed", 7, CONFIG)
    random_a, _ = datasets("random_hard", 7, CONFIG)
    random_b, _ = datasets("random_hard", 7, CONFIG)
    for name in ["train", "validation", "test"]:
        assert torch.equal(hard[name]["x"], random_a[name]["x"])
        assert torch.equal(random_a[name]["y"], random_b[name]["y"])
        assert not torch.equal(hard[name]["y"], random_a[name]["y"])
    tiny, _ = datasets("tiny_hard", 7, CONFIG)
    for name in ["validation", "test"]:
        assert torch.equal(tiny["train"]["x"], tiny[name]["x"])
        assert torch.equal(tiny["train"]["y"], tiny[name]["y"])


def test_all_classifier_inputs_have_same_batch_and_two_logit_outputs():
    data, _ = datasets("hard_closed", 7, CONFIG)
    batch = subset(data["train"], torch.arange(2))
    gru_width = matched_gru_width(CONFIG)
    counts = {}
    with torch.no_grad():
        for variant in ["gru", "v1", "tree", "no_transfer", "no_open"]:
            model = Classifier(variant, CONFIG, gru_width)
            assert model(batch).shape == (2, 2)
            counts[variant] = parameter_count(model)
    assert counts["tree"] == counts["no_transfer"] == counts["no_open"]
    assert abs(counts["gru"] - counts["tree"]) / counts["tree"] < 0.01


def test_balanced_accuracy_uses_equal_class_weight():
    class Fixed(nn.Module):
        def forward(self, batch):
            return torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]])

    score = evaluate(Fixed(), {"y": torch.tensor([0, 0, 0, 1])})
    assert score["accuracy"] == pytest.approx(0.75)
    assert score["balanced_accuracy"] == pytest.approx(0.5)
