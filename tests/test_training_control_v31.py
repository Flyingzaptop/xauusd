import pytest

from src.training_control_v31 import EarlyStop


def test_exact_best_and_short_best_are_distinct_from_patience_reference():
    control = EarlyStop()
    losses = [0.9, 0.8, 0.7, 0.75, 0.699995, 0.699991, 0.699989]
    decisions = [control.update(loss, epoch) for epoch, loss in enumerate(losses, 1)]
    assert decisions[3]["save_short_best"]
    assert decisions[3]["short_best_epoch"] == 3
    assert decisions[3]["reference_loss"] == pytest.approx(0.7)
    assert decisions[4]["save_full_best"] and not decisions[4]["significant_improvement"]
    assert decisions[5]["save_full_best"] and not decisions[5]["significant_improvement"]
    assert decisions[6]["save_full_best"] and decisions[6]["significant_improvement"]
    assert decisions[6]["bad_epochs"] == 0
    assert control.best_epoch == 7
    assert control.short_best_epoch == 3
    tie = control.update(losses[-1], 8)
    assert not tie["save_full_best"]
    assert control.best_epoch == 7


def test_earliest_patience_stop_is_epoch_nine():
    control = EarlyStop()
    decisions = [control.update(loss, epoch) for epoch, loss in
                 enumerate([0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], 1)]
    assert all(not item["stop"] for item in decisions[:8])
    assert decisions[8]["stop"] and decisions[8]["stop_reason"] == "patience"
    assert decisions[8]["bad_epochs"] == 5
    with pytest.raises(RuntimeError):
        control.update(0.5, 10)


def test_budget_cap_can_stop_continued_improvement_without_convergence_claim():
    control = EarlyStop()
    decisions = [control.update(1 - 0.01 * epoch, epoch) for epoch in range(1, 21)]
    assert all(not item["stop"] for item in decisions[:19])
    assert decisions[-1]["stop"] and decisions[-1]["stop_reason"] == "budget_cap"
    assert decisions[-1]["significant_improvement"]
    assert decisions[-1]["best_epoch"] == 20
    assert control.state_dict()["stop_reason"] == "budget_cap"


def test_resume_reproduces_same_decisions_and_rejects_policy_drift():
    losses = [0.9, 0.8, 0.7, 0.65, 0.649999, 0.649998, 0.649997, 0.649996, 0.649995]
    uninterrupted = EarlyStop()
    full = [uninterrupted.update(loss, epoch) for epoch, loss in enumerate(losses, 1)]
    partial = EarlyStop()
    for epoch, loss in enumerate(losses[:6], 1):
        partial.update(loss, epoch)
    state = partial.state_dict()
    resumed = EarlyStop()
    resumed.load_state_dict(state)
    rest = [resumed.update(loss, epoch) for epoch, loss in enumerate(losses[6:], 7)]
    assert rest == full[6:]
    assert resumed.state_dict() == uninterrupted.state_dict()
    with pytest.raises(ValueError):
        EarlyStop(max_epochs=21).load_state_dict(state)


def test_invalid_epoch_and_nonfinite_loss_rejected():
    control = EarlyStop()
    with pytest.raises(ValueError):
        control.update(float("nan"), 1)
    with pytest.raises(ValueError):
        control.update(0.8, 2)
    control.update(0.8, 1)
    with pytest.raises(ValueError):
        control.update(0.7, 1)
