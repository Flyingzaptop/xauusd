"""Deterministic v3.1 epoch stopping and checkpoint bookkeeping.

The exact minimum selects checkpoints; a separate min-delta reference controls
patience. This module does not save weights and does not claim convergence.
"""

from __future__ import annotations

import math
from typing import Any


class EarlyStop:
    """Sequential epoch controller with minimum-four, maximum-twenty policy."""

    def __init__(
        self,
        min_epochs: int = 4,
        max_epochs: int = 20,
        min_delta: float = 1e-5,
        patience: int = 5,
    ) -> None:
        if not (isinstance(min_epochs, int) and isinstance(max_epochs, int)
                and isinstance(patience, int)):
            raise TypeError("epoch limits and patience must be integers")
        if min_epochs < 1 or max_epochs < min_epochs or patience < 1:
            raise ValueError("require 1 <= min_epochs <= max_epochs and patience >= 1")
        if not math.isfinite(min_delta) or min_delta <= 0:
            raise ValueError("min_delta must be positive and finite")
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_delta = float(min_delta)
        self.patience = patience
        self.last_epoch = 0
        self.best_epoch: int | None = None
        self.best_loss = math.inf
        self.short_best_epoch: int | None = None
        self.short_best_loss: float | None = None
        self.reference_loss: float | None = None
        self.bad_epochs = 0
        self.should_stop = False
        self.stop_reason: str | None = None

    def update(self, loss: float, epoch: int) -> dict[str, Any]:
        """Record a finite selection loss for the next 1-based epoch."""
        if self.should_stop:
            raise RuntimeError("Cannot update an already stopped controller")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != self.last_epoch + 1:
            raise ValueError(f"Expected sequential epoch {self.last_epoch + 1}")
        measured = float(loss)
        if not math.isfinite(measured):
            raise ValueError("Selection loss must be finite")
        exact_improvement = measured < self.best_loss
        if exact_improvement:
            self.best_loss = measured
            self.best_epoch = epoch
        significant_improvement = False
        save_short_best = False
        if epoch == self.min_epochs:
            self.short_best_epoch = self.best_epoch
            self.short_best_loss = self.best_loss
            self.reference_loss = self.best_loss
            self.bad_epochs = 0
            save_short_best = True
        elif epoch > self.min_epochs:
            assert self.reference_loss is not None
            significant_improvement = measured <= self.reference_loss - self.min_delta
            if significant_improvement:
                self.reference_loss = measured
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
        self.last_epoch = epoch
        patience_exhausted = epoch > self.min_epochs and self.bad_epochs >= self.patience
        budget_cap_reached = epoch >= self.max_epochs
        self.should_stop = patience_exhausted or budget_cap_reached
        self.stop_reason = "patience" if patience_exhausted else "budget_cap" if budget_cap_reached else None
        return {
            "epoch": epoch,
            "selection_loss": measured,
            "save_full_best": exact_improvement,
            "save_short_best": save_short_best,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "short_best_epoch": self.short_best_epoch,
            "short_best_loss": self.short_best_loss,
            "reference_loss": self.reference_loss,
            "significant_improvement": significant_improvement,
            "bad_epochs": self.bad_epochs,
            "patience_exhausted": patience_exhausted,
            "budget_cap_reached": budget_cap_reached,
            "stop": self.should_stop,
            "stop_reason": self.stop_reason,
        }

    def state_dict(self) -> dict[str, Any]:
        """JSON-compatible state for exact interruption/resume bookkeeping."""
        return {
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "last_epoch": self.last_epoch,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss if self.best_epoch is not None else None,
            "short_best_epoch": self.short_best_epoch,
            "short_best_loss": self.short_best_loss,
            "reference_loss": self.reference_loss,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a saved state after validating its frozen policy and bounds."""
        policy = ("min_epochs", "max_epochs", "min_delta", "patience")
        if any(state.get(name) != getattr(self, name) for name in policy):
            raise ValueError("Saved early-stop policy differs from current policy")
        epoch = state.get("last_epoch")
        best_epoch = state.get("best_epoch")
        best_loss = state.get("best_loss")
        if not isinstance(epoch, int) or not 0 <= epoch <= self.max_epochs:
            raise ValueError("Saved last_epoch is invalid")
        if epoch == 0:
            if best_epoch is not None or best_loss is not None:
                raise ValueError("Empty history cannot have a best checkpoint")
        elif not isinstance(best_epoch, int) or not 1 <= best_epoch <= epoch \
                or best_loss is None or not math.isfinite(float(best_loss)):
            raise ValueError("Saved best checkpoint is invalid")
        short_epoch = state.get("short_best_epoch")
        short_loss = state.get("short_best_loss")
        reference = state.get("reference_loss")
        if epoch < self.min_epochs:
            if short_epoch is not None or short_loss is not None or reference is not None:
                raise ValueError("Patience reference exists before minimum epoch")
        elif not isinstance(short_epoch, int) or not 1 <= short_epoch <= self.min_epochs \
                or short_loss is None or not math.isfinite(float(short_loss)) \
                or reference is None or not math.isfinite(float(reference)):
            raise ValueError("Saved short-best/reference is invalid")
        bad = state.get("bad_epochs")
        if not isinstance(bad, int) or bad < 0 or bad > self.patience:
            raise ValueError("Saved bad-epoch count is invalid")
        stopped = state.get("should_stop")
        reason = state.get("stop_reason")
        if not isinstance(stopped, bool) or reason not in (None, "patience", "budget_cap"):
            raise ValueError("Saved stop status is invalid")
        if stopped != (reason is not None):
            raise ValueError("Saved stop flag/reason disagree")
        if reason == "patience" and bad < self.patience:
            raise ValueError("Patience stop lacks exhausted counter")
        if reason == "budget_cap" and epoch != self.max_epochs:
            raise ValueError("Budget stop is not at maximum epoch")
        self.last_epoch = epoch
        self.best_epoch = best_epoch
        self.best_loss = float(best_loss) if best_loss is not None else math.inf
        self.short_best_epoch = short_epoch
        self.short_best_loss = float(short_loss) if short_loss is not None else None
        self.reference_loss = float(reference) if reference is not None else None
        self.bad_epochs = bad
        self.should_stop = stopped
        self.stop_reason = reason
