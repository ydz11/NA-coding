from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """
    Simple early stopping helper.
    Mode: "max" (larger is better) for NDCG.
    """

    patience: int = 3
    mode: str = "max"

    best: float | None = None
    num_bad_epochs: int = 0

    def update(self, value: float) -> bool:
        """
        Update with the new metric value.
        Returns True if this is a new best.
        """
        if self.best is None:
            self.best = value
            self.num_bad_epochs = 0
            return True

        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.num_bad_epochs >= self.patience

