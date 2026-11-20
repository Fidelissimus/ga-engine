"""Stopping criteria shared by the single-population and island engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .errors import ConfigurationError


class StopReason(str, Enum):
    """Why a run ended. Members compare equal to their string values."""

    TARGET_FITNESS = "target_fitness"
    STAGNATION = "stagnation"
    MAX_GENERATIONS = "max_generations"
    CALLBACK = "callback"


@dataclass(frozen=True)
class StopCriteria:
    """The conditions that end a run.

    Args:
        max_generations: Generation budget (``>= 0``).
        stagnation_generations: Stop after this many generations without
            improvement (``>= 1``), or ``None``.
        target_fitness: Stop once a feasible solution at least this good
            exists, or ``None``. Single-objective only.

    Raises:
        ConfigurationError: If a value is out of range.
    """

    max_generations: int = 500
    stagnation_generations: Optional[int] = None
    target_fitness: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.max_generations, bool) or self.max_generations < 0:
            raise ConfigurationError("max_generations must be >= 0")
        if self.stagnation_generations is not None and self.stagnation_generations < 1:
            raise ConfigurationError("stagnation_generations must be >= 1 or None")

    def require_single_objective(self, multi_objective: bool) -> None:
        """Reject a target fitness in multi-objective mode.

        Raises:
            ConfigurationError: If ``target_fitness`` is set and ``multi_objective`` is true.
        """
        if self.target_fitness is not None and multi_objective:
            raise ConfigurationError(
                "target_fitness is supported only for single-objective optimization"
            )

    def check(
        self, *, generation: int, stagnation_count: int, target_reached: bool
    ) -> Optional[StopReason]:
        """Return the reason to stop now, or ``None`` to continue.

        Criteria are examined in this order: target, stagnation, budget.
        """
        if self.target_fitness is not None and target_reached:
            return StopReason.TARGET_FITNESS
        if (
            self.stagnation_generations is not None
            and stagnation_count >= self.stagnation_generations
        ):
            return StopReason.STAGNATION
        if generation >= self.max_generations:
            return StopReason.MAX_GENERATIONS
        return None
