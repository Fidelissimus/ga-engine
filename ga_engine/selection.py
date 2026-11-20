"""Parent selection: tournament, roulette, rank and stochastic universal sampling.

:class:`ParentSelector` is stateless apart from its configuration. Callers pass
the *quality information* of the current population as arrays, so the selector
needs no knowledge of individuals, constraints or objectives:

* single-objective: ``scores`` (penalised fitness, lower is better);
* multi-objective: ``ranks`` and ``crowding`` (NSGA-II metadata).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .enums import SelectionMethod
from .errors import ConfigurationError


class ParentSelector:
    """Draws parent indices according to a :class:`SelectionMethod`.

    Args:
        method: Selection strategy.
        tournament_size: Contestants per tournament (capped at the population size).

    Raises:
        ConfigurationError: If ``tournament_size`` is smaller than 1.
    """

    def __init__(self, method: SelectionMethod, tournament_size: int = 3) -> None:
        if tournament_size < 1:
            raise ConfigurationError("tournament_size must be >= 1")
        self.method = method
        self.tournament_size = tournament_size

    def select(
        self,
        rng: np.random.Generator,
        count: int,
        *,
        scores: Optional[np.ndarray] = None,
        ranks: Optional[np.ndarray] = None,
        crowding: Optional[np.ndarray] = None,
        distinct: bool = False,
    ) -> List[int]:
        """Select ``count`` parent indices.

        Args:
            rng: Random generator (the only source of randomness).
            count: Number of indices to draw.
            scores: Penalised scalar fitness per individual (single-objective).
            ranks: NSGA-II rank per individual (multi-objective).
            crowding: NSGA-II crowding distance per individual (multi-objective).
            distinct: If ``True`` the returned indices are all different, which
                needs ``count <= population size``. Otherwise selection is
                *with replacement*, as required for selection pressure.

        Returns:
            Selected indices into the population.

        Raises:
            ConfigurationError: If neither ``scores`` nor ``ranks``/``crowding``
                are given, or ``distinct`` is impossible.
        """
        if scores is None and (ranks is None or crowding is None):
            raise ConfigurationError("provide either scores or both ranks and crowding")
        size = len(scores) if scores is not None else len(ranks)  # type: ignore[arg-type]
        if distinct and count > size:
            raise ConfigurationError("cannot select more distinct parents than individuals")
        if count <= 0:
            return []

        if self.method == SelectionMethod.TOURNAMENT:
            quality = self._quality(scores, ranks, crowding)
            return self._tournament(rng, count, quality, distinct)

        weights = self._weights(scores, ranks, crowding)
        if distinct:
            return [int(i) for i in rng.choice(size, size=count, replace=False, p=weights)]
        if self.method == SelectionMethod.SUS:
            return self._sus(rng, count, weights)
        return [int(i) for i in rng.choice(size, size=count, replace=True, p=weights)]

    # -- quality ordering (tournament) -----------------------------------

    @staticmethod
    def _quality(
        scores: Optional[np.ndarray],
        ranks: Optional[np.ndarray],
        crowding: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return a lower-is-better key that totally orders the population."""
        if scores is not None:
            return np.asarray(scores, dtype=float)
        order = np.lexsort((-np.asarray(crowding, dtype=float), np.asarray(ranks, dtype=float)))
        quality = np.empty(len(order))
        quality[order] = np.arange(len(order), dtype=float)
        return quality

    def _tournament(
        self, rng: np.random.Generator, count: int, quality: np.ndarray, distinct: bool
    ) -> List[int]:
        """Run ``count`` tournaments; contestants of one tournament are distinct."""
        size = len(quality)
        contestants = min(self.tournament_size, size)
        chosen: List[int] = []
        available = np.arange(size)
        for _ in range(count):
            if distinct:
                take = min(contestants, len(available))
                picks = available[rng.choice(len(available), size=take, replace=False)]
            else:
                picks = rng.choice(size, size=contestants, replace=False)
            winner = int(picks[int(np.argmin(quality[picks]))])
            chosen.append(winner)
            if distinct:
                available = available[available != winner]
        return chosen

    # -- selection weights (roulette / rank / SUS) -----------------------

    def _weights(
        self,
        scores: Optional[np.ndarray],
        ranks: Optional[np.ndarray],
        crowding: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return strictly positive selection probabilities summing to one."""
        use_rank = self.method == SelectionMethod.RANK
        if scores is None:
            return self._nsga_weights(
                np.asarray(ranks, dtype=float), np.asarray(crowding, dtype=float), use_rank
            )
        return self._scalar_weights(np.asarray(scores, dtype=float), use_rank)

    @staticmethod
    def _scalar_weights(values: np.ndarray, use_rank: bool) -> np.ndarray:
        """Weights for minimisation: better (lower) values get larger weights."""
        n = len(values)
        if use_rank:
            order = np.argsort(np.where(np.isfinite(values), values, np.inf), kind="stable")
            ranks = np.empty(n)
            ranks[order] = np.arange(n, dtype=float)
            weights = n - ranks
            return np.asarray(weights / weights.sum())
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.full(n, 1.0 / n)
        safe = np.where(finite, values, np.max(values[finite]))
        if np.max(safe) - np.min(safe) <= 0.0:
            return np.full(n, 1.0 / n)
        weights = np.max(safe) - safe + 1e-12
        return np.asarray(weights / weights.sum())

    @staticmethod
    def _nsga_weights(ranks: np.ndarray, crowding: np.ndarray, use_rank: bool) -> np.ndarray:
        """Weights from NSGA-II rank, mildly boosted by crowding distance."""
        n = len(ranks)
        if use_rank:
            weights = np.maximum(n - ranks, 1.0)
            return np.asarray(weights / weights.sum())
        finite = np.isfinite(crowding)
        if not np.any(finite):
            crowd_norm = np.ones(n)
        else:
            clipped = np.where(finite, crowding, np.max(crowding[finite]))
            low, high = float(np.min(clipped)), float(np.max(clipped))
            crowd_norm = np.ones(n) if high == low else (clipped - low) / (high - low)
        weights = np.maximum((1.0 / (1.0 + ranks)) * (1.0 + 0.25 * crowd_norm), 1e-15)
        return np.asarray(weights / weights.sum())

    @staticmethod
    def _sus(rng: np.random.Generator, count: int, weights: np.ndarray) -> List[int]:
        """Stochastic universal sampling.

        One random offset places ``count`` equally spaced pointers over the
        cumulative weights. The result is shuffled because the pointer order
        follows population order and would otherwise pair neighbours.
        """
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        step = 1.0 / count
        start = float(rng.uniform(0.0, step))
        pointers = np.minimum(start + step * np.arange(count), np.nextafter(1.0, 0.0))
        indices = np.minimum(np.searchsorted(cumulative, pointers, side="left"), len(weights) - 1)
        rng.shuffle(indices)
        return [int(i) for i in indices]


__all__ = ["ParentSelector"]
