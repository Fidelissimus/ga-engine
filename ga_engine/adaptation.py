"""Adaptive mutation control (1/5-success-rule style).

:class:`AdaptiveMutationController` owns the *mutation intensity* the engine
uses (individual probability, per-gene probability and Gaussian step size) and
the bookkeeping needed to adapt it. When adaptation is disabled it simply holds
the constants, so the engine always reads its mutation settings from one place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AdaptiveMutationController:
    """Tracks mutation success and adapts mutation intensity.

    A *trial* is one offspring that actually underwent mutation; it is a
    *success* when it is strictly better than the better of its two parents.
    Once at least ``window`` trials have accumulated, the success ratio is
    compared with 1/5 (Rechenberg's rule): above 20 % the search is too timid
    so steps grow, below 20 % it is too erratic so steps shrink. The window
    lets statistics accumulate over several generations, which matters for
    steady-state runs that create only a couple of offspring per step.

    Args:
        enabled: Whether adaptation is active.
        mutation_prob: Initial probability that an offspring is mutated.
        gene_mutation_prob: Initial per-gene mutation probability.
        sigma: Initial Gaussian step size as a fraction of the gene range.
        dimensions: Genome length (sets the floor of the per-gene probability).
        adapt_sigma: Whether ``sigma`` participates in adaptation (Gaussian only).
        window: Minimum number of trials before an adaptation step.
    """

    GROW = 1.05
    SHRINK = 0.95
    SIGMA_GROW = 1.10
    SIGMA_SHRINK = 0.90

    def __init__(
        self,
        *,
        enabled: bool,
        mutation_prob: float,
        gene_mutation_prob: float,
        sigma: float,
        dimensions: int,
        adapt_sigma: bool,
        window: int = 10,
    ) -> None:
        self.enabled = enabled
        self.adapt_sigma = adapt_sigma
        self.window = window
        self._initial = (mutation_prob, gene_mutation_prob, sigma)
        # Bounds never exclude the user's starting value (e.g. a 0 probability).
        self._prob_bounds = (min(0.001, mutation_prob), max(0.95, mutation_prob))
        floor = 1.0 / dimensions / 10.0
        self._gene_bounds = (min(floor, gene_mutation_prob), max(1.0, gene_mutation_prob))
        self._sigma_bounds = (min(1e-6, sigma), max(1.0, sigma))
        self.mutation_prob = mutation_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.sigma = sigma
        self.success_count = 0
        self.trial_count = 0
        self.last_success_ratio: Optional[float] = None

    def reset(self) -> None:
        """Restore the initial intensities and clear all success statistics."""
        self.mutation_prob, self.gene_mutation_prob, self.sigma = self._initial
        self.success_count = 0
        self.trial_count = 0
        self.last_success_ratio = None

    def record_trial(self, success: bool) -> None:
        """Record the outcome of one mutated offspring (no-op when disabled)."""
        if not self.enabled:
            return
        self.trial_count += 1
        if success:
            self.success_count += 1

    def update(self) -> None:
        """Adapt the intensities if enough trials have accumulated."""
        if not self.enabled or self.trial_count < self.window:
            return
        ratio = self.success_count / self.trial_count
        self.last_success_ratio = ratio
        if ratio > 0.20:
            factor, sigma_factor = self.GROW, self.SIGMA_GROW
        elif ratio < 0.20:
            factor, sigma_factor = self.SHRINK, self.SIGMA_SHRINK
        else:
            factor = sigma_factor = 1.0
        self.mutation_prob = _clamp(self.mutation_prob * factor, self._prob_bounds)
        self.gene_mutation_prob = _clamp(self.gene_mutation_prob * factor, self._gene_bounds)
        if self.adapt_sigma:
            self.sigma = _clamp(self.sigma * sigma_factor, self._sigma_bounds)
        self.trial_count = 0
        self.success_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return the mutable state for checkpointing."""
        return {
            "mutation_prob": self.mutation_prob,
            "gene_mutation_prob": self.gene_mutation_prob,
            "sigma": self.sigma,
            "success_count": self.success_count,
            "trial_count": self.trial_count,
            "last_success_ratio": self.last_success_ratio,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the state produced by :meth:`state_dict`."""
        self.mutation_prob = float(state["mutation_prob"])
        self.gene_mutation_prob = float(state["gene_mutation_prob"])
        self.sigma = float(state["sigma"])
        self.success_count = int(state["success_count"])
        self.trial_count = int(state["trial_count"])
        ratio = state["last_success_ratio"]
        self.last_success_ratio = None if ratio is None else float(ratio)


def _clamp(value: float, bounds: "tuple[float, float]") -> float:
    """Clamp ``value`` into ``[bounds[0], bounds[1]]``."""
    return max(bounds[0], min(bounds[1], value))
