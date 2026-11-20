"""Continuous real-valued representation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..enums import CrossoverMethod, MutationMethod
from ..errors import ConfigurationError
from ._bounds import Bounds, parse_bounds
from ._exchange import single_point_exchange, two_point_exchange, uniform_exchange
from .base import Representation, get_param

class RealRepresentation(Representation):
    """Bounded real-valued vector.

    Crossover: single-point, two-point, uniform, SBX (default) and BLX-alpha.
    Mutation: Gaussian (default) and polynomial.

    Operator parameters (passed through the engine's ``crossover_params`` /
    ``mutation_params``): ``eta`` (SBX distribution index, default 20),
    ``alpha`` (BLX-alpha expansion, default 0.5), ``probability`` (uniform
    crossover, default 0.5) and ``eta_m`` (polynomial mutation index,
    default 20).

    Args:
        dimensions: Number of genes.
        bounds: Either one finite ``(lower, upper)`` pair shared by every gene
            (e.g. ``(-5.0, 5.0)``), or a sequence of ``dimensions`` such pairs
            (e.g. ``[(0, 1), (-10, 10), (100, 500)]``). Each needs
            ``lower < upper``.

    Attributes:
        lower_bounds: Per-gene lower bounds (array of length ``dimensions``).
        upper_bounds: Per-gene upper bounds.

    Raises:
        ConfigurationError: On invalid ``dimensions`` or ``bounds``.
    """

    GENOME_DTYPE = np.dtype(np.float64)
    SUPPORTED_CROSSOVERS = frozenset(
        {
            CrossoverMethod.SINGLE_POINT,
            CrossoverMethod.TWO_POINT,
            CrossoverMethod.UNIFORM,
            CrossoverMethod.SBX,
            CrossoverMethod.BLX_ALPHA,
        }
    )
    SUPPORTED_MUTATIONS = frozenset({MutationMethod.GAUSSIAN, MutationMethod.POLYNOMIAL})
    DEFAULT_CROSSOVER = CrossoverMethod.SBX
    DEFAULT_MUTATION = MutationMethod.GAUSSIAN
    CROSSOVER_PARAMS = {
        CrossoverMethod.SINGLE_POINT: frozenset(),
        CrossoverMethod.TWO_POINT: frozenset(),
        CrossoverMethod.UNIFORM: frozenset({"probability"}),
        CrossoverMethod.SBX: frozenset({"eta"}),
        CrossoverMethod.BLX_ALPHA: frozenset({"alpha"}),
    }
    MUTATION_PARAMS = {
        MutationMethod.GAUSSIAN: frozenset(),
        MutationMethod.POLYNOMIAL: frozenset({"eta_m"}),
    }

    def __init__(self, dimensions: int, bounds: Bounds = (-1.0, 1.0)):
        super().__init__(dimensions, None)
        self.lower_bounds, self.upper_bounds = parse_bounds(self.dimensions, bounds)
        self._span = self.upper_bounds - self.lower_bounds
        self._midpoint = 0.5 * (self.lower_bounds + self.upper_bounds)
        self.bounds = bounds

    def describe(self) -> Dict[str, Any]:
        """Describe type, dimensions and bounds (a pair, or one pair per gene)."""
        uniform = bool(
            np.all(self.lower_bounds == self.lower_bounds[0])
            and np.all(self.upper_bounds == self.upper_bounds[0])
        )
        bounds: Any
        if uniform:
            bounds = [float(self.lower_bounds[0]), float(self.upper_bounds[0])]
        else:
            bounds = [[float(lo), float(hi)] for lo, hi in zip(self.lower_bounds, self.upper_bounds)]
        return {"type": type(self).__name__, "dimensions": self.dimensions, "bounds": bounds}

    # -- creation / repair -------------------------------------------------

    def random_individual(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a genome uniformly from the bounds."""
        return np.asarray(rng.uniform(self.lower_bounds, self.upper_bounds))

    def repair(self, individual: np.ndarray) -> np.ndarray:
        """Clip to the bounds; NaN becomes the midpoint, ±inf the nearest bound."""
        arr = np.asarray(individual, dtype=float)
        arr = np.where(np.isnan(arr), self._midpoint, arr)
        return np.asarray(np.clip(arr, self.lower_bounds, self.upper_bounds))

    def validate(self, individual: np.ndarray) -> bool:
        """Return whether the genome is 1-D, right-sized, finite and within bounds."""
        arr = np.asarray(individual)
        return bool(
            arr.ndim == 1
            and len(arr) == self.dimensions
            and np.all(np.isfinite(arr))
            and np.all(arr >= self.lower_bounds)
            and np.all(arr <= self.upper_bounds)
        )

    def absolute_sigma(self, fraction: float) -> Any:
        """Scale a relative Gaussian step by each gene's range (returns an array)."""
        return fraction * self._span

    def diversity(self, genes: np.ndarray) -> float:
        """Mean per-gene standard deviation relative to the gene range."""
        if len(genes) <= 1:
            return 0.0
        return float(np.mean(np.std(genes.astype(float), axis=0) / self._span))

    # -- crossover ----------------------------------------------------------

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        method: CrossoverMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recombine two parents.

        Single-point and two-point crossover *exchange* gene segments (the
        textbook operators); they do not blend values.

        Raises:
            ConfigurationError: For an unsupported method or invalid parameter.
        """
        if method == CrossoverMethod.SBX:
            return self._sbx(parent1, parent2, rng, get_param(kwargs, "eta", 20.0))
        if method == CrossoverMethod.BLX_ALPHA:
            return self._blx_alpha(parent1, parent2, rng, get_param(kwargs, "alpha", 0.5))
        if method == CrossoverMethod.UNIFORM:
            return uniform_exchange(parent1, parent2, rng, get_param(kwargs, "probability", 0.5))
        if method == CrossoverMethod.SINGLE_POINT:
            return single_point_exchange(parent1, parent2, rng)
        if method == CrossoverMethod.TWO_POINT:
            return two_point_exchange(parent1, parent2, rng)
        raise ConfigurationError(f"Unsupported crossover for RealRepresentation: {method}")

    def _sbx(
        self, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, eta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover with distribution index ``eta``."""
        u = rng.random(p1.shape)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
        )
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        return (
            np.clip(c1, self.lower_bounds, self.upper_bounds),
            np.clip(c2, self.lower_bounds, self.upper_bounds),
        )

    def _blx_alpha(
        self, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BLX-alpha: sample each gene from the parents' interval widened by ``alpha``."""
        spread = alpha * np.abs(p1 - p2)
        low = np.minimum(p1, p2) - spread
        high = np.maximum(p1, p2) + spread
        return (
            np.clip(rng.uniform(low, high), self.lower_bounds, self.upper_bounds),
            np.clip(rng.uniform(low, high), self.lower_bounds, self.upper_bounds),
        )

    # -- mutation -----------------------------------------------------------

    def mutate(
        self,
        individual: np.ndarray,
        method: MutationMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """Mutate a copy of ``individual``.

        Keyword Args:
            mutation_prob: Per-gene mutation probability (default ``1/dimensions``).
            sigma: Absolute Gaussian standard deviation, a number or one value
                per gene (default 10 % of each gene's range).
            eta_m: Polynomial-mutation distribution index (default 20).

        Raises:
            ConfigurationError: For an unsupported method or invalid parameter.
        """
        probability = get_param(kwargs, "mutation_prob", 1.0 / self.dimensions)
        if method == MutationMethod.GAUSSIAN:
            sigma = self._sigma_per_gene(kwargs.get("sigma"))
            mutated: np.ndarray = np.asarray(individual, dtype=float).copy()
            mask = rng.random(self.dimensions) < probability
            count = int(np.sum(mask))
            if count:
                mutated[mask] += rng.normal(0.0, sigma[mask])
            return np.asarray(np.clip(mutated, self.lower_bounds, self.upper_bounds))
        if method == MutationMethod.POLYNOMIAL:
            return self._polynomial_mutation(
                individual, rng, get_param(kwargs, "eta_m", 20.0), probability
            )
        raise ConfigurationError(f"Unsupported mutation for RealRepresentation: {method}")

    def _sigma_per_gene(self, sigma: Any) -> np.ndarray:
        """Validate a Gaussian step (scalar or per-gene) and broadcast it to every gene."""
        values = self.absolute_sigma(0.1) if sigma is None else sigma
        array = np.broadcast_to(np.asarray(values, dtype=float), (self.dimensions,))
        if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
            raise ConfigurationError("parameter 'sigma' must be finite and >= 0")
        return array

    def _polynomial_mutation(
        self, x: np.ndarray, rng: np.random.Generator, eta_m: float, probability: float
    ) -> np.ndarray:
        """Deb's bounded polynomial mutation."""
        xl, xu, span = self.lower_bounds, self.upper_bounds, self._span
        x = np.asarray(x, dtype=float)
        mutated: np.ndarray = x.copy()
        mutate_mask = rng.random(x.shape) < probability
        if not np.any(mutate_mask):
            return mutated

        u = rng.random(x.shape)
        delta1 = (x - xl) / span
        delta2 = (xu - x) / span
        exponent = 1.0 / (eta_m + 1.0)
        delta_q = np.zeros_like(x)

        low = mutate_mask & (u <= 0.5)
        delta_q[low] = (
            2.0 * u[low] + (1.0 - 2.0 * u[low]) * (1.0 - delta1[low]) ** (eta_m + 1.0)
        ) ** exponent - 1.0

        high = mutate_mask & (u > 0.5)
        delta_q[high] = 1.0 - (
            2.0 * (1.0 - u[high])
            + 2.0 * (u[high] - 0.5) * (1.0 - delta2[high]) ** (eta_m + 1.0)
        ) ** exponent

        mutated += delta_q * span
        return np.asarray(np.clip(mutated, xl, xu))
