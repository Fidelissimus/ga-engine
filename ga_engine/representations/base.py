"""Abstract representation interface and shared operator-parameter helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, FrozenSet, Mapping, Optional, Tuple

import numpy as np

from ..enums import CrossoverMethod, MutationMethod
from ..errors import ConfigurationError

# Value rules for the operator parameters the built-in operators understand.
# Unknown names (e.g. from custom representations) are not constrained.
_PARAM_RULES: Dict[str, Tuple[str, Callable[[float], bool]]] = {
    "eta": ("> 0", lambda v: v > 0.0),
    "eta_m": ("> 0", lambda v: v > 0.0),
    "alpha": (">= 0", lambda v: v >= 0.0),
    "probability": ("in [0, 1]", lambda v: 0.0 <= v <= 1.0),
    "mutation_prob": ("in [0, 1]", lambda v: 0.0 <= v <= 1.0),
    "sigma": (">= 0 and finite", lambda v: v >= 0.0 and math.isfinite(v)),
    "max_step": (">= 1", lambda v: v >= 1.0),
}


def get_param(params: Mapping[str, Any], name: str, default: float) -> float:
    """Read one numeric operator parameter and validate its value.

    Args:
        params: Mapping of operator keyword arguments.
        name: Parameter name, e.g. ``"eta"``.
        default: Value used when ``name`` is absent.

    Returns:
        The parameter as a ``float``.

    Raises:
        ConfigurationError: If the value violates the rule for ``name``
            (NaN always violates it).
    """
    value = float(params.get(name, default))
    rule = _PARAM_RULES.get(name)
    if rule is not None and not rule[1](value):
        raise ConfigurationError(f"parameter {name!r} must be {rule[0]}, got {value!r}")
    return value


class Representation(ABC):
    """Interface between the engine and a genome encoding.

    Subclasses implement genome creation, recombination, mutation and repair.
    They may also declare capability metadata (the ``SUPPORTED_*``,
    ``DEFAULT_*`` and ``*_PARAMS`` class variables) so the engine can reject an
    unsupported operator at construction time instead of mid-run, and pick
    sensible default operators. Leaving the metadata as ``None`` opts out of
    those checks, so a minimal custom subclass works without declaring any
    metadata.

    Attributes:
        dimensions: Genome length.
        bounds: Optional ``(lower, upper)`` gene bounds.
    """

    #: NumPy dtype the engine coerces valid genomes to (``None`` = leave as is).
    GENOME_DTYPE: ClassVar[Optional[np.dtype]] = None
    #: Crossover methods this representation implements (``None`` = unchecked).
    SUPPORTED_CROSSOVERS: ClassVar[Optional[FrozenSet[CrossoverMethod]]] = None
    #: Mutation methods this representation implements (``None`` = unchecked).
    SUPPORTED_MUTATIONS: ClassVar[Optional[FrozenSet[MutationMethod]]] = None
    #: Crossover used when the engine is not told which one to use.
    DEFAULT_CROSSOVER: ClassVar[Optional[CrossoverMethod]] = None
    #: Mutation used when the engine is not told which one to use.
    DEFAULT_MUTATION: ClassVar[Optional[MutationMethod]] = None
    #: Accepted ``crossover_params`` keys per method (missing method = unchecked).
    CROSSOVER_PARAMS: ClassVar[Mapping[CrossoverMethod, FrozenSet[str]]] = {}
    #: Accepted ``mutation_params`` keys per method (missing method = unchecked).
    MUTATION_PARAMS: ClassVar[Mapping[MutationMethod, FrozenSet[str]]] = {}

    def __init__(self, dimensions: int, bounds: Optional[Tuple[float, float]] = None):
        """Initialise the representation.

        Args:
            dimensions: Positive genome length.
            bounds: Optional ``(lower, upper)`` gene bounds.

        Raises:
            ConfigurationError: If ``dimensions`` is not a positive integer.
        """
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, (int, np.integer))
            or int(dimensions) <= 0
        ):
            raise ConfigurationError("dimensions must be a positive integer")
        self.dimensions = int(dimensions)
        self.bounds: Any = bounds

    @abstractmethod
    def random_individual(self, rng: np.random.Generator) -> np.ndarray:
        """Generate one valid random genome."""

    @abstractmethod
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        method: CrossoverMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create two children from two parents (inputs are never modified)."""

    @abstractmethod
    def mutate(
        self,
        individual: np.ndarray,
        method: MutationMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return a mutated copy of a genome (the input is never modified)."""

    @abstractmethod
    def repair(self, individual: np.ndarray) -> np.ndarray:
        """Return a genome repaired to satisfy the hard representation rules."""

    def validate(self, individual: np.ndarray) -> bool:
        """Return whether a genome already satisfies the hard rules."""
        return True

    def make_valid(self, genes: np.ndarray) -> np.ndarray:
        """Return an owned, valid genome of the canonical dtype.

        A genome that already passes :meth:`validate` is only copied (and cast
        to :attr:`GENOME_DTYPE`); otherwise :meth:`repair` runs first.

        Args:
            genes: Any array-like genome.

        Returns:
            A new array that satisfies :meth:`validate`.

        Raises:
            ConfigurationError: If the genome is not a 1-D array of
                ``dimensions`` genes, or repair cannot produce a valid genome
                (a faulty representation).
        """
        array = np.asarray(genes)
        if array.ndim != 1 or len(array) != self.dimensions:
            raise ConfigurationError(
                f"genome must be one-dimensional with {self.dimensions} genes, "
                f"got shape {array.shape}"
            )
        if not self.validate(array):
            array = self.repair(array)
        dtype = self.GENOME_DTYPE
        array = np.array(array, dtype=dtype, copy=True) if dtype is not None else np.array(array, copy=True)
        if not self.validate(array):
            raise ConfigurationError(
                f"{type(self).__name__}.repair() failed to produce a valid genome"
            )
        return array

    def absolute_sigma(self, fraction: float) -> float:
        """Translate a relative Gaussian step size into gene units.

        Args:
            fraction: Step size as a fraction of the gene range.

        Returns:
            The absolute standard deviation. The default is the identity;
            bounded real-valued representations scale by the range.
        """
        return fraction

    def diversity(self, genes: np.ndarray) -> float:
        """Population diversity in ``[0, 1]`` for a ``(population, genes)`` matrix.

        The default is the mean positional mismatch (normalised Hamming
        distance) against up to ~20 reference rows, which keeps the cost
        linear in the population size.
        """
        count = len(genes)
        if count <= 1:
            return 0.0
        stride = max(1, count // 20)
        samples = genes[::stride]
        return float(np.mean([np.mean(sample != genes, axis=1) for sample in samples]))

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description used to validate checkpoints."""
        return {
            "type": type(self).__name__,
            "dimensions": self.dimensions,
            "bounds": [float(self.bounds[0]), float(self.bounds[1])]
            if self.bounds is not None
            else None,
        }

    def check_operator_config(
        self,
        crossover: CrossoverMethod,
        crossover_params: Mapping[str, Any],
        mutation: MutationMethod,
        mutation_params: Mapping[str, Any],
    ) -> None:
        """Fail fast if an operator or one of its parameters is unusable.

        Raises:
            ConfigurationError: If a method is unsupported, a parameter name is
                unknown for the method, or a parameter value is out of range.
        """
        name = type(self).__name__
        if self.SUPPORTED_CROSSOVERS is not None and crossover not in self.SUPPORTED_CROSSOVERS:
            supported = ", ".join(sorted(m.value for m in self.SUPPORTED_CROSSOVERS))
            raise ConfigurationError(
                f"{name} does not support crossover {crossover.value!r} (supported: {supported})"
            )
        if self.SUPPORTED_MUTATIONS is not None and mutation not in self.SUPPORTED_MUTATIONS:
            supported = ", ".join(sorted(m.value for m in self.SUPPORTED_MUTATIONS))
            raise ConfigurationError(
                f"{name} does not support mutation {mutation.value!r} (supported: {supported})"
            )
        self._check_params("crossover", crossover.value, crossover_params, self.CROSSOVER_PARAMS.get(crossover))
        self._check_params("mutation", mutation.value, mutation_params, self.MUTATION_PARAMS.get(mutation))

    def _check_params(
        self,
        kind: str,
        method: str,
        params: Mapping[str, Any],
        allowed: Optional[FrozenSet[str]],
    ) -> None:
        """Validate parameter names (if ``allowed`` is known) and values."""
        if allowed is not None:
            unknown = sorted(set(params) - allowed)
            if unknown:
                accepted = ", ".join(sorted(allowed)) or "none"
                raise ConfigurationError(
                    f"unknown {kind} parameter(s) {unknown} for {method!r} (accepted: {accepted})"
                )
        for key in params:
            if key in _PARAM_RULES:
                get_param(params, key, 0.0)  # raises ConfigurationError if out of range
