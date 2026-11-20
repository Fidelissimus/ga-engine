"""Validated, immutable engine configuration.

Validation lives here, apart from the evolutionary logic, so every constraint on
a setting is stated once and the engine can rely on well-formed values.
:func:`build_config` normalises and validates raw constructor arguments and
returns a frozen :class:`GAConfig`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from .enums import (
    CrossoverMethod,
    EvaluationErrorPolicy,
    MutationMethod,
    ReplacementMethod,
    SelectionMethod,
)
from .errors import ConfigurationError
from .representations.base import Representation

#: Operators assumed for custom representations that declare no defaults.
FALLBACK_CROSSOVER = CrossoverMethod.SBX
FALLBACK_MUTATION = MutationMethod.GAUSSIAN

#: Mutation keywords the engine manages itself; users may not override them.
RESERVED_MUTATION_PARAMS = frozenset({"mutation_prob", "sigma"})


@dataclass(frozen=True)
class GAConfig:
    """Resolved settings of a :class:`~ga_engine.engine.GeneticAlgorithm`.

    Attributes:
        population_size: Number of individuals.
        selection_method: Parent-selection strategy.
        crossover_method: Resolved recombination operator.
        mutation_method: Resolved mutation operator.
        replacement_method: Survivor-selection scheme.
        crossover_prob: Probability that a mating pair is recombined.
        mutation_prob: Initial probability that an offspring is mutated.
        gene_mutation_prob: Initial per-gene mutation probability.
        elitism_count: Elites preserved by generational replacement.
        tournament_size: Contestants per tournament.
        multi_objective: Whether NSGA-II mode is used.
        n_workers: Worker processes for evaluation (``<= 1`` = serial).
        adaptive_mutation: Whether mutation intensity adapts online.
        restart_on_stagnation: Stagnating generations before random
            individuals are injected (``None`` disables).
        mutation_sigma: Initial Gaussian step as a fraction of the gene range.
        constraint_penalty: Weight of constraint violation in scalar fitness.
        crossover_params: Extra keyword arguments for the crossover operator.
        mutation_params: Extra keyword arguments for the mutation operator.
        offspring_per_generation: Offspring created (and evaluated) per step.
        adaptation_window: Minimum mutation trials before an adaptation step.
        stagnation_tolerance: Minimum improvement that resets stagnation.
        evaluation_error_policy: Handling of failing user functions.
        verbose: Whether to attach console logging.
    """

    population_size: int
    selection_method: SelectionMethod
    crossover_method: CrossoverMethod
    mutation_method: MutationMethod
    replacement_method: ReplacementMethod
    crossover_prob: float
    mutation_prob: float
    gene_mutation_prob: float
    elitism_count: int
    tournament_size: int
    multi_objective: bool
    n_workers: int
    adaptive_mutation: bool
    restart_on_stagnation: Optional[int]
    mutation_sigma: float
    constraint_penalty: float
    crossover_params: Mapping[str, Any]
    mutation_params: Mapping[str, Any]
    offspring_per_generation: int
    adaptation_window: int
    stagnation_tolerance: float
    evaluation_error_policy: EvaluationErrorPolicy
    verbose: bool


def _integer(name: str, value: Any, minimum: int, maximum: Optional[int] = None) -> int:
    """Validate an integer setting (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigurationError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or (maximum is not None and number > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ConfigurationError(f"{name} must be {bound}")
    return number


def _probability(name: str, value: Any) -> float:
    """Validate a probability in ``[0, 1]``."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1]")
    return number


def _non_negative(name: str, value: Any) -> float:
    """Validate a finite number ``>= 0``."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ConfigurationError(f"{name} must be a finite number >= 0")
    return number


def _resolve_workers(n_workers: Any) -> int:
    """Translate ``n_workers`` (``-1`` = all CPUs) into a worker count."""
    if isinstance(n_workers, bool) or not isinstance(n_workers, (int, np.integer)):
        raise ConfigurationError("n_workers must be an integer")
    if n_workers == -1:
        return os.cpu_count() or 1
    if n_workers < 0:
        raise ConfigurationError("n_workers must be >= 0, or -1 for all CPUs")
    return int(n_workers)


def build_config(
    *,
    population_size: Any,
    representation: Representation,
    selection_method: SelectionMethod,
    crossover_method: Optional[CrossoverMethod],
    mutation_method: Optional[MutationMethod],
    replacement_method: ReplacementMethod,
    crossover_prob: Any,
    mutation_prob: Any,
    elitism_count: Any,
    tournament_size: Any,
    multi_objective: bool,
    n_workers: Any,
    adaptive_mutation: bool,
    restart_on_stagnation: Any,
    verbose: bool,
    gene_mutation_prob: Any,
    mutation_sigma: Any,
    constraint_penalty: Any,
    crossover_params: Optional[Mapping[str, Any]],
    mutation_params: Optional[Mapping[str, Any]],
    steady_state_offspring: Any,
    adaptation_window: Any,
    stagnation_tolerance: Any,
    evaluation_error_policy: EvaluationErrorPolicy,
) -> GAConfig:
    """Validate raw arguments and build a :class:`GAConfig`.

    Unset operators are resolved from the representation's defaults, and the
    resulting operator choice and parameters are checked against what the
    representation supports.

    Raises:
        ConfigurationError: If any value is invalid. ``ConfigurationError`` is a
            ``ValueError``.
    """
    size = _integer("population_size", population_size, 2)
    crossover_method = crossover_method or representation.DEFAULT_CROSSOVER or FALLBACK_CROSSOVER
    mutation_method = mutation_method or representation.DEFAULT_MUTATION or FALLBACK_MUTATION
    elites = _integer("elitism_count", elitism_count, 0, size)

    for name, enum_type, value in (
        ("selection_method", SelectionMethod, selection_method),
        ("crossover_method", CrossoverMethod, crossover_method),
        ("mutation_method", MutationMethod, mutation_method),
        ("replacement_method", ReplacementMethod, replacement_method),
        ("evaluation_error_policy", EvaluationErrorPolicy, evaluation_error_policy),
    ):
        if not isinstance(value, enum_type):
            raise ConfigurationError(f"{name} must be a {enum_type.__name__} member")

    if restart_on_stagnation is not None:
        restart_on_stagnation = _integer("restart_on_stagnation", restart_on_stagnation, 1)

    mutation_extra = dict(mutation_params or {})
    reserved = sorted(RESERVED_MUTATION_PARAMS & set(mutation_extra))
    if reserved:
        raise ConfigurationError(
            f"mutation_params may not set {reserved}; use mutation_prob, "
            "gene_mutation_prob and mutation_sigma instead"
        )

    if replacement_method == ReplacementMethod.STEADY_STATE:
        default_offspring = min(size, max(2, size // 10))
        offspring = (
            default_offspring
            if steady_state_offspring is None
            else _integer("steady_state_offspring", steady_state_offspring, 1, size)
        )
    else:
        offspring = size

    gene_probability = (
        1.0 / representation.dimensions
        if gene_mutation_prob is None
        else _probability("gene_mutation_prob", gene_mutation_prob)
    )

    crossover_extra = dict(crossover_params or {})
    representation.check_operator_config(
        crossover_method, crossover_extra, mutation_method, mutation_extra
    )

    return GAConfig(
        population_size=size,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_method=mutation_method,
        replacement_method=replacement_method,
        crossover_prob=_probability("crossover_prob", crossover_prob),
        mutation_prob=_probability("mutation_prob", mutation_prob),
        gene_mutation_prob=gene_probability,
        elitism_count=elites,
        tournament_size=_integer("tournament_size", tournament_size, 1),
        multi_objective=bool(multi_objective),
        n_workers=_resolve_workers(n_workers),
        adaptive_mutation=bool(adaptive_mutation),
        restart_on_stagnation=restart_on_stagnation,
        mutation_sigma=_non_negative("mutation_sigma", 0.1 if mutation_sigma is None else mutation_sigma),
        constraint_penalty=_non_negative("constraint_penalty", constraint_penalty),
        crossover_params=MappingProxyType(crossover_extra),
        mutation_params=MappingProxyType(mutation_extra),
        offspring_per_generation=offspring,
        adaptation_window=_integer("adaptation_window", adaptation_window, 1),
        stagnation_tolerance=_non_negative("stagnation_tolerance", stagnation_tolerance),
        evaluation_error_policy=evaluation_error_policy,
        verbose=bool(verbose),
    )
