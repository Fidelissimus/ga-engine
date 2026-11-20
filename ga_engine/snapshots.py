"""Mapping between a live engine and an :class:`EngineSnapshot`.

:mod:`ga_engine.checkpoint` knows the *file* format; this module knows which
parts of an engine make up its resumable state, and how to check a snapshot
against an engine before applying it. Loading is two-phase: :func:`validate_snapshot`
inspects everything and changes nothing, then :func:`install_snapshot` applies
the (already validated) state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .checkpoint import CHECKPOINT_VERSION, EngineSnapshot
from .errors import CheckpointError
from .models import Individual

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers (avoids a cycle)
    from .engine import GeneticAlgorithm


def capture_snapshot(engine: "GeneticAlgorithm") -> EngineSnapshot:
    """Capture the resumable state of ``engine`` (population and run state)."""
    adaptation = engine.adaptation.state_dict()
    progress = engine.progress.state_dict()
    description = engine.representation.describe()
    best = engine.progress.best_individual
    return EngineSnapshot(
        version=CHECKPOINT_VERSION,
        population_size=engine.population_size,
        representation=description["type"],
        dimensions=description["dimensions"],
        bounds=description["bounds"],
        multi_objective=engine.multi_objective,
        generation=engine.generation,
        stagnation=progress["stagnation"],
        generations_since_restart=progress["generations_since_restart"],
        mutation_prob=adaptation["mutation_prob"],
        gene_mutation_prob=adaptation["gene_mutation_prob"],
        sigma=adaptation["sigma"],
        success_count=adaptation["success_count"],
        trial_count=adaptation["trial_count"],
        last_success_ratio=adaptation["last_success_ratio"],
        objective_count=engine.evaluator.objective_count,
        evaluations=engine.evaluator.evaluations,
        evaluation_failures=engine.evaluator.failures,
        rng_state=dict(engine.rng.bit_generator.state),
        stats=engine.statistics.copy(),
        best_fitness_history=progress["best_fitness_history"],
        pareto_signature=progress["pareto_signature"],
        population=[ind.clone() for ind in engine.population],
        best_individual=best.clone() if best else None,
    )


def validate_snapshot(engine: "GeneticAlgorithm", snapshot: EngineSnapshot) -> None:
    """Check that ``snapshot`` fits ``engine``, without changing the engine.

    Raises:
        CheckpointError: On any mismatch (population size, representation type,
            dimensions, bounds, objective mode), an invalid genome, a fitness of
            the wrong shape, or an incompatible random state.
    """
    description = engine.representation.describe()
    if snapshot.population_size != engine.population_size:
        raise CheckpointError(
            f"Checkpoint population_size {snapshot.population_size} does not match "
            f"engine population_size {engine.population_size}"
        )
    if len(snapshot.population) != engine.population_size:
        raise CheckpointError("Checkpoint population does not contain population_size individuals")
    if snapshot.representation != description["type"]:
        raise CheckpointError(
            f"Checkpoint representation {snapshot.representation!r} does not match "
            f"{description['type']!r}"
        )
    if snapshot.dimensions != description["dimensions"]:
        raise CheckpointError(
            f"Checkpoint dimensions {snapshot.dimensions} do not match {description['dimensions']}"
        )
    if snapshot.bounds is not None and snapshot.bounds != description["bounds"]:
        raise CheckpointError(
            f"Checkpoint bounds {snapshot.bounds} do not match {description['bounds']}"
        )
    if snapshot.multi_objective != engine.multi_objective:
        raise CheckpointError("Checkpoint multi_objective flag does not match the engine")

    members = list(snapshot.population)
    if snapshot.best_individual is not None:
        members.append(snapshot.best_individual)
    for member in members:
        if not engine.representation.validate(member.genes):
            raise CheckpointError(
                "Checkpoint contains a genome that is invalid for this representation"
            )
        if isinstance(member.fitness, tuple) != engine.multi_objective:
            raise CheckpointError("Checkpoint fitness shape does not match the objective mode")
    try:
        probe = type(engine.rng.bit_generator)()
        probe.state = snapshot.rng_state
    except (ValueError, KeyError, TypeError) as exc:
        raise CheckpointError(f"Checkpoint random state is incompatible: {exc}") from exc


def _canonical(engine: "GeneticAlgorithm", individual: Individual) -> Individual:
    """Clone ``individual`` with genes in the representation's canonical dtype."""
    clone = individual.clone()
    clone.genes = engine.representation.make_valid(clone.genes)
    return clone


def install_snapshot(engine: "GeneticAlgorithm", snapshot: EngineSnapshot) -> None:
    """Install an already validated snapshot as the engine's state."""
    best = snapshot.best_individual
    engine.population = [_canonical(engine, ind) for ind in snapshot.population]
    engine.generation = snapshot.generation
    engine.progress.load_state_dict(
        {
            "stagnation": snapshot.stagnation,
            "generations_since_restart": snapshot.generations_since_restart,
            "pareto_signature": snapshot.pareto_signature,
            "best_fitness_history": snapshot.best_fitness_history,
        },
        _canonical(engine, best) if best is not None else None,
    )
    engine.adaptation.load_state_dict(
        {
            "mutation_prob": snapshot.mutation_prob,
            "gene_mutation_prob": snapshot.gene_mutation_prob,
            "sigma": snapshot.sigma,
            "success_count": snapshot.success_count,
            "trial_count": snapshot.trial_count,
            "last_success_ratio": snapshot.last_success_ratio,
        }
    )
    engine.evaluator.objective_count = snapshot.objective_count
    engine.evaluator.evaluations = snapshot.evaluations
    engine.evaluator.failures = snapshot.evaluation_failures
    engine.statistics.load(snapshot.stats)
    engine.rng.bit_generator.state = snapshot.rng_state
    if engine.multi_objective:
        engine.refresh_ranking()
