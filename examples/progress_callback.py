"""Report progress, stop on a custom budget, and warm-start from a known solution."""

from __future__ import annotations

import numpy as np

from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import rastrigin

EVALUATION_BUDGET = 1500


def report_and_check_budget(ga: GeneticAlgorithm) -> bool:
    """Print every 5th generation; return True (stop) once the budget is spent."""
    if ga.generation % 5 == 0:
        best = ga.best_individual.fitness  # type: ignore[union-attr]
        print(f"generation {ga.generation:>3}  evaluations={ga.evaluations:>4}  best={best:8.3f}")
    return ga.evaluations >= EVALUATION_BUDGET   # truthy => stop_reason == "callback"


def main() -> None:
    """Seed the population with a reasonable guess, then evolve within a budget."""
    ga = GeneticAlgorithm(
        population_size=50,
        representation=RealRepresentation(dimensions=6, bounds=(-5.12, 5.12)),
        fitness_function=rastrigin,
        rng_seed=8,
    )
    ga.initialize_population(initial_genomes=[np.full(6, 0.5)])   # warm start
    result = ga.run(max_generations=1000, resume=True, on_generation=report_and_check_budget)

    print(f"stop reason: {result.stop_reason}; best fitness {result.best_fitness:.3f}")


if __name__ == "__main__":
    main()
