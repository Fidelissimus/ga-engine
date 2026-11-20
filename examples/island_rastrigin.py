"""Run an island model with differently configured islands on Rastrigin."""

from __future__ import annotations

from ga_engine import (
    GeneticAlgorithm,
    IslandModel,
    MutationMethod,
    RealRepresentation,
    SelectionMethod,
)
from ga_engine.benchmarks import rastrigin


def make_island(seed: int, **overrides: object) -> GeneticAlgorithm:
    """Create one island; islands may differ in operators and rates."""
    return GeneticAlgorithm(
        population_size=40,
        representation=RealRepresentation(dimensions=10, bounds=(-5.12, 5.12)),
        fitness_function=rastrigin,
        rng_seed=seed,
        **overrides,  # type: ignore[arg-type]
    )


def main() -> None:
    """Evolve three islands with ring migration and report the global best."""
    islands = [
        make_island(1),
        make_island(2, selection_method=SelectionMethod.RANK, mutation_prob=0.3),
        make_island(3, mutation_method=MutationMethod.POLYNOMIAL, adaptive_mutation=True),
    ]
    with IslandModel(islands, migration_interval=10, migration_size=2) as model:
        result = model.run(max_generations=200)

    print(f"global best: {result.best_fitness:.3f} after {result.generation} generations")
    for index, stats in enumerate(result.island_stats or []):
        print(f"  island {index}: final best {stats['best_fitness'][-1]:.3f}")


if __name__ == "__main__":
    main()
