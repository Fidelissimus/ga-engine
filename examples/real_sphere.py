"""Minimise a continuous function with adaptive Gaussian mutation.

Run from the project root after ``pip install -e .``::

    python examples/real_sphere.py
"""

from __future__ import annotations

from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere


def main() -> None:
    """Solve the 10-dimensional sphere problem and print a short summary."""
    ga = GeneticAlgorithm(
        population_size=60,
        representation=RealRepresentation(dimensions=10, bounds=(-5.0, 5.0)),
        fitness_function=sphere,
        adaptive_mutation=True,
        rng_seed=42,
    )
    result = ga.run(max_generations=300, target_fitness=1e-2)

    print(f"stop reason : {result.stop_reason} at generation {result.generation}")
    print(f"best fitness: {result.best_fitness:.3e}")
    print(f"evaluations : {result.evaluations}")
    print(f"mutation prob adapted to {ga.mutation_prob:.3f}")


if __name__ == "__main__":
    main()
