"""Interrupt a run, save a checkpoint, and resume it exactly."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import rastrigin


def build() -> GeneticAlgorithm:
    """Create an engine with a fixed seed so the runs are comparable."""
    return GeneticAlgorithm(50, RealRepresentation(8, (-5.12, 5.12)), rastrigin, rng_seed=21)


def main() -> None:
    """Show that resuming from a checkpoint reproduces the uninterrupted run."""
    uninterrupted = build().run(max_generations=120)

    first_half = build()
    first_half.run(max_generations=60)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.json"  # use .npz for a compact binary file
        first_half.save_checkpoint(path)
        print(f"checkpoint written: {path.name} ({path.stat().st_size} bytes)")

        second_half = build()
        second_half.load_checkpoint(path)
    resumed = second_half.run(max_generations=120, resume=True)

    print(f"uninterrupted best: {uninterrupted.best_fitness!r}")
    print(f"resumed best      : {resumed.best_fitness!r}")
    print(f"identical         : {uninterrupted.best_fitness == resumed.best_fitness}")


if __name__ == "__main__":
    main()
