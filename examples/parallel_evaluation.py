"""Evaluate an expensive objective in parallel worker processes.

Parallel evaluation uses the ``spawn`` start method on every platform, so the
objective must live at module level and the entry point must be guarded with
``if __name__ == "__main__":`` (both are true here).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from ga_engine import GeneticAlgorithm, RealRepresentation


def expensive_sphere(x: np.ndarray) -> float:
    """A sphere function made artificially slow (a few milliseconds per call)."""
    total = 0.0
    for _ in range(4000):
        total = math.fsum(np.square(x)) + 0.0 * total
    return total


def timed_run(n_workers: int) -> tuple[float, float]:
    """Run 10 generations and return ``(seconds, best fitness)``."""
    representation = RealRepresentation(10, (-5.0, 5.0))
    with GeneticAlgorithm(
        80, representation, expensive_sphere, rng_seed=1, n_workers=n_workers
    ) as ga:
        started = time.perf_counter()
        result = ga.run(max_generations=10)
        return time.perf_counter() - started, result.best_fitness


def main() -> None:
    """Compare serial and parallel wall-clock time (results are identical)."""
    print(f"CPU cores available: {os.cpu_count()}")
    serial_time, serial_best = timed_run(n_workers=0)
    parallel_time, parallel_best = timed_run(n_workers=2)
    print(f"serial   : {serial_time:.2f}s  best={serial_best:.6f}")
    print(f"2 workers: {parallel_time:.2f}s  best={parallel_best:.6f}")
    print(f"same result: {serial_best == parallel_best}")
    print("The speed-up needs more than one physical core and an evaluation that "
          "costs much more than process start-up and data transfer.")


if __name__ == "__main__":
    main()
