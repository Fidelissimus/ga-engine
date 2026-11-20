# User guide

A tour of everything `ga-engine` can do, with code you can run. Every Python
block in this guide is executed by the test suite, and every `output` block is
compared with what the code really prints.

- [1. The big picture](#1-the-big-picture)
- [2. Defining a problem](#2-defining-a-problem)
- [3. Choosing a representation](#3-choosing-a-representation)
- [4. Running and stopping](#4-running-and-stopping)
- [5. Reading results](#5-reading-results)
- [6. Constraints](#6-constraints)
- [7. Multiple objectives](#7-multiple-objectives)
- [8. Warm starts](#8-warm-starts)
- [9. Tuning the search](#9-tuning-the-search)
- [10. Parallel evaluation](#10-parallel-evaluation)
- [11. Island model](#11-island-model)
- [12. Checkpoints and exact resume](#12-checkpoints-and-exact-resume)
- [13. When your objective function fails](#13-when-your-objective-function-fails)
- [14. Logging](#14-logging)
- [15. Writing your own representation](#15-writing-your-own-representation)
- [16. Reproducibility](#16-reproducibility)

## 1. The big picture

A genetic algorithm keeps a *population* of candidate solutions (individuals).
Each generation it selects good parents, recombines and mutates them into
offspring, evaluates the offspring with **your objective function**, and keeps
the best. `ga-engine` splits that into replaceable parts:

| Part | You choose it with | Role |
| --- | --- | --- |
| Representation | `representation=` | What a solution is, and how to recombine, mutate and repair it |
| Objective | `fitness_function=` | `f(genes) -> float` (or a sequence of floats); **minimised** |
| Selection | `selection_method=` | Which individuals become parents |
| Survival | `replacement_method=` | How the next population is formed |
| Stopping | arguments of `run()` | Budget, target, stagnation or callback |

## 2. Defining a problem

The objective receives a NumPy array (the genome) and returns a number. The
engine always **minimises**. To maximise, return the negative:

```python
from ga_engine import GeneticAlgorithm, RealRepresentation


def profit(x):                              # we want this to be as large as possible
    return 10 * x[0] - x[0] ** 2 + 5 * x[1] - 0.5 * x[1] ** 2


ga = GeneticAlgorithm(
    population_size=40,
    representation=RealRepresentation(dimensions=2, bounds=[(0, 10), (0, 10)]),
    fitness_function=lambda x: -profit(x),  # minimise the negative profit
    rng_seed=1,
)
result = ga.run(max_generations=80)
print(round(-result.best_fitness, 1))       # convert back to profit
```

```output
37.5
```

Rules for objective functions:

- Return a finite number. `NaN` is treated as "worst possible" and `-inf` is
  rejected as a failure (see [section 13](#13-when-your-objective-function-fails)).
- Do not modify the genome you are given.
- For [parallel evaluation](#10-parallel-evaluation) the function must be
  defined at module level (lambdas cannot be pickled).

## 3. Choosing a representation

| If your solution is... | Use | Notes |
| --- | --- | --- |
| a vector of real numbers | `RealRepresentation(dimensions, bounds)` | `bounds=(low, high)` for all genes, or one pair per gene |
| a vector of integers | `IntegerRepresentation(dimensions, bounds)` | inclusive integer bounds, shared or per gene |
| a bit string / yes-no choices | `BinaryRepresentation(dimensions)` | |
| an ordering (tour, schedule) | `PermutationRepresentation(dimensions)` | permutation of `0 .. n-1` |
| something else | subclass `Representation` | [section 15](#15-writing-your-own-representation) |

Genomes are homogeneous: all real, all integer, and so on. To mix kinds, encode
the integer parameters as reals and round them inside your objective, or write
a custom representation.

**Per-gene bounds** let each parameter have its own range:

```python
from ga_engine import RealRepresentation

rep = RealRepresentation(3, bounds=[(0, 1), (-10, 10), (100, 500)])
print(rep.describe())
```

```output
{'type': 'RealRepresentation', 'dimensions': 3, 'bounds': [[0.0, 1.0], [-10.0, 10.0], [100.0, 500.0]]}
```

Gaussian mutation scales its step by each gene's own range, so a wide gene and a
narrow gene mutate proportionally.

Leave `crossover_method` and `mutation_method` unset to get sensible defaults for
the genome (see [operators](operators.md)). Unsupported choices raise
`ConfigurationError` immediately, when you create the engine.

## 4. Running and stopping

`run()` stops at the **first** of these criteria:

| Argument | Meaning |
| --- | --- |
| `max_generations` | Generation budget (always applies; default 500) |
| `target_fitness` | A *feasible* solution at least this good exists (single-objective) |
| `stagnation_generations` | No improvement for this many generations |
| `on_generation` | Your callback returned a truthy value |

The result's `stop_reason` says which one fired: `"target_fitness"`,
`"stagnation"`, `"callback"` or `"max_generations"`.

The callback receives the engine after the initial population and after every
generation. Use it for progress output, a custom budget or any other rule:

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere

EVALUATION_BUDGET = 500


def watch(ga):
    if ga.generation % 4 == 0:
        print("generation", ga.generation, "evaluations", ga.evaluations)
    return ga.evaluations >= EVALUATION_BUDGET      # truthy => stop


ga = GeneticAlgorithm(20, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1)
result = ga.run(max_generations=1000, on_generation=watch)
print(result.stop_reason, result.generation)
```

```output
generation 0 evaluations 20
generation 4 evaluations 100
generation 8 evaluations 180
generation 12 evaluations 260
generation 16 evaluations 340
generation 20 evaluations 420
generation 24 evaluations 500
callback 24
```

`run(..., resume=True)` continues the current population instead of starting a
new one, so you can run in stages:

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere

ga = GeneticAlgorithm(20, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1)
ga.run(max_generations=10)
result = ga.run(max_generations=25, resume=True)   # continues from generation 10
print(result.generation, len(result.stats["generation"]))
```

```output
25 26
```

## 5. Reading results

`run()` returns a `RunResult`:

| Field | Meaning |
| --- | --- |
| `best_individual` | Best solution found (`.genes`, `.fitness`, `.constraint_violation`, `.is_feasible`) |
| `best_fitness` | Its fitness |
| `generation` | Generations completed |
| `stop_reason` | Why the run ended |
| `converged` | `True` if it stopped on the target or on stagnation |
| `evaluations`, `evaluation_failures` | Objective calls made, and how many failed |
| `stats` | Per-generation statistics (dictionary of lists) |
| `pareto_front` | Multi-objective only: the non-dominated individuals |
| `island_stats` | Island model only: statistics per island |
| `elapsed_seconds` | Wall-clock time of the call |

`result.stats` has one entry per generation, index 0 being the initial
population:

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere

result = GeneticAlgorithm(20, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1).run(max_generations=5)
print(list(result.stats)[:6])
print(len(result.stats["best_fitness"]))
```

```output
['generation', 'best_fitness', 'worst_fitness', 'mean_fitness', 'median_fitness', 'std_fitness']
6
```

`best_fitness` and friends describe the *current population*; the best-so-far
trajectory is `ga.best_fitness_history`. The full key list is in
[configuration.md](configuration.md#statistics-keys).

To plot convergence (matplotlib is not a dependency; install it yourself):

```python
# doctest: skip
import matplotlib.pyplot as plt

plt.plot(ga.best_fitness_history, label="best so far")
plt.plot(result.stats["mean_fitness"], label="population mean")
plt.xlabel("generation")
plt.yscale("log")
plt.legend()
plt.show()
```

## 6. Constraints

Return a *violation* from `constraint_function`: positive means violated, zero or
negative means satisfied. A sequence of violations is summed over its positive parts.

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere


def need_x0_at_least_one(x):
    return 1.0 - float(x[0])            # violated while x[0] < 1


ga = GeneticAlgorithm(
    60,
    RealRepresentation(3, (-5, 5)),
    sphere,
    constraint_function=need_x0_at_least_one,
    constraint_penalty=100.0,
    rng_seed=1,
)
result = ga.run(max_generations=100)
print(result.best_individual.is_feasible)
```

```output
True
```

Single-objective comparison uses `fitness + constraint_penalty * violation`.
**The penalty must be large enough that breaking the constraint never pays**;
otherwise the "best" individual can be infeasible. Always check
`result.best_individual.is_feasible`. Multi-objective runs use constrained
dominance (feasible beats infeasible; among infeasible solutions the smaller
violation wins) and ignore `constraint_penalty`.

## 7. Multiple objectives

Return a sequence of at least two objectives (always the same number) and set
`multi_objective=True`. The engine runs NSGA-II and returns the Pareto front:

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import zdt1

ga = GeneticAlgorithm(
    60, RealRepresentation(6, (0.0, 1.0)), zdt1, multi_objective=True, rng_seed=5
)
result = ga.run(max_generations=40)

front = sorted(result.pareto_front, key=lambda ind: ind.fitness[0])
print(len(front) > 5, front[0].fitness[0] <= front[-1].fitness[0])
```

```output
True True
```

Each front member's `fitness` is a tuple. `target_fitness` is not available in
this mode; stagnation means "the Pareto front stopped changing".

## 8. Warm starts

Seed the population with solutions you already know. They are repaired into
validity, and the remaining slots are filled randomly:

```python
import numpy as np

from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere

ga = GeneticAlgorithm(20, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1)
ga.initialize_population(initial_genomes=[np.zeros(3)])   # a known optimum
result = ga.run(max_generations=5, resume=True)           # resume=True keeps the seeded population
print(result.best_fitness)
```

```output
0.0
```

## 9. Tuning the search

Rules of thumb, not guarantees (genetic algorithms are stochastic and problems differ, so run several seeds for anything important):

| Symptom | Things to try |
| --- | --- |
| Converges early to a poor solution | Larger `population_size`; higher `mutation_prob`; `restart_on_stagnation`; fewer elites; the island model |
| Converges too slowly | Smaller `population_size`; `adaptive_mutation=True`; more `crossover_prob`; more `elitism_count` |
| Wobbles near the optimum | Smaller `mutation_sigma`; `adaptive_mutation=True`; for SBX a larger `eta` (`crossover_params={"eta": 40}`) |
| Integer genes converge slowly | Raise `mutation_prob` (0.3-0.8) and use `CREEP` mutation for ordinal settings. On the small integer problem in the README, the optimum was found for 15 of 30 seeds at `mutation_prob=0.1` and 28 of 30 at 0.5 (`CREEP` with `max_step=6` found it for all 30 at both rates) |
| Many generations without real progress | `stagnation_generations` (with `stagnation_tolerance` to ignore tiny gains) |

**Adaptive mutation and restarts.** `adaptive_mutation=True` applies the 1/5
success rule: when more than 20 % of mutated offspring beat their better parent,
mutation grows; otherwise it shrinks. `restart_on_stagnation=N` replaces the
worst ~10 % of the population with random individuals after `N` stagnating
generations (the best individual always survives).

**Steady-state replacement.** With `ReplacementMethod.STEADY_STATE` each
generation creates only a few offspring (`steady_state_offspring`, default
`max(2, population_size // 10)`) that compete with the whole population. A
"generation" is then much cheaper, so compare runs by `result.evaluations`:

```python
from ga_engine import GeneticAlgorithm, RealRepresentation, ReplacementMethod
from ga_engine.benchmarks import sphere

ga = GeneticAlgorithm(
    30, RealRepresentation(3, (-5, 5)), sphere,
    replacement_method=ReplacementMethod.STEADY_STATE, rng_seed=1,
)
result = ga.run(max_generations=100)
print(ga.config.offspring_per_generation, result.evaluations)   # 30 initial + 100 * 3
```

```output
3 330
```

Operator parameters are set with `crossover_params` and `mutation_params`, and
are validated against the chosen operator:

```python
from ga_engine import GeneticAlgorithm, MutationMethod, RealRepresentation
from ga_engine.benchmarks import sphere

ga = GeneticAlgorithm(
    30, RealRepresentation(3, (-5, 5)), sphere,
    mutation_method=MutationMethod.POLYNOMIAL,
    crossover_params={"eta": 30.0},      # SBX distribution index
    mutation_params={"eta_m": 25.0},     # polynomial-mutation index
    adaptive_mutation=True,
    restart_on_stagnation=25,
    rng_seed=2,
)
print(ga.run(max_generations=60).best_fitness < 1.0)
```

```output
True
```

All options with defaults: [configuration.md](configuration.md).

## 10. Parallel evaluation

For expensive objectives, evaluate offspring in worker processes with
`n_workers`. Three rules:

1. The fitness (and constraint) function must be **picklable**: defined at module
   level, or an instance of a module-level class. Lambdas and closures are
   rejected immediately with a clear `TypeError`.
2. Guard the entry point with `if __name__ == "__main__":`. Workers are started
   with the `spawn` method on every platform (the Windows default), which
   re-imports your script.
3. Use the engine as a context manager (or call `run()`) so the pool is released.

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import rastrigin     # module-level => picklable

if __name__ == "__main__":
    with GeneticAlgorithm(
        40, RealRepresentation(6, (-5.12, 5.12)), rastrigin, n_workers=2, rng_seed=1
    ) as ga:
        result = ga.run(max_generations=10)
    print(result.evaluations)
```

```output
440
```

Results are identical to serial evaluation. A speed-up appears only when one
evaluation costs far more than starting processes and sending genomes;
`n_workers=-1` uses all CPUs.

## 11. Island model

Several populations evolve separately and periodically exchange their best
individuals in a ring. Islands may differ in operators and rates (a cheap way to
keep diversity) but must share the same genome description and objective mode:

```python
from ga_engine import (
    GeneticAlgorithm, IslandModel, RealRepresentation, SelectionMethod,
)
from ga_engine.benchmarks import rastrigin


def island(seed, **options):
    return GeneticAlgorithm(
        30, RealRepresentation(6, (-5.12, 5.12)), rastrigin, rng_seed=seed, **options
    )


islands = [island(1), island(2, selection_method=SelectionMethod.RANK, mutation_prob=0.3), island(3)]
with IslandModel(islands, migration_interval=10, migration_size=2) as model:
    result = model.run(max_generations=40)

print(result.generation, len(result.island_stats))
```

```output
40 3
```

Single-objective results report the best solution across islands. In
multi-objective mode the result is the non-dominated set of the union of all
island fronts. `IslandModel.run` accepts the same stopping arguments as
`GeneticAlgorithm.run` (target: any island; stagnation: every island).

## 12. Checkpoints and exact resume

```python
import tempfile
from pathlib import Path

from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere


def build():
    return GeneticAlgorithm(30, RealRepresentation(4, (-5.0, 5.0)), sphere, rng_seed=11)


reference = build().run(max_generations=40)

first = build()
first.run(max_generations=20)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "run.json"        # use ".npz" for a compact binary file
    first.save_checkpoint(path)
    second = build()
    second.load_checkpoint(path)            # validates everything before changing anything
resumed = second.run(max_generations=40, resume=True)

print(resumed.best_fitness == reference.best_fitness)
```

```output
True
```

The engine you load into must be configured compatibly (population size,
representation, dimensions, bounds, objective mode); otherwise `CheckpointError` is
raised and the engine is left untouched. Writes are atomic, `.json` files are
strict JSON. Details: [checkpoints.md](checkpoints.md).

## 13. When your objective function fails

By default a failing evaluation (an exception, or a malformed return value) gets
an infinitely bad fitness, a summary warning is logged, and the failure is counted
in `result.evaluation_failures`. To abort instead, use the `RAISE` policy:

```python
from ga_engine import (
    EvaluationError, EvaluationErrorPolicy, GeneticAlgorithm, RealRepresentation,
)


def fragile(x):
    if x[0] > 0.9:
        raise ValueError("simulation diverged")
    return float(x[0] ** 2)


tolerant = GeneticAlgorithm(30, RealRepresentation(2, (0, 1)), fragile, rng_seed=1)
print(tolerant.run(max_generations=5).evaluation_failures > 0)

strict = GeneticAlgorithm(
    30, RealRepresentation(2, (0, 1)), fragile, rng_seed=1,
    evaluation_error_policy=EvaluationErrorPolicy.RAISE,
)
try:
    strict.run(max_generations=5)
except EvaluationError as error:
    print("aborted:", "simulation diverged" in str(error))
```

```output
True
aborted: True
```

## 14. Logging

The library never configures logging itself: it logs to the `ga_engine` logger,
which has a `NullHandler`. For a quick console view use `verbose=True` (DEBUG) or
call `enable_console_logging()`, which is idempotent:

```python
import logging

from ga_engine import enable_console_logging

enable_console_logging(logging.INFO)     # restarts, warnings and run summaries
```

If your application already configures logging, attach your own handler to
`logging.getLogger("ga_engine")` instead.

## 15. Writing your own representation

Subclass `Representation` and implement `random_individual`, `crossover`,
`mutate` and `repair` (add `validate` so the engine can detect invalid genomes).
Set `GENOME_DTYPE` so genomes get a canonical dtype. The example below optimises
a *subset*: `k` distinct items out of `n`.

```python
import numpy as np

from ga_engine import GeneticAlgorithm, Representation


class SubsetRepresentation(Representation):
    """A genome is k distinct item indices out of n, kept sorted."""

    GENOME_DTYPE = np.dtype(np.int64)

    def __init__(self, k: int, n: int):
        super().__init__(dimensions=k)
        self.n = n

    def random_individual(self, rng):
        return np.sort(rng.choice(self.n, self.dimensions, replace=False))

    def validate(self, genes):
        g = np.asarray(genes)
        return bool(
            g.ndim == 1 and len(g) == self.dimensions
            and np.issubdtype(g.dtype, np.integer)
            and len(set(g.tolist())) == len(g) and g.min() >= 0 and g.max() < self.n
        )

    def repair(self, genes):                      # drop duplicates / out-of-range, refill
        kept = [int(v) for v in np.asarray(genes).ravel() if 0 <= v < self.n]
        unique = list(dict.fromkeys(kept))[: self.dimensions]
        missing = [i for i in range(self.n) if i not in set(unique)]
        unique += missing[: self.dimensions - len(unique)]
        return np.sort(np.array(unique, dtype=np.int64))

    def crossover(self, parent1, parent2, method, rng, **kwargs):
        pool = np.union1d(parent1, parent2)       # children draw from the parents' items
        return (
            np.sort(rng.choice(pool, self.dimensions, replace=False)),
            np.sort(rng.choice(pool, self.dimensions, replace=False)),
        )

    def mutate(self, individual, method, rng, **kwargs):
        mutated = individual.copy()
        outside = np.setdiff1d(np.arange(self.n), mutated)
        if len(outside):
            mutated[rng.integers(self.dimensions)] = rng.choice(outside)
        return np.sort(mutated)


values = np.random.default_rng(0).random(30)      # pick the 5 most valuable items
ga = GeneticAlgorithm(
    population_size=40,
    representation=SubsetRepresentation(k=5, n=30),
    fitness_function=lambda subset: -float(values[subset].sum()),
    mutation_prob=0.5,
    rng_seed=1,
)
result = ga.run(max_generations=100)
best = sorted(result.best_individual.genes.tolist())
print(best == sorted(np.argsort(-values)[:5].tolist()))
```

```output
True
```

The optional class variables `SUPPORTED_CROSSOVERS`, `SUPPORTED_MUTATIONS`,
`DEFAULT_CROSSOVER`, `DEFAULT_MUTATION`, `CROSSOVER_PARAMS` and
`MUTATION_PARAMS` opt in to construction-time validation of operator choices. A
representation that leaves them unset (like the one above) simply receives the
engine's default method names, which it may ignore. Implementing `describe()`
(it has a sensible default) lets checkpoints verify genome compatibility.

## 16. Reproducibility

- Give `rng_seed` and every random choice comes from one `numpy.random.Generator`
  owned by the engine; the same code, seed and library versions give the same run.
- A run resumed from a checkpoint equals the uninterrupted run exactly.
- Parallel evaluation does not change results: only the objective function runs in
  workers; all randomness stays in the main process.
- Random streams are stable for a given NumPy version but not guaranteed across
  versions, so exact numbers in this documentation may differ slightly on other
  NumPy releases.
- Your objective function must itself be deterministic for results to be.
