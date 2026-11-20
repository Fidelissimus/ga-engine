# ga-engine: briefing for AI assistants

> **For humans:** paste or attach this file to your AI assistant, then describe
> your optimisation problem. The assistant should be able to write correct code
> using the `ga_engine` library without any other documentation.
>
> **For the assistant:** this file is the complete contract. Use only the names,
> parameters and behaviours listed here. Where something is not listed, it does
> not exist; see [Things that do not exist](#8-things-that-do-not-exist). Every
> code block in this file is executed by the library's test suite, so the
> templates are known to run.

## 1. What it is

`ga-engine` (import name `ga_engine`) is a genetic-algorithm library. You give it
an objective function and a description of what a solution looks like; it evolves
a population and returns the best solution.

- Depends only on NumPy. Python 3.9+.
- **Always minimises.** To maximise `f`, minimise `-f`.
- Genome kinds: real vectors, integer vectors, bit strings, permutations, or your
  own `Representation` subclass. A genome is a 1-D NumPy array. All genes in one
  genome have the same kind.
- Single-objective, or multi-objective (NSGA-II).
- Constraints, process-parallel evaluation, island model, checkpoints with exact
  resume, warm starts, progress callbacks.

It is **not**: a deep-learning framework, a GPU library, a distributed system, or
a plotting tool. It has no maximise flag and no mixed-type genome.

Install from the repository root: `pip install -e .`

## 2. Imports

```python
from ga_engine import (
    GeneticAlgorithm, IslandModel, GAConfig,
    RealRepresentation, IntegerRepresentation, BinaryRepresentation,
    PermutationRepresentation, Representation,
    Individual, RunResult, StopReason,
    SelectionMethod, CrossoverMethod, MutationMethod, ReplacementMethod,
    EvaluationErrorPolicy,
    GAEngineError, ConfigurationError, EvaluationError, CheckpointError,
    enable_console_logging,
)
from ga_engine.benchmarks import (
    sphere, rastrigin, rosenbrock, ackley, zdt1, onemax, KnapsackProblem, TspProblem,
)

print(GeneticAlgorithm.__name__)
```

```output
GeneticAlgorithm
```

## 3. Rules that are always true

1. **Minimise.** Return `-score` to maximise.
2. The objective is `f(genes: np.ndarray) -> float`. In multi-objective mode it
   returns a sequence of at least two floats, always the same number. Never
   modify `genes`.
3. Build the engine with positional or keyword arguments
   `GeneticAlgorithm(population_size, representation, fitness_function, ...)`;
   everything after those three has a default.
4. Run with `result = ga.run(max_generations=...)`. `result` is a `RunResult`
   dataclass: use attributes (`result.best_fitness`), never indexing
   (`result["best_fitness"]` fails).
5. The best solution is `result.best_individual.genes` (NumPy array) with
   `result.best_fitness`.
6. Operators are enum members (`SelectionMethod.RANK`), never strings. Leave
   `crossover_method` / `mutation_method` as `None` to get proper defaults for the genome.
7. Pass `rng_seed=<int>` for reproducible runs.
8. With `n_workers > 1` the objective must be a **module-level function** (not a
   lambda or closure) and the script must be guarded by
   `if __name__ == "__main__":`.
9. With constraints (single-objective) the `constraint_penalty` must be large
   enough that violating a constraint never pays; afterwards check
   `result.best_individual.is_feasible`.
10. Invalid settings raise `ConfigurationError` (a `ValueError`) when the engine is
    built, not during the run.

## 4. Templates

Copy the closest template and adapt it. All are complete and runnable.

### T1. Continuous variables

```python
import numpy as np
from ga_engine import GeneticAlgorithm, RealRepresentation


def objective(x: np.ndarray) -> float:
    return float(np.sum((x - 1.5) ** 2))           # minimum 0 at x = 1.5


ga = GeneticAlgorithm(
    population_size=50,
    representation=RealRepresentation(dimensions=4, bounds=(-5.0, 5.0)),
    fitness_function=objective,
    rng_seed=1,
)
result = ga.run(max_generations=100)
print(round(result.best_fitness, 3) < 0.05)
```

```output
True
```

### T2. Variables with different ranges (per-gene bounds)

```python
import numpy as np
from ga_engine import GeneticAlgorithm, RealRepresentation

# learning_rate in [1e-4, 1e-1], dropout in [0, 0.8], scale in [10, 1000]
rep = RealRepresentation(dimensions=3, bounds=[(1e-4, 1e-1), (0.0, 0.8), (10.0, 1000.0)])


def objective(p: np.ndarray) -> float:
    lr, dropout, scale = p
    return float((np.log10(lr) + 2.5) ** 2 + (dropout - 0.3) ** 2 + ((scale - 400) / 1000) ** 2)


result = GeneticAlgorithm(60, rep, objective, rng_seed=2).run(max_generations=120)
lr, dropout, scale = result.best_individual.genes
print(1e-4 <= lr <= 1e-1, 0.0 <= dropout <= 0.8, 10 <= scale <= 1000)
```

```output
True True True
```

### T3. Integer variables (inclusive bounds)

```python
import numpy as np
from ga_engine import GeneticAlgorithm, IntegerRepresentation, MutationMethod


def cost(settings: np.ndarray) -> float:
    depth, width = settings
    return float((depth - 7) ** 2 + (width - 40) ** 2)


ga = GeneticAlgorithm(
    population_size=30,
    representation=IntegerRepresentation(dimensions=2, bounds=[(1, 20), (8, 128)]),
    fitness_function=cost,
    mutation_method=MutationMethod.CREEP,     # nudge ordinal values; omit for RANDOM_RESET
    mutation_params={"max_step": 6},
    mutation_prob=0.5,
    rng_seed=3,
)
result = ga.run(max_generations=100, target_fitness=0.0)
print(result.stop_reason, result.best_individual.genes.tolist())
```

```output
target_fitness [7, 40]
```

### T4. Yes/no choices with a constraint (knapsack)

```python
import numpy as np
from ga_engine import BinaryRepresentation, GeneticAlgorithm

values = np.array([60, 100, 120, 80, 30, 70, 90, 50], dtype=float)
weights = np.array([10, 20, 30, 25, 5, 15, 22, 12], dtype=float)
capacity = 70.0


def neg_value(bits):                    # minimise the negative of the value
    return -float(values @ bits)


def overweight(bits):                   # positive = violated, 0 = satisfied
    return max(0.0, float(weights @ bits) - capacity)


ga = GeneticAlgorithm(
    population_size=40,
    representation=BinaryRepresentation(dimensions=len(values)),
    fitness_function=neg_value,
    constraint_function=overweight,
    constraint_penalty=1000.0,          # large: breaking the limit must never pay
    rng_seed=1,
)
result = ga.run(max_generations=80)
print(result.best_individual.is_feasible, -result.best_fitness)
```

```output
True 330.0
```

A genetic algorithm is stochastic: the optimum here is 330, and with these
settings it was found for 15 of 20 tried seeds (the others returned a feasible but
slightly worse solution). For important problems run several seeds or use an
`IslandModel`.

### T5. Ordering problem (tour, schedule): permutations

```python
import numpy as np
from ga_engine import GeneticAlgorithm, PermutationRepresentation


class TourLength:
    """Callable class: picklable, and carries its data."""

    def __init__(self, points):
        self.points = points

    def __call__(self, tour: np.ndarray) -> float:
        ordered = self.points[tour]
        return float(np.sum(np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1)))


points = np.random.default_rng(0).random((12, 2))
ga = GeneticAlgorithm(60, PermutationRepresentation(12), TourLength(points), mutation_prob=0.3, rng_seed=5)
result = ga.run(max_generations=100)
print(sorted(result.best_individual.genes.tolist()) == list(range(12)))   # a valid permutation
```

```output
True
```

### T6. Several objectives (Pareto front)

```python
from ga_engine import GeneticAlgorithm, RealRepresentation


def two_objectives(x):                  # both are minimised
    return float(x[0] ** 2 + x[1] ** 2), float((x[0] - 2) ** 2 + (x[1] - 2) ** 2)


ga = GeneticAlgorithm(
    population_size=60,
    representation=RealRepresentation(2, bounds=(-4.0, 4.0)),
    fitness_function=two_objectives,
    multi_objective=True,               # the objective returns >= 2 numbers
    rng_seed=6,
)
result = ga.run(max_generations=50)
for individual in sorted(result.pareto_front, key=lambda ind: ind.fitness[0])[:2]:
    print(len(individual.fitness))      # fitness is a tuple, one value per objective
```

```output
2
2
```

### T7. Expensive objective: parallel evaluation

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import rastrigin     # module-level function => picklable

if __name__ == "__main__":                     # REQUIRED when n_workers > 1
    with GeneticAlgorithm(
        40, RealRepresentation(5, (-5.12, 5.12)), rastrigin, n_workers=2, rng_seed=1
    ) as ga:                                   # releases the worker pool on exit
        result = ga.run(max_generations=5)
    print(result.evaluations)
```

```output
240
```

### T8. Island model

```python
from ga_engine import GeneticAlgorithm, IslandModel, RealRepresentation, SelectionMethod
from ga_engine.benchmarks import rastrigin


def island(seed, **options):
    return GeneticAlgorithm(
        30, RealRepresentation(5, (-5.12, 5.12)), rastrigin, rng_seed=seed, **options
    )


islands = [island(1), island(2, selection_method=SelectionMethod.RANK), island(3)]
with IslandModel(islands, migration_interval=10, migration_size=2) as model:
    result = model.run(max_generations=30)
print(result.generation, len(result.island_stats))
```

```output
30 3
```

### T9. Checkpoint, resume, warm start, callback

```python
import tempfile
from pathlib import Path

import numpy as np
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import sphere


def build():
    return GeneticAlgorithm(30, RealRepresentation(4, (-5.0, 5.0)), sphere, rng_seed=11)


# Warm start: seed known solutions, then continue with resume=True.
ga = build()
ga.initialize_population(initial_genomes=[np.zeros(4)])


def stop_after_budget(engine):                          # called after every generation
    return engine.evaluations >= 300                    # truthy => stop ("callback")


result = ga.run(max_generations=100, resume=True, on_generation=stop_after_budget)
print(result.stop_reason)

# Checkpoint and resume: the continued run equals an uninterrupted run exactly.
reference = build().run(max_generations=30)
first = build()
first.run(max_generations=10)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "state.json"                  # ".npz" is also supported
    first.save_checkpoint(path)
    second = build()                                    # must be configured identically
    second.load_checkpoint(path)
resumed = second.run(max_generations=30, resume=True)
print(resumed.best_fitness == reference.best_fitness)
```

```output
callback
True
```

### T10. A custom genome (here: k distinct items out of n)

Implement `random_individual`, `crossover`, `mutate`, `repair` (and `validate`).
Set `GENOME_DTYPE`. The engine calls `repair` whenever `validate` is false.

```python
import numpy as np
from ga_engine import GeneticAlgorithm, Representation


class SubsetRepresentation(Representation):
    GENOME_DTYPE = np.dtype(np.int64)

    def __init__(self, k: int, n: int):
        super().__init__(dimensions=k)          # dimensions = genome length
        self.n = n

    def random_individual(self, rng):
        return np.sort(rng.choice(self.n, self.dimensions, replace=False))

    def validate(self, genes):
        g = np.asarray(genes)
        return bool(g.ndim == 1 and len(g) == self.dimensions
                    and np.issubdtype(g.dtype, np.integer)
                    and len(set(g.tolist())) == len(g) and g.min() >= 0 and g.max() < self.n)

    def repair(self, genes):
        kept = [int(v) for v in np.asarray(genes).ravel() if 0 <= v < self.n]
        unique = list(dict.fromkeys(kept))[: self.dimensions]
        missing = [i for i in range(self.n) if i not in set(unique)]
        unique += missing[: self.dimensions - len(unique)]
        return np.sort(np.array(unique, dtype=np.int64))

    def crossover(self, parent1, parent2, method, rng, **kwargs):
        pool = np.union1d(parent1, parent2)
        return (np.sort(rng.choice(pool, self.dimensions, replace=False)),
                np.sort(rng.choice(pool, self.dimensions, replace=False)))

    def mutate(self, individual, method, rng, **kwargs):
        mutated = individual.copy()
        outside = np.setdiff1d(np.arange(self.n), mutated)
        if len(outside):
            mutated[rng.integers(self.dimensions)] = rng.choice(outside)
        return np.sort(mutated)


values = np.random.default_rng(0).random(30)
ga = GeneticAlgorithm(40, SubsetRepresentation(5, 30), lambda s: -float(values[s].sum()),
                      mutation_prob=0.5, rng_seed=1)
result = ga.run(max_generations=100)
print(sorted(result.best_individual.genes.tolist()) == sorted(np.argsort(-values)[:5].tolist()))
```

```output
True
```

## 5. Reference

### 5.1 `GeneticAlgorithm(...)` parameters

The first three are required; the rest are optional. Defaults are checked against
the code by the test suite.

<!-- params:start -->

| Parameter | Default | Meaning |
| --- | --- | --- |
| `population_size` | required | Number of individuals, integer >= 2 |
| `representation` | required | A `Representation` instance |
| `fitness_function` | required | `f(genes)` -> float, or sequence of floats if multi-objective. Minimised |
| `selection_method` | `SelectionMethod.TOURNAMENT` | `TOURNAMENT`, `ROULETTE`, `RANK`, `SUS` |
| `crossover_method` | `None` | `None` = the genome's default; else a `CrossoverMethod` member |
| `mutation_method` | `None` | `None` = the genome's default; else a `MutationMethod` member |
| `replacement_method` | `ReplacementMethod.GENERATIONAL` | `GENERATIONAL` or `STEADY_STATE` |
| `crossover_prob` | `0.8` | Probability a pair is recombined, in [0, 1] |
| `mutation_prob` | `0.1` | Probability an offspring is mutated, in [0, 1] |
| `elitism_count` | `1` | Best parents preserved by generational replacement, 0..population_size |
| `tournament_size` | `3` | Contestants per tournament, >= 1 |
| `rng_seed` | `None` | Integer seed for reproducibility |
| `multi_objective` | `False` | `True` = NSGA-II; the objective must return >= 2 numbers |
| `n_workers` | `0` | `<= 1` serial; `> 1` that many processes; `-1` all CPUs. Needs a module-level objective |
| `adaptive_mutation` | `False` | Adapt mutation intensity with the 1/5 success rule |
| `restart_on_stagnation` | `None` | After N stagnating generations replace the worst ~10 % with random individuals |
| `verbose` | `False` | `True` = console logging at DEBUG level |
| `gene_mutation_prob` | `None` | Per-gene mutation probability; `None` = 1/dimensions |
| `mutation_sigma` | `None` | Gaussian step as a fraction of the gene range; `None` = 0.1 |
| `constraint_function` | `None` | `g(genes)` -> violation (positive = violated) or a sequence of them |
| `constraint_penalty` | `1.0` | Weight of the violation in single-objective comparison; make it large |
| `crossover_params` | `None` | Dict of crossover parameters, e.g. `{"eta": 30.0}` |
| `mutation_params` | `None` | Dict of mutation parameters, e.g. `{"max_step": 5}`; may not contain `mutation_prob` or `sigma` |
| `steady_state_offspring` | `None` | Offspring per generation for `STEADY_STATE`; `None` = max(2, population_size // 10) |
| `adaptation_window` | `10` | Mutation trials needed before an adaptation step |
| `stagnation_tolerance` | `0.0` | Improvements smaller than this do not reset stagnation (single-objective) |
| `evaluation_error_policy` | `EvaluationErrorPolicy.PENALIZE` | `PENALIZE` (failed evaluations get infinite fitness, are counted and logged) or `RAISE` |

<!-- params:end -->

### 5.2 `run(...)` (same arguments on `GeneticAlgorithm` and `IslandModel`)

<!-- run-params:start -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `max_generations` | `500` | Generation budget, >= 0 |
| `stagnation_generations` | `None` | Stop after N generations without improvement (islands: every island) |
| `target_fitness` | `None` | Stop when a feasible solution this good exists; single-objective only |
| `resume` | `False` | Continue the current or loaded population instead of re-initialising |
| `on_generation` | `None` | `f(engine)` (islands: `f(model)`) called after the initial population and after each generation; a truthy return stops the run |

<!-- run-params:end -->

`stop_reason` is one of `"target_fitness"`, `"stagnation"`, `"callback"`,
`"max_generations"`. They are the values of the `StopReason` members
`TARGET_FITNESS`, `STAGNATION`, `CALLBACK` and `MAX_GENERATIONS`, and compare
equal to those strings.

### 5.3 `IslandModel(islands, migration_interval=20, migration_size=2)`

`islands`: at least two `GeneticAlgorithm` objects with identical genome
descriptions and objective mode. `migration_interval`: generations between
migrations (>= 1). `migration_size`: individuals sent per island (1..smallest
population). Ring topology. `run(...)` as above; the result is the best solution
across islands (multi-objective: the merged Pareto front); `result.island_stats`
has per-island statistics. Use as a context manager.

### 5.4 Genomes

| Class | Constructor | Default crossover / mutation |
| --- | --- | --- |
| `RealRepresentation` | `(dimensions, bounds=(-1.0, 1.0))`; `bounds` is `(low, high)` for all genes or `[(low, high), ...]` one pair per gene | `SBX` / `GAUSSIAN` |
| `IntegerRepresentation` | `(dimensions, bounds=(0, 1))`; inclusive integer bounds, shared or per gene | `UNIFORM` / `RANDOM_RESET` |
| `BinaryRepresentation` | `(dimensions)` | `SINGLE_POINT` / `BIT_FLIP` |
| `PermutationRepresentation` | `(dimensions)`; permutation of `0 .. n-1` | `ORDER` / `SWAP` |

Which operators exist for which genome (anything else raises `ConfigurationError`
at construction):

| Genome | Crossover | Mutation |
| --- | --- | --- |
| Real | `SINGLE_POINT`, `TWO_POINT`, `UNIFORM`, `SBX`, `BLX_ALPHA` | `GAUSSIAN`, `POLYNOMIAL` |
| Integer | `SINGLE_POINT`, `TWO_POINT`, `UNIFORM` | `RANDOM_RESET`, `CREEP` |
| Binary | `SINGLE_POINT`, `TWO_POINT`, `UNIFORM` | `BIT_FLIP` |
| Permutation | `ORDER`, `PMX` | `SWAP`, `SCRAMBLE`, `INVERSION` |

Operator parameters (keys of `crossover_params` / `mutation_params`):

| Operator | Parameter | Default | Range |
| --- | --- | --- | --- |
| `SBX` | `eta` | 20 | > 0 (larger = children closer to parents) |
| `BLX_ALPHA` | `alpha` | 0.5 | >= 0 |
| `UNIFORM` | `probability` | 0.5 | [0, 1] |
| `POLYNOMIAL` | `eta_m` | 20 | > 0 |
| `CREEP` | `max_step` | 1 | >= 1 |

Other enum members: `SelectionMethod.TOURNAMENT | ROULETTE | RANK | SUS`;
`ReplacementMethod.GENERATIONAL | STEADY_STATE`;
`EvaluationErrorPolicy.PENALIZE | RAISE`.

### 5.5 Results

`RunResult` fields: `best_individual` (an `Individual`), `best_fitness`,
`generation`, `elapsed_seconds`, `converged` (true for target or stagnation),
`stop_reason`, `stats`, `pareto_front` (multi-objective only, else `None`),
`evaluations`, `evaluation_failures`, `island_stats` (islands only).

`Individual` fields: `genes` (array), `fitness` (float, or tuple in
multi-objective mode), `age`, `crowding_distance`, `constraint_violation`, `rank`;
property `is_feasible`; method `clone()`.

`result.stats` is a dict of equal-length lists, one entry per generation (index 0
is the initial population): `generation`, `best_fitness`, `worst_fitness`,
`mean_fitness`, `median_fitness`, `std_fitness` (these describe the *current
population*), `pareto_front_size`, `feasible_count`, `mean_constraint_violation`,
`diversity`, `mutation_prob`, `gene_mutation_prob`, `mutation_success_ratio`,
`evaluations`, `evaluation_failures` (cumulative). For the best-so-far curve use
`ga.best_fitness_history`.

### 5.6 Engine members you may use

Methods: `run(...)`, `step()`, `initialize_population(initial_genomes=None)`,
`get_best()` -> `(genes, fitness)`, `get_stats()`, `get_pareto_front()`
(multi-objective only), `save_checkpoint(path)`, `load_checkpoint(path)` (suffix
`.json` or `.npz`), `close()`; context manager. Attributes: `population` (list of
`Individual`), `generation`, `best_individual`, `stagnation_count`,
`best_fitness_history`, `evaluations`, `evaluation_failures`, `mutation_prob`,
`config` (a `GAConfig`), `rng`.

### 5.7 Errors

`GAEngineError` is the base. `ConfigurationError` (also a `ValueError`): invalid
settings, raised at construction. `EvaluationError` (also a `RuntimeError`): your
objective failed under the `RAISE` policy, or returned inconsistent objective
counts. `CheckpointError` (also a `ValueError`): corrupt or incompatible
checkpoint; the engine is left untouched.

### 5.8 Logging

`enable_console_logging(level=logging.INFO)` (idempotent) or `verbose=True`. The
library logs to the `"ga_engine"` logger and never configures logging itself.

## 6. Choosing settings

| Situation | Recommendation |
| --- | --- |
| Unsure | Start with defaults; `population_size` 50-100 for ~10 variables; `max_generations` 200-500; `rng_seed` fixed |
| Premature convergence | Larger population, higher `mutation_prob`, `restart_on_stagnation`, or `IslandModel` |
| Slow convergence | Smaller population, `adaptive_mutation=True` |
| Integer genes converge slowly | Raise `mutation_prob` (0.3-0.8); `MutationMethod.CREEP` for ordinal values, `RANDOM_RESET` for categorical. On a small integer test, the optimum was found for 15 of 30 seeds at 0.1 and 28 of 30 at 0.5 |
| Noisy results near the optimum | `adaptive_mutation=True`, smaller `mutation_sigma`, larger SBX `eta` |
| Objective is slow | `n_workers` (module-level function + `__main__` guard), `target_fitness`, `stagnation_generations` |
| Need the trade-off between goals | `multi_objective=True` |
| Several local optima | `IslandModel` with differently configured islands |
| You know good starting points | `initialize_population(initial_genomes=[...])` then `run(resume=True)` |

These are heuristics, not guarantees.

## 7. Common mistakes

| Mistake | Correct |
| --- | --- |
| Treating a score as "higher is better" | Return the negative |
| `result["best_fitness"]` or unpacking `result` | `result.best_fitness`, `result.best_individual.genes` |
| `selection_method="rank"` (string) | `SelectionMethod.RANK` |
| `n_workers=4` with a `lambda` or nested function | Module-level function; add `if __name__ == "__main__":` |
| Leaving `constraint_penalty` at the default `1.0` for a hard limit | Set it large (e.g. `1000.0`) and check `is_feasible` |
| Returning a bare number with `multi_objective=True` | Return a tuple `(f1, f2)` |
| Expecting `target_fitness` to work with `multi_objective=True` | It is single-objective only |
| Passing `mutation_prob` or `sigma` inside `mutation_params` | Use the dedicated arguments |
| `run(resume=True)` with no population | Call `initialize_population()` or `load_checkpoint()` first |
| Loading a checkpoint into an engine with different size/bounds/representation | Build the engine exactly as when saving |
| Comparing steady-state runs by generation count | Compare `result.evaluations` |
| Mutating `genes` inside the objective | Treat it as read-only |

## 8. Things that do not exist

If you are about to write one of these, use the alternative instead.

| Does not exist | Use |
| --- | --- |
| `ga.fit()`, `ga.evolve()`, `ga.optimize()` | `ga.run(...)` |
| `maximize=True`, `minimize=...` | Return the negative of what you maximise |
| `pop_size=`, `generations=`, `mutation_rate=`, `crossover_rate=`, `elitism=`, `seed=`, `n_jobs=` | `population_size`, `max_generations` (in `run`), `mutation_prob`, `crossover_prob`, `elitism_count`, `rng_seed`, `n_workers` |
| `bounds=` on `GeneticAlgorithm` | Bounds belong to the representation: `RealRepresentation(n, bounds=...)` |
| `RealRepresentation(low=..., high=...)` | `RealRepresentation(n, bounds=(low, high))` |
| `callbacks=[...]` (a list) | One hook: `run(on_generation=f)` |
| `result.history`, `result.best_params`, `ga.plot()` | `result.stats`, `result.best_individual.genes`; plot with your own library |
| Mixed real+integer genomes | Encode integers as reals and round inside the objective, or write a `Representation` subclass |
| Asynchronous / distributed islands | Islands run in one process, one after another |

## 9. Checklist when answering a user

1. Confirm what is being optimised, its direction (apply the sign), the variables
   and their ranges, constraints, and the number of objectives.
2. Pick the genome: real / integer / binary / permutation / custom.
3. Write a complete script with imports, the objective, the engine, `run`, and
   how the result is read. Set `rng_seed`.
4. Add `if __name__ == "__main__":` whenever `n_workers > 1`.
5. For constraints: a large `constraint_penalty`, then print `is_feasible`.
6. Mention the stopping criteria you chose (`max_generations`, `target_fitness`,
   `stagnation_generations`) and why.
7. Use only the parameters in section 5. Do not invent options.
8. Prefer defaults; explain any non-default setting in one sentence.
