# ga-engine

**A modular, dependency-light genetic algorithm library for Python.**

Optimise real-valued, integer, binary and permutation problems; handle
constraints and multiple objectives (NSGA-II); evaluate in parallel; coordinate
island populations; and save checkpoints that resume *exactly*. NumPy is the
only dependency.

```python
from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import rastrigin

ga = GeneticAlgorithm(
    population_size=80,
    representation=RealRepresentation(dimensions=10, bounds=(-5.12, 5.12)),
    fitness_function=rastrigin,          # f(genes) -> float, minimised
    rng_seed=1,                          # reproducible
)
result = ga.run(max_generations=300)
print(f"best fitness {result.best_fitness:.3f} after {result.evaluations} evaluations")
```

> **Using an AI assistant?** Give it [`LLM_GUIDE.md`](LLM_GUIDE.md): a single,
> self-contained file that teaches an LLM the whole API, with templates for every
> problem type and a list of mistakes to avoid.

## Contents

- [Why ga-engine](#why-ga-engine)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Features](#features)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Development](#development)
- [Status and limitations](#status-and-limitations)
- [Contributing](#contributing)

## Why ga-engine

- **Small and readable.** Plain Python and NumPy, no hidden framework. Every
  concern (selection, variation, survival, evaluation, adaptation, persistence)
  is a small module you can read in one sitting; see
  [docs/architecture.md](docs/architecture.md).
- **Safe by default.** Operators and parameters are validated when you build the
  engine, not hours into a run. Failures in *your* objective function are
  reported, counted and handled by an explicit policy rather than swallowed.
- **Reproducible.** One seeded random generator drives everything. A run resumed
  from a checkpoint is bit-for-bit identical to an uninterrupted one.
- **Practical.** Per-gene bounds, integer genes, constraints, warm starts,
  progress callbacks, process-parallel evaluation and an island model are built in.

It is **not** a distributed-computing or GPU framework, and it does not include
gradient-based or covariance-matrix methods; see
[Status and limitations](#status-and-limitations).

## Installation

Requires Python **3.9+** and **NumPy 1.22+**. From a clone of this repository:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

NumPy is the single runtime dependency: it provides the vectorised numerics
(population sorting, operators) and the `numpy.random.Generator` that the
reproducibility guarantees are built on.

Without installing, put the project root on `PYTHONPATH`
(PowerShell: `$env:PYTHONPATH = "."`; bash: `export PYTHONPATH=$PWD`).

## Quick start

All objectives are **minimised** (to maximise `f`, return `-f`). An engine needs
a population size, a *representation* (what a solution looks like) and an
objective function:

```python
import numpy as np

from ga_engine import GeneticAlgorithm, IntegerRepresentation


def cost(settings: np.ndarray) -> float:
    depth, width = settings
    return float((depth - 7) ** 2 + (width - 40) ** 2)


ga = GeneticAlgorithm(
    population_size=30,
    representation=IntegerRepresentation(dimensions=2, bounds=[(1, 20), (8, 128)]),
    fitness_function=cost,
    mutation_prob=0.5,        # integer genes usually benefit from more mutation
    rng_seed=3,
)
result = ga.run(max_generations=60, target_fitness=0.0)

print(result.stop_reason)
print(result.best_individual.genes)       # the best settings found
print(result.best_fitness)
```

```output
target_fitness
[ 7 40]
0.0
```

`run()` returns a `RunResult` with the best individual, per-generation
statistics (`result.stats`), evaluation counts and, in multi-objective mode,
the Pareto front. It stops on `max_generations`, a `target_fitness`, a
`stagnation_generations` limit or an `on_generation` callback.

## Examples

Runnable programs are in [`examples/`](examples). Representative output
(NumPy 2.5; exact digits can differ between NumPy versions):

| Script | Shows | Output |
| --- | --- | --- |
| [`real_sphere.py`](examples/real_sphere.py) | Continuous optimisation with adaptive mutation | `stop reason : target_fitness at generation 228` |
| [`integer_hyperparameters.py`](examples/integer_hyperparameters.py) | Discrete settings with per-gene ranges and creep mutation | `tree_depth = 9`, `n_estimators = 340`, `min_samples_leaf = 4` |
| [`binary_knapsack.py`](examples/binary_knapsack.py) | Binary genome with a constraint penalty | `weight : 481 / capacity 483`, `feasible : True` |
| [`permutation_tsp.py`](examples/permutation_tsp.py) | Permutation genome, order crossover, inversion mutation | tour length `10.434` → `4.334` |
| [`multiobjective_zdt1.py`](examples/multiobjective_zdt1.py) | NSGA-II and the Pareto front | `front size : 100`, `mean gap to true front: 0.0791` |
| [`island_rastrigin.py`](examples/island_rastrigin.py) | Three differently configured islands with ring migration | `global best: 0.231 after 200 generations` |
| [`checkpoint_resume.py`](examples/checkpoint_resume.py) | Save, load and resume exactly | `identical : True` |
| [`progress_callback.py`](examples/progress_callback.py) | Progress output, custom budget stop, warm start | `stop reason: callback` |
| [`parallel_evaluation.py`](examples/parallel_evaluation.py) | Process-parallel evaluation | identical results to serial |

Run any of them from the project root, for example
`python examples/integer_hyperparameters.py`.

## Features

| Area | What you get |
| --- | --- |
| Genomes | `RealRepresentation` (shared or per-gene bounds), `IntegerRepresentation`, `BinaryRepresentation`, `PermutationRepresentation`, plus a small abstract class for your own |
| Selection | Tournament, roulette, rank, stochastic universal sampling |
| Crossover | SBX, BLX-α, uniform, single-point, two-point, order (OX), PMX |
| Mutation | Gaussian, polynomial, bit-flip, swap, scramble, inversion, random reset, creep |
| Objectives | Single-objective, or multi-objective with NSGA-II (vectorised non-dominated sorting + crowding distance) |
| Constraints | Penalty method (one objective) and constrained dominance (several) |
| Survival | Generational with elitism, or genuinely steady-state |
| Adaptation | 1/5-success-rule mutation adaptation, restart-on-stagnation, stagnation tolerance |
| Control | Stop on budget / target / stagnation / callback; warm start from known solutions; resume |
| Scaling | Process-parallel evaluation (`spawn`, works on Windows), ring-topology island model |
| Reliability | Atomic strict-JSON or NPZ checkpoints, exact resume, typed errors, explicit failure policy for user code |

## Documentation

| Document | Contents |
| --- | --- |
| [docs/user-guide.md](docs/user-guide.md) | Tutorial-style guide to every capability, with runnable code |
| [docs/configuration.md](docs/configuration.md) | Every option, its default and its validation rule |
| [docs/operators.md](docs/operators.md) | Operators per genome, their parameters and exact semantics |
| [docs/checkpoints.md](docs/checkpoints.md) | Checkpoint guarantees and file format |
| [docs/architecture.md](docs/architecture.md) | Modules, layers, data flow and design decisions |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Error messages and common problems, with fixes |
| [docs/api.md](docs/api.md) | Complete API reference (generated from the source) |
| [LLM_GUIDE.md](LLM_GUIDE.md) | One-file briefing for AI assistants |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

## Project layout

```text
ga-engine/
├── ga_engine/                  the package
│   ├── engine.py               GeneticAlgorithm: wiring and the generation loop
│   ├── islands.py              IslandModel
│   ├── config.py               GAConfig and validation of every setting
│   ├── reproduction.py         parent selection, crossover, mutation
│   ├── evaluation.py           running user code (serial / process pool, failure policy)
│   ├── replacement.py          survivor selection
│   ├── progress.py             best-so-far, stagnation, restart bookkeeping
│   ├── termination.py          stop criteria and reasons
│   ├── adaptation.py           adaptive mutation controller
│   ├── multiobjective.py       NSGA-II: dominance, fronts, crowding
│   ├── selection.py, fitness.py, migration.py, statistics.py
│   ├── snapshots.py            engine <-> checkpoint mapping
│   ├── checkpoint.py           file formats, atomic writes
│   ├── representations/        real, integer, binary, permutation
│   └── benchmarks.py           ready-made test problems
├── tests/                      standard-library unittest suite
├── examples/                   runnable programs
├── docs/                       guides and reference
├── tools/generate_api_docs.py  regenerates docs/api.md
├── LLM_GUIDE.md, llms.txt      material for AI assistants
└── pyproject.toml
```

## Development

```bash
python -m unittest discover -s tests -t .     # run the tests (standard library only)

pip install mypy
mypy                                          # strict static typing (configured in pyproject.toml)

python tools/generate_api_docs.py             # regenerate docs/api.md after changing docstrings
```

The test suite also executes every Python snippet in the documentation and fails
if `docs/api.md` is out of date, so the documentation cannot silently drift from
the code. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status and limitations

Verified while building this release:

- The full test suite passes on Python 3.12.3 with NumPy 2.5.4 (Linux). It covers
  parallel evaluation, atomic writes, corrupt-file handling and bit-exact resume.
- `mypy --strict` reports no issues, all examples run, and every Python block in the
  documentation is executed by the tests.

Not verified:

- Python versions other than 3.12, NumPy versions other than 2.5. Python 3.9 and
  NumPy 1.22 are the declared minimums; the sources contain no syntax newer than
  3.9, but they have not been run there.
- Windows and macOS. Parallel evaluation always uses the `spawn` start method (the
  Windows default), so that code path is exercised, but the atomic checkpoint
  write was only observed on Linux.
- Parallel *speed-up*: the development machine had one CPU core, so only result
  equality with serial evaluation was checked.
- Multi-objective populations beyond a few thousand individuals (non-dominated
  sorting needs O(n² · m) memory).

Design limits: islands run one after another in a single process (each island can
still evaluate in parallel); genomes are homogeneous (all real, all integer, ...);
there is no built-in plotting. Random streams are stable for a given NumPy version,
not guaranteed across versions.

## Contributing

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
Security-relevant reports: [SECURITY.md](SECURITY.md).

No license file is included yet; add one before publishing or redistributing.
