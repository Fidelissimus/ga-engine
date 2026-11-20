# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-19

First release.

### Added

- `GeneticAlgorithm`: single-population genetic algorithm with generational or
  steady-state replacement, elitism, adaptive mutation, restart on stagnation and
  an `on_generation` hook.
- NSGA-II multi-objective mode with vectorised non-dominated sorting and
  crowding distance; constrained dominance; penalty method for one objective.
- Representations: `RealRepresentation` (shared or per-gene bounds),
  `IntegerRepresentation`, `BinaryRepresentation`, `PermutationRepresentation`,
  and the `Representation` base class for custom genomes.
- Operators: tournament, roulette, rank and stochastic-universal-sampling
  selection; SBX, BLX-alpha, uniform, single-point, two-point, order and PMX
  crossover; Gaussian, polynomial, bit-flip, swap, scramble, inversion, random
  reset and creep mutation. Operator parameters via `crossover_params` and
  `mutation_params`, validated when the engine is created.
- `IslandModel`: ring-topology island model with merged Pareto fronts.
- Process-parallel evaluation (`spawn`, picklable functions, pools released on
  error) and an explicit evaluation failure policy (`PENALIZE` / `RAISE`) with
  `evaluation_failures` counters.
- Checkpoints: strict JSON or NPZ, atomic writes, all-or-nothing loads and exact
  resume (random state included).
- Warm starts with `initialize_population(initial_genomes=...)`.
- Typed error hierarchy: `GAEngineError`, `ConfigurationError`,
  `EvaluationError`, `CheckpointError`.
- Library-style logging (`enable_console_logging`).
- `ga_engine.benchmarks` with picklable test problems (sphere, Rastrigin,
  Rosenbrock, Ackley, ZDT1, OneMax, knapsack, TSP).
- Documentation: user guide, configuration and operator references, checkpoint
  format, architecture, troubleshooting, a generated API reference and
  `LLM_GUIDE.md` for AI assistants.
- A standard-library `unittest` suite that also executes the documentation's code,
  verifies its outputs and links, and checks the generated API reference.
- Type annotations (`py.typed`).
