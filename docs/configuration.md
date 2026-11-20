# Configuration reference

Every option of `GeneticAlgorithm(...)`. Validation lives in
[`ga_engine/config.py`](../ga_engine/config.py); an invalid value raises
`ConfigurationError` (a `ValueError`) **at construction time**. The resolved,
immutable settings are available as `ga.config` (a `GAConfig`).

## Required

| Argument | Type | Description |
| --- | --- | --- |
| `population_size` | `int` ≥ 2 | Number of individuals. |
| `representation` | `Representation` | Genome encoding and operators. |
| `fitness_function` | callable | `f(genes) -> float` (or a sequence of floats when `multi_objective=True`). Minimised. |

## Operators and selection

| Argument | Default | Description |
| --- | --- | --- |
| `selection_method` | `TOURNAMENT` | `TOURNAMENT`, `ROULETTE`, `RANK` or `SUS`. |
| `crossover_method` | `None` | `None` selects the representation's default (see [operators](operators.md)). |
| `mutation_method` | `None` | `None` selects the representation's default. |
| `tournament_size` | `3` | Contestants per tournament (capped at the population size). |
| `crossover_params` | `None` | Extra crossover keywords, e.g. `{"eta": 15.0}`. Names are validated against the operator. |
| `mutation_params` | `None` | Extra mutation keywords, e.g. `{"eta_m": 30.0}`. `mutation_prob` and `sigma` are reserved: use the dedicated arguments below. |

## Rates

| Argument | Default | Description |
| --- | --- | --- |
| `crossover_prob` | `0.8` | Probability that a mating pair is recombined (else the children are copies of the parents). Must be in [0, 1]. |
| `mutation_prob` | `0.1` | Probability that an *offspring* is mutated at all. In [0, 1]. |
| `gene_mutation_prob` | `1 / dimensions` | Per-gene mutation probability once an offspring is selected for mutation. In [0, 1]. |
| `mutation_sigma` | `0.1` | Gaussian step as a **fraction of each gene's range** (real genomes; with per-gene bounds every gene gets its own scale). Finite, ≥ 0. |

## Survival

| Argument | Default | Description |
| --- | --- | --- |
| `replacement_method` | `GENERATIONAL` | `GENERATIONAL`: the offspring replace the population, the best `elitism_count` parents displace the worst offspring if better. `STEADY_STATE`: a few offspring per generation compete with the whole population, the best `population_size` survive. Multi-objective runs always use NSGA-II (mu + lambda) survival; `replacement_method` only sets the offspring count there. |
| `elitism_count` | `1` | Elites kept by generational replacement. 0 … `population_size`. |
| `steady_state_offspring` | `max(2, population_size // 10)` | Offspring per generation in `STEADY_STATE` mode (1 … `population_size`). Pass `population_size` for classic (mu + lambda). Ignored for `GENERATIONAL`. |

A "generation" is one `step()`. In steady-state mode it creates
`steady_state_offspring` children, so compare runs by
`RunResult.evaluations`, not by generation count.

## Adaptation and restarts

| Argument | Default | Description |
| --- | --- | --- |
| `adaptive_mutation` | `False` | Adapt mutation probability, per-gene probability and Gaussian step with the 1/5 success rule: more than 20 % of mutated offspring beating their better parent grows the values (×1.05, σ ×1.10), fewer shrinks them (×0.95, σ ×0.90). |
| `adaptation_window` | `10` | Minimum number of mutation trials before an adaptation step; trials accumulate across generations until reached. |
| `restart_on_stagnation` | `None` | After this many generations without improvement (and at least this many since the previous restart), replace the worst ~10 % of the population (at least one, never the best) with random individuals. The stagnation counter is **not** reset, so `stagnation_generations` in `run()` still works. |
| `stagnation_tolerance` | `0.0` | Improvements smaller than this (single-objective) do not reset the stagnation counter. |

Adaptation bounds never exclude the value you start with: they are
`[min(0.001, p0), max(0.95, p0)]` for the mutation probability and analogous
for the others, so an initial `mutation_prob=0` stays 0.

## Constraints

| Argument | Default | Description |
| --- | --- | --- |
| `constraint_function` | `None` | `g(genes)` returning a violation (positive = violated) or a sequence of them (positive parts are summed). `NaN` means maximally infeasible. |
| `constraint_penalty` | `1.0` | Weight of the violation in single-objective comparison: `fitness + penalty * violation`. Finite, ≥ 0; `0` ignores constraints in scalar comparison. Must be large enough that violating a constraint is never profitable. |

## Evaluation

| Argument | Default | Description |
| --- | --- | --- |
| `multi_objective` | `False` | Enable NSGA-II. The objective function must return at least two values, always the same number. |
| `n_workers` | `0` | `<= 1` evaluates serially; `> 1` uses that many worker processes (`spawn`); `-1` means all CPUs. Functions must be picklable. |
| `evaluation_error_policy` | `PENALIZE` | `PENALIZE`: failures get infinite fitness, are logged (one warning per batch) and counted. `RAISE`: abort with `EvaluationError`. |

## Reproducibility and logging

| Argument | Default | Description |
| --- | --- | --- |
| `rng_seed` | `None` | Seed of the engine's private `numpy.random.Generator`. With a seed, a run is exactly reproducible (also across checkpoint/resume). |
| `verbose` | `False` | Attach a console handler to the `ga_engine` logger at DEBUG level. |

## `run()` arguments

| Argument | Default | Description |
| --- | --- | --- |
| `max_generations` | `500` | Generation budget (≥ 0). |
| `stagnation_generations` | `None` | Stop after this many generations without improvement (≥ 1). In multi-objective mode "improvement" means the Pareto front changed. |
| `target_fitness` | `None` | Stop when a *feasible* solution at least this good exists. Single-objective only. |
| `resume` | `False` | Continue the current or loaded population instead of re-initialising. |
| `on_generation` | `None` | Hook `f(engine)` called after the initial population is evaluated and after every generation (not before the first step when `resume=True`). A truthy return value stops the run with `stop_reason == "callback"`. |

Stopping checks run before every generation in this order: target, stagnation,
budget; the callback is consulted right after each generation. `RunResult.stop_reason`
is `"target_fitness"`, `"stagnation"`, `"max_generations"` or `"callback"`
(`converged` is true only for the first two).

`IslandModel.run` takes the same five arguments; its callback receives the
model, and "stagnation" means that *every* island stagnated.

## Other entry points

| Method | Purpose |
| --- | --- |
| `initialize_population(initial_genomes=None)` | Start a new run; optionally seed known solutions (`initial_genomes`, at most `population_size`, each repaired). |
| `step()` | Advance one generation manually. |
| `save_checkpoint(path)` / `load_checkpoint(path)` | Persist / restore the run state ([checkpoints](checkpoints.md)). |

## Statistics keys

`RunResult.stats` / `ga.get_stats()` return lists with one entry per
generation (index 0 is the initial population):

`generation`, `best_fitness`, `worst_fitness`, `mean_fitness`,
`median_fitness`, `std_fitness` (current population, raw fitness; tuples per
objective in multi-objective mode), `pareto_front_size`, `feasible_count`,
`mean_constraint_violation`, `diversity`, `mutation_prob`,
`gene_mutation_prob`, `mutation_success_ratio`, `evaluations`
(cumulative), `evaluation_failures` (cumulative).

The best-so-far trajectory is `ga.best_fitness_history`.
