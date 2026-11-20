# Troubleshooting

- [Error messages](#error-messages)
- [Behaviour that looks wrong but is not an error](#behaviour-that-looks-wrong-but-is-not-an-error)
- [Performance and memory](#performance-and-memory)
- [Reporting a problem](#reporting-a-problem)

## Error messages

Messages you may see, what they mean and how to fix them. The fragments in the
first column are taken from the source and verified by the test suite.

### Configuration (raised when the engine is created)

| Message fragment | Cause | Fix |
| --- | --- | --- |
| `does not support crossover` | The crossover (or, analogously, mutation) is not available for this genome | See the [compatibility table](operators.md#compatibility); or leave `crossover_method` / `mutation_method` unset for defaults |
| `parameter(s)` | An operator parameter name is wrong for the chosen operator, e.g. `alpha` with SBX | Check the parameter list in [operators.md](operators.md); SBX takes `eta`, BLX `alpha`, uniform `probability`, polynomial `eta_m`, creep `max_step` |
| `mutation_params may not set` | You passed `mutation_prob` or `sigma` inside `mutation_params` | Use the dedicated arguments `mutation_prob`, `gene_mutation_prob`, `mutation_sigma` |
| `must be an integer` | A count such as `population_size` was a float or a bool | Pass an `int` (`50`, not `50.0`) |
| `target_fitness is supported only for single-objective optimization` | `target_fitness` with `multi_objective=True` | Stop on `max_generations`, `stagnation_generations` or an `on_generation` callback instead |
| `integer bounds must be whole numbers` | `IntegerRepresentation` got bounds such as `(0.5, 3)` | Use whole numbers, e.g. `(0, 3)` |
| `every gene needs lower < upper` | A bounds pair has `low >= high` | Fix the pair; each gene needs a non-empty range |
| `IslandModel requires at least two islands` | Fewer than two engines were given | Provide two or more `GeneticAlgorithm` objects |
| `uses a different genome description` | Islands differ in representation type, dimensions or bounds | Make every island use an identical representation |
| `requires a picklable fitness_function` | `n_workers > 1` with a lambda, closure or local function | Define the function at module level (and guard the script with `if __name__ == "__main__":`) |
| `Use module-level functions when n_workers > 1.` | Same as above (second half of the message) | Same fix |

### Running

| Message fragment | Cause | Fix |
| --- | --- | --- |
| `resume=True requires an initialized or loaded population` | `run(resume=True)` before any population exists | Call `initialize_population()` or `load_checkpoint()` first, or drop `resume=True` |
| `Generational replacement requires exactly` | `replacement()` was called with the wrong number of offspring | Pass exactly `population_size` offspring, or use `ReplacementMethod.STEADY_STATE` |
| `initial genomes for a population of` | `initialize_population` got more seeds than `population_size` | Pass at most `population_size` genomes |
| `genome must be one-dimensional with` | A genome (e.g. a warm-start seed) has the wrong length or shape | Match the representation's `dimensions` |
| `Pareto front only available in multi-objective mode` | `get_pareto_front()` on a single-objective engine | Use `get_best()` / `result.best_individual` |

### Evaluating your objective function

| Message fragment | Cause | Fix |
| --- | --- | --- |
| `fitness evaluations failed; first error:` | Your function raised or returned something invalid (this is the *warning* of the default `PENALIZE` policy, or the message of `EvaluationError` under `RAISE`) | Fix the function; the text after the colon is the first underlying error. Failed evaluations are counted in `result.evaluation_failures` |
| `All individuals must return the same number of objectives` | A multi-objective function returned 2 values for one genome and 3 for another | Always return the same number of objectives |
| `Multi-objective fitness must be a sequence of at least two numbers` | `multi_objective=True` but the function returned a single number | Return a tuple/list such as `(f1, f2)` |
| `Every multi-objective fitness evaluation failed` | The first batch failed completely, so the number of objectives cannot be learned | Fix the function; test it on a random genome first |

### Checkpoints

| Message fragment | Cause | Fix |
| --- | --- | --- |
| `Checkpoint path must end with .npz or .json` | Unsupported file suffix | Use `.json` or `.npz` |
| `Cannot checkpoint before population initialization` | `save_checkpoint` before any population exists | Run or initialise first |
| `Malformed checkpoint` | The file is corrupt, truncated or not a checkpoint | Restore it from a backup; writes are atomic, so an interrupted save does not damage the previous file |
| `Unsupported checkpoint version` | The file was written by an incompatible format version | Use a checkpoint written by a supported version (see [checkpoints.md](checkpoints.md#version-history)) |
| `Checkpoint population_size` | The engine's `population_size` differs from the one saved | Build the engine with the same `population_size` as the saved run |
| `Checkpoint contains a genome that is invalid for this representation` | The engine's bounds, dimensions or genome type differ from the saved run | Rebuild the engine exactly as it was when the checkpoint was saved |

A failed `load_checkpoint` never changes the engine, so you can catch
`CheckpointError` and fall back to a fresh start.

## Behaviour that looks wrong but is not an error

| Symptom | Explanation | What to do |
| --- | --- | --- |
| `best_individual.is_feasible` is `False` | With a penalty, breaking a constraint can pay more than it costs | Increase `constraint_penalty` until violating never pays; always check feasibility |
| The best fitness *rises* between generations in `result.stats` | `stats["best_fitness"]` describes the current population (no elitism guarantee with `elitism_count=0`) | Use `ga.best_fitness_history` for the best-so-far curve |
| Steady-state runs "finish" in few evaluations | A generation creates only `steady_state_offspring` children | Compare by `result.evaluations`, or raise `max_generations` |
| `stop_reason` is `"max_generations"` but you expected stagnation | `stagnation_generations` defaults to off | Pass `stagnation_generations=N` |
| Tiny improvements keep resetting stagnation | Any strict improvement counts by default | Set `stagnation_tolerance` (single-objective) |
| Different numbers than in the documentation | Random streams are stable for a given NumPy version only | Compare qualitative results, or pin the NumPy version |
| Results differ between two runs with the same seed | Your objective function is non-deterministic, or you changed an option | Make the function deterministic; keep all options identical |
| `evaluations` is larger than `population_size * generations` | The initial population and restarts are evaluated too | Expected: `evaluations = population_size + generations * offspring_per_generation` (+ restarts) |

## Performance and memory

- **Parallel evaluation is slower than serial.** Starting processes and sending
  genomes costs time. It only pays off when one evaluation takes much longer than
  that overhead, and when the machine has more than one physical core.
- **Script re-runs itself or hangs on Windows/macOS with `n_workers > 1`.** Workers
  are started with `spawn`, which re-imports your script. Put the entry point under
  `if __name__ == "__main__":`.
- **Multi-objective runs get slow or use a lot of memory for large populations.**
  Non-dominated sorting needs `O(n² · m)` time and memory for `n` individuals and
  `m` objectives. Use a smaller population or split the problem across islands.
- **Per-generation overhead dominates for very cheap objectives.** Statistics
  (including a diversity measure) are computed every generation; this is small
  compared to any non-trivial objective.

## Reporting a problem

Open an issue with a minimal script that reproduces the problem, the output of
`python -c "import sys, numpy, ga_engine; print(sys.version, numpy.__version__, ga_engine.__version__)"`,
and the full traceback. For suspected security issues see [SECURITY.md](../SECURITY.md).
