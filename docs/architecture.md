# Architecture

How `ga-engine` is put together, why, and where to change things.

## Design goals

- **Separation of concerns.** Each module answers one question (what is
  "better"? how are parents chosen? who runs user code? when do we stop?) and can
  be tested alone.
- **A thin orchestrator.** `GeneticAlgorithm` owns the population and sequences a
  generation; every algorithmic decision is delegated to a component.
- **Fail early, never silently.** Configuration is validated at construction;
  errors carry context; the only broad exception handler sits at the boundary
  around *your* objective function and reports everything it catches
  ([Error handling](#error-handling)).
- **Determinism.** One `numpy.random.Generator` per engine is the only source of
  randomness, and it is part of every checkpoint.

## Layers

An arrow means "imports". The levels below are computed from the actual imports;
there are no cycles (`snapshots` refers to the engine only for type checking).

```text
 L5  islands
 L4  engine
 L3  evaluation  reproduction  replacement  progress  migration  snapshots
 L2  config  fitness  multiobjective  statistics  checkpoint
     representations/{real, integer, binary, permutation}
 L1  models  selection  termination  representations/base
 L0  enums  errors  aliases  adaptation  logging_utils
```

| Module | Responsibility |
| --- | --- |
| `enums`, `errors`, `aliases`, `logging_utils` | Vocabulary: operator enums, exception hierarchy, type aliases, library logging setup |
| `adaptation` | `AdaptiveMutationController`: owns the current mutation intensity and its 1/5-rule updates |
| `models` | `Individual` (genes, fitness, age, rank, crowding, violation) and `RunResult` |
| `representations/` | Genome creation, crossover, mutation, repair, validation, diversity, operator metadata (`base`, `real`, `integer`, `binary`, `permutation`) |
| `selection` | `ParentSelector`: tournament / roulette / rank / SUS over quality *arrays* |
| `termination` | `StopCriteria` and `StopReason`, shared by the engine and the island model |
| `fitness` | Normalising what user code returns; constraint penalty; the single definition of "better" (`FitnessComparator`) |
| `multiobjective` | Vectorised constrained dominance, non-dominated fronts, crowding distance, NSGA-II ordering |
| `config` | `GAConfig` (frozen) and `build_config`: validation of every setting, resolution of default operators |
| `statistics` | `GenerationStatistics`: one row per generation |
| `checkpoint` | `EngineSnapshot`, strict-JSON / NPZ I/O, atomic writes |
| `evaluation` | `Evaluator`: the only code that runs user functions; serial or process pool; failure policy; counters |
| `reproduction` | `Reproducer`: parent selection, recombination, mutation; reports mutation trials |
| `replacement` | Survivor-selection functions and `SurvivalPolicy`; age bookkeeping |
| `progress` | `ProgressTracker`: best-so-far, stagnation, restart counters |
| `migration` | Choosing emigrants and merging immigrants for the island model |
| `snapshots` | Mapping between a live engine and an `EngineSnapshot` (capture, validate, install) |
| `engine` | `GeneticAlgorithm`: wires the components together and sequences generations |
| `islands` | `IslandModel`: ring migration and merged results; uses only the engine's public API |
| `benchmarks` | Picklable test problems |

The engine depends on the abstract `Representation` only, never on a concrete
genome class, so a new genome type touches no other module. The components are
public attributes of the engine (`ga.reproducer`, `ga.evaluator`, `ga.survival`,
`ga.adaptation`, `ga.progress`, `ga.statistics`, `ga.comparator`), which keeps
them easy to inspect and unit-test.

## One generation

`GeneticAlgorithm.step()`:

```text
 population ──► Reproducer.produce
                  ├─ ParentSelector        parents (quality arrays from FitnessComparator,
                  │                        or NSGA-II rank / crowding)
                  ├─ crossover ──► make_valid ──► mutate ──► make_valid
                  └─ MutationTrial         what each child must beat to count as a success
 child genomes ──► Evaluator.evaluate      repair once more, run user code (serial or pool),
                                           apply the failure policy
 Reproducer.record_trial ──► AdaptiveMutationController
 population + offspring ──► SurvivalPolicy.select ──► next population (ages updated)
 AdaptiveMutationController.update()
 refresh NSGA ranks ──► ProgressTracker.update ──► maybe restart ──► record statistics
```

The number of offspring per step is `GAConfig.offspring_per_generation`: the
population size for generational replacement, a small number for steady-state.

## Key decisions

**One definition of "better".** Selection scores, survival, adaptation success
tests, best-so-far tracking, migration and the island model all use
`FitnessComparator` or the dominance matrix, so constraint handling cannot drift
between components.

**Selection works on arrays.** `ParentSelector` receives `scores` (or `ranks` +
`crowding`) instead of individuals, which keeps it independent of the data model.

**The representation owns validity.** `Representation.make_valid` validates,
repairs, casts to the canonical dtype and re-validates. The engine never needs
to know what "valid" means for a genome, and checkpoint loading uses the same path.

**Vectorised non-dominated sorting.** The dominance relation is one `(n, n)`
boolean matrix computed with NumPy: `O(n² · m)` time and memory. See the limits in
the README.

**Survivors are built from indices.** `SurvivalPolicy` ranks `parents + offspring`
and assembles survivors from the ranking, so a survivor's age follows from where
it came from.

**Restart replaces, it does not compete.** Random newcomers are almost always worse
than a converged population, so letting them compete would discard them at once.
Restart therefore *replaces* the worst individuals (never the best) and keeps a
separate counter, so a stagnation stop still works.

**Parallel evaluation ships functions once.** User functions travel to each
worker through the pool *initialiser*; only genomes are sent per batch. The
`spawn` start method is used on every platform. Pools are drained on success and
terminated on error. Only the objective runs in workers; all randomness stays in
the main process, so parallel and serial runs agree.

**Immutable configuration.** `GAConfig` is a frozen dataclass with read-only
parameter mappings; the engine never re-validates it.

**Two-phase checkpoint loading.** `snapshots.validate_snapshot` inspects the whole
snapshot against the engine and changes nothing; only then does
`install_snapshot` apply it, so a failed load leaves the engine untouched.

## Error handling

| Situation | Behaviour |
| --- | --- |
| Invalid argument, unsupported operator, bad operator parameter | `ConfigurationError` (a `ValueError`) at construction |
| User objective raises or returns something malformed | Policy `PENALIZE` (default): infinite fitness, **one summary warning per batch with the first error message**, counted in `evaluation_failures`. Policy `RAISE`: `EvaluationError` |
| Objective count changes between individuals | `EvaluationError` |
| Corrupt or incompatible checkpoint | `CheckpointError`; the engine is untouched |
| `KeyboardInterrupt`, `SystemExit` | Never caught; a running pool is terminated and the exception propagates |

The package contains exactly one broad `except Exception` that does not re-raise:
in `evaluation._call_user_functions`, the boundary around user code. It returns
the error message to its caller, which logs it, counts it and applies the policy.
A test parses the sources with `ast` and fails if any other broad handler
swallows an exception.

## Extending

- **New genome type:** subclass `Representation`
  ([user guide](user-guide.md#15-writing-your-own-representation)).
- **New crossover / mutation operator:** add an enum member in `enums.py`, support
  it in the relevant representation (including `SUPPORTED_*` and `*_PARAMS`), and
  document it in [operators.md](operators.md).
- **New selection scheme:** add an enum member and a branch in `selection.py`.
- **New survival scheme:** extend `SurvivalPolicy` in `replacement.py`.
- **New stop condition:** extend `StopCriteria` in `termination.py` (it is shared
  by `GeneticAlgorithm.run` and `IslandModel.run`), or use the `on_generation` hook.

## Testing strategy

Standard-library `unittest`, one test module per area (`tests/test_<area>.py`).
Techniques worth knowing before you change something:

- the vectorised dominance matrix is cross-checked against the scalar
  `FitnessComparator` on random data;
- resume tests compare an uninterrupted run with a split-and-resumed one for JSON
  and NPZ, adaptive mutation, restarts and NSGA-II;
- `tests/test_invariants.py` pins cross-cutting guarantees (selection pressure,
  restart behaviour, logging hygiene, strict JSON, no swallowed exceptions);
- `tests/test_docs.py` executes every Python block in the documentation, compares
  `output` blocks with real output, validates links and anchors, and checks that
  `docs/api.md` is current.
