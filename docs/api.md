# API reference

> Generated from the source by `tools/generate_api_docs.py`. Do not edit by hand;
> run `python tools/generate_api_docs.py` after changing a docstring or signature.

Everything listed here is importable from `ga_engine` (test problems from `ga_engine.benchmarks`).

## Contents

- **Engines**: [`GeneticAlgorithm`](#class-geneticalgorithm), [`IslandModel`](#class-islandmodel)
- **Configuration**: [`GAConfig`](#class-gaconfig)
- **Representations**: [`Representation`](#class-representation), [`RealRepresentation`](#class-realrepresentation), [`IntegerRepresentation`](#class-integerrepresentation), [`BinaryRepresentation`](#class-binaryrepresentation), [`PermutationRepresentation`](#class-permutationrepresentation)
- **Results and data**: [`RunResult`](#class-runresult), [`Individual`](#class-individual), [`StopReason`](#enum-stopreason)
- **Choices**: [`SelectionMethod`](#enum-selectionmethod), [`CrossoverMethod`](#enum-crossovermethod), [`MutationMethod`](#enum-mutationmethod), [`ReplacementMethod`](#enum-replacementmethod), [`EvaluationErrorPolicy`](#enum-evaluationerrorpolicy)
- **Errors**: [`GAEngineError`](#exception-gaengineerror), [`ConfigurationError`](#exception-configurationerror), [`EvaluationError`](#exception-evaluationerror), [`CheckpointError`](#exception-checkpointerror)
- **Utilities**: [`enable_console_logging`](#enable_console_logging)
- **Test problems** (`ga_engine.benchmarks`): [`sphere`](#sphere), [`rastrigin`](#rastrigin), [`rosenbrock`](#rosenbrock), [`ackley`](#ackley), [`zdt1`](#zdt1), [`onemax`](#onemax), [`KnapsackProblem`](#class-knapsackproblem), [`TspProblem`](#class-tspproblem)

## Engines

### class `GeneticAlgorithm`

*Defined in `ga_engine.engine`.*

```python
GeneticAlgorithm(
    population_size: int,
    representation: Representation,
    fitness_function: ObjectiveFunction,
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
    crossover_method: Optional[CrossoverMethod] = None,
    mutation_method: Optional[MutationMethod] = None,
    replacement_method: ReplacementMethod = ReplacementMethod.GENERATIONAL,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.1,
    elitism_count: int = 1,
    tournament_size: int = 3,
    rng_seed: Optional[int] = None,
    multi_objective: bool = False,
    n_workers: int = 0,
    adaptive_mutation: bool = False,
    restart_on_stagnation: Optional[int] = None,
    verbose: bool = False,
    gene_mutation_prob: Optional[float] = None,
    mutation_sigma: Optional[float] = None,
    constraint_function: Optional[ConstraintFunction] = None,
    constraint_penalty: float = 1.0,
    *,
    crossover_params: Optional[Mapping[str, Any]] = None,
    mutation_params: Optional[Mapping[str, Any]] = None,
    steady_state_offspring: Optional[int] = None,
    adaptation_window: int = 10,
    stagnation_tolerance: float = 0.0,
    evaluation_error_policy: EvaluationErrorPolicy = EvaluationErrorPolicy.PENALIZE,
) -> None
```

Single-population genetic algorithm with optional NSGA-II mode.

The engine can be used as a context manager; leaving the ``with`` block
shuts down any worker pool (aborting it if an exception is propagating):

```python
with GeneticAlgorithm(100, RealRepresentation(10, (-5, 5)), sphere) as ga:
    result = ga.run(max_generations=200)
```

The engine is not thread-safe.

**Args**

- `population_size`: Number of individuals (>= 2).
- `representation`: Genome encoding and operators.
- `fitness_function`: ``f(genes)`` -> float, or a sequence of floats when ``multi_objective`` is true. Minimised.
- `selection_method`: Parent-selection strategy.
- `crossover_method`: Recombination operator; ``None`` uses the representation's default.
- `mutation_method`: Mutation operator; ``None`` uses the representation's default.
- `replacement_method`: ``GENERATIONAL`` replaces the whole population; ``STEADY_STATE`` creates only a few offspring per generation that compete with the population (see ``steady_state_offspring``).
- `crossover_prob`: Probability a mating pair is recombined.
- `mutation_prob`: Probability an offspring is mutated.
- `elitism_count`: Elites kept by generational replacement.
- `tournament_size`: Contestants per tournament.
- `rng_seed`: Seed for the engine's private random generator.
- `multi_objective`: Enable NSGA-II (``fitness_function`` returns a sequence).
- `n_workers`: Processes for parallel evaluation; ``<= 1`` is serial and ``-1`` uses all CPUs. Functions must be picklable (module level).
- `adaptive_mutation`: Adapt mutation intensity with the 1/5 success rule.
- `restart_on_stagnation`: After this many stagnating generations, replace the worst ~10 % of the population with random individuals.
- `verbose`: Attach console logging at DEBUG level.
- `gene_mutation_prob`: Per-gene mutation probability (default ``1/dimensions``).
- `mutation_sigma`: Gaussian step size as a fraction of the gene range.
- `constraint_function`: ``g(genes)`` -> violation (positive = violated) or a sequence of violations.
- `constraint_penalty`: Weight of violations in single-objective comparison.
- `crossover_params`: Extra crossover keyword arguments, e.g. ``{"eta": 15}``.
- `mutation_params`: Extra mutation keyword arguments, e.g. ``{"eta_m": 30}``.
- `steady_state_offspring`: Offspring per generation in ``STEADY_STATE`` mode (default ``max(2, population_size // 10)``; pass ``population_size`` for a classic (mu + lambda) strategy).
- `adaptation_window`: Minimum mutation trials before adapting.
- `stagnation_tolerance`: Improvements smaller than this do not reset the stagnation counter (single-objective only).
- `evaluation_error_policy`: ``PENALIZE`` (default) or ``RAISE`` when user code fails.

**Attributes**

- `config`: The resolved, immutable `GAConfig`.
- `population`: Current individuals (empty until initialised).
- `generation`: Number of completed generations.
- `rng`: The engine's ``numpy.random.Generator``.

**Raises**

- `ConfigurationError`: If any argument is invalid or the representation does not support the requested operators.

**Properties**

- `best_individual`: Best (single-objective) or representative (multi-objective) solution so far.
- `stagnation_count`: Generations since the last (significant) improvement.
- `best_fitness_history`: Best-so-far fitness after every generation (index 0 = initial population).
- `mutation_prob`: Current probability that an offspring is mutated (may adapt).
- `gene_mutation_prob`: Current per-gene mutation probability (may adapt).
- `stats`: Live statistics dictionary; use `get_stats` for a safe copy.
- `evaluations`: Fitness evaluations performed since the population was initialised.
- `evaluation_failures`: How many of those evaluations failed.

#### `GeneticAlgorithm.evaluation_pool`

```python
evaluation_pool() -> Iterator[None]
```

Keep the worker pool alive for the duration of a ``with`` block.

The pool is started if parallel evaluation is configured and not yet
running, and shut down on exit only if this call started it. On error
the pool is terminated instead of drained. ``run()`` uses this
internally; call it yourself to share one pool across several steps.

#### `GeneticAlgorithm.close`

```python
close() -> None
```

Release the worker pool, if any. Safe to call repeatedly.

#### `GeneticAlgorithm.effective_fitness`

```python
effective_fitness(individual: Individual) -> float
```

Penalised scalar fitness used for single-objective comparison.

#### `GeneticAlgorithm.initialize_population`

```python
initialize_population(initial_genomes: Optional[Sequence[np.ndarray]] = None) -> None
```

Create and evaluate a population, starting a new run.

**Args**

- `initial_genomes`: Optional starting genomes (for example known good solutions). At most ``population_size`` are used; the remaining slots are filled randomly. Each is repaired into validity.

**Raises**

- `ConfigurationError`: If more genomes than ``population_size`` are given, or a genome has the wrong length.

#### `GeneticAlgorithm.select_parents`

```python
select_parents() -> Tuple[Individual, Individual]
```

Select two distinct parents and return clones of them.

**Raises**

- `RuntimeError`: If the population has not been initialised.

#### `GeneticAlgorithm.create_offspring`

```python
create_offspring(parent1: Individual, parent2: Individual) -> List[Individual]
```

Create, evaluate and return two children of the given parents.

This is the low-level building block of `step`; it also feeds the
adaptive-mutation statistics. It does not touch the population.

#### `GeneticAlgorithm.replacement`

```python
replacement(offspring: Sequence[Individual]) -> None
```

Form the next population from the current one and ``offspring``.

See `SurvivalPolicy` for the rules.

**Raises**

- `ConfigurationError`: If the offspring count does not suit the scheme.

#### `GeneticAlgorithm.refresh_ranking`

```python
refresh_ranking() -> List[List[int]]
```

Recompute NSGA-II ranks and crowding distances; returns the fronts.

#### `GeneticAlgorithm.step`

```python
step() -> None
```

Advance the search by one generation (initialising it if necessary).

In ``STEADY_STATE`` mode a "generation" is one step that creates only
``steady_state_offspring`` children, so many more generations are needed
for the same number of evaluations.

#### `GeneticAlgorithm.target_reached`

```python
target_reached(target_fitness: float) -> bool
```

Whether the best solution is feasible and at least as good as the target.

#### `GeneticAlgorithm.run`

```python
run(
    max_generations: int = 500,
    stagnation_generations: Optional[int] = None,
    target_fitness: Optional[float] = None,
    resume: bool = False,
    on_generation: Optional[GenerationCallback] = None,
) -> RunResult
```

Evolve until a stopping criterion is met.

**Args**

- `max_generations`: Generation budget (``>= 0``); the run stops when ``generation`` reaches it.
- `stagnation_generations`: Stop after this many generations without improvement (``>= 1``), or ``None`` to disable.
- `target_fitness`: Stop once a feasible solution at least this good is found (single-objective only).
- `resume`: Continue the current/loaded population instead of initialising a new one.
- `on_generation`: Optional hook called with the engine after the initial population is evaluated and after every generation. Return a truthy value to stop the run (``stop_reason`` is then ``"callback"``). Use it for progress output or custom criteria.

**Returns**

- A `RunResult`. ``stop_reason`` is ``"target_fitness"``, ``"stagnation"``, ``"callback"`` or ``"max_generations"``.

**Raises**

- `ConfigurationError`: On invalid arguments.
- `RuntimeError`: If ``resume`` is requested without a population.
- `EvaluationError`: Under the ``RAISE`` failure policy.

#### `GeneticAlgorithm.get_best`

```python
get_best() -> Tuple[np.ndarray, Fitness]
```

Return ``(genes, fitness)`` of the best (or representative) solution.

**Raises**

- `RuntimeError`: If no population exists yet.

#### `GeneticAlgorithm.get_stats`

```python
get_stats() -> Dict[str, List[Any]]
```

Return a deep copy of the accumulated generation statistics.

#### `GeneticAlgorithm.get_pareto_front`

```python
get_pareto_front() -> List[Individual]
```

Return clones of the current non-dominated set.

**Raises**

- `ConfigurationError`: If the engine is not in multi-objective mode.

#### `GeneticAlgorithm.select_migrants`

```python
select_migrants(count: int) -> List[Individual]
```

Return clones of the ``count`` best individuals for emigration.

#### `GeneticAlgorithm.accept_migrants`

```python
accept_migrants(migrants: Sequence[Individual]) -> None
```

Merge immigrants in, keeping the best ``population_size`` individuals.

#### `GeneticAlgorithm.save_checkpoint`

```python
save_checkpoint(path: PathLike) -> None
```

Persist the complete run state to a ``.json`` or ``.npz`` file.

The write is atomic. The fitness function and the engine settings are
*not* stored; see ``docs/checkpoints.md``.

**Raises**

- `CheckpointError`: If the suffix is unsupported or the population is empty.

#### `GeneticAlgorithm.load_checkpoint`

```python
load_checkpoint(path: PathLike) -> None
```

Restore run state from a checkpoint written by `save_checkpoint`.

The engine must be configured compatibly (population size,
representation type, dimensions, bounds, objective mode). Loading is
all-or-nothing: the file is parsed and validated completely before any
engine state changes. Continue with ``run(..., resume=True)``.

**Raises**

- `CheckpointError`: If the file is corrupt or incompatible.
- `FileNotFoundError`: If the file does not exist.

### class `IslandModel`

*Defined in `ga_engine.islands`.*

```python
IslandModel(
    islands: Sequence[GeneticAlgorithm],
    migration_interval: int = 20,
    migration_size: int = 2,
) -> None
```

Ring-topology island model.

The islands must share the same genome description (representation type,
dimensions, bounds) and objective mode so individuals can migrate between
them. They may differ in everything else (operators, rates, population
size, seeds), which is the usual way to add diversity.

**Args**

- `islands`: At least two engines.
- `migration_interval`: Generations between migrations (>= 1).
- `migration_size`: Individuals each island sends per migration; must not exceed the smallest island population.

**Raises**

- `ConfigurationError`: If the islands are incompatible or the migration settings are invalid.

**Properties**

- `generation`: Generation counter of the islands (they advance in lock-step).

#### `IslandModel.close`

```python
close() -> None
```

Release every island's worker pool.

#### `IslandModel.run`

```python
run(
    max_generations: int = 500,
    stagnation_generations: Optional[int] = None,
    target_fitness: Optional[float] = None,
    resume: bool = False,
    on_generation: Optional[IslandCallback] = None,
) -> RunResult
```

Evolve all islands in lock-step with periodic migration.

**Args**

- `max_generations`: Generation budget per island (``>= 0``).
- `stagnation_generations`: Stop when *every* island has stagnated for this many generations.
- `target_fitness`: Stop when *any* island reaches the target (single-objective only).
- `resume`: Continue the islands' current populations (they must all be initialised and at the same generation).
- `on_generation`: Optional hook called with this model after initialisation and after every generation (and its migration). Return a truthy value to stop (``stop_reason == "callback"``).

**Returns**

- A `RunResult`. Single-objective: the best solution over all islands. Multi-objective: the non-dominated set of the union of all island fronts. ``island_stats`` holds each island's statistics.

**Raises**

- `ConfigurationError`: On invalid arguments.
- `RuntimeError`: If ``resume`` is requested but the islands are not initialised or are out of sync.

## Configuration

### class `GAConfig`

*Defined in `ga_engine.config`.*

```python
GAConfig(
    population_size: int,
    selection_method: SelectionMethod,
    crossover_method: CrossoverMethod,
    mutation_method: MutationMethod,
    replacement_method: ReplacementMethod,
    crossover_prob: float,
    mutation_prob: float,
    gene_mutation_prob: float,
    elitism_count: int,
    tournament_size: int,
    multi_objective: bool,
    n_workers: int,
    adaptive_mutation: bool,
    restart_on_stagnation: Optional[int],
    mutation_sigma: float,
    constraint_penalty: float,
    crossover_params: Mapping[str, Any],
    mutation_params: Mapping[str, Any],
    offspring_per_generation: int,
    adaptation_window: int,
    stagnation_tolerance: float,
    evaluation_error_policy: EvaluationErrorPolicy,
    verbose: bool,
) -> None
```

Resolved settings of a `GeneticAlgorithm`.

**Attributes**

- `population_size`: Number of individuals.
- `selection_method`: Parent-selection strategy.
- `crossover_method`: Resolved recombination operator.
- `mutation_method`: Resolved mutation operator.
- `replacement_method`: Survivor-selection scheme.
- `crossover_prob`: Probability that a mating pair is recombined.
- `mutation_prob`: Initial probability that an offspring is mutated.
- `gene_mutation_prob`: Initial per-gene mutation probability.
- `elitism_count`: Elites preserved by generational replacement.
- `tournament_size`: Contestants per tournament.
- `multi_objective`: Whether NSGA-II mode is used.
- `n_workers`: Worker processes for evaluation (``<= 1`` = serial).
- `adaptive_mutation`: Whether mutation intensity adapts online.
- `restart_on_stagnation`: Stagnating generations before random individuals are injected (``None`` disables).
- `mutation_sigma`: Initial Gaussian step as a fraction of the gene range.
- `constraint_penalty`: Weight of constraint violation in scalar fitness.
- `crossover_params`: Extra keyword arguments for the crossover operator.
- `mutation_params`: Extra keyword arguments for the mutation operator.
- `offspring_per_generation`: Offspring created (and evaluated) per step.
- `adaptation_window`: Minimum mutation trials before an adaptation step.
- `stagnation_tolerance`: Minimum improvement that resets stagnation.
- `evaluation_error_policy`: Handling of failing user functions.
- `verbose`: Whether to attach console logging.

## Representations

### class `Representation`

*Defined in `ga_engine.representations.base`.*

```python
Representation(dimensions: int, bounds: Optional[Tuple[float, float]] = None)
```

Interface between the engine and a genome encoding.

Subclasses implement genome creation, recombination, mutation and repair.
They may also declare capability metadata (the ``SUPPORTED_*``,
``DEFAULT_*`` and ``*_PARAMS`` class variables) so the engine can reject an
unsupported operator at construction time instead of mid-run, and pick
sensible default operators. Leaving the metadata as ``None`` opts out of
those checks, so a minimal custom subclass works without declaring any
metadata.

**Attributes**

- `dimensions`: Genome length.
- `bounds`: Optional ``(lower, upper)`` gene bounds.

#### `Representation.random_individual`

```python
random_individual(rng: np.random.Generator) -> np.ndarray
```

Generate one valid random genome.

#### `Representation.crossover`

```python
crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: CrossoverMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]
```

Create two children from two parents (inputs are never modified).

#### `Representation.mutate`

```python
mutate(
    individual: np.ndarray,
    method: MutationMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> np.ndarray
```

Return a mutated copy of a genome (the input is never modified).

#### `Representation.repair`

```python
repair(individual: np.ndarray) -> np.ndarray
```

Return a genome repaired to satisfy the hard representation rules.

#### `Representation.validate`

```python
validate(individual: np.ndarray) -> bool
```

Return whether a genome already satisfies the hard rules.

#### `Representation.make_valid`

```python
make_valid(genes: np.ndarray) -> np.ndarray
```

Return an owned, valid genome of the canonical dtype.

A genome that already passes `validate` is only copied (and cast
to `GENOME_DTYPE`); otherwise `repair` runs first.

**Args**

- `genes`: Any array-like genome.

**Returns**

- A new array that satisfies `validate`.

**Raises**

- `ConfigurationError`: If the genome is not a 1-D array of ``dimensions`` genes, or repair cannot produce a valid genome (a faulty representation).

#### `Representation.absolute_sigma`

```python
absolute_sigma(fraction: float) -> float
```

Translate a relative Gaussian step size into gene units.

**Args**

- `fraction`: Step size as a fraction of the gene range.

**Returns**

- The absolute standard deviation. The default is the identity; bounded real-valued representations scale by the range.

#### `Representation.diversity`

```python
diversity(genes: np.ndarray) -> float
```

Population diversity in ``[0, 1]`` for a ``(population, genes)`` matrix.

The default is the mean positional mismatch (normalised Hamming
distance) against up to ~20 reference rows, which keeps the cost
linear in the population size.

#### `Representation.describe`

```python
describe() -> Dict[str, Any]
```

Return a JSON-friendly description used to validate checkpoints.

#### `Representation.check_operator_config`

```python
check_operator_config(
    crossover: CrossoverMethod,
    crossover_params: Mapping[str, Any],
    mutation: MutationMethod,
    mutation_params: Mapping[str, Any],
) -> None
```

Fail fast if an operator or one of its parameters is unusable.

**Raises**

- `ConfigurationError`: If a method is unsupported, a parameter name is unknown for the method, or a parameter value is out of range.

### class `RealRepresentation`

*Defined in `ga_engine.representations.real`.*

```python
RealRepresentation(dimensions: int, bounds: Bounds = (-1.0, 1.0))
```

Bounded real-valued vector.

Crossover: single-point, two-point, uniform, SBX (default) and BLX-alpha.
Mutation: Gaussian (default) and polynomial.

Operator parameters (passed through the engine's ``crossover_params`` /
``mutation_params``): ``eta`` (SBX distribution index, default 20),
``alpha`` (BLX-alpha expansion, default 0.5), ``probability`` (uniform
crossover, default 0.5) and ``eta_m`` (polynomial mutation index,
default 20).

**Args**

- `dimensions`: Number of genes.
- `bounds`: Either one finite ``(lower, upper)`` pair shared by every gene (e.g. ``(-5.0, 5.0)``), or a sequence of ``dimensions`` such pairs (e.g. ``[(0, 1), (-10, 10), (100, 500)]``). Each needs ``lower < upper``.

**Attributes**

- `lower_bounds`: Per-gene lower bounds (array of length ``dimensions``).
- `upper_bounds`: Per-gene upper bounds.

**Raises**

- `ConfigurationError`: On invalid ``dimensions`` or ``bounds``.

#### `RealRepresentation.describe`

```python
describe() -> Dict[str, Any]
```

Describe type, dimensions and bounds (a pair, or one pair per gene).

#### `RealRepresentation.random_individual`

```python
random_individual(rng: np.random.Generator) -> np.ndarray
```

Draw a genome uniformly from the bounds.

#### `RealRepresentation.repair`

```python
repair(individual: np.ndarray) -> np.ndarray
```

Clip to the bounds; NaN becomes the midpoint, ±inf the nearest bound.

#### `RealRepresentation.validate`

```python
validate(individual: np.ndarray) -> bool
```

Return whether the genome is 1-D, right-sized, finite and within bounds.

#### `RealRepresentation.absolute_sigma`

```python
absolute_sigma(fraction: float) -> Any
```

Scale a relative Gaussian step by each gene's range (returns an array).

#### `RealRepresentation.diversity`

```python
diversity(genes: np.ndarray) -> float
```

Mean per-gene standard deviation relative to the gene range.

#### `RealRepresentation.crossover`

```python
crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: CrossoverMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]
```

Recombine two parents.

Single-point and two-point crossover *exchange* gene segments (the
textbook operators); they do not blend values.

**Raises**

- `ConfigurationError`: For an unsupported method or invalid parameter.

#### `RealRepresentation.mutate`

```python
mutate(
    individual: np.ndarray,
    method: MutationMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> np.ndarray
```

Mutate a copy of ``individual``.

**Keyword Args**

- `mutation_prob`: Per-gene mutation probability (default ``1/dimensions``).
- `sigma`: Absolute Gaussian standard deviation, a number or one value per gene (default 10 % of each gene's range).
- `eta_m`: Polynomial-mutation distribution index (default 20).

**Raises**

- `ConfigurationError`: For an unsupported method or invalid parameter.

### class `IntegerRepresentation`

*Defined in `ga_engine.representations.integer`.*

```python
IntegerRepresentation(dimensions: int, bounds: Bounds = (0, 1))
```

Vector of bounded integers (both bounds inclusive).

Typical uses: tree depth, number of layers, discrete settings. Genes are
independent; each may have its own range.

Crossover: single-point, two-point and uniform (default uniform).
Mutation: random reset (default) and creep.

Operator parameters: ``probability`` for uniform crossover (default 0.5)
and ``max_step`` for creep mutation (default 1; the step is drawn uniformly
from ``1 .. max_step`` and its sign at random).

**Args**

- `dimensions`: Number of genes.
- `bounds`: One inclusive ``(low, high)`` pair of integers shared by every gene, e.g. ``(1, 10)``, or ``dimensions`` such pairs, e.g. ``[(1, 10), (0, 3), (50, 500)]``. Each needs ``low < high``.

**Attributes**

- `lower_bounds`: Per-gene inclusive lower bounds (``int64`` array).
- `upper_bounds`: Per-gene inclusive upper bounds.

**Raises**

- `ConfigurationError`: On invalid ``dimensions`` or ``bounds`` (including non-integer bounds).

#### `IntegerRepresentation.describe`

```python
describe() -> Dict[str, Any]
```

Describe type, dimensions and bounds (a pair, or one pair per gene).

#### `IntegerRepresentation.random_individual`

```python
random_individual(rng: np.random.Generator) -> np.ndarray
```

Draw each gene uniformly from its inclusive range.

#### `IntegerRepresentation.repair`

```python
repair(individual: np.ndarray) -> np.ndarray
```

Round to the nearest integer and clip into range; NaN becomes the lower bound.

#### `IntegerRepresentation.validate`

```python
validate(individual: np.ndarray) -> bool
```

Return whether the genome is a 1-D integer array of the right size within bounds.

#### `IntegerRepresentation.crossover`

```python
crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: CrossoverMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]
```

Recombine two parents by exchanging genes.

**Raises**

- `ConfigurationError`: For an unsupported method or invalid parameter.

#### `IntegerRepresentation.mutate`

```python
mutate(
    individual: np.ndarray,
    method: MutationMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> np.ndarray
```

Mutate a copy of ``individual``.

**Keyword Args**

- `mutation_prob`: Per-gene mutation probability (default ``1/dimensions``).
- `max_step`: Largest creep step (default 1, must be >= 1).

**Raises**

- `ConfigurationError`: For an unsupported method or invalid parameter.

### class `BinaryRepresentation`

*Defined in `ga_engine.representations.binary`.*

```python
BinaryRepresentation(dimensions: int)
```

Fixed-length vector of bits stored as ``int8``.

Crossover: single-point, two-point and uniform (default single-point).
Mutation: bit-flip (default).

Operator parameters: ``probability`` for uniform crossover (default 0.5).

**Args**

- `dimensions`: Number of bits.

**Raises**

- `ConfigurationError`: If ``dimensions`` is not a positive integer.

#### `BinaryRepresentation.random_individual`

```python
random_individual(rng: np.random.Generator) -> np.ndarray
```

Draw each bit uniformly at random.

#### `BinaryRepresentation.crossover`

```python
crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: CrossoverMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]
```

Recombine two parents.

**Raises**

- `ConfigurationError`: For an unsupported method or invalid parameter.

#### `BinaryRepresentation.mutate`

```python
mutate(
    individual: np.ndarray,
    method: MutationMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> np.ndarray
```

Flip each bit with probability ``mutation_prob`` (default ``1/dimensions``).

**Raises**

- `ConfigurationError`: For an unsupported method or invalid probability.

#### `BinaryRepresentation.repair`

```python
repair(individual: np.ndarray) -> np.ndarray
```

Round to the nearest bit; NaN becomes 0.

#### `BinaryRepresentation.validate`

```python
validate(individual: np.ndarray) -> bool
```

Return whether the genome is 1-D, right-sized and contains only 0/1.

### class `PermutationRepresentation`

*Defined in `ga_engine.representations.permutation`.*

```python
PermutationRepresentation(dimensions: int)
```

A permutation of ``0 .. dimensions-1`` stored as ``int64``.

Crossover: order crossover (OX, default) and partially mapped crossover
(PMX). Mutation: swap (default), scramble and inversion.

**Args**

- `dimensions`: Permutation length.

**Raises**

- `ConfigurationError`: If ``dimensions`` is not a positive integer.

#### `PermutationRepresentation.random_individual`

```python
random_individual(rng: np.random.Generator) -> np.ndarray
```

Return a uniformly random permutation.

#### `PermutationRepresentation.crossover`

```python
crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: CrossoverMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]
```

Recombine two permutations; children are always valid permutations.

**Raises**

- `ConfigurationError`: For an unsupported method.

#### `PermutationRepresentation.mutate`

```python
mutate(
    individual: np.ndarray,
    method: MutationMethod,
    rng: np.random.Generator,
    **kwargs: Any,
) -> np.ndarray
```

Apply swap, scramble or inversion to a copy of ``individual``.

The permutation operators act on the whole genome, so the per-gene
``mutation_prob`` keyword is accepted and ignored.

**Raises**

- `ConfigurationError`: For an unsupported method.

#### `PermutationRepresentation.repair`

```python
repair(individual: np.ndarray) -> np.ndarray
```

Replace duplicates and out-of-range values with the missing values.

**Raises**

- `ConfigurationError`: If the genome has the wrong length.

#### `PermutationRepresentation.validate`

```python
validate(individual: np.ndarray) -> bool
```

Return whether the genome is an integer permutation of ``0..n-1``.

Integer-valued floats are rejected here; `repair` normalises them.

## Results and data

### class `RunResult`

*Defined in `ga_engine.models`.*

```python
RunResult(
    best_individual: Individual,
    best_fitness: Fitness,
    generation: int,
    elapsed_seconds: float,
    converged: bool,
    stop_reason: str,
    stats: Dict[str, List[Any]],
    pareto_front: Optional[List[Individual]] = None,
    evaluations: int = 0,
    evaluation_failures: int = 0,
    island_stats: Optional[List[Dict[str, List[Any]]]] = None,
) -> None
```

Structured result returned by ``run()``.

**Attributes**

- `best_individual`: Best solution found (a representative of the Pareto front in multi-objective mode).
- `best_fitness`: Fitness of ``best_individual``.
- `generation`: Generation counter when the run stopped.
- `elapsed_seconds`: Wall-clock duration of the ``run()`` call.
- `converged`: ``True`` if a target or stagnation criterion stopped the run.
- `stop_reason`: ``"max_generations"``, ``"target_fitness"`` or ``"stagnation"``.
- `stats`: Per-generation statistics (see `statistics`).
- `pareto_front`: Non-dominated set (multi-objective only, else ``None``).
- `evaluations`: Cumulative number of fitness evaluations performed.
- `evaluation_failures`: How many of those evaluations failed.
- `island_stats`: Per-island statistics (island model only, else ``None``).

### class `Individual`

*Defined in `ga_engine.models`.*

```python
Individual(
    genes: np.ndarray,
    fitness: Fitness,
    age: int = 0,
    crowding_distance: float = 0.0,
    constraint_violation: float = 0.0,
    rank: int = 0,
) -> None
```

A candidate solution together with its evaluated metadata.

The class is deliberately mutable: ``age``, ``rank`` and
``crowding_distance`` are evolutionary bookkeeping rather than value-object
fields. Equality is identity-based (``eq=False``) because comparing NumPy
arrays with ``==`` is ambiguous.

**Attributes**

- `genes`: One-dimensional genome array.
- `fitness`: Scalar, or tuple of objective values (all minimised).
- `age`: Generations the individual has survived (0 for a newborn).
- `crowding_distance`: NSGA-II crowding distance (multi-objective only).
- `constraint_violation`: Aggregate violation; ``0.0`` means feasible.
- `rank`: NSGA-II non-domination rank, 0 being the best front.

**Properties**

- `is_feasible`: Whether the individual satisfies every constraint.

#### `Individual.clone`

```python
clone() -> 'Individual'
```

Return an independent copy (genes are copied, fitness is immutable).

### enum `StopReason`

*Defined in `ga_engine.termination`.*

Why a run ended. Members compare equal to their string values.

| Member | Value |
| --- | --- |
| `TARGET_FITNESS` | `'target_fitness'` |
| `STAGNATION` | `'stagnation'` |
| `MAX_GENERATIONS` | `'max_generations'` |
| `CALLBACK` | `'callback'` |

## Choices

### enum `SelectionMethod`

*Defined in `ga_engine.enums`.*

Parent-selection strategy.

| Member | Value |
| --- | --- |
| `TOURNAMENT` | `'tournament'` |
| `ROULETTE` | `'roulette'` |
| `RANK` | `'rank'` |
| `SUS` | `'sus'` |

### enum `CrossoverMethod`

*Defined in `ga_engine.enums`.*

Recombination operator. Availability depends on the representation.

| Member | Value |
| --- | --- |
| `SINGLE_POINT` | `'single_point'` |
| `TWO_POINT` | `'two_point'` |
| `UNIFORM` | `'uniform'` |
| `SBX` | `'sbx'` |
| `BLX_ALPHA` | `'blx_alpha'` |
| `ORDER` | `'order'` |
| `PMX` | `'pmx'` |

### enum `MutationMethod`

*Defined in `ga_engine.enums`.*

Mutation operator. Availability depends on the representation.

| Member | Value |
| --- | --- |
| `GAUSSIAN` | `'gaussian'` |
| `POLYNOMIAL` | `'polynomial'` |
| `BIT_FLIP` | `'bit_flip'` |
| `SWAP` | `'swap'` |
| `SCRAMBLE` | `'scramble'` |
| `INVERSION` | `'inversion'` |
| `RANDOM_RESET` | `'random_reset'` |
| `CREEP` | `'creep'` |

### enum `ReplacementMethod`

*Defined in `ga_engine.enums`.*

Survivor-selection scheme for single-objective problems.

| Member | Value |
| --- | --- |
| `GENERATIONAL` | `'generational'` |
| `STEADY_STATE` | `'steady_state'` |

### enum `EvaluationErrorPolicy`

*Defined in `ga_engine.enums`.*

What to do when the user's fitness/constraint function raises.

``PENALIZE`` assigns the individual an infinitely bad fitness, logs a
summary warning and counts the failure (visible in the run statistics).
``RAISE`` aborts immediately with `EvaluationError`.

| Member | Value |
| --- | --- |
| `PENALIZE` | `'penalize'` |
| `RAISE` | `'raise'` |

## Errors

### exception `GAEngineError`

*Defined in `ga_engine.errors`.*

```python
GAEngineError()
```

Base class for all errors raised deliberately by ``ga_engine``.

### exception `ConfigurationError`

*Defined in `ga_engine.errors`.*

```python
ConfigurationError()
```

An argument or combination of arguments is invalid.

Raised at construction time wherever possible (fail fast) rather than in
the middle of a run.

### exception `EvaluationError`

*Defined in `ga_engine.errors`.*

```python
EvaluationError()
```

User-supplied fitness or constraint code failed or broke its contract.

Raised when the evaluation error policy is ``RAISE``, when a multi-objective
function returns an inconsistent number of objectives, or when every
evaluation of the first batch fails so the objective count is unknowable.

### exception `CheckpointError`

*Defined in `ga_engine.errors`.*

```python
CheckpointError()
```

A checkpoint file is corrupt, unsupported, or incompatible with the engine.

## Utilities

### `enable_console_logging`

```python
enable_console_logging(
    level: int = 20,
    stream: Optional[TextIO] = None,
) -> logging.Handler
```

Attach (at most) one console handler to the ``ga_engine`` logger.

Calling this repeatedly reuses the existing handler and only updates the
level, so it is safe to call from many engine instances and threads.

**Args**

- `level`: Logging level for the package logger and the handler.
- `stream`: Destination stream; defaults to ``sys.stderr``. Ignored when a handler is already installed.

**Returns**

- The installed console handler.

## Test problems

Importable from `ga_engine.benchmarks`. All are minimisation problems and are picklable.

### `sphere`

```python
sphere(x: np.ndarray) -> float
```

Sphere function; minimum 0 at the origin.

### `rastrigin`

```python
rastrigin(x: np.ndarray) -> float
```

Rastrigin function (highly multimodal); minimum 0 at the origin, usually in [-5.12, 5.12].

### `rosenbrock`

```python
rosenbrock(x: np.ndarray) -> float
```

Rosenbrock valley; minimum 0 at (1, ..., 1). Needs at least two dimensions.

### `ackley`

```python
ackley(x: np.ndarray) -> float
```

Ackley function; minimum 0 at the origin.

### `zdt1`

```python
zdt1(x: np.ndarray) -> Tuple[float, float]
```

ZDT1 two-objective problem on ``[0, 1]^n`` (n >= 2).

The true Pareto front is ``f2 = 1 - sqrt(f1)`` for ``f1`` in ``[0, 1]``.

### `onemax`

```python
onemax(bits: np.ndarray) -> float
```

Number of zero bits; minimum 0 for the all-ones string.

### class `KnapsackProblem`

*Defined in `ga_engine.benchmarks`.*

```python
KnapsackProblem(values: np.ndarray, weights: np.ndarray, capacity: float) -> None
```

0/1 knapsack posed as constrained minimisation.

``objective`` is the negated total value; ``constraint`` is the amount by
which the total weight exceeds the capacity (0 when feasible).

#### `KnapsackProblem.random`

```python
random(n_items: int, seed: int = 0, capacity_ratio: float = 0.4) -> 'KnapsackProblem'
```

Create a random instance whose capacity is a fraction of the total weight.

#### `KnapsackProblem.objective`

```python
objective(bits: np.ndarray) -> float
```

Negative total value of the selected items.

#### `KnapsackProblem.constraint`

```python
constraint(bits: np.ndarray) -> float
```

Overweight of the selection (0 if it fits).

### class `TspProblem`

*Defined in `ga_engine.benchmarks`.*

```python
TspProblem(coordinates: np.ndarray) -> None
```

Euclidean travelling-salesman problem; a genome is a tour (permutation).

#### `TspProblem.random`

```python
random(n_cities: int, seed: int = 0) -> 'TspProblem'
```

Create ``n_cities`` uniformly random points in the unit square.
