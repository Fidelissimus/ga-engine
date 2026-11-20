# Operators

Which operators exist for which genome, their parameters, and their exact
semantics. Operators are chosen with `crossover_method=` / `mutation_method=`
(enum members from `ga_engine`) and tuned with `crossover_params=` /
`mutation_params=`.

## Compatibility

| | Real | Integer | Binary | Permutation |
| --- | :---: | :---: | :---: | :---: |
| `SINGLE_POINT` crossover | ✓ | ✓ | ✓ (default) | |
| `TWO_POINT` crossover | ✓ | ✓ | ✓ | |
| `UNIFORM` crossover | ✓ | ✓ (default) | ✓ | |
| `SBX` crossover | ✓ (default) | | | |
| `BLX_ALPHA` crossover | ✓ | | | |
| `ORDER` crossover | | | | ✓ (default) |
| `PMX` crossover | | | | ✓ |
| `GAUSSIAN` mutation | ✓ (default) | | | |
| `POLYNOMIAL` mutation | ✓ | | | |
| `BIT_FLIP` mutation | | | ✓ (default) | |
| `SWAP` mutation | | | | ✓ (default) |
| `SCRAMBLE` mutation | | | | ✓ |
| `INVERSION` mutation | | | | ✓ |
| `RANDOM_RESET` mutation | | ✓ (default) | | |
| `CREEP` mutation | | ✓ | | |

An unsupported combination raises `ConfigurationError` when the engine is built.

## Crossover

| Method | Parameter (default) | Behaviour |
| --- | --- | --- |
| `SINGLE_POINT` | none | One cut point; the children swap tails. Genomes shorter than 2 genes are copied. |
| `TWO_POINT` | none | Two distinct interior cut points; the children swap the middle segment. Genomes shorter than 3 genes are copied. |
| `UNIFORM` | `probability` (0.5), in [0, 1] | Each gene comes from parent 1 with this probability, else from parent 2; the second child gets the complement. |
| `SBX` | `eta` (20), > 0 | Simulated binary crossover. A larger `eta` keeps children closer to their parents. Children are clipped to the bounds. |
| `BLX_ALPHA` | `alpha` (0.5), ≥ 0 | Each child gene is drawn uniformly from the parents' interval widened by `alpha` times its length. Clipped to the bounds. |
| `ORDER` | none | Textbook OX1: the child keeps a segment of one parent; the other positions are filled cyclically from just after the segment with the remaining genes in the order of the other parent. |
| `PMX` | none | Partially mapped crossover: keeps a segment of one parent and resolves conflicts through the segment's mapping. |

Single-point and two-point crossover on **real** genomes *exchange* gene
segments. They do not blend values; use `SBX` or `BLX_ALPHA` for
recombination that creates new values.

## Mutation

Mutation is applied to an offspring with probability `mutation_prob`; the
operator below then decides which genes change.

| Method | Parameters | Behaviour |
| --- | --- | --- |
| `GAUSSIAN` | step: `mutation_sigma` (0.1 of the range); per gene: `gene_mutation_prob` | Adds N(0, σ) to each selected gene; clipped to the bounds. |
| `POLYNOMIAL` | `eta_m` (20), > 0; per gene: `gene_mutation_prob` | Deb's bounded polynomial mutation. Larger `eta_m` means smaller perturbations. |
| `BIT_FLIP` | per gene: `gene_mutation_prob` | Flips each selected bit. |
| `SWAP` | none | Exchanges two random positions. |
| `SCRAMBLE` | none | Shuffles a random segment. |
| `INVERSION` | none | Reverses a random segment. |
| `RANDOM_RESET` | per gene: `gene_mutation_prob` | Replaces each selected integer gene by a uniform draw from its range. Never gets stuck; best for categorical-like integers. |
| `CREEP` | `max_step` (1), ≥ 1; per gene: `gene_mutation_prob` | Moves each selected integer gene by a random step of 1 … `max_step` in a random direction, clipped to the range. Best for ordinal settings. |

The permutation mutations always act on the whole genome, so
`gene_mutation_prob` has no effect on them.

## Bounds

`RealRepresentation` and `IntegerRepresentation` accept either one `(low, high)`
pair shared by every gene, or one pair per gene:

```text
RealRepresentation(3, bounds=(-5.0, 5.0))                       # shared
RealRepresentation(3, bounds=[(0, 1), (-10, 10), (100, 500)])   # per gene
IntegerRepresentation(2, bounds=[(1, 20), (8, 128)])            # inclusive integers
```

Each pair needs `low < high` and finite values. Integer bounds must be whole
numbers and are inclusive on both ends. Genomes are always repaired into their
bounds (real: clip, NaN becomes the midpoint; integer: round, then clip).

## Selection

| Method | Behaviour |
| --- | --- |
| `TOURNAMENT` | Draws `tournament_size` distinct contestants (with replacement *across* tournaments) and picks the best. |
| `ROULETTE` | Fitness-proportionate for minimisation: weight = (worst finite fitness − fitness) + 1e-12; non-finite fitness gets the lowest weight; equal fitness gives uniform weights. |
| `RANK` | Linear ranking: the best individual gets weight *n*, the worst weight 1. |
| `SUS` | Stochastic universal sampling over the roulette weights: one random offset, equally spaced pointers; the result is shuffled. |

All weighted methods sample **with replacement**, which is what creates
selection pressure. In multi-objective mode, tournament compares by NSGA-II rank then crowding distance;
roulette weights combine rank (`1 / (1 + rank)`) with a mild crowding bonus, and
rank selection uses `n − rank`.

## Writing your own representation

Subclass `ga_engine.Representation` and implement `random_individual`,
`crossover`, `mutate` and `repair`; add `validate` so invalid genomes can be
detected, and set `GENOME_DTYPE`. The optional class variables
`SUPPORTED_CROSSOVERS`, `SUPPORTED_MUTATIONS`, `DEFAULT_CROSSOVER`,
`DEFAULT_MUTATION`, `CROSSOVER_PARAMS` and `MUTATION_PARAMS` opt in to
construction-time validation and per-genome defaults. If you leave them unset,
the engine passes its fallback method names (`SBX` / `GAUSSIAN`) to your
`crossover` / `mutate` without validating them; your methods may simply ignore
them. A complete, tested example is in the
[user guide](user-guide.md#15-writing-your-own-representation).
