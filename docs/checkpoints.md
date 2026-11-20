# Checkpoint format

`GeneticAlgorithm.save_checkpoint(path)` / `load_checkpoint(path)` persist and
restore the complete evolutionary state. Implementation:
[`ga_engine/checkpoint.py`](../ga_engine/checkpoint.py) (file format) and
[`ga_engine/snapshots.py`](../ga_engine/snapshots.py) (which parts of an engine form its
resumable state, and how a snapshot is checked against an engine).

## Guarantees

1. **Exact resume.** An engine configured identically and loaded from a
   checkpoint continues exactly as an uninterrupted run would have, because the
   random-number-generator state, the adaptive-mutation state, the stagnation
   and restart counters, the statistics and the best-so-far record are all
   restored. The test-suite verifies this bit-for-bit for single- and
   multi-objective runs, adaptive mutation and restarts, in both formats.
2. **Atomic writes.** Data is written to a temporary file in the target
   directory, flushed and `fsync`ed, then moved over the destination with
   `os.replace`. A crash or error while saving never replaces a good checkpoint
   with a truncated one, and the temporary file is removed on failure.
3. **All-or-nothing loads.** The file is parsed and validated completely
   (see [Compatibility checks](#compatibility-checks)) *before* any engine state
   is touched. A failed load leaves the engine exactly as it was.
4. **No code execution.** JSON is parsed with the standard library; NPZ is read
   with `allow_pickle=False`. Loading a checkpoint cannot run code. (It can
   still be *wrong*: only load files you trust to be yours.)

## What is **not** stored

The fitness and constraint functions, the representation object and every
engine setting (rates, operators, ...). You rebuild the engine in code and load
the state into it. Resuming with *different* operator settings is possible
but is then no longer a reproduction of the uninterrupted run.

## Choosing a format

The suffix decides: `.json` or `.npz` (case-insensitive); anything else raises
`CheckpointError`.

| | `.json` | `.npz` |
| --- | --- | --- |
| Readable / diffable | yes | no |
| Size | larger | compact (compressed arrays) |
| Portable to other languages | yes (strict JSON, see below) | needs a NumPy reader |
| Contents | identical schema | identical schema |

## Schema (version 3)

Top-level fields (JSON: keys of the object; NPZ: the `metadata` entry holds
everything except the population arrays):

| Field | Meaning |
| --- | --- |
| `version` | Format version (`3`). Readers accept `2` and `3`. |
| `population_size`, `representation`, `dimensions`, `bounds`, `multi_objective` | Description of the engine the state belongs to; checked on load. `bounds` is one `[low, high]` pair, or a list of pairs for per-gene bounds (`null` in version 2 files). |
| `generation`, `stagnation`, `generations_since_restart` | Run counters. |
| `mutation_prob`, `gene_mutation_prob`, `sigma`, `success_count`, `trial_count`, `last_success_ratio` | Adaptive-mutation state. |
| `objective_count` | Inferred objective count (multi-objective), else `null`. |
| `evaluations`, `evaluation_failures` | Cumulative counters. |
| `rng_state` | `numpy` bit-generator state (`bit_generator`, `state.state`, `state.inc`, `has_uint32`, `uinteger`). The two state integers exceed 64 bits; Python's `json` handles them, other parsers may need big-integer support. |
| `stats` | The per-generation statistics (see [configuration](configuration.md#statistics-keys)). |
| `best_fitness_history` | Best-so-far fitness per generation. |
| `pareto_signature` | Fingerprint of the Pareto front used for multi-objective stagnation detection, or `null`. |
| `population` | JSON: list of `{genes, fitness, age, crowding_distance, constraint_violation}`. NPZ: arrays `genes`, `fitnesses`, `ages`, `crowding`, `violations`. |
| `best_individual` | JSON: same shape as a population entry, or `null`. NPZ: arrays `best_genes`, `best_fitness`, `best_age`, `best_crowd`, `best_violation`. |

`fitness` is a number (single-objective) or a list of numbers (multi-objective).

### Strict JSON and non-finite numbers

Python's `json` module would write `Infinity` and `NaN`, which are not valid
JSON. Crowding distances are infinite for boundary solutions and failed
evaluations carry infinite fitness, so such values do occur. They are
encoded as tagged objects and decoded back on load:

```json
{ "crowding_distance": { "$float": "inf" } }
```

Accepted tags: `"inf"`, `"-inf"`, `"nan"`. Files written with `allow_nan=False`
parse in any JSON implementation. Version 2 files (see below) may contain bare
`Infinity` tokens; Python's parser still reads them.

## Compatibility checks

`load_checkpoint` raises `CheckpointError` (and changes nothing) if:

- the file is missing required fields, is not valid JSON / NPZ, or has an
  unsupported `version` (a missing file raises `FileNotFoundError`);
- `population_size`, `representation` type name, `dimensions`, `bounds` (when
  present) or `multi_objective` differ from the engine;
- the population does not contain exactly `population_size` individuals;
- any genome (including the best individual's) is invalid for the engine's
  representation, e.g. outside the bounds;
- a fitness has the wrong shape for the objective mode;
- the stored random state is incompatible with the engine's bit generator.

Genomes loaded from JSON are converted to the representation's canonical dtype
(`float64`, `int8`, `int64`; integer genomes are `int64`), so binary and permutation genomes keep their
types across a round trip.

## Version history

| Version | Notes |
| --- | --- |
| 3 | Current format: per-gene `bounds` (a pair, or a list of pairs), `generations_since_restart`, `evaluations`, `evaluation_failures`; strict-JSON float encoding; atomic writes. |
| 2 | Earlier layout, read-only support: missing fields default to `0` / `null`, and statistics lacking the two newer columns are padded with `null` so every list keeps the same length. |
