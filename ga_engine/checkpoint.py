"""Checkpoint (de)serialisation.

A checkpoint captures the complete evolutionary state, including the
random-number-generator state, so a resumed run continues exactly as an
uninterrupted run would have. Two on-disk formats share one schema:

* ``.json`` - human readable, **strict** JSON. Non-finite floats (``inf``,
  ``nan``) are written as ``{"$float": "inf"}`` objects instead of the
  non-standard ``Infinity`` tokens Python would emit by default, so the files
  can be parsed by any JSON implementation.
* ``.npz`` - compressed NumPy arrays for the population plus the same JSON
  metadata; compact for large populations. Loaded with ``allow_pickle=False``.

Files are written atomically (temporary file, ``fsync``, ``os.replace``), so a
crash while saving can never leave a truncated checkpoint in place of a good one.

This module only converts between files and :class:`EngineSnapshot`; deciding
whether a snapshot fits a particular engine is the engine's job.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .aliases import Fitness
from .errors import CheckpointError
from .models import Individual

CHECKPOINT_VERSION = 3
READABLE_VERSIONS = (2, 3)

PathLike = Union[str, "os.PathLike[str]"]

_FLOAT_TAG = "$float"
_POPULATION_FIELDS = ("population", "best_individual")


@dataclass
class EngineSnapshot:
    """Everything needed to restore an engine to an earlier point in a run."""

    version: int
    population_size: int
    representation: str
    dimensions: int
    bounds: Optional[List[Any]]
    multi_objective: bool
    generation: int
    stagnation: int
    generations_since_restart: int
    mutation_prob: float
    gene_mutation_prob: float
    sigma: float
    success_count: int
    trial_count: int
    last_success_ratio: Optional[float]
    objective_count: Optional[int]
    evaluations: int
    evaluation_failures: int
    rng_state: Dict[str, Any]
    stats: Dict[str, List[Any]]
    best_fitness_history: List[Fitness]
    pareto_signature: Optional[Tuple[Any, ...]]
    population: List[Individual]
    best_individual: Optional[Individual]


# ---------------------------------------------------------------------------
# Strict-JSON value encoding
# ---------------------------------------------------------------------------


def encode_value(value: Any) -> Any:
    """Convert NumPy/Python values into strictly JSON-serialisable ones."""
    if isinstance(value, np.ndarray):
        return [encode_value(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_FLOAT_TAG: "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    return value


def decode_value(value: Any) -> Any:
    """Invert :func:`encode_value` (tagged floats become ``float`` again)."""
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            return float(value[_FLOAT_TAG])
        return {k: decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    return value


def _dumps(payload: Any) -> str:
    """Serialise to strict JSON; raises ``ValueError`` if anything non-finite slipped through."""
    return json.dumps(encode_value(payload), allow_nan=False)


# ---------------------------------------------------------------------------
# Snapshot <-> plain dictionaries
# ---------------------------------------------------------------------------


def _metadata(snapshot: EngineSnapshot) -> Dict[str, Any]:
    """Return every snapshot field except the population arrays."""
    return {
        f.name: getattr(snapshot, f.name)
        for f in fields(snapshot)
        if f.name not in _POPULATION_FIELDS
    }


def _fitness_from_raw(raw: Any) -> Fitness:
    """Rebuild a fitness value from its JSON/NumPy representation."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        return tuple(float(v) for v in raw)
    return float(raw)


def _individual_from_entry(entry: Dict[str, Any]) -> Individual:
    """Rebuild an :class:`Individual` from a JSON population entry."""
    return Individual(
        genes=np.asarray(entry["genes"]),
        fitness=_fitness_from_raw(entry["fitness"]),
        age=int(entry.get("age", 0)),
        crowding_distance=float(entry.get("crowding_distance", 0.0)),
        constraint_violation=float(entry.get("constraint_violation", 0.0)),
    )


def _entry_from_individual(individual: Individual) -> Dict[str, Any]:
    """Describe an :class:`Individual` as a JSON-friendly dictionary."""
    return {
        "genes": individual.genes,
        "fitness": individual.fitness,
        "age": individual.age,
        "crowding_distance": individual.crowding_distance,
        "constraint_violation": individual.constraint_violation,
    }


def _build_snapshot(
    meta: Dict[str, Any], population: List[Individual], best: Optional[Individual]
) -> EngineSnapshot:
    """Create a snapshot from decoded metadata, applying defaults for v2 files."""
    version = int(meta["version"])
    if version not in READABLE_VERSIONS:
        raise CheckpointError(
            f"Unsupported checkpoint version {version}; readable versions: {READABLE_VERSIONS}"
        )
    signature = meta.get("pareto_signature")
    return EngineSnapshot(
        version=version,
        population_size=int(meta["population_size"]),
        representation=str(meta["representation"]),
        dimensions=int(meta["dimensions"]),
        bounds=meta.get("bounds"),
        multi_objective=bool(meta["multi_objective"]),
        generation=int(meta["generation"]),
        stagnation=int(meta["stagnation"]),
        generations_since_restart=int(meta.get("generations_since_restart", 0)),
        mutation_prob=float(meta["mutation_prob"]),
        gene_mutation_prob=float(meta["gene_mutation_prob"]),
        sigma=float(meta["sigma"]),
        success_count=int(meta.get("success_count", 0)),
        trial_count=int(meta.get("trial_count", 0)),
        last_success_ratio=meta.get("last_success_ratio"),
        objective_count=meta.get("objective_count"),
        evaluations=int(meta.get("evaluations", 0)),
        evaluation_failures=int(meta.get("evaluation_failures", 0)),
        rng_state=dict(meta["rng_state"]),
        stats={k: list(v) for k, v in dict(meta.get("stats", {})).items()},
        best_fitness_history=[
            tuple(v) if isinstance(v, list) else v for v in meta.get("best_fitness_history", [])
        ],
        pareto_signature=(
            tuple((tuple(f), float(v)) for f, v in signature) if signature is not None else None
        ),
        population=population,
        best_individual=best,
    )


# ---------------------------------------------------------------------------
# Atomic file output
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Write a file so readers see either the old or the complete new content."""
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_snapshot(path: PathLike, snapshot: EngineSnapshot) -> None:
    """Write ``snapshot`` to a ``.json`` or ``.npz`` file (chosen by suffix).

    Args:
        path: Destination; the suffix selects the format.
        snapshot: State to persist.

    Raises:
        CheckpointError: If the suffix is neither ``.json`` nor ``.npz``, or the
            population is empty.
        OSError: If the file cannot be written.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".json", ".npz"):
        raise CheckpointError("Checkpoint path must end with .npz or .json")
    if not snapshot.population:
        raise CheckpointError("Cannot checkpoint before population initialization")
    if suffix == ".json":
        _write_json(target, snapshot)
    else:
        _write_npz(target, snapshot)


def read_snapshot(path: PathLike) -> EngineSnapshot:
    """Read a ``.json`` or ``.npz`` checkpoint (format versions 2 and 3).

    Args:
        path: Checkpoint file.

    Returns:
        The parsed snapshot. It has been parsed but not yet matched against any
        engine configuration.

    Raises:
        CheckpointError: If the suffix is unsupported or the file is corrupt,
            incomplete or of an unsupported version.
        FileNotFoundError: If the file does not exist.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in (".json", ".npz"):
        raise CheckpointError("Checkpoint path must end with .npz or .json")
    try:
        return _read_json(source) if suffix == ".json" else _read_npz(source)
    except (ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"Malformed checkpoint {source}: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON format
# ---------------------------------------------------------------------------


def _write_json(path: Path, snapshot: EngineSnapshot) -> None:
    payload = _metadata(snapshot)
    payload["population"] = [_entry_from_individual(i) for i in snapshot.population]
    payload["best_individual"] = (
        _entry_from_individual(snapshot.best_individual) if snapshot.best_individual else None
    )
    data = json.dumps(encode_value(payload), indent=2, allow_nan=False).encode("utf-8")

    def write(handle: BinaryIO) -> None:
        handle.write(data)

    _atomic_write(path, write)


def _read_json(path: Path) -> EngineSnapshot:
    with path.open("r", encoding="utf-8") as handle:
        payload = decode_value(json.load(handle))
    population = [_individual_from_entry(e) for e in payload.get("population", [])]
    best_entry = payload.get("best_individual")
    best = _individual_from_entry(best_entry) if best_entry is not None else None
    return _build_snapshot(payload, population, best)


# ---------------------------------------------------------------------------
# NPZ format
# ---------------------------------------------------------------------------


def _write_npz(path: Path, snapshot: EngineSnapshot) -> None:
    population = snapshot.population
    width = len(population[0].fitness) if isinstance(population[0].fitness, tuple) else 1
    fitnesses = np.empty((len(population), width), dtype=float)
    for row, individual in enumerate(population):
        values = (
            list(individual.fitness)
            if isinstance(individual.fitness, tuple)
            else [individual.fitness]
        )
        if len(values) != width:
            raise CheckpointError("Inconsistent fitness dimensionality in population")
        fitnesses[row] = values

    genes = np.stack([i.genes for i in population])
    best = snapshot.best_individual
    if best is None:
        best_genes = np.empty((0,), dtype=genes.dtype)
        best_fitness = np.empty((0,), dtype=float)
    else:
        best_genes = best.genes
        best_fitness = np.asarray(
            list(best.fitness) if isinstance(best.fitness, tuple) else [best.fitness], dtype=float
        )
    arrays: Dict[str, Any] = {
        "genes": genes,
        "fitnesses": fitnesses,
        "ages": np.asarray([i.age for i in population], dtype=np.int64),
        "crowding": np.asarray([i.crowding_distance for i in population], dtype=float),
        "violations": np.asarray([i.constraint_violation for i in population], dtype=float),
        "best_genes": best_genes,
        "best_fitness": best_fitness,
        "best_age": np.asarray([best.age if best else 0], dtype=np.int64),
        "best_crowd": np.asarray([best.crowding_distance if best else 0.0], dtype=float),
        "best_violation": np.asarray([best.constraint_violation if best else 0.0], dtype=float),
        "metadata": np.asarray(_dumps(_metadata(snapshot))),
    }
    def write(handle: BinaryIO) -> None:
        np.savez_compressed(handle, **arrays)

    _atomic_write(path, write)


def _read_npz(path: Path) -> EngineSnapshot:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise CheckpointError("Checkpoint is missing metadata")
        meta = decode_value(json.loads(str(data["metadata"].item())))
        multi_objective = bool(meta["multi_objective"])
        genes, fits = data["genes"], data["fitnesses"]
        ages, crowding, violations = data["ages"], data["crowding"], data["violations"]
        population = [
            Individual(
                genes=genes[i].copy(),
                fitness=tuple(float(v) for v in fits[i]) if multi_objective else float(fits[i, 0]),
                age=int(ages[i]),
                crowding_distance=float(crowding[i]),
                constraint_violation=float(violations[i]),
            )
            for i in range(len(genes))
        ]
        best: Optional[Individual] = None
        if data["best_genes"].size:
            best_fit = data["best_fitness"]
            best = Individual(
                genes=data["best_genes"].copy(),
                fitness=(
                    tuple(float(v) for v in best_fit) if multi_objective else float(best_fit[0])
                ),
                age=int(data["best_age"][0]) if "best_age" in data.files else 0,
                crowding_distance=(
                    float(data["best_crowd"][0]) if "best_crowd" in data.files else 0.0
                ),
                constraint_violation=(
                    float(data["best_violation"][0]) if "best_violation" in data.files else 0.0
                ),
            )
    return _build_snapshot(meta, population, best)
