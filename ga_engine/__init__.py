"""ga_engine - a modular genetic algorithm toolkit.

Quick start::

    from ga_engine import GeneticAlgorithm, RealRepresentation
    from ga_engine.benchmarks import sphere

    ga = GeneticAlgorithm(60, RealRepresentation(10, (-5.0, 5.0)), sphere, rng_seed=1)
    result = ga.run(max_generations=200)
    print(result.best_fitness)

Public API (everything below is importable from the package root):

* engines: :class:`GeneticAlgorithm`, :class:`IslandModel`
* genomes: :class:`RealRepresentation`, :class:`IntegerRepresentation`,
  :class:`BinaryRepresentation`, :class:`PermutationRepresentation`, base class :class:`Representation`
* data: :class:`Individual`, :class:`RunResult`, :class:`StopReason`, :class:`GAConfig`
* choices: :class:`SelectionMethod`, :class:`CrossoverMethod`,
  :class:`MutationMethod`, :class:`ReplacementMethod`,
  :class:`EvaluationErrorPolicy`
* errors: :class:`GAEngineError` and subclasses
* utilities: :func:`enable_console_logging`
"""

from ._version import __version__
from .config import GAConfig
from .engine import GeneticAlgorithm
from .enums import (
    CrossoverMethod,
    EvaluationErrorPolicy,
    MutationMethod,
    ReplacementMethod,
    SelectionMethod,
)
from .errors import (
    CheckpointError,
    ConfigurationError,
    EvaluationError,
    GAEngineError,
)
from .islands import IslandModel
from .logging_utils import enable_console_logging
from .models import Individual, RunResult
from .termination import StopReason
from .representations import (
    BinaryRepresentation,
    IntegerRepresentation,
    PermutationRepresentation,
    RealRepresentation,
    Representation,
)

__all__ = [
    "__version__",
    "GeneticAlgorithm",
    "IslandModel",
    "GAConfig",
    "Representation",
    "RealRepresentation",
    "BinaryRepresentation",
    "IntegerRepresentation",
    "PermutationRepresentation",
    "Individual",
    "RunResult",
    "StopReason",
    "SelectionMethod",
    "CrossoverMethod",
    "MutationMethod",
    "ReplacementMethod",
    "EvaluationErrorPolicy",
    "GAEngineError",
    "ConfigurationError",
    "EvaluationError",
    "CheckpointError",
    "enable_console_logging",
]
