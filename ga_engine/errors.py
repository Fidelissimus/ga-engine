"""Exception hierarchy for ``ga_engine``.

Every error raised deliberately by the package derives from
:class:`GAEngineError`, so callers can catch the whole family with one
``except`` clause. The concrete classes also inherit from the closest built-in
exception (``ValueError`` / ``RuntimeError``), so callers that only know the
built-ins can still catch them.
"""

from __future__ import annotations


class GAEngineError(Exception):
    """Base class for all errors raised deliberately by ``ga_engine``."""


class ConfigurationError(GAEngineError, ValueError):
    """An argument or combination of arguments is invalid.

    Raised at construction time wherever possible (fail fast) rather than in
    the middle of a run.
    """


class EvaluationError(GAEngineError, RuntimeError):
    """User-supplied fitness or constraint code failed or broke its contract.

    Raised when the evaluation error policy is ``RAISE``, when a multi-objective
    function returns an inconsistent number of objectives, or when every
    evaluation of the first batch fails so the objective count is unknowable.
    """


class CheckpointError(GAEngineError, ValueError):
    """A checkpoint file is corrupt, unsupported, or incompatible with the engine."""
