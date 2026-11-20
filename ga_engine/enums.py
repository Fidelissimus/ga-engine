"""Enumerations that select the engine's operators and policies."""

from __future__ import annotations

from enum import Enum


class SelectionMethod(Enum):
    """Parent-selection strategy."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    SUS = "sus"


class CrossoverMethod(Enum):
    """Recombination operator. Availability depends on the representation."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    SBX = "sbx"
    BLX_ALPHA = "blx_alpha"
    ORDER = "order"
    PMX = "pmx"


class MutationMethod(Enum):
    """Mutation operator. Availability depends on the representation."""

    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    BIT_FLIP = "bit_flip"
    SWAP = "swap"
    SCRAMBLE = "scramble"
    INVERSION = "inversion"
    RANDOM_RESET = "random_reset"
    CREEP = "creep"


class ReplacementMethod(Enum):
    """Survivor-selection scheme for single-objective problems."""

    GENERATIONAL = "generational"
    STEADY_STATE = "steady_state"


class EvaluationErrorPolicy(Enum):
    """What to do when the user's fitness/constraint function raises.

    ``PENALIZE`` assigns the individual an infinitely bad fitness, logs a
    summary warning and counts the failure (visible in the run statistics).
    ``RAISE`` aborts immediately with :class:`ga_engine.errors.EvaluationError`.
    """

    PENALIZE = "penalize"
    RAISE = "raise"
