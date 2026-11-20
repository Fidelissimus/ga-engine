"""Segment-exchange crossovers shared by the real and binary representations."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def single_point_exchange(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap the tails of two genomes after one random cut point.

    Genomes shorter than two genes cannot be cut and are returned unchanged.
    """
    size = len(parent1)
    if size < 2:
        return parent1.copy(), parent2.copy()
    point = int(rng.integers(1, size))
    return (
        np.concatenate([parent1[:point], parent2[point:]]),
        np.concatenate([parent2[:point], parent1[point:]]),
    )


def two_point_exchange(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap the middle segment of two genomes between two random cut points.

    Two distinct interior cut points need at least three genes; shorter genomes
    are returned unchanged.
    """
    size = len(parent1)
    if size < 3:
        return parent1.copy(), parent2.copy()
    first, second = np.sort(rng.choice(np.arange(1, size), 2, replace=False))
    first, second = int(first), int(second)
    return (
        np.concatenate([parent1[:first], parent2[first:second], parent1[second:]]),
        np.concatenate([parent2[:first], parent1[first:second], parent2[second:]]),
    )


def uniform_exchange(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    probability: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take each gene from ``parent1`` with ``probability``, else from ``parent2``.

    The second child receives the complementary choice.
    """
    mask = rng.random(len(parent1)) < probability
    return np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)
