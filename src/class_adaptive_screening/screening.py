from __future__ import annotations

import numpy as np


def retained_mask(scores: np.ndarray, labels: np.ndarray, cutoffs: dict) -> np.ndarray:
    """Retain scores strictly above the cutoff; ties are removed."""
    thresholds = np.fromiter((cutoffs[label] for label in labels), dtype=float, count=len(labels))
    return scores > thresholds


def global_retained_mask(scores: np.ndarray, cutoff: float) -> np.ndarray:
    return scores > cutoff


def matched_random_mask(
    labels: np.ndarray, reference_retained: np.ndarray, seed: int
) -> np.ndarray:
    """Randomly retain the same number within every class as a reference rule."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        number = int(reference_retained[indices].sum())
        mask[rng.choice(indices, size=number, replace=False)] = True
    return mask
