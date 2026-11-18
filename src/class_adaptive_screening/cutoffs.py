from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta, norm

METHODS = ("fixed", "cantelli", "beta", "logit_normal", "empirical")


def class_cutoffs(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str,
    gamma: float = 0.20,
    fixed_cutoff: float = 0.30,
    epsilon: float = 1e-6,
) -> dict:
    """Estimate a lower-tail threshold separately for each recorded class.

    Beta and logit-normal fits clip probabilities away from zero and one.
    Empirical and Cantelli thresholds use the original scores.
    """
    if not 0 < gamma < 1:
        raise ValueError("gamma must lie strictly between zero and one")
    result = {}
    for label in np.unique(labels):
        values = np.asarray(scores[labels == label], dtype=float)
        clipped = np.clip(values, epsilon, 1 - epsilon)
        if method == "fixed":
            cutoff = fixed_cutoff
        elif method == "cantelli":
            cutoff = values.mean() - values.std(ddof=1) * np.sqrt((1 - gamma) / gamma)
            cutoff = np.clip(cutoff, 0.0, 1.0)
        elif method == "beta":
            a, b, _, _ = beta.fit(clipped, floc=0, fscale=1)
            cutoff = beta.ppf(gamma, a, b)
        elif method == "logit_normal":
            transformed = logit(clipped)
            cutoff = expit(transformed.mean() + norm.ppf(gamma) * transformed.std(ddof=1))
        elif method == "empirical":
            cutoff = np.quantile(values, gamma)
        else:
            raise ValueError(f"Unknown cutoff method: {method}")
        result[label] = float(cutoff)
    return result


def global_empirical_cutoff(scores: np.ndarray, gamma: float = 0.20) -> float:
    return float(np.quantile(scores, gamma))
