from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _logistic_model(max_iter: int, random_state: int | None = None) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=random_state)


def assigned_class_scores_in_sample(
    X: np.ndarray, y: np.ndarray, max_iter: int = 1000, seed: int = 2026
) -> np.ndarray:
    scores = np.empty(len(y), dtype=float)
    for label in np.unique(y):
        target = y == label
        model = _logistic_model(max_iter, seed).fit(X, target)
        scores[target] = model.predict_proba(X[target])[:, 1]
    return scores


def assigned_class_scores_cross_fitted(
    X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 2026, max_iter: int = 1000
) -> np.ndarray:
    """One-vs-rest out-of-fold probability corresponding to each recorded class."""
    scores = np.empty(len(y), dtype=float)
    # Match the archived analysis: each one-vs-rest problem is stratified
    # independently using its binary target and the same random seed.
    for label in np.unique(y):
        target = y == label
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        class_probabilities = np.empty(len(y), dtype=float)
        for train_index, holdout_index in splitter.split(X, target):
            model = _logistic_model(max_iter, seed).fit(X[train_index], target[train_index])
            class_probabilities[holdout_index] = model.predict_proba(X[holdout_index])[:, 1]
        scores[target] = class_probabilities[target]
    return scores
