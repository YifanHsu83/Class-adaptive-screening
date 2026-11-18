from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def evaluate_masks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    masks: dict,
    seeds: int = 50,
    trees: int = 100,
    max_depth: int = 8,
    seed_start: int = 1,
) -> pd.DataFrame:
    rows = []
    for seed in range(seed_start, seed_start + seeds):
        for method, mask_or_factory in masks.items():
            mask = mask_or_factory(seed) if callable(mask_or_factory) else mask_or_factory
            classifier = RandomForestClassifier(
                n_estimators=trees,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=-1,
            )
            classifier.fit(X_train[mask], y_train[mask])
            prediction = classifier.predict(X_evaluation)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "retained_n": int(mask.sum()),
                    "removed_n": int((~mask).sum()),
                    "retained_percentage": 100 * mask.mean(),
                    "accuracy": accuracy_score(y_evaluation, prediction),
                    "macro_f1": f1_score(y_evaluation, prediction, average="macro"),
                }
            )
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("method", sort=False)
        .agg(
            retained_n=("retained_n", "first"),
            removed_n=("removed_n", "first"),
            retained_percentage=("retained_percentage", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
        )
        .reset_index()
    )


def exploratory_training_f1(X, y, scores, method_cutoffs) -> pd.DataFrame:
    """Exact exploratory analysis used for the manuscript's class-level table."""
    rows = []
    for label in np.unique(y):
        binary_target = y == label
        target_indices = np.flatnonzero(binary_target)
        for method in ("original", "fixed", "cantelli", "beta", "logit_normal", "empirical"):
            retained = np.ones(len(y), dtype=bool)
            if method != "original":
                cutoff = method_cutoffs[method][label]
                retained[target_indices[scores[target_indices] <= cutoff]] = False
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=2026,
                n_jobs=-1,
            ).fit(X[retained], binary_target[retained])
            prediction = model.predict(X[retained])
            rows.append(
                {
                    "method": method,
                    "class": label,
                    "training_f1": f1_score(binary_target[retained], prediction, average="macro"),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result[result.method == "original"].set_index("class")["training_f1"]
    result["difference_from_original"] = result.apply(
        lambda row: row.training_f1 - baseline.loc[row["class"]], axis=1
    )
    return result
