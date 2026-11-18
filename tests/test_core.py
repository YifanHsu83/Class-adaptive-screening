import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from class_adaptive_screening.config import ExperimentConfig
from class_adaptive_screening.cutoffs import METHODS, class_cutoffs
from class_adaptive_screening.propensity import (
    assigned_class_scores_cross_fitted,
    assigned_class_scores_in_sample,
)
from class_adaptive_screening.screening import matched_random_mask, retained_mask


def synthetic(seed=11):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), 30)
    X = rng.normal(size=(90, 6))
    X[:, 0] += y * 1.3
    return X, y


def test_propensity_scores_are_probabilities():
    X, y = synthetic()
    for scores in (
        assigned_class_scores_in_sample(X, y),
        assigned_class_scores_cross_fitted(X, y, folds=3),
    ):
        assert scores.shape == (len(y),)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()


def test_crossfit_matches_archived_one_vs_rest_logic():
    X, y = synthetic()
    observed = assigned_class_scores_cross_fitted(X, y, folds=3, seed=2026)
    expected = np.empty(len(y))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    for label in np.unique(y):
        binary = (y == label).astype(int)
        probability = cross_val_predict(
            LogisticRegression(max_iter=1000, random_state=2026),
            X,
            binary,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        expected[binary == 1] = probability[binary == 1]
    np.testing.assert_allclose(observed, expected)


def test_every_cutoff_produces_a_valid_mask():
    X, y = synthetic()
    scores = assigned_class_scores_in_sample(X, y)
    for method in METHODS:
        cutoffs = class_cutoffs(scores, y, method)
        mask = retained_mask(scores, y, cutoffs)
        assert mask.dtype == bool
        assert 0 < mask.sum() <= len(mask)


def test_empirical_and_random_baseline_match_class_counts():
    X, y = synthetic()
    scores = assigned_class_scores_in_sample(X, y)
    empirical = retained_mask(scores, y, class_cutoffs(scores, y, "empirical"))
    random = matched_random_mask(y, empirical, seed=7)
    for label in np.unique(y):
        selected = y == label
        assert empirical[selected].sum() == random[selected].sum()


def test_manuscript_defaults():
    config = ExperimentConfig()
    assert config.gamma == 0.20
    assert config.fixed_cutoff == 0.30
    assert config.propensity_folds == 5
    assert config.crossfit_seed == 2026
    assert config.rf_trees == 100
    assert config.rf_max_depth == 8
    assert config.seeds == 50
    assert config.seed_start == 1
