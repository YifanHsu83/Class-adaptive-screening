from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import ExperimentConfig
from .cutoffs import METHODS, class_cutoffs, global_empirical_cutoff
from .diagnostics import distribution_diagnostics
from .evaluation import evaluate_masks, exploratory_training_f1, summarize_results
from .propensity import assigned_class_scores_cross_fitted, assigned_class_scores_in_sample
from .screening import global_retained_mask, matched_random_mask, retained_mask
from .statistics import paired_tests


def _output(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(path: Path, config: ExperimentConfig, command: str, datasets: list[str]):
    packages = {
        name: version(name)
        for name in ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "matplotlib")
    }
    manifest = {
        "command": command,
        "datasets": datasets,
        "config": config.__dict__,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    (path / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_masks(
    scores,
    labels,
    config: ExperimentConfig,
    gamma: float | None = None,
    include_baselines: bool = False,
):
    gamma = config.gamma if gamma is None else gamma
    masks = {"original": np.ones(len(labels), dtype=bool)}
    for method in METHODS:
        cutoffs = class_cutoffs(scores, labels, method, gamma, config.fixed_cutoff, config.epsilon)
        masks[method] = retained_mask(scores, labels, cutoffs)
    if include_baselines:
        empirical = masks["empirical"]
        masks["global_empirical"] = global_retained_mask(
            scores, global_empirical_cutoff(scores, gamma)
        )
        masks["matched_random"] = lambda seed: matched_random_mask(labels, empirical, seed)
    return masks


def run_evaluation(
    bundle,
    config: ExperimentConfig,
    score_design="cross_fitted",
    seeds=None,
    include_baselines=False,
):
    if score_design == "in_sample":
        scores = assigned_class_scores_in_sample(
            bundle.X_train, bundle.y_train, config.propensity_max_iter, config.crossfit_seed
        )
    elif score_design == "cross_fitted":
        scores = assigned_class_scores_cross_fitted(
            bundle.X_train,
            bundle.y_train,
            config.propensity_folds,
            config.crossfit_seed,
            config.propensity_max_iter,
        )
    else:
        raise ValueError("score_design must be 'in_sample' or 'cross_fitted'")
    masks = build_masks(scores, bundle.y_train, config, include_baselines=include_baselines)
    results = evaluate_masks(
        bundle.X_train,
        bundle.y_train,
        bundle.X_evaluation,
        bundle.y_evaluation,
        masks,
        config.seeds if seeds is None else seeds,
        config.rf_trees,
        config.rf_max_depth,
        config.seed_start,
    )
    return scores, masks, results


def primary(bundle, config: ExperimentConfig, output, seeds=None):
    output = _output(output)
    all_results = []
    cached = {}
    for design in ("in_sample", "cross_fitted"):
        scores, masks, results = run_evaluation(bundle, config, design, seeds)
        results.insert(0, "score_design", design)
        all_results.append(results)
        cached[design] = (scores, masks)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output / "method_results.csv", index=False)
    summarize_results(combined[combined.score_design == "in_sample"]).to_csv(
        output / "method_summary.csv", index=False
    )
    paired_tests(combined[combined.score_design == "in_sample"]).to_csv(
        output / "paired_tests.csv", index=False
    )
    distribution_diagnostics(
        cached["in_sample"][0],
        bundle.y_train,
        config.epsilon,
        config.gof_bootstrap_replicates,
        config.crossfit_seed,
    ).to_csv(output / "distribution_diagnostics.csv", index=False)
    method_cutoffs = {
        method: class_cutoffs(
            cached["in_sample"][0],
            bundle.y_train,
            method,
            config.gamma,
            config.fixed_cutoff,
            config.epsilon,
        )
        for method in METHODS
    }
    exploratory_training_f1(
        bundle.X_train,
        bundle.y_train,
        cached["in_sample"][0],
        method_cutoffs,
    ).to_csv(output / "exploratory_training_f1.csv", index=False)
    sensitivity(bundle, cached["in_sample"][0], config, output, seeds)
    return combined


def sensitivity(bundle, scores, config: ExperimentConfig, output, seeds=None):
    frames = []
    for gamma in config.sensitivity_gammas:
        masks = {"original": np.ones(len(bundle.y_train), dtype=bool)}
        for method in ("beta", "logit_normal", "empirical"):
            cutoffs = class_cutoffs(
                scores, bundle.y_train, method, gamma, config.fixed_cutoff, config.epsilon
            )
            masks[method] = retained_mask(scores, bundle.y_train, cutoffs)
        result = evaluate_masks(
            bundle.X_train,
            bundle.y_train,
            bundle.X_evaluation,
            bundle.y_evaluation,
            masks,
            config.seeds if seeds is None else seeds,
            config.rf_trees,
            config.rf_max_depth,
            config.seed_start,
        )
        result.insert(0, "gamma", gamma)
        frames.append(result)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(Path(output) / "sensitivity_results.csv", index=False)
    return frame


def extended(bundles: dict, config: ExperimentConfig, output, seeds=None):
    output = _output(output)
    frames = []
    for name, bundle in bundles.items():
        _, _, result = run_evaluation(bundle, config, "cross_fitted", seeds, include_baselines=True)
        result.insert(0, "dataset", name)
        frames.append(result)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(output / "extended_seed_results.csv", index=False)
    summary = (
        raw.groupby(["dataset", "method"], sort=False)
        .agg(
            retained_percentage=("retained_percentage", "first"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "extended_summary.csv", index=False)
    return raw


def known_noise(bundles: dict, config: ExperimentConfig, output):
    output = _output(output)
    rows, distributions = [], []
    for name, bundle in bundles.items():
        if bundle.clean_y_train is None:
            raise ValueError(f"{name} has no clean training labels")
        scores = assigned_class_scores_cross_fitted(
            bundle.X_train,
            bundle.y_train,
            config.propensity_folds,
            config.crossfit_seed,
            config.propensity_max_iter,
        )
        masks = build_masks(scores, bundle.y_train, config, include_baselines=True)
        disagreement = bundle.y_train != bundle.clean_y_train
        for method, retained_or_factory in masks.items():
            repetitions = (
                range(config.seed_start, config.seed_start + config.seeds)
                if callable(retained_or_factory)
                else (0,)
            )
            values = []
            for seed in repetitions:
                retained = (
                    retained_or_factory(seed)
                    if callable(retained_or_factory)
                    else retained_or_factory
                )
                flagged = ~retained
                values.append(
                    (
                        100 * flagged.mean(),
                        accuracy_score(disagreement, flagged),
                        f1_score(disagreement, flagged, average="macro"),
                    )
                )
            values = np.asarray(values)
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "flagged_percentage": values[:, 0].mean(),
                    "accuracy": values[:, 1].mean(),
                    "macro_f1": values[:, 2].mean(),
                }
            )
        distributions.append((name, scores, disagreement))
    result = pd.DataFrame(rows)
    result.to_csv(output / "known_noise_results.csv", index=False)
    plot_known_noise(distributions, output / "known_noise_histograms.pdf")
    return result


def plot_known_noise(distributions, path: Path):
    figure, axes = plt.subplots(
        1,
        len(distributions),
        figsize=(4.0 * len(distributions), 3.1),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 31)
    for axis, (name, scores, disagreement) in zip(axes, distributions):
        axis.hist(
            scores[~disagreement],
            bins=bins,
            density=True,
            alpha=0.55,
            color="#4C78A8",
            label="Label agreement",
        )
        axis.hist(
            scores[disagreement],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#F58518",
            label="Label disagreement",
        )
        axis.axvline(np.median(scores[~disagreement]), color="#4C78A8", linestyle="--")
        axis.axvline(np.median(scores[disagreement]), color="#F58518", linestyle="--")
        axis.set(title=name, xlabel="Cross-fitted assigned-class score", xlim=(0, 1))
    axes[0].set_ylabel("Normalized density")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
