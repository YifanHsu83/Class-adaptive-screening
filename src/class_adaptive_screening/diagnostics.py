from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta, norm


def _statistics(values, cdf):
    values = np.sort(np.asarray(values))
    n = len(values)
    probabilities = np.clip(cdf(values), 0, 1)
    ks = max(
        np.max(np.arange(1, n + 1) / n - probabilities),
        np.max(probabilities - np.arange(n) / n),
    )
    expected = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    cvm = 1 / (12 * n) + np.sum((probabilities - expected) ** 2)
    return float(ks), float(cvm)


def distribution_diagnostics(
    scores,
    labels,
    epsilon: float = 1e-6,
    bootstrap_replicates: int = 1000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Parametric-bootstrap diagnostics used in the manuscript."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in np.unique(labels):
        values = np.clip(scores[labels == label], epsilon, 1 - epsilon)

        a, b, _, _ = beta.fit(values, floc=0, fscale=1)
        ll_beta = beta.logpdf(values, a, b).sum()
        ks_beta, cvm_beta = _statistics(values, lambda x, aa=a, bb=b: beta.cdf(x, aa, bb))
        beta_boot = np.empty((bootstrap_replicates, 2))
        for index in range(bootstrap_replicates):
            simulated = np.clip(rng.beta(a, b, len(values)), epsilon, 1 - epsilon)
            sim_a, sim_b, _, _ = beta.fit(simulated, floc=0, fscale=1)
            beta_boot[index] = _statistics(
                simulated, lambda x, aa=sim_a, bb=sim_b: beta.cdf(x, aa, bb)
            )

        transformed = logit(values)
        mu, sigma = transformed.mean(), max(transformed.std(ddof=0), epsilon)
        ll_logit = (norm.logpdf(transformed, mu, sigma) - np.log(values) - np.log1p(-values)).sum()
        logit_cdf = lambda x, mm=mu, ss=sigma: norm.cdf(logit(x), mm, ss)
        ks_logit, cvm_logit = _statistics(values, logit_cdf)
        logit_boot = np.empty((bootstrap_replicates, 2))
        for index in range(bootstrap_replicates):
            simulated = np.clip(expit(rng.normal(mu, sigma, len(values))), epsilon, 1 - epsilon)
            sim_logit = logit(simulated)
            sim_mu = sim_logit.mean()
            sim_sigma = max(sim_logit.std(ddof=0), epsilon)
            logit_boot[index] = _statistics(
                simulated,
                lambda x, mm=sim_mu, ss=sim_sigma: norm.cdf(logit(x), mm, ss),
            )

        rows.extend(
            [
                {
                    "class": label,
                    "distribution": "beta",
                    "ks": ks_beta,
                    "ks_p": (1 + np.sum(beta_boot[:, 0] >= ks_beta)) / (bootstrap_replicates + 1),
                    "cvm": cvm_beta,
                    "cvm_p": (1 + np.sum(beta_boot[:, 1] >= cvm_beta)) / (bootstrap_replicates + 1),
                    "aic": 4 - 2 * ll_beta,
                    "p1": a,
                    "p2": b,
                },
                {
                    "class": label,
                    "distribution": "logit_normal",
                    "ks": ks_logit,
                    "ks_p": (1 + np.sum(logit_boot[:, 0] >= ks_logit)) / (bootstrap_replicates + 1),
                    "cvm": cvm_logit,
                    "cvm_p": (1 + np.sum(logit_boot[:, 1] >= cvm_logit))
                    / (bootstrap_replicates + 1),
                    "aic": 4 - 2 * ll_logit,
                    "p1": mu,
                    "p2": sigma,
                },
            ]
        )
    return pd.DataFrame(rows)
