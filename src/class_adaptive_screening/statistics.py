from __future__ import annotations

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


def paired_tests(results: pd.DataFrame, baseline: str = "original") -> pd.DataFrame:
    rows = []
    for metric in ("accuracy", "macro_f1"):
        pivot = results.pivot(index="seed", columns="method", values=metric)
        for method in pivot.columns:
            if method == baseline:
                continue
            difference = pivot[method] - pivot[baseline]
            t_result = ttest_rel(pivot[method], pivot[baseline])
            try:
                w_result = wilcoxon(difference, zero_method="wilcox")
                w_p = float(w_result.pvalue)
            except ValueError:
                w_p = 1.0
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "original_mean": pivot[baseline].mean(),
                    "filtered_mean": pivot[method].mean(),
                    "mean_difference": difference.mean(),
                    "improved": int((difference > 0).sum()),
                    "unchanged": int((difference == 0).sum()),
                    "worse": int((difference < 0).sum()),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": w_p,
                    "cohens_dz": difference.mean() / difference.std(ddof=1),
                }
            )
    frame = pd.DataFrame(rows)
    for metric in frame["metric"].unique():
        selected = frame["metric"] == metric
        frame.loc[selected, "paired_t_p_holm"] = multipletests(
            frame.loc[selected, "paired_t_p"], method="holm"
        )[1]
        frame.loc[selected, "wilcoxon_p_holm"] = multipletests(
            frame.loc[selected, "wilcoxon_p"], method="holm"
        )[1]
    return frame
