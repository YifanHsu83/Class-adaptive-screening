from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    gamma: float = 0.20
    fixed_cutoff: float = 0.30
    epsilon: float = 1e-6
    gof_bootstrap_replicates: int = 1000
    propensity_folds: int = 5
    crossfit_seed: int = 2026
    propensity_max_iter: int = 1000
    rf_trees: int = 100
    rf_max_depth: int = 8
    seeds: int = 50
    seed_start: int = 1
    sensitivity_gammas: tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    )

    def __post_init__(self) -> None:
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie strictly between 0 and 1")
        if not 0 <= self.fixed_cutoff <= 1:
            raise ValueError("fixed_cutoff must lie between 0 and 1")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must lie strictly between 0 and 0.5")
        if self.propensity_folds < 2:
            raise ValueError("propensity_folds must be at least 2")
        if self.seeds < 1:
            raise ValueError("seeds must be at least 1")
        if any(not 0 < gamma < 1 for gamma in self.sensitivity_gammas):
            raise ValueError("all sensitivity levels must lie strictly between 0 and 1")

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "sensitivity_gammas" in values:
            values["sensitivity_gammas"] = tuple(values["sensitivity_gammas"])
        return cls(**values)


def load_dataset_configs(path: str | Path) -> tuple[dict[str, dict[str, Any]], Path]:
    path = Path(path).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent
