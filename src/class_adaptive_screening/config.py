from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "sensitivity_gammas" in values:
            values["sensitivity_gammas"] = tuple(values["sensitivity_gammas"])
        return cls(**values)


def load_dataset_configs(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent
