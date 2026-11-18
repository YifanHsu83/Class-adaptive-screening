"""Create a small synthetic input to check installation without image downloads.

Run from the repository root: python examples/make_demo_data.py
These data exercise the pipeline; they do not reproduce the paper's results.
"""

import json
from pathlib import Path

import numpy as np


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "prepared" / "demo"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    class_centers = rng.normal(size=(3, 6))

    for split, per_class in (("train", 40), ("evaluation", 20)):
        labels = np.repeat(np.arange(3), per_class)
        features = class_centers[labels] + rng.normal(size=(len(labels), 6))
        np.savez_compressed(output / f"{split}.npz", X=features, y=labels)

    datasets = {"demo": {"format": "npz", "train": "train.npz", "evaluation": "evaluation.npz"}}
    (output / "datasets.json").write_text(json.dumps(datasets, indent=2), encoding="utf-8")
    print(f"Demo data saved to {output}")


if __name__ == "__main__":
    main()
