from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_evaluation: np.ndarray
    y_evaluation: np.ndarray
    clean_y_train: np.ndarray | None = None


def _resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def _csv_arrays(
    path: Path,
    feature_prefix: str,
    label: str,
    clean: str | None = None,
    expected_features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    frame = pd.read_csv(path)
    features = [column for column in frame.columns if column.startswith(feature_prefix)]
    if not features:
        raise ValueError(f"No columns beginning with {feature_prefix!r} in {path}")
    if expected_features is not None:
        if set(features) != set(expected_features):
            raise ValueError(f"Training and evaluation feature columns differ: {path}")
        # CSV column order can change during export. Match the training order.
        features = expected_features
    if label not in frame:
        raise ValueError(f"Missing label column {label!r} in {path}")
    if clean is not None and clean not in frame:
        raise ValueError(f"Missing clean-label column {clean!r} in {path}")
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame[label].to_numpy()
    clean_y = None if clean is None else frame[clean].to_numpy()
    return X, y, clean_y, features


def load_dataset(spec: dict[str, Any], base: Path) -> DatasetBundle:
    """Load frozen features, keeping evaluation columns in training order."""
    train_path = _resolve(spec["train"], base)
    evaluation_path = _resolve(spec["evaluation"], base)
    file_format = spec.get("format", "csv").lower()
    if file_format == "npz":
        train = np.load(train_path)
        evaluation = np.load(evaluation_path)
        return DatasetBundle(
            np.asarray(train["X"], dtype=np.float32),
            np.asarray(train["y"]),
            np.asarray(evaluation["X"], dtype=np.float32),
            np.asarray(evaluation["y"]),
            np.asarray(train["clean_y"]) if "clean_y" in train else None,
        )
    if file_format != "csv":
        raise ValueError(f"Unsupported format: {file_format}")
    X_train, y_train, clean, features = _csv_arrays(
        train_path, spec["feature_prefix"], spec["train_label"], spec.get("clean_train_label")
    )
    X_eval, y_eval, _, _ = _csv_arrays(
        evaluation_path,
        spec["feature_prefix"],
        spec["evaluation_label"],
        expected_features=features,
    )
    return DatasetBundle(X_train, y_train, X_eval, y_eval, clean)
