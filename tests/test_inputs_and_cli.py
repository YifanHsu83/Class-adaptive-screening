"""Regression checks for data alignment and command-line overrides."""

import json

import numpy as np
import pandas as pd
import pytest

from class_adaptive_screening import cli
from class_adaptive_screening.data import load_dataset


def write_dataset(tmp_path, evaluation):
    pd.DataFrame({"clip_0": [1, 2], "clip_1": [3, 4], "label": [0, 1]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    evaluation.to_csv(tmp_path / "evaluation.csv", index=False)
    return {
        "train": "train.csv",
        "evaluation": "evaluation.csv",
        "feature_prefix": "clip_",
        "train_label": "label",
        "evaluation_label": "label",
    }


def test_evaluation_features_follow_training_order(tmp_path):
    spec = write_dataset(tmp_path, pd.DataFrame({"clip_1": [30], "label": [0], "clip_0": [10]}))
    bundle = load_dataset(spec, tmp_path)
    np.testing.assert_array_equal(bundle.X_evaluation, [[10, 30]])


def test_different_feature_names_are_rejected(tmp_path):
    spec = write_dataset(tmp_path, pd.DataFrame({"clip_0": [10], "clip_2": [30], "label": [0]}))
    with pytest.raises(ValueError, match="feature columns differ"):
        load_dataset(spec, tmp_path)


@pytest.mark.parametrize("command", ["primary", "extended", "known-noise"])
def test_seed_override_reaches_experiment_and_manifest(tmp_path, monkeypatch, command):
    dataset_file = tmp_path / "datasets.json"
    experiment_file = tmp_path / "experiment.json"
    output = tmp_path / "output"
    dataset_file.write_text(json.dumps({"demo": {}}), encoding="utf-8")
    experiment_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "load_dataset", lambda spec, base: object())
    received = []

    def capture_run(bundle, config, destination):
        received.append(config.seeds)
        destination.mkdir()

    monkeypatch.setattr(cli, command.replace("-", "_"), capture_run)
    dataset_option = "--dataset" if command == "primary" else "--datasets"
    cli.main(
        [
            command,
            "--config",
            str(dataset_file),
            "--experiment",
            str(experiment_file),
            dataset_option,
            "demo",
            "--output",
            str(output),
            "--seeds",
            "2",
        ]
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert received == [2]
    assert manifest["config"]["seeds"] == 2


def test_zero_seeds_is_rejected():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            [
                "primary",
                "--config",
                "datasets.json",
                "--dataset",
                "demo",
                "--output",
                "outputs",
                "--seeds",
                "0",
            ]
        )
