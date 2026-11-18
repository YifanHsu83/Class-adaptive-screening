from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig, load_dataset_configs
from .data import load_dataset
from .experiments import extended, known_noise, primary, write_manifest


def positive_integer(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Class-adaptive screening experiments")
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    for name in ("primary", "extended", "known-noise"):
        item = subparsers.add_parser(name)
        item.add_argument("--config", required=True, help="Dataset JSON configuration")
        item.add_argument("--experiment", default="configs/experiment.json")
        item.add_argument("--output", required=True)
        item.add_argument(
            "--seeds",
            type=positive_integer,
            default=None,
            help="Number of random seeds; overrides the experiment configuration",
        )
        if name == "primary":
            item.add_argument("--dataset", required=True)
        else:
            item.add_argument("--datasets", nargs="+", required=True)
    return argument_parser


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = ExperimentConfig.from_json(args.experiment)
    # Keep the executed settings and the saved manifest in agreement.
    if args.seeds is not None:
        config = replace(config, seeds=args.seeds)
    specifications, base = load_dataset_configs(args.config)
    names = [args.dataset] if args.command == "primary" else args.datasets
    missing = [name for name in names if name not in specifications]
    if missing:
        raise KeyError(f"Datasets absent from configuration: {missing}")
    bundles = {name: load_dataset(specifications[name], base) for name in names}
    output = Path(args.output)
    if args.command == "primary":
        primary(bundles[names[0]], config, output)
    elif args.command == "extended":
        extended(bundles, config, output)
    else:
        known_noise(bundles, config, output)
    write_manifest(output, config, args.command, names)


if __name__ == "__main__":
    main()
