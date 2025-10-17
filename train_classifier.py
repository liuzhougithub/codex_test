"""Command line entry point for the generic classification trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from classification.config import load_config
from classification.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/toy_classification.json"),
        help="Path to a JSON config file describing the training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
