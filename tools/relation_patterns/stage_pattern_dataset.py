"""Stage a mined relation-pattern subset as a local MERC evaluation dataset.

This utility copies source splits into the layouts consumed by
``CustomTransductive`` or ``CustomInductive``. It never overwrites the original
dataset files, so pattern evaluation does not require swapping test splits in
place or invalidating an existing processed cache.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def validate_name(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or os.path.basename(value) != value:
        raise ValueError("dataset name and version must be single directory names")
    return value


def copy_splits(mapping: list[tuple[Path, Path]]) -> None:
    """Validate the complete staging operation before copying any file."""
    missing = [source for source, _ in mapping if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source split(s): " + ", ".join(str(path) for path in missing)
        )
    existing = [destination for _, destination in mapping if destination.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite staged split(s): "
            + ", ".join(str(path) for path in existing)
            + "; choose a new --name or remove the staged copy"
        )

    mapping[0][1].parent.mkdir(parents=True, exist_ok=True)
    for source, destination in mapping:
        shutil.copy2(source, destination)


def stage_transductive(args: argparse.Namespace) -> Path:
    destination = args.output_root / validate_name(args.name) / "raw"
    copy_splits([
        (args.train, destination / "train.txt"),
        (args.valid, destination / "valid.txt"),
        (args.subset, destination / "test.txt"),
    ])
    return destination


def stage_inductive(args: argparse.Namespace) -> Path:
    destination = (
        args.output_root
        / validate_name(args.name)
        / validate_name(args.version)
        / "raw"
    )
    copy_splits([
        (args.train_graph, destination / "transductive_train.txt"),
        (args.inference_graph, destination / "inference_graph.txt"),
        (args.valid, destination / "inf_valid.txt"),
        (args.subset, destination / "inf_test.txt"),
    ])
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    transductive = subparsers.add_parser("transductive")
    transductive.add_argument("--train", type=Path, required=True)
    transductive.add_argument("--valid", type=Path, required=True)
    transductive.add_argument("--subset", type=Path, required=True)
    transductive.add_argument("--output-root", type=Path, required=True)
    transductive.add_argument("--name", required=True)
    transductive.set_defaults(handler=stage_transductive)

    inductive = subparsers.add_parser("inductive")
    inductive.add_argument("--train-graph", type=Path, required=True)
    inductive.add_argument("--inference-graph", type=Path, required=True)
    inductive.add_argument("--valid", type=Path, required=True)
    inductive.add_argument("--subset", type=Path, required=True)
    inductive.add_argument("--output-root", type=Path, required=True)
    inductive.add_argument("--name", required=True)
    inductive.add_argument("--version", default="v1")
    inductive.set_defaults(handler=stage_inductive)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = arguments.handler(arguments)
    print(f"Staged pattern dataset in {output}")
