"""Apply the configured relation-stratified caps to oversized biomedical splits."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from common import read_triples, write_triples
from subsample_evaluation_splits import process_split


def preserve_full_split(current: Path, full: Path) -> Path:
    if full.exists():
        return full
    if not current.exists():
        raise FileNotFoundError(f"Missing both {current} and preserved full split {full}")
    shutil.copy2(current, full)
    return full


def apply_config(dataset_root: Path, config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    valid_input = str(config["valid_input"])
    test_input = str(config["test_input"])
    valid_output = str(config["valid_output"])
    test_output = str(config["test_output"])

    for dataset, ratio_value in config["datasets"].items():
        raw_dir = dataset_root / str(dataset) / "raw"
        train = read_triples(raw_dir / "train.txt")
        valid_source = preserve_full_split(
            raw_dir / valid_output, raw_dir / valid_input
        )
        test_source = preserve_full_split(
            raw_dir / test_output, raw_dir / test_input
        )
        ratio = float(ratio_value)
        valid = process_split(
            f"{dataset}/valid", read_triples(valid_source), train,
            ratio=ratio, seed=seed,
        )
        test = process_split(
            f"{dataset}/test", read_triples(test_source), train,
            ratio=ratio, seed=seed,
        )
        write_triples(raw_dir / valid_output, valid)
        write_triples(raw_dir / test_output, test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/biomedical/evaluation_subsampling.yaml"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_config(args.dataset_root, args.config)
