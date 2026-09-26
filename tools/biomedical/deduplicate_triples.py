"""Remove duplicate triples while preserving their first occurrence."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import Triple, iter_tsv_files, read_triples, write_triples


def deduplicate(triples: list[Triple]) -> list[Triple]:
    """Return the first occurrence of each triple in input order."""
    seen: set[Triple] = set()
    unique: list[Triple] = []
    for triple in triples:
        if triple not in seen:
            seen.add(triple)
            unique.append(triple)
    return unique


def process_directory(input_dir: Path, output_dir: Path) -> None:
    """Deduplicate every TSV graph in ``input_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in iter_tsv_files(input_dir):
        triples = read_triples(input_path)
        unique = deduplicate(triples)
        write_triples(output_dir / input_path.name, unique)
        print(
            f"{input_path.stem}: retained {len(unique):,}/{len(triples):,} triples "
            f"({len(triples) - len(unique):,} duplicates removed)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(args.input_dir, args.output_dir)

