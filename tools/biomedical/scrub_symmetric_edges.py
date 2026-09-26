"""Canonicalize undirected interactions while preserving directed biochemistry."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import Triple, iter_tsv_files, read_triples, write_triples

DEFAULT_TARGET_SOURCES = {
    "DIP",
    "DrugBank",
    "HPIDb",
    "I2D",
    "InnateDB",
    "IntAct",
    "MINT",
    "Molecular_Connections",
    "STRING",
    "UniProt",
    "bhf-ucl",
    "matrixdb",
    "mbinfo",
}

DIRECTED_KEYWORDS = (
    "REACTION",
    "PTM",
    "PHOSPHO",
    "METHYL",
    "UBIQUITIN",
    "ACETYL",
    "BIOCHEMICAL",
    "SIGNAL",
    "CLEAVAGE",
    "STIMULATE",
    "INHIBIT",
    "MODIFIES",
    "STRUCT",
)


def is_directed_relation(relation: str) -> bool:
    """Return whether the relation denotes a direction-sensitive operation."""
    upper = relation.upper()
    return any(keyword in upper for keyword in DIRECTED_KEYWORDS)


def scrub_graph(source_name: str, triples: list[Triple]) -> tuple[list[Triple], int]:
    """Canonicalize undirected edges for selected interaction databases."""
    cleaned: set[Triple] = set()
    protected = 0
    for head, relation, tail in triples:
        if source_name in DEFAULT_TARGET_SOURCES and not is_directed_relation(relation):
            first, second = sorted((head, tail))
            cleaned.add((first, relation, second))
        else:
            cleaned.add((head, relation, tail))
            if source_name in DEFAULT_TARGET_SOURCES:
                protected += 1
    return sorted(cleaned), protected


def process_directory(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in iter_tsv_files(input_dir):
        triples = read_triples(input_path)
        cleaned, protected = scrub_graph(input_path.stem, triples)
        write_triples(output_dir / input_path.name, cleaned)
        print(
            f"{input_path.stem}: retained {len(cleaned):,}/{len(triples):,} triples; "
            f"protected {protected:,} directed triples"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(args.input_dir, args.output_dir)

