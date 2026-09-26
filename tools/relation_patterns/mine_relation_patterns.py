"""Mine symmetry, low-reciprocity, and composition test subsets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, TypeAlias

Triple: TypeAlias = tuple[str, str, str]

TEST_FILENAMES = ("test.txt", "inf_test.txt", "test_ind.txt")
GENERATED_SUFFIXES = ("_symmetry.txt", "_anti_symmetry.txt", "_composition.txt")
SPARSE_KG_DIRECTORY = "sparsekg"


def parse_triple(line: str, *, head_tail_relation: bool = False) -> Triple | None:
    """Parse a tab- or whitespace-separated triple."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split("\t") if "\t" in stripped else stripped.split()
    if len(fields) != 3:
        return None
    head, second, third = (field.strip() for field in fields)
    if head_tail_relation:
        return head, third, second
    return head, second, third


def find_raw_directories(root: Path) -> list[Path]:
    """Find all directories named ``raw`` below ``root``."""
    return sorted(path for path in root.rglob("raw") if path.is_dir())


def detect_test_files(raw_dir: Path) -> list[Path]:
    """Return the evaluation files used by the dataset loader.

    Grail merges its inductive validation and test queries into the final test
    split by default, so both files must be partitioned. Other loaders expose a
    single test-query file.
    """
    by_name = {path.name.lower(): path for path in raw_dir.glob("*.txt")}
    grail_files = [
        by_name[name]
        for name in ("valid_ind.txt", "test_ind.txt")
        if name in by_name
    ]
    if len(grail_files) == 2:
        return grail_files
    for candidate in TEST_FILENAMES:
        if candidate in by_name:
            return [by_name[candidate]]
    return []


def uses_head_tail_relation(raw_dir: Path) -> bool:
    """Return whether raw triples use SparseKG's ``(head, tail, relation)`` order."""
    return any(part.casefold() == SPARSE_KG_DIRECTORY for part in raw_dir.parts)


def source_files(raw_dir: Path) -> list[Path]:
    """Return split files while excluding previously generated pattern subsets."""
    return [
        path
        for path in sorted(raw_dir.glob("*.txt"))
        if not path.name.lower().endswith(GENERATED_SUFFIXES)
    ]


def load_triples(
    paths: Iterable[Path], *, head_tail_relation: bool = False
) -> list[Triple]:
    triples: list[Triple] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                triple = parse_triple(line, head_tail_relation=head_tail_relation)
                if triple is not None:
                    triples.append(triple)
    return triples


def relation_pairs(triples: Iterable[Triple]) -> dict[str, set[tuple[str, str]]]:
    pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for head, relation, tail in triples:
        pairs[relation].add((head, tail))
    return dict(pairs)


def composition_candidates(triples: Iterable[Triple]) -> set[tuple[str, str]]:
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    for head, relation, tail in triples:
        outgoing[head].add(relation)
        incoming[tail].add(relation)
    candidates: set[tuple[str, str]] = set()
    for entity in sorted(incoming.keys() & outgoing.keys()):
        candidates.update(product(incoming[entity], outgoing[entity]))
    return candidates


def composition_relations(
    triples: list[Triple],
    pairs: dict[str, set[tuple[str, str]]],
    *,
    threshold: float,
) -> set[str]:
    adjacency: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for head, relation, tail in triples:
        adjacency[relation][head].add(tail)

    best_confidence = {relation: 0.0 for relation in pairs}
    for first_relation, second_relation in sorted(composition_candidates(triples)):
        left_hand_side: set[tuple[str, str]] = set()
        for start, middle in pairs.get(first_relation, set()):
            for end in adjacency.get(second_relation, {}).get(middle, set()):
                left_hand_side.add((start, end))
        if not left_hand_side:
            continue
        for target_relation, target_pairs in pairs.items():
            confidence = len(left_hand_side & target_pairs) / len(left_hand_side)
            best_confidence[target_relation] = max(
                best_confidence[target_relation], confidence
            )
    return {
        relation
        for relation, confidence in best_confidence.items()
        if confidence >= threshold
    }


def mine_relations(
    triples: list[Triple], *, threshold: float
) -> tuple[set[str], set[str], set[str]]:
    """Return relation sets for symmetry, low reciprocity, and composition."""
    pairs = relation_pairs(triples)
    symmetry: set[str] = set()
    low_reciprocity: set[str] = set()
    for relation, observed_pairs in pairs.items():
        reciprocal_pairs = {(tail, head) for head, tail in observed_pairs}
        reciprocity = len(observed_pairs & reciprocal_pairs) / len(observed_pairs)
        if reciprocity >= threshold:
            symmetry.add(relation)
        if 1.0 - reciprocity >= threshold:
            low_reciprocity.add(relation)
    composition = composition_relations(triples, pairs, threshold=threshold)
    return symmetry, low_reciprocity, composition


def write_subsets(
    test_path: Path,
    relation_sets: dict[str, set[str]],
    *,
    head_tail_relation: bool,
    overwrite: bool,
) -> dict[str, int]:
    buffers = {name: [] for name in relation_sets}
    with test_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            triple = parse_triple(line, head_tail_relation=head_tail_relation)
            if triple is None:
                continue
            relation = triple[1]
            for subset_name, relations in relation_sets.items():
                if relation in relations:
                    buffers[subset_name].append(line if line.endswith("\n") else f"{line}\n")

    counts: dict[str, int] = {}
    for subset_name, lines in buffers.items():
        if not lines:
            continue
        output_path = test_path.with_name(f"{test_path.stem}_{subset_name}.txt")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")
        output_path.write_text("".join(lines), encoding="utf-8")
        counts[subset_name] = len(lines)
    return counts


def process_raw_directory(raw_dir: Path, *, threshold: float, overwrite: bool) -> None:
    test_paths = detect_test_files(raw_dir)
    if not test_paths:
        print(f"Warning: no supported test file in {raw_dir}")
        return
    head_tail_relation = uses_head_tail_relation(raw_dir)
    triples = load_triples(
        source_files(raw_dir), head_tail_relation=head_tail_relation
    )
    if not triples:
        print(f"Warning: no triples parsed from {raw_dir}")
        return
    symmetry, low_reciprocity, composition = mine_relations(
        triples, threshold=threshold
    )
    relation_sets = {
        "symmetry": symmetry,
        "anti_symmetry": low_reciprocity,
        "composition": composition,
    }
    counts = {
        test_path.name: write_subsets(
            test_path,
            relation_sets,
            head_tail_relation=head_tail_relation,
            overwrite=overwrite,
        )
        for test_path in test_paths
    }
    print(
        f"{raw_dir}: relations(sym={len(symmetry)}, low-reciprocity="
        f"{len(low_reciprocity)}, comp={len(composition)}); triples={counts}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    raw_directories = find_raw_directories(args.root)
    if not raw_directories:
        raise FileNotFoundError(f"No raw directories found below {args.root}")
    for directory in raw_directories:
        process_raw_directory(
            directory, threshold=args.threshold, overwrite=args.overwrite
        )
