"""Create difficulty-stratified transductive train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from common import Triple, iter_tsv_files, read_triples, write_triples


@dataclass(frozen=True)
class SplitStats:
    dataset: str
    total: int
    train: int
    valid: int
    test: int


def split_graph(
    triples: list[Triple],
    *,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2026,
) -> tuple[list[Triple], list[Triple], list[Triple]]:
    """Split one graph while covering every entity and relation in training."""
    rng = random.Random(seed)
    degree: dict[str, int] = defaultdict(int)
    unseen_entities: set[str] = set()
    unseen_relations: set[str] = set()
    for head, relation, tail in triples:
        unseen_entities.update((head, tail))
        unseen_relations.add(relation)
        degree[head] += 1
        degree[tail] += 1

    shuffled = list(triples)
    rng.shuffle(shuffled)
    train: list[Triple] = []
    candidates: list[Triple] = []
    for triple in shuffled:
        head, relation, tail = triple
        if head in unseen_entities or tail in unseen_entities or relation in unseen_relations:
            train.append(triple)
            unseen_entities.discard(head)
            unseen_entities.discard(tail)
            unseen_relations.discard(relation)
        else:
            candidates.append(triple)

    scored = [
        (
            1.0
            / (
                math.log1p(degree[head] + 1.718)
                * math.log1p(degree[tail] + 1.718)
            ),
            (head, relation, tail),
        )
        for head, relation, tail in candidates
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    evaluation_cutoff = int(len(scored) * 0.8)
    evaluation_pool = [triple for _, triple in scored[:evaluation_cutoff]]
    train.extend(triple for _, triple in scored[evaluation_cutoff:])
    rng.shuffle(evaluation_pool)

    adjacency: dict[str, list[int]] = defaultdict(list)
    for index, (head, _, tail) in enumerate(evaluation_pool):
        adjacency[head].append(index)
        adjacency[tail].append(index)

    used: set[int] = set()

    def coherent_sample(target_size: int) -> list[Triple]:
        selected: list[Triple] = []
        for index, triple in enumerate(evaluation_pool):
            if len(selected) >= target_size:
                break
            if index in used:
                continue
            selected.append(triple)
            used.add(index)
            head, _, tail = triple
            neighbors = adjacency[head] + adjacency[tail]
            rng.shuffle(neighbors)
            for neighbor_index in neighbors:
                if neighbor_index not in used and len(selected) < target_size:
                    selected.append(evaluation_pool[neighbor_index])
                    used.add(neighbor_index)
                    break
        return selected

    total = len(triples)
    test = coherent_sample(int(total * test_ratio))
    valid = coherent_sample(int(total * valid_ratio))
    train.extend(
        triple for index, triple in enumerate(evaluation_pool) if index not in used
    )
    rng.shuffle(train)
    return train, valid, test


def write_report(path: Path, rows: list[SplitStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    report: Path,
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    if valid_ratio < 0 or test_ratio < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio and test_ratio must be non-negative and sum to less than 1")
    rows: list[SplitStats] = []
    for input_path in iter_tsv_files(input_dir):
        triples = read_triples(input_path)
        train, valid, test = split_graph(
            triples,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        raw_dir = output_dir / input_path.stem / "raw"
        write_triples(raw_dir / "train.txt", train)
        write_triples(raw_dir / "valid.txt", valid)
        write_triples(raw_dir / "test.txt", test)
        stats = SplitStats(
            dataset=input_path.stem,
            total=len(triples),
            train=len(train),
            valid=len(valid),
            test=len(test),
        )
        rows.append(stats)
        print(
            f"{stats.dataset}: train={stats.train:,}, valid={stats.valid:,}, "
            f"test={stats.test:,}"
        )
    if rows:
        write_report(report, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=Path("split_report.csv"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        args.input_dir,
        args.output_dir,
        args.report,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

