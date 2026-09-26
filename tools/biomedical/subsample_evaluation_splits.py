"""Optionally create relation-stratified validation and test subsets."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import Triple, read_triples, write_triples


def stratified_sample(
    triples: list[Triple], *, ratio: float, seed: int
) -> list[Triple]:
    """Sample each relation independently while retaining at least one row."""
    by_relation: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        by_relation[triple[1]].append(triple)
    rng = random.Random(seed)
    sampled: list[Triple] = []
    for relation in sorted(by_relation):
        rows = by_relation[relation]
        count = min(len(rows), max(1, round(len(rows) * ratio)))
        sampled.extend(rng.sample(rows, count))
    rng.shuffle(sampled)
    return sampled


def kl_divergence(reference: list[Triple], sample: list[Triple]) -> float:
    """Compute KL(reference || sample) over relation frequencies."""
    reference_counts = Counter(triple[1] for triple in reference)
    sample_counts = Counter(triple[1] for triple in sample)
    reference_total = sum(reference_counts.values())
    sample_total = sum(sample_counts.values())
    epsilon = 1e-9
    value = 0.0
    for relation, count in reference_counts.items():
        p_value = count / reference_total
        q_value = sample_counts.get(relation, 0) / sample_total if sample_total else 0.0
        value += p_value * math.log(p_value / max(q_value, epsilon))
    return value


def mean_train_degree(triples: list[Triple], train: list[Triple]) -> float:
    degree = Counter(entity for head, _, tail in train for entity in (head, tail))
    if not triples:
        return 0.0
    return sum((degree[head] + degree[tail]) / 2 for head, _, tail in triples) / len(triples)


def process_split(
    name: str,
    triples: list[Triple],
    train: list[Triple],
    *,
    ratio: float,
    seed: int,
) -> list[Triple]:
    sampled = stratified_sample(triples, ratio=ratio, seed=seed)
    original_degree = mean_train_degree(triples, train)
    sampled_degree = mean_train_degree(sampled, train)
    degree_difference = (
        abs(original_degree - sampled_degree) / original_degree
        if original_degree > 0
        else 0.0
    )
    print(
        f"{name}: sampled {len(sampled):,}/{len(triples):,}; "
        f"relation KL={kl_divergence(triples, sampled):.6f}; "
        f"mean-degree relative difference={degree_difference:.2%}"
    )
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-input", default="valid.txt")
    parser.add_argument("--test-input", default="test.txt")
    parser.add_argument("--valid-output", default="valid_sampled.txt")
    parser.add_argument("--test-output", default="test_sampled.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0 < args.ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    train_rows = read_triples(args.dataset_dir / "train.txt")
    valid_rows = read_triples(args.dataset_dir / args.valid_input)
    test_rows = read_triples(args.dataset_dir / args.test_input)
    sampled_valid = process_split(
        "validation", valid_rows, train_rows, ratio=args.ratio, seed=args.seed
    )
    sampled_test = process_split(
        "test", test_rows, train_rows, ratio=args.ratio, seed=args.seed
    )
    write_triples(args.dataset_dir / args.valid_output, sampled_valid)
    write_triples(args.dataset_dir / args.test_output, sampled_test)
