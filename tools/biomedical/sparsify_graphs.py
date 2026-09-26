"""Apply topology-aware, relation-stratified sparsification to large graphs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from common import Triple, iter_tsv_files, read_triples, write_triples


@dataclass(frozen=True)
class SparsifyStats:
    dataset: str
    nodes: int
    original_triples: int
    target_triples: int
    retained_triples: int


def dynamic_target(
    node_count: int,
    edge_count: int,
    *,
    target_average_degree: float,
    pareto_ratio: float,
) -> int:
    """Compute the target edge count used by the reported benchmark."""
    degree_target = int(node_count * target_average_degree / 2)
    pareto_target = int(edge_count * pareto_ratio)
    connectivity_floor = int(node_count * 3 / 2)
    return min(edge_count, max(degree_target, pareto_target, connectivity_floor))


def sparsify(
    triples: list[Triple],
    *,
    target_average_degree: float = 15.0,
    pareto_ratio: float = 0.25,
) -> tuple[list[Triple], int, int]:
    """Retain a degree-aware skeleton and allocate the remaining budget by relation."""
    degree: dict[str, int] = defaultdict(int)
    incident: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        head, _, tail = triple
        degree[head] += 1
        degree[tail] += 1
        incident[head].append(triple)
        incident[tail].append(triple)

    target = dynamic_target(
        len(degree),
        len(triples),
        target_average_degree=target_average_degree,
        pareto_ratio=pareto_ratio,
    )
    if target >= len(triples):
        return list(triples), target, len(degree)

    skeleton: set[Triple] = set()
    for node in sorted(degree):
        candidates = sorted(incident[node])
        best = max(
            candidates,
            key=lambda triple: (
                degree[triple[2] if triple[0] == node else triple[0]],
                triple,
            ),
        )
        skeleton.add(best)

    remaining_quota = target - len(skeleton)
    selected = sorted(skeleton)
    if remaining_quota <= 0:
        return selected, target, len(degree)

    candidates_by_relation: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        if triple not in skeleton:
            candidates_by_relation[triple[1]].append(triple)

    candidate_count = sum(len(rows) for rows in candidates_by_relation.values())
    if candidate_count == 0:
        return selected, target, len(degree)

    for relation in sorted(candidates_by_relation):
        candidates = candidates_by_relation[relation]
        relation_quota = int(remaining_quota * len(candidates) / candidate_count)
        if relation_quota <= 0:
            continue
        ranked = sorted(
            candidates,
            key=lambda triple: (
                math.log1p(degree[triple[0]]) * math.log1p(degree[triple[2]]),
                triple,
            ),
            reverse=True,
        )
        selected.extend(ranked[:relation_quota])

    return sorted(set(selected)), target, len(degree)


def write_report(path: Path, rows: list[SparsifyStats]) -> None:
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
    target_average_degree: float,
    pareto_ratio: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SparsifyStats] = []
    for input_path in iter_tsv_files(input_dir):
        triples = read_triples(input_path)
        retained, target, node_count = sparsify(
            triples,
            target_average_degree=target_average_degree,
            pareto_ratio=pareto_ratio,
        )
        write_triples(output_dir / input_path.name, retained)
        stats = SparsifyStats(
            dataset=input_path.stem,
            nodes=node_count,
            original_triples=len(triples),
            target_triples=target,
            retained_triples=len(retained),
        )
        rows.append(stats)
        print(
            f"{stats.dataset}: retained {stats.retained_triples:,}/"
            f"{stats.original_triples:,} triples (target={stats.target_triples:,})"
        )
    if rows:
        write_report(report, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=Path("sparsify_report.csv"))
    parser.add_argument("--target-average-degree", type=float, default=15.0)
    parser.add_argument("--pareto-ratio", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        args.input_dir,
        args.output_dir,
        args.report,
        target_average_degree=args.target_average_degree,
        pareto_ratio=args.pareto_ratio,
    )

