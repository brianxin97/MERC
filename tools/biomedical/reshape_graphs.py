"""Inject co-occurrence edges and add topology/flow relation suffixes."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from common import Triple, iter_tsv_files, read_triples, write_triples


@dataclass(frozen=True)
class ReshapeStats:
    dataset: str
    original_triples: int
    injected_triples: int
    output_triples: int
    bipartite_index: float
    flow_asymmetric_triples: int


class BiomedicalGraphReshaper:
    """Apply the topology-aware construction described in Appendix G."""

    def __init__(
        self,
        *,
        bipartite_threshold: float = 0.15,
        max_injected_per_target: int = 30,
        seed: int = 2026,
    ) -> None:
        self.bipartite_threshold = bipartite_threshold
        self.max_injected_per_target = max_injected_per_target
        self.rng = random.Random(seed)

    @staticmethod
    def bipartite_index(
        triples: list[Triple],
    ) -> tuple[float, dict[str, set[str]]]:
        heads: set[str] = set()
        tails: set[str] = set()
        target_to_sources: dict[str, set[str]] = defaultdict(set)
        for head, _, tail in triples:
            heads.add(head)
            tails.add(tail)
            target_to_sources[tail].add(head)
        union = heads | tails
        score = len(heads & tails) / len(union) if union else 0.0
        return score, target_to_sources

    def inject_cooccurrence(
        self, triples: list[Triple], target_to_sources: dict[str, set[str]]
    ) -> tuple[list[Triple], int]:
        augmented = list(triples)
        injected = 0
        for target in sorted(target_to_sources):
            sources = sorted(target_to_sources[target])
            count = len(sources)
            if 1 < count <= 50:
                added_for_target = 0
                for left_index in range(count):
                    for right_index in range(left_index + 1, count):
                        left = sources[left_index]
                        right = sources[right_index]
                        augmented.extend(
                            [
                                (left, "CO_OCCURS_WITH", right),
                                (right, "CO_OCCURS_WITH", left),
                            ]
                        )
                        added_for_target += 2
                        injected += 2
                        if added_for_target >= self.max_injected_per_target:
                            break
                    if added_for_target >= self.max_injected_per_target:
                        break
            elif count > 50:
                sampled = self.rng.sample(sources, min(count, 30))
                for index in range(len(sampled) - 1):
                    left, right = sampled[index], sampled[index + 1]
                    augmented.extend(
                        [
                            (left, "CO_OCCURS_WITH", right),
                            (right, "CO_OCCURS_WITH", left),
                        ]
                    )
                    injected += 2
        return augmented, injected

    @staticmethod
    def topology(
        triples: list[Triple],
    ) -> tuple[dict[str, set[str]], dict[str, float]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        degree: dict[str, int] = defaultdict(int)
        for head, _, tail in triples:
            adjacency[head].add(tail)
            adjacency[tail].add(head)
            degree[head] += 1
            degree[tail] += 1
        centrality: dict[str, float] = {}
        for node, neighbors in adjacency.items():
            mean_neighbor_degree = (
                sum(degree[neighbor] for neighbor in neighbors) / len(neighbors)
                if neighbors
                else 0.0
            )
            centrality[node] = degree[node] + mean_neighbor_degree
        return adjacency, centrality

    @staticmethod
    def edge_clustering(
        head: str, tail: str, adjacency: dict[str, set[str]]
    ) -> float:
        head_neighbors = adjacency[head]
        tail_neighbors = adjacency[tail]
        denominator = min(len(head_neighbors) - 1, len(tail_neighbors) - 1)
        if denominator <= 0:
            return 0.0
        return len(head_neighbors & tail_neighbors) / denominator

    def reshape(self, dataset: str, triples: list[Triple]) -> tuple[list[Triple], ReshapeStats]:
        bipartite_index, target_to_sources = self.bipartite_index(triples)
        working = list(triples)
        injected = 0
        if bipartite_index < self.bipartite_threshold:
            working, injected = self.inject_cooccurrence(working, target_to_sources)

        adjacency, centrality = self.topology(working)
        average_degree = (
            sum(len(neighbors) for neighbors in adjacency.values()) / len(adjacency)
            if adjacency
            else 0.0
        )
        clustering_threshold = 0.3 if average_degree > 5 else 0.1
        reshaped: set[Triple] = set()
        flow_asymmetric = 0

        for head, relation, tail in working:
            coefficient = self.edge_clustering(head, tail, adjacency)
            if coefficient >= clustering_threshold:
                topology_tag = "INTRAMODULAR"
            elif coefficient == 0 and len(adjacency[head]) > 2 and len(adjacency[tail]) > 2:
                topology_tag = "CROSSTALK"
            else:
                topology_tag = "SYSTEMIC"

            head_centrality = centrality[head]
            tail_centrality = centrality[tail]
            if head_centrality > 1.2 * tail_centrality:
                flow_tag = "DIVERGENT"
                flow_asymmetric += 1
            elif tail_centrality > 1.2 * head_centrality:
                flow_tag = "CONVERGENT"
                flow_asymmetric += 1
            else:
                flow_tag = "LATERAL"
            reshaped.add((head, f"{relation}_{topology_tag}_{flow_tag}", tail))

        output = sorted(reshaped)
        stats = ReshapeStats(
            dataset=dataset,
            original_triples=len(triples),
            injected_triples=injected,
            output_triples=len(output),
            bipartite_index=bipartite_index,
            flow_asymmetric_triples=flow_asymmetric,
        )
        return output, stats


def write_report(path: Path, rows: list[ReshapeStats]) -> None:
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
    seed: int,
    bipartite_threshold: float,
    max_injected_per_target: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reshaper = BiomedicalGraphReshaper(
        bipartite_threshold=bipartite_threshold,
        max_injected_per_target=max_injected_per_target,
        seed=seed,
    )
    rows: list[ReshapeStats] = []
    for input_path in iter_tsv_files(input_dir):
        output, stats = reshaper.reshape(input_path.stem, read_triples(input_path))
        write_triples(output_dir / input_path.name, output)
        rows.append(stats)
        print(
            f"{stats.dataset}: B={stats.bipartite_index:.4f}, "
            f"injected={stats.injected_triples:,}, output={stats.output_triples:,}"
        )
    if rows:
        write_report(report, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reshape_report.csv"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bipartite-threshold", type=float, default=0.15)
    parser.add_argument("--max-injected-per-target", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        args.input_dir,
        args.output_dir,
        args.report,
        seed=args.seed,
        bipartite_threshold=args.bipartite_threshold,
        max_injected_per_target=args.max_injected_per_target,
    )
