"""Export source-specific CKG relationships from a running Neo4j database.

Each output file is a tab-separated table with four columns:
``start_node``, ``rel_type``, ``end_node``, and ``attributes_json``.  The
attributes are retained because the next pipeline stage uses source-specific
properties to refine relation labels.

The export is batched by Neo4j's internal relationship ID and can resume from
a JSON checkpoint.  Do not modify the database while an export is running:
internal IDs and the maximum-ID boundary must remain stable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from tqdm import tqdm


DEFAULT_IGNORED_RELATIONS = (
    "MENTIONED_IN_PUBLICATION",
    "PUBLISHED_IN",
)
HEADER = ("start_node", "rel_type", "end_node", "attributes_json")


def parse_sources(value: object) -> list[str]:
    """Parse the semicolon- or comma-separated CKG ``source`` property."""
    if value is None:
        return []
    normalized = str(value).strip().replace("[", "").replace("]", "")
    delimiter = ";" if ";" in normalized else "," if "," in normalized else None
    parts = normalized.split(delimiter) if delimiter else [normalized]
    return [part.strip().strip('"').strip("'") for part in parts if part.strip().strip('"').strip("'")]


def safe_source_name(value: str) -> str:
    """Convert a CKG source label into a portable filename stem."""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", value.strip()).strip("._")
    return cleaned or "unknown_source"


class CKGExporter:
    """Batched, checkpointed exporter for a running CKG Neo4j instance."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str | None,
        output_dir: Path,
        checkpoint: Path,
        batch_size: int,
        ignored_relations: tuple[str, ...],
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.ignored_relations = ignored_relations

    def close(self) -> None:
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def get_max_relationship_id(self) -> int | None:
        """Return the maximum internal relationship ID, or ``None`` if empty."""
        with self.session() as session:
            record = session.run(
                "MATCH ()-[r]->() RETURN max(id(r)) AS max_id"
            ).single()
        return record["max_id"] if record is not None else None

    def load_checkpoint(self) -> int:
        if not self.checkpoint.exists():
            return 0
        try:
            payload = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            return int(payload.get("next_relationship_id", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot read checkpoint {self.checkpoint}: {error}") from error

    def save_checkpoint(self, next_relationship_id: int) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_relationship_id": next_relationship_id,
            "timestamp": time.time(),
        }
        temporary = self.checkpoint.with_suffix(self.checkpoint.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.checkpoint)

    def fetch_batch(self, start_id: int, end_id: int) -> list[dict[str, Any]]:
        query = """
        MATCH (a)-[r]->(b)
        WHERE id(r) >= $start_id AND id(r) < $end_id
          AND NOT type(r) IN $ignore_list
          AND r.source IS NOT NULL AND r.source <> ""
        RETURN a.id AS start_node,
               type(r) AS rel_type,
               b.id AS end_node,
               r.source AS source,
               properties(r) AS attributes
        """
        with self.session() as session:
            result = session.run(
                query,
                start_id=start_id,
                end_id=end_id,
                ignore_list=list(self.ignored_relations),
            )
            return [record.data() for record in result]

    def write_batch(self, rows_by_source: dict[str, list[list[object]]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for source, rows in rows_by_source.items():
            output_path = self.output_dir / f"{safe_source_name(source)}.txt"
            write_header = not output_path.exists() or output_path.stat().st_size == 0
            with output_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                if write_header:
                    writer.writerow(HEADER)
                writer.writerows(rows)

    def run(self) -> None:
        self.driver.verify_connectivity()
        max_id = self.get_max_relationship_id()
        if max_id is None:
            print("The database contains no relationships; nothing was exported.")
            return

        next_id = self.load_checkpoint()
        if next_id > max_id + 1:
            raise ValueError(
                f"Checkpoint starts at relationship ID {next_id}, beyond current maximum {max_id}."
            )

        print(f"Maximum Neo4j relationship ID: {max_id}")
        print(f"Starting at relationship ID: {next_id}")
        progress = tqdm(total=max_id + 1, initial=next_id, unit="id", desc="CKG export")
        try:
            for current_start in range(next_id, max_id + 1, self.batch_size):
                current_end = min(current_start + self.batch_size, max_id + 1)
                records = self.fetch_batch(current_start, current_end)
                rows_by_source: dict[str, list[list[object]]] = defaultdict(list)

                for record in records:
                    sources = parse_sources(record.get("source"))
                    if not sources:
                        continue
                    attributes_json = json.dumps(
                        record.get("attributes") or {}, ensure_ascii=False
                    )
                    row = [
                        record.get("start_node"),
                        record.get("rel_type"),
                        record.get("end_node"),
                        attributes_json,
                    ]
                    for source in sources:
                        rows_by_source[source].append(row)

                if rows_by_source:
                    self.write_batch(rows_by_source)
                self.save_checkpoint(current_end)
                progress.update(current_end - current_start)
                del records, rows_by_source
                gc.collect()
        finally:
            progress.close()

        self.checkpoint.unlink(missing_ok=True)
        print(f"Export completed: {self.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="directory for source-specific exports")
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (default: NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j user (default: NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Neo4j password (prefer the NEO4J_PASSWORD environment variable)",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE"),
        help="optional database name (default: NEO4J_DATABASE)",
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint path (default: OUTPUT_DIR/ckg_export_checkpoint.json)",
    )
    parser.add_argument(
        "--ignore-relation",
        action="append",
        dest="ignored_relations",
        help="relationship type to exclude; repeat to replace the default list",
    )
    args = parser.parse_args()
    if not args.password:
        parser.error("set NEO4J_PASSWORD or provide --password")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "ckg_export_checkpoint.json"
    return args


def main() -> None:
    args = parse_args()
    exporter = CKGExporter(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        ignored_relations=tuple(args.ignored_relations or DEFAULT_IGNORED_RELATIONS),
    )
    try:
        exporter.run()
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
