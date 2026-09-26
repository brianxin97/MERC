"""Shared TSV input/output helpers for biomedical graph construction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TypeAlias

Triple: TypeAlias = tuple[str, str, str]


def iter_tsv_files(directory: Path) -> Iterator[Path]:
    """Yield visible ``.txt`` files in deterministic filename order."""
    for path in sorted(directory.glob("*.txt")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def read_triples(path: Path) -> list[Triple]:
    """Read tab-separated triples, ignoring blank and malformed rows."""
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != 3:
                print(f"Warning: skipped malformed row {line_number} in {path}")
                continue
            triples.append((fields[0], fields[1], fields[2]))
    return triples


def write_triples(path: Path, triples: Iterable[Triple]) -> None:
    """Write triples as normalized UTF-8 TSV records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for head, relation, tail in triples:
            stream.write(f"{head}\t{relation}\t{tail}\n")

