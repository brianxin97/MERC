"""Generate a deterministic SHA-256 manifest for files below a directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=["train.txt", "valid.txt", "test.txt"],
        help="Basenames to include (default: train.txt valid.txt test.txt)",
    )
    parser.add_argument(
        "--dataset-list",
        type=Path,
        help="Optional text file containing one included dataset directory per line",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    allowed = set(args.filenames)
    datasets = None
    if args.dataset_list:
        datasets = {
            line.strip()
            for line in args.dataset_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    files = sorted(
        path
        for path in args.root.rglob("*.txt")
        if path.is_file()
        and path.name in allowed
        and (datasets is None or path.relative_to(args.root).parts[0] in datasets)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("sha256\tpath\n")
        for path in files:
            stream.write(f"{sha256(path)}\t{path.relative_to(args.root).as_posix()}\n")
    print(f"Wrote {len(files)} entries to {args.output}")
