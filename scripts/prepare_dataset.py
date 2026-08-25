#!/usr/bin/env python3
"""Build a clean AMP/putative-non-AMP dataset with leakage-aware splits."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.data import build_labeled_dataset  # noqa: E402
from amp_finder.evaluation import write_json  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-fasta", type=Path, required=True)
    parser.add_argument("--negative-parent-fasta", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/amp_dataset.csv")
    )
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--split-mode", choices=["similarity", "random"], default="similarity"
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, metadata = build_labeled_dataset(
        args.positive_fasta,
        args.negative_parent_fasta,
        min_length=args.min_length,
        max_length=args.max_length,
        negative_ratio=args.negative_ratio,
        split_mode=args.split_mode,
        similarity_threshold=args.similarity_threshold,
        random_state=args.seed,
    )
    metadata["source_files"] = {
        "positive_fasta": {
            "path": str(args.positive_fasta),
            "sha256": file_sha256(args.positive_fasta),
        },
        "negative_parent_fasta": {
            "path": str(args.negative_parent_fasta),
            "sha256": file_sha256(args.negative_parent_fasta),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    write_json(metadata, metadata_path)
    print(f"Saved {len(dataset):,} rows to {args.output}")
    print(f"Saved data-quality and split metadata to {metadata_path}")


if __name__ == "__main__":
    main()
