#!/usr/bin/env python3
"""Extract frozen ESM-2 mean-pooled embeddings for the prepared dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.esm import DEFAULT_ESM_MODEL, embed_sequences, save_embedding_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/amp_dataset.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/esm2_embeddings.npz")
    )
    parser.add_argument("--model-name", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset)
    required = {"sequence", "label", "split", "split_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    embeddings = embed_sequences(
        frame["sequence"],
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_embedding_bundle(
        args.output,
        embeddings=embeddings,
        sequences=frame["sequence"].to_numpy(),
        labels=frame["label"].to_numpy(),
        splits=frame["split"].to_numpy(),
        groups=frame["split_group"].to_numpy(),
        model_name=args.model_name,
    )
    print(f"Saved {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
