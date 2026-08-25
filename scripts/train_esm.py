#!/usr/bin/env python3
"""Train logistic regression on frozen ESM-2 embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.esm import load_embedding_bundle  # noqa: E402
from amp_finder.evaluation import write_json  # noqa: E402
from amp_finder.modeling import (  # noqa: E402
    save_artifact,
    train_esm_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/processed/esm2_embeddings.npz")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("models/esm2_logreg.joblib")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("outputs/esm2_predictions.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Class-stratified test-set bootstrap resamples (minimum 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_embedding_bundle(args.embeddings)
    artifact, predictions = train_esm_logistic_regression(
        bundle["embeddings"],
        bundle["labels"],
        bundle["splits"],
        bundle["groups"],
        embedding_model_name=str(bundle["model_name"]),
        random_state=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    save_artifact(artifact, args.model_output)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.insert(0, "sequence", bundle["sequences"])
    predictions.to_csv(args.predictions_output, index=False)
    write_json(artifact["metadata"], args.model_output.with_suffix(".metrics.json"))
    print(f"Saved model to {args.model_output}")
    print(f"Test MCC: {artifact['metadata']['test_metrics']['mcc']:.3f}")


if __name__ == "__main__":
    main()
