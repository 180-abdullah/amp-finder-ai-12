#!/usr/bin/env python3
"""Train and evaluate the explainable Random Forest baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.evaluation import write_json  # noqa: E402
from amp_finder.modeling import save_artifact, train_feature_random_forest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/amp_dataset.csv")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("models/baseline_rf.joblib")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("outputs/baseline_predictions.csv"),
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
    frame = pd.read_csv(args.dataset)
    artifact, predictions = train_feature_random_forest(
        frame,
        random_state=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        scientific_status="real_data_demonstration",
        dataset_notes=f"Prepared dataset: {args.dataset}",
    )
    save_artifact(artifact, args.model_output)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)

    metrics_path = args.model_output.with_suffix(".metrics.json")
    write_json(artifact["metadata"], metrics_path)
    importance_path = args.model_output.with_suffix(".feature_importance.csv")
    pd.DataFrame(
        artifact["feature_importance"].items(), columns=["feature", "importance"]
    ).to_csv(importance_path, index=False)
    print(f"Saved model to {args.model_output}")
    print(f"Saved predictions to {args.predictions_output}")
    print(f"Test MCC: {artifact['metadata']['test_metrics']['mcc']:.3f}")


if __name__ == "__main__":
    main()
