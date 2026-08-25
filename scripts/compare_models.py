#!/usr/bin/env python3
"""Create a compact held-out-test comparison for the two intended models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.modeling import load_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("models/baseline_rf.joblib"))
    parser.add_argument("--esm", type=Path, default=Path("models/esm2_logreg.joblib"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/model_comparison.csv")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("outputs/figures/model_comparison.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in [args.baseline, args.esm]:
        artifact = load_artifact(path)
        metrics = artifact["metadata"]["test_metrics"]
        rows.append(
            {
                "model": artifact["model_name"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "mcc": metrics["mcc"],
                "f1": metrics["f1"],
            }
        )
    comparison = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)

    plot_data = comparison.set_index("model")[["average_precision", "mcc", "balanced_accuracy"]]
    axis = plot_data.plot.bar(
        figsize=(9, 5), color=["#2F6BFF", "#D59B2D", "#667085"], rot=0
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test metric")
    axis.set_title("Model comparison on the same leakage-aware test split")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="Metric", frameon=False)
    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, dpi=180, bbox_inches="tight")
    print(f"Saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
