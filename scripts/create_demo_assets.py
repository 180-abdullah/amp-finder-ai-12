#!/usr/bin/env python3
"""Create deterministic toy data/model for UI and installation smoke tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.modeling import save_artifact, train_feature_random_forest  # noqa: E402
from amp_finder.sequence import sequence_sha256  # noqa: E402
from amp_finder.splitting import assign_splits  # noqa: E402


def weighted_sequence(
    random_generator: random.Random,
    length: int,
    alphabet: str,
    weights: list[float],
) -> str:
    return "".join(random_generator.choices(alphabet, weights=weights, k=length))


def unique_sequences(
    random_generator: random.Random,
    *,
    count: int,
    alphabet: str,
    weights: list[float],
    occupied: set[str],
) -> list[str]:
    sequences: list[str] = []
    while len(sequences) < count:
        length = random_generator.randint(10, 42)
        sequence = weighted_sequence(random_generator, length, alphabet, weights)
        if sequence not in occupied:
            occupied.add(sequence)
            sequences.append(sequence)
    return sequences


def main() -> None:
    random_generator = random.Random(20260825)
    occupied: set[str] = set()
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    amp_weights = [
        7, 1, 1, 1, 5, 2, 1, 8, 13, 8, 5, 2, 2, 3, 12, 3, 3, 7, 2, 7
    ]
    non_amp_weights = [
        7, 2, 7, 8, 4, 7, 3, 6, 3, 7, 5, 4, 4, 6, 3, 7, 6, 5, 2, 6
    ]
    amps = unique_sequences(
        random_generator,
        count=120,
        alphabet=alphabet,
        weights=amp_weights,
        occupied=occupied,
    )
    non_amps = unique_sequences(
        random_generator,
        count=120,
        alphabet=alphabet,
        weights=non_amp_weights,
        occupied=occupied,
    )

    frame = pd.DataFrame(
        [
            {"sequence": sequence, "label": 1, "class_name": "Synthetic AMP-like"}
            for sequence in amps
        ]
        + [
            {"sequence": sequence, "label": 0, "class_name": "Synthetic non-AMP-like"}
            for sequence in non_amps
        ]
    )
    frame["source"] = "Synthetic UI smoke-test data"
    frame["source_id"] = [f"toy_{index:04d}" for index in range(len(frame))]
    frame["parent_id"] = frame["source_id"]
    frame["sequence_sha256"] = frame["sequence"].map(sequence_sha256)
    frame["length"] = frame["sequence"].str.len()
    frame = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    frame = assign_splits(frame, mode="random", random_state=42)

    data_path = PROJECT_ROOT / "data/demo/demo_sequences.csv"
    model_path = PROJECT_ROOT / "models/demo_baseline_rf.joblib"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False)
    artifact, _ = train_feature_random_forest(
        frame,
        random_state=42,
        scientific_status="synthetic_ui_demo_only",
        dataset_notes=(
            "Deterministic synthetic sequences created only to smoke-test the "
            "application. Metrics and predictions are not scientific evidence."
        ),
    )
    save_artifact(artifact, model_path)
    print(f"Saved toy data to {data_path}")
    print(f"Saved toy UI model to {model_path}")


if __name__ == "__main__":
    main()
