"""Deterministic random and similarity-aware train/validation/test splits."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def greedy_similarity_groups(
    sequences: list[str], threshold: float = 0.80
) -> np.ndarray:
    """Group peptides by greedy normalized edit similarity.

    This transparent Python fallback is suitable for a demonstration dataset.
    For a formal benchmark, replace it with an alignment-aware clustering tool
    such as MMseqs2 or CD-HIT and provide those cluster identifiers as groups.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("Similarity threshold must be in (0, 1].")

    order = sorted(range(len(sequences)), key=lambda index: (-len(sequences[index]), sequences[index]))
    representatives: list[str] = []
    representative_groups_by_length: dict[int, list[int]] = defaultdict(list)
    assignments = np.full(len(sequences), -1, dtype=int)

    for original_index in order:
        sequence = sequences[original_index]
        sequence_length = len(sequence)
        minimum_length = max(1, math.ceil(sequence_length * threshold))
        maximum_length = math.floor(sequence_length / threshold)

        best_group = -1
        best_similarity = -1.0
        for candidate_length in range(minimum_length, maximum_length + 1):
            for group_id in representative_groups_by_length.get(candidate_length, []):
                similarity = Levenshtein.normalized_similarity(
                    sequence, representatives[group_id]
                )
                if similarity >= threshold and similarity > best_similarity:
                    best_group = group_id
                    best_similarity = similarity

        if best_group < 0:
            best_group = len(representatives)
            representatives.append(sequence)
            representative_groups_by_length[sequence_length].append(best_group)
        assignments[original_index] = best_group

    return assignments


def _best_stratified_group_fold(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups[indices])
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} independent groups; found {len(unique_groups)}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    target_positive_rate = float(labels[indices].mean())
    target_fraction = 1.0 / n_splits
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    local_x = np.zeros((len(indices), 1))
    local_y = labels[indices]
    local_groups = groups[indices]
    for local_train, local_holdout in splitter.split(local_x, local_y, local_groups):
        holdout_rate = float(local_y[local_holdout].mean())
        holdout_fraction = len(local_holdout) / len(indices)
        score = abs(holdout_rate - target_positive_rate) + abs(
            holdout_fraction - target_fraction
        )
        candidate = (score, indices[local_train], indices[local_holdout])
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    return best[1], best[2]


def assign_splits(
    frame: pd.DataFrame,
    *,
    mode: str = "similarity",
    similarity_threshold: float = 0.80,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add ``split_group`` and ``split`` columns to a labeled dataset."""

    required = {"sequence", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Both AMP (1) and non-AMP (0) labels are required.")

    output = frame.copy().reset_index(drop=True)
    labels = output["label"].to_numpy(dtype=int)
    all_indices = np.arange(len(output))

    if mode == "random":
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=0.20,
            stratify=labels,
            random_state=random_state,
        )
        train_indices, validation_indices = train_test_split(
            train_indices,
            test_size=0.25,
            stratify=labels[train_indices],
            random_state=random_state + 1,
        )
        groups = np.arange(len(output), dtype=int)
    elif mode == "similarity":
        groups = greedy_similarity_groups(
            output["sequence"].tolist(), threshold=similarity_threshold
        )
        train_and_validation, test_indices = _best_stratified_group_fold(
            all_indices,
            labels,
            groups,
            n_splits=5,
            random_state=random_state,
        )
        train_indices, validation_indices = _best_stratified_group_fold(
            train_and_validation,
            labels,
            groups,
            n_splits=4,
            random_state=random_state + 1,
        )
    else:
        raise ValueError("Split mode must be 'random' or 'similarity'.")

    split_values = np.full(len(output), "", dtype=object)
    split_values[train_indices] = "train"
    split_values[validation_indices] = "validation"
    split_values[test_indices] = "test"
    if np.any(split_values == ""):
        raise RuntimeError("At least one row was not assigned to a split.")

    output["split_group"] = [f"cluster_{group:06d}" for group in groups]
    output["split"] = split_values
    validate_split_integrity(output)
    return output


def validate_split_integrity(frame: pd.DataFrame) -> None:
    """Raise if groups leak between partitions or exact sequences conflict."""

    if frame["sequence"].duplicated().any():
        raise ValueError("Exact duplicate sequences remain in the dataset.")

    cross_label_counts = frame.groupby("sequence")["label"].nunique()
    if (cross_label_counts > 1).any():
        raise ValueError("At least one sequence has conflicting labels.")

    group_split_counts = frame.groupby("split_group")["split"].nunique()
    if (group_split_counts > 1).any():
        raise ValueError("Similarity group leakage detected across splits.")

    for split_name, subset in frame.groupby("split"):
        if subset["label"].nunique() < 2:
            raise ValueError(f"Split '{split_name}' does not contain both classes.")


def split_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact counts and leakage checks for metadata/reporting."""

    validate_split_integrity(frame)
    by_split = {}
    for split_name, subset in frame.groupby("split", sort=False):
        by_split[split_name] = {
            "rows": int(len(subset)),
            "amps": int(subset["label"].sum()),
            "putative_non_amps": int((subset["label"] == 0).sum()),
            "positive_rate": float(subset["label"].mean()),
            "groups": int(subset["split_group"].nunique()),
        }
    return {
        "rows": int(len(frame)),
        "unique_sequences": int(frame["sequence"].nunique()),
        "groups": int(frame["split_group"].nunique()),
        "group_leakage": False,
        "splits": by_split,
    }
