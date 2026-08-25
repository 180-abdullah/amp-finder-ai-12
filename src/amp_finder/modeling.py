"""Training and persistence for feature and ESM-embedding classifiers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import (
    binary_classification_metrics,
    bootstrap_binary_metric_intervals,
    feature_distribution_summary,
    select_mcc_threshold,
)
from .features import FEATURE_NAMES, extract_feature_frame
from .splitting import split_diagnostics, validate_split_integrity

ARTIFACT_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_training_frame(frame: pd.DataFrame) -> None:
    required = {"sequence", "label", "split", "split_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    expected_splits = {"train", "validation", "test"}
    if set(frame["split"].unique()) != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}; found {sorted(frame['split'].unique())}."
        )
    validate_split_integrity(frame)


def train_feature_random_forest(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    bootstrap_resamples: int = 1000,
    scientific_status: str = "real_data_demonstration",
    dataset_notes: str = "APD positives and length-matched UniProt fragments.",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit a fixed Random Forest and select its threshold on validation data."""

    _validate_training_frame(frame)
    features = extract_feature_frame(frame["sequence"])
    labels = frame["label"].to_numpy(dtype=int)
    train_mask = frame["split"].eq("train").to_numpy()
    validation_mask = frame["split"].eq("validation").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()

    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    estimator.fit(features.loc[train_mask, FEATURE_NAMES], labels[train_mask])

    validation_scores = estimator.predict_proba(
        features.loc[validation_mask, FEATURE_NAMES]
    )[:, 1]
    threshold = select_mcc_threshold(labels[validation_mask], validation_scores)
    test_scores = estimator.predict_proba(features.loc[test_mask, FEATURE_NAMES])[:, 1]

    validation_metrics = binary_classification_metrics(
        labels[validation_mask], validation_scores, threshold=threshold
    )
    test_metrics = binary_classification_metrics(
        labels[test_mask], test_scores, threshold=threshold
    )
    forest = estimator.named_steps["model"]
    feature_importance = dict(
        sorted(
            zip(FEATURE_NAMES, forest.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": "feature_random_forest",
        "model_name": "Random Forest on explainable peptide features",
        "estimator": estimator,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "created_at_utc": _utc_timestamp(),
        "sequence_length_min": int(frame["sequence"].str.len().min()),
        "sequence_length_max": int(frame["sequence"].str.len().max()),
        "feature_summary": feature_distribution_summary(features.loc[train_mask], labels[train_mask]),
        "feature_importance": feature_importance,
        "metadata": {
            "scientific_status": scientific_status,
            "dataset_notes": dataset_notes,
            "negative_class_caveat": (
                "UniProt-derived negatives are putative negatives, not peptides "
                "experimentally proven to lack antimicrobial activity."
            ),
            "score_caveat": (
                "The output is an AMP-likeness model score, not a calibrated "
                "clinical probability or experimental activity measurement."
            ),
            "random_state": random_state,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_confidence_intervals": bootstrap_binary_metric_intervals(
                labels[test_mask],
                test_scores,
                threshold=threshold,
                n_resamples=bootstrap_resamples,
                random_state=random_state,
            ),
            "split_diagnostics": split_diagnostics(frame),
        },
    }

    predictions = frame[["sequence", "label", "split", "split_group"]].copy()
    predictions["score"] = estimator.predict_proba(features[FEATURE_NAMES])[:, 1]
    predictions["prediction"] = (predictions["score"] >= threshold).astype(int)
    return artifact, predictions


def train_esm_logistic_regression(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    groups: np.ndarray,
    *,
    embedding_model_name: str,
    random_state: int = 42,
    bootstrap_resamples: int = 1000,
    scientific_status: str = "real_data_demonstration",
    dataset_notes: str = "ESM-2 mean embeddings of the prepared AMP dataset.",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit logistic regression to frozen ESM-2 mean-pooled embeddings."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    splits = np.asarray(splits).astype(str)
    groups = np.asarray(groups).astype(str)
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("Embeddings must be a 2D array aligned with labels.")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Embeddings require train, validation, and test splits.")
    for group in np.unique(groups):
        if len(np.unique(splits[groups == group])) > 1:
            raise ValueError(f"Group leakage detected for {group}.")

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    estimator.fit(embeddings[train_mask], labels[train_mask])
    validation_scores = estimator.predict_proba(embeddings[validation_mask])[:, 1]
    threshold = select_mcc_threshold(labels[validation_mask], validation_scores)
    test_scores = estimator.predict_proba(embeddings[test_mask])[:, 1]

    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": "esm2_logistic_regression",
        "model_name": "ESM-2 embeddings + Logistic Regression",
        "estimator": estimator,
        "embedding_model_name": embedding_model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "threshold": threshold,
        "created_at_utc": _utc_timestamp(),
        "metadata": {
            "scientific_status": scientific_status,
            "dataset_notes": dataset_notes,
            "score_caveat": (
                "The output is an AMP-likeness model score, not a calibrated "
                "clinical probability or experimental activity measurement."
            ),
            "random_state": random_state,
            "validation_metrics": binary_classification_metrics(
                labels[validation_mask], validation_scores, threshold=threshold
            ),
            "test_metrics": binary_classification_metrics(
                labels[test_mask], test_scores, threshold=threshold
            ),
            "test_confidence_intervals": bootstrap_binary_metric_intervals(
                labels[test_mask],
                test_scores,
                threshold=threshold,
                n_resamples=bootstrap_resamples,
                random_state=random_state,
            ),
        },
    }
    all_scores = estimator.predict_proba(embeddings)[:, 1]
    predictions = pd.DataFrame(
        {
            "label": labels,
            "split": splits,
            "split_group": groups,
            "score": all_scores,
            "prediction": (all_scores >= threshold).astype(int),
        }
    )
    return artifact, predictions


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Persist a versioned model artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a model artifact."""

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact version.")
    if "estimator" not in artifact or "threshold" not in artifact:
        raise ValueError("Model artifact is incomplete.")
    return artifact
