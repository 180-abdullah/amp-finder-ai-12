"""Model metrics, validation-threshold selection, and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_mcc_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select a classification threshold on validation data only."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both classes.")

    candidates = np.unique(
        np.concatenate([np.linspace(0.05, 0.95, 181), scores])
    )
    best_threshold = 0.5
    best_key = (-np.inf, -np.inf)
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        mcc = matthews_corrcoef(labels, predictions)
        tie_breaker = -abs(float(threshold) - 0.5)
        if (mcc, tie_breaker) > best_key:
            best_key = (mcc, tie_breaker)
            best_threshold = float(threshold)
    return best_threshold


def binary_classification_metrics(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    """Return threshold-free and thresholded binary metrics."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_binary_metric_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate stratified percentile-bootstrap intervals on a fixed test set.

    The threshold must already have been selected outside the test set. These
    intervals quantify finite-test-sample uncertainty only; they do not account
    for data-curation choices, model refitting, assay noise, or domain shift.
    """

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("Labels and scores must be aligned one-dimensional arrays.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Bootstrap intervals require both binary classes.")
    if n_resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must be between 0 and 1.")

    metric_names = (
        "roc_auc",
        "average_precision",
        "mcc",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
        "f1",
        "brier_score",
    )
    estimates = binary_classification_metrics(labels, scores, threshold=threshold)
    distributions = {name: np.empty(n_resamples, dtype=float) for name in metric_names}
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    random = np.random.default_rng(random_state)

    for resample_index in range(n_resamples):
        sampled_indices = np.concatenate(
            [
                random.choice(negative_indices, size=len(negative_indices), replace=True),
                random.choice(positive_indices, size=len(positive_indices), replace=True),
            ]
        )
        sampled_metrics = binary_classification_metrics(
            labels[sampled_indices], scores[sampled_indices], threshold=threshold
        )
        for name in metric_names:
            distributions[name][resample_index] = float(sampled_metrics[name])

    alpha = (1.0 - confidence_level) / 2.0
    intervals = {
        name: {
            "estimate": float(estimates[name]),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for name, values in distributions.items()
    }
    return {
        "method": "class-stratified percentile bootstrap on the fixed held-out test set",
        "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples),
        "random_state": int(random_state),
        "scope_caveat": (
            "Intervals reflect finite test-sample uncertainty conditional on the fitted model, "
            "fixed threshold, labels, and data pipeline; they do not quantify domain shift or label error."
        ),
        "metrics": intervals,
    }


def feature_distribution_summary(
    features: pd.DataFrame, labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """Create compact class-specific medians and robust ranges for app context."""

    labels = np.asarray(labels, dtype=int)
    summary: dict[str, dict[str, float]] = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype=float)
        amp_values = values[labels == 1]
        non_amp_values = values[labels == 0]
        summary[column] = {
            "overall_q05": float(np.quantile(values, 0.05)),
            "overall_median": float(np.median(values)),
            "overall_q95": float(np.quantile(values, 0.95)),
            "amp_median": float(np.median(amp_values)),
            "non_amp_median": float(np.median(non_amp_values)),
        }
    return summary


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write deterministic, human-readable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
