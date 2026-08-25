"""Shared single and batch inference for the Streamlit application."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .esm import embed_sequences
from .features import FEATURE_NAMES, extract_feature_frame
from .sequence import normalize_sequence


def interpretation_label(
    score: float, threshold: float, uncertainty_margin: float = 0.10
) -> str:
    """Return a cautious three-way screening interpretation."""

    lower = max(0.0, threshold - uncertainty_margin)
    upper = min(1.0, threshold + uncertainty_margin)
    if lower <= score <= upper:
        return "Uncertain / borderline"
    return "AMP-like" if score > upper else "Non-AMP-like"


def predict_sequences(
    artifact: dict[str, Any],
    sequences: Iterable[str],
    *,
    esm_batch_size: int = 16,
    esm_device: str = "auto",
) -> pd.DataFrame:
    """Predict normalized sequences with a supported artifact."""

    normalized = [normalize_sequence(sequence) for sequence in sequences]
    if not normalized:
        return pd.DataFrame(columns=["sequence", "score", "interpretation"])

    kind = artifact.get("kind")
    estimator = artifact["estimator"]
    if kind == "feature_random_forest":
        features = extract_feature_frame(normalized)
        scores = estimator.predict_proba(features[artifact["feature_names"]])[:, 1]
        output = pd.concat(
            [pd.DataFrame({"sequence": normalized}), features.reset_index(drop=True)],
            axis=1,
        )
    elif kind == "esm2_logistic_regression":
        embeddings = embed_sequences(
            normalized,
            model_name=artifact["embedding_model_name"],
            batch_size=esm_batch_size,
            device=esm_device,
        )
        scores = estimator.predict_proba(embeddings)[:, 1]
        output = pd.DataFrame({"sequence": normalized})
    else:
        raise ValueError(f"Unsupported model kind: {kind!r}")

    threshold = float(artifact["threshold"])
    output["score"] = np.asarray(scores, dtype=float)
    output["threshold"] = threshold
    output["interpretation"] = [
        interpretation_label(float(score), threshold) for score in scores
    ]
    output["binary_prediction"] = (output["score"] >= threshold).astype(int)
    return output


def biological_context(
    feature_row: pd.Series | dict[str, float], artifact: dict[str, Any]
) -> list[str]:
    """Generate descriptive, non-causal context from explainable features."""

    features = dict(feature_row)
    statements: list[str] = []
    charge = float(features["net_charge_pH7"])
    gravy = float(features["gravy"])
    moment = float(features["hydrophobic_moment"])
    length = int(round(float(features["length"])))

    if charge >= 2.0:
        statements.append(
            "The peptide is strongly cationic at pH 7, a common but non-specific AMP characteristic."
        )
    elif charge > 0:
        statements.append("The peptide has a mildly positive estimated charge at pH 7.")
    else:
        statements.append("The peptide is neutral or negatively charged at pH 7.")

    if -0.5 <= gravy <= 1.2:
        statements.append(
            "Its mean hydropathy is in a mixed polar/hydrophobic range compatible with membrane interaction."
        )
    elif gravy > 1.2:
        statements.append(
            "It is highly hydrophobic; this may aid membrane binding but can also raise aggregation or toxicity concerns."
        )
    else:
        statements.append("It is relatively hydrophilic by the Kyte-Doolittle scale.")

    if moment >= 0.30:
        statements.append(
            "The alpha-helical hydrophobic-moment descriptor suggests amphipathic residue organization."
        )
    else:
        statements.append(
            "The alpha-helical hydrophobic-moment descriptor is modest; other conformations may still be relevant."
        )

    if length <= 50:
        statements.append(f"Its length ({length} aa) is within a commonly studied short-peptide range.")
    else:
        statements.append(f"Its length ({length} aa) is above the common 10–50 aa teaching range.")

    if artifact.get("kind") == "feature_random_forest":
        statements.append(
            "These are descriptive feature cues, not local causal explanations of the model score."
        )
    return statements
