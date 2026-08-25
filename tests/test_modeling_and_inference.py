from pathlib import Path

import numpy as np
import pandas as pd

from amp_finder.inference import interpretation_label, predict_sequences
from amp_finder.esm import load_embedding_bundle, save_embedding_bundle
from amp_finder.modeling import (
    load_artifact,
    save_artifact,
    train_esm_logistic_regression,
    train_feature_random_forest,
)
from amp_finder.splitting import assign_splits


def synthetic_training_frame() -> pd.DataFrame:
    positive = [
        "K" * 5 + "AILMFWVY"[index % 8] * 5 + "R" * 5 + "A" * (index % 5)
        for index in range(80)
    ]
    negative = [
        "D" * 5 + "STNQ"[index % 4] * 5 + "E" * 5 + "G" * (index % 5)
        for index in range(80)
    ]
    frame = pd.DataFrame(
        [{"sequence": sequence, "label": 1} for sequence in positive]
        + [{"sequence": sequence, "label": 0} for sequence in negative]
    ).drop_duplicates("sequence")
    return assign_splits(frame.reset_index(drop=True), mode="random", random_state=5)


def test_feature_model_round_trip_and_prediction(tmp_path: Path):
    frame = synthetic_training_frame()
    artifact, predictions = train_feature_random_forest(
        frame, scientific_status="test_only"
    )
    path = tmp_path / "model.joblib"
    save_artifact(artifact, path)
    loaded = load_artifact(path)
    result = predict_sequences(loaded, ["KWKLFKKIGAVLKVL"])
    assert len(predictions) == len(frame)
    assert result.loc[0, "score"] >= 0
    assert result.loc[0, "score"] <= 1
    assert result.loc[0, "interpretation"] in {
        "AMP-like",
        "Non-AMP-like",
        "Uncertain / borderline",
    }


def test_esm_logistic_training_uses_aligned_group_splits():
    random = np.random.default_rng(42)
    labels = np.array([0] * 60 + [1] * 60)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def encoded_sequence(index: int, label: int) -> str:
        encoded = []
        value = index
        for _ in range(6):
            encoded.append(alphabet[value % len(alphabet)])
            value //= len(alphabet)
        return ("KK" if label else "DD") + "".join(encoded) + "ACDEFG"

    frame = assign_splits(
        pd.DataFrame(
            {
                "sequence": [
                    encoded_sequence(index, int(label))
                    for index, label in enumerate(labels)
                ],
                "label": labels,
            }
        ),
        mode="random",
        random_state=12,
    )
    n = len(frame)
    aligned_labels = frame["label"].to_numpy()
    aligned_embeddings = random.normal(size=(n, 16)).astype(np.float32)
    aligned_embeddings[aligned_labels == 1, 0] += 2.5
    artifact, predictions = train_esm_logistic_regression(
        aligned_embeddings,
        aligned_labels,
        frame["split"].to_numpy(),
        frame["split_group"].to_numpy(),
        embedding_model_name="test/esm",
    )
    assert artifact["kind"] == "esm2_logistic_regression"
    assert len(predictions) == n
    assert predictions["score"].between(0, 1).all()


def test_interpretation_has_an_uncertainty_band():
    assert interpretation_label(0.50, 0.50) == "Uncertain / borderline"
    assert interpretation_label(0.90, 0.50) == "AMP-like"
    assert interpretation_label(0.10, 0.50) == "Non-AMP-like"


def test_embedding_bundle_round_trip(tmp_path: Path):
    path = tmp_path / "embeddings.npz"
    embeddings = np.arange(24, dtype=np.float32).reshape(3, 8)
    save_embedding_bundle(
        path,
        embeddings=embeddings,
        sequences=np.array(["ACDEF", "KKKKK", "DDDDD"]),
        labels=np.array([0, 1, 0]),
        splits=np.array(["train", "validation", "test"]),
        groups=np.array(["g0", "g1", "g2"]),
        model_name="test/esm",
    )
    bundle = load_embedding_bundle(path)
    assert np.array_equal(bundle["embeddings"], embeddings)
    assert bundle["model_name"] == "test/esm"
