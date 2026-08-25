"""Frozen ESM-2 embedding extraction with lazy optional dependencies."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .sequence import normalize_sequence

DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"


def _resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def embed_sequences(
    sequences: Iterable[str],
    *,
    model_name: str = DEFAULT_ESM_MODEL,
    batch_size: int = 16,
    device: str = "auto",
    max_residues: int = 200,
) -> np.ndarray:
    """Mean-pool residue embeddings from a frozen Hugging Face ESM-2 model."""

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "ESM-2 requires the optional dependencies in requirements-esm.txt."
        ) from error

    normalized = [
        normalize_sequence(sequence, min_length=5, max_length=max_residues)
        for sequence in sequences
    ]
    if not normalized:
        return np.empty((0, 0), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    selected_device = _resolve_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(selected_device)
    model.eval()

    pooled_batches: list[np.ndarray] = []
    for start in range(0, len(normalized), batch_size):
        batch = normalized[start : start + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_residues + 2,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = tokens.pop("special_tokens_mask")
        tokens = {key: value.to(selected_device) for key, value in tokens.items()}
        special_tokens_mask = special_tokens_mask.to(selected_device)

        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
        valid_residues = tokens["attention_mask"].bool() & ~special_tokens_mask.bool()
        weights = valid_residues.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled_batches.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(pooled_batches, axis=0)


def save_embedding_bundle(
    path: str | Path,
    *,
    embeddings: np.ndarray,
    sequences: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    groups: np.ndarray,
    model_name: str,
) -> None:
    """Save aligned embeddings and provenance fields in one compressed file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        sequences=np.asarray(sequences).astype(str),
        labels=np.asarray(labels, dtype=int),
        splits=np.asarray(splits).astype(str),
        groups=np.asarray(groups).astype(str),
        model_name=np.asarray(model_name),
    )


def load_embedding_bundle(path: str | Path) -> dict[str, np.ndarray | str]:
    """Load an embedding bundle with safe, non-pickled NumPy arrays."""

    with np.load(path, allow_pickle=False) as bundle:
        return {
            "embeddings": bundle["embeddings"],
            "sequences": bundle["sequences"].astype(str),
            "labels": bundle["labels"].astype(int),
            "splits": bundle["splits"].astype(str),
            "groups": bundle["groups"].astype(str),
            "model_name": str(bundle["model_name"].item()),
        }
