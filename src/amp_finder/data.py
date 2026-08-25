"""FASTA ingestion, negative-fragment generation, and labeled dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import pandas as pd
from Bio import SeqIO

from .constants import CANONICAL_AA
from .sequence import SequenceValidationError, normalize_sequence, sequence_sha256
from .splitting import assign_splits, split_diagnostics


@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read a FASTA file without changing its source identifiers."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records = [
        FastaRecord(
            record_id=str(record.id),
            description=str(record.description),
            sequence=str(record.seq).upper().replace("-", ""),
        )
        for record in SeqIO.parse(path, "fasta")
    ]
    if not records:
        raise ValueError(f"No FASTA records found in {path}.")
    return records


def clean_positive_records(
    records: list[FastaRecord], *, min_length: int, max_length: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Validate, length-filter, and exact-deduplicate positive peptides."""

    cleaned: list[dict[str, object]] = []
    seen: set[str] = set()
    report = {
        "input_records": len(records),
        "invalid_or_noncanonical": 0,
        "outside_length_range": 0,
        "exact_duplicates_removed": 0,
    }

    for record in records:
        try:
            sequence = normalize_sequence(
                record.sequence, min_length=min_length, max_length=max_length
            )
        except SequenceValidationError as error:
            if "too short" in str(error) or "too long" in str(error):
                report["outside_length_range"] += 1
            else:
                report["invalid_or_noncanonical"] += 1
            continue

        if sequence in seen:
            report["exact_duplicates_removed"] += 1
            continue
        seen.add(sequence)
        cleaned.append(
            {
                "sequence": sequence,
                "label": 1,
                "class_name": "AMP",
                "source": "APD",
                "source_id": record.record_id,
                "parent_id": record.record_id,
            }
        )
    report["retained_records"] = len(cleaned)
    return cleaned, report


def _valid_parent_sequence(raw_sequence: str) -> str:
    sequence = "".join(raw_sequence.split()).upper().replace("-", "")
    return sequence if sequence and set(sequence).issubset(CANONICAL_AA) else ""


def sample_negative_fragments(
    parent_records: list[FastaRecord],
    target_lengths: list[int],
    *,
    forbidden_sequences: set[str],
    random_state: int = 42,
) -> list[dict[str, object]]:
    """Create deterministic, length-matched putative non-AMP fragments."""

    random = Random(random_state)
    valid_parents = [
        (record.record_id, _valid_parent_sequence(record.sequence))
        for record in parent_records
    ]
    valid_parents = [(record_id, sequence) for record_id, sequence in valid_parents if sequence]
    if not valid_parents:
        raise ValueError("No canonical parent proteins are available for fragmentation.")

    fragments: list[dict[str, object]] = []
    used_sequences: set[str] = set()
    shuffled_lengths = list(target_lengths)
    random.shuffle(shuffled_lengths)

    for fragment_index, target_length in enumerate(shuffled_lengths):
        candidates = [item for item in valid_parents if len(item[1]) >= target_length]
        if not candidates:
            raise ValueError(
                f"No parent protein is long enough for a {target_length}-aa fragment."
            )

        accepted: tuple[str, str, int] | None = None
        for _ in range(250):
            parent_id, parent_sequence = random.choice(candidates)
            start = random.randint(0, len(parent_sequence) - target_length)
            fragment = parent_sequence[start : start + target_length]
            if fragment in forbidden_sequences or fragment in used_sequences:
                continue
            accepted = (parent_id, fragment, start)
            break

        if accepted is None:
            raise RuntimeError(
                "Could not generate enough unique negative fragments. Provide more "
                "parent proteins or lower the requested negative ratio."
            )

        parent_id, fragment, start = accepted
        used_sequences.add(fragment)
        fragments.append(
            {
                "sequence": fragment,
                "label": 0,
                "class_name": "Putative non-AMP",
                "source": "UniProt fragment",
                "source_id": f"{parent_id}:{start + 1}-{start + target_length}",
                "parent_id": parent_id,
                "fragment_index": fragment_index,
            }
        )
    return fragments


def build_labeled_dataset(
    positive_fasta: str | Path,
    negative_parent_fasta: str | Path,
    *,
    min_length: int = 8,
    max_length: int = 80,
    negative_ratio: float = 1.0,
    split_mode: str = "similarity",
    similarity_threshold: float = 0.80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a balanced, traceable dataset from APD and UniProt FASTA inputs."""

    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be greater than zero.")

    positive_records = read_fasta(positive_fasta)
    parent_records = read_fasta(negative_parent_fasta)
    positives, cleaning_report = clean_positive_records(
        positive_records, min_length=min_length, max_length=max_length
    )
    if len(positives) < 20:
        raise ValueError(
            "Fewer than 20 valid positive peptides remain; this is insufficient "
            "for train/validation/test demonstration splits."
        )

    number_of_negatives = int(round(len(positives) * negative_ratio))
    random = Random(random_state)
    positive_lengths = [len(row["sequence"]) for row in positives]
    target_lengths = [random.choice(positive_lengths) for _ in range(number_of_negatives)]
    forbidden = {str(row["sequence"]) for row in positives}
    negatives = sample_negative_fragments(
        parent_records,
        target_lengths,
        forbidden_sequences=forbidden,
        random_state=random_state,
    )

    dataset = pd.DataFrame(positives + negatives)
    dataset["sequence_sha256"] = dataset["sequence"].map(sequence_sha256)
    dataset["length"] = dataset["sequence"].str.len()
    dataset = dataset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    dataset = assign_splits(
        dataset,
        mode=split_mode,
        similarity_threshold=similarity_threshold,
        random_state=random_state,
    )

    metadata: dict[str, object] = {
        "random_state": random_state,
        "min_length": min_length,
        "max_length": max_length,
        "negative_ratio": negative_ratio,
        "negative_label_interpretation": (
            "Putative non-AMP fragments; absence of an AMP annotation is not "
            "experimental proof of no antimicrobial activity."
        ),
        "split_mode": split_mode,
        "similarity_threshold": similarity_threshold,
        "positive_cleaning": cleaning_report,
        "negative_parent_records": len(parent_records),
        "diagnostics": split_diagnostics(dataset),
    }
    return dataset, metadata
