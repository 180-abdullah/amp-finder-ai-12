"""Sequence input validation and lightweight sequence utilities."""

from __future__ import annotations

import hashlib
import re

from .constants import CANONICAL_AA


class SequenceValidationError(ValueError):
    """Raised when a peptide cannot be represented by canonical amino acids."""


def normalize_sequence(
    raw_sequence: str,
    *,
    min_length: int = 5,
    max_length: int = 200,
) -> str:
    """Return a canonical uppercase peptide sequence.

    Whitespace and alignment hyphens are removed. A single FASTA header is
    accepted for convenience. Ambiguous/non-canonical residues are rejected
    rather than silently guessed.
    """

    if raw_sequence is None:
        raise SequenceValidationError("Sequence is missing.")

    text = str(raw_sequence).strip()
    if not text:
        raise SequenceValidationError("Sequence is empty.")

    if text.startswith(">"):
        lines = text.splitlines()
        text = "".join(lines[1:])

    sequence = re.sub(r"[\s-]+", "", text).upper()
    if not sequence:
        raise SequenceValidationError("No amino-acid residues were found.")

    invalid = sorted(set(sequence) - CANONICAL_AA)
    if invalid:
        invalid_text = ", ".join(invalid)
        raise SequenceValidationError(
            "Only the 20 canonical one-letter amino-acid codes are accepted. "
            f"Invalid symbol(s): {invalid_text}."
        )

    if len(sequence) < min_length:
        raise SequenceValidationError(
            f"Sequence is too short ({len(sequence)} aa); minimum is {min_length}."
        )
    if len(sequence) > max_length:
        raise SequenceValidationError(
            f"Sequence is too long ({len(sequence)} aa); maximum is {max_length}."
        )
    return sequence


def sequence_sha256(sequence: str) -> str:
    """Return a stable, non-reversible identifier for a normalized sequence."""

    return hashlib.sha256(sequence.encode("ascii")).hexdigest()
