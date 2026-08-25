"""Explainable physicochemical features for peptide sequences."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .constants import AMINO_ACIDS, EISENBERG_HYDROPHOBICITY
from .sequence import normalize_sequence

CORE_FEATURE_NAMES = [
    "length",
    "net_charge_pH7",
    "charge_density",
    "gravy",
    "hydrophobic_moment",
    "aromaticity",
    "isoelectric_point",
    "molecular_weight",
    "instability_index",
    "shannon_entropy",
    "fraction_hydrophobic",
    "fraction_basic",
    "fraction_acidic",
    "fraction_polar",
]
COMPOSITION_FEATURE_NAMES = [f"aa_{aa}" for aa in AMINO_ACIDS]
FEATURE_NAMES = CORE_FEATURE_NAMES + COMPOSITION_FEATURE_NAMES


def alpha_helical_hydrophobic_moment(sequence: str, angle_degrees: float = 100.0) -> float:
    """Calculate a normalized Eisenberg hydrophobic moment.

    The 100-degree residue angle is an alpha-helix assumption. Therefore this is
    an amphipathicity descriptor, not evidence that the peptide forms a helix.
    """

    angle = math.radians(angle_degrees)
    x_component = 0.0
    y_component = 0.0
    for position, residue in enumerate(sequence):
        hydrophobicity = EISENBERG_HYDROPHOBICITY[residue]
        x_component += hydrophobicity * math.cos(position * angle)
        y_component += hydrophobicity * math.sin(position * angle)
    return math.sqrt(x_component**2 + y_component**2) / len(sequence)


def shannon_entropy(sequence: str) -> float:
    """Return amino-acid composition entropy in bits."""

    counts = np.array([sequence.count(aa) for aa in AMINO_ACIDS], dtype=float)
    probabilities = counts[counts > 0] / len(sequence)
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_features(sequence: str) -> dict[str, float]:
    """Extract deterministic sequence and physicochemical features."""

    sequence = normalize_sequence(sequence)
    analysis = ProteinAnalysis(sequence)
    length = len(sequence)
    # Compute fractions directly. Biopython changed its percent API from
    # fractions to 0–100 percentages, so this avoids version-dependent units.
    composition = {aa: sequence.count(aa) / length for aa in AMINO_ACIDS}
    net_charge = float(analysis.charge_at_pH(7.0))

    features: dict[str, float] = {
        "length": float(length),
        "net_charge_pH7": net_charge,
        "charge_density": net_charge / length,
        "gravy": float(analysis.gravy()),
        "hydrophobic_moment": alpha_helical_hydrophobic_moment(sequence),
        "aromaticity": float(analysis.aromaticity()),
        "isoelectric_point": float(analysis.isoelectric_point()),
        "molecular_weight": float(analysis.molecular_weight()),
        "instability_index": float(analysis.instability_index()),
        "shannon_entropy": shannon_entropy(sequence),
        "fraction_hydrophobic": sum(composition[aa] for aa in "AILMFWVY"),
        "fraction_basic": sum(composition[aa] for aa in "KRH"),
        "fraction_acidic": sum(composition[aa] for aa in "DE"),
        "fraction_polar": sum(composition[aa] for aa in "STNQ"),
    }
    features.update({f"aa_{aa}": float(composition[aa]) for aa in AMINO_ACIDS})
    return features


def extract_feature_frame(sequences: Iterable[str]) -> pd.DataFrame:
    """Return one ordered feature row per sequence."""

    rows = [extract_features(sequence) for sequence in sequences]
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float)
