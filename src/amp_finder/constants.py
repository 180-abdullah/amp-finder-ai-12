"""Shared biological constants and display labels."""

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_AA = frozenset(AMINO_ACIDS)

# Eisenberg consensus hydrophobicity scale, used only for the alpha-helical
# hydrophobic moment. GRAVY is calculated separately with Kyte-Doolittle.
EISENBERG_HYDROPHOBICITY = {
    "A": 0.25,
    "R": -1.76,
    "N": -0.64,
    "D": -0.72,
    "C": 0.04,
    "Q": -0.69,
    "E": -0.62,
    "G": 0.16,
    "H": -0.40,
    "I": 0.73,
    "L": 0.53,
    "K": -1.10,
    "M": 0.26,
    "F": 0.61,
    "P": -0.07,
    "S": -0.26,
    "T": -0.18,
    "W": 0.37,
    "Y": 0.02,
    "V": 0.54,
}

FEATURE_LABELS = {
    "length": "Length (aa)",
    "net_charge_pH7": "Net charge at pH 7",
    "charge_density": "Charge density",
    "gravy": "Mean hydropathy (GRAVY)",
    "hydrophobic_moment": "Alpha-helical hydrophobic moment",
    "aromaticity": "Aromaticity",
    "isoelectric_point": "Isoelectric point",
    "molecular_weight": "Molecular weight (Da)",
    "instability_index": "Instability index",
    "shannon_entropy": "Sequence entropy",
    "fraction_hydrophobic": "Hydrophobic-residue fraction",
    "fraction_basic": "Basic-residue fraction",
    "fraction_acidic": "Acidic-residue fraction",
    "fraction_polar": "Polar-residue fraction",
}
