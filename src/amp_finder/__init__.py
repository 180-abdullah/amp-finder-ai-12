"""AMP Finder AI: explainable antimicrobial-peptide screening utilities."""

from .features import FEATURE_NAMES, extract_feature_frame, extract_features
from .sequence import SequenceValidationError, normalize_sequence

__all__ = [
    "FEATURE_NAMES",
    "SequenceValidationError",
    "extract_feature_frame",
    "extract_features",
    "normalize_sequence",
]

__version__ = "1.0.0"
