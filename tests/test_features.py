import numpy as np

from amp_finder.constants import AMINO_ACIDS
from amp_finder.features import FEATURE_NAMES, extract_feature_frame, extract_features


def test_feature_schema_and_finite_values():
    features = extract_features("KWKLFKKIGAVLKVL")
    assert list(features) == FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()
    assert features["length"] == 15


def test_amino_acid_composition_sums_to_one():
    features = extract_features("ACDEFGHIKLMNPQRSTVWY")
    composition_sum = sum(features[f"aa_{aa}"] for aa in AMINO_ACIDS)
    assert np.isclose(composition_sum, 1.0)


def test_charge_direction_is_biologically_sensible():
    assert extract_features("KKKKK")["net_charge_pH7"] > 0
    assert extract_features("DDDDD")["net_charge_pH7"] < 0


def test_feature_frame_preserves_row_count_and_order():
    frame = extract_feature_frame(["KKKKK", "DDDDD"])
    assert frame.shape == (2, len(FEATURE_NAMES))
    assert frame.iloc[0]["net_charge_pH7"] > frame.iloc[1]["net_charge_pH7"]
