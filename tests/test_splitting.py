import pandas as pd

from amp_finder.splitting import (
    assign_splits,
    greedy_similarity_groups,
    split_diagnostics,
    validate_split_integrity,
)


def encode_index(index: int, prefix: str) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    digits = []
    value = index
    for _ in range(8):
        digits.append(alphabet[value % len(alphabet)])
        value //= len(alphabet)
    return prefix + "".join(digits) + ("K" if prefix == "K" else "D")


def balanced_frame(rows_per_class: int = 60) -> pd.DataFrame:
    rows = []
    for label, prefix in [(1, "K"), (0, "D")]:
        for index in range(rows_per_class):
            rows.append({"sequence": encode_index(index, prefix), "label": label})
    return pd.DataFrame(rows)


def test_greedy_similarity_grouping_keeps_near_identical_sequences_together():
    groups = greedy_similarity_groups(
        ["AAAAAKKKKK", "AAAAAKKKKR", "DDDDDEEEEE"], threshold=0.80
    )
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_random_split_contains_both_classes_and_has_no_leakage():
    result = assign_splits(balanced_frame(), mode="random", random_state=42)
    assert set(result["split"]) == {"train", "validation", "test"}
    validate_split_integrity(result)
    diagnostics = split_diagnostics(result)
    assert diagnostics["group_leakage"] is False
    assert sum(item["rows"] for item in diagnostics["splits"].values()) == len(result)
