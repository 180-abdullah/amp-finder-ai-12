import random
from pathlib import Path

from amp_finder.data import build_labeled_dataset


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f">{record_id}\n{sequence}\n" for record_id, sequence in records),
        encoding="utf-8",
    )


def test_full_fasta_to_split_dataset_pipeline(tmp_path: Path):
    generator = random.Random(123)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    positives: set[str] = set()
    while len(positives) < 30:
        length = generator.randint(10, 20)
        positives.add("".join(generator.choices(alphabet, k=length)))
    parents = [
        (
            f"parent_{index}",
            "".join(generator.choices(alphabet, k=180)),
        )
        for index in range(12)
    ]
    positive_path = tmp_path / "positive.fasta"
    parent_path = tmp_path / "parents.fasta"
    write_fasta(
        positive_path,
        [(f"amp_{index}", sequence) for index, sequence in enumerate(sorted(positives))],
    )
    write_fasta(parent_path, parents)

    dataset, metadata = build_labeled_dataset(
        positive_path,
        parent_path,
        min_length=8,
        max_length=30,
        negative_ratio=1.0,
        split_mode="similarity",
        similarity_threshold=0.95,
        random_state=42,
    )
    assert len(dataset) == 60
    assert dataset["sequence"].nunique() == 60
    assert dataset["label"].value_counts().to_dict() == {1: 30, 0: 30}
    assert set(dataset["split"]) == {"train", "validation", "test"}
    assert metadata["diagnostics"]["group_leakage"] is False
    assert "Putative non-AMP" in metadata["negative_label_interpretation"]
