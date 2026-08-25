from pathlib import Path

import pandas as pd

from amp_finder.features import extract_feature_frame
from amp_finder.modeling import load_artifact
from amp_finder.research import (
    dataset_quality_audit,
    dome_readiness_matrix,
    endpoint_evidence_matrix,
    reference_library_frame,
    sequence_research_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _status_by_domain(audit: pd.DataFrame) -> dict[str, str]:
    return audit.set_index("Domain")["Status"].to_dict()


def test_bundled_demo_dataset_passes_software_quality_checks():
    frame = pd.read_csv(PROJECT_ROOT / "data/demo/demo_sequences.csv")
    audit = dataset_quality_audit(frame)

    assert set(audit.columns) == {"Domain", "Status", "Evidence"}
    assert set(audit["Status"]) == {"Pass"}


def test_dataset_audit_detects_conflicts_duplicates_and_group_leakage():
    frame = pd.DataFrame(
        {
            "sequence": ["KLLK", "KLLK", "MDEA", "VVVV"],
            "label": [1, 0, 0, 0],
            "split": ["train", "test", "train", "test"],
            "split_group": ["shared", "shared", "acidic", "hydrophobic"],
            "source": ["source"] * 4,
            "source_id": ["a", "b", "c", "d"],
            "parent_id": ["a", "b", "c", "d"],
        }
    )

    status = _status_by_domain(dataset_quality_audit(frame))
    assert status["Exact duplicates"] == "Needs revision"
    assert status["Cross-label conflicts"] == "Critical issue"
    assert status["Group-disjoint split"] == "Critical issue"


def test_sequence_audit_keeps_chemical_and_model_domain_limits_visible():
    artifact = load_artifact(PROJECT_ROOT / "models/demo_baseline_rf.joblib")
    features = extract_feature_frame(["KWKLFKKIGAVLKVL"]).iloc[0]
    audit = sequence_research_audit(features, artifact)

    assert len(audit) == 5
    chemistry = audit.loc[audit["Check"].eq("Chemical representation")].iloc[0]
    assert chemistry["Status"] == "Sequence-only limitation"
    assert "D-residues" in chemistry["Implication"]


def test_translational_matrix_never_implies_unmeasured_activity():
    matrix = endpoint_evidence_matrix()

    assert matrix.loc[0, "Status"] == "Estimated"
    assert matrix.loc[0, "Endpoint"] == "AMP-like sequence pattern"
    assert set(matrix.loc[1:, "Status"]) == {"Not evaluated"}
    assert {"Target-species potency", "Hemolysis / cytotoxicity", "In-vivo efficacy"}.issubset(
        set(matrix["Endpoint"])
    )


def test_dome_readiness_blocks_claims_for_synthetic_data():
    artifact = load_artifact(PROJECT_ROOT / "models/demo_baseline_rf.joblib")
    readiness = dome_readiness_matrix(artifact, dataset_is_scientific=False)

    data_row = readiness.loc[readiness["DOME domain"].eq("Data")].iloc[0]
    evaluation_row = readiness.loc[readiness["DOME domain"].eq("Evaluation")].iloc[0]
    assert data_row["Readiness"] == "Blocked for claims"
    assert evaluation_row["Readiness"] == "Needs scientific run"


def test_reference_library_is_broad_and_uses_stable_links():
    references = reference_library_frame()

    assert len(references) >= 10
    assert references["url"].str.startswith("https://").all()
    assert {"Biological ML reporting", "Clinical translation", "Benchmark validity"}.issubset(
        set(references["topic"])
    )
