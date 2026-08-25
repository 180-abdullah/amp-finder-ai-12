"""Research-facing audits, evidence maps, and scholar reference metadata.

The functions in this module deliberately separate what the application has
measured from what still requires another computational model or a wet-lab
assay. They are suitable for the Streamlit research interface and unit tests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import CANONICAL_AA


REFERENCE_LIBRARY: tuple[dict[str, str], ...] = (
    {
        "topic": "Global health context",
        "resource": "WHO antimicrobial resistance fact sheet",
        "evidence_type": "Authoritative public-health source",
        "year": "2026",
        "url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        "use": "Current AMR burden, resistance prevalence, and R&D context.",
    },
    {
        "topic": "Global health context",
        "resource": "WHO bacterial priority pathogens list",
        "evidence_type": "Authoritative priority-setting report",
        "year": "2024",
        "url": "https://www.who.int/publications/i/item/9789240093461",
        "use": "Frames the bacterial targets for which new interventions are most urgent.",
    },
    {
        "topic": "Positive data",
        "resource": "Antimicrobial Peptide Database",
        "evidence_type": "Curated biological database",
        "year": "Current release",
        "url": "https://aps.unmc.edu/downloads",
        "use": "Source of curated AMP sequences; release and download date must be recorded.",
    },
    {
        "topic": "Activity and safety data",
        "resource": "DBAASP",
        "evidence_type": "Manually curated experimental database",
        "year": "Current release",
        "url": "https://dbaasp.org/home",
        "use": "Target-specific activity, assay conditions, structure, hemolysis, and cytotoxicity.",
    },
    {
        "topic": "Putative-negative data",
        "resource": "UniProt website API",
        "evidence_type": "Primary database/API publication",
        "year": "2025",
        "url": "https://doi.org/10.1093/nar/gkaf394",
        "use": "Reproducible retrieval of reviewed protein records and provenance metadata.",
    },
    {
        "topic": "Protein language models",
        "resource": "ESM-2",
        "evidence_type": "Primary methods paper",
        "year": "2023",
        "url": "https://doi.org/10.1126/science.ade2574",
        "use": "Frozen contextual sequence representations for the modern comparison model.",
    },
    {
        "topic": "Benchmark validity",
        "resource": "Negative-data bias in AMP prediction",
        "evidence_type": "Primary benchmarking study",
        "year": "2022",
        "url": "https://doi.org/10.1093/bib/bbac343",
        "use": "Shows why negative-set construction can materially alter apparent performance.",
    },
    {
        "topic": "Biological ML reporting",
        "resource": "DOME recommendations",
        "evidence_type": "Community reporting recommendations",
        "year": "2021",
        "url": "https://doi.org/10.1038/s41592-021-01205-4",
        "use": "Data, Optimization, Model, and Evaluation transparency for biological ML.",
    },
    {
        "topic": "Clinical translation",
        "resource": "TRIPOD+AI",
        "evidence_type": "Clinical prediction reporting guideline",
        "year": "2024",
        "url": "https://doi.org/10.1136/bmj-2023-078378",
        "use": "Relevant only if the work becomes an individual-level clinical prediction study.",
    },
    {
        "topic": "Clinical translation",
        "resource": "PROBAST+AI",
        "evidence_type": "Clinical prediction risk-of-bias tool",
        "year": "2025",
        "url": "https://doi.org/10.1136/bmj-2024-082505",
        "use": "Relevant to quality and applicability assessment in a clinical prediction context.",
    },
)


def endpoint_evidence_matrix() -> pd.DataFrame:
    """Return the translational evidence ladder for one sequence prediction."""

    return pd.DataFrame(
        [
            {
                "Endpoint": "AMP-like sequence pattern",
                "Status": "Estimated",
                "Required evidence": "Validated binary classifier",
                "Interpretation": "Prioritization signal only",
            },
            {
                "Endpoint": "Target-species potency",
                "Status": "Not evaluated",
                "Required evidence": "MIC/MBC under specified assay conditions",
                "Interpretation": "Cannot infer susceptible organism or dose",
            },
            {
                "Endpoint": "Antimicrobial spectrum",
                "Status": "Not evaluated",
                "Required evidence": "Panel of target strains",
                "Interpretation": "Binary AMP labels do not define spectrum",
            },
            {
                "Endpoint": "Hemolysis / cytotoxicity",
                "Status": "Not evaluated",
                "Required evidence": "HC50/CC50 and relevant cell assays",
                "Interpretation": "Activity without selectivity is not a therapeutic result",
            },
            {
                "Endpoint": "Stability and exposure",
                "Status": "Not evaluated",
                "Required evidence": "Protease, serum, solubility, and formulation studies",
                "Interpretation": "Sequence score does not establish usable exposure",
            },
            {
                "Endpoint": "In-vivo efficacy",
                "Status": "Not evaluated",
                "Required evidence": "Ethically approved disease model",
                "Interpretation": "Requires an independent translational program",
            },
        ]
    )


def analysis_plan_matrix() -> pd.DataFrame:
    """Return a compact pre-specified analysis plan for the project."""

    return pd.DataFrame(
        [
            {
                "Objective": "Describe biological differences",
                "Outcome": "Feature distributions",
                "Method": "Effect plots and distribution summaries",
                "Diagnostic": "Class balance, outliers, source stratification",
                "Robustness": "Repeat by source and length band",
            },
            {
                "Objective": "Estimate binary discrimination",
                "Outcome": "AMP vs putative non-AMP label",
                "Method": "Random Forest and ESM-2 + Logistic Regression",
                "Diagnostic": "ROC/PR curves, confusion matrix, calibration",
                "Robustness": "Similarity-aware versus random split",
            },
            {
                "Objective": "Choose a screening threshold",
                "Outcome": "Validation MCC",
                "Method": "Threshold search on validation data only",
                "Diagnostic": "Sensitivity-specificity trade-off",
                "Robustness": "Report results across plausible thresholds",
            },
            {
                "Objective": "Assess generalization risk",
                "Outcome": "Held-out performance",
                "Method": "Group-disjoint test evaluation",
                "Diagnostic": "Similarity leakage and subgroup coverage",
                "Robustness": "External database/release validation",
            },
        ]
    )


def sequence_research_audit(
    feature_row: pd.Series | dict[str, float], artifact: dict[str, Any]
) -> pd.DataFrame:
    """Describe model-domain and sequence-quality checks without inventing assays."""

    features = dict(feature_row)
    length = int(round(float(features["length"])))
    entropy = float(features["shannon_entropy"])
    hydrophobic_fraction = float(features["fraction_hydrophobic"])
    charge_density = float(features["charge_density"])
    training_min = int(artifact.get("sequence_length_min", 0))
    training_max = int(artifact.get("sequence_length_max", 10**9))

    rows: list[dict[str, str]] = []
    in_length_domain = training_min <= length <= training_max
    rows.append(
        {
            "Check": "Length domain",
            "Status": "Within model range" if in_length_domain else "Outside model range",
            "Evidence": f"{length} aa; model range {training_min}–{training_max} aa",
            "Implication": (
                "No length extrapolation flag"
                if in_length_domain
                else "Treat score as out-of-domain and do not rank with in-domain peptides"
            ),
        }
    )
    rows.append(
        {
            "Check": "Sequence complexity",
            "Status": "Low-complexity flag" if entropy < 2.5 else "No strong flag",
            "Evidence": f"Shannon entropy {entropy:.2f} bits",
            "Implication": (
                "Composition-driven predictions may be unstable"
                if entropy < 2.5
                else "Residue diversity is not unusually low by this teaching rule"
            ),
        }
    )
    rows.append(
        {
            "Check": "Hydrophobic burden",
            "Status": "High-hydrophobicity flag" if hydrophobic_fraction > 0.65 else "No strong flag",
            "Evidence": f"Hydrophobic residue fraction {hydrophobic_fraction:.2f}",
            "Implication": (
                "Aggregation, solubility, and host-cell toxicity require attention"
                if hydrophobic_fraction > 0.65
                else "No extreme composition flag; experimental safety remains unknown"
            ),
        }
    )
    rows.append(
        {
            "Check": "Charge density",
            "Status": "Extreme-charge flag" if abs(charge_density) > 0.40 else "No strong flag",
            "Evidence": f"Charge per residue {charge_density:.3f}",
            "Implication": (
                "May be outside common training behavior; inspect nearest sequences"
                if abs(charge_density) > 0.40
                else "Charge is not extreme by this teaching rule"
            ),
        }
    )
    rows.append(
        {
            "Check": "Chemical representation",
            "Status": "Sequence-only limitation",
            "Evidence": "Canonical L-amino-acid one-letter sequence",
            "Implication": "Termini, D-residues, cyclization, disulfide connectivity, and other modifications are absent",
        }
    )
    return pd.DataFrame(rows)


def dataset_quality_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit the minimum trust conditions for a labeled peptide dataset."""

    if "sequence" not in frame.columns or "label" not in frame.columns:
        raise ValueError("Dataset must contain sequence and label columns.")

    normalized = frame["sequence"].astype(str).str.upper().str.replace(r"\s+|-", "", regex=True)
    canonical = normalized.map(lambda value: bool(value) and set(value).issubset(CANONICAL_AA))
    duplicates = int(normalized.duplicated().sum())
    conflicts = int(
        pd.DataFrame({"sequence": normalized, "label": frame["label"]})
        .groupby("sequence")["label"]
        .nunique()
        .gt(1)
        .sum()
    )
    positive_rate = float(pd.to_numeric(frame["label"], errors="coerce").mean())

    if {"split", "split_group"}.issubset(frame.columns):
        group_split_counts = frame.groupby("split_group")["split"].nunique()
        leaking_groups = int((group_split_counts > 1).sum())
        split_evidence = f"{leaking_groups} group(s) occur in more than one split"
        split_status = "Pass" if leaking_groups == 0 else "Critical issue"
    else:
        split_evidence = "split and/or split_group column unavailable"
        split_status = "Not assessable"

    provenance_columns = {"source", "source_id", "parent_id"}
    provenance_present = provenance_columns.issubset(frame.columns)

    return pd.DataFrame(
        [
            {
                "Domain": "Canonical sequences",
                "Status": "Pass" if canonical.all() else "Critical issue",
                "Evidence": f"{int(canonical.sum())}/{len(frame)} rows use only 20 canonical residues",
            },
            {
                "Domain": "Exact duplicates",
                "Status": "Pass" if duplicates == 0 else "Needs revision",
                "Evidence": f"{duplicates} repeated sequence row(s)",
            },
            {
                "Domain": "Cross-label conflicts",
                "Status": "Pass" if conflicts == 0 else "Critical issue",
                "Evidence": f"{conflicts} sequence(s) carry both labels",
            },
            {
                "Domain": "Class balance",
                "Status": "Pass" if 0.25 <= positive_rate <= 0.75 else "Needs revision",
                "Evidence": f"Positive-class prevalence {positive_rate:.1%}",
            },
            {
                "Domain": "Group-disjoint split",
                "Status": split_status,
                "Evidence": split_evidence,
            },
            {
                "Domain": "Record provenance",
                "Status": "Pass" if provenance_present else "Needs revision",
                "Evidence": (
                    "source, source_id, and parent_id columns present"
                    if provenance_present
                    else "one or more provenance columns missing"
                ),
            },
        ]
    )


def dome_readiness_matrix(
    artifact: dict[str, Any], *, dataset_is_scientific: bool
) -> pd.DataFrame:
    """Return a non-numeric DOME-aligned readiness assessment."""

    metadata = artifact.get("metadata", {})
    diagnostics = metadata.get("split_diagnostics", {})
    group_leakage = diagnostics.get("group_leakage")
    has_test_metrics = bool(metadata.get("test_metrics"))
    status = metadata.get("scientific_status", "unknown")

    return pd.DataFrame(
        [
            {
                "DOME domain": "Data",
                "Readiness": "Ready" if dataset_is_scientific else "Blocked for claims",
                "What is present": "Provenance-aware APD/UniProt pipeline",
                "Remaining risk": (
                    "Verify releases, exclusions, duplicates, and assay heterogeneity"
                    if dataset_is_scientific
                    else "Bundled data are synthetic and cannot support biological findings"
                ),
            },
            {
                "DOME domain": "Optimization",
                "Readiness": "Ready",
                "What is present": "Validation-only threshold selection",
                "Remaining risk": "Document every tuned hyperparameter and search range",
            },
            {
                "DOME domain": "Model",
                "Readiness": "Ready" if status != "unknown" else "Needs revision",
                "What is present": "Versioned feature model and optional frozen ESM-2 encoder",
                "Remaining risk": "Probability calibration and subgroup behavior need explicit review",
            },
            {
                "DOME domain": "Evaluation",
                "Readiness": (
                    "Ready for internal evaluation"
                    if has_test_metrics and group_leakage is False and dataset_is_scientific
                    else "Needs scientific run"
                ),
                "What is present": "Held-out metrics, MCC threshold, split diagnostics",
                "Remaining risk": "External-release, target-specific, potency, and safety validation are absent",
            },
        ]
    )


def reference_library_frame() -> pd.DataFrame:
    """Return a fresh table so callers cannot mutate the module constant."""

    return pd.DataFrame(REFERENCE_LIBRARY).copy()
