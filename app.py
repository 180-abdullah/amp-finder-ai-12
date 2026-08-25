"""Research-facing Streamlit interface for AMP Finder AI."""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_finder.constants import AMINO_ACIDS, FEATURE_LABELS  # noqa: E402
from amp_finder.features import extract_feature_frame  # noqa: E402
from amp_finder.inference import biological_context, predict_sequences  # noqa: E402
from amp_finder.modeling import load_artifact  # noqa: E402
from amp_finder.research import (  # noqa: E402
    analysis_plan_matrix,
    dataset_quality_audit,
    dome_readiness_matrix,
    endpoint_evidence_matrix,
    reference_library_frame,
    sequence_research_audit,
)
from amp_finder.sequence import SequenceValidationError, normalize_sequence  # noqa: E402


st.set_page_config(
    page_title="AMP Finder AI — Research Edition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = PROJECT_ROOT / "assets/research.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


MODEL_PATHS = [
    PROJECT_ROOT / "models/baseline_rf.joblib",
    PROJECT_ROOT / "models/esm2_logreg.joblib",
    PROJECT_ROOT / "models/demo_baseline_rf.joblib",
]
DATASET_PATHS = [
    (PROJECT_ROOT / "data/processed/amp_dataset.csv", "Research dataset", True),
    (PROJECT_ROOT / "data/demo/demo_sequences.csv", "Synthetic teaching dataset", False),
]

COLORS = {
    "background": "#07110f",
    "panel": "#10231f",
    "text": "#effff8",
    "muted": "#a8c4ba",
    "grid": "rgba(201, 255, 234, 0.10)",
    "amp": "#46e6c2",
    "non_amp": "#ffbd72",
    "accent": "#c6ff4a",
    "danger": "#ff7e79",
}


@st.cache_resource(show_spinner=False)
def cached_load_artifact(path: str) -> dict[str, Any]:
    return load_artifact(path)


@st.cache_data(show_spinner=False)
def cached_load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_feature_frame(sequences: tuple[str, ...]) -> pd.DataFrame:
    return extract_feature_frame(sequences)


def discover_models() -> dict[str, tuple[Path, dict[str, Any]]]:
    models: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in MODEL_PATHS:
        if not path.exists():
            continue
        artifact = cached_load_artifact(str(path))
        status = artifact.get("metadata", {}).get("scientific_status", "unknown")
        if status == "synthetic_ui_demo_only":
            label = "Toy UI model — synthetic data"
        else:
            label = artifact.get("model_name", path.stem)
        models[label] = (path, artifact)
    return models


def discover_dataset() -> tuple[Path | None, pd.DataFrame, str, bool]:
    for path, label, is_scientific in DATASET_PATHS:
        if path.exists():
            return path, cached_load_dataset(str(path)), label, is_scientific
    return None, pd.DataFrame(), "No dataset found", False


def style_figure(figure: go.Figure, *, height: int, legend: bool = True) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=64, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "family": "Inter, Arial, sans-serif"},
        legend={
            "orientation": "h",
            "y": -0.18,
            "font": {"color": COLORS["muted"]},
        },
        showlegend=legend,
        hoverlabel={"bgcolor": COLORS["panel"], "font_color": COLORS["text"]},
    )
    figure.update_xaxes(
        gridcolor=COLORS["grid"],
        zerolinecolor="rgba(239,255,248,0.25)",
        linecolor=COLORS["grid"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    figure.update_yaxes(
        gridcolor=COLORS["grid"],
        zerolinecolor="rgba(239,255,248,0.25)",
        linecolor=COLORS["grid"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    return figure


def score_gauge(score: float, threshold: float) -> go.Figure:
    lower = max(0.0, threshold - 0.10) * 100
    upper = min(1.0, threshold + 0.10) * 100
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 45, "color": COLORS["text"]}},
            title={"text": "AMP-likeness score", "font": {"size": 17, "color": COLORS["muted"]}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": "%",
                    "tickcolor": COLORS["muted"],
                    "tickfont": {"color": COLORS["muted"]},
                },
                "bar": {"color": COLORS["accent"], "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": COLORS["grid"],
                "steps": [
                    {"range": [0, lower], "color": "rgba(255,189,114,0.11)"},
                    {"range": [lower, upper], "color": "rgba(255,189,114,0.28)"},
                    {"range": [upper, 100], "color": "rgba(70,230,194,0.14)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["danger"], "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    return style_figure(figure, height=300, legend=False)


def normalized_feature_figure(feature_row: pd.Series, artifact: dict[str, Any]) -> go.Figure | None:
    summary = artifact.get("feature_summary", {})
    selected = [
        "net_charge_pH7",
        "charge_density",
        "gravy",
        "hydrophobic_moment",
        "fraction_hydrophobic",
    ]
    if not all(feature in summary for feature in selected):
        return None

    def scale(value: float, feature: str) -> float:
        lower = float(summary[feature]["overall_q05"])
        upper = float(summary[feature]["overall_q95"])
        if upper <= lower:
            return 0.5
        return max(0.0, min(1.0, (value - lower) / (upper - lower)))

    labels = [FEATURE_LABELS[feature] for feature in selected]
    peptide_values = [scale(float(feature_row[feature]), feature) for feature in selected]
    amp_medians = [scale(float(summary[feature]["amp_median"]), feature) for feature in selected]
    non_amp_medians = [scale(float(summary[feature]["non_amp_median"]), feature) for feature in selected]
    figure = go.Figure()
    figure.add_bar(
        y=labels,
        x=peptide_values,
        name="Input peptide",
        orientation="h",
        marker={"color": COLORS["accent"], "line": {"color": COLORS["text"], "width": 0.4}},
    )
    figure.add_scatter(
        y=labels,
        x=amp_medians,
        name="Training AMP median",
        mode="markers",
        marker={"symbol": "diamond", "size": 11, "color": COLORS["amp"]},
    )
    figure.add_scatter(
        y=labels,
        x=non_amp_medians,
        name="Training non-AMP median",
        mode="markers",
        marker={"symbol": "circle-open", "size": 11, "color": COLORS["non_amp"]},
    )
    figure.update_layout(
        title="Feature position within the training range",
        barmode="overlay",
    )
    figure.update_xaxes(
        title="Normalized from training 5th to 95th percentile",
        range=[0, 1],
        tickvals=[0, 0.5, 1],
        ticktext=["Low", "Middle", "High"],
    )
    figure.update_yaxes(autorange="reversed")
    return style_figure(figure, height=410)


def render_hero() -> None:
    st.markdown(
        """
        <div class="research-hero">
          <div class="hero-copy">
            <div class="research-kicker">AMP Finder AI · Research Edition</div>
            <h1>Map peptide sequence <span>to biological evidence.</span></h1>
            <p>
              A transparent antimicrobial-peptide research workspace combining interpretable
              descriptors, protein language models, provenance-aware data, and explicit
              translational limits.
            </p>
            <div class="hero-badges">
              <span class="hero-badge">Explainable ML</span>
              <span class="hero-badge">ESM-2 comparison</span>
              <span class="hero-badge">Similarity-aware evaluation</span>
              <span class="hero-badge">DOME-aligned reporting</span>
            </div>
          </div>
          <div class="hero-visual" aria-hidden="true">
            <div class="amp-orbit orbit-a"></div>
            <div class="amp-orbit orbit-b"></div>
            <div class="amp-orbit orbit-c"></div>
            <div class="peptide-core">
              <strong>KWK · LFK · KIG</strong>
              <small>sequence → evidence</small>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_notice(artifact: dict[str, Any]) -> None:
    status = artifact.get("metadata", {}).get("scientific_status", "unknown")
    if status == "synthetic_ui_demo_only":
        st.markdown(
            """
            <div class="toy-banner"><strong>Toy mode is active.</strong> The bundled model uses
            synthetic sequences only to verify the software and interface. Its scores and apparent
            metrics are not biological findings. Train the real-data pipeline before scholarly use.</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="science-banner"><strong>Screening mode.</strong> The model ranks AMP-like
            sequence patterns. It does not establish potency, target spectrum, toxicity, stability,
            novelty, or clinical usefulness.</div>
            """,
            unsafe_allow_html=True,
        )


def render_overview() -> None:
    st.markdown('<div class="section-kicker">Why this problem matters</div>', unsafe_allow_html=True)
    st.markdown("## Antimicrobial resistance is a discovery and evidence problem")
    st.markdown(
        '<p class="section-lead">Sequence models can reduce a very large candidate space, but the scientific value comes from connecting prioritization to transparent data, uncertainty, and experimental follow-up.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="impact-grid">
          <div class="impact-card">
            <span class="impact-number">4.7M+</span>
            <span class="impact-label">deaths associated with bacterial AMR globally in 2021</span>
            <span class="impact-source">WHO fact sheet, updated 16 July 2026</span>
          </div>
          <div class="impact-card">
            <span class="impact-number">1 in 6</span>
            <span class="impact-label">laboratory-confirmed bacterial infections resistant to antibiotics in 2023</span>
            <span class="impact-source">WHO global surveillance summary</span>
          </div>
          <div class="impact-card">
            <span class="impact-number">24</span>
            <span class="impact-label">priority pathogens across 15 families in the WHO 2024 list</span>
            <span class="impact-source">WHO Bacterial Priority Pathogens List 2024</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Sources: [WHO antimicrobial resistance fact sheet](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance) · "
        "[WHO bacterial priority pathogens list, 2024](https://www.who.int/publications/i/item/9789240093461)"
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### The focused research question")
        st.markdown(
            """
            <div class="research-card">
              <strong>Can sequence-only models distinguish curated antimicrobial peptides from
              length-matched putative non-AMPs under similarity-aware evaluation?</strong>
              <p>This is a binary recognition problem—not a potency, toxicity, or therapeutic-efficacy study.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### The decision this platform supports")
        st.markdown(
            """
            <div class="research-card">
              <strong>Which sequences deserve more expensive investigation?</strong>
              <p>Use the score, descriptor profile, quality flags, provenance, and uncertainty to prioritize candidates for database checks and controlled assays.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker">Evidence ladder</div>', unsafe_allow_html=True)
    st.markdown("## A prediction is the beginning—not the conclusion")
    st.markdown(
        """
        <div class="evidence-ladder">
          <div class="evidence-step"><span class="step-index">01</span><strong>Sequence screen</strong><span>AMP-likeness, domain checks, and model uncertainty.</span></div>
          <div class="evidence-step"><span class="step-index">02</span><strong>In-silico triage</strong><span>Novelty, structure, solubility, toxicity, and target-aware models.</span></div>
          <div class="evidence-step"><span class="step-index">03</span><strong>Wet-lab evidence</strong><span>MIC/MBC, time-kill, hemolysis, cytotoxicity, and stability assays.</span></div>
          <div class="evidence-step"><span class="step-index">04</span><strong>Translation</strong><span>Selectivity, formulation, exposure, in-vivo efficacy, and reproducibility.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Research modules</div>', unsafe_allow_html=True)
    columns = st.columns(3, gap="large")
    modules = [
        (
            "Predictor Lab",
            "Screen one peptide or a CSV batch, inspect descriptors, quality flags, uncertainty, and export an auditable record.",
        ),
        (
            "Data Observatory",
            "Interrogate class balance, sequence validity, split leakage, length, charge, hydropathy, and composition.",
        ),
        (
            "Methods & Scholar Library",
            "Review the analysis plan, DOME readiness, model card, translational gaps, databases, standards, and primary literature.",
        ),
    ]
    for column, (title, body) in zip(columns, modules):
        with column:
            st.markdown(
                f'<div class="research-card"><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )


def research_record(
    result: pd.Series,
    artifact: dict[str, Any],
    audit: pd.DataFrame,
) -> dict[str, Any]:
    selected_features = [
        "length",
        "net_charge_pH7",
        "charge_density",
        "gravy",
        "hydrophobic_moment",
        "aromaticity",
        "isoelectric_point",
        "molecular_weight",
        "shannon_entropy",
    ]
    return {
        "record_type": "AMP Finder AI screening record",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence": str(result["sequence"]),
        "model_name": artifact.get("model_name", "unknown"),
        "model_kind": artifact.get("kind", "unknown"),
        "scientific_status": artifact.get("metadata", {}).get("scientific_status", "unknown"),
        "amp_likeness_score": float(result["score"]),
        "decision_threshold": float(result["threshold"]),
        "interpretation": str(result["interpretation"]),
        "features": {name: float(result[name]) for name in selected_features if name in result.index},
        "sequence_audit": audit.to_dict(orient="records"),
        "unmeasured_endpoints": [
            "target-species potency",
            "spectrum",
            "hemolysis",
            "cytotoxicity",
            "stability",
            "in-vivo efficacy",
        ],
        "mandatory_caveat": (
            "This is a sequence-pattern screening result, not experimental evidence or a clinical probability."
        ),
    }


def render_single_prediction(artifact: dict[str, Any], raw_sequence: str) -> None:
    try:
        normalized = normalize_sequence(raw_sequence)
        with st.spinner("Extracting biological descriptors and running the model…"):
            result = predict_sequences(artifact, [normalized]).iloc[0]
    except (SequenceValidationError, ValueError, ImportError, OSError) as error:
        st.error(str(error))
        return

    score = float(result["score"])
    threshold = float(result["threshold"])
    interpretation = str(result["interpretation"])
    audit = sequence_research_audit(result, artifact)

    st.markdown(
        f"""
        <div class="result-shell">
          <div class="result-label">Screening interpretation</div>
          <div class="result-value">{interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.plotly_chart(score_gauge(score, threshold), use_container_width=True)
        st.caption(
            f"Decision threshold {threshold:.3f}; the ±0.10 zone is intentionally reported as uncertain."
        )
    with right:
        metric_columns = st.columns(2)
        metric_columns[0].metric("Length", f"{int(result['length'])} aa")
        metric_columns[1].metric("Net charge at pH 7", f"{float(result['net_charge_pH7']):+.2f}")
        metric_columns[0].metric("Mean hydropathy", f"{float(result['gravy']):.2f}")
        metric_columns[1].metric("Hydrophobic moment", f"{float(result['hydrophobic_moment']):.3f}")
        st.markdown("#### Biological context")
        for statement in biological_context(result, artifact)[:4]:
            st.markdown(f"- {statement}")

    if artifact.get("kind") == "feature_random_forest":
        feature_figure = normalized_feature_figure(result, artifact)
        if feature_figure is not None:
            st.plotly_chart(feature_figure, use_container_width=True)

    audit_tab, evidence_tab, action_tab = st.tabs(
        ["Sequence audit", "Translational evidence", "Recommended next studies"]
    )
    with audit_tab:
        st.markdown(
            "Flags are deterministic teaching rules and model-domain checks. They are not toxicity or stability predictions."
        )
        st.dataframe(audit, use_container_width=True, hide_index=True)
    with evidence_tab:
        st.dataframe(endpoint_evidence_matrix(), use_container_width=True, hide_index=True)
    with action_tab:
        st.markdown(
            """
            1. **Check novelty and leakage:** search the training set and current AMP databases for near neighbors.
            2. **Define the target:** specify organism, strain, resistance phenotype, and intended assay conditions.
            3. **Evaluate safety-facing endpoints:** hemolysis, mammalian-cell cytotoxicity, aggregation, and solubility.
            4. **Measure activity:** MIC/MBC with biological and technical replicates plus appropriate controls.
            5. **Assess robustness:** protease/serum stability, salt/pH sensitivity, batch reproducibility, and an external sequence set.
            """
        )

    record = research_record(result, artifact, audit)
    st.download_button(
        "Download auditable JSON record",
        data=json.dumps(record, indent=2),
        file_name="amp_finder_research_record.json",
        mime="application/json",
    )
    with st.expander("Interpretation guardrail"):
        st.markdown(
            """
            - **Supported:** the sequence resembles patterns learned from the chosen labels.
            - **Unsupported:** experimental activity, MIC, target spectrum, selectivity, stability, novelty, or clinical utility.
            - **Important:** a high score can reflect dataset construction, homologs, composition, or other shortcuts.
            """
        )


def render_batch_prediction(artifact: dict[str, Any]) -> None:
    st.markdown("### Batch candidate triage")
    st.write(
        "Upload a CSV with a `sequence` column. The public-demo safety limit is 500 rows, and invalid rows remain visible in the output."
    )
    example = "sequence\nKWKLFKKIGAVLKVL\nLLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES\n"
    st.download_button(
        "Download CSV template",
        data=example,
        file_name="amp_finder_batch_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Upload sequence CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        input_frame = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        return
    if "sequence" not in input_frame.columns:
        st.error("The CSV must contain a column named `sequence`.")
        return
    if len(input_frame) > 500:
        st.error("This public-demo safety limit is 500 rows per batch.")
        return

    if st.button("Run batch screening", type="primary"):
        valid_sequences: list[str] = []
        valid_indices: list[int] = []
        errors: dict[int, str] = {}
        for index, raw in input_frame["sequence"].items():
            try:
                valid_sequences.append(normalize_sequence(raw))
                valid_indices.append(index)
            except SequenceValidationError as error:
                errors[index] = str(error)

        output = input_frame.copy()
        output["normalized_sequence"] = ""
        output["score"] = pd.NA
        output["interpretation"] = "Invalid"
        output["validation_error"] = ""
        try:
            if valid_sequences:
                with st.spinner(f"Screening {len(valid_sequences)} valid sequence(s)…"):
                    predictions = predict_sequences(artifact, valid_sequences)
                for row_position, original_index in enumerate(valid_indices):
                    output.at[original_index, "normalized_sequence"] = predictions.iloc[row_position]["sequence"]
                    output.at[original_index, "score"] = float(predictions.iloc[row_position]["score"])
                    output.at[original_index, "interpretation"] = predictions.iloc[row_position]["interpretation"]
            for original_index, message in errors.items():
                output.at[original_index, "validation_error"] = message
        except (ValueError, ImportError, OSError) as error:
            st.error(str(error))
            return

        st.success(f"Processed {len(valid_sequences)} valid sequence(s); {len(errors)} invalid row(s).")
        if valid_sequences:
            valid_output = output.loc[valid_indices].copy()
            valid_output["score"] = pd.to_numeric(valid_output["score"])
            plot_columns = st.columns(2, gap="large")
            with plot_columns[0]:
                score_figure = go.Figure(
                    go.Histogram(
                        x=valid_output["score"],
                        nbinsx=20,
                        marker={"color": COLORS["amp"], "line": {"color": COLORS["text"], "width": 0.4}},
                    )
                )
                score_figure.update_layout(title="Batch score distribution")
                score_figure.update_xaxes(title="AMP-likeness score", range=[0, 1])
                score_figure.update_yaxes(title="Sequences")
                st.plotly_chart(style_figure(score_figure, height=350, legend=False), use_container_width=True)
            with plot_columns[1]:
                counts = valid_output["interpretation"].value_counts()
                result_figure = go.Figure(
                    go.Bar(
                        x=counts.values,
                        y=counts.index,
                        orientation="h",
                        marker={"color": COLORS["non_amp"], "line": {"color": COLORS["text"], "width": 0.4}},
                        text=counts.values,
                        textposition="outside",
                    )
                )
                result_figure.update_layout(title="Screening interpretation counts")
                result_figure.update_xaxes(title="Sequences", rangemode="tozero")
                result_figure.update_yaxes(title="")
                st.plotly_chart(style_figure(result_figure, height=350, legend=False), use_container_width=True)

        st.dataframe(output, use_container_width=True, hide_index=True)
        output_buffer = io.StringIO()
        output.to_csv(output_buffer, index=False)
        st.download_button(
            "Download screening results",
            data=output_buffer.getvalue(),
            file_name="amp_finder_results.csv",
            mime="text/csv",
        )


def length_distribution_figure(frame: pd.DataFrame, feature_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for label, color, name in [
        (1, COLORS["amp"], "AMP"),
        (0, COLORS["non_amp"], "Putative non-AMP"),
    ]:
        mask = frame["label"].astype(int).eq(label).to_numpy()
        figure.add_histogram(
            x=feature_frame.loc[mask, "length"],
            name=name,
            nbinsx=24,
            histnorm="probability",
            opacity=0.70,
            marker={"color": color, "line": {"color": COLORS["text"], "width": 0.35}},
        )
    figure.update_layout(title="Peptide length distribution", barmode="overlay")
    figure.update_xaxes(title="Length (amino acids)")
    figure.update_yaxes(title="Within-class proportion")
    return style_figure(figure, height=420)


def charge_hydropathy_figure(frame: pd.DataFrame, feature_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for label, color, symbol, name in [
        (1, COLORS["amp"], "circle", "AMP"),
        (0, COLORS["non_amp"], "x", "Putative non-AMP"),
    ]:
        mask = frame["label"].astype(int).eq(label).to_numpy()
        rows = feature_frame.loc[mask]
        figure.add_scattergl(
            x=rows["gravy"],
            y=rows["net_charge_pH7"],
            name=name,
            mode="markers",
            marker={"color": color, "symbol": symbol, "size": 8, "opacity": 0.72},
            customdata=rows["length"],
            hovertemplate="Hydropathy %{x:.2f}<br>Charge %{y:.2f}<br>Length %{customdata:.0f} aa<extra>%{fullData.name}</extra>",
        )
    figure.add_hline(y=0, line_color="rgba(239,255,248,0.34)", line_width=1)
    figure.update_layout(title="Charge–hydropathy landscape")
    figure.update_xaxes(title="Mean hydropathy (GRAVY)")
    figure.update_yaxes(title="Estimated net charge at pH 7")
    return style_figure(figure, height=420)


def composition_difference_figure(frame: pd.DataFrame, feature_frame: pd.DataFrame) -> go.Figure:
    composition_columns = [f"aa_{amino_acid}" for amino_acid in AMINO_ACIDS]
    labeled = feature_frame[composition_columns].copy()
    labeled["label"] = frame["label"].astype(int).to_numpy()
    means = labeled.groupby("label")[composition_columns].mean()
    differences = (means.loc[1] - means.loc[0]).rename("difference").reset_index()
    differences["residue"] = differences["index"].str.replace("aa_", "", regex=False)
    differences["absolute"] = differences["difference"].abs()
    differences = differences.nlargest(12, "absolute").sort_values("difference")
    colors = [COLORS["amp"] if value >= 0 else COLORS["non_amp"] for value in differences["difference"]]
    figure = go.Figure(
        go.Bar(
            x=differences["difference"],
            y=differences["residue"],
            orientation="h",
            marker={"color": colors, "line": {"color": COLORS["text"], "width": 0.35}},
            text=[f"{value:+.3f}" for value in differences["difference"]],
            textposition="outside",
        )
    )
    figure.update_layout(title="Largest amino-acid composition differences")
    figure.update_xaxes(title="Mean fraction: AMP minus putative non-AMP")
    figure.update_yaxes(title="Residue")
    return style_figure(figure, height=440, legend=False)


def render_data_observatory(
    dataset: pd.DataFrame,
    dataset_path: Path | None,
    dataset_label: str,
    dataset_is_scientific: bool,
) -> None:
    st.markdown("### Data Observatory")
    st.markdown(
        "Inspect what the classifier can learn before interpreting any metric. Data construction is part of the model."
    )
    banner_class = "science-banner" if dataset_is_scientific else "toy-banner"
    st.markdown(
        f'<div class="{banner_class}"><strong>Active dataset:</strong> {dataset_label}. '
        + (
            "Audit the recorded database releases and metadata before publication."
            if dataset_is_scientific
            else "This view demonstrates the analysis workflow only; it cannot support biological conclusions."
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if dataset.empty:
        st.error("No dataset is available. Prepare `data/processed/amp_dataset.csv` first.")
        return

    normalized_sequences = tuple(dataset["sequence"].astype(str))
    feature_frame = cached_feature_frame(normalized_sequences)
    quality = dataset_quality_audit(dataset)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Records", f"{len(dataset):,}")
    metric_columns[1].metric("Unique sequences", f"{dataset['sequence'].nunique():,}")
    metric_columns[2].metric("Median length", f"{feature_frame['length'].median():.0f} aa")
    metric_columns[3].metric("AMP prevalence", f"{dataset['label'].astype(float).mean():.1%}")
    st.caption(f"Source file: `{dataset_path.name if dataset_path else 'unavailable'}`")

    first, second = st.columns(2, gap="large")
    with first:
        st.plotly_chart(length_distribution_figure(dataset, feature_frame), use_container_width=True)
    with second:
        st.plotly_chart(charge_hydropathy_figure(dataset, feature_frame), use_container_width=True)
    st.plotly_chart(composition_difference_figure(dataset, feature_frame), use_container_width=True)

    quality_tab, split_tab, preview_tab = st.tabs(["Data quality audit", "Split profile", "Record preview"])
    with quality_tab:
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.caption(
            "A passing software audit does not resolve biological label noise, database overlap, assay heterogeneity, or publication bias."
        )
    with split_tab:
        if "split" in dataset.columns:
            profile = (
                dataset.groupby("split", observed=False)
                .agg(
                    records=("sequence", "size"),
                    unique_sequences=("sequence", "nunique"),
                    amps=("label", "sum"),
                    positive_rate=("label", "mean"),
                )
                .reset_index()
            )
            if "split_group" in dataset.columns:
                group_counts = dataset.groupby("split")["split_group"].nunique().rename("groups")
                profile = profile.merge(group_counts, on="split", how="left")
            st.dataframe(profile, use_container_width=True, hide_index=True)
        else:
            st.warning("No split column is available in this dataset.")
    with preview_tab:
        preview_columns = [
            column
            for column in ["sequence", "label", "class_name", "source", "source_id", "split"]
            if column in dataset.columns
        ]
        st.dataframe(dataset[preview_columns].head(25), use_container_width=True, hide_index=True)


def model_importance_figure(artifact: dict[str, Any]) -> go.Figure | None:
    if artifact.get("kind") != "feature_random_forest" or not artifact.get("feature_importance"):
        return None
    importance = (
        pd.DataFrame(artifact["feature_importance"].items(), columns=["feature", "importance"])
        .head(12)
        .sort_values("importance")
    )
    importance["label"] = importance["feature"].map(
        lambda name: FEATURE_LABELS.get(name, name.replace("aa_", "Composition: "))
    )
    figure = go.Figure(
        go.Bar(
            x=importance["importance"],
            y=importance["label"],
            orientation="h",
            marker={"color": COLORS["amp"], "line": {"color": COLORS["text"], "width": 0.4}},
        )
    )
    figure.update_layout(title="Global feature importance")
    figure.update_xaxes(title="Random Forest impurity importance")
    figure.update_yaxes(title="")
    return style_figure(figure, height=470, legend=False)


def render_method_pipeline() -> None:
    st.markdown(
        """
        <div class="method-pipeline">
          <div class="method-step"><span class="step-index">01 · DEFINE</span><strong>Research question</strong><span>Binary AMP recognition, explicitly separated from potency and safety.</span></div>
          <div class="method-step"><span class="step-index">02 · CURATE</span><strong>Traceable labels</strong><span>APD positives, reviewed UniProt parent proteins, and putative-negative caveats.</span></div>
          <div class="method-step"><span class="step-index">03 · SPLIT</span><strong>Leakage control</strong><span>Exact deduplication plus group-disjoint similarity-aware partitions.</span></div>
          <div class="method-step"><span class="step-index">04 · REPRESENT</span><strong>Two evidence views</strong><span>Biological descriptors versus frozen ESM-2 embeddings.</span></div>
          <div class="method-step"><span class="step-index">05 · VALIDATE</span><strong>Threshold discipline</strong><span>Validation-only threshold choice and one held-out test evaluation.</span></div>
          <div class="method-step"><span class="step-index">06 · REPORT</span><strong>Calibrated claims</strong><span>Metrics, provenance, limitations, uncertainty, and experimental gap.</span></div>
          <div class="method-step"><span class="step-index">07 · STRESS TEST</span><strong>Sensitivity analyses</strong><span>Random versus similarity splits, negative definitions, sources, and length bands.</span></div>
          <div class="method-step"><span class="step-index">08 · TRANSLATE</span><strong>External evidence</strong><span>Novelty, target-specific activity, safety, stability, and wet-lab validation.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(
    artifact: dict[str, Any],
    model_path: Path,
    dataset_is_scientific: bool,
) -> None:
    metadata = artifact.get("metadata", {})
    status = metadata.get("scientific_status", "unknown")
    st.markdown("### Model card")
    identity_columns = st.columns(4)
    identity_columns[0].metric("Model", artifact.get("model_name", "Unknown"))
    identity_columns[1].metric("Representation", artifact.get("kind", "Unknown"))
    identity_columns[2].metric("Threshold", f"{float(artifact['threshold']):.3f}")
    identity_columns[3].metric("Status", status.replace("_", " ").title())
    st.caption(
        f"Artifact `{model_path.name}` · threshold chosen from validation MCC · output is not a calibrated clinical probability"
    )

    if status == "synthetic_ui_demo_only":
        st.warning("Toy-model performance is intentionally suppressed because it is not scientific evidence.")
    else:
        test_metrics = metadata.get("test_metrics", {})
        if test_metrics:
            metric_columns = st.columns(4)
            metric_columns[0].metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
            metric_columns[1].metric("Test PR-AUC", f"{test_metrics['average_precision']:.3f}")
            metric_columns[2].metric("Test MCC", f"{test_metrics['mcc']:.3f}")
            metric_columns[3].metric("Balanced accuracy", f"{test_metrics['balanced_accuracy']:.3f}")
            confidence = metadata.get("test_confidence_intervals", {})
            mcc_interval = confidence.get("metrics", {}).get("mcc")
            if mcc_interval:
                level = float(confidence.get("confidence_level", 0.95))
                st.caption(
                    f"MCC {level:.0%} stratified-bootstrap interval: "
                    f"{mcc_interval['lower']:.3f}–{mcc_interval['upper']:.3f} "
                    f"({int(confidence['n_resamples']):,} resamples). Conditional on this fixed test set and pipeline."
                )

    importance_figure = model_importance_figure(artifact)
    if importance_figure is not None:
        st.plotly_chart(importance_figure, use_container_width=True)
        st.caption(
            "Impurity importance is global and association-based; it is not a local or causal explanation."
        )

    st.markdown("#### Scientific limitations")
    st.markdown(
        """
        1. UniProt fragments are **putative negatives**, not peptides experimentally proven inactive.
        2. AMP databases mix organisms, mechanisms, assay conditions, modifications, and publication practices.
        3. Similarity-aware splitting reduces one leakage route but cannot eliminate all structural or database bias.
        4. Binary classification does not estimate MIC, target spectrum, cytotoxicity, hemolysis, stability, or therapeutic index.
        5. Model scores require calibration and external validation before any decision-oriented probability claim.
        """
    )
    st.markdown("#### DOME-aligned readiness")
    st.dataframe(
        dome_readiness_matrix(artifact, dataset_is_scientific=dataset_is_scientific),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "DOME is used as a qualitative transparency framework, not a numeric quality score."
    )


def render_methods(
    artifact: dict[str, Any],
    model_path: Path,
    dataset_is_scientific: bool,
) -> None:
    st.markdown("### Methods & validation")
    st.write(
        "The scientific contribution is the transparent chain from question and labels to split, model, evaluation, and bounded inference."
    )
    render_method_pipeline()
    plan_tab, card_tab, reproducibility_tab = st.tabs(
        ["Pre-specified analysis plan", "Model card & DOME", "Reproducibility contract"]
    )
    with plan_tab:
        st.dataframe(analysis_plan_matrix(), use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Primary estimand:** discrimination on a group-disjoint held-out sequence set drawn from the declared data-generating process.  
            **Primary operating metric:** MCC at a validation-selected threshold.  
            **Supporting metrics:** ROC-AUC, PR-AUC, sensitivity, specificity, balanced accuracy, F1, Brier score, and confusion matrix.  
            **Minimum robustness checks:** random vs similarity split; alternative negative construction; source/time split; class balance; length strata; nearest-neighbor audit.
            """
        )
    with card_tab:
        render_model_card(artifact, model_path, dataset_is_scientific)
    with reproducibility_tab:
        st.code(
            """python scripts/fetch_uniprot_negatives.py --output data/raw/uniprot_negative_parents.fasta --max-records 5000
python scripts/prepare_dataset.py --positive-fasta data/raw/apd_positive.fasta --negative-parent-fasta data/raw/uniprot_negative_parents.fasta --output data/processed/amp_dataset.csv --split-mode similarity --seed 42
python scripts/train_baseline.py --dataset data/processed/amp_dataset.csv --model-output models/baseline_rf.joblib
python scripts/extract_esm_embeddings.py --dataset data/processed/amp_dataset.csv --output data/processed/esm2_embeddings.npz
python scripts/train_esm.py --embeddings data/processed/esm2_embeddings.npz --model-output models/esm2_logreg.joblib
python scripts/compare_models.py
pytest -q""",
            language="bash",
        )
        st.markdown(
            "Record database release/date, query text, input hashes, exclusions, random seed, split threshold, package versions, model artifact hash, and every post-hoc deviation."
        )
        st.markdown(
            "[DOME recommendations for biological ML](https://doi.org/10.1038/s41592-021-01205-4) · "
            "[Negative-data bias in AMP prediction](https://doi.org/10.1093/bib/bbac343)"
        )


def render_scholar_library() -> None:
    st.markdown("### Scholar Library")
    st.write(
        "A study-oriented map of primary databases, methods papers, reporting standards, and the questions each source can actually answer."
    )
    references = reference_library_frame()
    topics = ["All topics"] + sorted(references["topic"].unique().tolist())
    filter_columns = st.columns([0.8, 1.2])
    with filter_columns[0]:
        selected_topic = st.selectbox("Evidence topic", topics)
    with filter_columns[1]:
        query = st.text_input("Search resources", placeholder="e.g., bias, activity, DOME, UniProt")
    filtered = references.copy()
    if selected_topic != "All topics":
        filtered = filtered.loc[filtered["topic"].eq(selected_topic)]
    if query.strip():
        query_lower = query.strip().lower()
        mask = filtered.astype(str).apply(
            lambda column: column.str.lower().str.contains(query_lower, regex=False)
        ).any(axis=1)
        filtered = filtered.loc[mask]

    for row in filtered.to_dict(orient="records"):
        st.markdown(
            f"#### [{row['resource']}]({row['url']})  \n"
            f"**{row['evidence_type']} · {row['year']}**  \n"
            f"{row['use']}"
        )

    st.markdown('<div class="section-kicker">Choose the right research question</div>', unsafe_allow_html=True)
    question_matrix = pd.DataFrame(
        [
            {
                "Question": "Does the sequence look AMP-like?",
                "Required label": "Curated AMP vs defensible comparator",
                "Output": "Binary score/class",
                "Evaluation": "Discrimination, calibration, leakage control",
            },
            {
                "Question": "How potent is it against organism X?",
                "Required label": "Assay-specific MIC/MBC for organism/strain X",
                "Output": "Potency regression or ordinal class",
                "Evaluation": "Error with assay/source-aware external validation",
            },
            {
                "Question": "What organisms might it affect?",
                "Required label": "Target panel with observed activities/inactivities",
                "Output": "Multilabel spectrum",
                "Evaluation": "Per-target PR metrics and coverage",
            },
            {
                "Question": "Is it selectively safe?",
                "Required label": "Hemolysis/cytotoxicity plus antimicrobial potency",
                "Output": "Safety and selectivity endpoints",
                "Evaluation": "Endpoint-specific error and therapeutic-index uncertainty",
            },
        ]
    )
    st.dataframe(question_matrix, use_container_width=True, hide_index=True)

    st.markdown("#### Critical-reading checklist")
    checklist_columns = st.columns(2, gap="large")
    with checklist_columns[0]:
        st.markdown(
            """
            - Are labels experimentally defined or inferred from missing annotation?
            - Were database versions, dates, and licensing constraints reported?
            - Are exact duplicates and near homologs separated across partitions?
            - Was the threshold chosen without looking at the test set?
            - Are class imbalance and prevalence visible?
            """
        )
    with checklist_columns[1]:
        st.markdown(
            """
            - Are uncertainty, calibration, and confidence intervals reported?
            - Is comparison performed on the same held-out data and estimand?
            - Is there external, temporal, or source-based validation?
            - Are potency, spectrum, safety, and stability kept distinct?
            - Do claims stop where the measured endpoints stop?
            """
        )

    st.markdown("#### Translation and reporting standards")
    st.info(
        "DOME is the directly relevant biological-ML reporting framework here. TRIPOD+AI and PROBAST+AI become relevant if the work is redesigned as a clinical individual-level prediction study; they should not be used to imply that this sequence classifier is clinically validated."
    )

    glossary = {
        "AMP-likeness": "Similarity to patterns learned from training labels; not experimental antimicrobial activity.",
        "Putative non-AMP": "A sequence used as a comparator without experimental proof of inactivity.",
        "Similarity leakage": "Information transfer when homologous or near-duplicate sequences occur across train and evaluation partitions.",
        "Calibration": "Agreement between predicted probabilities and observed outcome frequencies in the target population.",
        "MIC": "Minimum inhibitory concentration under a specified organism, medium, protocol, and endpoint definition.",
        "Therapeutic index": "A comparison of antimicrobial potency with host-toxicity endpoints; it cannot be inferred from AMP classification alone.",
    }
    for term, definition in glossary.items():
        with st.expander(term):
            st.write(definition)


render_hero()
models = discover_models()
if not models:
    st.error(
        "No model artifact was found. Run `python scripts/create_demo_assets.py` for a UI smoke test, "
        "or train the real baseline with `python scripts/train_baseline.py`."
    )
    st.stop()

dataset_path, active_dataset, dataset_label, dataset_is_scientific = discover_dataset()

with st.sidebar:
    st.markdown("## AMP Finder AI")
    st.caption("Research intelligence for peptide sequence screening")
    st.markdown("---")
    selected_label = st.selectbox("Active model", list(models.keys()))
    selected_path, selected_artifact = models[selected_label]
    model_status = selected_artifact.get("metadata", {}).get("scientific_status", "unknown")
    st.markdown(f"**Threshold:** {float(selected_artifact['threshold']):.3f}")
    st.markdown(f"**Model status:** `{model_status}`")
    st.markdown(f"**Dataset:** {dataset_label}")
    st.markdown("---")
    st.markdown("**Research guardrails**")
    st.markdown(
        "- Prioritization, not discovery proof\n"
        "- Putative negatives remain uncertain\n"
        "- Similarity-aware evaluation preferred\n"
        "- Wet-lab validation is mandatory"
    )
    st.markdown("---")
    st.markdown(
        '<p class="small-muted">Sequences remain local to this app unless you explicitly download an output record.</p>',
        unsafe_allow_html=True,
    )

render_model_notice(selected_artifact)
overview_tab, predictor_tab, data_tab, methods_tab, library_tab = st.tabs(
    ["Research overview", "Predictor Lab", "Data Observatory", "Methods & validation", "Scholar Library"]
)

with overview_tab:
    render_overview()

with predictor_tab:
    single_tab, batch_tab = st.tabs(["Single peptide", "Batch CSV"])
    with single_tab:
        st.markdown(
            '<div class="research-callout"><strong>Primary question:</strong> Does this sequence resemble patterns learned from antimicrobial-peptide labels, and what evidence is still missing?</div>',
            unsafe_allow_html=True,
        )
        sample_choice = st.selectbox(
            "Example or custom input",
            [
                "Custom sequence",
                "Short cationic example",
                "LL-37 sequence example",
                "Acidic contrast example",
            ],
        )
        sample_values = {
            "Custom sequence": "KWKLFKKIGAVLKVL",
            "Short cationic example": "KLLKLLKKLLK",
            "LL-37 sequence example": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
            "Acidic contrast example": "MDEDDDNQNEEGTQ",
        }
        sequence_text = st.text_area(
            "Peptide sequence (one-letter amino-acid code)",
            value=sample_values[sample_choice],
            height=110,
            help="Canonical residues only. Whitespace and alignment hyphens are removed.",
        )
        if st.button("Analyze sequence", type="primary", use_container_width=True):
            render_single_prediction(selected_artifact, sequence_text)
    with batch_tab:
        render_batch_prediction(selected_artifact)

with data_tab:
    render_data_observatory(
        active_dataset,
        dataset_path,
        dataset_label,
        dataset_is_scientific,
    )

with methods_tab:
    render_methods(selected_artifact, selected_path, dataset_is_scientific)

with library_tab:
    render_scholar_library()

st.markdown(
    """
    <div class="footer-note">
      AMP Finder AI Research Edition · Educational and candidate-prioritization software ·
      Always report dataset provenance, split strategy, held-out metrics, calibration status,
      and the experimental-validation gap.
    </div>
    """,
    unsafe_allow_html=True,
)
