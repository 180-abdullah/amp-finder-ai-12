#!/usr/bin/env python3
"""Generate reader-facing Jupyter notebooks with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def notebook(cells):
    document = nbf.v4.new_notebook(cells=cells)
    document["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    }
    return document


COMMON_SETUP = r'''
from pathlib import Path
import sys

def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src/amp_finder").exists():
            return candidate
    raise FileNotFoundError("Run this notebook inside the AMP Finder AI project.")

PROJECT_ROOT = find_project_root(Path.cwd())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
(PROJECT_ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
print("Project root:", PROJECT_ROOT)
'''


def build_biology_notebook():
    cells = [
        markdown(
            """
# AMP Finder AI — Biological Understanding

## Goal

Inspect what separates the operational AMP and putative-non-AMP labels before training any model. The notebook focuses on charge, hydropathy, length, amphipathicity-related descriptors, and amino-acid composition.

If the real processed dataset is absent, the notebook uses the bundled synthetic UI dataset and labels every result as a technical demonstration.
            """
        ),
        markdown(
            """
## Setup

For Colab, clone the repository and install `requirements-dev.txt` first. Raw APD data are not bundled; follow `docs/DATA_ACQUISITION.md`.
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from amp_finder.constants import AMINO_ACIDS
from amp_finder.features import extract_feature_frame

sns.set_theme(style="whitegrid", context="notebook")
real_dataset_path = PROJECT_ROOT / "data/processed/amp_dataset.csv"
demo_dataset_path = PROJECT_ROOT / "data/demo/demo_sequences.csv"
dataset_path = real_dataset_path if real_dataset_path.exists() else demo_dataset_path
dataset = pd.read_csv(dataset_path)
is_demo = dataset_path == demo_dataset_path
print("Dataset:", dataset_path)
print("Status:", "SYNTHETIC UI DEMO — NOT SCIENTIFIC EVIDENCE" if is_demo else "Prepared APD/UniProt demonstration data")
print("Rows:", len(dataset))
dataset.head()
            '''
        ),
        markdown("## Steps\n\n### 1. Check label, sequence, and split integrity"),
        code(
            r'''
required = {"sequence", "label", "split", "split_group"}
assert required.issubset(dataset.columns), f"Missing: {required - set(dataset.columns)}"
assert set(dataset["label"].unique()) == {0, 1}
assert not dataset["sequence"].duplicated().any(), "Exact sequence duplicates found"
assert (dataset.groupby("split_group")["split"].nunique() == 1).all(), "Group leakage found"

quality_table = (
    dataset.groupby(["split", "label"], observed=True)
    .size()
    .rename("rows")
    .reset_index()
)
quality_table["class"] = quality_table["label"].map({1: "AMP", 0: "Putative non-AMP"})
quality_table
            '''
        ),
        markdown("### 2. Extract explainable biological features"),
        code(
            r'''
features = extract_feature_frame(dataset["sequence"])
overlapping_feature_columns = [column for column in features.columns if column in dataset.columns]
dataset_context = dataset.drop(columns=overlapping_feature_columns).reset_index(drop=True)
analysis = pd.concat([dataset_context, features], axis=1)
analysis["class"] = analysis["label"].map({1: "AMP", 0: "Putative non-AMP"})

summary_columns = ["length", "net_charge_pH7", "gravy", "hydrophobic_moment", "aromaticity", "isoelectric_point"]
summary = analysis.groupby("class", observed=True)[summary_columns].agg(["median", "mean", "std"])
summary.round(3)
            '''
        ),
        markdown("### 3. Compare the main feature distributions"),
        code(
            r'''
palette = {"AMP": "#2F6BFF", "Putative non-AMP": "#D59B2D"}
figure, axes = plt.subplots(2, 2, figsize=(12, 8))
for axis, feature, title in zip(
    axes.ravel(),
    ["length", "net_charge_pH7", "gravy", "hydrophobic_moment"],
    ["Sequence length", "Estimated net charge at pH 7", "Mean hydropathy (GRAVY)", "Alpha-helical hydrophobic moment"],
):
    sns.histplot(
        data=analysis,
        x=feature,
        hue="class",
        hue_order=["AMP", "Putative non-AMP"],
        palette=palette,
        element="step",
        stat="density",
        common_norm=False,
        ax=axis,
        legend=(feature == "net_charge_pH7"),
    )
    axis.set_title(title)
    axis.set_xlabel(title)
    axis.set_ylabel("Density")
    if feature == "net_charge_pH7" and axis.get_legend() is not None:
        axis.get_legend().set_title("Operational label")
figure.suptitle("Biological feature distributions by operational label", fontsize=15, y=1.02)
figure.tight_layout()
overview_path = PROJECT_ROOT / "outputs/figures/01_biological_overview.png"
figure.savefig(overview_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", overview_path)
            '''
        ),
        markdown("### 4. Examine charge and hydropathy together"),
        code(
            r'''
figure, axis = plt.subplots(figsize=(8.5, 6))
sns.scatterplot(
    data=analysis,
    x="gravy",
    y="net_charge_pH7",
    hue="class",
    hue_order=["AMP", "Putative non-AMP"],
    palette=palette,
    alpha=0.70,
    s=48,
    ax=axis,
)
axis.axhline(0, color="#667085", linewidth=1, linestyle="--")
axis.set_title("Charge–hydropathy sequence space")
axis.set_xlabel("Mean hydropathy (Kyte–Doolittle GRAVY)")
axis.set_ylabel("Estimated net charge at pH 7")
axis.legend(title="Operational label", frameon=False)
figure.tight_layout()
scatter_path = PROJECT_ROOT / "outputs/figures/01_charge_hydropathy.png"
figure.savefig(scatter_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", scatter_path)
            '''
        ),
        markdown("### 5. Compare amino-acid composition"),
        code(
            r'''
composition_columns = [f"aa_{aa}" for aa in AMINO_ACIDS]
composition_means = analysis.groupby("class", observed=True)[composition_columns].mean().T
composition_means.index = [name.replace("aa_", "") for name in composition_means.index]
composition_difference = (
    composition_means["AMP"] - composition_means["Putative non-AMP"]
).sort_values()

figure, axis = plt.subplots(figsize=(9, 6))
colors = ["#D59B2D" if value < 0 else "#2F6BFF" for value in composition_difference]
axis.barh(composition_difference.index, composition_difference.values, color=colors)
axis.axvline(0, color="#344054", linewidth=1)
axis.set_title("Mean amino-acid composition difference")
axis.set_xlabel("AMP minus putative-non-AMP fraction")
axis.set_ylabel("Residue")
figure.tight_layout()
composition_path = PROJECT_ROOT / "outputs/figures/01_composition_difference.png"
figure.savefig(composition_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", composition_path)
            '''
        ),
        markdown(
            """
## Checks

The code below confirms that feature rows are complete and physically plausible. It does not establish that the labels are biologically perfect.
            """
        ),
        code(
            r'''
assert len(features) == len(dataset)
assert np.isfinite(features.to_numpy()).all()
assert (features["length"] == dataset["sequence"].str.len()).all()
composition_sum = features[[f"aa_{aa}" for aa in AMINO_ACIDS]].sum(axis=1)
assert np.allclose(composition_sum, 1.0, atol=1e-8)
print("Feature integrity checks passed.")

if is_demo:
    print("IMPORTANT: observed class differences were deliberately created in synthetic data and are not biological findings.")
else:
    print("Interpret differences as dataset associations, not universal AMP mechanisms.")
            '''
        ),
        markdown(
            """
## Next Steps

1. If this notebook used synthetic data, prepare the real APD/UniProt dataset and rerun it.
2. Describe distributions and overlap, not only class means.
3. Run `02_baseline_random_forest.ipynb` on the same fixed partitions.
4. Preserve the figures and metadata with the model card.
            """
        ),
    ]
    return notebook(cells)


def build_baseline_notebook():
    cells = [
        markdown(
            """
# AMP Finder AI — Explainable Random Forest Baseline

## Goal

Train one understandable model on fixed biological features, choose its decision threshold using validation data, evaluate it once on held-out test data, and save a versioned artifact for the Streamlit app.
            """
        ),
        markdown(
            """
## Setup

Run the biological-understanding notebook first. This notebook falls back to synthetic data only for a technical execution check.
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from amp_finder.modeling import save_artifact, train_feature_random_forest

sns.set_theme(style="whitegrid", context="notebook")
real_dataset_path = PROJECT_ROOT / "data/processed/amp_dataset.csv"
demo_dataset_path = PROJECT_ROOT / "data/demo/demo_sequences.csv"
dataset_path = real_dataset_path if real_dataset_path.exists() else demo_dataset_path
dataset = pd.read_csv(dataset_path)
is_demo = dataset_path == demo_dataset_path
print("Dataset:", dataset_path)
print("Status:", "SYNTHETIC UI DEMO — NOT SCIENTIFIC EVIDENCE" if is_demo else "Prepared APD/UniProt demonstration data")
dataset.groupby(["split", "label"]).size().unstack(fill_value=0)
            '''
        ),
        markdown("## Steps\n\n### 1. Fit on train and select the threshold on validation"),
        code(
            r'''
status = "synthetic_ui_demo_only" if is_demo else "real_data_demonstration"
artifact, predictions = train_feature_random_forest(
    dataset,
    random_state=42,
    scientific_status=status,
    dataset_notes=str(dataset_path),
)
print("Selected validation threshold:", round(artifact["threshold"], 3))
pd.DataFrame(
    [artifact["metadata"]["validation_metrics"], artifact["metadata"]["test_metrics"]],
    index=["validation", "held-out test"],
)[["n", "roc_auc", "average_precision", "balanced_accuracy", "mcc", "f1", "sensitivity_recall", "specificity", "brier_score"]].round(3)
            '''
        ),
        markdown("### 2. Inspect held-out discrimination and errors"),
        code(
            r'''
test_predictions = predictions.query("split == 'test'").copy()
test_labels = test_predictions["label"].to_numpy()
test_scores = test_predictions["score"].to_numpy()
test_calls = test_predictions["prediction"].to_numpy()

fpr, tpr, _ = roc_curve(test_labels, test_scores)
precision, recall, _ = precision_recall_curve(test_labels, test_scores)
matrix = confusion_matrix(test_labels, test_calls, labels=[0, 1])

figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
axes[0].plot(fpr, tpr, color="#2F6BFF", linewidth=2)
axes[0].plot([0, 1], [0, 1], color="#98A2B3", linestyle="--")
axes[0].set(title="Held-out ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")

axes[1].plot(recall, precision, color="#D59B2D", linewidth=2)
axes[1].set(title="Held-out precision–recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))

sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2])
axes[2].set(title="Held-out confusion matrix", xlabel="Predicted label", ylabel="True label")
axes[2].set_xticklabels(["Putative non-AMP", "AMP"], rotation=20)
axes[2].set_yticklabels(["Putative non-AMP", "AMP"], rotation=0)
figure.tight_layout()
evaluation_path = PROJECT_ROOT / "outputs/figures/02_baseline_evaluation.png"
figure.savefig(evaluation_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", evaluation_path)
            '''
        ),
        markdown("### 3. Inspect global feature importance"),
        code(
            r'''
importance = (
    pd.DataFrame(artifact["feature_importance"].items(), columns=["feature", "importance"])
    .head(12)
    .sort_values("importance")
)
figure, axis = plt.subplots(figsize=(8.5, 6))
axis.barh(importance["feature"], importance["importance"], color="#2F6BFF")
axis.set_title("Random Forest global feature importance")
axis.set_xlabel("Impurity importance")
axis.set_ylabel("")
figure.tight_layout()
importance_path = PROJECT_ROOT / "outputs/figures/02_feature_importance.png"
figure.savefig(importance_path, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", importance_path)
            '''
        ),
        markdown("### 4. Save the app artifact and predictions"),
        code(
            r'''
model_path = PROJECT_ROOT / ("models/notebook_demo_baseline_rf.joblib" if is_demo else "models/baseline_rf.joblib")
prediction_path = PROJECT_ROOT / ("outputs/notebook_demo_predictions.csv" if is_demo else "outputs/baseline_predictions.csv")
save_artifact(artifact, model_path)
predictions.to_csv(prediction_path, index=False)
print("Saved model:", model_path)
print("Saved predictions:", prediction_path)
            '''
        ),
        markdown("## Checks"),
        code(
            r'''
assert set(predictions["split"]) == {"train", "validation", "test"}
assert predictions["score"].between(0, 1).all()
assert 0 < artifact["threshold"] < 1
assert artifact["metadata"]["test_metrics"]["n"] == len(test_predictions)
assert not dataset["sequence"].duplicated().any()
assert (dataset.groupby("split_group")["split"].nunique() == 1).all()
print("Training, prediction-range, held-out-count, duplicate, and split-group checks passed.")
if is_demo:
    print("Do not share these toy metrics. Rerun after creating data/processed/amp_dataset.csv.")
            '''
        ),
        markdown(
            """
## Next Steps

1. Replace synthetic data with the prepared APD/UniProt dataset if necessary.
2. Run the ESM-2 notebook on the identical split assignments.
3. Compare held-out metrics and failure cases—not only the highest number.
4. Launch `streamlit run app.py` and confirm the real-data artifact is selected.
            """
        ),
    ]
    return notebook(cells)


def build_esm_notebook():
    cells = [
        markdown(
            """
# AMP Finder AI — Frozen ESM-2 Embeddings

## Goal

Encode each peptide with the small frozen ESM-2 checkpoint, fit Logistic Regression on those embeddings, and compare it fairly with the feature model using identical held-out rows.

This notebook downloads official model weights on first execution. In Colab, enable a GPU when available.
            """
        ),
        markdown(
            """
## Setup

Install optional dependencies before running:

```python
%pip install -q -r requirements-esm.txt
%pip install -q -e .
```
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
import importlib.util
missing = [name for name in ["torch", "transformers"] if importlib.util.find_spec(name) is None]
if missing:
    raise ImportError(f"Install requirements-esm.txt first. Missing: {missing}")

import pandas as pd
import matplotlib.pyplot as plt

from amp_finder.esm import DEFAULT_ESM_MODEL, embed_sequences, save_embedding_bundle
from amp_finder.modeling import load_artifact, save_artifact, train_esm_logistic_regression

real_dataset_path = PROJECT_ROOT / "data/processed/amp_dataset.csv"
demo_dataset_path = PROJECT_ROOT / "data/demo/demo_sequences.csv"
dataset_path = real_dataset_path if real_dataset_path.exists() else demo_dataset_path
dataset = pd.read_csv(dataset_path)
is_demo = dataset_path == demo_dataset_path
print("Dataset:", dataset_path)
print("Status:", "SYNTHETIC UI DEMO — NOT SCIENTIFIC EVIDENCE" if is_demo else "Prepared APD/UniProt demonstration data")
            '''
        ),
        markdown("## Steps\n\n### 1. Extract frozen mean-pooled ESM-2 embeddings"),
        code(
            r'''
embedding_path = PROJECT_ROOT / ("data/processed/esm2_demo_embeddings.npz" if is_demo else "data/processed/esm2_embeddings.npz")
embeddings = embed_sequences(
    dataset["sequence"],
    model_name=DEFAULT_ESM_MODEL,
    batch_size=16,
    device="auto",
)
save_embedding_bundle(
    embedding_path,
    embeddings=embeddings,
    sequences=dataset["sequence"].to_numpy(),
    labels=dataset["label"].to_numpy(),
    splits=dataset["split"].to_numpy(),
    groups=dataset["split_group"].to_numpy(),
    model_name=DEFAULT_ESM_MODEL,
)
print("Embedding shape:", embeddings.shape)
print("Saved:", embedding_path)
            '''
        ),
        markdown("### 2. Fit Logistic Regression and evaluate held-out data"),
        code(
            r'''
status = "synthetic_ui_demo_only" if is_demo else "real_data_demonstration"
artifact, predictions = train_esm_logistic_regression(
    embeddings,
    dataset["label"].to_numpy(),
    dataset["split"].to_numpy(),
    dataset["split_group"].to_numpy(),
    embedding_model_name=DEFAULT_ESM_MODEL,
    scientific_status=status,
    dataset_notes=str(dataset_path),
)
pd.DataFrame(
    [artifact["metadata"]["validation_metrics"], artifact["metadata"]["test_metrics"]],
    index=["validation", "held-out test"],
)[["n", "roc_auc", "average_precision", "balanced_accuracy", "mcc", "f1", "sensitivity_recall", "specificity", "brier_score"]].round(3)
            '''
        ),
        markdown("### 3. Save the ESM-2 classifier and compare with the baseline"),
        code(
            r'''
model_path = PROJECT_ROOT / ("models/notebook_demo_esm2_logreg.joblib" if is_demo else "models/esm2_logreg.joblib")
prediction_path = PROJECT_ROOT / ("outputs/notebook_demo_esm2_predictions.csv" if is_demo else "outputs/esm2_predictions.csv")
save_artifact(artifact, model_path)
predictions.insert(0, "sequence", dataset["sequence"].to_numpy())
predictions.to_csv(prediction_path, index=False)
print("Saved model:", model_path)

baseline_path = PROJECT_ROOT / ("models/notebook_demo_baseline_rf.joblib" if is_demo else "models/baseline_rf.joblib")
if baseline_path.exists():
    baseline = load_artifact(baseline_path)
    comparison = pd.DataFrame(
        [baseline["metadata"]["test_metrics"], artifact["metadata"]["test_metrics"]],
        index=["Feature Random Forest", "ESM-2 + Logistic Regression"],
    )[["roc_auc", "average_precision", "balanced_accuracy", "mcc", "f1"]]
    display(comparison.round(3))
    axis = comparison[["average_precision", "mcc", "balanced_accuracy"]].plot.bar(
        figsize=(9, 5), color=["#2F6BFF", "#D59B2D", "#667085"], rot=0
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test metric")
    axis.set_title("Models evaluated on identical held-out rows")
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    comparison_path = PROJECT_ROOT / "outputs/figures/03_model_comparison.png"
    plt.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved:", comparison_path)
else:
    print("Baseline artifact not found; run notebook 02 before model comparison.")
            '''
        ),
        markdown("## Checks"),
        code(
            r'''
assert embeddings.ndim == 2
assert embeddings.shape[0] == len(dataset)
assert predictions["score"].between(0, 1).all()
assert set(predictions["split"]) == {"train", "validation", "test"}
assert artifact["embedding_model_name"] == DEFAULT_ESM_MODEL
assert artifact["metadata"]["test_metrics"]["n"] == int((dataset["split"] == "test").sum())
print("Embedding alignment, score range, split, model-name, and held-out-count checks passed.")
if is_demo:
    print("Do not share these toy metrics. Rerun after creating data/processed/amp_dataset.csv.")
            '''
        ),
        markdown(
            """
## Next Steps

1. Investigate where the two models disagree.
2. Report the same held-out metrics and split details for both.
3. If ESM-2 performs worse, discuss small-data limitations instead of hiding the result.
4. Use external validation before making any discovery claim.
            """
        ),
    ]
    return notebook(cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_biological_understanding.ipynb": build_biology_notebook(),
        "02_baseline_random_forest.ipynb": build_baseline_notebook(),
        "03_esm2_logistic_regression.ipynb": build_esm_notebook(),
    }
    for filename, document in notebooks.items():
        destination = NOTEBOOK_DIR / filename
        nbf.write(document, destination)
        print("Generated", destination)


if __name__ == "__main__":
    main()
