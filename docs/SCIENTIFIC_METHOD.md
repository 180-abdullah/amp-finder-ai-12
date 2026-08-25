# Scientific method

## Question and prediction target

The task is binary sequence screening:

```text
peptide sequence → AMP-likeness score → cautious screening interpretation
```

The positive label means “included in the selected APD release.” The negative label means “a length-matched fragment from a reviewed UniProt protein not carrying the selected antimicrobial annotation.” The labels are therefore operational training labels, not perfect biological truth.

## Biological features

The Random Forest uses 34 deterministic features:

- sequence length;
- fractional composition of all 20 canonical amino acids;
- Biopython net charge estimate at pH 7 and charge density;
- Kyte–Doolittle mean hydropathy (GRAVY);
- alpha-helical hydrophobic moment using a 100° residue angle and the Eisenberg scale;
- aromaticity;
- isoelectric point;
- molecular weight;
- instability index;
- Shannon composition entropy;
- hydrophobic, basic, acidic, and polar residue fractions.

The hydrophobic moment is an assumed alpha-helical descriptor. It does not prove that the peptide adopts an alpha helix.

## Models

### Random Forest

The fixed baseline has 500 trees, balanced subsampling, square-root feature sampling, and a minimum leaf size of two. No test-set tuning is performed.

Why it is useful:

- learns nonlinear feature interactions;
- handles composition features well;
- supplies a global feature-importance summary;
- is easy to explain and fast to run.

Limitation: impurity importance is global and can favor correlated or high-variance features. It must not be called a causal explanation of an individual prediction.

### ESM-2 plus Logistic Regression

The modern model uses the frozen `facebook/esm2_t6_8M_UR50D` checkpoint. Residue embeddings are mean-pooled while excluding special and padding tokens. A scaled, class-balanced Logistic Regression classifier is fitted on the frozen embeddings.

This design intentionally separates representation learning from task classification. It demonstrates modern protein AI without pretending that a small AMP dataset is sufficient to responsibly fine-tune a large model.

## Evaluation design

Partitions are approximately 60% train, 20% validation, and 20% test.

- Training data fit the estimator.
- Validation data choose the decision threshold by maximizing MCC.
- Test data are used once for final metrics.
- The same partitions are used for both models.

The preferred split groups similar sequences before partitioning. A random stratified split may be run only as a teaching contrast because it can inflate performance through sequence-family leakage.

Report ROC-AUC, average precision, MCC, balanced accuracy, sensitivity, specificity, F1, Brier score, confusion matrix, sample counts, threshold, and split method. The training scripts also calculate class-stratified percentile-bootstrap intervals on the fixed test set. These intervals quantify finite-test-sample uncertainty conditional on the fitted pipeline; they do not capture label error or domain shift.

The full pre-analysis specification, including estimands, hypotheses, robustness checks, sample-coverage guidance, and claim policy, is in [Research protocol](RESEARCH_PROTOCOL.md).

## What the project can support

Defensible statement:

> The model prioritizes peptide sequences that resemble the operational AMP labels in this dataset and provides biological descriptors that help inspect the result.

Unsupported statements:

- “The model discovered a new antibiotic.”
- “A 94% score means a 94% chance of killing bacteria.”
- “The peptide is safe.”
- “The peptide will work clinically.”
- “ESM-2 understands antimicrobial mechanisms.”

## Scientific limitations

1. Negative-label uncertainty can bias decision boundaries.
2. APD entries vary in organism, assay, target, modification, and activity definition.
3. Modified residues and terminal chemistry are not represented by canonical sequence alone.
4. Similarity grouping reduces but cannot remove all dataset-family and source bias.
5. A binary sequence label does not capture MIC, target organism, toxicity, hemolysis, solubility, or stability.
6. Model scores are not calibrated therapeutic probabilities.
7. Protein language model pretraining data may overlap sequence families represented in the benchmark.

## Credible next extensions

- add toxicity/hemolysis as a separate multi-objective screen;
- predict organism-specific antibacterial activity only with curated target labels;
- add MIC regression after rigorous unit/assay harmonization;
- use an external independent benchmark;
- cluster with MMseqs2/CD-HIT and verify cross-split identity;
- synthesize a small blinded candidate set for wet-lab validation.

These are extensions, not claims made by this version.
