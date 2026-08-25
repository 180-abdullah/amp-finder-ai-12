# Research protocol: sequence-only antimicrobial-peptide screening

Protocol status: pre-analysis template for the real-data demonstration  
Version: 2.0  
Date: 2026-08-25

This protocol turns AMP Finder AI into an auditable computational study. It is intentionally more rigorous than a portfolio notebook, but it is not a preregistered clinical study and does not authorize therapeutic claims.

## 1. Problem and scope

Antimicrobial resistance creates a need for new intervention strategies. Antimicrobial peptides (AMPs) are biologically interesting candidate molecules, but a sequence classifier addresses only an early prioritization question:

> Can sequence-only models distinguish curated AMP sequences from a declared putative-non-AMP comparator under group-disjoint evaluation?

The unit of analysis is one canonical amino-acid sequence. The prediction target is the operational dataset label, not minimum inhibitory concentration (MIC), organism-specific spectrum, mechanism, hemolysis, cytotoxicity, stability, in-vivo efficacy, or clinical benefit.

## 2. Objectives and estimands

### Primary objective

Estimate binary discrimination on a held-out, similarity-group-disjoint test set drawn from the declared data-construction process.

### Primary estimand

Matthews correlation coefficient (MCC) on the untouched test partition at a threshold chosen by maximizing MCC on the validation partition.

The estimand is conditional on:

- the selected APD release and inclusion criteria;
- the UniProt query and negative-fragment sampling procedure;
- canonical sequence representation;
- the chosen similarity-grouping method and threshold;
- the fitted model and validation-selected operating threshold.

It does not estimate prospective wet-lab success or performance in an unspecified biological population.

### Secondary objectives

1. Compare a biologically interpretable Random Forest with frozen ESM-2 embeddings plus Logistic Regression on identical partitions.
2. Describe differences in length, charge, hydropathy, amphipathicity, aromaticity, isoelectric point, entropy, and residue composition.
3. Quantify sensitivity to negative-set construction and sequence-similarity leakage.
4. Identify model-domain, provenance, and chemical-representation limitations for individual predictions.

### Secondary metrics

- ROC-AUC and average precision/PR-AUC;
- balanced accuracy, sensitivity, specificity, precision, and F1;
- Brier score as a probability-quality diagnostic, without claiming deployment calibration;
- confusion matrix and class-specific counts;
- class-stratified percentile-bootstrap intervals on the fixed test set.

## 3. Hypotheses

The study is primarily estimation-focused. The following directional hypotheses guide interpretation but should not replace effect sizes or uncertainty:

- H1: AMP labels will be associated with higher net positive charge than the putative-negative comparator.
- H2: AMP labels will occupy a distinct but overlapping charge–hydropathy region.
- H3: similarity-aware evaluation will produce no better—and potentially worse—performance than a random sequence split because it reduces near-neighbor information transfer.
- H4: ESM-2 embeddings may improve generalization, but any improvement must be evaluated on the same held-out sequences and with uncertainty visible.

## 4. Data sources and provenance

### Positive class

Use a recorded release of the Antimicrobial Peptide Database (APD). Save the original file outside version control and record:

- database/release name;
- retrieval URL and UTC date;
- file SHA-256 hash;
- license or access conditions;
- parsing and exclusion counts;
- whether activity, organism, modification, or peptide type filters were applied.

The positive label means inclusion in the declared APD extraction—not universal activity under all assay conditions.

### Comparator class

Retrieve reviewed bacterial UniProt parent proteins through the documented API query, excluding records with the selected antimicrobial keyword. Sample length-matched internal fragments with a fixed seed.

Call these records **putative non-AMPs**. Missing antimicrobial annotation is not experimental evidence of inactivity. This uncertainty is a core source of label bias and must remain visible in every report.

### Experimental activity extension

If the research question changes to potency, spectrum, or safety, rebuild the target table from assay-specific data such as DBAASP. Harmonize organism, strain, medium, salt, pH, endpoint, units, censoring, and modifications before modeling. Do not relabel the binary APD/UniProt task as MIC prediction.

## 5. Eligibility and representation

Default computational eligibility criteria:

- 8–80 residues;
- only the 20 canonical amino-acid one-letter codes;
- no exact duplicate sequence rows;
- no sequence assigned both labels;
- one recorded source ID and parent ID per row;
- modifications, termini, D-residues, cyclization, and disulfide connectivity either represented separately or explicitly declared absent.

Where multiple records map to the same unmodified sequence, preserve assay/source information in a separate long-form table rather than silently collapsing incompatible measurements.

## 6. Dataset construction

1. Normalize case and remove allowed FASTA whitespace.
2. Exclude noncanonical or out-of-range sequences with a reason code.
3. Deduplicate within each source.
4. Remove or adjudicate cross-label exact conflicts.
5. Draw one or more length-matched putative-negative samples using recorded random seeds.
6. Compute record hashes and source metadata.
7. Assign similarity groups before model fitting.
8. Freeze the prepared dataset and calculate a SHA-256 hash.

The full flow, including every excluded count, should be summarized in a flow diagram or machine-readable ledger.

## 7. Partitioning and leakage control

The preferred demonstration split is approximately 60% training, 20% validation, and 20% test, with each similarity group assigned to only one partition.

The included greedy normalized-edit grouping is transparent and suitable for teaching. A serious benchmark should repeat the analysis with an alignment-aware clustering tool such as MMseqs2 or CD-HIT and report the clustering parameters.

Required leakage checks:

- exact sequence overlap across partitions;
- similarity-group overlap across partitions;
- parent-protein overlap for negative fragments;
- source/accession overlap where relevant;
- nearest-neighbor identity from test to training;
- possible pretraining overlap as a limitation for protein language models.

The random stratified split is a sensitivity analysis, not the headline result.

## 8. Predictors and models

### Explainable feature model

The Random Forest uses 34 deterministic sequence features:

- length;
- 20 residue fractions;
- net charge estimate at pH 7 and charge density;
- Kyte–Doolittle mean hydropathy;
- alpha-helical hydrophobic moment using a stated 100° angle assumption;
- aromaticity and isoelectric point;
- molecular weight and instability index;
- Shannon composition entropy;
- hydrophobic, basic, acidic, and polar fractions.

The hydrophobic moment assumes an alpha-helical geometry. It is a descriptor, not structural evidence.

### Protein-language-model comparison

Use the frozen `facebook/esm2_t6_8M_UR50D` encoder, exclude special/padding tokens, mean-pool residue embeddings, and fit class-balanced Logistic Regression. The encoder is not fine-tuned.

Both models must use the same records and partitions. Model selection must not depend on test-set results.

## 9. Analysis plan

### Descriptive analysis

- show counts, class prevalence, sources, lengths, duplicates, and exclusions;
- plot length distributions and charge–hydropathy landscapes by label;
- report medians and robust intervals rather than only means;
- show composition differences with denominators and class labels;
- stratify by source and length band where the sample supports it.

### Model training

- fit only on the training partition;
- use fixed, documented hyperparameters for the primary comparison;
- choose the operating threshold only on validation data;
- evaluate once on the test partition;
- preserve per-record scores for error analysis.

### Uncertainty

The software reports class-stratified percentile-bootstrap intervals for held-out metrics. These intervals condition on the fixed model, labels, threshold, and pipeline. They do not incorporate label error, model-refitting variability, database selection, or deployment-domain shift.

For a publication-grade study, add cluster-aware resampling and, where feasible, repeat the complete data-construction and training process across planned seeds or resampled clusters.

### Model comparison

Compare the two models on the same test records. Report paired differences with a paired resampling strategy if inferential comparison is needed. Do not select the “winner” from unpaired point estimates.

### Error analysis

Predefine error-analysis strata:

- sequence-length bands;
- net-charge bands;
- hydropathy bands;
- source/database;
- similarity to the training set;
- low-complexity and out-of-domain flags.

Error analysis is descriptive unless multiplicity and inferential procedures were prespecified.

## 10. Robustness and falsification checks

Minimum sensitivity analyses:

1. random versus similarity-aware partitioning;
2. alternative similarity thresholds;
3. at least two defensible negative-set constructions;
4. source- or release-held-out validation;
5. balanced versus natural/target prevalence where known;
6. removal of near-boundary sequence lengths;
7. feature-only versus ESM-2 representation;
8. label-permutation control to detect pipeline leakage;
9. nearest-neighbor baseline to reveal memorization;
10. probability calibration assessed only on data not used to fit the calibrator.

Any sensitivity analysis introduced after seeing the test result must be labeled post hoc.

## 11. Sample size and coverage

There is no universal sequence-classification sample size that guarantees biological validity. Before freezing the study:

- define the smallest effect or operating sensitivity/specificity worth detecting;
- estimate the needed number of independent similarity groups, not only sequence rows;
- ensure both classes and major length/source strata are represented in validation and test sets;
- use pilot estimates cautiously because they are often optimistic;
- report interval width and independent-group counts as design diagnostics.

If the test set is too small for useful uncertainty bounds, treat the work as feasibility evidence and do not compensate by repeating random splits until a favorable result appears.

## 12. Missing data and exclusions

Sequence-only features should be deterministic for eligible inputs. Source metadata can be missing. Report missingness per field, preserve an explicit missing category where appropriate, and do not infer missing assay values as inactivity.

Every exclusion should have a machine-readable reason. Never remove difficult cases solely because they lower performance.

## 13. Reproducibility contract

Archive or record:

- source queries, release dates, URLs, and file hashes;
- raw-to-processed exclusion ledger;
- environment/package versions;
- code commit identifier;
- random seeds;
- split-grouping method and parameters;
- prepared-dataset hash;
- model artifact and hash;
- validation threshold and selection rule;
- predictions for every validation/test record;
- all planned deviations and their dates.

Core reproduction commands are displayed in the website's **Methods & validation** module.

## 14. Claim policy

Supported:

> On the declared held-out dataset, this model discriminated the operational AMP labels from the putative-negative comparator with the reported uncertainty and limitations.

Unsupported without additional evidence:

- “The model discovered an antibiotic.”
- “A score of 0.9 means a 90% chance of antimicrobial activity.”
- “The sequence is active against pathogen X.”
- “The sequence is non-toxic, stable, novel, or clinically useful.”

## 15. Experimental bridge

A computationally prioritized candidate should proceed through a separate, reviewed plan:

1. exact/near-neighbor and intellectual-property search;
2. target organism and assay definition;
3. synthesis and analytical quality control;
4. MIC/MBC and time-kill measurements with controls and replicates;
5. hemolysis and relevant mammalian-cell cytotoxicity;
6. solubility, aggregation, salt/pH, protease, and serum stability;
7. selectivity/therapeutic-index assessment;
8. mechanistic studies if warranted;
9. independently replicated and ethically approved in-vivo work only after sufficient evidence.

## 16. Reporting frameworks

Use DOME (Data, Optimization, Model, Evaluation) as the directly relevant biological-ML transparency framework. TRIPOD+AI and PROBAST+AI become relevant only if the project is redesigned as an individual-level clinical prediction study; citing them does not make a peptide sequence classifier clinically validated.

See [Evidence ledger](EVIDENCE_LEDGER.md) and [Result interpretation](RESULT_INTERPRETATION.md) before communicating results.

