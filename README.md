# AMP Finder AI — Research Edition

A research-facing, explainable machine-learning platform that asks a focused biological question:

> Does a peptide sequence resemble patterns learned from known antimicrobial peptides?

The project connects antimicrobial resistance, peptide biology, classical machine learning, protein language models, data-quality auditing, uncertainty estimation, and a modern interactive Streamlit workspace—without claiming to discover or validate a new antibiotic.

![Research status](https://img.shields.io/badge/status-research%20demonstration-c6ff4a)
![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-46e6c2)
![Tests](https://img.shields.io/badge/tests-28%20passing-46e6c2)

## What makes the Research Edition different

The website is organized as a small research environment rather than a prediction form:

| Workspace | Scholarly purpose |
|---|---|
| **Research overview** | Frames AMR, the estimand, the decision supported, and the full evidence ladder |
| **Predictor Lab** | Screens one sequence or a batch; exposes model-domain checks, biological descriptors, uncertainty, unmeasured endpoints, and auditable JSON/CSV exports |
| **Data Observatory** | Audits sequence validity, duplicates, cross-label conflicts, class balance, group leakage, provenance, length, charge, hydropathy, and composition |
| **Methods & validation** | Shows the pre-specified analysis plan, model card, DOME-aligned readiness, held-out metrics, bootstrap uncertainty, and exact reproduction commands |
| **Scholar Library** | Connects the project to APD, DBAASP, UniProt, ESM-2, benchmark-bias literature, DOME, TRIPOD+AI, and PROBAST+AI with a question/endpoint map |

The visual system is award-informed—drawing on current Webby/Awwwards science and data-storytelling patterns—without claiming an award or copying a specific site. See [Design research](docs/DESIGN_RESEARCH.md).

## Why this version is scientifically stronger

- Builds a biological understanding layer before prediction: length, charge, hydropathy, amphipathicity, aromaticity, pI, and amino-acid composition.
- Uses one interpretable baseline: Random Forest on 34 explicit sequence features.
- Adds one modern comparison: frozen ESM-2 embeddings plus Logistic Regression.
- Chooses the classification threshold on validation data and reports performance once on held-out test data.
- Reports class-stratified percentile-bootstrap intervals for held-out metrics and states exactly what those intervals do not capture.
- Supports a similarity-aware split to reduce near-duplicate leakage.
- Calls UniProt-derived negatives **putative non-AMPs**, because absence of an annotation is not experimental proof of inactivity.
- Calls the output an **AMP-likeness score**, not a clinical probability.

## What is included

| Component | Purpose |
|---|---|
| `scripts/prepare_dataset.py` | Cleans APD sequences, creates length-matched UniProt fragments, removes exact conflicts, and assigns splits |
| `scripts/train_baseline.py` | Trains and evaluates the explainable Random Forest |
| `scripts/extract_esm_embeddings.py` | Extracts mean-pooled embeddings from `facebook/esm2_t6_8M_UR50D` |
| `scripts/train_esm.py` | Trains Logistic Regression on frozen ESM-2 embeddings |
| `app.py` | Five-module research website with screening, evidence audits, data observatory, methods, and scholar library |
| `src/amp_finder/research.py` | Dataset/sequence audits, evidence matrix, DOME readiness, analysis plan, and curated reference metadata |
| `notebooks/` | Colab-ready biological analysis, baseline training, and ESM-2 walkthroughs |
| `tests/` | Unit, data-split, artifact, notebook, and Streamlit smoke checks |
| `docs/RESEARCH_PROTOCOL.md` | Study question, estimands, hypotheses, data, leakage control, analysis, uncertainty, robustness, claim policy, and experimental bridge |
| `docs/EVIDENCE_LEDGER.md` | Claim-to-source map and endpoint-by-endpoint evidence status |
| `docs/DESIGN_RESEARCH.md` | Award-site research, information architecture, visual rationale, and accessibility principles |
| `docs/` | Data acquisition, method, interpretation, deployment, communication, and research-governance guides |
| `VALIDATION.md` | Packaging-time test record and explicit validation boundaries |

## Quick technical demo

The repository contains a synthetic toy model so the interface can be tested immediately. Its scores are **not scientific predictions** and the app displays a prominent warning.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python scripts/create_demo_assets.py
streamlit run app.py
```

Or double-click `run_demo.bat`.

### macOS/Linux

```bash
bash run_demo.sh
```

Streamlit will normally open `http://localhost:8501`.

## Build the real demonstration

### 1. Obtain the positive FASTA

The original project concept specified APD3. APD3 is the historical 2016 release; the official APD site now provides newer APD releases/current sequence downloads. This project accepts either an APD3 FASTA for exact historical reproduction or a current APD FASTA for an updated demonstration.

Download the FASTA from the [official APD download/search interface](https://aps.unmc.edu/downloads), record the release and download date, and save it as:

```text
data/raw/apd_positive.fasta
```

The repository does not scrape or redistribute APD data. See [Data acquisition](docs/DATA_ACQUISITION.md).

### 2. Obtain reviewed UniProt parent proteins

```bash
python scripts/fetch_uniprot_negatives.py \
  --output data/raw/uniprot_negative_parents.fasta \
  --max-records 5000
```

The default query uses reviewed bacterial proteins, length 100–1000 aa, while excluding records carrying the `Antimicrobial` keyword. The exact query, date, count, and file hash are saved automatically.

### 3. Prepare the labeled dataset

```bash
python scripts/prepare_dataset.py \
  --positive-fasta data/raw/apd_positive.fasta \
  --negative-parent-fasta data/raw/uniprot_negative_parents.fasta \
  --output data/processed/amp_dataset.csv \
  --min-length 8 \
  --max-length 80 \
  --negative-ratio 1.0 \
  --split-mode similarity \
  --similarity-threshold 0.80 \
  --seed 42
```

The built-in similarity grouping uses transparent normalized edit similarity. It is appropriate for this educational demonstration. For a formal benchmark, create cluster IDs with an alignment-aware tool such as MMseqs2 or CD-HIT and keep each cluster in only one partition.

### 4. Train the explainable baseline

```bash
python scripts/train_baseline.py \
  --dataset data/processed/amp_dataset.csv \
  --model-output models/baseline_rf.joblib
```

The app automatically prefers `baseline_rf.joblib` over the toy model.

### 5. Add ESM-2

Install the optional dependencies (Colab already provides PyTorch; a clean Linux installation can be large):

```bash
python -m pip install -r requirements-esm.txt
```

Extract frozen embeddings and train Logistic Regression:

```bash
python scripts/extract_esm_embeddings.py \
  --dataset data/processed/amp_dataset.csv \
  --output data/processed/esm2_embeddings.npz \
  --batch-size 16 \
  --device auto

python scripts/train_esm.py \
  --embeddings data/processed/esm2_embeddings.npz \
  --model-output models/esm2_logreg.joblib
```

The selected 8-million-parameter ESM-2 checkpoint is intentionally small enough for Colab or CPU demonstrations. The project uses the frozen model as a sequence encoder; it does not fine-tune ESM-2.

### 6. Compare only the two intended models

```bash
python scripts/compare_models.py
```

This produces `outputs/model_comparison.csv` and one comparison figure. Use the leakage-aware test split for the headline result; a random-split result may be shown only as a teaching contrast.

### 7. Launch the app

```bash
streamlit run app.py
```

## Evaluation to report

Do not choose one convenient accuracy number. Report:

- ROC-AUC and average precision/PR-AUC
- Matthews correlation coefficient (MCC)
- balanced accuracy
- sensitivity and specificity
- F1 score
- confusion matrix
- validation-selected threshold
- test-set size, class balance, and split method
- 95% class-stratified bootstrap intervals, including the resample count and scope caveat

The primary operating estimand in the included protocol is test-set MCC at a validation-selected threshold. ROC-AUC and PR-AUC remain important threshold-free supporting metrics. Read [Research protocol](docs/RESEARCH_PROTOCOL.md) before declaring an analysis plan.

If the ESM-2 model does not improve over the feature model, that is still an interesting and credible result: on a small curated peptide dataset, simple biological descriptors may already capture much of the available label signal.

## How to interpret an app result

`AMP-like` means the sequence is above the model's validation-selected threshold and outside its uncertainty band. It does **not** mean:

- experimentally antimicrobial;
- active against a particular bacterium;
- low MIC;
- non-toxic or non-hemolytic;
- stable in serum;
- novel relative to known peptides;
- clinically useful.

Read [Result interpretation](docs/RESULT_INTERPRETATION.md) before sharing screenshots.

## Repository map

```text
amp-finder-ai/
├── app.py
├── assets/research.css       # modern responsive visual system
├── src/amp_finder/          # reusable package
├── scripts/                 # data, training, ESM-2, comparison
├── notebooks/               # Colab-ready walkthroughs
├── data/                    # raw inputs excluded; toy data included
├── models/                  # toy model included; real models generated
├── outputs/                 # metrics, predictions, figures
├── tests/
└── docs/
```

## Test the project

```bash
python -m pip install -r requirements-dev.txt
pytest
```

The notebook generator and execution checks are described in `notebooks/README.md`.
The checks completed for this distribution—including research-audit and interface assertions—are recorded in [VALIDATION.md](VALIDATION.md).

## Deployment

For Streamlit Community Cloud, push the repository to GitHub, add the trained feature-model artifact under `models/`, select `app.py` as the entrypoint, and deploy. The default `requirements.txt` intentionally deploys the lightweight feature model. A full ESM-enabled environment uses `requirements-full.txt` and downloads official model weights on first use; confirm that the chosen host has enough memory and storage.

See [Deployment guide](docs/DEPLOYMENT_GUIDE.md) for Streamlit Cloud and Docker instructions.

## Strong, accurate project description

> I built a research-facing antimicrobial-peptide screening platform that compares biologically designed features with frozen ESM-2 sequence embeddings. It combines provenance-aware data construction, similarity-group-disjoint evaluation, validation-only thresholding, held-out bootstrap uncertainty, dataset and sequence audits, and an explicit evidence ladder. The tool prioritizes AMP-like sequences for further study; it does not claim experimental discovery, potency, safety, or clinical usefulness.

Ready-to-use GitHub text, a LinkedIn post, a professor pitch, and a screenshot checklist are in [GitHub and LinkedIn guide](docs/GITHUB_LINKEDIN_GUIDE.md).

## Primary sources

1. Wang G, Li X, Wang Z. APD3: the antimicrobial peptide database as a tool for research and education. *Nucleic Acids Research*. 2016;44(D1):D1087–D1093. [doi:10.1093/nar/gkv1278](https://doi.org/10.1093/nar/gkv1278)
2. Ahmad S, et al. The UniProt website API: facilitating programmatic access to protein knowledge. *Nucleic Acids Research*. 2025;53(W1):W547–W553. [doi:10.1093/nar/gkaf394](https://doi.org/10.1093/nar/gkaf394)
3. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. 2023;379:1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
4. Sidorczuk K, et al. Benchmarks in antimicrobial peptide prediction are biased due to the selection of negative data. *Briefings in Bioinformatics*. 2022;23(5):bbac343. [doi:10.1093/bib/bbac343](https://doi.org/10.1093/bib/bbac343)
5. Walsh I, et al. DOME: recommendations for supervised machine learning validation in biology. *Nature Methods*. 2021. [doi:10.1038/s41592-021-01205-4](https://doi.org/10.1038/s41592-021-01205-4)

## License and data rights

The project code is MIT-licensed. APD data, UniProt data, ESM-2 weights, and other third-party assets remain under their respective terms. The code license does not grant permission to redistribute those assets.
