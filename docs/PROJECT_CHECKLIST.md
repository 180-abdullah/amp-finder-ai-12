# Project completion checklist

## Data

- [ ] Download APD FASTA from the official source.
- [ ] Record APD release and download date.
- [ ] Download reviewed UniProt parent proteins with saved query metadata.
- [ ] Build processed dataset and review metadata JSON.
- [ ] Verify duplicates, conflicts, class balance, length balance, and group leakage.

## Analysis

- [ ] Run `01_biological_understanding.ipynb`.
- [ ] Explain why charge and hydropathy matter without saying they are AMP-specific.
- [ ] Export the four main biological figures.

## Models

- [ ] Train Random Forest.
- [ ] Confirm threshold came from validation data.
- [ ] Confirm test data were not used for tuning.
- [ ] Extract ESM-2 embeddings.
- [ ] Train ESM Logistic Regression.
- [ ] Compare both on identical held-out rows.
- [ ] Report MCC, PR-AUC, balanced accuracy, sensitivity, specificity, and confusion matrix.
- [ ] Report held-out bootstrap intervals and retain their scope caveat.
- [ ] Run the planned random-vs-similarity and negative-set sensitivity analyses.
- [ ] Review every post-hoc analysis against `docs/RESEARCH_PROTOCOL.md`.

## App

- [ ] Add `models/baseline_rf.joblib`.
- [ ] Confirm real model is selected by default.
- [ ] Test valid, invalid, short, long, and FASTA-formatted input.
- [ ] Test batch CSV and downloaded results.
- [ ] Read every disclaimer for accuracy.
- [ ] Verify all five workspaces on desktop and mobile widths.
- [ ] Confirm the Data Observatory passes canonical, conflict, provenance, and split-group audits.

## GitHub and sharing

- [ ] Replace all `[Your Name]`, repository, and demo placeholders.
- [ ] Add app and analysis screenshots.
- [ ] Add live Streamlit URL.
- [ ] Confirm raw data are not committed without permission.
- [ ] Run `pytest`.
- [ ] Make a clean clone and repeat installation.
- [ ] Publish only after the synthetic toy warning is absent.
- [ ] Update `docs/EVIDENCE_LEDGER.md` for every new public claim or endpoint.
