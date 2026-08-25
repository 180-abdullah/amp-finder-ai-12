# Validation record

Validation date: 2026-08-25

This file records the checks performed on the distributed project. It is a
software-validation record, not evidence that the bundled synthetic demo model
has biological validity.

## Completed checks

- Python source compilation completed without syntax errors.
- The automated test suite completed with **28 passing tests**.
- The complete demo-data baseline pipeline ran end to end: validation,
  deduplication, feature extraction, similarity-aware splitting, model fitting,
  threshold selection, held-out evaluation, artifact saving, and inference.
- The Streamlit server started successfully and passed its health check.
- Streamlit's application test runner loaded the app, submitted a sequence,
  rendered the result, created the Plotly explanation charts, and produced the
  sequence-audit/evidence tables without an uncaught exception.
- Research-specific tests verified canonical-sequence auditing, duplicate and
  cross-label-conflict detection, split-group leakage detection, explicit
  unmeasured translational endpoints, DOME claim blocking for toy data, and the
  curated reference-library schema.
- Class-stratified percentile-bootstrap metric intervals were checked for
  deterministic behavior, bounds, and an explicit scope caveat.
- The modern five-module interface rendered the Research Edition identity,
  toy-model warning, Data Observatory, Scholar Library, and single-sequence
  research audit.
- Notebook 01 and Notebook 02 were executed cell by cell with the bundled demo
  data. Their plots, model artifact, predictions, and internal assertions were
  generated successfully.
- All three notebooks passed structural validation with `nbformat`.

## Deliberate validation boundary

Notebook 03 requires PyTorch, Transformers, and a download of the ESM-2 model
checkpoint. Its code path and notebook structure were checked, but the model
download and embedding extraction were not executed in the packaging
environment. Run it in Google Colab or another environment with the optional
dependencies from `requirements-esm.txt`.

The bundled `models/demo_baseline_rf.joblib` is trained only on synthetic
sequences so that the interface can be tested immediately. Do not quote its
scores as scientific results. Rebuild the dataset from documented APD and
UniProt sources, run the complete training workflow, and report the generated
held-out metrics before presenting biological conclusions.

## Reproduce the software checks

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q src scripts app.py tests
pytest -q
python scripts/validate_notebook_cells.py \
  notebooks/01_biological_understanding.ipynb \
  notebooks/02_baseline_random_forest.ipynb
python scripts/smoke_test_streamlit.py
```
