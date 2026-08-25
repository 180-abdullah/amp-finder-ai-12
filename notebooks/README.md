# Notebooks

The notebooks are generated with `nbformat` so their structure is reproducible:

```bash
python scripts/generate_notebooks.py
```

Execution checks for the CPU notebooks:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_biological_understanding.ipynb

python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/02_baseline_random_forest.ipynb
```

The ESM-2 notebook requires `requirements-esm.txt` and first-run access to the official Hugging Face model files:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/03_esm2_logistic_regression.ipynb
```

Run them in numeric order. If `data/processed/amp_dataset.csv` is absent, notebooks 01 and 02 use bundled synthetic data and visibly label all outputs as non-scientific technical checks.

In a restricted container that prohibits Jupyter kernel sockets, validate the same code cells in one process:

```bash
python scripts/validate_notebook_cells.py \
  notebooks/01_biological_understanding.ipynb \
  notebooks/02_baseline_random_forest.ipynb
```
