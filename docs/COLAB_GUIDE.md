# Google Colab guide

## Recommended route

1. Upload this project to a GitHub repository.
2. Open Colab and select **File → Open notebook → GitHub**.
3. Enter your repository URL.
4. Run the notebooks in numeric order.

## Clone and install in one Colab cell

Replace the placeholder URL:

```python
!git clone https://github.com/YOUR-USERNAME/amp-finder-ai.git
%cd amp-finder-ai
!python -m pip install -q -r requirements-dev.txt
!python -m pip install -q -e .
```

Upload the two FASTA files into `data/raw/` using the Colab Files sidebar, or mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Do not publish raw APD data to a public repository unless the source terms explicitly permit redistribution.

## Notebook order

1. `01_biological_understanding.ipynb`
2. `02_baseline_random_forest.ipynb`
3. `03_esm2_logistic_regression.ipynb`

The first two run on CPU. For the ESM-2 notebook, select **Runtime → Change runtime type → T4 GPU** when available. The 8M checkpoint also runs on CPU, but more slowly.

## Save trained artifacts

Download these files from the Colab Files sidebar:

- `models/baseline_rf.joblib`
- `models/esm2_logreg.joblib` (after the optional ESM notebook)
- `outputs/model_comparison.csv`
- `outputs/figures/model_comparison.png`

Add the real model artifact to your GitHub repository before deploying the Streamlit app. Confirm that the app no longer shows the synthetic toy-model warning.
