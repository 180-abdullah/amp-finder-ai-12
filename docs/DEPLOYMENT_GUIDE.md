# Deployment guide

## Local Streamlit

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
streamlit run app.py
```

The default deployment is feature-model-only. ESM-2 inference additionally requires `requirements-esm.txt` and downloads the selected model checkpoint on first use. A clean Linux PyTorch installation can be large, so enable it only on a host with adequate storage and memory.

## Streamlit Community Cloud

1. Create a public or private GitHub repository.
2. Push this project.
3. Train the real baseline and add `models/baseline_rf.joblib`.
4. Optionally add `models/esm2_logreg.joblib`; if doing so, use the dependencies in `requirements-full.txt` and verify host limits.
5. In Streamlit Community Cloud, choose the repository, branch, and `app.py`.
6. Use Python 3.11.
7. Deploy and wait for dependency/model download.
8. Open the **Science & model card** tab and verify that the status is not `synthetic_ui_demo_only`.
9. Test one valid sequence, one invalid sequence, and one CSV batch.

No secret is required for the public UniProt or Hugging Face resources used here. If a future model is private, store its token in Streamlit Secrets and never commit it.

## Docker

```bash
docker build -t amp-finder-ai .
docker run --rm -p 8501:8501 amp-finder-ai
```

Open `http://localhost:8501`.

## Deployment checklist

- [ ] Real model artifact present.
- [ ] Toy warning absent for the selected model.
- [ ] Data release and source query documented.
- [ ] Test metrics match the saved model card.
- [ ] Single-sequence prediction works.
- [ ] Invalid residues produce a useful error.
- [ ] Batch CSV download works.
- [ ] Mobile/narrow layout is readable.
- [ ] App disclaimer is visible.
- [ ] GitHub README does not claim discovery or clinical utility.

## Resource notes

- The Random Forest is small and fast.
- ESM-2 first-use startup is slower because model files are downloaded and cached.
- Public free hosting can sleep or restart; treat it as a portfolio demo, not a production screening service.
