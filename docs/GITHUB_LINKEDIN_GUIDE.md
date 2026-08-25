# GitHub, LinkedIn, and professor outreach guide

Replace bracketed placeholders before posting.

## GitHub repository settings

Repository name:

```text
amp-finder-ai
```

Description:

```text
Explainable AMP screening with physicochemical features, Random Forest, ESM-2 embeddings, leakage-aware evaluation, and a Streamlit demo.
```

Topics:

```text
bioinformatics antimicrobial-peptides machine-learning protein-language-model esm2 streamlit antibiotic-resistance computational-biology
```

Pin these items near the top of the README:

1. one app screenshot;
2. one biological-distribution figure;
3. one held-out model-comparison figure;
4. the limitations paragraph;
5. the live-demo link.

## LinkedIn post

Antibiotic resistance motivates the search for alternatives to conventional antibiotics, including antimicrobial peptides (AMPs). I built **AMP Finder AI**, a small but scientifically careful demonstration that screens peptide sequences for AMP-like patterns.

The project has two layers:

- an explainable Random Forest using sequence length, amino-acid composition, charge, hydropathy, aromaticity, isoelectric point, and amphipathicity-related features;
- frozen ESM-2 protein-language-model embeddings with Logistic Regression.

I also focused on evaluation quality. Instead of relying only on a random split, the pipeline groups similar sequences to reduce train–test leakage, selects the decision threshold on validation data, and reports held-out MCC, balanced accuracy, ROC-AUC, and PR-AUC. UniProt fragments are described as *putative negatives* rather than proven inactive peptides.

The Streamlit app accepts a sequence, returns an AMP-likeness score, and explains the biological descriptors. It explicitly does **not** claim experimental activity, safety, MIC, or clinical usefulness.

GitHub: [LINK]\
Live demo: [LINK]

I would appreciate feedback from researchers working in antimicrobial peptides, microbiology, computational biology, or protein AI—especially on stronger external validation and negative-set design.

#Bioinformatics #MachineLearning #AntimicrobialPeptides #ESM2 #ComputationalBiology #AntibioticResistance

## Short version

I built **AMP Finder AI**, an explainable peptide-screening demo that compares physicochemical features + Random Forest with frozen ESM-2 embeddings + Logistic Regression. The project uses length-matched putative negatives, validation-selected thresholds, and similarity-aware splitting to reduce inflated evaluation. It prioritizes AMP-like sequences for further study; it does not claim experimental discovery. [GitHub] [Demo]

## 60-second professor pitch

> I wanted a project that is small enough to understand completely but connected to a real biological need. I built an antimicrobial-peptide screening pipeline using APD positives and length-matched UniProt fragments as putative negatives. First, I analyzed charge, hydropathy, length, amino-acid composition, and amphipathicity-related features. Then I trained a Random Forest and compared it with frozen ESM-2 embeddings plus Logistic Regression. The important part is the evaluation: I keep similar sequences in the same split, choose the threshold on validation data, and reserve the test split for final metrics. The app explains that the output is an AMP-likeness score, not experimental proof. I would value your advice on negative-set design, external validation, and which biological endpoint would be most useful to add next.

## Email to a professor

Subject: Request for brief feedback on an explainable AMP screening project

Dear Professor [Name],

I am [Your Name], a [program/year] student at [University]. I recently built a small computational-biology demonstration called **AMP Finder AI**. It compares an explainable physicochemical-feature model with frozen ESM-2 sequence embeddings for antimicrobial-peptide screening.

I tried to keep the claims scientifically careful: UniProt fragments are treated as putative negatives, similar sequences are grouped before splitting, the classification threshold is selected on validation data, and the app clearly separates model score from experimental activity.

Project link: [GitHub]\
Demo link: [Streamlit]

If you have time for a brief look, I would be grateful for feedback on the negative-set design and the most meaningful next validation step. I am not presenting this as a discovered antibiotic; my goal is to demonstrate that I understand how AI can support—but not replace—biological validation.

Sincerely,\
[Your Name]\
[Program, University]

## Screenshot checklist

Capture exactly these four images:

1. App title, sequence input, and real-model status.
2. One prediction with score and biological feature table.
3. Biological understanding notebook: charge-versus-hydropathy figure.
4. Held-out baseline-versus-ESM comparison figure with split method in the caption.

Do not post a screenshot while the app says **Toy UI model — synthetic data**.

## Claims checklist before publishing

- [ ] I say “AMP-like” or “screening,” not “discovered antibiotic.”
- [ ] I call the score a score, not a true probability.
- [ ] I state which APD release was used.
- [ ] I state the UniProt query/download date.
- [ ] I call the negative class putative.
- [ ] I state the split method and held-out sample count.
- [ ] I include MCC/PR-AUC, not accuracy alone.
- [ ] I mention toxicity, MIC, and wet-lab validation as missing.
- [ ] I do not claim novelty without a sequence-similarity/database check.
