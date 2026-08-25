# Data acquisition and label design

## 1. Positive class: APD

Use the official [APD download page](https://aps.unmc.edu/downloads) or an APD database search followed by FASTA export.

The original idea named APD3. APD3 is a valid historical dataset tied to the 2016 paper. By 2026 the official APD site presents newer releases. Choose one of these two defensible paths and state it explicitly:

- **Historical reproduction:** use an archived APD3 FASTA and cite the exact release/source.
- **Current demonstration:** use the current official APD FASTA and report its release/download date while citing both the database and the relevant release paper.

Save the downloaded file as `data/raw/apd_positive.fasta`. Do not rename or alter source headers before running the pipeline.

Record:

- database/release name;
- download date;
- query or activity filter, if any;
- original record count;
- permitted use and redistribution terms.

The default project question is broad AMP classification. If the downloaded set contains non-antibacterial activities, call the task `antimicrobial peptide prediction`, not `antibacterial peptide prediction`.

## 2. Putative-negative parent proteins: UniProt

Run:

```bash
python scripts/fetch_uniprot_negatives.py \
  --output data/raw/uniprot_negative_parents.fasta \
  --max-records 5000
```

Default query:

```text
(reviewed:true) AND (taxonomy_id:2) AND (length:[100 TO 1000]) AND NOT (keyword:Antimicrobial)
```

The downloader uses the official UniProtKB REST API and writes a companion metadata JSON with the exact query, UTC timestamp, record count, and SHA-256 hash.

Why full proteins rather than randomly generated strings?

- Fragments retain natural amino-acid usage and local sequence patterns.
- Length matching prevents the model from winning simply because the classes have different lengths.
- Reviewed entries improve annotation quality.

Why are they only **putative negatives**?

- “No antimicrobial annotation” is not the same as “experimentally inactive.”
- An intracellular protein fragment could accidentally have membrane-active chemistry.
- Database annotations are incomplete and change over time.

This label uncertainty must appear in the README, model card, and any public post.

## 3. Inclusion and exclusion rules

Default rules:

- 20 canonical amino acids only;
- peptide length 8–80 aa;
- exact positive duplicates removed;
- exact sequence conflicts excluded from negatives;
- one positive-to-one putative-negative ratio;
- negative fragment lengths sampled from the observed positive length distribution;
- deterministic seed 42;
- source and parent identifiers retained.

Do not silently replace `X`, `B`, `Z`, `J`, `U`, or `O`. Drop those sequences and report the count.

## 4. Build and audit the dataset

```bash
python scripts/prepare_dataset.py \
  --positive-fasta data/raw/apd_positive.fasta \
  --negative-parent-fasta data/raw/uniprot_negative_parents.fasta \
  --output data/processed/amp_dataset.csv \
  --split-mode similarity \
  --similarity-threshold 0.80
```

Inspect both outputs:

- `data/processed/amp_dataset.csv`
- `data/processed/amp_dataset.metadata.json`

Quality checklist:

- [ ] Both labels are present in train, validation, and test.
- [ ] No exact sequence appears twice.
- [ ] No exact sequence has conflicting labels.
- [ ] No `split_group` appears in multiple partitions.
- [ ] Length distributions are comparable across labels.
- [ ] Dropped/retained APD counts are documented.
- [ ] Positive source release and negative query are recoverable.
- [ ] “Putative negative” wording is used consistently.

## 5. Important upgrade for a formal study

The built-in greedy edit-similarity grouping is transparent and dependency-light. It is not a substitute for a carefully validated sequence-identity clustering protocol. For research-grade benchmarking:

1. cluster all sequences with MMseqs2 or CD-HIT at a predeclared identity/coverage rule;
2. assign whole clusters to partitions;
3. verify the maximum cross-split similarity;
4. preserve cluster-tool version and command;
5. use an external independent benchmark if available.

That upgrade is outside this portfolio project's claims but is the correct next methodological step.
