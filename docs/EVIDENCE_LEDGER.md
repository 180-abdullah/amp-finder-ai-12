# Evidence ledger

This ledger separates external facts, model outputs, assumptions, and unmeasured endpoints. Update it whenever a data source, claim, or endpoint changes.

## Claim-to-source map

| Claim or design choice | Evidence source | What it supports | What it does not support |
|---|---|---|---|
| AMR is a major global-health problem | [WHO AMR fact sheet](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance) | Current burden and surveillance framing | Efficacy of any peptide |
| Priority bacterial threats should guide target selection | [WHO Bacterial Priority Pathogens List 2024](https://www.who.int/publications/i/item/9789240093461) | Public-health priority setting | A peptide's target spectrum |
| Curated AMP sequences can form an operational positive class | [APD downloads](https://aps.unmc.edu/downloads) and the APD release publication | Sequence curation and database membership | Universal activity under every assay |
| Target-specific activity and safety require experimental records | [DBAASP](https://dbaasp.org/home) | Assay-linked activity, structure, hemolysis, and cytotoxicity data | Automatic harmonization across assays |
| Reviewed protein data can be retrieved reproducibly | [UniProt website API paper](https://doi.org/10.1093/nar/gkaf394) | API provenance and programmatic access | Proof that unannotated fragments are inactive |
| Frozen protein-language-model embeddings provide a modern representation | [ESM-2 primary paper](https://doi.org/10.1126/science.ade2574) | Model family and learned protein representations | AMP mechanism understanding or clinical validation |
| AMP benchmarks are sensitive to negative-data construction | [Sidorczuk et al.](https://doi.org/10.1093/bib/bbac343) | Risk of performance inflation and comparator bias | One universally correct negative set |
| Biological ML should report data, optimization, model, and evaluation details | [DOME recommendations](https://doi.org/10.1038/s41592-021-01205-4) | Reporting and transparency structure | A numeric certification score |
| Clinical prediction reporting/risk-of-bias tools have a defined scope | [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) and [PROBAST+AI](https://doi.org/10.1136/bmj-2024-082505) | Individual-level clinical prediction studies | Clinical status for this sequence-screening demonstration |

## Endpoint status for the current project

| Endpoint | Current evidence | Status | Evidence needed to advance |
|---|---|---|---|
| AMP-like sequence pattern | Binary classifier score and descriptor profile | Estimated | Independent external sequence validation |
| Target-species potency | None | Not evaluated | MIC/MBC for a declared organism, strain, and protocol |
| Spectrum | None | Not evaluated | Multi-organism activity panel with observed negatives |
| Hemolysis | None | Not evaluated | HC50 or a declared hemolysis endpoint |
| Mammalian cytotoxicity | None | Not evaluated | Cell-type-specific CC50/viability assay |
| Solubility/aggregation | Composition flags only | Not measured | Experimental or separately validated endpoint model |
| Protease/serum stability | None | Not evaluated | Time-resolved stability measurements |
| Novelty | No live similarity search in the classifier | Not evaluated | Current database and literature similarity search |
| In-vivo efficacy | None | Not evaluated | Ethically approved, independently reviewed study |
| Clinical usefulness | None | Outside scope | A complete translational and clinical development program |

## Internal evidence hierarchy

1. **Software evidence:** tests, compilation, schema checks, deterministic artifacts.
2. **Internal model evidence:** held-out metrics on the declared constructed dataset.
3. **External computational evidence:** source/release/target-held-out validation.
4. **Experimental evidence:** replicated activity, selectivity, and stability assays.
5. **Translational evidence:** exposure, in-vivo efficacy, safety, formulation, and independent reproduction.

A result cannot be promoted to a higher level by visual polish or by a larger model.

## Maintenance rule

For every number shown publicly, retain:

- numerator and denominator;
- dataset and release;
- partition and split method;
- metric definition and threshold;
- uncertainty method;
- code/model version;
- caveat describing the target population and unmeasured endpoints.

