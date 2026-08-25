# Result interpretation

## Read the app in this order

1. **Model status:** toy synthetic model or real-data demonstration model.
2. **AMP-likeness score:** a model ranking score.
3. **Decision threshold:** chosen from validation data, not from the test sequence.
4. **Uncertainty band:** scores near the threshold are shown as borderline.
5. **Biological context:** charge, hydropathy, amphipathicity, length, and other descriptors.
6. **Limitations:** missing potency, toxicity, stability, specificity, and experimental evidence.

## Example language

Good:

> The feature model assigned this sequence an AMP-likeness score of 0.82, above the validation-selected threshold of 0.63. The sequence is cationic and has moderate hydropathy. This makes it a candidate for further computational checks and experimental testing, not a confirmed AMP.

Bad:

> The sequence is 82% antimicrobial and will kill bacteria.

## Why “probability” is avoided

Random Forest vote fractions and Logistic Regression outputs can look probabilistic, but their calibration depends on label quality, prevalence, sampling, split design, and domain shift. The training dataset is deliberately constructed and balanced, while real candidate prevalence is unknown. Therefore the app says **score**.

## Candidate triage after a high score

1. Check exact and near matches against APD/other peptide databases.
2. Review charge, hydropathy, amphipathicity, aggregation, and solubility.
3. Run toxicity and hemolysis screening as separate models.
4. Check synthesis feasibility and sequence liabilities.
5. Select negative and positive experimental controls.
6. Measure activity against defined organisms with a documented assay.
7. Evaluate cytotoxicity and therapeutic index.

No computational result replaces these steps.
