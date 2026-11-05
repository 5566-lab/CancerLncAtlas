Three RNA predictor arms for non-RNA assay endpoints
===================================================

Current experiment
------------------
Compare lncRNA, protein-coding mRNA and their union for BRCA, LUAD and COAD.
The current default is config/m0_three_arm_unscreened_20261105.json.
All measured, annotated genes enter training-local quality checks. There is
no RNA-outcome correlation cutoff, p-value cutoff, FDR cutoff, or top-k
association preselection. The screened run of 2026-11-05 is a reference
experiment, reproducible with its explicit original --config.

The primary targets are PROTEIN_PATHWAY and PHOSPHO_PTMSEA. Measured
METHYLATION_PATHWAY targets form a separate secondary analysis. These
prepared native assay targets currently cover CPTAC only; this is not
TCGA replication or prediction of individual raw protein abundances.
RNA-derived pathways, States, TF scores and deconvolution scores are excluded.

Inputs and fitting
------------------
Use the same measured GENCODE v36 gene-level log2(TPM+1) data, patient
splits, targets and clinical/purity values as the screened reference.
lncRNA and protein_coding annotation classes are disjoint. No historical
expression filters or zero-filled missing genes are introduced.

Quality checks use only the current training rows: require at least two
observed expression values per training cohort and positive residual
variance after clinical/purity projection. Fit missing-expression medians
and scaling on these training rows. Required clinical/purity values are not
imputed. The same quality definitions apply to all three RNA arms.

Age, sex, stage and fixed source DNA purity remain unpenalized covariates.
Only RNA coefficients receive the Lasso penalty. Keep the same 25 penalty
ratios and existing inner holdouts to isolate the effect of removing
association screening. Five outer folds estimate prediction performance.
After evaluation, fit final signatures using the median penalty ratio from
at least three tuned folds. Zero-RNA tuned folds retain their penalty ratio
for final tuning; they are still counted as failed RNA models.

Failure policy
--------------
A fit with no nonzero RNA coefficient is FAILED_NO_RNA_SELECTED, not a
successful clinical-only model. Zero-variable RNA, inadequate observations
and convergence failures are also recorded explicitly. Failed fits export
neither successful predictions nor a successful model.

A target/arm is a complete training success only when all five outer fits
produce a nonempty RNA model with held-out predictions, and the final fit
also selects RNA. A final nonempty signature with failed outer folds is
retained separately as unvalidated, excluded from FINAL_SIGNATURES.tsv.gz.

Do not force RNA selection or eliminate the zero-RNA penalty candidate to
make training appear successful. Do not substitute clinical-only predictions
for failed RNA fits. Clinical-only baselines are retained only for successful
RNA fits as a reference comparator.

Evaluation
----------
Keep all attempted targets in failure denominators. Report per-arm failures
and successes in TARGET_TRAINING_STATUS.tsv and ARM_TARGET_METRICS.tsv.
The three-arm R2 comparison uses the intersection of targets successful in
all three arms, and asserts identical held-out patients, labels and baseline
predictions. Report failure coverage alongside these conditional metrics:
better scores on a selected successful subset alone do not establish a
better overall model.

FINAL_SIGNATURES.tsv.gz contains complete training successes only.
ALL_NONEMPTY_FINAL_SIGNATURES.tsv.gz additionally retains nonempty final
fits with failed evaluation folds, explicitly flagged as unvalidated.
Do not treat a nonempty signature as proof of useful prediction; examine
out-of-fold R2 and improvement over covariates.

When --previous points to the screened reference, reclassify that run by
the same zero-RNA failure policy. SCREENING_POLICY_COMPARISON.tsv reports
success coverage before/after and performance on identical targets
successful under both policies. Do not compare medians over changing sets.

Bootstrap intervals use paired patient draws of fixed out-of-fold errors;
they do not refit models or correct for multiple pathway comparisons.
Pathways overlap and their counts are descriptive, not independent
biological replications.

Upstream labels
---------------
Native assay preprocessing is retained, including upstream cohort-centered
phosphosite normalization. It was not refit within outer folds. This is
retrospective prediction of fixed scores, not fully inductive validation of
the complete assay pipeline, absolute pathway activation or causation.

Execution and delivery
----------------------
Run all preparation, audits, tests, training and figures on hostname 149.
Record target_host=149 before running. At most 24 total fitting workers,
one BLAS thread per worker, no paid GPU.

Run scripts/run_m0_three_arm.py with --aligned, --splits, --covariates,
--cancer, --output, --workers and optional --config. After all three cancers
complete, run scripts/finalize_m0_three_arm.py --results OUTPUT
--previous SCREENED_OUTPUT.

Export paired PNG/PDF figures and summary/failure/signature tables to a new
M0_THREE_ARM_UNSCREENED_demo_20261105_r1 directory in the user's model image
folder. Patient-level data and fitted patient-level predictions stay on 149
and are excluded from Git and the public delivery archive.
