Three RNA predictor arms for non-RNA assay endpoints
===================================================

Question and scope
------------------
For BRCA, LUAD and COAD, compare lncRNA, protein-coding mRNA, and their union.
The primary endpoints are prepared PROTEIN_PATHWAY and PHOSPHO_PTMSEA scores.
METHYLATION_PATHWAY is a separate secondary layer, only when measured.
RNA pathways, RNA-derived States, TF footprints and RNA deconvolution scores
are excluded by an explicit assay-layer whitelist.

The current native assay labels in these aligned inputs are CPTAC-only.
TCGA patients without these labels cannot contribute supervised examples.
This experiment does not constitute a TCGA replication, does not benchmark
individual raw protein abundances, and does not claim causal regulation or
absolute pathway activation. All-missing layers remain missing, not zero.

Matched comparison
------------------
Use the same measured GENCODE v36 gene-level log2(TPM+1) data. lncRNA features
are annotated lncRNA genes; mRNA features are annotated protein_coding genes.
The combined arm is their disjoint union. No historical expression filters
or zero-filled missing genes are introduced. The lncRNA axis is checked
against the existing joint M0 input values.

Restrict the existing joint-demo outer and inner patient splits to CPTAC;
never reassign patients per arm. Training-only partial rank correlation
adjusts for age, sex, stage and fixed source DNA purity, then applies
abs(rho) >= 0.10 and BH FDR <= 0.05. BH includes all candidate genes within
each arm and target, so the combined arm has a larger testing family.
This compares the complete predefined pipelines, not only feature additions
with an otherwise fixed selected mRNA set. It is not the full HHGT screening
procedure, which also has other adjustment requirements.

Clinical and purity coefficients are unpenalized. Only RNA coefficients have
an L1 penalty. Missing required clinical or purity values exclude that
patient for the corresponding fit; no purity imputation or RNA substitution.
Use identical Lasso alpha grids and inner holdouts, followed by five outer
fold evaluations. Numeric transforms, clinical encodings, missing expression
medians and screening are estimated only from the relevant training rows.
Final models are fit after outer evaluation using the median tuned penalty
ratio from at least three valid outer folds; these fits do not supply test
performance estimates.

Before comparison, assert identical patient-target-fold keys, target values
and clinical-only baseline predictions across the three arms. Reject a
comparison if these differ. Report pooled out-of-fold R2, RMSE and Spearman
per target, since each native endpoint has one unchanged score definition
across folds, as well as foldwise metrics. Layer summaries include only
targets evaluated in all five folds. A best arm can still have negative R2;
report positive R2 counts and improvement over the same clinical-only model.

The main quantity is combined-minus-mRNA held-out R2. Confidence intervals
use 1000 paired patient bootstrap draws of fixed OOF predictions. They do
not incorporate training/model-selection variability, do not adjust for
multiple targets and must not be read as replicated biological discoveries.
Many pathway gene sets overlap, so target counts are descriptive, not
independent biological replications.

Upstream-label limitation
-------------------------
Native protein, phosphosite and methylation score preprocessing is retained
from the prepared assay products. Some upstream normalizations (including
cohort-centered phosphosite inputs) used the original assay cohort and are
not refit within the outer training folds. Predictor fitting is training-only,
but this benchmark is a retrospective comparison on those fixed labels,
not a fully inductive validation of the whole assay-scoring pipeline.

Execution and outputs
---------------------
All preparation, tests, training and figures run on host 149. Queue behind
existing joint-demo jobs, keeping at most 24 total training workers and one
BLAS thread per fitting worker. An isolated source checkout preserves the
currently running joint demo. No paid GPU is required.

Run scripts/run_m0_three_arm.py with --aligned, --splits, --covariates,
--cancer, --output and --workers. The aligned, split and covariate directories
are the completed joint-demo inputs. Then run scripts/finalize_m0_three_arm.py
--results OUTPUT after all three cancers complete.

Key artifacts: TARGET_COMPARISON.tsv, FOLD_COMPARISON.tsv,
PAIRED_OOF_PREDICTIONS.parquet, TARGETS.tsv availability, FINAL_SIGNATURES.tsv.gz,
FINAL_COVARIATE_COEFFICIENTS.tsv and hashed fitted model checkpoints.
Finalization produces four matched PNG/PDF figures per cancer plus aggregate
comparison and coverage tables. Patient-level values remain in the private
server results and are not included in Git publication.
