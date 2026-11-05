Joint TCGA/CPTAC M0 with clinical adjustment
============================================

Scope and status
----------------
This implementation jointly fits BRCA, LUAD and COAD. It supersedes the separate
native-score Lasso demo for shared, recomputable RNA targets. Full fitted results
are a separate artifact and require COMPLETE receipts; source publication and
unit tests alone do not establish completed training or useful signatures.
The HHGT model and its Pathway=1.0, State=0.5 losses are unchanged.

Prediction and objective
------------------------
For each cancer and target, prediction is Y_hat = X beta + Z gamma. X contains
restored, commonly measured GENCODE v36 lncRNA log2(TPM+1). Z includes an
intercept, age, sex, stage, cohort, fixed DNA purity and cohort-by-purity terms.
The fitted objective is sum_i w_i (Y_i - X_i beta - Z_i gamma)^2 / 2 plus
lambda times the L1 norm of the standardized lncRNA coefficients. The total
weight of each observed training cohort is equal. Gamma is UNPENALIZED.
Weighted projection removes Z from both X and Y during optimization; gamma is
then refitted conditional on the selected beta. This is mathematically the
partially penalized joint regression, not residualizing Y alone.

On 2026-11-05 the user explicitly selected clinical/purity predictors in the
final model, superseding the earlier lncRNA-only input requirement. Final
prediction therefore needs the clinical variables and purity, except for
explicit target-specific exclusions. The shared lncRNA coefficients are one
signature per cancer/target. Clinical coefficients and cohort calibration are
exported separately; a gene list alone is not a complete prediction model.

Fixed clinical values and purity
--------------------------------
TCGA age/sex/stage use the R6.2 merged covariate snapshot, whose stage fields
include the recorded clinical-survival harmonization. DNA purity uses its
absolute_purity column, NOT the legacy purity column (which contains ESTIMATE).
CPTAC age/sex/stage use CLINICAL_ALIGNED; purity uses the published WES_purity
values in PUBLISHED_PHENOTYPES. The exact original field and cohort/source label
are retained. The values are not recomputed from lncRNA, averaged across
methods, replaced with RNA estimates, or fitted on test patients.

The methods are recorded as TCGA_ABSOLUTE and CPTAC_PUBLISHED_WES rather than
assuming that two measurements from different projects are interchangeable.
Separate cohort purity slopes are fitted where both cohorts are observed.
For leave-cohort-out evaluation, the source-trained calibration is used without
fitting any reference on the held-out cohort; transfer is an empirical test,
not an assertion that assay differences have vanished.

Required covariate missingness is complete-case, without imputation. An unseen
clinical category or variation of a training-constant clinical field yields
an unavailable prediction and is counted as reduced prediction coverage.
Sex and stage are categorical; age and purity are numeric. When the target is
the same source DNA purity value, purity and its interactions are excluded.
An RNA ESTIMATE target is not the same source as DNA purity and is not treated
as an identical measurement. Covariate-only baselines use the exact same
training patients and encodings as their corresponding full models.

Clinical plus purity coverage before target-specific missingness:

======= ====== ======== =====================
Cancer  Cohort Patients Complete clinical+DNA
======= ====== ======== =====================
BRCA    TCGA   1090     992
BRCA    CPTAC  121      100
LUAD    TCGA   509      459
LUAD    CPTAC  110      108
COAD    TCGA   454      375
COAD    CPTAC  106      102
======= ====== ======== =====================

Screening and validation
------------------------
Within each training partition, rank X and Y on observed pairs, project both
onto the complete clinical/purity/cohort design, and correlate the residuals.
Use abs(rho) >= 0.10 and BH FDR <= 0.05, as explicitly requested by the user.
BH includes every supplied lncRNA for each cancer/target; untestable pairs
remain p=1. Require at least 12 observed pairs and positive residual degrees
of freedom, computed using the actual design rank. No validation or test
observations enter this screen. This is NOT the full HHGT A1 association:
local lncRNA CNV, methylation-derived leukocyte fraction and macrophage
composition adjustment have not been silently substituted or reproduced.

Five cohort-stratified outer folds contain inner training and validation
partitions. Selection, clinical encodings, X scaling, and learned Y references
are refitted inside their respective training partitions. The validation
criterion gives equal weight to cohorts. Both leave-TCGA-out and
leave-CPTAC-out are evaluated separately. Final alpha ratios use the median
of at least three successful outer-fold tuning results; transfer results do
not tune the model. No clinical test labels fit a normalization reference.

Report foldwise R2, Spearman, covariate-only R2, delta R2 over the covariate-only
baseline, and incremental R2 against its prediction errors. Do not pool raw
Y across folds with different training-fitted references. Nonempty Lasso
coefficients do not by themselves establish predictive value or causality.

Expression and Y harmonization
------------------------------
Restore TCGA full TPM before historical filtering. Intersect measured stable
GENCODE v36 IDs with CPTAC, sum actual TPM for symbol aggregation, then apply
log2(TPM+1). Never zero-fill historically deleted expression. The shared X
contains 16,871 / 16,868 / 16,871 lncRNAs in BRCA / LUAD / COAD. Only
training-local measured/variable features can enter the fitted model.

RNA pathways use a fixed common protein-coding background and raw ssGSEA
(alpha 0.25) matching the internal tied-rank behavior of GSVA 2.4.9. Constant
genes remain in the fixed scoring background. The normalization range is
fitted only on training scores. Missing background measurements remain
unavailable labels. This is a new harmonized version, not unchanged legacy
logCPM or cohort-normalized ssGSEA labels.

TF/PROGENy gene and program standardization, xCell minima, CAF PCA weights,
RNA stemness min/max, EXTEND penalty/min/max and APM Gaussian-GSVA references
are fitted only on training patients. Fixed-formula State components use the
same definitions and measured background for both cohorts. DNArep_Kang uses
explicit gene-wise training median comparisons; it does not reproduce the
R matrix/vector recycling behavior found in the installed implementation.

The cosine ESTIMATE-to-purity conversion on the new RNA background produced
invalid negative values. It is EXCLUDED from shared formal Y and from
covariates. Previously supplied native ESTIMATE purity endpoints remain
separate cohort-qualified targets with their original limitations. Relative
immune/stromal enrichment values are not claimed to be absolute cell fractions.

Non-recomputable native State values are explicitly cohort-qualified rather
than pooled. Protein, phosphorylation and methylation endpoints remain their
observed native assay targets; only patients with that target are trained.
Their upstream published normalization has not been re-fitted inductively.
Exact patient IDs do not overlap between cohorts, but a complete biological
alias crosswalk was unavailable; no stronger non-overlap claim is made.

Entrypoints and outputs
-----------------------
Preparation: prepare_m0_joint_rna.py, cache_m0_joint_scores.py,
prepare_m0_joint_splits.py, materialize_m0_joint_apm.R,
prepare_m0_joint_covariates.py. Scientific execution is restricted to host 149.
Training: run_m0_joint_lasso.py --aligned ... --cache ... --splits ...
--covariates ... --cancer BRCA --output ... --workers 8.
Run BRCA, LUAD, COAD with at most 24 total CPU workers and BLAS threads limited.

Each cancer exports final model objects, FINAL_GENE_SETS, per-target clinical
encodings and coefficients, screening results, fold predictions, per-cohort
and transfer validation, and target-level statuses. Each completed phase has
hashes and can be resumed only with unchanged code/config/input receipts.
The native SC sample-summary table is not a bulk-patient purity covariate:
its 617 SC sample identifiers had zero exact matches to the demo bulk patients.
No mapped single-cell purity artifact was found in the checked current sources.
