.. warning::

   HISTORICAL RUN ONLY. MOGSA and the derived parent-scoring pipeline were retired
   on 2026-11-04. Scores and prepared artifacts described below are not authorized
   targets for new training. Commands below document the old run and are no
   longer executable in the current code. See PATHWAY_TARGET_POLICY_20261104.rst.

Five-assay pathway scoring and source-patient fitting
====================================================

The current requested design adds gene CNV and promoter methylation to RNA,
measured protein and phosphorylation. All five assays enter MOGSA. This addition
does not replace protein/phosphorylation, disable gene-node omics, or change the
joint task weights: Pathway=1.0 and State=0.5. Both heads update the shared encoder.

Scores and model labels are different quantities. S(patient,pathway) is a
continuous patient score. The HHGT target Y(cancer,lncRNA,pathway) records an
adjusted association across a separate label-patient cohort. Neither is a causal
intervention effect. CNV/methylation in both score construction and graph inputs
can create biological dependence even when the patients are disjoint; an
independent assay or perturbation evaluation is still needed for biological claims.

Input identities and missing measurements
----------------------------------------

RNA is measured protein-coding gene logCPM. CNV is the verified gene-midpoint
signed log2 copy ratio. Methylation is verified promoter beta for TSS -1000/+100.
Protein and phosphorylation are targeted TCGA RPPA measurements, not whole-proteome
mass spectrometry. Phosphorylation is not adjusted for total protein abundance and
must not be described as measured kinase activity.

The RPPA input has 7,790 aliquots and 198 features. Exact normalized antibody
annotations map 144 protein and 49 phosphorylation features. Three unresolved
features and two non-phosphorylation modifications are excluded. Mapped Caution
and UnderEvaluation annotations retain their original status; inclusion is not a
claim that every antibody has Valid status. A multi-gene antibody remains one
feature and annotates a pathway only when all possible genes belong to that set.

Match the frozen RNA patient and sample/vial, require exact aliquot quality
annotations, and exclude official pathology/Do_not_use records. CORE resolves to
COAD or READ only when the exact official aliquot cancer agrees with RNA authority.
Multiple qualifying aliquots for one patient are unavailable, not averaged.
There are 10,432 RNA patients across 33 cancers, 7,088 matched quality-passed RPPA
patients, and 5,649 patients with at least one observed feature in every assay
across 32 cancers. These are input-presence counts, not final score availability.

Source files are the GDC PanCanAtlas RPPA table and GDC antibody reference:

* https://gdc.cancer.gov/about-data/publications/pancanatlas
* https://api.gdc.cancer.gov/data/fcbb373e-28d4-4818-92f3-601ede3da5e1
* https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files
* https://api.gdc.cancer.gov/v0/data/62647302-b4d3-4a81-a7c0-d141f5dbd300

Original missing values and masks remain missing. There is no 95% per-assay
coverage threshold and no filling of raw measurements. The current fit requires
at least 20 paired source patients, at least 10 observed features per block per
fit patient, and at least 10 complete variable features retained per block.
These are explicit implementation eligibility settings, not MOGSA package
requirements or demonstrated biological thresholds. LAML has no five-assay pair;
OV has 9 and UVM has 12, so they cannot pass this current fit configuration.

RNA consensus and native multi-omics scores
-----------------------------------------

The descriptive RNA consensus is

  R_ip = (z_p(ssGSEA_ip) + z_p(singscore_ip) + z_p(GSVA_ip)) / 3.

The z transformation is within each cancer and pathway. ssGSEA uses
normalize=FALSE; GSVA uses Gaussian kcdf; singscore uses a single positive gene
set. The three methods share RNA/gene sets, so agreement is not independent
biological validation. Existing gene sets do not acquire activating/inhibiting
subsets automatically. Eligible sets have at least 10 measured genes and at most
500 original members. Program scores such as existing PROGENY/DoRothEA resource
scores move unchanged to State. Their existing implementation is weighted gene
z-scores followed by program z-scores, not native PROGENy/VIPER package output.

MOGSA consumes the five raw feature-by-patient blocks, not the three RNA score
matrices. The RNA consensus is a descriptive comparator and does not replace the
RNA assay or become a global-fitted prepared label. This avoids counting several
transformations of the same RNA as separate molecular assays.

Native mogsa::moa uses proc.row='center_ssq1', w.data='lambda1', statis=FALSE.
mogsa::mogsa uses up to 10 components, limited by fitted rank and patient count.
The native score reconstruction is numerically checked. Per-assay contributions
sum to the MOGSA score and are saved separately. These are package score
components, not causal feature attributions. PATHWAY_ASSAY_COVERAGE.tsv records
annotated measured features in each layer; five input blocks do not imply five
directly annotated measurements for every pathway.

Prepared scoring fits feature selection, centering, scaling, loadings and pathway
coordinates only on the exact source-patient cohort for that view. Validation and
label patients do not fit these parameters. Whole-cohort descriptive scores are
not accepted as prepared targets. Cache identity binds input hashes, patient axes,
source patients, code, catalog, component count and fitting scope.

For complete vectors, the frozen projection equals native MOGSA. For partial
vectors, estimate factors from observed entries O using the fixed fitted loadings
Q and weights A:

  f_i = (Q_O' A_O Q_O)^(-1) Q_O' A_O x_i,O ;   M_ip = C_p f_i.

Every requested assay must contribute an observed retained feature and the
weighted observed design must have numerical full rank. No fixed coverage
percentage is applied. This observed-feature least-squares projection is an
explicit extension, not a native MOGSA missing-data prediction API. It estimates
a latent score, not missing raw measurements. The numerical condition, observed
counts and residual norm are saved. An entirely missing assay or singular design
produces unavailable scores; rank sufficiency does not establish prediction
accuracy for sparse observations.

Parent pathways
---------------

The reviewed catalog contains 2,547 gene sets and 237 candidate Reactome parents
with at least three directly verified child pathways. Parent identity comes from
the Reactome hierarchy, not post-hoc lncRNA correlations. lavaan fits one factor
to standardized child MOGSA scores on the same source cohort:

  z(M_ij) = lambda_j F_i,parent + epsilon_ij.

Use MLR, std.lv=TRUE, at least max(30,3*children) source patients, full-rank child
scores, convergence, nonnegative variances, consistently oriented standardized
loadings >=0.3 and SRMR<=0.08. A three-child model is just identified and explicitly
not globally fit-validated. No jitter, loading truncation or forced parent scores.
Supported parent IDs use lavaan regression factor scores; unsupported registered
parent IDs are unavailable. Other pathways keep native MOGSA scores. Child IDs
remain available. The final S is therefore a parent factor or a native pathway
score, not their arithmetic sum. Acceptance settings are explicit modeling choices.

Association Y and joint loss
----------------------------

Association statistics use finite common patients, the existing covariate contract
and original callable local lncRNA CNV. At least 12 patients and residual degrees
of freedom are required. In a label cohort, let Z contain admitted covariates and
M_Z remove their linear effects. Let c be local signed lncRNA CNV:

  u = M_Z rank(lncRNA) - beta_hat M_Z c,
  v = M_Z rank(S),
  r = cor(u,v).

The pre-existing local CNV adjustment acts on lncRNA, not on the pathway score.
BH q values use the full tested cancer/task family. Pathway strong positives have
abs(r)>=0.20 and q<=0.05. Weak positives have abs(r)>=0.15 and q<=0.10, excluding
strong positives, with half weight. Y_Pathway is strong OR weak. State uses
Y_State=1[abs(r)>=0.20] without this FDR criterion. Unavailable labels do not enter
the loss; a serialized zero does not turn missing data into negative evidence.
Direction is 1[r>=0] for available pairs with abs(r)>=0.15.

For each task t, the implemented loss is

  L_t = sum_available(w_i BCE(y_i,sigmoid(b_i+alpha*delta_i))) / sum_available(w_i)
        + 0.25 mean_direction(BCE_direction)
        + 1e-4 mean_available(delta_i^2),
  L = L_Pathway + 0.5 L_State + R_gate.

b is the matched LASSO baseline logit; delta is the bounded graph residual.
R_gate is 1e-5 times the selected clamped cancer/global modality-gate means.
State weights balance target families. AdamW weight decay 1e-5 is an optimizer
setting, separate from this explicit loss. Frozen score preprocessors do not
freeze the shared encoder or State task head.

Reproduction on host 149
------------------------

All scientific export, scoring, preparation and tests run on host 149. The
configuration defaults to cancer_type=all; cancer_type=list plus cancer_list
selects a demo. No paid instance is needed for this CPU scoring workflow.

1. Run scripts/export_pathway_gene_inputs_149.py with its verified config, RNA
   root, gene dimension, Reactome hierarchy/xrefs and output arguments. This is
   only an intermediate RNA/CNV/methylation product, not a five-assay score input.
2. Run scripts/export_pathway_fusion_inputs_149.py with --gene-inputs, --rppa,
   --antibody-annotation, --sample-quality, --patient-map, their corresponding
   --*-sha256 bindings, and a new --output directory. Use --cancer-type all or
   --cancer-type list --cancer-list BRCA LUAD.
3. Bind the resulting COMPLETE.json path/hash, pinned R libraries and fresh cache
   in pathway_fusion. The required format is
   CLA_SOURCE_FITTED_FIVE_ASSAY_PATHWAY_FUSION_V2. Bind runtime controller/imports
   to the current code digest, then use the ordinary cla.py prepare workflow.
4. Optional descriptive scores use scripts/score_pathway_fusion_149.py with
   --raw-inputs, --output, --r-libraries and --workers. Never use their global
   matrices as prepared labels.

Pinned versions are mogsa 1.44.0, lavaan 0.7.3, GSVA 2.4.9 and singscore 1.30.0.
Tests cover exact antibody matching, sample/vial/QC ambiguity, missing assays,
native reconstruction, heldout-fit invariance, observed projection, parent factor
recovery and prepared lineage. Actual availability and integration results must
come from run receipts; package checks alone are not trained-model evidence.


Executed validation results
---------------------------

The aggregate provenance is recorded in
provenance/joint_mainline_20261101/FIVE_ASSAY_SCORING_VALIDATION.json.
All 33 cancers were processed. RNA consensus scores cover 10432 patients;
five-assay scores cover 5628 patients in 30 cancers. There are 508
supported cancer/parent models. Missing or unsupported scores remain unavailable.
The BRCA/LUAD real-data diagnostic passed formal prepared validation with 8
source/label views, 482 primary State targets, six-column node inputs and unchanged
Pathway/State weights 1.0/0.5. A new HHGT model has not been trained.

The joint regression suite passed 487 tests with 1 skip; after the
target-contract update, 59 focused tests passed. The final script-snapshot
suite passed 15 tests. These suites overlap and must not be summed as
independent tests. Native MOGSA/projection/lavaan numerical tests also passed.

.. list-table:: Final patient and parent coverage
   :header-rows: 1

   * - Cancer
     - RNA patients
     - Five-assay input pairs
     - Patients with final scores
     - Supported parents
   * - ACC
     - 79
     - 44
     - 44
     - 16
   * - BLCA
     - 404
     - 334
     - 334
     - 16
   * - BRCA
     - 1090
     - 624
     - 624
     - 23
   * - CESC
     - 304
     - 157
     - 157
     - 13
   * - CHOL
     - 35
     - 28
     - 28
     - 0
   * - COAD
     - 454
     - 234
     - 234
     - 18
   * - DLBC
     - 48
     - 33
     - 33
     - 15
   * - ESCA
     - 184
     - 119
     - 119
     - 13
   * - GBM
     - 252
     - 51
     - 51
     - 21
   * - HNSC
     - 520
     - 196
     - 196
     - 12
   * - KICH
     - 66
     - 62
     - 62
     - 14
   * - KIRC
     - 531
     - 276
     - 276
     - 16
   * - KIRP
     - 290
     - 206
     - 206
     - 19
   * - LAML
     - 151
     - 0
     - 0
     - 0
   * - LGG
     - 516
     - 424
     - 424
     - 13
   * - LIHC
     - 371
     - 177
     - 177
     - 17
   * - LUAD
     - 509
     - 310
     - 310
     - 19
   * - LUSC
     - 499
     - 251
     - 251
     - 18
   * - MESO
     - 87
     - 59
     - 59
     - 14
   * - OV
     - 427
     - 9
     - 0
     - 0
   * - PAAD
     - 178
     - 100
     - 100
     - 15
   * - PCPG
     - 179
     - 79
     - 79
     - 20
   * - PRAD
     - 496
     - 293
     - 293
     - 24
   * - READ
     - 166
     - 71
     - 71
     - 17
   * - SARC
     - 259
     - 185
     - 185
     - 21
   * - SKCM
     - 469
     - 91
     - 91
     - 17
   * - STAD
     - 412
     - 287
     - 287
     - 25
   * - TGCT
     - 150
     - 118
     - 118
     - 18
   * - THCA
     - 505
     - 363
     - 363
     - 18
   * - THYM
     - 120
     - 87
     - 87
     - 11
   * - UCEC
     - 544
     - 322
     - 322
     - 22
   * - UCS
     - 57
     - 47
     - 47
     - 23
   * - UVM
     - 80
     - 12
     - 0
     - 0
