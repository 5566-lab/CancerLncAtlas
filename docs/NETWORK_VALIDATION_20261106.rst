PPI/RPI network validation: prespecified three-cancer experiment
==============================================================

This extension checks whether the earlier near-zero incremental graph result
comes from its modeling assumptions, tuning instability, or lack of useful
network information for these patient outcomes. It retains the existing raw
inputs, native targets, clinical adjustment and three RNA arms. It does not
redefine the outcome as a network gene-set score.

Execution and provenance
------------------------

All preparation, validation, fitting and reporting run on host ``149``. The
experiment is specified in ``config/network_validation_20261106.json`` before
new fits. Input checksums bind the existing multiomics preparation; typed
network authority and unambiguous gene mappings are verified independently.
The first mapping draft is retained as a diagnostic artifact. Revision 2
correctly attaches lncRNA gene CNV/mutation features to the lncRNA anchor.
An evidence audit then found a substantive admission error: expression/abundance
and reporter evidence had been accepted as physical RPI via the
experimental_unspecified class. The provisional run was stopped and invalidated
before its held-out predictions were inspected. Network-control revision 3
requires an explicit physical method and physical_binding evidence family.
The corrected run uses unchanged patient measurements, targets and splits.
Its prespecified amendment is recorded in the experiment configuration.

CPTAC BRCA, LUAD and COAD have 121, 110 and 106 input patients, respectively.
Each prepared target table contains 8,750 candidate outcomes, including
unavailable outcomes. A candidate is not claimed as successfully trained merely
because its row exists. Every inner fit needs at least 12 training observations
and every inner validation needs at least 3. Complete evaluation requires all
five outer folds and the final fit; failed targets and negative R2 remain in
the audit. The report records the resulting eligible population.

The original M0/Lasso functions and graph-ridge solver remain intact.
The graph-ridge launch entry now rejects historical input receipts without
the explicit physical-evidence gate. ``scripts/run_m0_three_arm.py`` adds an explicit
``--estimator network_validation`` entry point; its default remains Lasso.

Why test another constraint?
----------------------------

The legacy Laplacian acts on RNA coefficients, including RNA proxies for PPI
protein nodes. It softly penalizes differences between connected coefficients.
Protein association or RNA-protein binding alone does not establish matching regression signs; those
signs also depend on correlated predictors and are not causal edge signs.
The legacy penalty permits opposite signs but penalizes them. Added patient
omics blocks are ordinary ridge blocks in that implementation.

The new model uses biological network communities to share regularization
strength across measured RNA, CNV, mutation, protein and phosphosite features.
Every feature retains its own coefficient, and opposite coefficient signs
inside a module receive no additional penalty. The model is network-module
ridge, not a signed Laplacian, aGrace, or a causal model.

PPI-only, RPI-only and combined partitions are generated from the fixed network
prior. No outcome or held-out expression enters topology construction. Features
without an unambiguous mapping or connection remain in modality-specific
background groups. Phosphosite features keep their site identities; mapping
them to the parent gene anchor does not merge their measured values.

Kernel and fitting definition
-----------------------------

For every training split, fit feature quality checks, missing-value medians,
scaling and clinical encodings on that training population. Protein and
phosphosite values retain the prior training-range clipping contract. No
correlation or FDR admission threshold is applied. Each allowed assay block has
unit centered training kernel trace and equal prior block mass.

Let the equal-modality baseline kernel be the sum of raw group kernels G_g.
Define m_g as the centered training trace of G_g and K_g = G_g / m_g, so that
sum_g m_g = 1 and sum_g m_g K_g equals the baseline kernel exactly. Clinical
covariates Z are unpenalized; M projects off their training column space.

.. math::

   a_g = \frac{y_r^T (M K_{g,tt} M)y_r}
                    {\|M K_{g,tt} M\|_F\,\|y_r\|_2^2},\qquad y_r=My_t

   \alpha_g(p) = \frac{m_g(a_g+10^{-12})^p}
                          {\sum_h m_h(a_h+10^{-12})^p},\qquad
   K_\alpha=\sum_g\alpha_g K_g.

The alignment power p is tuned inside each outer training set. Power zero
recovers equal-modality ridge. The adaptive-omics control uses exactly the same
weighting and tuning procedure with assay blocks instead of network modules.
For target-observed training rows, dual ridge solves the residualized kernel
with penalty n*lambda; the clinical coefficient is then recovered without
penalization. There is no supervised fitting on outer test rows.

``cc_hhgt/network_ridge_predict.py`` reconstructs standardized feature
coefficients and predicts new rows from frozen model, preprocessing, feature
mapping and input-reference artifacts. Prediction does not read outcomes or
refit any parameter. Ridge coefficients are dense; they are not a Lasso-selected
fixed gene set. Protein/phosphosite clipping remains part of the input transform.

Comparisons and validation
--------------------------

* Two outer repetitions of five folds. Repeat 0 exactly reuses the original
  authoritative outer folds; repeat 1 uses seed 20261106.
* Each outer fit and final fit uses two repetitions of three-fold inner tuning.
* Lambda grid: 17 logarithmic values from 1e-5 to 1e3.
* Legacy Laplacian eta: 0, 0.01, 0.1, 1, 10, 100.
* Group alignment power: 0, 0.5, 1, 2.
* RNA-only, shared omics-only, equal-omics ridge and adaptive-omics ridge controls.
* PPI, RPI, union and five null unions for both legacy and module models.
* All three RNA arms share identical eligible omics and clinical covariates.

Null graphs use independent degree-preserving swaps within PPI and bipartite
RPI. Unweighted node degrees and the edge-weight multiset are preserved;
weighted node strengths and community sizes are not. Five null seeds are a
mechanistic diagnostic, not an automatic permutation p-value. The corrected RNA projection enforces coding-to-coding PPI and
lncRNA-to-coding RPI. The typed network uses biological node classes.
The original contaminated graph remains a historical diagnostic and is not
the network used by any corrected legacy or module fit.

Primary endpoints are protein and phosphosite scores. The primary contrasts
are union-module ridge versus adaptive-omics ridge and versus the five null
module models. RNA-derived scores are secondary reconstruction endpoints.
The report shows both outer repeats, paired target changes, and patient-level
bootstrap intervals. Bootstrap samples patients within cancer, preserves their
paired outcomes and both repeats, averages targets within score families, and
weights families and cancers equally. Its intervals condition on already fitted
CV models; it is not a full retraining bootstrap or external validation.

Target/source exclusions
------------------------

Protein outcomes exclude every protein input. Phosphosite outcomes exclude every
phosphosite input. Mutation/CNV-derived State targets retain the existing
source-specific exclusions. DNA purity outcomes exclude the purity covariate
and both genomic input blocks. RNA-derived targets retain lncRNA as authorized;
mRNA-containing models can reconstruct their own score constituents. Existing
lncRNA overlap audits remain applicable. No outcome is imputed.

Native upstream outcome scoring keeps its original cohort preprocessing. These
scores are relative assay-derived endpoints, not absolute pathway activation.
Repeated nested CV protects the newly fitted model stages, but does not turn
retrospectively prepared native outcomes into prospective external validation.

Reproduction
------------

Prepare verified typed controls on 149::

  python scripts/prepare_network_validation.py \
    --inputs /dell_2/DSC/CancerLncAtlas/inputs/graph_ridge_20261105_r1 \
    --output /dell_2/DSC/CancerLncAtlas/inputs/network_validation_20261106_r3 \
    --cancer BRCA

Use ``scripts/materialize_verified_graph_inputs.py`` to reuse unchanged assay
files under ``inputs/graph_ridge_20261106_verified_r1`` with the corrected
network envelope; source assays are linked, never overwritten.

Run each of repeat0_outer0 through repeat1_outer4 and final, for all three
cancers. The existing linear-model entry point is::

  python scripts/run_m0_three_arm.py --estimator network_validation \
    --omics-input /dell_2/DSC/CancerLncAtlas/inputs/graph_ridge_20261106_verified_r1 \
    --network-controls /dell_2/DSC/CancerLncAtlas/inputs/network_validation_20261106_r3 \
    --output /dell_2/DSC/CancerLncAtlas/results/model/NETWORK_VALIDATION_demo_20261106_r2 \
    --cancer BRCA --phase repeat0_outer0

After every phase completes, run ``scripts/summarize_network_validation.py``,
``scripts/bootstrap_network_validation.py`` and ``scripts/plot_network_validation.py``.
The final completion audit and returned figure manifest bind their outputs.

Relevant methodological background
---------------------------------

The rationale for questioning unsigned coefficient smoothing is consistent with
work distinguishing ordinary graph-constrained regression and adaptive/sign-aware
penalties. This implementation uses module weighting and does not claim to
implement those papers' adaptive-Laplacian algorithms:

* https://pmc.ncbi.nlm.nih.gov/articles/PMC3423227/
* https://pmc.ncbi.nlm.nih.gov/articles/PMC3217586/


Independent diagnostic findings (patient fits still pending)
------------------------------------------------------------

The corrected typed RPI graphs contain 77,083 / 73,390 / 75,304 edges for
BRCA / LUAD / COAD. Their connected lncRNA counts are 3,523 / 3,435 / 3,615,
out of about 16,870 candidate lncRNAs before training-only quality checks.
These are topology coverage counts, not successfully trained gene sets.
Degree-preserving RPI nulls retain 43.4--45.8 percent of original edges;
PPI nulls retain about 2.7 percent. Interpretation must retain this difference
in the strength of the topology perturbation. Equal degree does not preserve
weighted node strengths or community sizes. A continuation of one null chain
per cancer from five to ten and twenty successful swaps per edge leaves
original-edge overlap nearly unchanged (BRCA 45.5 percent, LUAD 44.1 percent,
COAD 43.5 percent at twenty). This supports stability of this overlap
diagnostic; it is not a proof of full Markov-chain convergence.

An independent synthetic diagnostic checks whether the implemented module
weighting can exploit a deliberately informative grouping. Ten fixed seeds use
100 training samples, 200 independent test samples and 400 features in twenty
modules. All parameters use the same nested training-only grids as the patient
model; the independent test outcomes do not select parameters. Both a correct
partition and a size-preserving shuffled partition are supplied directly. This
checks the weighting estimator, not biological edge validity or the community
construction algorithm.

When the signal is concentrated in one module, mean test R2 is 0.802 for the
correct grouping, 0.204 for equal ridge and 0.225 for shuffled grouping. With
opposite coefficient signs in that module, the corresponding values are
0.820, 0.206 and 0.208. Correct grouping improves over both controls in all ten
seeds for these two settings. When signal is dispersed over every module,
mean R2 is 0.243 / 0.244 / 0.243: the supplied grouping has no useful advantage.
The no-feature-signal setting retains an unpenalized predictive clinical
covariate, so its R2 near 0.48 is not an RNA prediction gain.

The full per-seed results, fixed plan and bound core hashes are in
``provenance/network_validation_20261106/diagnostics``. Reproduce on host 149
with the pinned Python environment and the repository on PYTHONPATH::

  python scripts/diagnose_network_prior.py --output /NEW/DIAGNOSTIC_DIRECTORY

These simulated results demonstrate sensitivity under the stated data
construction. They do not establish that physical PPI/RPI improves the actual
patient outcomes. That conclusion still requires all prespecified patient
phases, failure/coverage audits, paired comparisons and uncertainty intervals.


Matched RNA increment and interval interpretation
-------------------------------------------------

The prespecified RNA-increment comparison uses equal-modality ridge with each
RNA arm versus the shared omics-only ridge, retaining clinical covariates and
target-specific assay exclusions. Comparing module ridge with plain omics-only
ridge would also change regularization, so that difference is not attributed
solely to RNA. The supplemental reporter verifies every prediction phase for
exact patient and observed-target pairing, then exports per-target and
per-cancer results plus figure 11. All three RNA arms are retained.

The patient bootstrap holds each target's original common-support variance
fixed and resamples paired prediction losses. Its intervals are expressed in
Delta R2 units. It neither refits CV models nor recomputes target variance for
each bootstrap replicate. This restriction accompanies the existing statement
that the intervals are conditional on the fitted models.

Saved final-model verification
------------------------------

After a cancer's final phase is complete, run
``scripts/audit_network_final_models.py`` with the verified input, network,
result and new audit-output directories. It selects the first lexicographic
fully fitted target per layer and purity-covariate policy, without consulting
prediction performance. For each selected target it checks all three RNA arms
and the shared omics-only control across base, adaptive-omics, PPI, RPI, union
and one null-module estimator.

The audit verifies bound artifact hashes, reconstructs frozen preprocessing,
compares exported-coefficient predictions with the training kernel and an
independent normal-equation solve, permutes query rows, and checks rejection of
missing required clinical data. It covers the base/module prediction helper;
legacy Laplacian models are outside this helper's export contract. This is a
bounded saved-artifact check, not independent external prediction validation.
A four-target synthetic fixture checks 88 saved models, both purity policies
and rejection of a checksum change. Real-patient audit receipts must be reviewed
separately before claiming the final artifacts have passed.

Resolution-four follow-up (structural amendment)
------------------------------------------------

A graph-only audit found that the resolution-one union assigns 99.43%, 99.71%
and 99.39% of connected BRCA, LUAD and COAD lncRNA anchors to one community.
Their coefficients remain independent, but almost all connected lncRNAs share
one network shrinkage multiplier. Unconnected lncRNAs remain in the separate
background group. Thus this representation supplies coarse lncRNA information.
Degree-preserving nulls also change community counts and sizes; edge overlap
alone does not describe how much the fitted grouping changes.

A subsequent topology-only probe used resolutions 1, 2, 4 and 8 with seed 1979.
Resolution four is the smallest tested value for which the largest lncRNA group
contains at most 25% of connected lncRNAs in every cancer. This structural
criterion was set after inspecting the topology probe, without inspecting
patient predictive metrics. Resolution four yields 84/86/88 lncRNA-containing
communities and largest fractions 21.71%/18.08%/17.15% for BRCA/LUAD/COAD.
This is an exploratory, structurally motivated amendment, not a claim that the
original experiment prespecified this choice or that finer modules predict better.

The follow-up retains all original candidate targets, patients, three RNA arms,
clinical adjustment, target-source exclusions, two-by-five outer splits and
two-by-three inner tuning. The ridge and weighting equations are unchanged.
It fits the union and the same five degree-preserving null graphs, all partitioned
at resolution four. The original experiment remains a complete coarse-module
and Laplacian comparison. Coarse, fine, adaptive-omics and equal-ridge methods
must be compared on matching patient/target support before a performance claim.

Prepare separate network inputs on host149::

  python scripts/prepare_network_resolution4.py \\
    --networks /PATH/TO/VERIFIED_NETWORK_R3 \\
    --output /NEW/RESOLUTION4_INPUTS

The existing three-arm entry accepts the additional estimator::

  python scripts/run_m0_three_arm.py --estimator network_resolution4 \\
    --omics-input /PATH/TO/VERIFIED_PATIENT_INPUTS \\
    --network-controls /PATH/TO/RESOLUTION4_INPUTS \\
    --output /NEW/RESOLUTION4_RESULTS --cancer BRCA --phase repeat0_outer0

The follow-up runner validates the resolution-four provenance and reuses the
frozen core solver. Its synthetic regression test compares it numerically with
the original module solver on identical partitions, checks the exact zero-power
ridge fallback, verifies 72 saved models, and rejects resolution-one provenance.
Actual patient performance and final artifacts remain pending until the new
33 phases and their independent audits complete.

Resolution-four comparative report
----------------------------------

After both experiments and their independent audits complete, the
scripts/report_network_resolution_validation.py reporter compares the fine union
with the original coarse union, adaptive-omics ridge, equal ridge, the legacy
RNA Laplacian and five fine random-network controls. It binds the original
summary and phase files, checks identical input receipts, shared core code and
split definitions, and pairs every prediction with the original equal-ridge
patient's observed label. A missing patient, changed label or duplicate
prediction rejects the comparison.

It exports all candidate training statuses, successful and failed target
counts, actual observed/evaluated patient coverage, selected hyperparameters,
per-cancer/per-repeat paired differences and three-arm comparisons. Target
evaluation requires five outer fits and the final fit; finite negative R2 is
retained. Six PNG/PDF figure pairs include one figure per RNA arm, arm
differences, fine-versus-coarse/adaptive differences, and primary protein/phospho
patient-bootstrap intervals.

The primary bootstrap uses the same patient support across both repeats and
all compared methods. It holds the original common-support target variance
fixed, uses equal score-family and cancer weights, and averages the errors of
five null models rather than their predictions. Intervals are conditional on
the fitted models. All-layer figures average only targets available in both
outer repetitions. The report preserves the exploratory status of the
topology-based resolution change and the retrospective, platform-native label
definition; it makes no TCPA-CPTAC calibration claim.

Synthetic report tests verify an independently calculated R2, model difference
and five-null error contrast, retain unavailable and failed targets and
negative R2, reject incorrect pairing, and check all six figure pairs. These
test outputs are separate from actual patient results. A generated report is
explicitly marked as requiring scientific and visual review.

Vectorized frozen-model export
------------------------------

The optional cc_hhgt.network_ridge_predict_fast API exports the same coefficients
as the frozen predictor without constructing a pandas row for every feature.
Its compile_model function stores coefficients, feature order, nuisance
coefficients and frozen preprocessing once; predict_compiled reuses them for
subsequent query batches. The compiled artifact contains fitted parameters,
not reference-patient rows. This changes export and prediction efficiency,
not model fitting, network weights, target definitions or validation splits.

The regression test compares 272 saved base/module models with the frozen API,
including missing assays, availability indicators, dropped constant features,
background groups, query reordering and missing clinical data. It also checks
88 models through the independent normal-equation artifact audit using cached
compiled exports. Legacy Laplacian models retain their separate prediction
contract.

One completed BRCA union model was additionally checked on actual inputs:
92,833 exported coefficients and all 109 supported predictions were identical
between implementations. Compilation took approximately 1.46 seconds and the
compiled full-cohort prediction took 2.08 seconds in that run. The original
export took 85.25 seconds under cProfile, mainly in repeated pandas row access;
profiling overhead means these timings are not a controlled speedup ratio.
This bounded check demonstrates export equivalence, not heldout predictive
performance or universal runtime.

Run scripts/audit_network_compiled_models.py with the same input, network and
result directories and a new output directory. Use --resolution4 for the six
fine-module variants. An optional --validation-receipt binds the tested export
implementation. The audit retains the original target-selection and independent
kernel/refit checks, writes explicit export-code bindings, and compiles each
checked model once. Operational audit supervisors must monitor live progress
without applying a fixed short deadline to an otherwise productive audit.


Complete-repeat plot correction
-------------------------------

Review of the completed original experiment found 27 cancer-target selections
that completed only one of the two outer-CV repeats. The original figure 05
network-ablation averaging included 243 such model-target combinations, and
figure 07 included 756. Its caption described a two-repeat mean, so these two
figures require a complete-repeat restriction. The per-repeat prediction
metrics and failure tables are correct and remain unchanged.

After the frozen original report generator finishes, run the reviewed
two-figure producer on host149::

  python scripts/plot_network_validation_complete_repeats.py \
    --summary /PATH/TO/ORIGINAL_REPORT \
    --output /NEW/REVIEWED_TWO_FIGURE_DIRECTORY

Use its figure 05 and figure 07 PNG/PDF exports for the reviewed report.
The command verifies source checksums, rejects duplicate/unknown repeats,
requires both repeats 0 and 1, and preserves finite negative R2. It writes
the exact plotted values, heatmap cells and excluded singleton counts.
The original frozen generator is retained for the running experiment's source
bindings; the complete-repeat producer is a required report-review step.

The regression test includes incomplete repeats, failed targets and negative
scores, verifies both PNG/PDF pairs and exact plotted means, rejects invalid
repeat identities and source checksum drift, and leaves raw metric files
unchanged. On actual original results, an independent pivot reproduced all
146,804 figure-07 and 47,187 figure-05 plotted means exactly. All 37,715
negative figure-07 model-target means were retained. These are model-target
counts, not patient counts.

Aggregate correction/test evidence is in
provenance/network_validation_20261106/repeat_plot_correction.
This correction changes visual aggregation only. It neither refits models nor
establishes predictive benefit; the fine-module comparison still requires its
full experiment and review.


Reviewed original experiment results
------------------------------------

The original 33-phase experiment is complete. This subsection reports its
results only; the resolution-four follow-up remains pending, so the full
network-improvement goal is not yet complete. Inputs and native outcomes are
CPTAC BRCA, LUAD and COAD, not calibrated TCPA-CPTAC pooled labels.

Primary comparisons use 4,325 protein and 918 phospho cancer-target selections
complete in both outer repeats and all registered primary controls. With the
combined RNA arm, coarse union-module ridge versus adaptive-omics ridge has
family/cancer-balanced mean paired Delta R2 of -0.04825 for protein
(95% patient-bootstrap interval -0.05188 to -0.04488) and -0.02064 for phospho
(-0.02318 to -0.01815). It improves over equal-omics ridge by +0.04153 and
+0.01995, respectively. Thus beating the equal-weight baseline does not
establish an advantage over the stronger non-network comparator.

Relative to the mean squared error of five null-network models, its changes
are +0.00503 for protein and +0.00065 for phospho. These diagnostic controls
do not fix community sizes or weighted node strengths and are not a formal
five-permutation significance test.

Adding lncRNA to the mRNA arm of coarse union-module ridge gives protein
Delta R2 +0.00048 (interval -0.00174 to +0.00280) and phospho +0.00330
(+0.00182 to +0.00479). The corresponding adaptive-omics changes are
-0.00794 and +0.00264. These results do not support a uniformly positive
increment from lncRNA. Bootstrap intervals condition on fitted CV models
and fixed common-support target variance; models are not refitted.

A numerical review independently reconstructed all 1,599,118 successful finite
metric rows from their sufficient statistics, retaining 312,578 negative R2
rows. All 144 primary contrast points were reproduced with maximum absolute
difference 1.46e-16. Candidate training status retains 3,045,000 records:
3 cancers x 2 repeats x 8,750 targets x 58 model-arm attempts. These are
model-target records, not independent patients. All eleven original PNGs and
their eleven rendered PDF previews were visually inspected.

Figure 09 is a literal zero-parameter-selection count. Its complement must
not be interpreted as effective graph use. All lncRNA rows of the PPI-only
RNA Laplacian are exactly zero in these prepared networks. Across 27,571
successful cancer-target-repeat pairs, the lncRNA PPI-Laplacian arm and equal
ridge differ in R2 by at most 1.57e-12. Nonzero parameter choices in this
degenerate case do not imply predictive graph influence. PPI module ridge
can still reweight shared non-RNA feature groups and is a different estimator.

RNA-derived scores in mRNA-containing arms remain score reconstruction;
upstream native cohort preprocessing is unchanged. Phospho scores have not
been adjusted for total-protein abundance. Methylation outcomes in this
demo are available only in LUAD and have negative median held-out R2.
The small available non-RNA State subset does not establish performance for
the full State catalog. These are internal retrospective comparisons,
not external validation, absolute pathway activation or causal lncRNA effects.

Aggregate original-scope review evidence is in
provenance/network_validation_20261106/original_review.
