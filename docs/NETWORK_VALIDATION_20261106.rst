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
