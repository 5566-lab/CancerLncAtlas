Graph-regularized ridge extension of the existing M0 demo
======================================================

The existing ``scripts/run_m0_three_arm.py`` entry now accepts
``--estimator graph_ridge --omics-input <verified-input-root>``. Its default
Lasso behavior is retained. Existing patient splits, native outcomes,
clinical covariate encoding, three-arm metrics, paired comparisons and figure
functions are reused. This is a CPTAC BRCA/LUAD/COAD demo, not a new TCGA run.

Model
-----

For one cancer and target, minimize

.. math::

   n^{-1}\|y-C\gamma-\widetilde X_R\beta-
       \sum_{b\in B_y}\widetilde X_b\theta_b\|^2
   +\lambda\{\beta^T(I+\eta L)\beta+\sum_{b\in B_y}\|\theta_b\|^2\}.

``C`` contains the existing unpenalized intercept, age, sex, stage and fixed
DNA-purity covariates. ``X_R`` selects lncRNA, mRNA or their union. ``B_y`` is
the set of target-eligible measured CNV, binary mutation, protein-abundance
and phosphosite blocks. The additional blocks have ridge penalties; the
PPI/RPI graph penalty acts on RNA coefficients. No RNA correlation/FDR
admission or sparsity selection is used.

Each block is standardized using training medians, means and standard
deviations, then divided by the square root of its centered TRAINING kernel
trace per patient, and by the square root of the number of included blocks.
Thus a layer with more measured features does not automatically receive a
larger aggregate kernel weight. The graph-kernel trace is fitted separately
for each arm and graph strength. Predictor preprocessing uses all eligible
training patients, including training patients with a missing particular Y;
test-patient statistics are never fitted. Clinical encoding is reused from
``m0_covariates``. Target fitting and tuning use only observed Y.

``L`` is the normalized Laplacian of the frozen PPI/RPI prior. PPI protein
pairs are projected through unambiguous protein-to-gene mappings. RPI maps
lncRNAs to their partner's encoding gene. Predicted and unknown RPI classes
are excluded; context-specific records must match the cancer. STRING PPI is
an association prior, not a claim that every edge is a direct physical
interaction. Undirected duplicate pairs use maximum confidence, not summed
confidence. No expression-correlation or pathway-membership edges are added.

The full RNA gene axis is retained in the prior. An arm-excluded gene has no
expression column in that arm, but can be a latent coefficient connecting
network paths. Eliminating these latent coefficients gives the corresponding
Schur-complement prior. These nodes supply topology only; no mRNA values enter
the lncRNA arm. Both isolated and connected measured genes remain eligible.
An unsigned Laplacian favors smooth normalized coefficients; it does not
supply activation, inhibition, regulatory direction or causal evidence.

The dual kernel is ``X (I + eta L)^(-1) X.T``. Cached sparse LU solves the same
positive-definite system as the primal estimator. Every solve checks its
relative residual. Tests compare the dual against an explicit primal model
with unpenalized covariates and a latent bridge node.

Validation and comparisons
--------------------------

The original five outer folds and inner holdouts are unchanged. Inner
validation selects ``eta in {0, 0.1, 1, 10}`` and 13 log-spaced ridge penalties
from ``1e-4`` to ``1e2``. Final fits use the modal selected graph strength
(lower on a tie) and nearest-grid median log ridge penalty from outer tuning.
Held-out labels do not select their own model's hyperparameters.

Four variants are evaluated on matched patients:

* RNA-only ridge, three RNA arms;
* RNA plus eligible measured multiomics, ordinary ridge;
* the same multiomics inputs with graph-regularized ridge;
* a shared eligible-multiomics-only baseline, to quantify added RNA information.

Clinical/purity covariates occur in all variants under the same target policy.
Primary results require successful fitting in all five outer folds and the
final fit, supported held-out predictions in all folds, and nonconstant Y.
All unavailable/failed targets remain in the status table. Negative held-out
R2 is reported, not removed. Unlike Lasso, ridge does not produce a sparse
selected gene set; dense coefficients are not evidence of RNA usefulness.
Paired improvement over the shared-omics-only baseline measures that question.

Target and missingness policy
-----------------------------

Protein outcomes exclude the entire protein input block. Phosphosite outcomes
exclude the entire phosphosite input block. Mutation-derived State outcomes
exclude mutation; ploidy excludes CNV; HRD excludes both. DNA purity excludes
its own covariate and both genomic blocks. Methylation is an outcome here,
not an added predictor block.

RNA-derived outcomes retain lncRNA expression predictors. The pathway scorer selects coding
gene symbols, but its upstream symbol matrix sums same-symbol loci; some
lncRNA and coding loci share symbols (for example RGS5), so even that filter
is not a proof of zero lncRNA contribution. TF and some State implementations
accept all RNA. Constituent and ranking-background overlap are audited
instead of assuming all RNA-derived scores are lncRNA-independent.
mRNA and combined arms can reconstruct the underlying score definition.
None of these RNA comparisons establishes protein activation or causality.
Native Y and upstream cohort preprocessing are unchanged; this remains a
retrospective benchmark, not fully inductive validation of score generation.

Only training columns with at least two finite measurements and nonzero
variance are retained. Missing feature measurements use training medians;
each measured-omics block also includes a whole-assay availability indicator.
A missing mutation assay is not filled as wild type. Unmatched patient IDs
remain missing; an R-style X prefix or dot-for-hyphen spelling is accepted
only when it maps uniquely to an existing authoritative patient ID.

Existing and new data
---------------------

Existing TCGA gene-CNV matrices under ``gene_node_omics_20261029_r1`` and MC3
mutation products are retained. Their TCGA patient axis cannot substitute for
CPTAC measurements. Existing CPTAC RNA, protein, phosphosite and outcome
matrices are reused by checksum. Six gene-level CNV/mutation files were
retrieved from the official LinkedOmics CPTAC BRCA, LUAD and COAD pages;
source URLs, SHA256 and unmatched patient IDs are retained in the run receipt.
The directory inventory was limited to recorded roots/depth; absence from
that inventory is not proof no historical copy exists anywhere.

References
----------

* https://www.linkedomics.org/data_download/CPTAC-BRCA/
* https://www.linkedomics.org/data_download/CPTAC-LUAD/
* https://www.linkedomics.org/data_download/CPTAC-COAD/
* https://pmc.ncbi.nlm.nih.gov/articles/PMC3423227/
