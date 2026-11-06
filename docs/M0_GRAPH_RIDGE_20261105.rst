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

Revision 2: sparse-assay support bounds
--------------------------------------

The first completed demo revealed severe extrapolation for sparsely observed
protein/phosphosite features: near-constant training measurements could give
held-out kernel diagonals orders of magnitude beyond the training range.
The r1 outputs and code commit are retained; their deterioration is not treated
as biological evidence that added omics contain no information.

Revision 2 bounds each protein/phosphosite observation by that feature's
observed TRAINING minimum and maximum before applying the training median,
mean and standard deviation. It adds no measured-coverage threshold, removes
no sample after seeing its error, and uses no validation/test outcome to set
a bound. RNA, CNV, mutation, Y and split authorities are unchanged. The same
transform applies inside inner tuning and outer evaluation. Pure RNA-ridge
OOF predictions must match r1 within 1e-10; every sparse-assay query kernel
has a checked training-support bound. This is a development correction and
remains a retrospective demo, not independent external validation.

Reproduce on server 149
-----------------------

Run from the repository in the existing scientific environment with the
recorded dependency directories on ``PYTHONPATH``. Use one BLAS thread per
process. These commands reuse the demo's authoritative upstream paths; they
are not a general external-cohort import interface. The checked-in source
spec pins the six public genomic downloads by URL, size and SHA256::

   python scripts/download_graph_ridge_inputs.py --work "$WORK" --verify-only
   # For a new download directory, omit --verify-only. Use --direct-network
   # only when the server's configured HTTP proxy is unavailable.
   python scripts/prepare_graph_ridge.py --work "$WORK" --output "$INPUTS"
   python scripts/run_m0_three_arm.py --estimator graph_ridge \
       --omics-input "$INPUTS" --cancer BRCA --output "$OUTPUT"

Repeat the training command for LUAD and COAD. Then run
``scripts/audit_graph_ridge.py`` before ``scripts/report_graph_ridge.py``.
The original Lasso command retains its ``--aligned``, ``--splits`` and
``--covariates`` arguments and remains the default estimator. No historical
result directory is overwritten by the new versioned outputs.

Completed r2 demo
-----------------

The 2026-11-06 r2 run completed on server 149. The graph, three-arm and joint
M0 tests passed (29 tests), and all 18 cancer/phase audits passed. Pure RNA
ridge held-out predictions match r1 within 1e-10. The largest bounded assay
kernel diagonal is 3.4361; graph solve residuals are below 4.66e-14.
Seven existing M0 core modules are byte-identical to baseline commit
``6a5ef2d8da65486e2b88322ece1ad9c6cde72d27``. All five original top-level
functions in ``run_m0_three_arm.py`` are unchanged; its CLI dispatch is extended.

The following are pooled cancer-by-target medians on common successful targets.
Each RNA arm also includes the same target-eligible multiomics and unpenalized
covariates. Improvements count numerical paired increases, not independent
statistical discoveries; many pathways share genes.

.. list-table:: Graph-ridge multiomics demo, held-out R2
   :header-rows: 1

   * - Target family
     - Cancer-target pairs
     - lncRNA arm
     - mRNA arm
     - Combined arm
     - Median paired combined-minus-mRNA R2
   * - Protein pathway
     - 4358
     - 0.0907
     - 0.0992
     - 0.0978
     - -0.0024
   * - Phosphosite program
     - 948
     - 0.0376
     - 0.0252
     - 0.0312
     - +0.0044
   * - Methylation pathway
     - 1606
     - -0.0210
     - -0.0156
     - -0.0169
     - -0.0004
   * - RNA pathway
     - 5703
     - 0.2887
     - 0.3910
     - 0.3630
     - -0.0262
   * - RNA-derived State
     - 402
     - 0.2248
     - 0.3226
     - 0.3008
     - -0.0229
   * - TF
     - 813
     - 0.4427
     - 0.7872
     - 0.7080
     - -0.0580
   * - Other State
     - 9
     - 0.0520
     - 0.0329
     - 0.0408
     - -0.0001

Protein improves for 1946/4358 pairs (44.7%); phosphosite programs improve for
614/948 (64.8%). The phosphosite increment is small and differs by cancer;
COAD phosphosite median R2 remains negative. RNA score reconstruction is
stronger with mRNA than after adding lncRNA in this configuration. These
results do not establish that lncRNA contains no biological information.
Use the paired ablation tables to distinguish RNA choice, added assays and
graph regularization; use the matched old-Lasso table when comparing methods
because the successful-target populations differ substantially.

Matched ablations further separate the changes: for the combined RNA arm,
adding patient multiomics to the same ordinary ridge gives median paired R2
increments of +0.1024 (protein) and +0.0554 (phosphosite). Adding the graph on
top of those same inputs gives -0.00009 and -0.00012, respectively. These
results do not support a general benefit from this graph penalty.

On the old Lasso common-success subset, with identical held-out patients and
Y, the new combined model has median paired R2 changes of +0.1236 across 705
protein endpoints and +0.1975 across 131 phosphosite endpoints. RNA pathway
and RNA-State reconstruction decline on their matched old subsets. This
supports retaining the original implementation and endpoint-specific
comparisons rather than assuming the new estimator dominates every task.

Result root:
``/dell_2/DSC/CancerLncAtlas/results/model/M0_GRAPH_RIDGE_MULTIOMICS_demo_20261106_r2``.
Report root:
``/dell_2/DSC/CancerLncAtlas/results/model/M0_GRAPH_RIDGE_MULTIOMICS_REPORT_demo_20261106_r2``.
Patient-level inputs and predictions remain outside the public Git repository.


Physical RPI evidence correction (2026-11-06)
-------------------------------------------

The original RPI admission rule included ``experimental_unspecified``. An audit
found expression/abundance, reporter and unspecified evidence in that class;
these are not physical RNA-protein interaction evidence. The historical graph
gain figures therefore describe the old contaminated prior and must not be
interpreted as a validated physical PPI/RPI result. Patient measurements, native
Y and the non-network ridge comparisons are unchanged.

The preparation gate now requires an explicit physical method and a
``physical_binding`` evidence family. Historical input receipts are rejected
by the graph-ridge launcher. Corrected controlled experiments are documented in
``docs/NETWORK_VALIDATION_20261106.rst``; their final results are reported only
after all repeated nested-CV phases finish.
