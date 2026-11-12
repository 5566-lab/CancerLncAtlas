Source-association replication control
=====================================

This page describes the existing source-association-summary runner. It predicts
relationship strength in a separate patient partition from source-patient
relationship summaries. It is a replication control, not the requested
molecular-input M0. References below to M0 name the runner's existing API;
they do not establish a matched molecular baseline for CCHHGT.

Endpoint
--------

The analysis unit is cancer x lncRNA x exact pathway. BRCA, LUAD and COAD
are fitted separately. The primary target axis is the evidence-defined KEGG
backbone. Its original definitions and members determine pathway identity.
TF, kinase and PROGENy programs are Pathway context; classification alone
does not introduce supervised targets or additional losses.

The continuous endpoint is the observed fused ``rms_effect`` of the exact
relationship, computed in the frozen label-patient partition. Missing
relationships retain an unavailable mask. They are not assigned a negative
label or a measured effect of zero. Named mutation, pathogen and exposure
mechanisms require matching condition evidence for supervision.

For each cancer, the linear comparator is::

    predicted_relation_strength(c,l,k) = intercept(c) + beta(c)^T u_source(c,l,k)

The coefficients act on source-derived relationship descriptors. They are
not coefficients for a patient-level lncRNA expression signature.

Source features and adjustment
------------------------------

The numeric feature block keeps TCGA and CPTAC separate. For each cohort and
each of RNA, State readouts, protein, phospho and methylation, it records:
mean signed rho, RMS rho, maximum absolute rho, mean log(1+n), log(1+readout
count), and an availability flag. Each cohort also contributes the measured
mean, standard deviation and measured fraction of lncRNA expression and
local CNV. This gives 72 columns. An absent measurement block has zero values
and a zero availability flag.

Source associations use only source patients, with the frozen
``CLA_SHARED_CORE_DNA_PURITY_RELATION_DEMO_V1`` adjustment. Target-specific
State policies and direct RNA membership exclusions are applied before
aggregation. A partial candidate set does not define a new FDR family.
Label-patient rho, p, q and effect values do not enter the input block.

Clinical variables and purity enter this patient-stage adjustment. The
resulting relationship model does not invent a patient clinical vector for
an aggregated cancer--lncRNA--pathway row.

Fitting and comparison
----------------------

``fit_strength_cancer`` fits an L1-penalized linear regression. Numeric
standardization is fitted on available training relations. The default
``numeric_only`` design excludes lncRNA and pathway identity terms. An
explicit ``exact`` identity ablation learns its vocabulary from training
rows only.

The penalty grid uses ``alpha = alpha_max / C`` with ``alpha_max`` determined
from the training design. Nonconverged fits are excluded. Validation selects
the penalty by mean squared error averaged across exact pathways, so pathways
with more available pairs do not dominate model selection. Saved outputs
include coefficients, scaler parameters, validation predictions and metrics
with exact pathway IDs.

M0 and CCHHGT compare the same primary candidates, source numeric inputs,
continuous labels, availability masks, patient partitions and evaluation
units. CCHHGT additionally uses molecular-node information and typed graph
connections. Its joint loss retains Pathway=1.0 and State=0.5. The standalone
M0 relation runner evaluates the primary pathway task; joint State baseline
predictions belong to the matched prepared training package.

``materialize_relation_baseline`` produces both task baselines from the four
frozen source components. Pathway fitting balances observed exact targets;
State fitting additionally balances biological families. State validation
averages targets within families, then families. A PROGENy readout supporting
the Hypoxia State does not create a second ``pathway_footprint`` State family.

The graph view retains separate RPPA antibody-group and CPTAC protein/site
nodes. A multi-gene antibody remains one measurement with component edges.
Gene CNV and promoter methylation retain separate value and availability
columns. Source means and coexpression edges use only the declared source
patients. Exact KEGG membership is retained independently of context links.

``cla.py prepare-relation --plan PLAN.json --output PREPARED`` assembles a
diagnostic outer-fold-0 capsule from all four graph views and a verified
baseline component. Its plan names ``baseline``, ``static``, ``graph_views``
(train_label_fold2/3/4 and validation), ``model``, ``training`` and ``gates``.
The original ``cla.py worker`` consumes this explicitly typed continuous
capsule. The older binary preparation entry does not reinterpret these labels.

Prepared verification checks graph/target identity, source-patient partitions,
label masks and every baseline prediction against the same source components.
The 72-column context reaches both joint heads. Modality column declarations
cover both cohorts and gene-node assays; removing only graph inputs while
retaining the baseline measures a graph-input ablation. A complete information
ablation also requires refitting the baseline with that modality excluded.

Three comparisons have different meanings:

* Numeric Lasso versus the full CCHHGT measures the combined gain from
  molecular-node information and graph modeling.
* A model with identical node inputs and no message passing isolates the
  contribution of message passing.
* Removing PPI or RPI while holding all other inputs and candidates fixed
  measures the contribution of those edge sources.

Input contract
--------------

``run_m0_relation`` accepts a verified prepared capsule with an explicit
``relation_objective=observed_fused_rms_effect``. It verifies the bound
target policy, exact identity map and source/label split before fitting.
The frozen outer test is excluded from fitting and validation selection.
Feature components and a FunctionalRelationContext directory are inputs to
preparation, not substitutes for a prepared capsule.

The binary branch is used only for a capsule explicitly declaring binary
labels. It cannot turn unobserved fused relationships into negatives.

Run on server 149::

    python scripts/run_m0_relation.py \
      --prepared /path/to/verified_relation_prepared \
      --policy /path/to/TARGET_POLICY.tsv \
      --out /path/to/m0_relation_results \
      --cancers BRCA --penalty l1 --identity-mode numeric_only

Each cancer and each outer fold uses its own capsule. Training rows from
different outer folds are not concatenated.
