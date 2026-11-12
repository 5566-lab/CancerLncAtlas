Cancer-specific lncRNA--exact-pathway relations
===============================================

The analysis unit is cancer x lncRNA x exact pathway. BRCA, LUAD and COAD
are fitted and evaluated separately. The cancer backbone is defined from
biological evidence, independently of patient coverage and lncRNA associations.

KEGG cancer mechanisms
----------------------

The frozen MSigDB KEGG_MEDICUS 2025.1.Hs inventory contains 658 exact NETWORK
definitions. The evidence-defined backbone retains 477: 199 assigned to KEGG
Cancer / Cancer signaling, 23 additional mechanisms with explicit malignant
DISEASE annotations, and 255 supported by reviewed cancer-mechanism literature.
The remaining 181 stay in the source inventory with their exclusion basis.
These counts are results of this snapshot; code does not require a fixed count.

The backbone includes 334 Reference mechanisms, 78 Variant mechanisms,
47 Pathogen mechanisms and 18 environmental-factor mechanisms. A NETWORK
mechanism is often a reaction module or signaling branch, rather than a broad
KEGG pathway map. Its N identifier, official name, reaction, original source
definition and original gene membership define the target. Reference,
mutation, pathogen and exposure mechanisms remain distinct.

Cross-database integration
--------------------------

Formal equivalence mappings support identity merging. Parent-child relations
and shared molecular components are retained as relations. Protein similarity
uses complete-linkage groups with pairwise Jaccard at least 0.5; group
membership does not itself assert identity equivalence.

Other catalog entries are linked to the KEGG backbone using protein-component
enrichment in the frozen protein universe. BH correction covers all candidate
identity-by-backbone pairs, including zero-overlap pairs. A context link
requires at least three shared proteins and q <= 0.05. Exact TF/kinase
molecular identities have a separate direct membership relation to the
original KEGG proteins; a single regulator is not tested as a multigene set. Source/target coverage,
Jaccard and p/q are exported. The overlap p/q are annotation-set enrichment
statistics, not patient association statistics.

The resulting catalog contains 477 primary mechanisms, 1,652 associated
context identities and 197 State identities. TF programs, kinase programs and
PROGENy readouts display as Pathway. Their display category does not create
an independent loss. Genuine State remains an explicit auxiliary task, with
Pathway weight 1.0 and State weight 0.5.

Labels and applicability
------------------------

Exact identity bindings retain original KEGG and explicitly equivalent source
readouts. Verified TF/kinase molecular identities additionally bind member-state
readouts to the pathway. Overlap-only context links do not create supervision
bindings. Whole-pathway and component evidence, and each measurement layer,
retain separate output strata alongside the continuous RMS relation strength.
BH correction is recomputed for the current explicit task after new evidence
is bound. Different platforms retain their patient readout units. See
``MEMBER_STATE_EVIDENCE.rst`` for the complete binding and composition rules.

Generic host-component readouts cannot identify a named mutation, pathogen
or exposure. The 143 conditional definitions remain in the biological
backbone; their current generic-readout relation labels are masked as missing.
This rule is shared by the fused-label view and binary preparation. Missing
labels are not negative relations.

The three-cancer view contains 477 primary and 194 State columns. Three other
State definitions have no separate source-label column. In train_label_fold2,
observed primary/State targets are BRCA 270/193, LUAD 269/189 and COAD 269/189.
Phospho evidence is estimable for151,155,150 primary identities, respectively.
Each count requires at least one estimable lncRNA relationship; it does not
assert complete measurement of the entire pathway. In BRCA,247 identities have
same-pathway evidence and175 have member-state evidence, with152 shared:
their union is270, including23 with member evidence only. These are observation
counts in that split, not a biological selection rule or significance threshold.

The two O-mannosylation modules N00627 (POMT to POMK) and N00699 (FKTN to
LARGE) are included on focused cancer evidence for dystroglycan/matriglycan
biology (PMID:36494657). Neither has an observed fused relation label in the
current BRCA/LUAD/COAD train_label_fold2 views. Their definition does not
create a training label; directions of individual biosynthesis steps may differ.

M0 and CCHHGT comparison
-----------------------

The existing72-column source-association runner described below measures
replication across patient partitions. It is not the requested molecular-input
M0; using its API name does not establish a matched molecular baseline.
The molecular baseline requires an explicit molecular feature representation
with the same target readouts and labels as CCHHGT. No model fitting is part
of this Y and source-composition delivery.

The linear relation comparator uses, for each cancer::

    s(c,l,k) = b(c) + beta(c)^T u_source(c,l,k)

Its numeric inputs come only from the source-patient partition. The label
partition supplies Y. Default numeric_only excludes lncRNA/pathway identity
terms. For the selected continuous endpoint, Y is the observed fused
``rms_effect`` for that exact cancer--lncRNA--pathway relation. Unobserved
relations remain masked, including those with ``membership=0`` but no
measured effect. M0 fits an L1-penalized linear regression on the observed
training relations. Its penalty is selected by validation exact-target macro
MSE, with the same relation masks and target weights as HHGT. L1-penalized
logistic regression remains available only for explicitly binary prepared
endpoints. Label-side rho/p/q are never features.

Context identities enter the HHGT graph through fixed protein/gene composition
and typed connections to the primary mechanism. No context target enters the
supervised axis and no independent loss is added.

M0 and CCHHGT must use identical exact target IDs, patient splits,
applicability masks, labels and evaluation units. Comparing the full HHGT
to M0 measures the combined gain of node information and graph modeling.
A no-message-passing model with the same node inputs isolates topology gain.
TF/kinase auxiliary losses require an explicit experiment choice.

Fused relationship evidence has positive and unlabelled states.
``membership=0`` is not a confirmed negative, so binary conversion is rejected.
Continuous training uses observed effect sizes rather than treating absent
evidence as a zero effect. The prepared capsule must bind the frozen fused
view and its source/label patient roles before a matched training result can
be reported; the older RNA-score preparation does not meet that contract.

Reproduction
------------

Run all preparation on host 149. The entry points are::

    scripts/build_kegg_cancer_scope.py
    scripts/build_relation_catalog.py
    scripts/build_functional_target_view.py
    scripts/configure_cancer_relation_demo.py
    scripts/export_cancer_dictionary.py

The matched continuous relation workflow then uses::

    scripts/prepare_relation_components.py
    scripts/prepare_relation_baseline.py
    scripts/prepare_relation_graph_view.py
    cla.py prepare-relation --plan PLAN.json --output PREPARED
    cla.py worker --prepared PREPARED --output RUN --device cpu --deadline UNIX_TIME

Each prepared plan names the verified baseline, static graph and all four
source graph views for one cancer, plus model, training and diagnostic metric
settings. The prepared capsule aligns exact canonical target IDs across graph
nodes, M0 predictions and observed continuous labels. The 72 numeric columns,
cohort-specific assay nodes and missing-value masks are specified in
``M0_RELATION_BASELINE.rst``. Final prediction files preserve the exact lncRNA
and target axes and the corresponding M0 prediction for every candidate.

The scope build verifies the retrieved literature registry and original
membership digests. The catalog seals explicit target policies and crosswalks.
The target view seals source-column mappings and label applicability.
Demo configuration writes one cancer per capsule and retains the full
biological primary axis. The dictionary exports definitions, evidence,
cross-database links, protein clusters, molecular composition and readout
methods; source-only execution records remain in the project log.
