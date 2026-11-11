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

Exact source IDs select original KEGG label columns. An output identity
redirect does not replace a KEGG label with an existing Reactome column.
All original p/q correction families remain frozen. Different platforms
retain their own patient readout units; the target is relation evidence.

Generic host-component readouts cannot identify a named mutation, pathogen
or exposure. The 143 conditional definitions remain in the biological
backbone; their current generic-readout relation labels are masked as missing.
This rule is shared by the fused-label view and binary preparation. Missing
labels are not negative relations.

The three-cancer view contains 477 primary and 194 State columns. Three other
State definitions have no separate source-label column. In train_label_fold2,
observed primary/State targets are BRCA 243/193, LUAD 242/189 and COAD 242/189.
These are counts of estimable relationships in that split, not a new
biological selection rule.

The two O-mannosylation modules N00627 (POMT to POMK) and N00699 (FKTN to
LARGE) are included on focused cancer evidence for dystroglycan/matriglycan
biology (PMID:36494657). Neither has an observed fused relation label in the
current BRCA/LUAD/COAD train_label_fold2 views. Their definition does not
create a training label; directions of individual biosynthesis steps may differ.

M0 and CCHHGT comparison
-----------------------

The linear relation comparator uses, for each cancer::

    s(c,l,k) = b(c) + beta(c)^T u_source(c,l,k)

Its numeric inputs come only from the source-patient partition. The label
partition supplies Y. Default numeric_only excludes lncRNA/pathway identity
terms. L1-penalized logistic regression is the comparator for explicitly
binary prepared endpoints; L2 is a separate option. Hyperparameters are
selected on validation patients. Label-side rho/p/q are never features.

Context identities enter the HHGT graph through fixed protein/gene composition
and typed connections to the primary mechanism. No context target enters the
supervised axis and no independent loss is added.

M0 and CCHHGT must use identical exact target IDs, patient splits,
applicability masks, labels and evaluation units. Comparing the full HHGT
to M0 measures the combined gain of node information and graph modeling.
A no-message-passing model with the same node inputs isolates topology gain.
TF/kinase auxiliary losses require an explicit experiment choice.

Fused relationship evidence has positive and unlabelled states.
membership=0 is not a confirmed negative. Its binary conversion is rejected;
the existing binary comparator cannot silently consume that evidence release.
This dictionary revision does not change that statistical endpoint.

Reproduction
------------

Run all preparation on host 149. The entry points are::

    scripts/build_kegg_cancer_scope.py
    scripts/build_relation_catalog.py
    scripts/build_functional_target_view.py
    scripts/configure_cancer_relation_demo.py
    scripts/export_cancer_dictionary.py

The scope build verifies the retrieved literature registry and original
membership digests. The catalog seals explicit target policies and crosswalks.
The target view seals source-column mappings and label applicability.
Demo configuration writes one cancer per capsule and retains the full
biological primary axis. The dictionary exports definitions, evidence,
cross-database links, protein clusters, molecular composition and readout
methods; source-only execution records remain in the project log.
