Cancer-specific lncRNA--exact-pathway analysis
==============================================

The scientific output is cancer x lncRNA x exact pathway. Pathway identities,
source members, readout methods and measurement masks are fixed before
lncRNA modeling, independently of expression correlations and significance.

KEGG and cross-database integration
-----------------------------------

The frozen MSigDB KEGG_MEDICUS 2025.1.Hs inventory contains 658 NETWORK
definitions. The evidence-defined backbone retains 477: 199 in official
cancer categories, 23 with additional malignant DISEASE annotations and
255 supported by reviewed cancer-mechanism literature. These include
334 Reference, 78 Variant, 47 Pathogen and 18 environmental mechanisms.
Each retains its NETWORK N identifier, official name, condition and original
members. NETWORK mechanisms are not equivalent in size to broad pathway maps.

Formal equivalence mappings merge identities. Parent-child relationships,
protein similarity and regulator membership remain typed links.
Complete-linkage protein groups require pairwise Jaccard >= 0.5.
Protein-component enrichment uses a frozen protein background and BH correction
over all candidate pairs, including zero overlap. A context link requires
at least three shared proteins and q <= 0.05. These are annotation-set tests.
TF/kinase components use verified molecular identity; a complex requires
all known components. Original source lists are retained in full.

The catalog contains 477 primary mechanisms, 1,652 context identities and
197 State identities. TF, kinase and PROGENy display as Pathway. Classification
does not create independent losses. Pathway=1.0 and State=0.5.

Measured foundation for Y
-------------------------

Each readout has a biological definition, platform, measurement layer,
scoring method, patient value and observed mask. Exact-identity readouts
and member-state readouts have different roles. An overlap-only source
retains its full protein list and structural link; it is not renamed as
a measurement of the whole KEGG mechanism.

Data are patient x source-readout matrices. A measured zero or constant
column is observed. Missing remains NaN. Coverage counts patients with
measurements, separately for identity and member-state readouts.
No lncRNA association threshold enters these counts.

Measurement and applicability are separate: generic member measurements
cannot establish the mutation, pathogen or exposure required by a conditional
mechanism. All 197 State identities remain in the structural axis.
Oncogenic dedifferentiation, Epithelial and mesenchymal programs, and
Cancer-associated extracellular-matrix programs have no direct source
column in the inspected demo streams; no values are invented for them.

TCGA RPPA protein readouts use source-reference standardized antibody
gene-group means. CPTAC protein readouts use samplewise ssGSEA.
Phospho readouts retain source-specific sites and scoring methods.
Shared identity or a cohort covariate does not establish equal measurement
units. The exporter preserves measured inputs to Y; it does not choose
a scalar fusion method or calculate an lncRNA-by-pathway correlation matrix.

Reproduction
------------

All preparation runs on host149. Structural entry points are
scripts/build_kegg_cancer_scope.py, scripts/build_relation_catalog.py,
and cla_engine.pathway_graph_sources.materialize_graph_sources.

Export measured inputs using scripts/export_measured_readouts.py with
--bindings, --output, --cancer and --context. The exporter verifies source
scores and structural mappings, retains fixed identities regardless of
coverage, and reads only declared development patient partitions.
Association arrays, lncRNA expression and outer-test values are not inputs.
See MEMBER_STATE_EVIDENCE.rst and M0_RELATION_BASELINE.rst.
