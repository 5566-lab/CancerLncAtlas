Functional Pathway and State relationships
=========================================

The research unit is cancer x lncRNA x canonical Pathway or State. Patient
measurements are denoted by S; relationship evidence is denoted by Y. The
current release covers BRCA, LUAD and COAD, combining TCGA with matched TCPA
RPPA500 and CPTAC at the association-evidence level.

Biological identity and composition
-----------------------------------

There are 3,456 scored biological identities: 2,595 have a Pathway role and
864 have a State role, with three identities shared between these roles.
Four additional structural context nodes describe composition and hierarchy.
Stable CLA-P / CLA-S IDs, source identifiers and aliases preserve the naming.
Equivalent biological scopes share an identity. Parent-child relations,
branches, shared reaction cores and extensions retain their distinct scopes.
Protein lists, phosphosites, genes, probes, reaction participants, catalysts,
regulators and complexes remain traceable in the companion dictionary.

Examples include equivalent WNT scopes from Reactome and WikiPathways,
Hallmark/Buffa/PROGENY readouts of the hypoxia transcriptional response,
the shared tyrosine catabolic core with Reactome extensions, and the explicit
HRD = LOH + TAI + LST composition. A measurement readout binding is distinct
from equivalence between whole pathway definitions.

Patient scores and adjustment
------------------------------

RNA pathways use common-background ssGSEA. RNA State definitions use their
specified signatures and source-fitted transformations; PROGENY uses the
official top100 model. RPPA total-protein scores summarize source-standardized
antibody gene groups. RPPA phosphosite scores use exact site mapping and
matched total-protein adjustment. CPTAC protein scores use ssGSEA; directional
PTM-SEA uses source-patient site medians. Promoter methylation scores summarize
measured promoter-associated gene beta values. Native phenotypes retain their
source units. Score_definitions and Score_coverage identify the actual method,
platform, reference, data form and observed coverage of every readout.
Each Score_definitions row also supplies its definition_database, source_resource
and a score_matrix path relative to the versioned data root. These separate the
biological definition source from the platform that measured the patient.

Each context has disjoint source and label patients. Source patients fit
applicable score references and covariate encoding. Associations are estimated
in label patients. The shared covariate design includes age, sex, stage,
DNA-derived purity and macrophage fractions, with target-specific adjustment
policies. Purity targets exclude their own purity covariate. Signed local CNV
adjusts the lncRNA residual. Direct RNA score members are excluded from their
own readout's relationship labels. All 16,889 lncRNA identities are retained;
missing expression or CNV remains missing. CPTAC-BRCA has measured local CNV
for 588 lncRNAs, which limits its estimable relations.

Fusion into Y
-------------

Each retained readout supplies an adjusted correlation and a two-sided p-value.
The combination statistic is T = -2 sum_j w_j log(p_j). Exact duplicate rank
readouts with the same patients and adjustment policy share total weight one.
The null distribution is simulated with 100,000 correlated-normal draws using
the label-readout residual Gram matrix and observed measurement masks.
Single-readout and perfectly dependent cases are retained exactly; unknown
dependence uses a Bonferroni bound. Cohort blocks are independent after the
ambiguous retrospective BRCA case is excluded from the joint sealed-label
analysis. No common absolute TCPA/CPTAC activity scale is assumed.

BH correction covers all estimable lncRNA x identity pairs within each cancer,
context and primary task. Strong support requires combined q <= 0.05 and at
least one observed absolute adjusted rho >= 0.20. Weak support requires
q <= 0.10 and absolute rho >= 0.15, excluding strong support. Opposing readout
directions remain explicit. Fusion significance and individually supported
assay or cohort flags are separate fields. Pathway weight is 1.0 and State
weight is 0.5; shared identities are counted once with Pathway-primary weight.

The method was checked with 257,000 permutations across 26 estimable real-data
cases, retaining measurement missingness and covariate adjustment. All 15
materialized contexts pass array, checksum, patient-separation, support-rule
and count checks. Independent reconstruction of four representative targets
in each of the three training-label pilots reproduces the saved combined p.
These checks establish the stated label computation, not predictive or causal
performance.

Counts from the independent label fold
--------------------------------------

* BRCA: 2,721 estimable targets; 2,720 targets with supported relations; 4,448,465 strong and 1,696,866 weak relations.
* LUAD: 2,753 estimable targets; 2,112 targets with supported relations; 84,695 strong and 53,530 weak relations.
* COAD: 2,703 estimable targets; 2,674 targets with supported relations; 160,277 strong and 120,807 weak relations.

These counts use sealed_test once per cancer. They are not sums over the
training, validation and test contexts. Most supported relations still have
only RNA readouts individually reaching the support threshold; the dictionary
reports non-RNA and joint-only support separately.

Reading the data
----------------

The versioned data root on server 149 is::

    /dsk2/user/dengsc/CancerLncAtlas/processed/functional_y_20261110_r2

Use its Y subdirectory::

    from cla_engine.functional_y import load_functional_context
    target = load_functional_context("/path/to/release/Y", "BRCA", "sealed_test")
    print(target.counts())
    masks = target.supervision_masks()

The command-line inspector uses the same reader::

    python scripts/inspect_functional_y.py --catalog /path/to/release/Y \
        --cancer BRCA --context sealed_test

The format is CLA_FUNCTIONAL_RELATION_FUSED_V1. Membership codes are
0 unlabelled, 1 weak support and 2 strong support. Unmeasured associations
have observed=0 and NaN p/q. The reader exposes positive and unlabelled masks;
it does not create a confirmed negative class. The existing binary training
preparation requires an explicit objective adaptation before consuming this
format. The current release supplies relationship labels, patient scores,
composition, mappings, coverage, a dictionary and a presentation.

Reproduction
------------

provenance/functional_y_20261110/source preserves the executed fusion,
materialization, validation and reporting scripts. Configuration and source
hashes are recorded with the release. The scientific work directory supplies
the referenced patient matrices and source annotations. All scientific
computation and rendering run on server 149. Source scores and associated
reference records are indexed by PATIENT_SCORE_INDEX.tsv.
