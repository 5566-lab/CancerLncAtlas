Exact backbone Y, source hierarchy and current supervision scope
===============================================================

The fixed patient Y values and masks remain unchanged. Its catalogue has 476
Pathway identities and 48 State entries. An explicit supervised view selects
335 pathways with existing nonempty eligibility/applicability masks in each of
the six demo cohorts. The 141 event-conditioned definitions remain mechanism
annotations. The original construction passed an empty patient-event table;
that fact does not establish absence of events across cancers. Existing CNV,
mutation and public event inventories require appropriate patient-level mapping
and event interpretation before those targets can return to supervision.

An exact backbone owns member measurements, equivalent pathway readouts and
specified TF/kinase component readouts. A similar subpathway is structural
context, not an equivalent identity. WNT N00056 retains 46 members. Reactome
WNT ligand biogenesis and trafficking shares 18 members; its 8 additional
members and whole-program score do not expand this WNT Y. The BRCA WNT vector
has 155 registered dimensions for TCGA and 260 for CPTAC, with original masks.

The model specification uses the same observed outputs and patient folds for
M0 and HHGT. M0 fits source/platform-specific linear Lasso outputs with agreed
unpenalized clinical and independent-purity covariates. HHGT requires patient
conditioned graph encoding and matching observation decoders. Platform-specific
outputs preserve measurement identities; a cohort covariate is not numerical
calibration. Non-RNA target layers and derivatives are excluded from inputs.
RNA reconstruction hides the target and its direct mRNA constituents, while
lncRNA input remains available. Shared observation columns are counted once;
the prescribed Pathway/State weights remain 1.0/0.5. Model-comparison results are
then aggregated to the frozen backbone to assess predictive lncRNA associations.

Implementation audit: the fixed-Y accessor exists, but the new M0 fitting adapter
and HHGT patient-observation decoder are not yet wired into training. The old
HHGT relation-logit head and RMS-effect preparation are not this task. The
28-slide report therefore presents these models as a subsequent design, with
no new model training or lncRNA-derived Y calculation.

Resource counts are frozen project inventory: four shared composition resources
with 2740 definitions; 14 RNA scoring-resource families with 411 definitions;
PTMsigDB with 624 directional signatures. Source databases, scoring methods and
patient projects use separate denominators. The inventory workbook lists input,
bound and measured dimensions separately.
