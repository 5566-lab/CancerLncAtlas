MOGSA retired (2026-11-04)
=========================

The user retired MOGSA from the modeling design. Its scoring implementation,
derived lavaan parent-scoring pipeline and dedicated R dependency lock are
removed. The old scoring CLI returns MOGSA_RETIRED before reading input or
creating output. Configurations and prepared artifacts from this pipeline are
rejected, including a disabled config that still carries old fusion bindings.

config/149.json explicitly sets pathway_target_policy.status=REPLACEMENT_REQUIRED.
Preparation and training cannot silently substitute the existing RNA
activity_score. Independent pre-existing RNA workflows and synthetic fixtures
remain distinguishable; they are not a validated replacement for this design.

Verified RNA, gene CNV, promoter methylation and RPPA measurements remain intact.
The five-assay raw exporter is data preparation only and selects no scoring
algorithm. CNV/promoter methylation remain source-patient gene-node inputs.
State targets and the Pathway=1.0, State=0.5 joint objective are unchanged.

The proposed separate RNA, protein, phosphorylation, CNV and methylation
targets, missing-label handling and per-assay heads are not implemented by this
retirement. They require explicit score definitions and validation. No new
patient scores, prepared training targets, HHGT model or learned embeddings
are claimed by this change.

Historical five-assay results are preserved as historical evidence, not current
training authorization. The original files and manifests must not be rewritten
to pretend that another scoring algorithm produced them. The dated report
PATHWAY_FUSION_20261101.rst describes that retired run only.

Method terminology: MOGSA emits per-sample gene-set scores, but estimates its
reference structure from a cohort. Cohort fitting does not in itself make a
method impossible to use in predictive modeling. Retirement is the user's
model-design decision, not a claim that the original method is invalid.
