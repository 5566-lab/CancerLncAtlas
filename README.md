# CancerLncAtlas CC-HHGT v3.2

CancerLncAtlas studies **cancer × lncRNA × exact pathway**, with a
KEGG cancer-mechanism backbone and separate Pathway and State tasks.
Task weights are Pathway=1.0 and State=0.5. TF, kinase and PROGENy
readouts belong to Pathway; classification does not create independent losses.

The current Y-definition work fixes biological identities, original source
members and patient measurements before lncRNA relationship modeling.
Measured values, missingness and readout methods are exported independently
of lncRNA expression or association statistics. Model training is outside
the current delivery.

All preparation, audits and training execute on host **149**.
The development export excludes the sealed outer-test patients.

## Cancer mechanism backbone and measured targets

The evidence-defined backbone retains **477** NETWORK mechanisms from 658
source definitions: 199 in official cancer categories, 23 with additional
malignant-disease annotations and 255 supported by reviewed mechanism
literature. Original NETWORK IDs, names, conditions and members are retained.
Cross-database equivalence, hierarchy, protein similarity and regulator
membership are separate relationships. The catalog includes 1,652 context
identities and 197 State identities.

Use `scripts/export_measured_readouts.py` to export patient-by-readout values,
observed masks, fixed identities and structural bindings for BRCA, LUAD or
COAD. The exporter does not fit a model or combine lncRNA correlations into Y.
The old association-RMS materialization entry is disabled. Y groups patient
observations under an exact pathway identity, preserving assay and readout
dimensions. The typed registry retains unique measurements and separate
pathway membership; it does not calculate lncRNA associations. See
[the Y representation](docs/MEASURED_Y_REPRESENTATION.rst).

RNA scoring also requires a coding-gene dependency check: some original
scores use an all-gene background, mixed gene symbols or noncoding signature
members. `cla_engine.protein_coding_scoring` selects protein-coding gene IDs
before symbol aggregation. Scores recomputed with that background must retain
their method version rather than be described as unchanged published scores.

See [biological integration and measurements](docs/CANCER_RELATION_TASK.rst),
[member-state mappings](docs/MEMBER_STATE_EVIDENCE.rst) and
[patient-input linear M0 specification](docs/M0_RELATION_BASELINE.rst).
M0 uses patient molecular features and unpenalized clinical/purity covariates;
source-association summaries are not the requested molecular baseline.

## Reviewed three-cancer network comparison (2026-11-06)

Both network experiments completed: **BRCA, LUAD and COAD only**, with
33 phases each = 3 cancers × (2 repeats × 5 outer folds + 1 final fit).
All 66 phase eligibility audits and both sets of 30 heldout-identity audits passed.
Finer modules improve combined-arm protein/phospho mean paired R² by only
+0.00255/+0.00117 over coarse modules, remaining below adaptive-omics Ridge
by -0.04570/-0.01947. Adding lncRNA to mRNA within the fine model gives
+0.00092 for protein (interval crosses zero) and +0.00349 for phospho.
These are CPTAC native retrospective targets, not pooled calibrated TCGA/CPTAC Y.
See [complete methods and final results](docs/NETWORK_VALIDATION_20261106.rst)
and [Chinese conclusions](provenance/network_validation_20261106/final_review/FINAL_CONCLUSIONS_ZH.txt).
The graph-ridge and Lasso sections below describe earlier experiments.

## Graph-regularized ridge multiomics demo (2026-11-06)

The existing three-arm entry supports `--estimator graph_ridge --omics-input <verified-root>`.
It reuses the CPTAC BRCA/LUAD/COAD RNA, outcomes, folds, clinical/purity handling and
plot functions. PPI/RPI regularize RNA coefficients; matched gene CNV, mutation,
protein and phosphosite measurements enter as target-eligible predictor blocks.
The comparison includes ordinary ridge on the same inputs and a shared-omics-only
baseline. Protein/phosphosite and genomic-derived outcomes exclude their own
measurement blocks. RNA-derived labels retain lncRNA predictors, with constituent
and scoring-background overlap disclosed. See [model and validation details](docs/M0_GRAPH_RIDGE_20261105.rst).
The 2026-11-06 evidence audit found nonphysical expression/reporter records in the
old RPI prior. Those historical graph-gain results are not validated physical-RPI
results. The launcher now requires explicit physical evidence; corrected network
controls and repeated nested CV are described in
[network validation](docs/NETWORK_VALIDATION_20261106.rst).
The r2 three-cancer fit is complete: 29 tests and all 18 result audits passed.
Training-only support bounds prevent sparse-assay extrapolation; pure RNA
baselines remain unchanged. Protein paired median combined-minus-mRNA R2 is
-0.0024; phosphosite programs +0.0044. These are small, target-dependent effects,
not a general gain from adding lncRNA. Original Lasso code and prior results are retained.

## Joint TCGA/CPTAC M0 with clinical covariates (2026-11-05)

The joint demo restores a common lncRNA expression axis and fits shared lncRNA
coefficients within BRCA, LUAD and COAD. Age, sex, stage, fixed source DNA purity
and cohort terms enter as **unpenalized** covariates, as explicitly selected by
the user. Training-only partial-rank screening uses `|rho| >= 0.10, FDR <= 0.05`.
Validation reports the improvement over a clinical/purity-only baseline.
See [methods, input requirements and limitations](docs/M0_JOINT_CLINICAL_DEMO.rst).
A completed model requires result receipts; code availability is not training completion.

## Correlation-screened Lasso demo (2026-11-05)

The three-cancer extension restores the full shared TCGA/CPTAC lncRNA input
axis, screens associations inside each training split, and fits Lasso to export
fixed nonzero gene sets with weights and fold selection frequencies. The demo
is limited to BRCA, LUAD and COAD; cohorts retain their original score labels
and are fitted separately. There is no cross-cohort Y calibration claim.
See [methods, output contracts and scope](docs/M0_CORRELATION_LASSO_DEMO.rst).

All six cohort/cancer demo runs are complete and audited, with 13 PNG and 13 PDF figures.
See [completion receipts and case counts](provenance/joint_mainline_20261101/M0_LASSO_DEMO_20261105.json).
These six models predict native patient scores within each cohort and cancer.

## M0 lncRNA-only linear prediction (2026-11-04)

The separate M0 baseline uses all available annotated lncRNA expression jointly
in per-cancer Ridge regressions for existing native Pathway and State scores.
TCGA and CPTAC remain separate cohorts. Five patient-disjoint outer folds,
validation-only regularization selection, explicit missing labels and a
training-mean comparator produce auditable out-of-fold predictions. This is a
baseline for frozen cohort scores; its input is an individual patient's
lncRNA expression vector.

Run `scripts/run_m0_linear.py` on host 149 and export figures with
`scripts/plot_m0_linear.py`. See [M0 methods, outputs and limitations](docs/M0_LINEAR_BASELINE.md).

## CPTAC external cohort integration (2026-11-04)

The CPTAC extension adds measured RNA, protein, site-level phosphorylation,
promoter-related gene methylation and published clinical data for 1,043 tumor
patients in ten harmonized cohorts. The initial per-assay Pathway scores and
485-State source axis retain missing masks and explicit native-scale metadata.
The same gene/cancer/target identifiers do not establish equal calibration.
The validated reader and preflight binding preserve the existing TCGA inputs;
this extension does not train a new model or enable pooled targets.
See [methods, coverage and demo commands](docs/CPTAC_INTEGRATION_20261104.rst).

## Reproducible entry points

Run these commands from a full source checkout on host 149. Install the
dependencies listed in `requirements/` into an isolated environment. An editable
install can expose the CLI; an isolated wheel alone does not contain the full
configuration, downstream scripts and reviewed-source capsule.
Create a local copy of `config/149.json` and replace the `/path/to/` placeholders
with verified inputs. Keep patient identifiers, local controller keys, raw data
and trained outputs outside Git. Preserve source hashes and the declared
patient-fold, label, target-role and baseline contracts.

```bash
python cla.py package-check
python cla.py backend-check --output NEW_BACKEND_CHECK_DIRECTORY --device cpu
python cla.py controller-init --config LOCAL_CONFIG.json --role 149
python cla.py import-local VERIFIED_RAW_INPUT_ARCHIVE.zip --config LOCAL_CONFIG.json
python cla.py preflight --config LOCAL_CONFIG.json --output NEW_PREFLIGHT_DIRECTORY
python cla.py prepare --config LOCAL_CONFIG.json --fold 0 --profile diagnostic --output NEW_PREPARED_DIRECTORY
```

These model CLI commands retain the existing association-label workflow and
are not an entry point for the current measured-Y definition. The measured
export is `scripts/export_measured_readouts.py`; its output does not authorize
a model run.

`prepare` creates patient-disjoint input/label views and fits the matched
baseline. The worker accepts only validated joint prepared data. Training is a
separate, budget-bound command; do not rename prior prepared files or checkpoints
to bypass source/model signatures. Use `python cla.py worker --help` for the
explicit prepared directory, output directory, device and deadline arguments.

`bash run_model_workflow.sh --help` invokes this same CLI. The repository root
contains one `cc_hhgt` import tree; the executor does not load a hidden older copy.

## Cancer scope and a small demo

The top-level configuration parameter `cancer_type` accepts `all` or `list`.
`all` uses every cancer in the frozen patient map (33 in the current cohort).
`list` requires a nonempty `cancer_list`; an unknown cancer is an error.
The public configuration defaults to `"cancer_type": "all", "cancer_list": []`.
The requested demo uses BRCA, LUAD and COAD. For an existing model CLI configuration:

```json
"cancer_type": "list",
"cancer_list": ["BRCA", "LUAD", "COAD"]
```

Or override them on the command line on host 149:

```bash
python cla.py prepare --config LOCAL_CONFIG.json --profile diagnostic --cancer_type list --cancer_list BRCA LUAD COAD --output NEW_DEMO_PREPARED
python cla.py prepare --config LOCAL_CONFIG.json --profile diagnostic --cancer_type all --output NEW_ALL_CANCERS_PREPARED
```

Hyphenated aliases `--cancer-type` and `--cancer-list` also work. IDs are
trimmed, uppercased, deduplicated and sorted. CLI options override configuration.
`--cancer_type all` clears a configured list. Explicit selection overrides the
old profile-level `cancers` field; old bound configurations without the new
parameter retain their profile's cancer scope until explicitly overridden.

Cancer scope and data-size profile are separate: `diagnostic` retains its
lncRNA/pathway caps even with `all`. Only selected cancers contribute patient
views, baselines, cancer nodes, node-omics aggregates and subsequent training.
Global input checks and the fixed cross-cancer expression eligibility rule still
use their original authority; they are not recalculated from the selected demo cancers.
`CANCER_SCOPE.json` records the selection in preflight and prepared outputs.
The prepared validator binds this record to configuration and node identities.

Training consumes the prepared scope; it does not re-filter an existing capsule.
A diagnostic worker rejects cancer overrides; select the cancers during `prepare`.
Use a fresh output directory when changing scope. `preflight`, `prepare-formal`,
`run-queue`, formal `worker` and `gate` accept the same options; pass identical
overrides at each configuration-bound stage, or save the selection in the bound
configuration once. Formal admission and evaluation gates remain required;
a small demo does not establish a pan-cancer result.

## Inputs and targets

The biological output axis contains 477 exact KEGG mechanisms and 197 State
identities. Patient measurements are linked to this axis by verified pathway
identity or explicit molecular membership. Source-specific readouts, protein
members and modification sites retain their provenance and observed masks.
lncRNA correlations, their significance and their RMS are not the Y values.

TF, kinase and PROGENy readouts belong to Pathway. They describe regulatory
programs or member states; adding an independent supervised loss requires an
explicit task definition. Display category and database record count do not
set loss weights.

Recomputed RNA readouts use protein-coding gene IDs before symbol aggregation.
The fixed LM22 coding reference contains 517 genes and supplies 22 relative
cell-composition readouts per patient, with its own method version. Published
TLS12 and RNA-derived purity remain separately identified source readouts.
See [measurement definitions](docs/MEASURED_READOUT_DEFINITIONS.rst) and
[the patient-input M0 specification](docs/M0_RELATION_BASELINE.rst).

The TCGA protein/phosphorylation input is the hash-pinned official TCPA
RPPA500 download. Its 447 columns contain 433 markers with measurements; the
quality-screened exact RNA sample-vial match covers 7,213 of 10,432 patients
(69.14%). Other PTMs and non-gene measurements remain separate. The current
download and paper/portal annotations have 29 different marker identities,
which are explicitly reconciled against exact GDC antibody records. See
[the RPPA500 input contract and reproducible commands](docs/RPPA500_INPUTS_20261104.rst).
This replaces measured inputs; separate protein/phosphorylation heads are not
implemented by this change.

Retrospective BRCA and both OV centers now have independently checked five-fold
common-protein calibration diagnostics. They do not replace native training Y;
COAD calibration is explicitly withheld pending its input-scale/zero contract.
See [the actual calibration results and limits](docs/TCPA_CPTAC_CALIBRATION_STATUS_20261106.rst).

Verified gene-level CNV and promoter methylation measurements remain supported.
Joint preparation aggregates their original patient matrices using each view's
source patients, excluding that view's label patients. An outer-fold aggregate
cannot replace an inner-view input. See
[the node-input contract](docs/NODE_OMICS_INPUTS_20261029.rst) and the unified
training contract for binding and validation details.

KO/KD/OE/rescue regulatory-network readers and graph adapters remain available.
The cross-source study/sample split is not yet complete, so these graphs are not
silently inserted into primary joint training. See
[regulatory-network commands and evidence limits](docs/CAUSAL_REGULATORY_NETWORKS_20261101.rst).

Clinical, drug, evidence and other downstream modules retain their own outcome
and provenance contracts. Their frozen-encoder behavior does not apply to the
joint State task. Remaining requirements are tracked in `CODE_COMPLETION_PLAN.json`.

## Repository layout

- `cla.py`, `cla_engine/`: canonical joint workflow and data/validation contracts.
- `cc_hhgt/`: shared neural encoder, data interfaces and downstream modules.
- `scripts/`: acquisition, input conversion and downstream tools.
- `config/`, `configs/`, `schemas/`, `sql/`: configuration and data contracts.
- `tests/`, `provenance/joint_mainline_20261101/`: regression checks and reviewed integration evidence.
- `website/`, `docs/`, `data/`: application, protocols and public input instructions.

## Data and license

Raw TCGA/GDC, single-cell, clinical, drug and materialized patient data are not
distributed here. Use permitted sources and record their checksums. See
`SECURITY.md`, `data/README.md`, `CITATION.cff`, `LICENSE` and `NOTICE`.

### Three-arm comparison for non-RNA assay targets

The BRCA/LUAD/COAD demo compares lncRNA, mRNA and their union on identical held-out CPTAC protein/phosphosite endpoints (measured methylation as a secondary layer), with unpenalized clinical and fixed DNA purity covariates. See [the comparison contract](docs/M0_THREE_ARM_DEMO.rst) for scope, outputs and native-label normalization limits. The current default removes correlation/FDR admission and marks zero-RNA fits as training failures. All five folds and the final fit must select RNA for a complete success; failure rates accompany performance on common successful targets. Code readiness and result completion are recorded separately in the provenance receipt.

Functional Pathway/State identities, measurement definitions and relationship Y are described in [the construction guide](docs/FUNCTIONAL_PATHWAY_STATE_Y.rst). The current fixed ID catalog is in `data/functional_targets_20261110`.
