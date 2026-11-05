# CancerLncAtlas CC-HHGT v3.2

CancerLncAtlas learns cancer-specific lncRNA associations with exact pathways
and tumor states using one trainable PyG CC-HHGT encoder. The canonical entry
point is `cla.py`, configured by `config/149.json`. Its objective is
`Pathway loss + 0.5 * State loss + regularization`. Both tasks update the same
encoder. State targets and scores remain separate from Pathway targets and scores.

The unified executor includes the subsequent R6.2 preparation, exact sparse
LASSO baseline, joint optimization, checkpoint, evaluation and core-export
implementation. `config/model_v3_2_full_multitask.yaml` describes the same
architecture. The frozen State training workflow has been removed.

All preparation, audits and training execute on host **149**. No command in this
integration starts paid compute. The reviewed code and tests are not a newly
trained model or a production release.

## Pathway target status (2026-11-04)

MOGSA and its derived parent targets are retired. The public configuration
requires a replacement target design before preparation or training; it does
not silently fall back to RNA scores. Separate per-assay heads are not yet
implemented. See [the retirement and target policy](docs/PATHWAY_TARGET_POLICY_20261104.rst).

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
These are separate M0 models for frozen native scores; the HHGT target design remains pending.

## M0 lncRNA-only linear prediction (2026-11-04)

The separate M0 baseline uses all available annotated lncRNA expression jointly
in per-cancer Ridge regressions for existing native Pathway and State scores.
TCGA and CPTAC remain separate cohorts. Five patient-disjoint outer folds,
validation-only regularization selection, explicit missing labels and a
training-mean comparator produce auditable out-of-fold predictions. This is a
baseline for frozen cohort scores, not a replacement for the pending HHGT target
design or a claim of fully inductive raw-data validation.

Run `scripts/run_m0_linear.py` on host 149 and export figures with
`scripts/plot_m0_linear.py`. See [M0 methods, outputs and limitations](docs/M0_LINEAR_BASELINE.md).

## CPTAC external cohort integration (2026-11-04)

The CPTAC extension adds measured RNA, protein, site-level phosphorylation,
promoter-related gene methylation and published clinical data for 1,043 tumor
patients in ten harmonized cohorts. Per-assay Pathway scores and the current
485-State axis retain missing masks and explicit native-scale metadata.
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
For a two-cancer demo, set these fields in your bound configuration:

```json
"cancer_type": "list",
"cancer_list": ["BRCA", "LUAD"]
```

Or override them on the command line on host 149:

```bash
python cla.py prepare --config LOCAL_CONFIG.json --profile diagnostic --cancer_type list --cancer_list BRCA LUAD --output NEW_DEMO_PREPARED
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
use their original authority; they are not recalculated from two demo cancers.
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

The main labels are cancer × lncRNA × exact-pathway/State association labels,
derived from patient measurements in disjoint folds. They are not patient-level
continuous regression targets or causal intervention effects. State now has 484
primary targets plus auxiliary ESTIMATEScore, covering intrinsic transcriptional
states, immune function, cell composition and genomic measurements. Original
score definitions and missing-value masks are retained. See
[the State catalog and scoring workflow](docs/STATE_EXPANSION_20261101.rst) and
[the new score additions and complete protein mapping](docs/STATE_SIGNATURES_20261104.rst) and
[the unified training contract](docs/JOINT_MAINLINE_20261101.rst).

The previous five-assay MOGSA/lavaan scoring design was retired on 2026-11-04.
Its scores and prepared labels cannot be used for new training. Verified raw
assays are retained for a future per-assay design. Existing PROGENY/DoRothEA
resource scores remain in State using their legacy weighted implementation.
See [the current target policy](docs/PATHWAY_TARGET_POLICY_20261104.rst) and
[the historical scoring report](docs/PATHWAY_FUSION_20261101.rst).

The TCGA protein/phosphorylation input is the hash-pinned official TCPA
RPPA500 download. Its 447 columns contain 433 markers with measurements; the
quality-screened exact RNA sample-vial match covers 7,213 of 10,432 patients
(69.14%). Other PTMs and non-gene measurements remain separate. The current
download and paper/portal annotations have 29 different marker identities,
which are explicitly reconciled against exact GDC antibody records. See
[the RPPA500 input contract and reproducible commands](docs/RPPA500_INPUTS_20261104.rst).
This replaces measured inputs; separate protein/phosphorylation heads are not
implemented by this change.

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

The BRCA/LUAD/COAD demo compares lncRNA, mRNA and their union on identical held-out CPTAC protein/phosphosite endpoints (measured methylation as a secondary layer), with unpenalized clinical and fixed DNA purity covariates. See [the comparison contract](docs/M0_THREE_ARM_DEMO.rst) for scope, outputs and native-label normalization limits. Code readiness and result completion are recorded separately in the provenance receipt.
