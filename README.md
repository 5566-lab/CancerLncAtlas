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

## Inputs and targets

The main labels are cancer × lncRNA × exact-pathway/State association labels,
derived from patient measurements in disjoint folds. They are not patient-level
continuous regression targets or causal intervention effects. State has 11
primary targets; four additional measurements retain auxiliary roles and do not
silently become primary losses. See
[the unified training contract](docs/JOINT_MAINLINE_20261101.rst).

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
