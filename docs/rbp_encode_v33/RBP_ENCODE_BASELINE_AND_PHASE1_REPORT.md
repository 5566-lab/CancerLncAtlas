# RBP_ENCODE_BASELINE_AND_PHASE1_REPORT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phases covered**: 0 (audit) and 1 (assay taxonomy)
**Date**: 2026-09-21
**Server**: `149` (`MyLabServer`)
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. Work root

```
${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/
├── code/        copy of the immutable origin capsule
├── reports/     this report, RBP_CURRENT_CODE_AUDIT.md, test logs
├── manifests/   RBP_ASSAY_TAXONOMY.tsv, WORKROOT_SHA256_MANIFEST.tsv
├── inputs/      (empty — Phase 5/6)
├── outputs/     (empty — Phase 3+)
├── runtime/     wheels + pytest/sympy --target dir
└── env.sh       canonical environment
```

**Immutability evidence.** A recursive SHA256 comparison of the work root against the
origin capsule `${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/code`
reports:

```
modified: 0   added: 2   removed: 0
  +  cc_hhgt/v32/rbp_assay.py
  +  tests/test_v32_rbp_assay.py
```

The origin capsule was re-hashed after copying and is unchanged. No formal V3.2 artifact
is used as an output target.

`manifests/WORKROOT_SHA256_MANIFEST.tsv` records 1209 files (`sha256 / size / path`).

---

## 2. Environment (Phase 12 prerequisite — closed)

| item | value |
|---|---|
| interpreter | `/usr/bin/python3` 3.10.12 |
| package root | `${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/python_packages_v32_drug_20260825` |
| extended target | `$RBP_ROOT/runtime/py` |
| numpy / pandas | 2.2.6 / 2.3.3 |
| pyarrow / duckdb | 25.0.0 / 1.5.0 |
| torch | 2.7.1+cpu |
| sympy / mpmath | 1.14.0 / 1.4.1 |
| pytest | 9.1.1 |
| network | SOCKS5 `127.0.0.1:1080` (the `7890` proxy in the environment is dead) |

### Three environment defects found and fixed

These are recorded so the CPU/GPU runbooks do not rediscover them.

1. **`pip install --target` silently loses packages on this sshfs mount.**
   pip reported `Successfully installed pytest-9.1.1` while `PY_EXTRA/pytest/` did not
   exist. Fixed by extracting wheels with `zipfile` directly. This is the mechanism now
   used for every package added to `PY_EXTRA`.

2. **`sympy` in the CPU-torch target dir is an incomplete extraction.**
   `.../v32_cpu_torch_2_13_0_20260830_r1/site-packages/sympy/` has 32 sub-directories but
   **no `__init__.py`**, so Python resolved it as an empty namespace package. Every
   `torch.optim.*` construction died with
   `ModuleNotFoundError: No module named 'sympy'` and later
   `ImportError: cannot import name 'S' from 'sympy' (unknown location)`.
   Fixed by installing a complete sympy 1.14.0 + mpmath into `PY_EXTRA`, which precedes
   the broken directory on `PYTHONPATH`.

3. **PyPI simple-index URLs include PEP 740 integrity payloads.**
   Grepping the simple index for `sympy-*.whl` first returned
   `https://pypi.org/integrity/sympy/1.14.0/sympy-1.14.0-py3-none-any.whl` — a
   4864-byte attestation blob, not a wheel. Wheel URLs must be filtered to
   `files.pythonhosted.org`.

Verified after repair:

```
sympy 1.14.0
AdamW forward/backward/step OK, loss = 0.385229
```

---

## 3. Test baseline (Phase 12 gate reference)

Command: the RBP/evidence/graph/interaction subset in `$RBP_CODE`.

| run | result |
|---|---|
| before any change | 3 failed, 61 passed, 1 skipped |
| after Phase 1 + env repair | **2 failed, 105 passed, 1 skipped** |

### The two remaining failures are pre-existing and environmental

```
tests/test_v32_formal_prepare_graph_authority.py::test_safe_minimal_authority_builds_three_true_masks
tests/test_v32_formal_prepare_graph_authority.py::test_fold_expression_leakage_or_sha_drift_fails_closed
```

Both fail with:

```
cc_hhgt.v32.patient_fold_authority.PatientFoldAuthorityError:
  Patient fold manifest is missing or unsafe
  path: code/artifacts/v32_patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv
```

The `artifacts/` directory **does not exist in the immutable origin capsule either**
(verified directly), so these two tests cannot pass in this capsule regardless of any
change made by this task. They are fixture-availability failures, not code defects.

### One baseline failure that was environmental and is now fixed

```
tests/test_v32_evidence_training.py::test_private_attention_head_minimal_training_keeps_core_as_detached_input
```

This failed because of the broken `sympy` (defect 2 above). After installing a complete
sympy it **passes**, which is why the failure count dropped from 3 to 2 while the pass
count rose. This is an improvement, not a regression.

**Phase 12 "no new failures" gate is therefore anchored at: 2 known pre-existing failures,
both fixture-availability.**

---

## 4. Phase 1 deliverable — `cc_hhgt/v32/rbp_assay.py`

SHA256 `ccbcf5023df4992b…` · 24-row rule table · `TAXONOMY_VERSION = RBP_ASSAY_TAXONOMY_V1`

### Levels produced

| level | purpose | vocabulary size |
|---|---|---|
| `experiment_raw` | verbatim provenance, never altered | open |
| `assay_subtype` | fine-grained category | 18 |
| `graph_assay_class` | **only** level allowed to type graph relations | 8 |
| `experiment_family` | historical coarse family, for legacy equivalence | 7 |

`assay_subtype`: `eclip, par_clip, iclip, hits_clip, clip_unspecified, rip, chirp, rap,
chart, rna_pulldown, emsa, co_ip, functional_perturbation, reporter_assay,
expression_or_abundance, computational_prediction, other_experimental, unspecified`

`graph_assay_class`: `eclip, other_clip, rip, rna_capture, other_physical,
experimental_unspecified, predicted, unknown`

### Determinism

Rules are an ordered tuple of `re` patterns evaluated first-match-wins over a lowercased
copy of the raw string. No randomness, no hashing of unordered containers, no model
inference, no data-dependent state. Verified by a test that re-classifies a corpus 50×
and asserts identical results.

### Boundary discipline (new — the historical regex was also *noisy*, not only lossy)

The historical `experiment_family` matched bare substrings, so:

| text | historical result | why |
|---|---|---|
| `transcript abundance measured` | `physical_binding` | `rip` inside "sc**rip**t" |
| `manuscript curation` | `physical_binding` | `rip` inside "sc**rip**t" |
| `description of the method` | `physical_binding` | `rip` inside "sc**rip**t" |
| `graph based inference` | `physical_binding` | `rap` inside "g**rap**h" |
| `rapid amplification` | `physical_binding` | `rap` prefix |

The new module uses `(?<![a-z0-9])` / `(?![a-z0-9])` guards instead of `\b`, because
method strings routinely contain hyphens, digits and slashes (`PAR-CLIP`, `HITS-CLIP`,
`RIP-Seq`) where `\b` behaves inconsistently. A test asserts both facts simultaneously:
the legacy helper still reproduces the historical false positives, while the new
classifier does not.

### Fail-closed behaviour

| input | `assay_subtype` | `graph_assay_class` | `is_experimental` |
|---|---|---|---|
| `""` / whitespace / `NaN` | `unspecified` | `unknown` | **False** |
| unrecognised non-empty text | `other_experimental` | `experimental_unspecified` | True |
| `is_predicted=True` | `computational_prediction` | `predicted` | **False** |
| `is_experimental=False` | `computational_prediction` | `predicted` | **False** |

An explicit computational flag always beats any textual match, so a prediction record
cannot be laundered into a binding observation.

### Legacy equivalence

`legacy_experiment_family()` reproduces the historical function verbatim, and a test
asserts it equals `evidence_interaction_rematerialization.experiment_family` over a
33-string corpus. **This is the anchor that makes ablation mode A
(`legacy_generic_binding`) provably reproduce the old semantics.**

### Relation type mapping

```
eclip → binds_protein_eclip          rip → binds_protein_rip
other_clip → binds_protein_other_clip  rna_capture → binds_protein_rna_capture
other_physical → binds_protein_other_physical
experimental_unspecified → binds_protein_experimental_unspecified
predicted → binds_protein_predicted    unknown → binds_protein_unknown
```

Unknown input always fails closed to `binds_protein_unknown`.

### Test result

```
tests/test_v32_rbp_assay.py   43 passed in 1.58s
```

Covering: determinism, closed vocabularies, eCLIP≠RIP at both subtype and graph-class
level, `physical_binding` family equality **not** erasing subtype, verbatim
`experiment_raw`, fail-closed `unknown`, prediction-never-experimental, historical false
positives removed, legacy equivalence, relation-type closure, and frame integration
(index preservation, fallback, empty frame).

### Deliverable

`manifests/RBP_ASSAY_TAXONOMY.tsv` — 24 rows, columns
`rule_order, matched_rule, assay_subtype, graph_assay_class, experiment_family, regex,
relation_type, taxonomy_version`.

---

## 5. Phase 2 design decision (recorded before implementation)

Phase 2 must extend `EVENT_FEATURE_FIELDS` from
`(route_type, source_database, source_dataset, experiment_type, relation_type, tissue,
cell_line, species, member_type)`
to
`(route_type, source_database, source_dataset, experiment_family, assay_subtype,
relation_type, tissue, cell_line, species, member_type)`.

`_event_feature_matrix()` hashes each field as a categorical token
(`token = f"{field}={value}"` → sha256 → bucket). Therefore **renaming the field changes
the hash bucket even when the value is identical**, which means a naive rename would
silently change the legacy model's forward output and invalidate ablation mode A.

The implementation will therefore make the field list **configuration-driven**:

* mode A (`legacy_generic_binding`) → original tuple with `experiment_type`;
* modes B/C/D → extended tuple with `experiment_family` + `assay_subtype`.

The normalised frame will carry `experiment_type` *and* the four new columns so both
modes are served from one row representation. This is required for the four ablation
modes to be genuinely comparable.

`auxiliary_gradients_into_core = false` and `changes_primary_ranking = false` are
unchanged by this task.

---

## 6. Status

| phase | status |
|---|---|
| 0 — real-code baseline audit | **COMPLETE** (`RBP_CURRENT_CODE_AUDIT.md`) |
| 1 — assay taxonomy | **COMPLETE** (module + 43 tests + TSV) |
| 2 — Evidence Transformer | **DESIGNED**, not yet implemented |
| 3–15 | not started |
| Paid GPU | **NOT STARTED** (correct — CPU gate not closed) |
| Formal V3.2 artifacts overwritten | **NO** |
| Origin capsule modified | **NO** |
