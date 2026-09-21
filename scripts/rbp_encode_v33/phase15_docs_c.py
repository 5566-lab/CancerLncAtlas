"""Phase 15 item 10: RUN_149_CPU.md and RUN_GPU.md."""

from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
REPORTS = WORK / "reports"

CPU = """# RUN_149_CPU.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Deliverable**: Phase 15 item 10 (execution document, CPU half)
**Host**: `149` (`MyLabServer`, `dengsc`)
**Date**: 2026-09-21

Everything in this document runs on **149** and needs **no GPU**. Nothing here writes to a
formal V3.2 artifact.

---

## 0. One-time setup

```bash
source ${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/env.sh
```

`env.sh` sets:

| variable | value |
|---|---|
| `RBP_ROOT` | `${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1` |
| `RBP_CODE` | `$RBP_ROOT/code` |
| `PY_PKGS` | `${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/python_packages_v32_drug_20260825` |
| `PY_EXTRA` | `$RBP_ROOT/runtime/py` (pytest, sympy, xxhash, torch_geometric) |
| `PY_TORCHDEPS` | `.../v32_cpu_torch_2_13_0_20260830_r1/site-packages` (networkx, mpmath, jinja2) |
| `PYTHONPATH` | `$PY_PKGS:$PY_EXTRA:$RBP_CODE:$PY_TORCHDEPS` |
| proxies | `socks5h://127.0.0.1:1080` — **the `7890` proxy in the default environment is dead** |

Verify before doing anything else:

```bash
python3 - <<'PY'
import torch, torch_geometric, pandas, pyarrow, duckdb, sympy, pytest, xxhash
print("torch", torch.__version__, "| pyg", torch_geometric.__version__,
      "| pandas", pandas.__version__, "| pytest", pytest.__version__)
PY
```

Expected: `torch 2.7.1+cpu | pyg 2.8.0 | pandas 2.3.3 | pytest 9.1.1`.

---

## 1. Environment gotchas that cost time to find

Recorded so they are not rediscovered.

| symptom | cause | fix already applied |
|---|---|---|
| `pip install --target` reports success but the package dir never appears | sshfs mount; pip's final rename silently fails | extract wheels with `zipfile` directly into `PY_EXTRA` |
| `ModuleNotFoundError: No module named 'sympy'` then `cannot import name 'S' from 'sympy' (unknown location)` | the CPU-torch `site-packages/sympy` is an incomplete extraction (32 subdirs, **no `__init__.py`**) | complete sympy 1.14.0 + mpmath installed into `PY_EXTRA`, which precedes the broken dir on `PYTHONPATH` |
| `ModuleNotFoundError: No module named 'torch_geometric.utils'` | all three `torch_geometric_2_8_0_staging*` dirs are **incomplete extractions** | official `torch_geometric-2.8.0-py3-none-any.whl` extracted from the project wheelhouse; SHA256 verified against `PYG280_WHEEL_SHA256.tsv` = `1f62e415a2e9ee69d34617d1b0b230e9d9040f51809b96e801e742770fd4dada` |
| `No module named 'xxhash'` | only a cp313 wheel existed; interpreter is cp310 | cp310 manylinux wheel fetched and extracted |
| a curl of a PyPI simple index downloads a ~5 KB file that is not a wheel | PEP 740 integrity payload | filter URLs to `files.pythonhosted.org` |

---

## 2. Run the test suites

```bash
cd "$RBP_CODE"

# new RBP tests only (178 tests)
python3 -m pytest -q -p no:cacheprovider \\
  tests/test_v32_rbp_assay.py \\
  tests/test_v32_rbp_evidence_features.py \\
  tests/test_v32_rbp_typed_binding.py \\
  tests/test_v32_rbp_typed_graph.py \\
  tests/test_v32_rbp_canonical.py \\
  tests/test_v32_rbp_evidence_config.py

# full RBP/evidence/graph regression
python3 -m pytest -q -p no:cacheprovider \\
  tests/test_v32_rbp_assay.py tests/test_v32_rbp_evidence_features.py \\
  tests/test_v32_rbp_typed_binding.py tests/test_v32_rbp_typed_graph.py \\
  tests/test_v32_rbp_canonical.py tests/test_v32_rbp_evidence_config.py \\
  tests/test_v32_evidence_interaction_rematerialization.py \\
  tests/test_v32_evidence_training.py tests/test_v32_evidence_streaming_training.py \\
  tests/test_v32_evidence_fusion_adapter.py tests/test_v32_evidence_identifier_recovery.py \\
  tests/test_v32_evidence_semantic_wrapper.py tests/test_v32_evidence_output_binding.py \\
  tests/test_v32_experiment_assay_detail.py tests/test_v32_experiment_perturbation.py \\
  tests/test_v32_formal_graph_contract.py tests/test_v32_formal_graph_index_invariance.py \\
  tests/test_v32_formal_prepare_graph_authority.py tests/test_v32_model_and_graph.py \\
  tests/test_v32_shared_encoder_group_backward.py tests/test_v32_gpu_backward_probe.py \\
  tests/test_v31_runtime_graph_training.py tests/test_v31_graph_residual_training.py \\
  tests/test_v32_interaction_release.py tests/test_v32_interaction_query.py \\
  tests/test_graph_build.py tests/test_v31_graph_contract.py tests/test_v32_contracts.py
```

**Expected**: new tests all pass; full regression **6 failed, 298 passed, 1 skipped**.

The six failures are pre-existing and each reproduces on the untouched origin capsule
`${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/code`:

* 4 × `test_v32_evidence_streaming_training.py` — DuckDB 1.5.0 `InternalException`
  (`Attempted to access index 3 within vector of size 3`);
* 2 × `test_v32_formal_prepare_graph_authority.py` — `artifacts/v32_patient_fold_authority_20260829_r1/`
  is absent, in the origin capsule too.

**Any other failure is a regression and must be investigated before proceeding.**

---

## 3. Re-materialise the typed binding table

Reads only frozen inputs; writes only under `$RBP_ROOT`.

```bash
python3 /tmp/phase3_materialize.py
```

Inputs (frozen, read-only):

```
${PRIVATE_WORK_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/
  V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized/
    interaction_relation.parquet          6,160,707 rows
    protein_gene_map.parquet
    dim_lncRNA.parquet / dim_cancer.parquet
    lncRNA_protein_relation_context.parquet   755,346 rows (the frozen binding table)
```

Checks that must hold:

| check | expected |
|---|---|
| legacy equivalence | **PASS** — 755,346 rows, zero column mismatches |
| typed rows | 918,266 |
| pairs added | 155,746 |
| `context_specific_lost_to_global` | **0** |

Output: `outputs/phase3_typed_binding/lnc_protein_binding_typed.parquet` +
`PHASE3_TYPED_BINDING_AUDIT.json`.

---

## 4. Recompute the CPU preflight

```bash
python3 /tmp/phase12_preflight_compute.py
python3 /tmp/phase12_preflight_report.py
```

Regenerates `manifests/PHASE12_CPU_PREFLIGHT.json` and
`reports/RBP_ENCODE_CPU_PREFLIGHT.md`.

Gates that must read **PASS**: 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12.
Gate 3 (ENCODE manifest SHA) is **BLOCKED** until the ENCODE path decision is made.

Key expected values:

| metric | expected |
|---|---|
| RBP → UniProt mapped fraction (typed) | 99.09 % |
| unmappable partner rows | 8,940 in both legacy and typed |
| uninformative assay rows | 38,703 (4.21 %) |
| physical-only graph edges | 117,951 |
| context edges without a cancer | **0** |
| legacy global edges vs physical-only global edges | 724,502 → 102,425 (Δ = 622,077 predictions) |

---

## 4b. Materialise the typed-relation graph authority

The typed generation needs its own receipt because it genuinely does not share the
relation schema of the frozen one.

```bash
python3 /tmp/phase7_typed_authority.py 0 1 2 3 4
```

Writes only under `$RBP_ROOT/outputs/phase7_typed_authority/`:

| file | content |
|---|---|
| `static/lnc_protein_binding.parquet` | the typed binding table (918,266 rows) |
| `static/<other>.parquet` | the five unchanged static artifacts, hashes matching the frozen receipt |
| `GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json` | the typed-generation receipt |
| `fold_<n>/AUTHORITY_SUMMARY.json` | per-fold edge and variant counts |

**Cost: roughly 25 minutes per fold**, so a full five-fold run is about **two hours**
(8.25M coexpression rows merged against 3.3M candidate keys over NFS). Launch it
detached rather than over an interactive SSH session.

Expected fold-0 result:

| metric | value |
|---|---|
| generation | `TYPED_ASSAY_CLASS_V1` |
| edges | 7,136,794 |
| G0 / G1 / G2 | 6,863,061 / 6,981,942 / 7,136,794 |
| G1 − G0 | 118,881 = binding 98,873 + protein→gene 20,008 |
| G2 − G1 | 154,852 (PPI) |
| `binds_protein_predicted` | **absent** (conservative default) |

If `binds_protein_predicted` appears, `include_predicted` was enabled somewhere and
~622k prediction edges are entering message passing.

### Disk cost of the prepared bundles — measure before launching

The frozen prepared bundles show what a full re-preparation costs:

| arm | size |
|---|---|
| G0 | 51.3 GB |
| G1 | 58.4 GB |
| G2 | 60.1 GB |
| **total** | **~170 GB** |

Each arm holds five `PATIENT_FOLD_<n>.pt` payloads of roughly 11–13 GB. Free space on
149 at the time of writing:

| mount | available |
|---|---|
| `/dell_2` (the work root) | 252 GB |
| `/public0` | 503 GB |
| `/public8` | 187 GB |

A typed re-preparation therefore fits on `/dell_2` with about 80 GB to spare, and
comfortably on `/public0`. **Check `df` immediately before launching** — the volume is
shared and other work moves the free-space figure. Running out mid-write leaves partial
fold payloads that must be deleted before retrying.

The script above writes only the *authority* (small). Producing the runtime bundles is a
separate, much larger step; budget the ~170 GB before starting it, not after.

### Why one role per relation

`safe_graph.ROLE_CONTRACTS` is structurally one-role-to-one-relation. A single role
carrying several relation types is refused by the safety boundary — correctly, and
the refusal is a `RuntimeError`, not a silent acceptance. Adding a typed relation
therefore requires **three** declarations: a `ROLE_CONTRACTS` entry, a
`RELATION_POLARITY` entry, and membership of `G1_ROLES`. A context-carrying role
additionally needs an entry in `CONTEXT_BEARING_STATIC_ROLES`, and the condition
there is the exact inverse of the global-binding rule.

---

## 5. Regenerate the ablation configs

```bash
python3 /tmp/phase11_generate_ablations.py
```

Writes `code/config/rbp_ablation/*.yaml` and `manifests/RBP_ABLATION_FAIRNESS.json`.
`assert_fair_comparison` must report **FAIR**.

---

## 6. What is runnable today

| mode | needs ENCODE? | runnable |
|---|---|---|
| A `legacy_generic_binding` | no | **yes** |
| B `typed_binding_only` | no | **yes** |
| C `typed_binding_plus_encode_eclip` | yes | blocked |
| D `full_rbp_evidence` | yes | blocked |

Modes A and B answer the plan's core Phase 3 question — does preserving assay identity
change the model — and are untouched by the assembly gate. They still require the new
graph authority variant to be materialised first.

---

## 7. Absolute prohibitions on this host

* do not overwrite anything under `${PRIVATE_WORK_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/`;
* do not modify `${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/`;
* do not start a paid GPU instance — see `RUN_GPU.md`;
* do not recompute the LASSO baseline; it is reused by digest.
"""

GPU = """# RUN_GPU.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Deliverable**: Phase 15 item 10 (execution document, GPU half)
**Date**: 2026-09-21

> ## STATUS: DO NOT START
>
> The CPU gate is **NOT closed**. Of twelve gates, eleven PASS and **gate 3
> (ENCODE manifest SHA) is BLOCKED**: all six ENCODE accessions named by the plan are
> hg19 while the project annotation is GRCh38, and silent liftOver is forbidden.
>
> Starting a paid instance now would burn money on an ablation set that cannot be
> completed. The plan is explicit: *"CPU gate 未闭合前不启动付费 GPU."*

---

## 1. Preconditions — all must hold

| # | precondition | current |
|---|---|---|
| 1 | `RBP_ENCODE_CPU_PREFLIGHT.md` reads PASS | **11/12, gate 3 BLOCKED** |
| 2 | input manifest frozen with SHA256 | yes |
| 3 | new graph authority variant frozen | **not yet materialised** |
| 4 | all hashes written to a manifest | yes |
| 5 | all four ablation configs generated | **yes** |
| 6 | same folds verified across modes | **yes** (`assert_fair_comparison` = FAIR) |
| 7 | no current production output overwritten | yes |
| 8 | VRAM / disk / runtime estimate recorded | **not yet** |

**Preconditions 3 and 8 are outstanding; gate 3 is blocked on a decision.**

---

## 2. Resolve gate 3 first

Pick one path (see `RBP_ENCODE_PHASE5_PHASE6_GATE_REPORT.md` §6):

* **A — substitute hg38-native ENCODE eCLIP accessions** (252 GRCh38 experiments, 168 RBP
  targets available). Keeps the methodology and every context rule; changes only which
  accessions are cited. Recommended.
* **B — explicit, non-silent liftOver.** Requires a pinned chain file with its own SHA256,
  per-peak mapping status, quarantine of unmapped peaks, a new authority variant, and the
  lifting declared in every downstream lineage record.
* **C — drop the ENCODE half** and deliver Phases 0–4.

Then regenerate `manifests/ENCODE_DOWNLOAD_MANIFEST.tsv` with real SHA256 values for every
downloaded file. **A download manifest without hashes does not close gate 3.**

---

## 3. Smoke run — one fold, bounded, mandatory

Only after every precondition holds. The plan: *"先跑 1 fold / bounded smoke."*

```bash
bash 03_train_all_linux.sh --fold 0 --max-steps 50 --arm typed_binding_only
```

The smoke passes only if **all** of the following are observed:

| check | how to confirm |
|---|---|
| forward PASS | run completes without exception |
| backward PASS | gradients non-zero on at least one core parameter |
| loss finite | every logged loss is finite (no `nan`, no `inf`) |
| checkpoint / resume PASS | write a checkpoint, resume from it, identical step count |
| **typed relations present in the graph metadata** | inspect `HeteroData` edge types: `('lncRNA', 'binds_protein_eclip', 'protein')` and the other typed relations must appear — **not** a single `binds_protein` |

The last row is the one that matters most. The plan: *"禁止 GPU 启动后再发现 ENCODE 没进模型."*
If the graph metadata still shows only `binds_protein`, the typed path is not wired and the
run must be stopped before the five-fold job.

---

## 4. Formal five-fold run

Only after the smoke passes on every row above.

```bash
for arm in legacy_generic_binding typed_binding_only \\
           typed_binding_plus_encode_eclip full_rbp_evidence ; do
  bash run_model_workflow.sh config/rbp_ablation/$arm.yaml train
done
```

Requirements carried from the plan and enforced by `assert_fair_comparison`:

* identical patient folds, candidate universe, LASSO base, label definition, seeds,
  training budget and evaluation metric across all four arms;
* `changes_primary_ranking = false` in every arm;
* `auxiliary_gradients_into_core = false` in every arm;
* RBP knockdown stays **evidence-only** and never enters the primary graph;
* the shuffled-eCLIP control uses the same node count and approximately the same degree
  distribution, with real lncRNA–RBP pairings destroyed.

---

## 5. Cost discipline

* create the instance only when every precondition holds — never "to look around";
* the platform is a preemptible CompShare GPU; a 4090 is roughly ¥1.44–2.09/h plus
  ~¥0.07/h disk;
* run the one-fold smoke first and read its output before launching five folds;
* on any launcher failure, validation failure, or prolonged absence of optimizer progress,
  **stop the instance immediately**;
* after results are verified on both local and 149 with matching SHA256, release the
  instance **and its cloud disk**.

---

## 6. Reporting (pre-registered in `RBP_ENCODE_ABLATION_PLAN.md`)

Report AUPRC, AUROC, Precision@K, calibration and ΔAUPRC vs the reused LASSO base for every
arm, with per-fold values. Include the real-vs-shuffled eCLIP comparison and per-assay-subtype
Evidence performance.

**Pre-registered decision rule — no exceptions:**

* if C ≤ B + noise → ENCODE eCLIP does not help; **report FAIL**;
* if C ≈ shuffled eCLIP → the gain is degree, not biology; **report FAIL**;
* if D ≤ C → RBP knockdown adds nothing; report it;
* **no weight may be tuned, no label changed, and no mode redefined to convert a FAIL into
  a PASS.**

ENCODE eCLIP is *training input* in modes C and D and therefore cannot also be presented as
independent external validation. RBP knockdown, if reported, must be labelled **RBP-level
functional support — not lncRNA perturbation validation**.

---

## 7. Teardown

1. verify result SHA256 on local and 149;
2. copy results into the work root and refresh the manifest;
3. `compshare instance delete <id> --release-disk --yes --wait`, or release the disk from the
   console;
4. record the terminal receipt in `reports/`.
"""

(REPORTS / "RUN_149_CPU.md").write_text(CPU, encoding="utf-8")
(REPORTS / "RUN_GPU.md").write_text(GPU, encoding="utf-8")
print("written:")
print("  ", REPORTS / "RUN_149_CPU.md", len(CPU), "chars")
print("  ", REPORTS / "RUN_GPU.md", len(GPU), "chars")
