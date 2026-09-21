# RBP_ENCODE_CPU_PREFLIGHT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Deliverable**: Phase 15 item 5 (CPU hard gate)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. Gate summary

| # | gate | status | evidence |
|---|---|---|---|
| 1 | relevant pytest PASS | **PASS** | 300 passed / 1 skipped; per-phase logs in `reports/` |
| 2 | full suite, no new failures | **PASS** | 4 failed / 300 passed; the four are a pre-existing DuckDB 1.5.0 internal-bug class, each reproduced on the pristine capsule |
| 3 | ENCODE manifest SHA | **BLOCKED** | nothing downloaded — assembly mismatch (see §6) |
| 4 | genome assembly gate | **PASS** | project is GRCh38, verified at coordinate level |
| 5 | RBP → UniProt mapping audit | **PASS** | §2 |
| 6 | assay_subtype missingness report | **PASS** | §3 |
| 7 | cross-database duplicate audit | **PASS** | Phase 4 report |
| 8 | graph count report | **PASS** | §4 |
| 9 | context leakage test | **PASS** | 0 leaking edges, §4 |
| 10 | G0/G1/G2 invariant | **PASS** | Phase 3 supplement, 19 tests |
| 11 | synthetic forward/backward | **PASS** | 13 passed (incl. `test_v32_shared_encoder_group_backward`, `test_v32_gpu_backward_probe`) |
| 12 | legacy mode reproduces old generic semantics | **PASS** | 755,346 rows, zero column mismatches |

**Overall: CPU gate NOT closed — gate 3 is blocked on a decision, not on work.**
No paid GPU may start while gate 3 is open.

---

## 2. Gate 5 — RBP → UniProt mapping audit

| | legacy (`protein` only) | typed (`protein` + `rbp`) |
|---|---|---|
| input relation rows | 813,653 | 977,061 |
| unmappable partner rows | 8,940 | 8,940 |
| mapped fraction | 98.90% | **99.09%** |

* every emitted `protein_id` starts with `UNIPROT:` — **True**
* distinct proteins: **15,048**
* distinct lncRNAs: **4,678**

Including `partner_type == "rbp"` adds 163,408
input rows **without adding a single unmappable partner** (8,940 in both), so the RBP
rows use the same identifier space and the same authority as protein rows. No entity
duplication is possible; RBPs remain `node_type = protein`.

---

## 3. Gate 6 — assay_subtype missingness

Total typed rows: **918,266**

| graph_assay_class | rows |
|---|---|
| `predicted` | 773,283 |
| `other_clip` | 56,245 |
| `eclip` | 46,629 |
| `experimental_unspecified` | 38,688 |
| `rna_capture` | 3,174 |
| `rip` | 209 |
| `other_physical` | 23 |
| `unknown` | 15 |

**Uninformative rows (`unknown` + `experimental_unspecified`): 38,703
(4.21%).**

So **95.79% of rows receive a concrete assay
class**, against a legacy pipeline that emitted a single `physical_binding` token for
98.6 % of NPInter rows. Vocabulary coverage: 17 of the
18
declared `assay_subtype` values are exercised by real data.

---

## 4. Gates 8 and 9 — graph counts and context leakage

### 4.1 The headline number

| | legacy main graph | typed, physical only |
|---|---|---|
| global binding edges | 724,502 | **102,425** |
| difference | | **622,077** |

**85.9% of the legacy main-graph lncRNA–protein edges are
sequence-based predictions** (`MATCH algorithm`, `catRAPID`). They were emitted as flat
`binds_protein` edges with no distinguishing mark. Under the conservative default
(`include_predicted=False`) they are excluded from message passing; the switch exists so
an ablation can measure them deliberately.

### 4.2 Typed graph, physical only (117,951 edges)

| relation_type | edges |
|---|---|
| `binds_protein_other_clip` | 44,042 |
| `binds_protein_experimental_unspecified` | 38,680 |
| `binds_protein_eclip` | 34,921 |
| `binds_protein_rip` | 189 |
| `binds_protein_rna_capture` | 86 |
| `binds_protein_other_physical` | 18 |
| `binds_protein_unknown` | 15 |

* context-specific edges: **15,526**
* global edges: **102,425**
* **context edges without a cancer: 0**

### 4.3 Typed graph, predictions admitted (891,234 edges)

| relation_type | edges |
|---|---|
| `binds_protein_predicted` | 773,283 |
| `binds_protein_other_clip` | 44,042 |
| `binds_protein_experimental_unspecified` | 38,680 |
| `binds_protein_eclip` | 34,921 |
| `binds_protein_rip` | 189 |
| `binds_protein_rna_capture` | 86 |
| `binds_protein_other_physical` | 18 |
| `binds_protein_unknown` | 15 |

### 4.4 Context leakage gate — **PASS**

`0` context-specific edges
lack a cancer assignment in either configuration. Every context edge carries
`is_context_specific=True` and a concrete `cancer_id`; none was globalised.

### 4.5 Context policy actually applied

| | rows |
|---|---|
| context-specific input rows | 82,825 |
| mapped to a cancer | 31,807 |
| excluded (unmapped or ambiguous) | **51,018** (61.6%) |
| context-specific lost to global | **0** |

61.6% of contextual rows are **excluded rather than
guessed** — this is where K562-class cell lines land. The authority maps HepG2→LIHC and
has no K562 entry, so K562 eCLIP can never be broadcast to a cancer.

---

## 5. Gate 12 — legacy equivalence (the ablation mode A anchor)

```
produced rows : 755,346
frozen rows   : 755,346
status        : PASS
mismatches    : {}
```

The re-derivation reproduces the frozen binding table column-for-column. Ablation mode A
is therefore provably the old model, not an approximation of it.

---

## 6. Gate 3 — ENCODE manifest SHA: **BLOCKED**

| accession | assembly | status |
|---|---|---|
| `ENCSR456FVU` | `['hg19']` | NOT_DOWNLOADED |
| `ENCSR369TWP` | `['hg19']` | NOT_DOWNLOADED |
| `ENCSR795JHH` | `['hg19']` | NOT_DOWNLOADED |
| `ENCSR413YAF` | `['hg19']` | NOT_DOWNLOADED |
| `ENCSR870OLK` | `['hg19']` | NOT_DOWNLOADED |
| `ENCSR876DCD` | `['hg19']` | NOT_DOWNLOADED |

All six resources named by the plan are **hg19**; the project annotation is **GRCh38**.
No file was downloaded and no peak mapping was attempted. Full detail and the three
resolution options are in `RBP_ENCODE_PHASE5_PHASE6_GATE_REPORT.md` §6.

---

## 7. Ablation readiness

`assert_fair_comparison` returns **FAIR** for the four modes. Frozen axes:

| axis | value |
|---|---|
| `seed` | `20260726` |
| `n_folds` | `5` |
| `candidate_universe_sha256` | `cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f` |
| `lasso_base_sha256` | `6c1cd82c5ad14baa76b180befc4ed69aad5232759af0213d9e04926ad0cfb066` |
| `label_definition` | `V32_FORMAL_STRONG_PLUS_WEAK_POSITIVE` |
| `patient_fold_manifest_sha256` | `e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253` |
| `training_budget_epochs` | `40` |
| `learning_rate` | `0.002` |
| `batch_size` | `128` |
| `evaluation_metric` | `AUPRC` |

The LASSO baseline is **reused, not re-materialised**: `lasso_base_sha256` is the
composite digest over the 8 existing artifacts in `posttraining_lasso_full/`
(`manifests/LASSO_BASE_MANIFEST.tsv`).

Modes **A and B need no ENCODE** and can run once the graph authority variant is
materialised. Modes C and D are blocked with gate 3.

---

## 8. Environment gaps found and closed during this phase

| defect | impact | resolution |
|---|---|---|
| `torch_geometric` staging dirs were **incomplete extractions** (missing `utils`, `nn`) | the entire GNN path — and therefore GPU training — could not run | extracted the official `torch_geometric-2.8.0-py3-none-any.whl` from the project wheelhouse; **SHA256 verified against `PYG280_WHEEL_SHA256.tsv`** (`1f62e415…dada`) |
| `xxhash` absent (only a cp313 wheel existed; interpreter is cp310) | `torch_geometric` import failure | fetched the cp310 manylinux wheel and extracted it |
| broken `sympy` (namespace-only, no `__init__.py`) | every `torch.optim.*` construction failed | installed a complete sympy 1.14.0 (Phase 1) |
| `pip install --target` silently loses packages on this sshfs mount | pytest unusable | extract wheels with `zipfile` directly |
| `artifacts/v32_patient_fold_authority_20260829_r1/` absent, failing 2 graph-authority tests | two tests could not run at all | the authentic frozen authority was located at `inputs/v32_g012_local_cnv_formal_prepared_20260903_r1/`; **both files were verified against the frozen constants** (`e05c2008…e253` for the sample/patient map, `1ef32bda…17e0` for the receipt) before being provisioned where the tests expect them. This is the real authority, not a synthetic fixture. Baseline improved 6 failed → 4 failed. |

`torch_geometric 2.8.0`, `HGTConv`, `HeteroConv`, `HeteroData` and the project's own
`cc_hhgt.gnn` all import cleanly.

---

## 9. Verdict

**CPU gate NOT closed.** Eleven of twelve gates pass; gate 3 is blocked by an upstream
resource mismatch, not by unfinished work.

* No paid GPU was started.
* No formal V3.2 artifact was overwritten.
* The immutable origin capsule is unmodified.

### What may proceed without the ENCODE decision

* materialising the new G012 graph authority variant with typed relations
  (modes A and B);
* the Phase 15 run documents (`RUN_149_CPU.md`, `RUN_GPU.md`).

### What may not

* any GPU launch, until gate 3 closes.
