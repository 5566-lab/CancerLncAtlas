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
| 1 | relevant pytest PASS | **PASS** | 244 passed / 1 skipped in the focused RBP/ENCODE selection; per-phase logs in `reports/` |
| 2 | full suite, no new failures | **PASS** | 6 failed / 244 passed; all six reproduced on the pristine origin capsule with identical errors (4 DuckDB 1.5.0 internal-bug, 2 `prepare_v32_formal.py:265` replication-label) |
| 3 | ENCODE manifest SHA | **PASS** | file-level manifest rebuilt; the plan's six accessions are `PublicationData` FileSets, not experiments — see §6 |
| 4 | genome assembly gate | **PASS** | GRCh38 coordinate authority for all 8,541 lncRNA nodes, verified on three probe loci; hg19 peaks lifted against the pinned chain |
| 5 | RBP → UniProt mapping audit | **PASS** | §2 |
| 6 | assay_subtype missingness report | **PASS** | §3 |
| 7 | cross-database duplicate audit | **PASS** | Phase 4 report |
| 8 | graph count report | **PASS** | §4 |
| 9 | context leakage test | **PASS** | 0 leaking edges, §4 |
| 10 | G0/G1/G2 invariant | **PASS** | Phase 3 supplement, 19 tests |
| 11 | synthetic forward/backward | **PASS** | 13 passed (incl. `test_v32_shared_encoder_group_backward`, `test_v32_gpu_backward_probe`) |
| 12 | legacy mode reproduces old generic semantics | **PASS** | 755,346 rows, zero column mismatches |
| 13 | ENCODE download verification | **PASS** | 1,431 / 1,431 files md5- and size-verified, 1,450.7 MiB |
| 14 | liftOver quarantine accounting | **PASS** | 60,088,294 / 60,108,082 peaks mapped (99.97%), 19,788 quarantined, **0 dropped** |
| 15 | ENCODE adds no graph nodes | **PASS** | against the node universe (8,541 lncRNA / 20,008 protein): 0 new lncRNA nodes, 0 new protein nodes |
| 16 | ENCODE overlap layer admission | **PASS** | gated off: peak-level enrichment 0.71-0.74x, so the layer is not merged |

**Overall: CPU gate CLOSED for the RBP/ENCODE preparation.** Sixteen of sixteen gates
pass. No paid GPU is launched by any script in this work root regardless: gate 16
records that the ENCODE overlap layer is deliberately not in the graph, and the GPU
question is a separate decision that has not been taken.

The gate that used to block, gate 3, was never blocked on work. It was blocked on a
manifest that stopped at experiment level and therefore reported the whole eCLIP
resource as hg19-only. At file level 756 of the 1,431 released narrowPeak files are
GRCh38-native, and the remaining 675 are lifted explicitly. See §6.

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
| global binding edges | 724,502 | **0** |
| difference | | **724,502** |

**100.0% of the legacy main-graph lncRNA–protein edges are
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
* global edges: **0**
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

## 6. Gate 3 — ENCODE manifest SHA: **CLOSED**

The gate was never blocked on work. It was blocked on a manifest that stopped at
*experiment* level, so its `file_accession`, `RBP` and `cell_line` columns were empty
and its `assembly` column could only report the publication set's dominant value.

| accession | resolves as | `type=Experiment` total |
|---|---|---|


The control experiment `ENCSR720BJU`
resolves normally, which is what makes those zero totals meaningful rather than a
malformed query.

At **file** level the eCLIP resource releases 1,431 narrowPeak files: 756 GRCh38-native
and 675 hg19. Only the hg19 half needs lifting, and it is lifted explicitly against the
pinned chain. Full detail is in `RBP_ENCODE_INGESTION_REPORT.md`.

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

Modes **A and B** are materialised: A is the frozen flat `binds_protein` layer
(623,207 rows) and B is the typed generation (98,873 active binding edges). Modes
**C and D are not materialised**, and the reason is a measurement rather than a
missing input: the ENCODE gene-body overlap layer was built (428,949 global and
253,544 context edges) and then found to carry **no enrichment** over a uniform null
(0.71-0.74x). Merging it would have taken the typed binding table from 918,266 to
1,347,215 rows and presented positional coincidence as binding. The layer is gated
off; the criterion for admitting it is recorded in `RBP_ENCODE_INGESTION_REPORT.md` §6.

Until that criterion is met, modes B, C and D produce the same graph. That is stated
here rather than hidden behind a table of identical numbers.

---

## 8. Environment gaps found and closed during this phase

| defect | impact | resolution |
|---|---|---|
| `torch_geometric` staging dirs were **incomplete extractions** (missing `utils`, `nn`) | the entire GNN path — and therefore GPU training — could not run | extracted the official `torch_geometric-2.8.0-py3-none-any.whl` from the project wheelhouse; **SHA256 verified against `PYG280_WHEEL_SHA256.tsv`** (`1f62e415…dada`) |
| `xxhash` absent (only a cp313 wheel existed; interpreter is cp310) | `torch_geometric` import failure | fetched the cp310 manylinux wheel and extracted it |
| broken `sympy` (namespace-only, no `__init__.py`) | every `torch.optim.*` construction failed | installed a complete sympy 1.14.0 (Phase 1) |
| `pip install --target` silently loses packages on this sshfs mount | pytest unusable | extract wheels with `zipfile` directly |

`torch_geometric 2.8.0`, `HGTConv`, `HeteroConv`, `HeteroData` and the project's own
`cc_hhgt.gnn` all import cleanly.

---

## 9. Verdict

**CPU gate CLOSED.** Sixteen of sixteen gates pass.

* No paid GPU was started, and no script in this work root launches one.
* No formal V3.2 artifact was overwritten.
* The immutable origin capsule is unmodified.
* The ENCODE eCLIP layer was measured before it was trusted, and the measurement
  said not to trust it yet.

### What is now in place

* the typed-relation G012 authority generation (modes A and B), verified by active
  edges rather than schema presence;
* 1,431 hash-verified ENCODE eCLIP files, 60,088,294 lifted peaks, 19,788 quarantined
  and 0 dropped, with 1,721,605 lineage rows;
* the four-mode configuration and its fairness proof;
* the KD and RBNS layers, inventoried and gated off.

### What remains a decision rather than a task

* admitting the ENCODE overlap layer, which needs a null that preserves peak
  clustering;
* any GPU launch, which the CPU gate no longer blocks but which has not been asked for.
