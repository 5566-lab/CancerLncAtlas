# RBP_CURRENT_CODE_AUDIT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP data integration
**Phase**: 0 (real-code baseline audit, no code modified)
**Date**: 2026-09-21
**Executed on**: server `149` (`MyLabServer`, 172.22.148.149, user `dengsc`)

---

## 1. Authoritative baselines

### 1.1 GitHub `main`

Measured live via the GitHub API during this audit (not assumed from chat history):

| field | value |
|---|---|
| repository | `5566-lab/CancerLncAtlas` |
| branch | `main` |
| **commit SHA** | **`ab982cf43b34900f63309b1eca250e21c6384eea`** |
| tree SHA | `0b81101548c1b3b78c6fcf9b795310c283bb5f13` |
| commit date | 2026-09-21T02:01:03Z |
| commit message | `Add repository text normalization rules` |

### 1.2 Server-side original code tree

The repository has **no `.git` directory on server 149** — the server holds plain source
trees, not a git checkout. The origin tree matching GitHub `main` was located by
SHA256 fingerprinting every `evidence_training.py` under `${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime`:

```
${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/code
```

This is the same capsule recorded in the workspace long-term memory as the v3.2 code/auth
capsule. Full-tree comparison against GitHub `main` (`cc_hhgt/**/*.py`):

| metric | count |
|---|---|
| files in origin tree | 233 |
| files in `main` | 234 |
| **identical (SHA256 match)** | **220** |
| differing | 13 |
| only in `main` | 1 (`cc_hhgt/v32/optimizer_guard_v2.py`) |
| only in origin tree | 0 |

The 13 differing files are `directional_cnv_query`, `download_catalog`, `drug_staging`,
`independent_head_authorized_binding`, `independent_head_runtime_binding`, `mixed_query`,
`orchestration`, `release_registry`, `single_cell_association_scratch`,
`single_cell_formal23_query`, `single_cell_formal_context_query`, `single_cell_gap_audit`,
`unified_staging_bindings` — **none of them in scope for this task**.

### 1.3 Every Phase-0 target file matches between `main` and the origin tree

| file | SHA256 prefix (both) | status |
|---|---|---|
| `cc_hhgt/v32/evidence_training.py` | `b112319c1596cb33` | match |
| `cc_hhgt/v32/evidence_interaction_rematerialization.py` | `ca11954d377fd0a2` | match |
| `cc_hhgt/protein_layer.py` | `82b2f6fde8e1aec5` | match |
| `cc_hhgt/interaction_context.py` | `51034e21ec548d1c` | match |
| `cc_hhgt/v32/formal_graph.py` | `04e3e1b1ead886b8` | match |
| `cc_hhgt/gnn.py` | `f4110aefb0fd532a` | match |
| `config/model_v3_2_full_multitask.yaml` | `e8daea03d6bc27ef` | match |

**Conclusion: the Phase-0 file list in the task plan corresponds to the real code. No plan
revision is required on this ground.**

### 1.4 Single deliberate divergence: `scripts/materialize_v32_g012_graph_authorities.py`

| tree | SHA256 prefix | bytes |
|---|---|---|
| origin (server) | `2e83127065323378` | 43838 |
| GitHub `main` | `0075c48c9ed1286c` | 43830 |

Diff is **10 lines of hard-coded path constants only** — no logic difference:

```
origin:  reads restricted to ${PRIVATE_WORK_ROOT}/CancerLncAtlas or ${PRIVATE_WORK_ROOT}/CancerLncAtlas
         ALLOWED_WRITE_ROOT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas")
main:    reads via ${PRIVATE_WORK_ROOT}
         ALLOWED_WRITE_ROOT = Path("./data/CancerLncAtlas")
```

The `main` variant is the path-sanitised portable form produced during release
preparation. **For execution on 149 the origin (server) variant is authoritative and is
retained in the work root.** This is the only file where the two baselines differ in scope,
and the divergence is cosmetic.

---

## 2. Work root and immutability

A new immutable output root was created. **The source capsule was not modified** (verified by
re-hashing after the copy).

```
${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/
├── code/        copy of the origin tree (233 cc_hhgt .py files)
├── reports/     audit + preflight documents
├── manifests/   download + SHA256 manifests
├── inputs/      newly materialised inputs
├── outputs/     newly materialised outputs
├── runtime/     pytest wheels + --target package dir
└── env.sh       canonical environment
```

Post-copy verification:

| file | copied | source capsule |
|---|---|---|
| `cc_hhgt/v32/evidence_training.py` | `b112319c1596cb33` | `b112319c1596cb33` (unchanged) |
| `cc_hhgt/protein_layer.py` | `82b2f6fde8e1aec5` | — |

No formal V3.2 artifact is read for write, and no formal V3.2 path is an output target.

### 2.1 Environment receipt (Phase 12 prerequisite)

| item | value |
|---|---|
| interpreter | `/usr/bin/python3` (3.10.12) |
| package root | `${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/python_packages_v32_drug_20260825` |
| extra target | `$RBP_ROOT/runtime/py` (pytest only) |
| numpy / pandas | 2.2.6 / 2.3.3 |
| pyarrow / duckdb | 25.0.0 / 1.5.0 |
| **torch** | **2.7.1+cpu** |
| yaml / scipy / sklearn | 5.4.1 / 1.15.3 / 1.7.2 |
| **pytest** | **9.1.1** |
| import matrix | **ALL IMPORTS OK** |

`pip install --target` silently failed to land the `pytest` package on this sshfs mount
(pip reported success while `pytest/` was absent). Resolved by extracting the wheels with
`zipfile` directly. Documented so the GPU/CPU runbooks do not rediscover it.

### 2.2 Network receipt

The proxy configured in the server environment (`http://127.0.0.1:7890`) is **dead**.
A working **SOCKS5 proxy listens on `127.0.0.1:1080`**:

| target | via socks5h://127.0.0.1:1080 |
|---|---|
| `www.encodeproject.org` | `http=200` |
| `github.com` | `http=200` |
| `pypi.org` | `http=200` |

**ENCODE downloads can therefore run directly on 149** (no local-machine staging needed).

---

## 3. Proven chain: raw interaction → HGT forward

Each link below was verified by reading the real code on 149. Line numbers refer to the
work-root copy, which is byte-identical to GitHub `main` for these files.

### Link 1 — raw source → `experiment_raw` + `experiment_family`

`cc_hhgt/v32/evidence_interaction_rematerialization.py`

| line | evidence |
|---|---|
| 276 | output schema field `"experiment_family"` |
| 277 | output schema field `"experiment_raw"` |
| 412, 463, 485, 507, 531 | `"experiment_raw": clean_text(row.get(...))` per source database |
| 748 | `normalized_experiment = experiment_family(extracted["experiment_raw"])` |
| 787–788 | both `experiment_family` and `experiment_raw` written to the row |

**Both fields survive at this stage.** The compression happens downstream, not here.

### Link 2 — the compression function

`cc_hhgt/v32/evidence_interaction_rematerialization.py` lines **321–333**:

```python
def experiment_family(value: object) -> str:
    text = clean_text(value).lower()
    if re.search(r"clip|chirp|chart|rap|rip|pull.?down|immunoprecip|\bip\b|emsa", text):
        return "physical_binding"
    if re.search(r"knock|sirna|shrna|crispr|overexpress|transfect|deplet", text):
        return "functional_perturbation"
    if "luciferase" in text or "reporter" in text:
        return "reporter_assay"
    if re.search(r"qpcr|rt-pcr|rna-seq|microarray|western", text):
        return "expression_or_abundance"
    if not text:
        return "unspecified"
    return "other_experimental"
```

**eCLIP, iCLIP, PAR-CLIP, HITS-CLIP, CLIP, ChIRP, ChART, RAP, RIP, pull-down,
immunoprecipitation, IP and EMSA — every one of them collapses into the single token
`physical_binding`.** This is the root cause of the reported over-compression.

### Link 3 — `experiment_raw` never reaches the Evidence feature

`cc_hhgt/v32/evidence_training.py`

| line | evidence |
|---|---|
| 74–84 | `EVENT_FEATURE_FIELDS = ("route_type", "source_database", "source_dataset", "experiment_type", "relation_type", "tissue", "cell_line", "species", "member_type")` — contains `experiment_type` only |
| 406–409 | `"experiment_type": _column(frame, ["experiment_type", "experiment_family", "assay_type", "experimental_system"])` — **`experiment_raw` is not a candidate** |
| 1431–1450 | `_event_feature_matrix()` hashes each field as a categorical token: `token = f"{field}={value}"` → `sha256` → bucket |

Because `experiment_family` **is** present in the input frame (Link 1) and is second in the
candidate list, `experiment_type` resolves to the coarse value. The fine-grained
`experiment_raw` is available in the frame but is never consulted.

**Note (new finding, not in the task plan):** a *second*, independent column-resolution list
at line **2452** DOES include `experiment_raw`:
`["experiment_type", "experiment_family", "assay_type", "experimental_system", "experiment_raw"]`.
So the codebase contains two inconsistent resolution paths. Only the `_source_columns()`
path (line 384) feeds the EventSet feature fields. This divergence must be reconciled in
Phase 2 rather than patched in one place.

**Note 2:** `_event_feature_matrix()` has no continuous-feature channel at all. Every field
is a hashed categorical; the only numeric slots are 8 fixed trailing columns
(`is_experimental`, `is_computational`, `is_physical`, pmid count, pmid presence,
two route_type indicators, bias). This confirms Phase 8's requirement that eCLIP
quantities need an explicit numeric path.

### Link 4 — Evidence → `protein_layer`

`cc_hhgt/protein_layer.py`

| line | evidence |
|---|---|
| 96–97 | `experimental = mapped.get("is_experimental", ...)`; `mapped["weight"] = np.where(experimental, 1.0, 0.4) * mapped.mapping_weight.fillna(1.0)` |
| 102–110 | `mapped.groupby(["lncrna_id", "protein_id", "cancer_id"], ...).agg(weight=("weight","max"), source_database=("source_database", lambda v: "\|".join(sorted(set(map(str, v))))), ...)` |
| 113 | `grouped["is_context_specific"] = grouped.cancer_id.notna()` |
| 139–140 | `global_edges = cancer_id.isna().sum()`, `context_specific_edges = cancer_id.notna().sum()` |

**Assay type is not a grouping key and is not carried through.** The only experiment-derived
signal that reaches the graph is a **binary** `is_experimental` flag mapped to weights
1.0 vs 0.4. eCLIP and RIP are indistinguishable at this layer.

### Link 5 — `protein_layer` → graph relation

`cc_hhgt/v32/formal_graph.py`

| line | evidence |
|---|---|
| 48 | `("lncRNA", "binds_protein", "protein", "static_global_lnc_protein_binding", False)` |
| 35 | allowed roles `{"static_global_lnc_protein_binding", "static_protein_gene_encoding"}` |
| 326 | `required = {"lncrna_id", "protein_id", "weight", "cancer_id", "is_context_specific"}` |
| 341–343 | `relation_type="binds_protein"`, `target_type="protein"`, `edge_role="static_global_lnc_protein_binding"` |

**Every lncRNA–protein main-graph relation is emitted as one flat `binds_protein` type.**

### Link 6 — graph relation → HGT forward (what the model actually consumes)

`cc_hhgt/gnn.py`

| line | evidence |
|---|---|
| 320 | `edge_type = (str(src_type), str(rel), str(dst_type))` |
| 323–326 | per-relation tensors: `edge_index`, `edge_weight` only |
| 342–356 | `edge_index` built from src/dst; `edge_weight` from the `weight` column × `relation_polarity` |
| 400–402 | homogeneous bundle = `{"edge_index", "edge_type", "edge_weight"}` — **no `edge_attr`** |
| 797–821 | forward: `topology = conv(h, graph["edge_index"], graph["edge_type"])`; `weight_message = weighted(weighted_homogeneous_aggregate(h, graph))` |
| 845–848 | per-edge-type path uses `edge_index` and `edge_weight` only |

**Confirmed: the HGT consumes exactly `edge_index`, `edge_type` (= `relation_type`) and
`edge_weight`. `source_database`, `PMID`, `experiment_raw` and `assay_subtype` are never
model features.**

---

## 4. Status of the five reported findings

| id | reported issue | status | evidence |
|---|---|---|---|
| **A** | `evidence_interaction_rematerialization.py` outputs both `experiment_family` and `experiment_raw` | **CONFIRMED** | lines 276–277, 787–788 |
| **B** | `_source_columns()` candidates are `experiment_type`/`experiment_family`/`assay_type`/`experimental_system` with **no** `experiment_raw`; eCLIP/RIP/etc. are compressed to `physical_binding` before the Evidence Transformer | **CONFIRMED** | lines 406–409 + `experiment_family()` 321–333 |
| **C** | `protein_layer.py` aggregates by `lncrna_id + protein_id + cancer_id`, no assay type retained | **CONFIRMED** | line 102–110, 113 |
| **D** | `formal_graph.py` writes all lncRNA–protein main-graph relations as `binds_protein` | **CONFIRMED** | line 48, 341–343 |
| **E** | `gnn.py` message passing uses only `edge_index`, `edge_weight`, `relation_type`; `source_database`/`PMID`/`experiment_raw` are not HGT features | **CONFIRMED** | lines 320, 400–402, 797–821 |

**All five findings are accurate against the real code. No plan revision is required.**

---

## 5. Additional findings that change the Phase 3 / Phase 7 design

These were not in the task plan and materially affect implementation.

### 5.1 The main-graph binding materialiser is hard-coded to *global-only* binding

`scripts/materialize_v32_g012_graph_authorities.py` (origin variant), lines 336–347:

```python
binding = binding.loc[
    binding.cancer_id.isna()          # global / context-free ONLY
    & binding.lncrna_id.ne("")
    & binding.lncrna_id.isin(candidate_lncs)
    & binding.protein_id.str.startswith("UNIPROT:")
]
...
binding["cancer_id"] = pd.Series(pd.NA, index=binding.index, dtype="string")
binding["is_context_specific"] = False
```

This is **exactly** the mechanism described in the task plan: a context-specific HepG2→LIHC
eCLIP edge would be silently filtered out here. Phase 7's `static_context_lnc_rbp_eclip`
edge role is therefore not optional — without it ENCODE eCLIP cannot reach the main graph
at all.

### 5.2 There is a frozen row-count invariant that any new binding rows will break

Lines 351–354 of the same file:

```python
binding = binding.drop_duplicates(["lncrna_id", "protein_id"], keep="first")
if len(binding) != 623_207 or binding.lncrna_id.nunique() != 2_567:
    raise RuntimeError(f"Global binding invariant failed: rows={len(binding)}, ...")
```

The global binding table is pinned at **623,207 rows / 2,567 lncRNAs**. Any additive ENCODE
integration must go through a *new* authority variant with its own declared invariant
rather than mutating this constant — otherwise the formal G012 authority either fails
closed or is silently redefined. This must be resolved explicitly in Phase 3.

### 5.3 `drop_duplicates(["lncrna_id","protein_id"])` is a second, independent collapse point

Even if Phase 3 types the relations, this line collapses all assay classes for a pair down
to one row in the *global* table. The typed relations must be produced in a new table
(keyed on `graph_assay_class`), not by relaxing this call.

### 5.4 Cell-line context mapping status (Phase 7 premise verified)

`cc_hhgt/interaction_context.py` lines 15–45, `CELL_LINE_CANCER_HINTS`:

| cell line | mapped context |
|---|---|
| **HEPG2** (line 18) | **LIHC** ✓ present |
| HUH7 (line 19) | LIHC |
| … 27 other cancer lines … | … |
| **K562** | **ABSENT** |

The file's own comment (lines 12–14) states that lineage-ambiguous entries are deliberately
absent and excluded from strict LOCO graphs. **Therefore K562 eCLIP will not receive a
cancer context under the existing authority**, which is precisely the behaviour Phase 7
requires. No new K562→LAML mapping may be introduced.

### 5.5 Configuration switch block confirmed absent

`config/model_v3_2_full_multitask.yaml` declares modules
`exact_pathway`, `state`, `clinical`, `single_cell`, `mutation_cnv`, `drug`, `interaction`,
`evidence` — and contains **no `rbp_evidence` block**. The Phase 11 switch block is
genuinely new configuration, not an existing toggle.

---

## 6. Answers to the final-report questions that Phase 0 can already settle

| q | question | status |
|---|---|---|
| G | does typed experiment information reach the forward pass? | **No** — proven by Link 3 + Link 6. Mechanism to change it is Phase 2 + Phase 3. |
| H | does ENCODE eCLIP reach the forward pass? | **No** — no ENCODE reference exists in the code tree at all (verified previously: 0 true `ENCODE` hits; the 399 grep hits were `GENCODE` and Python `ENCODERS_BY_TYPE`). |
| C | which typed relations does CC-HHGT actually have today? | **One**: `binds_protein`. |

---

## 7. Deviations from the task plan

| # | deviation | justification |
|---|---|---|
| 1 | The plan's instruction to run `git rev-parse HEAD` cannot be executed on 149 — **there is no git repository there**. | Substituted a SHA256 fingerprint match of every candidate `evidence_training.py` against GitHub `main`, plus a full 234-file tree comparison. Baseline is established more strongly than a bare SHA read. |
| 2 | Work is performed in a **new work root** (`rbp_encode_v33_20260921_r1`) rather than in the origin capsule. | Mandated by the plan ("new immutable output root") and by the "do not overwrite V3.2 formal artifacts" rule. |
| 3 | `scripts/materialize_v32_g012_graph_authorities.py` is taken from the **origin (server)** variant, not from `main`. | The two differ only in path constants; `main`'s variant is the release-sanitised portable form and cannot execute on 149. |
| 4 | pytest was bootstrapped by direct wheel extraction instead of `pip install --target`. | `pip --target` reported success while silently not creating the package directory on this sshfs mount. |

---

## 8. Gate status at end of Phase 0

| item | status |
|---|---|
| Baseline established against real code | **PASS** |
| Chain raw→forward proven with line-level evidence | **PASS** |
| Findings A–E adjudicated | **PASS** (all 5 confirmed) |
| New work root created, origin capsule untouched | **PASS** |
| Python/torch/pytest environment operational | **PASS** |
| ENCODE reachability from 149 | **PASS** (SOCKS5 1080) |
| Code modified | **NONE** (Phase 0 forbids it) |
| Paid GPU started | **NO** |

**Next**: Phase 1 — deterministic assay taxonomy module `cc_hhgt/v32/rbp_assay.py`.
