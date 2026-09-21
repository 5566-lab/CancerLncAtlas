# RBP_ENCODE_PHASE2_PHASE4_REPORT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phases covered**: 2 (Evidence Transformer) and 4 (cross-database assay audit)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## Part A — Phase 2: Evidence Transformer now knows which experiment it saw

### A.1 What was changed

`cc_hhgt/v32/evidence_training.py`

| location | change |
|---|---|
| imports | `from .rbp_assay import classify_assay` |
| `LEGACY_EVENT_FEATURE_FIELDS` | new constant holding the **original** field tuple |
| `EVENT_FEATURE_FIELDS` | now the assay-aware tuple: `experiment_type` replaced by `experiment_family` + `assay_subtype` |
| `event_feature_fields(*, preserve_assay_type)` | selects the field list per ablation mode |
| `_source_columns()` | resolves `experiment_raw`, `experiment_family`, `assay_subtype`, `graph_assay_class` |
| `_normalize_source_rows()` | classifies each row and emits the four taxonomy columns |
| `build_exact_event_bags()` | carries the four columns into `events`, `physical_facts` |
| `_event_feature_matrix()` | takes `preserve_assay_type` and hashes the selected field list |
| `build_bag_examples()`, `run_evidence_training()` | thread the flag through |

`cc_hhgt/v32/evidence_streaming_training.py` — the DuckDB streaming path had to be kept
column-identical to the batch path (`test_v32_evidence_streaming_training` asserts this).
Updated in four places: `_normalized_raw_schema`, the `physical_all` projection, the
`all_exact_events` view, and `event_public_columns`.

### A.2 Why the field list is configuration-driven rather than simply renamed

`_event_feature_matrix()` hashes `f"{field}={value}"` into a bucket. **Renaming a field
changes its bucket even when the value is unchanged.** A naive rename of
`experiment_type` → `assay_subtype` would therefore silently change the *legacy* model's
forward output and destroy the comparability of ablation mode A.

The two field lists are consequently kept genuinely distinct, and mode A selects the
original tuple. A test recomputes the mode-A matrix by hand from
`LEGACY_EVENT_FEATURE_FIELDS` and asserts bit-for-bit equality.

### A.3 Two real defects found by the new tests (both fixed)

**(1) `experiment_raw` was being falsified.** When the raw string was absent, the code
fell back to the coarse family for classification and then wrote **the fallback value**
into `experiment_raw`. That misrepresents a coarse family token as observed text.
Fixed: classification may use the fallback; `experiment_raw` always records the *true*
original (empty when absent).

**(2) Flag/class inconsistency for textual predictions.** A string matched by the
computational rule returned `graph_assay_class="predicted"` while still reporting
`is_experimental=True, is_predicted=False`. Fixed: the flags are now **derived from the
class**, so a prediction can never claim to be an observation.

### A.4 An input-contract finding

`evidence_interaction_rematerialization.py` defines two column tuples:

| tuple | table | carries `experiment_raw`? |
|---|---|---|
| `OUTPUT_COLUMNS` (line 246) | `interaction_relation` | **yes** |
| `EMPTY_EVENT_COLUMNS` (line 292) | `evidence_event` | **no** — only `experiment_family` |

Both tables feed `_normalize_source_rows`. Therefore:

* binding rows arriving via **`interaction_relation`** (where NPInter/RNAInter land) get
  the **full fine-grained taxonomy**;
* rows arriving via **`evidence_event`** can only fall back to the coarse family.

This is a deliberate scope decision, not an oversight: `EMPTY_EVENT_COLUMNS` is duplicated
with strict tuple-equality assertions in **five** files, so widening it is an input-contract
change that belongs with the Phase 5/6 materialisation work rather than Phase 2. It is
tracked as an explicit Phase 6 prerequisite. The behaviour is covered by a test
(`test_coarse_family_fallback_recovers_physical_semantics`).

### A.5 Tests

`tests/test_v32_rbp_evidence_features.py` — 14 tests, all passing:

* eCLIP vs RIP produce **different** EventSet feature rows under `preserve_assay_type=True`;
* under `preserve_assay_type=False` they produce **identical** rows (the legacy collapse,
  pinned so mode A stays honest);
* the legacy matrix reproduced by hand from the original field list matches exactly;
* the taxonomy columns reach `events`; `experiment_raw` is verbatim;
* `experiment_raw` is absent from both feature field lists (unbounded bucket risk);
* predicted rows never become experimental; unknown does not raise.

**Test totals: 71 passing** (43 Phase-1 taxonomy + 14 real-data regression + 14 Phase-2).

---

## Part B — Phase 4: what the real data actually contains

Source tables located at `${PRIVATE_WORK_ROOT}/CancerLncAtlas/input/` (the same paths named in
`SOURCE_SPECS`). Analysis run with DuckDB over the real files; results written to
`manifests/PHASE4_NPINTER_ASSAY_COUNTS.json`.

### B.1 NPInter5 comprehensive table

`npinter5_human_lncRNA_interactions.comp.tsv` — 173 MB, **635,758 rows**

Partner types:

| partner_type | rows |
|---|---|
| protein | 509,866 |
| miRNA | 117,056 |
| mRNA | 7,150 |
| lncRNA | 701 |
| ncRNA / pseudogene / snRNA / snoRNA / DNA | 1,682 |

Columns include **`methods`** (the assay string) and **`partner_type`** — this is the
table that carries RBP-relevant, fine-grained assay information.

### B.2 Question A — how many relations carry a concrete assay type

**Every one of the 635,758 rows has a non-empty `methods` string** (`rows with empty
methods: 0`). The historical pipeline had a concrete assay available for 100 % of them
and used it for nothing but a 4-way family collapse.

The 500 distinct `methods` values are genuine assay names. Top entries:

| methods | rows | new subtype |
|---|---|---|
| `eCLIP` | 153,411 | eclip |
| `PAR-CLIP` | 130,304 | par_clip |
| `Conserved miRNAs target sites predicted by TargetScan and miRanda overlap with the AGO CLIP dataset` | 116,499 | **computational_prediction** |
| `CLIP-Seq` | 64,153 | clip_unspecified |
| `CLIP` | 60,565 | clip_unspecified |
| `HITS-CLIP` | 39,801 | hits_clip |
| `iCLIP` | 15,467 | iclip |
| `easyCLIP` | 13,056 | clip_unspecified |
| `CHART-seq` | 3,094 | chart |
| `fPARCLIP` | 3,249 | clip_unspecified |

### B.3 Question B — how much was collapsed

| | rows |
|---|---|
| historically classified `physical_binding` | **626,756** |
| …of which now receive a finer subtype | **509,928** |
| share of NPInter5 flattened into one token | **98.6 %** |

Corrected distribution under the new taxonomy (all 635,758 rows):

| graph_assay_class | rows |
|---|---|
| other_clip | 339,972 |
| **eclip** | **166,399** |
| **predicted** | **116,803** |
| experimental_unspecified | 9,002 |
| rna_capture | 3,370 |
| rip | 184 |
| other_physical | 28 |

| assay_subtype | rows |
|---|---|
| eclip | 166,399 |
| clip_unspecified | 148,749 |
| par_clip | 135,389 |
| computational_prediction | 116,803 |
| hits_clip | 40,326 |
| iclip | 15,508 |
| other_experimental | 8,931 |
| chart | 3,097 |
| rip | 184 |
| rna_pulldown | 174 |
| chirp | 99 |
| reporter_assay | 92 |
| emsa / functional_perturbation / expression_or_abundance | 2 each |
| co_ip | 1 |

### B.4 The most serious finding: 116,803 predictions were presented as binding evidence

The single largest `methods` value after the genuine CLIP assays is:

```
"Conserved miRNAs target sites predicted by TargetScan and miRanda
 overlap with the AGO CLIP dataset"                              116,499 rows
```

This is a **computational target-site prediction**. It merely *references* the AGO CLIP
dataset as an overlap set. The historical regex matched the substring `clip` and recorded
**all 116,499 rows as `physical_binding`**.

With the corrected taxonomy the total reaching `predicted` is **116,803 rows — 18.6 % of
everything the old pipeline called physical binding.** Those rows were being used as
experimental interaction evidence in the Evidence layer.

This is a correctness defect in the shipped V3.2 evidence chain, independent of the ENCODE
work, and it is now fixed and regression-tested.

### B.5 Two taxonomy defects that only real data could reveal (both fixed)

**(1) Prefixed CLIP variants were demoted.** A strict left boundary
(`(?<![a-z0-9])clip`) excluded `easyCLIP` (13,056), `fPARCLIP` (3,249), `fCLIP` (976),
`GoldCLIP` (484), `pCLIP` (375), `seCLIP-Seq` (238), `PARCLIP` (237) — **18,378 genuine
CLIP rows** that the historical regex correctly called physical binding. That would have
been a *regression* against legacy. Fixed with a bounded alphabetic-prefix allowance;
a test asserts the allowance still rejects `transcript`/`description`/`manuscript` prose.

**(2) Rule precedence.** Prediction language now outranks assay keyword matching, so a row
that self-describes as predicted can never be recorded as an observation.

Both fixes have dedicated regression tests built from the real strings.

### B.6 Caveat on the protein/RBP question

`partner_type` is literally `protein` (509,866 rows) — NPInter does not label RBP
membership. Determining which of those partners are *bona fide* RBPs requires an external
RBP gene list and is deferred to the Phase 3/4 protein-authority mapping step. The
**assay-side** counts above are unaffected by that: they are computed over all rows.

---

## Part C — Regression status

Command: the RBP/evidence/graph/interaction subset in `$RBP_CODE`.

| run | result |
|---|---|
| before any change | 3 failed, 61 passed |
| after Phase 1 + env repair | 2 failed, 105 passed |
| **after Phase 2 (with streaming + extra evidence files added to the set)** | **6 failed, 151 passed** |

The six failures are **all pre-existing** and each was proven to fail identically on the
pristine immutable capsule:

| failure | cause | proof |
|---|---|---|
| 4 × `test_v32_evidence_streaming_training.py` | DuckDB 1.5.0 `InternalException: Attempted to access index 3 within vector of size 3` in `COPY (...)` | ran the same tests on the untouched origin capsule: **4 failed, identical** |
| 2 × `test_v32_formal_prepare_graph_authority.py` | `artifacts/v32_patient_fold_authority_20260829_r1/` absent — **also absent in the origin capsule** | direct directory check |

**New failures introduced by Phase 2: zero.**

Environment gap additionally recorded: `fastapi` is missing, so
`tests/test_v32_evidence_streaming_website.py` cannot be collected. Not relevant to the
RBP evidence chain.

---

## Part D — Status

| phase | status |
|---|---|
| 0 — real-code baseline audit | COMPLETE |
| 1 — assay taxonomy | COMPLETE (71 tests) |
| **2 — Evidence Transformer** | **COMPLETE** (14 tests, legacy equivalence pinned) |
| **4 — cross-database assay audit** | **COMPLETE for NPInter5**; RNAInter4 / LncTarD2 / LncRNA2Target2 / LncACTdb4 pending |
| 3 — typed main-graph relations | not started |
| 5–15 | not started |
| Paid GPU | NOT STARTED (CPU gate open) |
| Formal V3.2 artifacts overwritten | NO |
| Origin capsule modified | NO |

### Next actions

1. Extend the Phase 4 audit to the other four databases (note: **RNAInter4 has no method
   column at all** — 3,267,964 rows that can only ever reach the coarse family; this must
   be stated plainly in the final report).
2. Phase 3: typed binding relations in `protein_layer.py` / `formal_graph.py`, including
   the `graph_assay_class` grouping key and the frozen 623,207-row invariant.
3. Decide the `EMPTY_EVENT_COLUMNS` widening as part of Phase 6.
