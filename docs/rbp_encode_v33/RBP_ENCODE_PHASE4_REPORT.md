# RBP_ENCODE_PHASE4_REPORT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phase**: 4 (cross-database duplicate audit + canonical experiment collapse)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. What Phase 4 was asked to establish

> "同一原始实验不能因为被 RNAInter 和 NPInter 同时收录就变成两个独立实验."

The plan anticipated that one wet-lab experiment indexed by several curated
databases would inflate apparent evidence multiplicity, and asked for a
`canonical_experiment_key` collapse with the source list preserved.

The audit was run on the real frozen `interaction_relation.parquet`
(**6,160,707 rows**) using the standardised fields already present there.

### Source composition

| source_database | rows | share |
|---|---|---|
| RNAInter | 5,510,363 | 89.4 % |
| NPInter | 635,758 | 10.3 % |
| LncTarD | 8,343 | 0.14 % |
| LncACTdb | 3,955 | 0.06 % |
| LncRNA2Target | 2,288 | 0.04 % |

---

## 2. Headline finding: the RBP path cannot be de-duplicated at all

**Every one of the 348,364 lncRNA–RBP rows comes from RNAInter, and RNAInter
contributes zero PMIDs.**

| source_database | rows | with PMID | coverage |
|---|---|---|---|
| **RNAInter** | 5,510,363 | **0** | **0.0 %** |
| NPInter | 635,758 | 635,758 | 100.0 % |
| LncTarD | 8,343 | 8,343 | 100.0 % |
| LncACTdb | 3,955 | 3,955 | 100.0 % |
| LncRNA2Target | 2,288 | 2,288 | 100.0 % |

The root cause is **upstream**, not a pipeline defect. The RNAInter4 source table's
own columns are:

```
RNAInterID, Interactor1.Symbol, Category1, Species1,
Interactor2.Symbol, Category2, Species2,
Raw_ID1, Raw_ID2, score, strong, weak, predict
```

There is **no PMID column in RNAInter4 at all**. Nothing downstream can recover it.

PMID presence by partner type confirms the split is structural:

| partner_type | rows | with PMID | coverage |
|---|---|---|---|
| tf | 2,830,640 | 1,580 | 0.1 % |
| protein | 2,586,646 | 509,867 | 19.7 % |
| **rbp** | **348,364** | **0** | **0.0 %** |
| mirna | 212,300 | 118,481 | 55.8 % |
| mrna | 78,161 | 7,150 | 9.1 % |
| gene | 5,917 | 5,917 | 100.0 % |
| pcg | 5,080 | 5,063 | 99.7 % |

Within the RBP rows, **even the physical assays have no PMID**:

| graph_assay_class | rows | with PMID |
|---|---|---|
| predicted | 227,624 | 0 |
| other_clip | 84,968 | 0 |
| eclip | 35,195 | 0 |
| experimental_unspecified | 484 | 0 |
| rip | 80 | 0 |
| rna_capture | 9 | 0 |
| other_physical | 4 | 0 |
| **physical subtotal** | **120,256** | **0** |

### Consequence for the plan's Phase 4 objective

The objective is **structurally inapplicable to the RBP path**:

* NPInter carries 100 % PMIDs but labels its partners `protein`, never `rbp`;
* RNAInter carries the `rbp` rows but 0 % PMIDs.

The two can therefore **never be cross-checked for the same experiment**. Any claim
that RBP evidence has been de-duplicated across databases would be false. This
limitation is recorded here rather than papered over, and it materially qualifies
what can be asserted about RBP evidence independence in the final report.

---

## 3. Cross-database duplication in the PMID-anchored subset is small

Canonical collapse over the whole table:

```json
{
  "input_rows": 6160707,
  "rows_with_pmid": 480781,
  "rows_without_pmid": 5679926,
  "canonical_experiments": 96093,
  "canonical_collapsed_rows": 384688,
  "canonical_experiments_seen_in_multiple_databases": 14,
  "unanchored_rows_preserved_separately": 5679926,
  "output_rows": 5776019
}
```

Read carefully:

* **Only 7.8 % of rows carry a PMID**, so the collapse is confined to that subset.
* 480,781 anchored rows reduce to **96,093 canonical experiments** — the great
  majority of that reduction is *within* RNAInter/NPInter record duplication, not
  across databases.
* **Only 14 canonical experiments are seen in more than one database.**

So the specific failure mode the plan worried about — "RNAInter and NPInter both
index the same experiment, creating two independent observations" — **does occur,
but is rare (14 experiments)**, not systemic. The real multiplicity problem in this
dataset is the opposite one: 92 % of rows have no publication anchor at all.

---

## 4. The `canonical_experiment_key` implementation

New module: `cc_hhgt/v32/rbp_canonical.py`

```
canonical_experiment_key(lncrna_id, partner_id, pmid, assay_subtype, cell_line, tissue)
    -> "CEXPV1:" + sha256(field1 \x1f field2 ...)[:32]
```

* **Deterministic** — a pure function of the six declared fields; verified by
  repeated evaluation.
* **Case- and padding-insensitive** — `" hepg2 "` and `"HepG2"` are the same
  experiment.
* **Placeholder-aware** — `NA`, `unknown`, `-`, `nan` normalise to absent, so a
  missing cell line never silently becomes a distinct experiment.
* **Assay-aware** — `assay_subtype` is part of the key, so an eCLIP and a RIP
  experiment on the same pair and PMID stay distinct (a test asserts this).

### Fail-closed rule for PMID-less rows

A row without a PMID is **never merged**. Without a publication anchor there is no
way to distinguish "one experiment indexed twice" from "two experiments", so the
row keeps its own identity. The audit reports them separately
(`unanchored_rows_preserved_separately`), and `duplicate_report()` surfaces
cross-database groupings using a *relaxed* key **for reporting only** — never for
collapsing. Guessing here would silently destroy evidence.

Collapsed rows carry the required provenance:

| column | meaning |
|---|---|
| `source_record_count` | how many records formed this experiment |
| `source_database_list` | sorted `\|`-joined distinct databases |
| `source_database_count` | how many distinct databases |

---

## 5. Relationship to the main graph

The plan requires that graph weighting "本来就不能重复计权". This already holds and
was verified in Phase 3: the typed materialiser groups by
`(lncrna_id, protein_id, cancer_id, graph_assay_class)` with `weight = max`, so ten
duplicate database records for one eCLIP pair produce **one** edge — and a Phase 3
test asserts exactly that (`n_source_records=10`, one row).

Phase 4 therefore adds the **Evidence-layer** half of the guarantee: the same
experiment cannot be counted as several independent events when it is canonicalised
by `(pair, PMID, assay, cell line, tissue)`.

---

## 6. Tests

`tests/test_v32_rbp_canonical.py` — **20 tests, all passing**

* key determinism (20× repeat), namespacing/versioning, field-order contract;
* **every one of the six key components changes the key** (parametrised);
* case/padding invariance and placeholder normalisation;
* three databases → one experiment with the full provenance triple;
* different assay on the same PMID stays separate; different cell line stays separate;
* **PMID-less rows are never merged** (3 similar rows stay 3, all with
  `source_record_count=1`);
* anchored and unanchored rows handled separately in one frame;
* missing required column raises; genuinely empty input returns empty safely;
* `duplicate_report` surfaces cross-database groups and is empty for a single database.

**Cumulative RBP test count: 112 passing**
(57 taxonomy + 14 evidence features + 17 typed binding + 20 canonical + 4 real-data regressions).

---

## 7. Answers contributed to the final report

| question | answer |
|---|---|
| Do RNAInter and NPInter double-count the same experiment? | Yes, but only **14 canonical experiments** are multi-database. |
| Can RBP evidence be de-duplicated across databases? | **No.** All RBP rows come from RNAInter, which has **0 % PMID coverage**; NPInter, which is 100 % PMID-annotated, never labels partners `rbp`. |
| How much of the evidence has a publication anchor? | 480,781 of 6,160,707 rows (**7.8 %**). |
| Are duplicate records inflating graph weight? | **No** — Phase 3 groups with `weight=max`; a test pins ten duplicate records to one edge. |

---

## 8. Status

| phase | status |
|---|---|
| 0 — real-code baseline audit | COMPLETE |
| 1 — assay taxonomy | COMPLETE |
| 2 — Evidence Transformer | COMPLETE |
| 3 — typed main-graph relations | COMPLETE at binding-table layer |
| **4 — cross-database duplicate audit** | **COMPLETE** |
| 5–15 | not started |
| Paid GPU | NOT STARTED |
| Formal V3.2 artifacts overwritten | NO |
| Origin capsule modified | NO |

### Next actions

1. The plan's own Phase 4 note — "没有 PMID 的记录不要武断合并；只报告潜在重复" — is
   implemented and verified. No further Phase 4 work is outstanding.
2. Phase 5/6: locate the six ENCODE accessions in the project directories first
   (the plan forbids a full-disk search), then download processed reproducible peaks.
   The genome-assembly gate must resolve the project annotation lineage before any
   peak mapping; a mismatch is FAIL-CLOSED and silent liftOver is forbidden.
3. Resolve the `binds_protein_predicted` placement decision (Phase 3 report §8) before
   wiring the typed relation map into a new G012 authority variant.
