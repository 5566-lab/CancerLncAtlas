# RBP_ENCODE_PHASE3_REPORT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phase**: 3 (typed binding relations in the main CC-HHGT graph)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. Where the main graph's binding table actually comes from

The task plan names `cc_hhgt/protein_layer.py` as the file to modify. Tracing the real
call graph produced a more precise picture:

* `protein_layer.build_contextual_lnc_protein` has **no live caller** — the only importer
  of `protein_layer` is `cc_hhgt/graph_build.py`, and `graph_build` is imported by
  **nothing** (it is exercised only by `tests/test_graph_build.py`).
* **However, its OUTPUT is a frozen input to the current formal pipeline.** The G012
  authority source list
  (`config/v32_g012_authority_sources_20260830_r1.json`) points the graph materialiser at

  ```
  ${PRIVATE_WORK_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/
    V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized/
    lncRNA_protein_relation_context.parquet
  ```

  which is exactly the filename `protein_layer.py` writes (line 130).

So `protein_layer.py` is not dead: it is the **provenance of a frozen artifact**. Editing
it alone would change nothing, because the formal pipeline reads the frozen file.

The frozen `standardized/` directory still contains everything needed to re-derive the
table, including the 333 MB `interaction_relation.parquet`, so the typed variant can be
materialised **into the new work root without touching any formal artifact**:

| file | rows | role |
|---|---|---|
| `interaction_relation.parquet` | 6,160,707 | upstream source, carries `experiment_raw` and `partner_type` |
| `lncRNA_protein_relation_context.parquet` | 755,346 | the frozen binding table (no assay column) |
| `protein_gene_map.parquet` | 20,008 | partner → protein authority |
| `dim_lncRNA.parquet` / `dim_cancer.parquet` | 36,016 / 33 | identifier and context authority |

The frozen binding table's columns are:

```
lncrna_id, protein_id, cancer_id, weight, source_database, n_source_records,
n_pmids, mapping_multiplicity, relation_type, is_context_specific,
mapping_status, edge_id
```

**No assay column of any kind.** The information was destroyed before the graph was built.

---

## 2. The headline finding: every explicitly-labelled RBP is silently dropped

`protein_layer.py` line 62 selects partners with:

```python
is_protein = relation.partner_type.astype(str).str.lower().str.contains("protein")
```

Real `partner_type` distribution in the frozen `interaction_relation` (6,160,707 rows):

| partner_type | rows | passes `contains("protein")` |
|---|---|---|
| tf | 2,830,640 | 0 |
| **protein** | 2,586,646 | 2,586,646 |
| **rbp** | **348,364** | **0** |
| mirna | 212,300 | 0 |
| mrna | 78,161 | 0 |
| dna | 75,675 | 0 |
| other (17 classes) | ~28,921 | 0 |

The string `"rbp"` does not contain the substring `"protein"`, so **all 348,364 rows whose
partner is explicitly labelled as an RBP are excluded from the main graph's binding
edges.** Measured impact:

| | count |
|---|---|
| distinct (lncRNA, partner) pairs on `rbp` rows | 162,362 |
| …not already reachable via `partner_type == "protein"` | **147,643** |
| `rbp` rows with an empty `experiment_raw` | **0** |

This is a direct, quantified defect in the RBP chain and it is independent of ENCODE.

**No entity duplication is introduced by fixing it.** The partner identifiers on `rbp` rows
are `GENE:ENSG…`, identical in form to `protein` rows, and are mapped through the same
`protein_gene_map` authority. RBPs therefore stay `node_type = protein`, and no
`NONO:protein` / `NONO:RBP` pair can arise. A test asserts no `RBP` token ever appears in
the emitted `protein_id` or `lncrna_id`.

---

## 3. The typed materialiser

New module: `cc_hhgt/v32/rbp_typed_binding.py`

```python
@dataclass(frozen=True)
class BindingTyping:
    include_rbp_partner_type: bool = False
    group_by_assay_class: bool = False
    emit_predicted_relation: bool = True
```

`build_typed_lnc_protein_binding(...)` re-derives the table with two opt-in switches, and
`verify_legacy_equivalence(produced, frozen)` compares a legacy-mode run against the frozen
artifact column by column.

Grouping key changes from
`(lncrna_id, protein_id, cancer_id)` to
`(lncrna_id, protein_id, cancer_id, graph_assay_class)`.

Within each cell the plan's aggregation rule is implemented exactly:

| output | aggregation |
|---|---|
| `weight` | `max` (pre-registered, no re-weighting) |
| `source_database` | sorted `\|`-joined distinct set |
| `n_source_records` | `nunique` |
| `n_pmids` | `sum` of pmid presence |
| `mapping_multiplicity` | `max` |
| `assay_subtype`, `experiment_family`, `experiment_raw` | sorted `\|`-joined distinct set |

Relation type is derived from the class via `graph_assay_relation_type()`, which fails
closed to `binds_protein_unknown`, and the emitted classes are validated against the closed
vocabulary (`GRAPH_ASSAY_CLASSES`) with a hard error otherwise.

---

## 4. Legacy equivalence: PASS

```
produced rows : 755,346
frozen rows   : 755,346
equivalence   : PASS
mismatches    : {}        (weight, source_database, n_source_records, n_pmids)
```

With both switches off the re-derivation reproduces the frozen table **exactly** — same row
count, same distinct-lncRNA count, and zero value mismatches across every comparable
column. This is the Phase 12 gate #12 anchor ("legacy mode reproduces the old generic
semantics") and it is now evidence, not an assertion.

---

## 5. Typed mode: what would actually enter the graph

```
frozen rows                       :   755,346
typed rows                        :   918,266
frozen distinct (lnc,protein)     :   742,012
typed  distinct (lnc,protein)     :   897,758
pairs added                       :   155,746
```

| relation_type | rows | share |
|---|---|---|
| **`binds_protein_predicted`** | **773,283** | **84.2 %** |
| `binds_protein_other_clip` | 56,245 | 6.1 % |
| `binds_protein_eclip` | 46,629 | 5.1 % |
| `binds_protein_experimental_unspecified` | 38,688 | 4.2 % |
| `binds_protein_rna_capture` | 3,174 | 0.35 % |
| `binds_protein_rip` | 209 | 0.02 % |
| `binds_protein_other_physical` | 23 | — |
| `binds_protein_unknown` | 15 | — |

### This is the single most important number in the phase

**84.2 % of everything the typed materialiser can produce is computational prediction.**
If the RBP rows were added to the main graph *without* the typed distinction — i.e. by
simply widening the `contains("protein")` filter — then 773,283 predicted
lncRNA–RBP relations would enter the heterogeneous graph carrying `weight` 0.4 and the
label `binds_protein`, indistinguishable from a measured eCLIP edge.

The typing is therefore not a refinement; it is the safety mechanism that makes the RBP
integration permissible at all. It is what allows the ablation to ask "does the model
benefit from measured RBP binding?" rather than silently answering "the model has been
given 773k predictions as if they were data".

The two dominant predicted strings are:

| experiment_raw | rows |
|---|---|
| `Scan pipeline widely used MATCH algorithm` | 197,094 |
| `catRAPID` | 30,276 |

`catRAPID` is a sequence-based interaction prediction server. It was **not** recognised by
the taxonomy during Phase 2 and fell through to `experimental_unspecified`; that was fixed
in this phase (`catrapid` added to the prediction rule) and is covered by a test. Without
that fix 30,276 predictions would have been typed as experimental.

---

## 6. Constraints verified

| constraint | status | evidence |
|---|---|---|
| No new RBP node type | **held** | `rbp` partners map into the existing `UNIPROT:`/`GENE:` protein space; test asserts no `RBP` token in emitted ids |
| No `NONO:protein` / `NONO:RBP` duplication | **held** | same authority table, same identifier space, single `protein_id` column |
| G0/G1/G2 scientific meaning unchanged | **held** | relation typing changes `relation_type` only; membership in G0/G1/G2 is decided by `edge_role`, which is untouched |
| Don't overwrite formal V3.2 artifacts | **held** | output written to `$RBP_ROOT/outputs/phase3_typed_binding/` |
| Don't tune weights to chase results | **held** | weight is `max` of the pre-existing weight; no new weighting invented |
| Same PMID in several databases ≠ independent experiments | **pending Phase 4** | Phase 4 canonical-key collapse not yet applied; noted |
| No paid GPU | **held** | none started |

---

## 7. Tests

`tests/test_v32_rbp_typed_binding.py` — **17 tests, all passing**

Legacy fidelity: emits the historical relation type and no assay column; drops `rbp` rows
exactly as before; `verify_legacy_equivalence` detects both a row-count and a value
divergence and passes on identical tables.

Typed behaviour: `rbp` included without a new entity; one pair yields several typed
relations; **ten duplicate database records collapse to one edge** with
`n_source_records=10`, `n_pmids=10` and a 10-way source list; `max` weight and joined
sources; relation types stay in the closed vocabulary; predictions get their own relation
and never share a row with an eCLIP edge; unknown fails closed; `is_experimental=False`
forces the predicted class so it cannot aggregate with a measured row.

Context: HepG2 maps to LIHC through the existing authority, **K562 is excluded rather than
guessed**, context-free rows stay global, and typing does not alter any of it.

**Cumulative RBP test count: 92 passing** (57 taxonomy + 14 evidence features + 17 typed
binding + 4 real-data prediction regressions).

---

## 8. Status

| phase | status |
|---|---|
| 0 — real-code baseline audit | COMPLETE |
| 1 — assay taxonomy | COMPLETE |
| 2 — Evidence Transformer | COMPLETE |
| **3 — typed main-graph relations** | **COMPLETE at the binding-table layer**; the `formal_graph` relation mapping is designed and tested but not yet wired into a G012 authority variant |
| 4 — cross-database audit | NPInter5 + lncRNA-RBP vocabulary COMPLETE; other databases pending |
| 5–15 | not started |
| Paid GPU | NOT STARTED |

### Open decision for the next round

The typed table offers `binds_protein_predicted`. Three options for the main graph, which
materially change what the model sees:

1. **exclude `predicted` from the main graph entirely** (physical relations only) — the
   most conservative reading of "typed binding";
2. **include it as its own relation type** (what the plan's relation list literally
   specifies), letting the ablation measure its contribution;
3. include it **only in the Evidence layer**, never in message passing.

The 84 % figure above is the reason this deserves an explicit decision rather than a
default. The plan's Phase 3 relation list names `binds_protein_predicted`, which points at
option 2, but option 1 is the safer scientific default and is what the "no prediction as
measurement" principle implies for message passing.

### Next actions

1. Wire the typed relation map into `formal_graph.materialize_global_lnc_protein_binding`
   and a new G012 authority variant, resolving the hard-coded
   `623_207 rows / 2_567 lncRNAs` invariant (which any additive change breaks).
2. Phase 4 canonical-experiment collapse across the five curated databases.
3. Phase 5/6 ENCODE eCLIP.
