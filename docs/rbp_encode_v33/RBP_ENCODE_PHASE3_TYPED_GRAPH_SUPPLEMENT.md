# RBP_ENCODE_PHASE3_TYPED_GRAPH_SUPPLEMENT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Scope**: typed binding relations wired into the formal G0/G1/G2 graph (Phase 3 completion + Phase 7 prerequisites)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. What was added

`cc_hhgt/v32/formal_graph.py` only. Nothing else in the graph chain was touched.

### 1.1 New edge role

```python
GLOBAL_BINDING_ROLE = "static_global_lnc_protein_binding"   # unchanged
CONTEXT_ECLIP_ROLE  = "static_context_lnc_rbp_eclip"        # new
```

`CONTEXT_ECLIP_ROLE` is the role the plan's Phase 7 requires. It admits
context-specific eCLIP edges **with `cancer_id` preserved and
`is_context_specific=True`** — they are never globalised.

`G1_ROLES` now contains both binding roles, so:

| arm | content |
|---|---|
| **G0** | no binding of any kind (unchanged meaning) |
| **G1** | + typed lncRNA–protein binding (global and approved context eCLIP) + protein→gene |
| **G2** | G1 + PPI |

The scientific meaning of the three arms is unchanged: a binding role is still a
binding role, and PPI is still G2-only.

### 1.2 Relation schema

`FORMAL_RELATION_SCHEMA` grew from 9 to 18 rows — the eight
`binds_protein_<graph_assay_class>` types plus a context-scoped eCLIP row:

```
binds_protein                       static_global_lnc_protein_binding   (legacy, kept)
binds_protein_eclip                 static_global_lnc_protein_binding
binds_protein_other_clip            static_global_lnc_protein_binding
binds_protein_rip                   static_global_lnc_protein_binding
binds_protein_rna_capture           static_global_lnc_protein_binding
binds_protein_other_physical        static_global_lnc_protein_binding
binds_protein_experimental_unspecified  static_global_lnc_protein_binding
binds_protein_predicted             static_global_lnc_protein_binding
binds_protein_unknown               static_global_lnc_protein_binding
binds_protein_eclip                 static_context_lnc_rbp_eclip        (new, context)
```

Registering them was **mandatory**: `build_formal_graph_authority` rejects any
emitted tuple that is not in the schema
(`"Fresh graph emitted an unregistered typed relation"`), so attempting this
without the registration would have failed loudly rather than silently.

### 1.3 New builder

```python
materialize_typed_lnc_protein_binding(
    binding, *, include_predicted=False, include_context_eclip=True,
    candidate_lncrnas=None,
)
```

`materialize_global_lnc_protein_binding` is **left byte-identical**. The frozen
G012 authority, its `623,207 rows / 2,567 lncRNAs` invariant, and its
`drop_duplicates(["lncrna_id","protein_id"])` behaviour are all untouched. The
typed path is purely additive, which is why no invariant constant had to be
renegotiated.

---

## 2. A graph-key constraint discovered while wiring this

`EDGE_KEYS` in `cc_hhgt/v32/safe_graph.py` is:

```python
('source_type', 'source_id', 'relation_type', 'target_type', 'target_id')
```

It carries **neither `edge_role` nor `cancer_id`**. Consequently the graph can
hold **only one edge per (source, relation, target) triple** — a pair supported
both by a context-free record and by a context-specific eCLIP record collides.

The first implementation let `_max_magnitude` resolve the collision, and it kept
the **global** edge. That is the dangerous direction: it takes
context-specific evidence and presents it as applying to every cancer, which the
task explicitly forbids ("不得把 K562/HepG2 等 context-specific ENCODE 数据偷偷广播成
pan-cancer/global").

The collision is now resolved **explicitly, before** `_max_magnitude`:

> when both a global and a context-specific edge exist for the same triple, the
> **context-specific edge wins** and the global duplicate is dropped.

Dropping the global edge under-claims (the relation is asserted only for the
mapped cancer); keeping it would over-claim. Under-claiming is the safe
direction, and it is the direction the task's constraints require.

This is pinned by
`test_context_edge_takes_precedence_over_a_global_edge_for_the_same_pair`.

---

## 3. Predictions are opt-in, defaulting to excluded

`include_predicted` defaults to **`False`**.

Phase 3's measurement on real data: **84.2 % of typed lncRNA–RBP relations are
sequence-based predictions** (773,283 of 918,266). Admitting them into message
passing by default would present roughly 773k predictions as measurements. The
switch exists so an ablation can measure their contribution deliberately — this
is the placement decision flagged in the Phase 3 report §8, implemented with the
conservative default and an explicit opt-in rather than chosen silently.

---

## 4. Tests

`tests/test_v32_rbp_typed_graph.py` — **19 tests, all passing**

Typed emission: one relation type per assay class; a declared `relation_type`
that disagrees with `graph_assay_class` raises; an out-of-vocabulary class raises.

Predictions: excluded by default, admitted only on request.

Context: a context-specific eCLIP edge keeps `cancer_id` and its own role and is
never globalised; **non-eCLIP context rows are not admitted at all** (only eCLIP
has an approved context role); context admission can be switched off; a single
mapped cancer never fans out; candidate restriction applies to both paths.

**Collision rule**: context beats global for the same pair; a global edge
survives when nothing competes.

**G0/G1/G2 invariants**: binding roles are absent from G0 and present from G1;
G1 still has no PPI and G2 does; `CONTEXT_ECLIP_ROLE` is a binding role and not a
G2-only role; an unknown arm raises; edges emitted by the typed builder select
correctly in all three arms (`G0` none, `G1` all, `G2` all).

**Regression: 6 failed, 232 passed, 1 skipped** — the same six pre-existing
failures (4 × DuckDB 1.5.0 incompatibility, 2 × missing `artifacts/` fixture),
each previously reproduced on the untouched origin capsule. **Zero new
failures.** Notably `tests/test_v32_formal_graph_contract.py` still passes with
the enlarged schema.

---

## 5. Constraints verified

| constraint | status |
|---|---|
| No new RBP node type | held — `node_type` vocabulary untouched |
| Context-specific data never globalised | held — enforced by an explicit collision rule and tested |
| G0/G1/G2 scientific meaning unchanged | held — role membership only extended to approved binding roles |
| Frozen G012 invariant untouched | held — legacy builder and materialiser unmodified |
| Formal V3.2 artifacts not overwritten | held — no formal path written |
| No paid GPU | held |

---

## 6. Status

| phase | status |
|---|---|
| 0 — real-code baseline audit | COMPLETE |
| 1 — assay taxonomy | COMPLETE |
| 2 — Evidence Transformer | COMPLETE |
| **3 — typed main-graph relations** | **COMPLETE** (binding table + formal graph + G0/G1/G2 invariants) |
| 4 — cross-database duplicate audit | COMPLETE |
| 5 — ENCODE resource audit + manifest | COMPLETE |
| 6 — eCLIP peak mapping | FAIL-CLOSED (assembly mismatch) — awaiting the §6 decision in the Phase 5/6 gate report |
| 7 — context eCLIP edge role | **edge role and admission policy implemented and tested**; the ENCODE half is blocked on Phase 6 |
| 8–15 | not started |

**Cumulative RBP test count: 131 passing**
(57 taxonomy + 14 evidence features + 17 typed binding + 19 typed graph + 20 canonical + 4 real-data regressions).

### What remains blocked and what does not

Blocked on the ENCODE-path decision (Phase 5/6 report §6):
Phases 6, 9, 10, and the ENCODE half of 7 and 13.

Not blocked, and available now:
* Phase 8's numeric eCLIP feature channel — the Evidence-side work does not
  depend on which ENCODE accession set is chosen, only on its schema;
* Phase 11's `rbp_evidence` configuration block and the four ablation modes;
* Phase 12's CPU preflight, for everything except the ENCODE manifest hash gate.
