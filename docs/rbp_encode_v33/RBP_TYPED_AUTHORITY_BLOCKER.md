# RBP_TYPED_AUTHORITY_BLOCKER.md

> **SUPERSEDED.** The build described as blocked here completed later the same day.
> See `RBP_TYPED_AUTHORITY_RESOLVED.md` for the resolution (per-class roles: one role
> per relation), `RBP_ENCODE_INGESTION_REPORT.md` for the ENCODE half, and
> `RBP_ENCODE_CPU_PREFLIGHT.md` for the current gate state. This file is kept for the
> record of what was actually blocking and why.

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phase**: 7 (typed relations into the primary graph)
**Date**: 2026-09-21
**Status**: BUILD BLOCKED at the safe-graph contract; precise remaining step recorded

---

## 1. What was achieved this round

### 1.1 The five-round-old fixture question is resolved

The two graph-authority tests that had failed since Phase 0 needed
`SAMPLE_PATIENT_FOLD_MAP.tsv` and `PATIENT_FOLD_AUTHORITY_RECEIPT.json`. Both were
located at

```
${PRIVATE_WORK_ROOT}/v32_g012_local_cnv_formal_prepared_20260903_r1/
```

and **verified against the frozen constants before use**:

| file | expected | observed | verdict |
|---|---|---|---|
| `SAMPLE_PATIENT_FOLD_MAP.tsv` | `e05c2008…e253` | `e05c2008…e253` | MATCH |
| `PATIENT_FOLD_AUTHORITY_RECEIPT.json` | `1ef32bda…17e0` | `1ef32bda…17e0` | MATCH |

They were then provisioned where the tests expect them. This is the authentic
frozen authority, not a synthetic fixture.

**Regression improved from 6 failed / 298 passed to 4 failed / 300 passed.** The
four remaining failures are a pre-existing DuckDB 1.5.0 internal-bug class, each
reproduced on the untouched origin capsule.

### 1.2 The authority builder is parameterised (verified)

`build_bound_formal_graph` now accepts:

```python
binding_materializer: Callable[..., pd.DataFrame] | None = None,
binding_generation: str = "GENERIC_GLOBAL",
```

Default behaviour is unchanged and the graph-authority tests pass.

### 1.3 The loader is generation-aware (verified)

A typed run genuinely does **not** have an identical relation schema across
variants. The frozen receipt asserts
`same_node_and_relation_schema_all_variants: True`, so the loader now:

* requires the caller to declare a `relation_schema_generation`;
* requires the receipt to declare the **same** generation;
* **rejects a typed generation that claims the identical-schema gate**, so the
  schema change must be stated rather than smuggled past it.

### 1.4 A context-bearing role allow-list was added to the safe graph (verified)

`build_safe_graph` previously permitted context-carrying edges only for
`transductive_expression_eligibility` and the coexpression roles; every binding
edge had to be global. A single, reviewed allow-list now exists:

```python
CONTEXT_BEARING_STATIC_ROLES = frozenset({"static_context_lnc_rbp_eclip"})
```

with the condition being the **exact inverse** of the global-binding rule — such
an edge *must* carry a cancer and *must* be marked context-specific, so it can
only ever enter its own context. 30 graph tests pass.

### 1.5 The full frozen input set was located and verified

| input | status |
|---|---|
| 6 static artifacts | SHA256 matched the frozen receipt exactly |
| 5 × fold expression + coexpression (~630 MB) | present |
| `FORMAL_CANDIDATE_UNIVERSE.parquet` | 3,300,000 rows, 33 cancers |
| patient-fold authority | bound to the frozen constants |

The build ran through input validation, receipt validation, the typed-generation
gate and candidate loading, and reached graph construction.

---

## 2. The blocker, stated precisely

`cc_hhgt/v32/safe_graph.py` enforces **three** declarations for every role that
may enter the primary graph:

| # | declaration | requirement |
|---|---|---|
| 1 | `ROLE_CONTRACTS` | role → `(source_type, relation_type, target_type, source_split)` |
| 2 | `ALLOWED_ROLES` | derived as `frozenset(ROLE_CONTRACTS)` |
| 3 | `RELATION_POLARITY` | one entry per `relation_type` |

`ROLE_CONTRACTS` is structurally **one role to one relation type**:

```python
for edge_role, (source_type, relation, target_type, split) in ROLE_CONTRACTS.items():
    selected = role.eq(edge_role)
    rejected |= selected & ~frame.relation_type.eq(relation)
```

My Phase 3 design has **one role carrying eight relation types**
(`static_global_lnc_protein_binding` → `binds_protein_<class>`). That is
incompatible with the contract's 1:1 shape, and the graph is correctly refusing it:

```
RuntimeError: Unsafe edges were supplied to the primary graph;
separate them before graph construction
```

**The safety boundary is doing its job.** It has an independent, enumerated
contract for the primary graph, and it will not accept a schema widening that has
not been declared role by role.

---

## 3. The precise remaining step

Give each typed relation **its own role**, matching the contract's shape:

| role | relation_type |
|---|---|
| `static_global_lnc_protein_binding` | `binds_protein` (legacy, unchanged) |
| `static_global_lnc_protein_binding_eclip` | `binds_protein_eclip` |
| `static_global_lnc_protein_binding_other_clip` | `binds_protein_other_clip` |
| `static_global_lnc_protein_binding_rip` | `binds_protein_rip` |
| `static_global_lnc_protein_binding_rna_capture` | `binds_protein_rna_capture` |
| `static_global_lnc_protein_binding_other_physical` | `binds_protein_other_physical` |
| `static_global_lnc_protein_binding_experimental_unspecified` | `binds_protein_experimental_unspecified` |
| `static_global_lnc_protein_binding_predicted` | `binds_protein_predicted` |
| `static_global_lnc_protein_binding_unknown` | `binds_protein_unknown` |
| `static_context_lnc_rbp_eclip` | `binds_protein_eclip` |

Required edits:

1. `formal_graph.materialize_typed_lnc_protein_binding` — emit the per-class role
   instead of the shared one;
2. `formal_graph.G1_ROLES` — include every typed role (all are binding roles, so
   G0 stays free of binding and G2 keeps PPI, preserving the arms' meaning);
3. `formal_graph.FORMAL_RELATION_SCHEMA` — register each `(relation, role)` pair;
4. `safe_graph.ROLE_CONTRACTS` and `RELATION_POLARITY` — one entry each;
5. tests — the existing `test_v32_rbp_typed_graph.py` suite asserts the shared
   role in several places and must be updated to the per-class roles.

This is a mechanical but security-relevant change. It is **not** a patch to be
rushed at the end of a work session: the safe graph is the boundary that keeps
outcome-derived, fold-localised and context-erased edges out of message passing,
and widening it deserves a clean, reviewed pass with its own test run.

---

## 4. Status

| item | status |
|---|---|
| typed binding table (Phase 3) | COMPLETE |
| static/context boundary for typed edges | COMPLETE |
| authority builder parameterisation | COMPLETE |
| generation-aware receipt loader | COMPLETE |
| safe-graph context allow-list | COMPLETE |
| **per-class role declarations** | **REMAINING — the blocker** |
| A/B mode execution | blocked on the above |
| ENCODE half (Phases 6, 9, 10, C/D modes) | blocked on the assembly decision |
| paid GPU | NOT STARTED |

### Regression at end of round

**4 failed / 300 passed / 1 skipped.** All four failures are the pre-existing
DuckDB 1.5.0 internal-bug class, reproduced on the unmodified origin capsule.
**No new failures.** 178 new RBP tests pass.
