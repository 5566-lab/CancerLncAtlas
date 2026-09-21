# RBP_TYPED_AUTHORITY_RESOLVED.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phase**: 7 (typed relations into the primary graph)
**Date**: 2026-09-21
**Status**: **RESOLVED** — supersedes `RBP_TYPED_AUTHORITY_BLOCKER.md`

---

## 1. The blocker and its cause

`build_safe_graph` refused the typed relations:

```
RuntimeError: Unsafe edges were supplied to the primary graph;
separate them before graph construction
```

The cause was a **structural mismatch**, not a bug. `cc_hhgt/v32/safe_graph.py`
enforces three declarations per role, and `ROLE_CONTRACTS` is
**one-role-to-one-relation**:

```python
for edge_role, (source_type, relation, target_type, split) in ROLE_CONTRACTS.items():
    rejected |= role.eq(edge_role) & ~frame.relation_type.eq(relation)
```

The original design had **one role carrying eight relation types**, which that
contract cannot express. The safety boundary was correct to refuse it.

## 2. The resolution: one dedicated role per relation

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
| `static_context_lnc_rbp_eclip` | `binds_protein_eclip` (context-scoped) |

Files changed:

| file | change |
|---|---|
| `cc_hhgt/v32/formal_graph.py` | `typed_global_role()` helper; `TYPED_GLOBAL_ROLES`; `G1_ROLES` extended with every typed role; schema rows re-pointed to per-class roles; `materialize_typed_lnc_protein_binding` emits the per-class role |
| `cc_hhgt/v32/safe_graph.py` | nine `ROLE_CONTRACTS` entries; eight `RELATION_POLARITY` entries; `CONTEXT_BEARING_STATIC_ROLES` allow-list |
| `tests/test_v32_rbp_typed_graph.py` | assertions updated from the shared role to per-class roles |

**G0/G1/G2 meaning is unchanged**: every typed role is a binding role, so it
belongs to G1 and G2 and is absent from G0, exactly like the legacy binding role.
PPI remains G2-only.

## 3. Consistency verification

A dedicated check asserts the whole declaration set lines up:

```
G1_ROLES          : 11
schema rows       : 18
ALLOWED_ROLES     : 18
RELATION_POLARITY : 17
  typed roles not in G1_ROLES      : none
  typed roles lacking a contract   : none
  context role not in allow-list   : none
  schema/contract alignment problems: NONE
  emitted but undeclared relations : none
  emitted but undeclared roles     : none
RESULT: CONSISTENT
```

17 relation types for 18 roles, because `binds_protein_eclip` is shared by the
global and context eCLIP roles — which is exactly why they need distinct roles.

## 4. Build result — fold 0

```
generation : TYPED_ASSAY_CLASS_V1
receipt    : GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json (bdd82248ff27a627…)
edges      : 7,136,794
G0 / G1 / G2 : 6,863,061 / 6,981,942 / 7,136,794
```

### The arms are arithmetically exact, not approximately right

| difference | edges | composition |
|---|---|---|
| G2 − G1 | 154,852 | PPI only |
| G1 − G0 | 118,881 | binding 98,873 + protein→gene 20,008 |

`98,873 + 20,008 = 118,881` exactly, and `20,008` is the authority's declared
`protein_gene` row count. The typed build therefore reproduces the known graph
structure while adding the typed binding relations.

| relation_type emitted | edges |
|---|---|
| `binds_protein_other_clip` | 36,998 |
| `binds_protein_eclip` | 31,402 |
| `binds_protein_experimental_unspecified` | 30,191 |
| `binds_protein_rip` | 176 |
| `binds_protein_rna_capture` | 75 |
| `binds_protein_other_physical` | 16 |
| `binds_protein_unknown` | 15 |

**`binds_protein_predicted` is absent** — the conservative default
(`include_predicted=False`) is in force, so the 622,077 prediction edges measured
in Phase 12 stay out of message passing unless an ablation opts in.

The binding payload records its provenance:

```
binding_generation   : TYPED_ASSAY_CLASS_V1
binding_materializer : materialize_typed_lnc_protein_binding
```

## 5. Constraints held

| constraint | status |
|---|---|
| No new RBP node type | held — `node_type` vocabulary untouched; RBPs remain `protein` |
| Context-specific data never globalised | held — enforced by the safe-graph allow-list, which requires a cancer **and** the context flag |
| G0/G1/G2 scientific meaning unchanged | held — verified arithmetically above |
| Frozen G012 authority untouched | held — a new receipt generation was written; the frozen one is read-only input |
| Formal V3.2 artifacts not overwritten | held — all output under the new work root |
| No paid GPU | held |

## 6. Cost note for the run book

One fold takes roughly **25 minutes** (8.25M coexpression rows merged against
3.3M candidate keys over NFS). A full five-fold materialisation is therefore about
**two hours** of wall time. The five-fold run is launched in the background.

## 7. Status

| item | status |
|---|---|
| typed binding table | COMPLETE |
| safe-graph and relation-schema declarations | COMPLETE, verified consistent |
| typed authority, fold 0 | **COMPLETE** |
| typed authority, folds 1–4 | running |
| A/B ablation execution | unblocked once all five folds land |
| ENCODE half (Phases 6, 9, 10, C/D modes) | still blocked on the assembly decision |
| paid GPU | NOT STARTED |
