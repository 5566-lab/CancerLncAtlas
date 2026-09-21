# RBP_ENCODE_GRAPH_SCHEMA_DIFF.md

**Deliverable**: Phase 15 item 8 (old/new graph schema diff)
**Date**: 2026-09-21

## Relation schema

* before: **9** registered tuples
* after: **18** registered tuples
* binding relations: **1** → **10**

### Binding relations after the change

| relation_type | edge_role |
|---|---|
| `binds_protein` | `static_global_lnc_protein_binding` |
| `binds_protein_eclip` | `static_global_lnc_protein_binding` |
| `binds_protein_other_clip` | `static_global_lnc_protein_binding` |
| `binds_protein_rip` | `static_global_lnc_protein_binding` |
| `binds_protein_rna_capture` | `static_global_lnc_protein_binding` |
| `binds_protein_other_physical` | `static_global_lnc_protein_binding` |
| `binds_protein_experimental_unspecified` | `static_global_lnc_protein_binding` |
| `binds_protein_predicted` | `static_global_lnc_protein_binding` |
| `binds_protein_unknown` | `static_global_lnc_protein_binding` |
| `binds_protein_eclip` | `static_context_lnc_rbp_eclip` |

### Edge roles

| role | before | after |
|---|---|---|
| `static_global_lnc_protein_binding` | present | present (unchanged) |
| `static_context_lnc_rbp_eclip` | absent | **new** |
| `static_protein_gene_encoding` | present | present |
| `static_symmetric_ppi` | present | present |

### G0 / G1 / G2 membership

`G1_ROLES` is now:

```
static_context_lnc_rbp_eclip
static_global_lnc_protein_binding
static_protein_gene_encoding
```

Both binding roles are G1 roles, so the scientific meaning of the three arms is unchanged:

| arm | content | changed? |
|---|---|---|
| G0 | no binding of any kind | **no** |
| G1 | + typed lncRNA–protein binding (global and approved context eCLIP) + protein→gene | extended only by approved binding roles |
| G2 | G1 + PPI | **no** |

### Node vocabulary

**Unchanged.** No new node type. `rbp` partners map into the existing `protein` node space via the same `protein_gene_map` authority, in the same `UNIPROT:` / `GENE:` identifier space, so no `NONO:protein` / `NONO:RBP` duplication is constructible. A test asserts no `RBP` token appears in any emitted `protein_id` or `lncrna_id`.

### A constraint discovered while wiring this

`EDGE_KEYS = (source_type, source_id, relation_type, target_type, target_id)` carries neither `edge_role` nor `cancer_id`. The graph can therefore hold only **one edge per (source, relation, target) triple**, so a pair supported both by a context-free record and by a context-specific eCLIP record collides. The collision is resolved explicitly: **the context-specific edge wins** and the global duplicate is dropped. Keeping the global edge would broadcast context-specific evidence to every cancer.

### Assay vocabulary

* `assay_subtype`: 18 values
* `graph_assay_class`: 8 values — `eclip, other_clip, rip, rna_capture, other_physical, experimental_unspecified, predicted, unknown`

Only `graph_assay_class` is permitted to type a graph relation, which keeps relation-type cardinality bounded.
