# RBP_AB_BASELINE_COMPARISON.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Purpose**: ablations A vs B — what the typed binding generation actually changes in the graph
**Date**: 2026-09-21
**Server**: `149`

---

## 1. Where the numbers come from

**Mode A (legacy).** Read directly from the frozen prepared authority records, so no
rebuild is needed and no re-derivation can drift:

`inputs/v32_g012_local_cnv_formal_prepared_20260903_r1/GRAPH_AUTHORITIES/PATIENT_FOLD_<n>.json`

**Mode B (typed).** `outputs/phase7_typed_authority/fold_0/AUTHORITY_SUMMARY.json`, built
from the same frozen inputs under the `TYPED_ASSAY_CLASS_V1` generation.

---

## 2. Frozen (mode A) counts

| fold | G0 | G1 | G2 | G1 − G0 | G2 − G1 |
|---|---|---|---|---|---|
| 0 | 6,863,061 | 7,506,276 | 7,661,128 | 643,215 | 154,852 |
| 1 | 6,889,245 | 7,532,460 | 7,687,312 | 643,215 | 154,852 |
| 2 | 6,927,631 | 7,570,846 | 7,725,698 | 643,215 | 154,852 |
| 3 | 6,951,342 | 7,594,557 | 7,749,409 | 643,215 | 154,852 |
| 4 | 6,902,331 | 7,545,546 | 7,700,398 | 643,215 | 154,852 |

Two exact invariances hold across all five folds:

* **G1 − G0 = 643,215** — constant. Binding and protein→gene encoding are static, so they
  cannot depend on the fold.
* **G2 − G1 = 154,852** — constant. PPI is likewise fold-independent.

These constants are a useful sanity check on any rebuild: a typed run that changes either
of them has changed graph structure, not just relation typing.

---

## 3. Typed (mode B) counts, fold 0

| | edges |
|---|---|
| G0 | 6,863,061 |
| G1 | 6,981,942 |
| G2 | 7,136,794 |
| G1 − G0 | 118,881 |
| G2 − G1 | **154,852** |

**G0 is bit-identical to the frozen value (6,863,061).** The typed generation reproduces
the same node universe and the same non-binding edge set exactly, which is the strongest
available evidence that only the binding layer changed.

**G2 − G1 is bit-identical too (154,852).** PPI is untouched.

---

## 4. The difference, exactly

| | edges |
|---|---|
| legacy G1 − G0 | 643,215 |
| − protein→gene encoding | 20,008 |
| **= legacy binding edges** | **623,207** |
| typed G1 − G0 | 118,881 |
| − protein→gene encoding | 20,008 |
| **= typed binding edges** | **98,873** |
| **difference** | **524,334** |

`643,215 − 118,881 = 524,334`, and independently
`623,207 − 98,873 = 524,334`.

**`623,207` is exactly the count the frozen graph authority declares** as
`global_binding` in its own receipt. The legacy graph's binding layer is therefore fully
accounted for, and the typed difference is precisely the set of binding edges the
conservative default excludes.

### What those 524,334 edges are

They are the sequence-based predictions: `MATCH algorithm` and `catRAPID` evidence on
lncRNA–RBP pairs. Phase 12 measured 622,077 such rows in the binding table; after the
graph's own key de-duplication they become **524,334 distinct edges**, or

**84.1 % of the legacy binding layer** (524,334 / 623,207).

The row-level figure was 84.2 %. The two agree, as they must.

---

## 5. What mode B actually emits, fold 0

| relation_type | edges |
|---|---|
| `binds_protein_other_clip` | 36,998 |
| `binds_protein_eclip` | 31,402 |
| `binds_protein_experimental_unspecified` | 30,191 |
| `binds_protein_rip` | 176 |
| `binds_protein_rna_capture` | 75 |
| `binds_protein_other_physical` | 16 |
| `binds_protein_unknown` | 15 |
| **total** | **98,873** |

`binds_protein_predicted` is **absent**, which is the conservative default
(`include_predicted=False`) working as intended.

Against legacy's single flat `binds_protein` relation carrying all 623,207 edges, mode B
replaces it with seven typed relations carrying 98,873 measured edges.

---

## 6. What the A/B ablation will therefore measure

The comparison is now precisely specified before any training:

| | mode A | mode B |
|---|---|---|
| binding relation types | 1 | 7 |
| binding edges | 623,207 | 98,873 |
| of which predictions | **524,334 (84.1 %)** | **0** |
| total G2 edges | 7,661,128 | 7,136,794 |

So **A vs B isolates two things at once**, and the report must not conflate them:

1. **assay typing** — one relation becomes seven, so message passing can distinguish an
   eCLIP edge from a RIP edge;
2. **prediction exclusion** — 84 % of the legacy binding layer is removed from message
   passing.

If a reviewer wants them separated, a third arm is available and cheap:
`typed_binding_plus_predicted` (`include_predicted=True`) keeps the seven relation types but
restores the 524,334 predicted edges as `binds_protein_predicted`. **A vs that arm isolates
typing alone; that arm vs B isolates prediction exclusion alone.** The four headline modes
in `RBP_ENCODE_ABLATION_PLAN.md` are unchanged; this is an optional decomposition, and it
costs nothing extra because the materialiser already supports the switch.

---

## 7. Cost

The frozen prepared bundles total about **170 GB** (G0 51.3, G1 58.4, G2 60.1). Producing
the mode B equivalent is the same order. Check `df` immediately before launching; `/dell_2`
had 252 GB free and `/public0` 503 GB.

---

## 8. The node universe is identical, so the prepared payload can be swapped rather than rebuilt

A prepared payload stores **node indices**, not node ids:

```python
"candidate_batch": {
    "l": torch.tensor([maps["lncRNA"][str(x)] for x in sub.lncrna_id], ...),
    "p": torch.tensor([maps["pathway"][str(x)] for x in sub.pathway_id], ...),
    "c": torch.tensor([maps["cancer"][str(x)]  for x in sub.cancer_id],  ...),
},
"base_logit": torch.tensor(logits[...]),   # the LASSO baseline
```

If the typed generation changed the node universe, reusing those indices would
**silently misalign every batch with the graph** — no error, just wrong rows.

Measured against the frozen authority records:

| node type | frozen | typed |
|---|---|---|
| cancer | 33 | 33 |
| gene | 21,981 | 21,981 |
| lncRNA | 8,541 | 8,541 |
| pathway | 2,135 | 2,135 |
| pathway_family | 85 | 85 |
| protein | 20,008 | 20,008 |

`node_sha256` is `a02f49648ff3d263c56c490eaf4abc06056c11a8d95520b2112dbd04ca21a491`
for **all five folds**, and the typed fold-0 counts match it exactly.

**Conclusion:** `node_maps` agree, so the frozen `train_batches` and
`validation_batches` — including their LASSO `base_logit` tensors — can be reused
**verbatim** with the typed graph bundle.

This matters for two reasons:

1. **No LASSO re-fit is required.** Re-running the preparation entrypoint would
   refit `_fast_l1` per fold; the reuse requirement says not to. Swapping the
   bundle reuses the existing baseline exactly.
2. **The typed prepared payload becomes a surgical substitution**, not a
   re-preparation: load the frozen fold payload, replace `bundle`,
   `formal_graph_authority` and `formal_graph_variant`, and write it to the new
   work root. Everything else — batches, logits, label contract, patient-fold
   binding — is carried over unchanged.

The frozen receipt asserts `same_relation_schema_all_variants: True`. The typed
generation correctly declares that gate as **False**, because it genuinely does
not share the relation schema. The node universe being identical while the
relation schema differs is the precise shape of this change: same graph, same
rows, differently typed binding edges.

The ~170 GB footprint noted in section 7 still applies, since the bundles are
what occupy the space — the saving is in avoiding a LASSO re-fit and a full
re-preparation, not in payload size.