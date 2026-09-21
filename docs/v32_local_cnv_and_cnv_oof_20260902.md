# V3.2 local-CNV audit and independent CNV OOF

> **Current closure (2026-09-04):** The local-CNV confounding audit and the
> independent five-fold CNV-only OOF are complete and independently audited.
> This document is a frozen record of those results, not an instruction to
> restart either run. The corrected local-CNV authority is already used for
> future G0/G1/G2 preparation; no historical CNV checkpoint or prediction may
> be substituted.

> **Receipt amendment (2026-09-04):** Exact server receipt hashes, the
> sensitive/mediated candidate counts, and a stale-document reconciliation are
> recorded in
> `docs/V32_LOCAL_CNV_AUDIT_RECONCILIATION_20260904.md`.  Use that note and
> the server receipts as the current authority; this file remains a frozen
> scientific summary.

Run IDs: `v32-cnv-only-20260902-r1` (superseded audit workspace), formal audit
`v32_local_cnv_confounding_audit_20260902_r3`, and CNV OOF
`v32_cnv_head_oof_20260902_r6`.  
Scope: local-lncRNA CNV confounding audit plus the independent CNV-only head,
five patient folds. Mutation, ATAC, methylation, Drug, Evidence, single-cell,
and router/HHGT retraining are outside this run.

The CNV head has a CNV-only contract. It reads the full 33-cancer compact
segment-CNV
store and frozen V3.2 lncRNA/pathway core exports; mutation tables are not
arguments and are not loaded as features. Typed unavailable values remain null,
never zero. Each fold is fit from scratch with validation-only checkpoint
selection, then test-fold predictions are written with `cnv_probability`.

Formal server outputs:

- `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/`
- `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/`

The first root is complete for all 33 cancers and has an independent PASS. It
contains the matched A0/A1/A2 pair tables, G0 edge stability, recomputed FDR,
and the frozen decision `LOCAL_CNV_CORE_RETRAIN_REQUIRED`. Its global audit
summary reports 16,500,000 pair-fold rows, 18.27% baseline-label flips,
4.96% direction flips, median pathway-rank Spearman 0.9778, and median G0 edge
Jaccard 0.6230.

The second root is also complete: five fresh CNV-only checkpoints and typed
OOF predictions for all 33 cancers. `OOF_MANIFEST.json` and
`INDEPENDENT_AUDIT.json` both pass; available OOF rows total 6,361,208, global
log loss is 0.181204, and core parameter hashes are unchanged before/after.
No mutation features, old checkpoint, old prediction, router, or sealed-test
data were used.

The full33 CNV store preflight is bound to 33 cancers, 10,432 patients, 10,364
processed segment patients, and 68 typed-unavailable patients. No historical
CNV checkpoint or prediction is accepted by the runner.

The exact completion gate was satisfied by
`./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/OOF_MANIFEST.json`
(`checkpoint_count=5`, `cnv_only=true`) and its independent audit. The CNV
head is a completed independent/staging artifact, not part of the primary
HHGT score. The directional website overlay is separately bound at
`./data/CancerLncAtlas/results/v32_directional_cnv_website_20260903_r1/`
with `staging_queryable=true` and `release_ready=false`. The audit decision
`LOCAL_CNV_CORE_RETRAIN_REQUIRED` is a scientific data-authority decision: it
does not by itself authorize a paid-GPU run, router training, sealed-test
evaluation, or publication of a changed primary score.

## Interpretation and non-repeat policy

The 18.27% baseline-label flips, 4.96% direction flips, and median G0 edge
Jaccard of 0.6230 demonstrate that local CNV is material to the old association
and G0 edge authorities. They do not mean the CNV overlay should be averaged
into the primary score. Missing CNV remains typed-unavailable (`null`), never
zero. The 33-cancer audit and the five-fold OOF are the current authorities;
future status reports should cite these receipts rather than schedule another
identical rescue.
