# V3.2 local-CNV authority/document reconciliation (2026-09-04)

This note reconciles the checked-in status documents with the completed
server-side local-CNV audit and independent CNV-only OOF.  It is a
documentation amendment only: no data, model, binding, or training process was
changed by this check.

## Authoritative server receipts

All paths below are on COMPUTE_HOST host `149` and were inspected read-only.

| Deliverable | Authority | SHA-256 | Result |
| --- | --- | --- | --- |
| Local-CNV decision | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/LOCAL_CNV_DECISION.json` | `bb2869c83f1fc8c2fd692d90a2a9668da58ce3d6a92b9e1ec1a9b2feea554fa6` | `status=PASS`; decision `LOCAL_CNV_CORE_RETRAIN_REQUIRED` |
| Independent local-CNV audit | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/LOCAL_CNV_INDEPENDENT_AUDIT.json` | `86679fd374d83d3ec050efc769cf5c064e1073491efbd8082dd118c56d67169a` | `status=PASS`; 33 cancers, 330 files, 165 folds |
| Global confounding summary | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/local_cnv_confounding_global_summary.json` | `908bd92002d0e5e5c662480484b12535f3920f0c7b15b5dc4f25711160f04353` | 16,500,000 candidate rows |
| Audit completion | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/SUCCESS.json` | `82de6bab7e2b479189dd0c1701ff09b475327b8d1dbb83c6d413caeb14937c7b` | `status=SUCCESS` |
| CNV-only OOF manifest | `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/OOF_MANIFEST.json` | `c42a323c781fe91a48a9ffba3e1f44fea428ecc03246ec6d1867c6adf587a307` | 5 fresh checkpoints, 165 prediction files, `cnv_only=true` |
| CNV-only metrics | `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/METRICS.json` | `789082bee4b419b83bb4a55232539f12241f82817041f7437d65abdd50c55577` | 6,361,208 available rows; global log loss `0.18120417074813497` |
| CNV-only independent audit | `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/INDEPENDENT_AUDIT.json` | `f230365b775e3afb6af7a7b56aae411752c28e69e111cb19550ddce0ae49a452` | `status=PASS`; typed-unavailable never zero |

## Scientific values that must be quoted together

The confounding decision is not inferred from the OOF log loss.  The formal
audit reports:

- baseline positive-label flip rate `0.18273194012582256`;
- direction flip rate `0.049638150218087616`;
- G0 sign-flip rate `0.0` and 17 typed-unavailable G0 folds;
- median cancer/pathway rank Spearman `0.977797585284823`;
- median G0 edge Jaccard `0.6230302453332099` (edge turnover
  `0.37696975466679006`);
- median lncRNA-direction top-k Jaccard `0.6666666666666666`;
- `2,782,549` CNV-sensitive candidates and `437,617` CNV-mediated candidates;
- `mutation_used=false`, `primary_or_g012_modified=false`, and
  `sealed_test_read=false`.

These values cross the frozen core-retrain gate.  Therefore the corrected
local-CNV authority must feed the next G0/G1/G2 preparation/training, while the
independent CNV-only OOF remains an auxiliary overlay and does not alter the
current primary score.

## Documentation consistency findings

### Current and consistent

`docs/v32_local_cnv_and_cnv_oof_20260902.md` already has the correct decision,
scope, headline metrics, typed-unavailable policy, and server roots.  Its only
material omission was the receipt hashes and the exact sensitive/mediated
counts; those are supplied above.

`docs/V32_CURRENT_MAINLINE_STATUS_20260903.md` correctly states that corrected
G0/G1/G2 preparation is complete but optimizer training has not started.  The
local-CNV decision and the separation of the overlay from the primary score
are also correct.

### Historical snapshots that contain stale status fields

The following files are dated snapshots and should not be read as the current
state without this amendment:

1. `docs/v32_fresh_g012_preparation_design_20260829.md` still labels the
   full-33 CNV five-fold OOF expert `NOT_READY` and says no OOF authority
   exists.  That was true on 2026-08-29; it is superseded by
   `v32_cnv_head_oof_20260902_r6`.
2. `docs/v32_g012_33_cancer_status_matrix_20260829.tsv` has
   `cnv_five_fold_oof=NOT_READY` for all rows.  It is an as-of matrix, not a
   live registry; its CNV column must not be used for current release gating.
3. `docs/NEXT_AGENT_STATUS_20260902.md` previously said that CNV OOF
   training/inference was absent.  It now carries an amendment stating that
   the independent CNV-only OOF is complete, while local-CNV-corrected primary
   G0/G1/G2 training remains outstanding.
4. `docs/V32_INDEPENDENT_HEAD_STAGING_CLOSURE_20260903.md` previously said the
   directional CNV overlay was pending formal binding.  It now records the
   server overlay as `status=PASS`, `staging_queryable=true`, and
   `release_ready=false`; production promotion is still blocked.

The dated 2026-08-29 design and status-matrix snapshots are intentionally
retained as historical records; the current handoff/closure docs were amended
only to point at the new authority.  A current status reader should use this
reconciliation plus the server receipts above.

## Non-claims and remaining gates

- The audit does **not** prove that the primary HHGT score has already been
  retrained; no G0/G1/G2 optimizer run was observed at this check.
- The CNV-only OOF is not a winner, router comparison, or sealed-test result.
- Typed-unavailable CNV values remain null/unavailable, never zero.
- No historical checkpoint or prediction is an accepted substitute for the
  corrected local-CNV inputs.
