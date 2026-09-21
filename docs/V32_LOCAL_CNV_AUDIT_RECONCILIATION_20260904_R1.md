# V3.2 local-CNV audit reconciliation — revision 1 (2026-09-04)

## Purpose and scope

This revision records a read-only, server-side consistency check of the
completed local-CNV confounding audit and the independent CNV-only five-fold
OOF.  It does not read parquet/OOF payloads, alter any authority, delete data,
start a process, or promote a website release.

| Field | Value |
| --- | --- |
| Check time (UTC) | `2026-09-04T11:14:38.151117+00:00` |
| Execution host | `149` (`COMPUTE_HOST`) |
| Mode | read-only; `payloads_read=false` |
| Inspector result | `PASS` |

The source receipts below are the authorities.  The local report and its JSON
receipt are documentation artifacts only; they are not replacements for the
server receipts.

## Authoritative receipts

| Item | Server path | SHA-256 | Observed result |
| --- | --- | --- | --- |
| Local-CNV decision | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/LOCAL_CNV_DECISION.json` | `bb2869c83f1fc8c2fd692d90a2a9668da58ce3d6a92b9e1ec1a9b2feea554fa6` | `PASS`; `LOCAL_CNV_CORE_RETRAIN_REQUIRED` |
| Independent local-CNV audit | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/LOCAL_CNV_INDEPENDENT_AUDIT.json` | `86679fd374d83d3ec050efc769cf5c064e1073491efbd8082dd118c56d67169a` | `PASS`; 33 cancers, 165 folds, 330 files |
| Global confounding summary | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/local_cnv_confounding_global_summary.json` | `908bd92002d0e5e5c662480484b12535f3920f0c7b15b5dc4f25711160f04353` | 16,500,000 candidate rows |
| Audit completion | `./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/SUCCESS.json` | `82de6bab7e2b479189dd0c1701ff09b475327b8d1dbb83c6d413caeb14937c7b` | `SUCCESS` |
| CNV-only OOF manifest | `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/OOF_MANIFEST.json` | `c42a323c781fe91a48a9ffba3e1f44fea428ecc03246ec6d1867c6adf587a307` | 5 checkpoints, 165 prediction files |
| CNV-only metrics | `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/METRICS.json` | `789082bee4b419b83bb4a55232539f12241f82817041f7437d65abdd50c55577` | 6,361,208 available rows; log loss `0.18120417074813497` |
| CNV-only independent audit | `./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/INDEPENDENT_AUDIT.json` | `f230365b775e3afb6af7a7b56aae411752c28e69e111cb19550ddce0ae49a452` | `PASS`; typed-unavailable values never zero |

## Structural and scientific checks

All of the following were true in the authoritative r3 audit:

- 33/33 cancer closure, 165/165 fold closure, and exactly 100,000 candidate
  rows per pair/fold partition.
- Candidate keys, input authority/schema, pair-file hashes, G0 file hashes,
  receipt hashes, labels, and official FDR were recomputed successfully.
- A0/A1 matched cohorts were equal; signed continuous CNV was used; separate
  amplification/deletion fields were retained; real lncRNA–gene G0 edges were
  present.
- Mutation and sealed-test fields were absent from the audit inputs.  This is
  an explicit contract check, not an assumption based on file names.

The estimands are kept separate:

| Layer | Meaning |
| --- | --- |
| A0_FULL | Baseline association on the declared training patients. |
| A0_MATCHED | The same matched, local-CNV-callable cohort without local adjustment. |
| A1_LOCAL | A0_MATCHED with the lncRNA-side local CNV adjustment. |
| A1-on-A2 | Matched comparator used when both sides are callable. |
| A2_LOCAL_PATHWAY | A1-style adjustment with pathway-side CNV included. |
| E0/E1 and G0 | Real lncRNA–gene edge stability before/after adjustment; E1 is signed. |

These are association and stability checks, not a causal intervention test.
Typed-unavailable observations remain null/NaN with an explicit reason and are
never encoded as numeric zero.

### Headline values

| Metric | Value |
| --- | ---: |
| Baseline positive-label flip rate | `0.18273194012582256` |
| Direction flip rate | `0.049638150218087616` |
| G0 sign-flip rate | `0.0` |
| G0 typed-unavailable folds | `17` |
| Median cancer/pathway rank Spearman | `0.977797585284823` |
| Median G0 edge Jaccard | `0.6230302453332099` |
| Median lncRNA-direction top-k Jaccard | `0.6666666666666666` |
| CNV-sensitive candidates | `2,782,549` |
| CNV-mediated candidates | `437,617` |

The positive-label and direction changes cross the frozen decision gate, hence
the authoritative decision is `LOCAL_CNV_CORE_RETRAIN_REQUIRED`.  It means
the corrected local-CNV inputs must feed the next primary G0/G1/G2 training;
it does not mean that the auxiliary CNV head is already part of the primary
score.

## Independent CNV-only OOF

The OOF artifact is a separate overlay and passed its independent audit:

- 5 fresh checkpoints and 165 prediction files;
- 6,361,208 available rows, global log loss `0.18120417074813497`, Brier
  `0.04375090127579527`;
- `cnv_only=true`, `mutation_features_used=false`,
  `old_checkpoint_loaded=false`, and `old_predictions_used=false`;
- core parameters before/after are identical; typed-unavailable rows are not
  zero-filled.

The OOF log loss is not evidence that the primary HHGT score improved.  No
primary score, G0/G1/G2 checkpoint, router, winner lock, or sealed-test result
was changed by this artifact.

## Provenance consistency finding

There is a versioned code distinction that must remain explicit in future
reports:

1. The formal server OOF at `v32_cnv_head_oof_20260902_r6` is the V1 artifact
   produced from the server-side `cc_hhgt/v32/cnv_only_training.py` snapshot.
2. The checked-in `cc_hhgt/v32/cnv_only_training_v2.py` is a newer, candidate
   runner (formal r3 association labels and a different output contract).  It
   is not the generator of the already-issued r6 receipt and has not been
   silently substituted for it.
3. The checked-in structural auditor is V2, while the server r3 independent
   receipt declares its own V1 format.  The receipt, not an unexecuted local
   script, is authoritative for r3.  This is code/documentation drift to
   track, not a failed scientific audit.

Consequently, reports must not claim that r6 was generated by the V2 runner,
or that the V2 runner has been independently audited, until a new server run
creates a new receipt.

## Boundary and remaining release gates

Confirmed absent from this local-CNV run: Mutation, ATAC, methylation, Drug,
Evidence Transformer, external router, and sealed-test data.  Those modules
remain independent overlays or later primary-training inputs.

| Gate | Status at this check |
| --- | --- |
| Formal local-CNV r3 audit | PASS |
| Independent CNV-only OOF | PASS |
| Corrected G0/G1/G2 input preparation | Complete on 149; 15 payloads exist |
| Rebind/hash gate for all 15 corrected payloads | Pending |
| Corrected primary G0/G1/G2 optimizer training | Not started/ not observed |
| Hierarchical-gating vs external-router fair comparison | Pending |
| Winner lock and sealed test | Pending; sealed test not read |
| Website staging promotion | Pending; CNV overlay remains staging-only |

No authority, data payload, model checkpoint, or training process was changed
by this documentation audit.

## Current closure amendment (2026-09-05)

The historical table above records the state at the 2026-09-04 inspection.  It
is superseded for current execution by the following host-149 facts:

- The 15-file, pre-local-CNV materialization
  `formal_prepared_20260830_r3_affine_pyg280_localtorch` and its matching
  transfer archive were explicitly deleted after an exact dependency and
  open-handle check.  They are not valid inputs and are not used as a
  baseline for any subsequent training.
- The 15 corrected local-CNV payloads under
  `v32_g012_local_cnv_formal_prepared_20260903_r1` completed the one-time r5
  rebind/hash closure.  No payload was copied or rewritten during that
  rebind.  The corrected static authorization records
  `old_baseline_used=false` and `training_started=false`.
- The legacy parent directory is retained only for the static graph and
  fold-level expression/coexpression authorities referenced by the corrected
  payload lineage.  It is not the deleted baseline payload and must not be
  interpreted as a trainable fallback.

Accordingly, the current gate values are: corrected input/rebind **complete**;
primary G0/G1/G2 optimizer training **not started**; and the local-CNV audit
decision remains `LOCAL_CNV_CORE_RETRAIN_REQUIRED`.  This amendment does not
rewrite the immutable server receipts or change the audit estimand.
