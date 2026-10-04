# G2 vs frozen L1 — complete five-fold validation-tier evaluation (2026-10-04)

This supersedes `../g2_eval_20261004/`, which was a 3-of-5-fold snapshot written while fold 3 was
still training. `G2_VS_L1_SUMMARY.json` here records `complete_five_fold: true`.

Produced on host `149`. Evaluator: `$R/eval/eval_g2_validation_149.py` (the authoritative copy on
the mount — it hard-codes its output root to `$R/eval/fold_<N>` and ignores `EVAL_OUT_ROOT`).
Aggregated by `summarise_g2_eval_149.py`. No sealed-test access at any point.

## Headline

| fold | rows | L1 AUPRC | G2 AUPRC | ΔAUPRC | ΔAUROC | Δlogloss | Spearman(L1,G2) |
|---|---|---|---|---|---|---|---|
| 0 | 3,300,000 | 0.300709 | 0.456138 | +0.155429 | +0.078526 | −0.362074 | 0.924066 |
| 1 | 3,300,000 | 0.306285 | 0.473734 | +0.167449 | +0.078740 | −0.367468 | 0.928009 |
| 2 | 3,300,000 | 0.310652 | 0.475661 | +0.165010 | +0.076964 | −0.367092 | 0.921649 |
| 3 | 3,300,000 | 0.290048 | 0.438200 | +0.148151 | +0.078140 | −0.362234 | 0.920228 |
| 4 | 3,300,000 | 0.292338 | 0.438276 | +0.145938 | +0.076093 | −0.358891 | 0.926269 |

**Mean ΔAUPRC = +0.156395** (sd 0.009678, range +0.145938 … +0.167449); G2 ahead in **5 of 5**
folds. Mean ΔAUROC ≈ +0.0777, mean Δlogloss ≈ −0.3636.

Same rows, same folds, same candidate keys, same availability mask, same labels in both arms. The
only difference is the graph residual `gate · tanh(raw_graph_residual)` added to the frozen L1
logit.

## Training behind the five checkpoints

| fold | completed cycles | stop reason | best validation loss | optimizer steps | runtime chunks |
|---|---|---|---|---|---|
| 0 | 104 | EARLY_STOPPING | 0.27241788221742685 | 10,605 | 26 |
| 1 | 41 | EARLY_STOPPING | 0.28638931512856775 | 4,242 | 26 |
| 2 | 87 | EARLY_STOPPING | 0.2767589824230766 | 8,888 | 26 |
| 3 | 111 | EARLY_STOPPING | 0.2783104576549853 | 11,312 | **27** |
| 4 | 93 | EARLY_STOPPING | 0.27403199377938575 | 9,494 | 26 |

All five wrote `SUCCESS.json`. Fold 3 is the one worth naming: it reached `stale_cycles = 6`
against `patience_coverage_cycles = 6` at cycle 111 of a 120-cycle cap, so it stopped by the
early-stopping rule — `did_not_hit_hard_cap: true` — and not by the cap.
`require_early_stopping_before_cap` was true in this run, so a cap stop would have written
`FAILURE.json` and no `SUCCESS.json` at all. That did not happen, and nothing here is a
relabelled failure.

The folds did **not** train for the same number of cycles (41 to 111). They were not padded or
retrained to match: the early-stopping rule is the registered one and it fired when it fired.

## Graph-contribution diagnostics (descriptive, not a benefit claim)

| fold | gate mean | mean abs raw residual | share with \|Δp\| < 1e-6 | identity max err | agg equiv | graph unavailable |
|---|---|---|---|---|---|---|
| 0 | 0.704913 | 2.999510 | 9.64% | 2.91e-07 | 0.0 | 0.0 |
| 1 | 0.715942 | 3.015354 | 4.46% | 2.96e-07 | 0.0 | 0.0 |
| 2 | 0.710148 | 2.942358 | 8.05% | 2.96e-07 | 0.0 | 0.0 |
| 3 | 0.706778 | 2.963442 | 9.60% | 2.89e-07 | 0.0 | 0.0 |
| 4 | 0.707793 | 2.985296 | 8.63% | 2.95e-07 | 0.0 | 0.0 |

Read these as *what the graph residual did*, not as *what it bought*:

* the gate sits near 0.71 in every fold and the raw residual near 2.96, so the residual is
  active rather than switched off;
* 4–10% of rows move by less than 1e-6 in probability, i.e. for a small minority of rows the
  graph changes essentially nothing;
* `identity_max_abs_error` ≈ 2.9e-07 in every fold confirms `z_final = z_l1 + gate·tanh(raw)`
  holds to float32 precision, and `incremental_aggregation_equivalence_max_abs_diff = 0.0`
  confirms the incremental Kahan accumulation is bit-identical to the reference;
* `share_graph_unavailable = 0.0` in every fold — no row was scored without a graph.

## A structure in the evaluation set that bounds how the pooled number reads

Twelve to fifteen percent of each fold's 3,300,000 rows sit in cancers with **no positive label at
all**, so they can only contribute false positives:

| fold | cancers with zero positives | their rows | share |
|---|---|---|---|
| 0 | ACC, CHOL, DLBC, UCS | 400,000 | 12.1% |
| 1 | ACC, CHOL, DLBC, KICH, UCS | 500,000 | 15.2% |
| 2 | CHOL, DLBC, MESO, UCS | 400,000 | 12.1% |
| 3 | ACC, CHOL, DLBC, UCS | 400,000 | 12.1% |
| 4 | CHOL, DLBC, KICH, UCS | 400,000 | 12.1% |

**CHOL, DLBC and UCS are zero-positive in all five folds.** Both arms are scored on identical
rows, so ΔAUPRC stays a fair comparison, but the absolute AUPRC is deflated by rows that can
never yield a true positive, and the per-cancer entries for those cancers are uninformative
rather than bad.

The pooled gain is not carried by one cancer: the largest per-fold gains are
BRCA/LGG/KIRC (+0.24 to +0.28) on folds 0–3 and LGG/THCA/STAD/PRAD (+0.23 to +0.25) on fold 4.

## The per-fold graphs are not identical

| fold | graph edges | binding edges added | runtime chunks |
|---|---|---|---|
| 0 | 7,190,167 | 152,246 | 26 |
| 1 | 7,216,351 | 152,246 | 26 |
| 2 | 7,254,737 | 152,246 | 26 |
| 3 | 7,278,448 | 152,246 | 27 |
| 4 | 7,229,437 | 152,246 | 26 |

Every fold receives the same 152,246 physical-binding edges on top of a different prepared
payload, so the rotating-edge partition — and with it the runtime chunk count — is fold-specific.
The core-embedding estimand is a weighted mean over that partition, i.e. 26 terms for four folds
and 27 for fold 3. See `../V32_CORE_EMBEDDING_ESTIMAND_DECISION.md` §5.

## Caveats — read before quoting the headline

* **Validation tier only.** No sealed-test access, so this is a development conclusion, not an
  independent confirmation.
* **Only the G2 arm was trained.** A gain cannot be attributed specifically to ENCODE eCLIP, to
  assay typing, or to the graph in general. That needs matched controls (G0/G1, shuffled eCLIP)
  which were deliberately not run.
* **Δlogloss mixes ranking and calibration.** The pre-specified headline is ΔAUPRC.
* **Gate/residual statistics describe the graph's contribution, not its benefit.**
* The figures sit close to the historical `[H5]` diagnostic (L1 0.301776, CC-HHGT 0.448534) whose
  provenance was later downgraded — a reason for scrutiny, not celebration.

## Files

| file | content |
|---|---|
| `G2_VS_L1_SUMMARY.md` / `.json` | per-fold and mean deltas, graph-contribution statistics, caveats |
| `RECEIPT_FOLD_{0,1,2,3,4}.json` | full per-fold receipt: both arms' metrics, calibration, per-cancer breakdown, gate/residual quantiles, identity error, aggregation-equivalence probe |
| `CORE_EMBEDDING_MANIFEST.json` | `CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1` export over all five folds (per-fold checkpoint hash, chunk weights, per-node-type usability and chunk deviation) |
| `RESULT_RETURN_FOLD_{3,4}.json` | size-verified result-return receipts for the two folds returned on 2026-10-04 |
| `INSTANCE_EVIDENCE_SHA256SUMS.tsv` | checksums of every instance-side provenance file swept to 149 before the paid GPU was released |

## Leakage

Unchanged from the 3-fold snapshot: `../g2_eval_20261004/LEAK_EDGE_INSPECTION.json` reports
`PASS_NO_LNCRNA_PATHWAY_EDGE` — 0 of 7,190,167 edges touch both an lncRNA and a pathway, and all
edges are `outcome_derived=False`. That inspection does **not** establish the absence of two-hop
lncRNA→gene→pathway paths, and no shuffled-edge control was run; both limitations are recorded in
`../g2_eval_20261004/LEAK_EDGE_INSPECTION_20261004.md`.
