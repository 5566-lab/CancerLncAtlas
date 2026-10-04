# G2 vs frozen L1 — validation-tier evaluation (2026-10-04)

Produced on host `149` by `cc_hhgt`-adjacent tooling at
`$R/eval/v32-g2-l1-eval-20261003-r2/`, aggregated by `summarise_g2_eval_149.py`.

**Status: partial — 3 of 5 folds.** `G2_VS_L1_SUMMARY.json` records
`complete_five_fold: false`. Fold 3 was still training and fold 4's payload had only just
been staged when this was written.

## What was compared

Identical rows, folds, candidate keys, availability mask and labels. The only difference is
the graph residual `gate * tanh(raw_graph_residual)` added to the frozen L1 logit
(`base_logit`), which comes from the prepared payload rather than from a recomputed baseline.
Validation tier only — no sealed-test access.

## A structure in the evaluation set that bounds how the pooled number reads

Twelve to fifteen percent of each fold's 3,300,000 rows sit in cancers that carry **no positive
label at all**, so they can only ever contribute false positives to precision:

| fold | cancers with zero positives | their rows | share |
|---|---|---|---|
| 0 | ACC, CHOL, DLBC, UCS | 400,000 | 12.1% |
| 1 | ACC, CHOL, DLBC, KICH, UCS | 500,000 | 15.2% |
| 2 | CHOL, DLBC, MESO, UCS | 400,000 | 12.1% |

CHOL, DLBC and UCS are zero-positive in **all three** folds. Among the remaining rows the
positive rate is 16.15% on fold 0.

Both arms are scored on exactly the same rows, so ΔAUPRC remains a fair comparison. But the
**absolute** AUPRC is deflated by rows that can never yield a true positive, and the per-cancer
entries for those cancers are uninformative rather than bad. Any report quoting the pooled
figure should say which cancers it is diluted by.

The largest per-cancer gains on fold 0 are BRCA (+0.2768), LGG (+0.2738) and KIRC (+0.2600),
so the pooled improvement is not carried by a single cancer.


## Files

| file | content |
|---|---|
| `G2_VS_L1_SUMMARY.md` / `.json` | per-fold and mean deltas, caveats, and the honest-boundary list |
| `RECEIPT_FOLD_{0,1,2}.json` | per-fold receipt: metrics for both arms, calibration, per-cancer breakdown, residual/gate distributions, identity error, aggregation-equivalence check |

## Integrity checks carried in every receipt

* `identity_max_abs_error_all_rows` ≈ 2.9e-07 — the residual identity
  `z_final = z_l1 + gate·tanh(raw)` holds to float32 precision.
* `incremental_aggregation_equivalence_max_abs_diff = 0.0` — the incremental Kahan
  accumulation is bit-identical to the reference `_kahan_weighted_tensor_sum`.
* `share_graph_unavailable = 0.0` and edges 7,190,167 / binding 152,246 matching the
  authorization.

## Read the caveats before quoting the headline

Mean ΔAUPRC is **+0.1626** with G2 ahead in 3/3 folds. That number must not be reported
without the boundaries recorded in the summary: validation tier only; no matched controls, so
the gain cannot be attributed to ENCODE eCLIP, assay typing, or the graph specifically; and the
figures sit almost exactly on the historical `[H5]` diagnostic (L1 0.301776, CC-HHGT 0.448534)
whose provenance was later downgraded — which is a reason for scrutiny, not celebration.