# G2 vs frozen L1 — validation-tier comparison

Folds present: [0, 1, 2, 3, 4] of [0, 1, 2, 3, 4]

Same rows, same folds, same candidate keys, same labels; the only difference is the
graph residual `gate * tanh(raw)` added to the frozen L1 logit.

| fold | rows | L1 AUPRC | G2 AUPRC | ΔAUPRC | ΔAUROC | Δlogloss | Spearman(L1,G2) | gate mean | identity err |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3,300,000 | 0.300709 | 0.456138 | +0.155429 | +0.078526 | -0.362074 | 0.924066 | 0.7049 | 2.91e-07 |
| 1 | 3,300,000 | 0.306285 | 0.473734 | +0.167449 | +0.078740 | -0.367468 | 0.928009 | 0.7159 | 2.96e-07 |
| 2 | 3,300,000 | 0.310652 | 0.475661 | +0.165010 | +0.076964 | -0.367092 | 0.921649 | 0.7101 | 2.96e-07 |
| 3 | 3,300,000 | 0.290048 | 0.438200 | +0.148151 | +0.078140 | -0.362234 | 0.920228 | 0.7068 | 2.89e-07 |
| 4 | 3,300,000 | 0.292338 | 0.438276 | +0.145938 | +0.076093 | -0.358891 | 0.926269 | 0.7078 | 2.94e-07 |

**Mean ΔAUPRC = +0.156395** (sd 0.009678, range +0.145938 … +0.167449)
across 5 fold(s); G2 ahead in 5, L1 ahead in 0.

## Caveats

- Validation tier only; no sealed-test access, so this is a development conclusion, not an independent confirmation.
- Only the G2 arm was trained, so a gain cannot be attributed specifically to ENCODE eCLIP, to assay typing, or to the graph in general: that needs matched controls (G0/G1, shuffled eCLIP) which were deliberately not run.
- delta_logloss mixes ranking and calibration; the pre-specified headline is delta_auprc.
- gate/residual statistics describe the graph's contribution, not its benefit.
