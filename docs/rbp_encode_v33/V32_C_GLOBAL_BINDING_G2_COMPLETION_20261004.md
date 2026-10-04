# C_GLOBAL_BINDING_G2 — completion record, 2026-10-04

Host `149` for every CPU step; the CompShare V100S instance only ever ran the five trainings and
the core-embedding export. This records what finished, what it produced, and what it does not
establish.

## 1. Five-fold training

| fold | status | stop reason | completed cycles | best validation loss | optimizer steps | runtime chunks |
|---|---|---|---|---|---|---|
| 0 | SUCCESS | EARLY_STOPPING | 104 | 0.27241788221742685 | 10,605 | 26 |
| 1 | SUCCESS | EARLY_STOPPING | 41 | 0.28638931512856775 | 4,242 | 26 |
| 2 | SUCCESS | EARLY_STOPPING | 87 | 0.2767589824230766 | 8,888 | 26 |
| 3 | SUCCESS | EARLY_STOPPING | 111 | 0.2783104576549853 | 11,312 | **27** |
| 4 | SUCCESS | EARLY_STOPPING | 93 | 0.27403199377938575 | 9,494 | 26 |

All five wrote `SUCCESS.json` and were returned to `149` under
`PASS_RESULT_RETURN_SIZE_VERIFIED_NO_REHASH` receipts.

**Fold 3 is the one that could have failed and did not.** The run config sets
`require_early_stopping_before_cap: true` with `patience_coverage_cycles: 6` and
`max_coverage_cycles: 120`, and `training.py:3501` writes `FAILURE.json` with
`status: FAILED_HARD_CAP` and raises if the cap is reached without early stopping — in which case
no `SUCCESS.json` is ever written and the standard collector refuses the fold. Fold 3 sat at
`stale_cycles = 5` after cycle 110 and reached 6 at cycle 111, so the registered early-stopping
rule fired 9 cycles before the cap (`did_not_hit_hard_cap: true`). Nothing was relabelled and no
`FAILURE` was reinterpreted.

The folds did **not** train for the same number of cycles (41 to 111). They were not padded,
resampled or retrained to match; the early-stopping rule is the registered one and it fired when
it fired.

## 2. Core-embedding export — `CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1`

One manifest over all five folds, six node types × 96 features:
cancer 33, gene 21,981, lncRNA 8,541, pathway 2,135, pathway_family 85, protein 20,008.
30 exported files on `149`, every one sha256-verified against the instance-side manifest.

`embedding_estimand = REGISTERED_CHUNK_WEIGHTED_MEAN_OF_ENCODER_OUTPUTS`, i.e.
`Σ_c w_c · encoder(chunk_c)` with `w_c` taken unchanged from
`cc_hhgt.v32.training._runtime_chunk_contract`. It is not claimed to equal
`encoder(union_graph)`.

The chunk partition is **fold-specific**: every fold receives the same 152,246 physical-binding
edges on top of a different prepared payload, so the rotating-edge partition differs and fold 3
lands on 27 chunks where the other four land on 26. The estimand is therefore a weighted mean over
26 terms for four folds and 27 for one — which is the registered definition read literally. No
fold was forced onto a common chunk count. Full detail in
[`V32_CORE_EMBEDDING_ESTIMAND_DECISION.md`](V32_CORE_EMBEDDING_ESTIMAND_DECISION.md) §5.

The export is an accumulator by design (the runner loads and rewrites an existing manifest), so
fold 3 was exported on its own into the existing manifest. The pre-export manifest was archived
first, and the four carried-over fold records were checked to be byte-identical afterwards
(24 export entries compared, 0 drifted).

## 3. G2 vs frozen L1 — complete five-fold validation evaluation

**Mean ΔAUPRC = +0.156395** (sd 0.009678, range +0.145938 … +0.167449), G2 ahead in **5 of 5**
folds. Mean ΔAUROC ≈ +0.0777, mean Δlogloss ≈ −0.3636. Full table and caveats in
[`g2_eval_20261004_fivefold/README.md`](g2_eval_20261004_fivefold/README.md); this supersedes the
3-of-5 snapshot in `g2_eval_20261004/`.

Integrity checks hold in every fold: `identity_max_abs_error ≈ 2.9e-07` (the residual identity
`z_final = z_l1 + gate·tanh(raw)`), `incremental_aggregation_equivalence_max_abs_diff = 0.0`
(incremental Kahan accumulation is bit-identical to the reference), `share_graph_unavailable = 0.0`.

Graph-contribution statistics (descriptive, not a benefit claim): gate mean 0.705–0.716, mean
absolute raw residual 2.94–3.02, and 4.5–9.6% of rows move less than 1e-6 in probability.

Twelve to fifteen percent of each fold's rows sit in cancers with **no positive label at all**
(CHOL, DLBC and UCS in all five folds), which deflates the absolute AUPRC without biasing the
paired ΔAUPRC.

## 4. Single-cell private head

`CC_HHGT_V3_2_SINGLE_CELL_PRIVATE_HEAD_V1`, five folds, `status: SUCCESS`,
`release_ready: true`, contract validation `PASS`. Each fold's `core_checkpoint_sha256` matches
the checkpoint the export manifest recorded for that fold, and the five folds' validation rows sum
to exactly the 178,064 available rows. The core's parameter hash is identical before and after
(`core_parameters_frozen: true`). Detail in
[`single_cell_head_20261004/README.md`](single_cell_head_20261004/README.md).

Two boundaries recorded there rather than glossed: `FIGURE_MANIFEST.json` is 33 entries all
`NULL_WITH_REASON` (`NO_REGISTERED_SINGLE_CELL_FIGURE`), and `partition_isolation` is
`DATASET_LEVEL_AGGREGATED_ASSOCIATIONS_NOT_DONOR_LEVEL`.

## 5. Paid compute released

Everything GPU-dependent finished and was verified on `149` first: five fold results returned
size-verified, the core embeddings collected with all 30 files sha256-verified, five evaluation
receipts present, and the instance-side provenance tree (configs, approvals, launch preflights,
core-embedding LINEAGE files, export logs) swept to
`$R/instance_evidence/` under a `SHA256SUMS.tsv` that verifies.

Then:

```
compshare instance delete uhost-1vsbwarp9m3s --release-disk --yes
  → TerminateCompShareInstance RetCode 0, failed: []
compshare instance list → items: []
compshare storage disk list → items: []
```

Instance `uhost-1vsbwarp9m3s` (`cancerlncatlas-c-v100s-f0-20260927`) and its disk
`bsi-196a1da7f1` are both gone; no paid resource remains. Two files were confirmed **absent** on
the instance before deletion and are therefore not in the evidence tree: fold 1's
`FOLD1_SSH_LAUNCH.log` and fold 4's `LAUNCH_PREFLIGHT.json`.

## 6. What none of this establishes

* **Validation tier only.** No sealed-test access at any point.
* **Only the G2 arm was trained.** The gain cannot be attributed to ENCODE eCLIP, to assay
  typing, or to the graph in general. That needs matched controls (G0/G1, shuffled eCLIP) which
  were deliberately not run.
* **The leakage inspection is one-directional.** `PASS_NO_LNCRNA_PATHWAY_EDGE` shows 0 of
  7,190,167 edges touch both an lncRNA and a pathway and that all edges are
  `outcome_derived=False`. It does not rule out two-hop lncRNA→gene→pathway paths, and no
  shuffled-edge control was run.
* **The single-cell head has no measured performance here.** Its run criteria are contract and
  lineage checks, and they passed; no AUPRC/AUROC is claimed.
* The G2 figures sit close to the historical `[H5]` diagnostic (L1 0.301776, CC-HHGT 0.448534)
  whose provenance was later downgraded — a reason for scrutiny, not celebration.
