# G2 validation evaluation smoke — fold 0 (2026-10-03)

Host `149`, CPU only. Payloads read from local disk (see the sshfs hang note below).
Runner log: `$R/eval/smoke_local_fold0.log`.

## Result: PASS

```
[15:31:02] fold=0 src=12836939525B overlay=3299701429B global_binding=3214311B
[15:33:25] payload loaded in 142.9s
[15:35:13] overlay loaded in 108.6s   format=CANCERLNCATLAS_V32_C_GRAPH_OVERLAY_V1
[15:45:32] overlay applied and verified in 617.1s
[15:45:36] graph edges=7190167 binding_edges=152246   (match authorization)
[15:45:41] checkpoint loaded: cycle=103 variant=G2
           val_loss=0.27241788221742685 arch=HHGT_FORMAL_CORE_EXTERNAL_ROUTER
[15:46:57] chunks=26 validation_batches=403 weight_sum=1.000000000000000
[15:47:25] chunk 0 graph materialised in 27.8s
[15:48:02] chunk 0 encoder forward in 36.7s
[15:48:02] batch 0: 0.67s rows=8192 graph_available=8192/8192 identity_err_all=2.299e-07
[15:48:03]           gate min/mean/max = 0.0000/0.7037/1.0000  |raw_res| mean=3.0126 p95=3.0856
[15:48:03]           base_logit mean=-0.9479  final_logit mean=-1.6482  mean|final-base|=0.7002
[15:48:03] batch 1: 0.64s rows=8192 graph_available=8192/8192 identity_err_all=2.692e-07
[15:48:03] SMOKE OK
```

Two independent correctness checks passed:

* `graph edges = 7,190,167` and `binding_edges = 152,246` equal the authorized
  `expected_graph_edges` / `expected_binding_edges` — the C overlay's semantic gate (eCLIP
  present, predicted = 0, legacy flat = 0, no cancer-specific binding) held on the real payload.
* the checkpoint's `validation_loss` equals fold 0's `SUCCESS.json` best exactly.

## What the numbers say about the graph contribution

The bounded-residual design is `z_final = z_l1 + gate · tanh(raw_residual)`, and the
change log had recorded a caution that the `tanh` cap plus the gate might leave the graph
contribution negligible. On fold 0's validation rows that is **not** what happens:

| quantity | value |
|---|---|
| identity max abs error | 2.3e-07 (float32) — the formula holds exactly |
| gate min / mean / max | 0.0000 / **0.7037** / 1.0000 |
| abs raw residual mean / p95 | **3.0126** / 3.0856 |
| mean abs(final − base) | **0.7002** |
| base_logit mean → final_logit mean | −0.9479 → −1.6482 |

`tanh` saturates at |raw| ≈ 3, so the achievable contribution is essentially `gate × 1`.
With a mean gate of 0.70 the observed shift of 0.70 is **at the ceiling the design permits**:
the gate is open and the residual is saturated, not suppressed. The graph shifts scores
downward on average.

This is a descriptive statement about fold 0's validation rows only. It is not an AUPRC
result, and it does not by itself establish that the shift improves ranking.

## Cost structure (measured, not estimated)

| phase | per fold | ×5 |
|---|---|---|
| payload load | 143 s | 12 min |
| overlay load | 109 s | 9 min |
| overlay apply + graph verification | 617 s | 51 min |
| chunk contract + model construction | 76 s | 6 min |
| per chunk: 27.8 s graph + 36.7 s encode + 403 × 0.65 s decode | 326 s | — |
| chunk loop (×26) | ~2.35 h | ~11.8 h |
| **total** | **~2.6 h** | **~13 h** |

The setup phase is single-threaded pandas on 7.2M edges; torch's 24 threads only engage
during the chunk loop.

## Why the payloads are read from local disk

`/dell_2` is a `fuse.sshfs` mount. A `torch.load` of the 12.8 GB payload through it hung
outright — `wchan=request_wait_answer`, `rchar` frozen at 15,834,900 bytes across 2,316 read
syscalls for more than ten minutes — while a sequential `dd` on the same file still returned
2.1 MB/s. The same payload from local disk takes **142.9 s** (~270 MB/s). Payloads are therefore
pulled from the training instance over plain SSH (~17 MB/s) rather than copied through the mount.
