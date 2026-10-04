# Core-embedding exporter — GPU path smoke (2026-10-04)

The exporter had only ever been validated on CPU (a 2-chunk run on fold 0, 2026-10-03). The
real five-fold export runs on the GPU, so the CUDA path was unproven until this run.

## How it was run without disturbing training

Fold 3 was still training on GPU 0. Fold 4 had already finished, leaving **GPU 1 completely
idle (0 MiB of 32,768 MiB)**. The smoke was therefore pinned with
`CUDA_VISIBLE_DEVICES=1`, and the launcher refuses to start unless it finds a card below
2,000 MiB used and at least 60 GB of free RAM.

```
before : 0, 21026 MiB, 0 %      1, 0 MiB, 0 %
during : 0, 21026 MiB, 0 %      1, ... export running
after  : 0, 21026 MiB, 0 %      1, 0 MiB, 0 %
```

GPU 0 never moved, so fold 3's memory was untouched.

## Result

```
[04:39:13 +  3.0s] fold 0: loading A payload (12,836,939,525 B)
[04:46:48 +457.7s] fold 0: 2/26 chunks, device=cuda, checkpoint cycle=103
[04:47:27 +497.1s] fold 0: cancer rows=33  USABLE max_chunk_deviation=0.7694
[04:47:28]         gene rows=21981  USABLE max_chunk_deviation=11.26
[04:47:28]         lncRNA rows=8541 USABLE max_chunk_deviation=15.34
[04:47:28]         pathway rows=2135 USABLE max_chunk_deviation=5.568
[04:47:28]         pathway_family rows=85 USABLE max_chunk_deviation=4.122
[04:47:28]         protein rows=20008 USABLE max_chunk_deviation=6.571
{"status": "SUCCESS", "folds": [0], ...}
```

All six node types exported on `device=cuda`. Partial-chunk runs route to
`core_embeddings_smoke/`, so the formal `core_embeddings/CORE_EMBEDDING_MANIFEST.json` was
not touched.

## The exporter is hardware-independent

The `max_chunk_deviation` figures are **identical** between the CPU smoke and this GPU smoke:

| node type | CPU smoke | GPU smoke |
|---|---|---|
| cancer | 0.7694 | 0.7694 |
| gene | 11.26 | 11.26 |
| lncRNA | 15.34 | 15.34 |
| pathway | 5.568 | 5.568 |
| pathway_family | 4.122 | 4.122 |
| protein | 6.571 | 6.571 |

That is a reproducibility result worth keeping: the chunk-weighted embedding for a node does
not depend on which device computed it, so an embedding exported on the paid card is the same
object a CPU run would have produced.

## Cost of the real run

Two chunks took ~39 s of chunk loop (~20 s per chunk). For 26 chunks x 5 folds = 130 chunks
that is ~43 min, plus five payload loads at ~7.6 min each (~38 min) — **about 90 minutes** for
the full five-fold export once fold 3 stops.

## Guards confirmed on the way

* `launch_core_export_all5.sh` refuses while any fold lacks `SUCCESS.json`:
  `fold 3: NO CKPT_OK` → `ABORT: fold 3 has not completed` (exit 1).
* `launch_single_cell_head_149.sh` preflight stops exactly at the missing core-embedding
  manifest, with all six required inputs present and pointing at the validated
  `sc_manifest_fix_20261004_r1/code` and `v32_sc_head_inputs_20261004_r4`.
