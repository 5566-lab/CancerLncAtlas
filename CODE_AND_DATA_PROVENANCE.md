# Code and data provenance

This bundle freezes the CancerLncAtlas CC-HHGT v2.3 GPU orchestration while
retaining the corrected v2.1 label, feature and LOCO-isolation semantics.

## Corrected semantics

- `strong_positive` and `weak_positive` are both positive observations.
- Weak positives receive the configured lower confidence weight.
- `unlabeled` candidates contribute to the PU negative-risk term.
- The unlabeled sampling ratio is calculated from the sampled positives.
- Node degree features are recomputed after LOCO context exclusion and relation sampling.
- Test and validation cancer-specific edges are excluded before graph construction.

## Upstream model inputs

The graph and candidate tables were built on the server by stages 00–10 and
are copied as immutable local inputs under:

`results/model/cc_hhgt_v2_1_gpu/tables/`

The GPU package does not contain raw TCGA expression, single-cell RDS/H5,
PRISM/GDSC raw matrices, DepMap raw expression, or any download cache.

## Frozen source

The bundle manifest and `BUNDLE_SHA256SUMS` record every transferred file.
The CPU feasibility validation used Python 3.11, PyTorch 2.13 CPU and PyG
2.8.0.post1. The GPU environment intentionally installs a CUDA-enabled PyTorch
wheel and PyG 2.8.0.

The GPU-specific changes add mixed precision, peak-VRAM recording, resumable
single-GPU orchestration, CUDA OOM edge-cap fallback, float16 node-embedding
export and a three-seed stability analysis. They do not change the corrected
label or fold-isolation rules.

## Multiseed contract

The primary seed `20260726` writes the production 93 model/fold directories.
Seeds `20261726` and `20262726` write another 186 directories under
`models_multiseed/` without overwriting production checkpoints. Cancer folds
are identical across seeds; initialization, sampled graph relations and
sampled training pairs change.

Temperature calibration is fit independently from each fold's validation
cancer. Reported 95% confidence intervals are Student-t intervals across the
three seed-level mean LOCO metrics. Every model/fold exports a float16
node-type tensor bundle whose rows align with
`graph_node.node_index_within_type`.

## Isolated external evidence

Lnc2Cancer, LncRNADisease, RNADisease and the official GSE85011 GEO sample
metadata snapshot are frozen under `input/` and standardized separately from
the graph/evidence training chain. The external manifest records source counts,
lncRNA/TCGA mappings, PMID overlap with training evidence and GSE85011 target
overlap.

GSE85011 contributes only the experimentally supported statement that a locus
was a growth-modifier hit selected for follow-up RNA-seq in the listed cell
line. The downloaded GEO Series metadata does not contain a per-locus screen
effect or direction, so the exported direction is deliberately `unknown`.
