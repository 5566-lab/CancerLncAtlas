# V3.2 core-embedding estimand for the auxiliary private heads

Decision record, 2026-10-03. Host `149` for all CPU work; export itself requires CUDA.

## 1. The gap this closes

The auxiliary heads (`single_cell`, `state`, `genomic`, `drug`, `clinical`, `evidence`) are
declared in `config/model_v3_2_full_multitask.yaml` with
`initialization_policy: FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH`, and
`single_cell_training.validate_core_manifest()` requires a
`CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1` export of the **newly trained** core. That export
did not exist for the corrected core, and the only exporter in the tree could not produce it:

| obstacle | evidence |
|---|---|
| stale checkpoint format | `scripts/export_v32_core_embeddings.py:21` pins `CC_HHGT_V3_2_FULL_TRAINING_STATE_V1`; `training.py:24` emits `..._V4_GROUP_SCHEDULE` |
| CUDA-only | `export_v32_core_embeddings.py:85` raises when CUDA is unavailable |
| single-graph only | it calls `model.encoder.encode(graph)` once; every chunk-aware encode site (`training.py:2303`, `validation_prediction_inference.py:117`) yields **logits**, never node embeddings |
| no overlay support | it never loads the C global-binding overlay, and `c_graph_overlay.apply_c_graph_overlay` sets `result["graph"] = None` |

Because the V4 core is defined over a 26–29 chunk runtime graph, a node has **no single
chunk-independent embedding**: rotating edges put the same node in a different neighbourhood
in each chunk.

## 2. The decision

    core_embedding(node) = Σ_c  w_c · encoder(chunk_c)[node]

with `w_c` taken unchanged from `cc_hhgt.v32.training._runtime_chunk_contract` — the same
registered weights the formal validation/test logit aggregation uses, which are derived from
per-chunk rotating-edge counts and are asserted to sum to 1.

Recorded in the emitted manifest as
`embedding_estimand = REGISTERED_CHUNK_WEIGHTED_MEAN_OF_ENCODER_OUTPUTS`.

### Why this choice

* It reuses the **registered** aggregation rule instead of inventing a new one, so the head's
  inputs are on the same footing as the primary model's own outputs.
* Every alternative is worse: encoding the union graph presents the encoder with a graph it
  never saw during training; picking one arbitrary chunk privileges that chunk; exporting all
  chunks multiplies the artifact and pushes the choice into every consumer.

### Honest limitations, recorded in the artifact

* It is **not** claimed to equal `encoder(union_graph)`. The union-graph encode is computed
  separately and reported as `union_graph_encode_diagnostic`, purely so the two can be compared.
* The decoder applies a nonlinearity per chunk, so a weighted mean of encoder outputs is not a
  weighted mean of decoder outputs. The estimator is defined in encoder space; no
  compositionality through `tanh` is claimed.
* `max_abs_chunk_deviation_from_weighted_mean` is recorded per node type. Measured on fold 0 it
  is 4.1–15.3, i.e. **the chunk choice is material, not cosmetic** — the estimand decision was
  necessary rather than a formality.

## 3. Verification performed

The runner is `export_core_embeddings_instance.py`, staged on the training instance **outside**
the hashed code tree so `code_tree_sha256` remains
`bad3135ba858bf36fedd4a102eb509ddb110872844d5bf1def90a93274c0ba58`.

A CPU/2-chunk smoke on fold 0 exercised the full chain (authorized A payload → C overlay with
its node-universe / eCLIP / predicted-zero / legacy-flat / global-source checks → V4 checkpoint
→ chunked encode → weighted mean → parquet) and produced all six node types:

```
gene rows=21981  lncRNA rows=8541  pathway rows=2135
pathway_family rows=85  protein rows=20008  cancer rows=33
all status=USABLE
```

`single_cell_training.validate_core_manifest()` was run against the smoke manifest and passed
every check up to the intended `Core export requires exactly folds 0..4`, confirming the schema
is consumable by the real consumer.

The smoke record's `artifact_hashes.code_sha256` reproduced
`bad3135ba858bf36fedd4a102eb509ddb110872844d5bf1def90a93274c0ba58`, confirming the embeddings
come from the same tree that trained the fold.

## 4. Consequence worth noting

`single_cell_training.candidate_core()` zeroes the lncRNA branch unless
`embedding_usability(core.lncRNA)["usable_for_node_discrimination"]` holds, with the comment
that the historical formal export had a constant lncRNA vector. **That no longer applies**:
the smoke export reports `lncRNA rows=8541, unique_embedding_vectors=7116,
varying_feature_count=96, status=USABLE`. The single-cell head will therefore consume real
lncRNA core features, and any comparison against an older single-cell result must account for
the changed feature set.

Partial (chunk-limited) runs write to `core_embeddings_smoke/` and never to the formal
`core_embeddings/CORE_EMBEDDING_MANIFEST.json`.
