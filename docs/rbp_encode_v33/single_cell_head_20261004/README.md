# V3.2 single-cell private head — trained on the frozen five-fold core (2026-10-04)

Host `149`, CPU only. This is the auxiliary head the model config declares as
`FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH`; it consumes the core embeddings exported from the
five `C_GLOBAL_BINDING_G2` checkpoints and never writes back into the core.

Run id: `v32-single-cell-head-source-verified-20261004-r1`
Output root: `$R/results/single_cell_head_v32-single-cell-head-source-verified-20261004-r1`
(`$R` = `/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1`)

## Result

`SUCCESS.json` → `status: SUCCESS`, `release_ready: true`, `trained_folds: 5`,
`contract_validation_status: PASS`, `all_unavailable_rows_have_reason: true`,
`all_probabilities_null: false`.

The full-model contract validator (`cc_hhgt.v32.full_model_contract.validate_module_lineage`,
`fail_closed: true`) returned `PASS` for `module_id: single_cell` against
`training_status: SUCCESS`.

## Five private heads, one per core fold

| fold | checkpoint | core checkpoint sha256 (from the export manifest) | fresh init | train rows | validation rows |
|---|---|---|---|---|---|
| 0 | `checkpoints/single_cell_fold_0.pt` | `8a36a8fada6d77f2…` | yes | 121,801 | 28,768 |
| 1 | `checkpoints/single_cell_fold_1.pt` | `c36f440b74fab0bd…` | yes | 118,075 | 31,221 |
| 2 | `checkpoints/single_cell_fold_2.pt` | `01c7e7dfd195a454…` | yes | 76,100 | 70,743 |
| 3 | `checkpoints/single_cell_fold_3.pt` | `742260a374e8b6e6…` | yes | 87,484 | 19,837 |
| 4 | `checkpoints/single_cell_fold_4.pt` | `c4a97da8dc659a88…` | yes | 130,732 | 27,495 |

Format string: `CC_HHGT_V3_2_SINGLE_CELL_PRIVATE_HEAD_V1`.

Two checks worth naming, both read off `CHECKPOINT_MANIFEST.json`:

* **The per-fold `core_checkpoint_sha256` values match the checkpoints the export manifest
  recorded for the same folds**, so each head is trained against the embeddings actually
  produced from its own fold's core, not a neighbour's.
* **The five folds' validation rows sum to 178,064**, which is exactly
  `available_rows` in `SUCCESS.json` — no evaluation row is double counted or dropped.

`core_parameters_before_sha256 == core_parameters_after_sha256`
(`4a41bb1e79c2da11…`) with `core_parameters_frozen: true` and
`core_detached_and_frozen: true`: the core did not move during head training.

`old_checkpoint_loaded`, `old_predictions_used_as_features`, `old_rankings_used_as_outputs` and
`old_family_support_used` are all `false`; `all_private_heads_random_initialization: true` with a
distinct `initial_parameter_sha256` per fold.

## Inputs and the cohort that was actually used

`MANIFEST_RECONCILIATION.json` → `status: PASS`, `target_host: 149`.

| quantity | value |
|---|---|
| datasets in the formal single-cell manifest | 33 |
| `formal_eligible` datasets | **28** |
| qualified cancers | **28** |
| cancers that bound at least one training row | **27** |
| `fold_splits_pairwise_disjoint` | `true` |
| `cross_source_cell_and_namespaced_patient_overlap` | `0` |
| `fold_reassignment_policy` | `EXISTING_SORTED_BLOCK_ROUND_ROBIN_RECOMPUTED_FOR_NEW_COHORT_NO_OLD_HEAD_RESUME` |
| `partition_isolation` | `DATASET_LEVEL_AGGREGATED_ASSOCIATIONS_NOT_DONOR_LEVEL` |
| `split_unit` | `dataset_or_donor` |

The gap between 28 bound cancers and 27 training cancers is **DLBC**, which is qualified and
bound but yields no association rows after the donor-replication filter.

`partition_isolation` is quoted verbatim because it is a real boundary: the associations are
aggregated at dataset level, so "disjoint folds" here means disjoint datasets/donors in the
manifest, not donor-level rows in every cancer.

## Typed predictions

`single_cell_typed_predictions.parquet` — 778,064 rows, sha256
`bd35241336192f075375dd05ee291e55dc03e8d479bc55aac6c8641ae4121db9`.

| property | value |
|---|---|
| `prediction_format` | `CC_HHGT_V3_2_SINGLE_CELL_TYPED_PREDICTIONS_V1` |
| `target_level` | `dataset_x_celltype_x_lncrna_x_exact_pathway` (single level, all rows) |
| `changes_primary_ranking` | `false` (all rows) |
| available rows | 178,064 |
| null rows with a recorded reason | 600,000 |
| distinct lncRNA / pathway / cell-type class / dataset | 4,902 / 2,135 / 4 / 33 |
| `single_cell_replication_probability` | min 0.333916, max 0.733086, mean 0.548677 |

Cancers with at least one available row: **27**. Cancers with zero available rows: **DLBC, KICH,
KIRP, LIHC, THCA, UCS** (100,000 rows each, all carrying a reason).

The 600,000 null rows are not silent:

| `single_cell_unavailable_reason` | rows |
|---|---|
| `NO_FORMAL_QUALIFIED_SINGLE_CELL_DATASET` | 200,000 |
| `DONOR_CELLTYPE_METADATA_UNAVAILABLE` | 200,000 |
| `NO_NON_STAGING_DONOR_OR_DATASET_ASSOCIATION` | 100,000 |
| `LIMITED_LNCRNA_FEATURE_UNIVERSE` | 100,000 |

`staging_never_promoted: true` and `staging_datasets_not_promoted: []`.

## The lncRNA core is usable, so the old fallback is gone

`constant_lncrna_core_masked: false`, and every fold's `core_embedding_usability` reports
lncRNA `status: USABLE` with 8,541 rows × 96 features, `varying_feature_count: 96`, and
7,906–8,282 distinct embedding vectors. `lncrna_fallback` is `null` in all five folds.
Pathway embeddings are likewise `USABLE` (2,135 rows, 2,135 distinct vectors).

This matters because `single_cell_training.candidate_core()` masks the lncRNA branch when the
core lncRNA embedding is constant. It is not constant here, so the head consumes real lncRNA
core features — and any comparison against an older single-cell result must account for the
changed feature set.

## Not claimed

* This is **not** a performance result. No AUPRC/AUROC is reported for this head here; the run's
  own success criteria are contract and lineage checks, and they passed.
* `FIGURE_MANIFEST.json` has 33 entries, **all** `NULL_WITH_REASON`
  (`NO_REGISTERED_SINGLE_CELL_FIGURE`). No single-cell figure was produced, and none is claimed.
* `cell_state` is null in every prediction row; only `cell_type` / `cell_type_major` are populated
  (immune / malignant / stromal, plus null on unavailable rows).
* The head has **zero fusion weight** into the primary model (frozen 2026-09-04), consistent with
  `changes_primary_ranking: false` on every row.

## Files

| file | content |
|---|---|
| `SUCCESS.json` | run status, row counts, prediction and lineage hashes |
| `LINEAGE.json` | full lineage: input hashes, core freeze hashes, policies, fold/seed list |
| `CHECKPOINT_MANIFEST.json` | `CC_HHGT_V3_2_SINGLE_CELL_PRIVATE_HEAD_V1` manifest: per-fold checkpoint hashes, init hashes, core usability audit |
| `RUN_CONFIG.json` | training hyperparameters and the four registered policies |
| `FULL_MODEL_CONTRACT_VALIDATION.json` | fail-closed module-lineage validation verdict |
| `INPUT_LINEAGE_AUDIT.json` | read-only audit of the four input artifacts (all `PASS`, all `outcome_derived: false`) |
| `FIGURE_MANIFEST.json` | 33 entries, all null with reason |
| `MANIFEST_RECONCILIATION.json` | cohort reconciliation: 28 formal-eligible datasets, 27 training cancers, disjoint folds |

The two data products (`single_cell_typed_predictions.parquet`, `activity.parquet`) and the five
`checkpoints/*.pt` files stay on `149` under the output root: this repository's `.gitignore`
excludes `*.parquet` and `*.pt`, and their sha256 values are recorded above and in `LINEAGE.json`.
