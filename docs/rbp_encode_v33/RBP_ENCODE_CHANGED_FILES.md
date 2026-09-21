# RBP_ENCODE_CHANGED_FILES.md

**Deliverable**: Phase 15 item 7 (modified-file list + SHA256)
**Date**: 2026-09-21

Origin baseline capsule (never modified):
`${PRIVATE_WORK_ROOT}/CancerLncAtlas/runtime/tools/v32_g012_corrected_paid_gpu_20260905_r1/code`

| status | file | work-root SHA256 | origin SHA256 |
|---|---|---|---|
| NEW | `cc_hhgt/v32/rbp_assay.py` | `33112d8d0ff6957d9cc3fccad7430b80…` | `(new file)…` |
| NEW | `cc_hhgt/v32/rbp_typed_binding.py` | `65c331e099eafaa790a86f94d16972bc…` | `(new file)…` |
| NEW | `cc_hhgt/v32/rbp_canonical.py` | `713e1732dd9324ee451e53e68c1e3ee3…` | `(new file)…` |
| NEW | `cc_hhgt/v32/rbp_evidence_config.py` | `cd9e846ffb4cb965b735b7143d47ed51…` | `(new file)…` |
| NEW | `tests/test_v32_rbp_assay.py` | `1bfab1857975725599ca99d4e94ffed2…` | `(new file)…` |
| NEW | `tests/test_v32_rbp_evidence_features.py` | `9b6a5fb37acc4b4309d56aaa91f0f69e…` | `(new file)…` |
| NEW | `tests/test_v32_rbp_typed_binding.py` | `a844b5a405d5148bf8f0a86aab39f60f…` | `(new file)…` |
| NEW | `tests/test_v32_rbp_typed_graph.py` | `e60995e999c982dc53ca7bf292012151…` | `(new file)…` |
| NEW | `tests/test_v32_rbp_canonical.py` | `afa65a080fee6a70d65c32c6e5748684…` | `(new file)…` |
| NEW | `tests/test_v32_rbp_evidence_config.py` | `c7adf47ed5a2f381813c90537eac6686…` | `(new file)…` |
| MODIFIED | `cc_hhgt/v32/evidence_training.py` | `88534ca47310df01ef137ba45bf99df2…` | `b112319c1596cb3337a6590828a3cdb2…` |
| MODIFIED | `cc_hhgt/v32/evidence_streaming_training.py` | `182f80e0780e728e6b5426795d75628c…` | `213dd8772ec63d3e37533c9c53f224f2…` |
| MODIFIED | `cc_hhgt/v32/formal_graph.py` | `9de3d0048c084b1f4f715a33744a5651…` | `04e3e1b1ead886b8cd1b08209de5060f…` |

## Files created outside the origin capsule

| file | purpose |
|---|---|
| `cc_hhgt/v32/rbp_assay.py` | deterministic assay taxonomy (two levels + graph class) |
| `cc_hhgt/v32/rbp_typed_binding.py` | typed lncRNA–protein/RBP binding materialiser + legacy-equivalence verifier |
| `cc_hhgt/v32/rbp_canonical.py` | canonical experiment identity and cross-database collapse |
| `cc_hhgt/v32/rbp_evidence_config.py` | `rbp_evidence` switch block, four modes, fairness enforcement |
| `config/rbp_ablation/*.yaml` | four generated ablation configs |

## Source files modified

| file | change |
|---|---|
| `cc_hhgt/v32/evidence_training.py` | taxonomy columns reach the event frame; assay-aware feature fields; `preserve_assay_type` threaded through `_event_feature_matrix` / `build_bag_examples` / `run_evidence_training`. `experiment_raw` provenance fixed (never the fallback). |
| `cc_hhgt/v32/evidence_streaming_training.py` | DuckDB streaming path kept column-identical: schema, `physical_all`, `all_exact_events`, `event_public_columns`. |
| `cc_hhgt/v32/formal_graph.py` | `CONTEXT_ECLIP_ROLE` added to `G1_ROLES`; schema 9 → 18 rows; `materialize_typed_lnc_protein_binding` added; explicit global/context collision rule. `materialize_global_lnc_protein_binding` left byte-identical. |

The frozen G012 invariant (`623,207 rows / 2,567 lncRNAs`) was never renegotiated: the legacy builder is untouched.
