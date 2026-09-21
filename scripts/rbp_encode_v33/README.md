# RBP / ENCODE v3.3 analysis scripts

These scripts produced the RBP evidence repair and the typed-binding graph
generation. They are the executable counterpart to `docs/rbp_encode_v33/`.

Paths are written as `${PRIVATE_WORK_ROOT}` following this repository's
convention. Set it to the absolute root that contains `CancerLncAtlas/` before
running anything, for example:

```bash
export PRIVATE_WORK_ROOT=/path/to/your/data/root
```

## Order of operations

| # | script | purpose |
|---|---|---|
| 1 | `phase3_materialize.py` | build the typed lncRNA–protein/RBP binding table from the frozen `standardized/` inputs; proves legacy equivalence |
| 2 | `phase4_canonical.py` | canonical experiment identity and cross-database duplicate audit |
| 3 | `phase4_npinter_counts.py` | NPInter5 assay distribution; quantifies the historical collapse and the mislabelled predictions |
| 4 | `phase5_encode_manifest.py` | audit the requested ENCODE accessions and emit the download manifest; fails closed on an assembly mismatch |
| 5 | `phase7_typed_authority.py` | materialise the typed-relation G012 authority under its own receipt generation |
| 6 | `phase7b_typed_payload.py` | produce the typed prepared payload by surgical substitution, reusing the frozen batches and their LASSO logits |
| 7 | `phase14_bundle_check.py` | verify that typed relations appear in the `HeteroData` metadata, not merely in a table |
| 8 | `phase11_generate_ablations.py` | write the four ablation configs and the fairness report |
| 9 | `phase11_lasso_binding.py` | fingerprint the existing LASSO baseline so it is reused by digest rather than refitted |
| 10 | `phase12_preflight_compute.py`, `phase12_preflight_report.py` | compute and render the CPU hard-gate report |
| 11 | `phase15_docs_*.py` | regenerate the deliverable documents from live state |
| — | `monitor.sh` | print job status, fold progress and the combined authority summary |

## Environment

Everything runs on the CPU host. See `docs/rbp_encode_v33/RUN_149_CPU.md` for the
environment setup and the four packaging defects that had to be worked around
(`pip install --target` losing packages on this filesystem, an incomplete `sympy`,
incomplete `torch_geometric` staging directories, and a cp313-only `xxhash` wheel
against a cp310 interpreter).

## Notes that matter

* **`phase7b_typed_payload.py` fails closed.** It compares the node universe
  against the frozen authority records before touching anything. Reusing the
  frozen index-based batches across a changed node universe would misalign
  silently, so the script refuses rather than proceeding.
* **The LASSO baseline is not refitted.** The frozen payload already carries the
  per-fold `base_logit`, and the node universe is identical between generations,
  so it is carried over verbatim.
* The typed generation declares `same_relation_schema_all_variants: False`. That
  is accurate: it genuinely does not share the legacy relation schema. The node
  universe is what stays identical.
