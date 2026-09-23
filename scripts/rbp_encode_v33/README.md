# RBP / ENCODE v3.3 analysis scripts

These scripts produced the RBP evidence repair, the typed-binding graph
generation, and the ENCODE eCLIP ingestion. They are the executable counterpart
to `docs/rbp_encode_v33/`.

Paths are written as placeholders following this repository's convention. **Two
distinct roots exist on the analysis host** and collapsing them into one is a
real bug, not a stylistic choice:

| placeholder | host path | holds |
|---|---|---|
| `${PRIVATE_WORK_ROOT}` | `/dell_2/DSC` | work root, frozen `inputs/`, `runtime/` |
| `${PRIVATE_ARCHIVE_ROOT}` | `/public8/DSC` | frozen archive: `results/v3_state_formal_runs/` |

```bash
export PRIVATE_WORK_ROOT=/path/to/your/main/root
export PRIVATE_ARCHIVE_ROOT=/path/to/your/archive/root
```

Scripts that read the frozen standardized assets (`phase3_materialize.py`,
`phase11_lasso_binding.py`, `pathb_step0_lncrna_coords.py`) use
`${PRIVATE_ARCHIVE_ROOT}`; everything else uses `${PRIVATE_WORK_ROOT}`.

## Order of operations

### A. RBP evidence repair and the typed graph

| # | script | purpose |
|---|---|---|
| 1 | `phase3_materialize.py` | build the typed lncRNA–protein/RBP binding table from the frozen `standardized/` inputs; proves legacy equivalence |
| 2 | `phase4_canonical.py` | canonical experiment identity and cross-database duplicate audit |
| 3 | `phase4_npinter_counts.py` | NPInter5 assay distribution; quantifies the historical collapse and the mislabelled predictions |
| 4 | `phase7_typed_authority.py` | materialise the typed-relation G012 authority under its own receipt generation |
| 5 | `phase7b_typed_payload.py` | produce the typed prepared payload by surgical substitution, reusing the frozen batches and their LASSO logits |
| 6 | `phase11_generate_ablations.py` | write the four ablation configs and the fairness report |
| 7 | `phase11_lasso_binding.py` | fingerprint the existing LASSO baseline so it is reused by digest rather than refitted |
| 8 | `phase12_preflight_compute.py`, `phase12_preflight_report.py` | compute and render the CPU hard-gate report |
| 9 | `phase15_docs_*.py` | regenerate the deliverable documents from live state |

### B0. The reproducible-peak and exon correction (current)

Supersedes section B. Run in this order.

| # | script | purpose |
|---|---|---|
| A | `step_a_reproducible.py` | fetch `preferred_default`; show that it is **not** the reproducible flag |
| A2 | `step_a2_structure.py` | each experiment ships 3 peak calls per assembly; identify the combined call |
| A3 | `step_a3_peakclass.py` | count real peaks per class: combined is 1.5 % of either single replicate |
| B | `step_b_exons.py` | strand-aware exon authority from the GENCODE releases already named by the assets |
| C+D | `step_cd_overlap.py` | lift the hg19 reproducible peaks, overlap at exon level, dedup by experiment |
| D2 | `step_d2_merge.py` | ruling B on the context gate + the ENCODE v2 edges, one merged table |
| E | `step_e_materialize.sh` | **the single materialisation** of the v2 generation, folds 0-4 |
| F | `step_f_shuffles.py` | the two shuffle controls for C vs C-shuffle |

Measurements that back the corrections:

| script | question |
|---|---|
| `ruling_b_impact.py` | what ruling B changes, with the frozen 623,207 reproduced first |
| `cell_line_accounting.py` | which cell lines from which databases never reach the graph |
| `measure_context.py`, `measure_context2.py` | what "global" means in this project, and what the gate drops |
| `measure_predicted.py` | what admitting `binds_protein_predicted` would do, under the candidate filter |

### B. ENCODE eCLIP ingestion (first attempt, superseded)

Run in this order. Each step writes a receipt the next one checks.

| # | script | purpose |
|---|---|---|
| 0 | `pathb_step0_lncrna_coords.py` | pin the GRCh38 coordinate authority for the 8,541 lncRNA nodes; **fails closed** unless the probe loci verify as GRCh38 |
| 1 | `pathb_step1_chain.py` | pin the hg19→hg38 chain by SHA256 and verify its direction by chromosome **size** |
| 2 | `phase6_encode_manifest.py` | file-level ENCODE manifest: proves the plan's six accessions are `PublicationData` FileSets, then enumerates eCLIP narrowPeak files with their per-file assembly |
| 3 | `phase6b_encode_download.py` | download the eCLIP narrowPeak surface, verifying md5 **and** byte count; unverifiable payloads are deleted |
| 4 | `pathb_step2_liftover.py` | explicitly lift the hg19 peaks to GRCh38; peaks are quarantined, never truncated |
| 5 | `phase3_encode_eclip_overlap.py` | overlap every peak with the lncRNA loci and emit typed `binds_protein_eclip` edges |
| 6 | `pathb_step5_lineage.py` | declare the lineage of every edge back to file accession, md5 and chain hash |
| 7 | `phase3b_merge_encode_binding.py` | merge ENCODE into the typed binding layer **without overwriting** the pre-ENCODE artifact |
| 8 | `phase7_typed_authority.py` (again) | materialise the ENCODE-inclusive authority, with `RBP_BINDING_PARQUET` and `RBP_AUTHORITY_OUT` pointed at new paths |
| 9 | `phase13_four_mode_ablation.py` | the CPU half of the four-mode ablation |
| 10 | `phase9_10_layers.py` | declare the knockdown and RBNS layers, gated off |
| 11 | `phase15_docs_d.py` | generate the ENCODE ingestion report from the receipts |
| — | `phase14_bundle_check_v2.py` | verify the bundle arms by **active edges**, not schema presence |
| — | `monitor.sh` | print job status, fold progress and the combined authority summary |

## Environment

Everything runs on the CPU host. See `docs/rbp_encode_v33/RUN_149_CPU.md` for the
environment setup and the four packaging defects that had to be worked around
(`pip install --target` losing packages on this filesystem, an incomplete `sympy`,
incomplete `torch_geometric` staging directories, and a cp313-only `xxhash` wheel
against a cp310 interpreter).

Network: the host has no direct outbound route. A SOCKS relay is required, and its
address is environment-overridable because the original `127.0.0.1:1080` relay
died mid-download:

```bash
export ENCODE_PROXY=socks5h://127.0.0.1:1080   # default
export ENCODE_DOWNLOAD_WORKERS=4               # default 3; 8 saturates some relays
```

## Notes that matter

* **`phase7_typed_authority.py` refuses to overwrite a generation.** Set
  `RBP_AUTHORITY_OUT` to a new directory, or `RBP_ALLOW_OVERWRITE=1` to replace one
  deliberately. The pre-ENCODE authority stays on disk next to the ENCODE one.
* **`phase7b_typed_payload.py` fails closed.** It compares the node universe
  against the frozen authority records before touching anything. Reusing the
  frozen index-based batches across a changed node universe would misalign
  silently, so the script refuses rather than proceeding.
* **The LASSO baseline is not refitted.** The frozen payload already carries the
  per-fold `base_logit`, and the node universe is identical between generations,
  so it is carried over verbatim.
* **Schema presence is not active edges.** Every bundle arm carries the complete
  G2 relation schema as empty tensors, so `HeteroData.edge_types` lists every
  relation regardless of arm. `phase14_bundle_check_v2.py` reports both and is the
  only check that means anything.
* The typed generation declares `same_relation_schema_all_variants: False`. That
  is accurate: it genuinely does not share the legacy relation schema. The node
  universe is what stays identical.
