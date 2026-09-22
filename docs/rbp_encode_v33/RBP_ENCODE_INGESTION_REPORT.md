# ENCODE RBP ingestion report

Generated 2026-09-22T06:27:00.907929+00:00 by `scripts/rbp_encode_v33/phase15_docs_d.py` from the pipeline receipts.

This report answers one question: **did ENCODE data actually reach the model, and can every edge be traced back to bytes on disk?**

## 1. The plan's accessions were not experiments

A `type=Experiment` search returns HTTP 404 / `total = 0` for all six accessions named by the plan, while the control experiment resolves. The six are `PublicationData` FileSets, so the plan's manifest could only ever report the FileSet's own dominant assembly.

| accession | `type=Experiment` total |
| --- | --- |
| ENCSR456FVU | 0 (http None) |
| ENCSR369TWP | 0 (http None) |
| ENCSR795JHH | 0 (http None) |
| ENCSR413YAF | 0 (http None) |
| ENCSR870OLK | 0 (http None) |
| ENCSR876DCD | 0 (http None) |
| ENCSR720BJU | 1 (http None) |

File-level enumeration of those FileSets:

| accession | role | files | FileSet `assembly` field |
| --- | --- | --- | --- |
| ENCSR456FVU | eclip_peaks_fileset | 2,670 | ['hg19'] |
| ENCSR369TWP | kd_rnaseq_hepg2 | 2,128 | ['hg19'] |
| ENCSR795JHH | kd_rnaseq_k562 | 2,112 | ['hg19'] |
| ENCSR413YAF | kd_processed_hepg2_k562 | 1,888 | ['hg19'] |
| ENCSR870OLK | kd_batch_corrected | 944 | ['hg19'] |
| ENCSR876DCD | rbns | 827 | ['hg19'] |

## 2. The resource is not hg19-only

The plan parked ENCODE as `ASSEMBLY_MISMATCH` because the FileSet reports `hg19`. At file level the eCLIP narrowPeak resource releases both assemblies:

| assembly | biosample | files | MiB | RBPs |
| --- | --- | --- | --- | --- |
| GRCh38 | HepG2 | 315 | 372.8 | 105 |
| GRCh38 | K562 | 435 | 384.0 | 139 |
| GRCh38 | adrenal gland | 6 | 4.7 | 2 |
| hg19 | HepG2 | 309 | 367.8 | 103 |
| hg19 | K562 | 360 | 316.5 | 120 |
| hg19 | adrenal gland | 6 | 4.9 | 2 |

Total: **1431 files, 1450.7 MiB**.

The GRCh38-native half needs no coordinate change at all; only the hg19 half is lifted, and it is lifted explicitly.

## 3. Coordinate authority

| item | value |
| --- | --- |
| item | value |
| --- | --- |
| lncRNA nodes | 8541 |
| nodes without coordinates | 0 |
| coordinate source | /public8/DSC/CancerLncAtlas/results/v3_state_formal_runs/V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized/dim_lncRNA.parquet |
| annotation release | {'GENCODE_v50': 34866, 'GENCODE_v36_GDC_DR45': 1150} |
| assembly | GRCh38 |
| external annotation introduced | False |

Assembly verification on three probe loci (GRCh38 truth):

| locus | observed | expected GRCh38 | delta (start, end) | verdict |
| --- | --- | --- | --- | --- |
| NEAT1 | chr11-65422774-65445540 | chr11-65422774-65445540 | (0, 0) | GRCh38 |
| XIST | chrX-73820649-73852714 | chrX-73820649-73852753 | (0, 39) | GRCh38 |
| MALAT1 | chr11-65497606-65508073 | chr11-65497688-65506516 | (82, 1,557) | GRCh38 |

The coordinates come from the project's own standardised `dim_lncRNA` asset, so the `LNC:ENSG…` identifier space cannot drift. No external annotation is introduced.

## 4. Download verification

| item | value |
| --- | --- |
| files on the consumption surface | 1,431 |
| verified downloads | 1,431 |
| verified bytes | 1450.7 MiB |
| failures | 0 |

Verification: md5sum and byte count must both equal the values published by ENCODE.

## 5. Explicit hg19 to GRCh38 liftOver

Silent liftOver is forbidden, so the chain file is pinned and its direction is verified by chromosome **size** (`chr1` 249,250,621 hg19 -> 248,956,422 GRCh38), not by chromosome name, which is identical in both assemblies.

| item | value |
| --- | --- |
| chain sha256 | 5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d |
| chain records | 1278 |
| direction verified by | chromosome size (hg19 chr1 249250621 -> hg38 chr1 248956422) |
| aligned blocks indexed | 53950 |

| outcome | peaks |
| --- | --- |
| total | 60,108,082 |
| mapped | 60,088,294 (99.97%) |
| quarantined | 19,788 |
| dropped | 0 |

Quarantine reasons:

| reason | peaks |
| --- | --- |
| `chromosome_absent_from_chain` | 0 |
| `not_contained_in_single_aligned_block` | 19,788 |
| `lifted_interval_out_of_query_bounds` | 0 |

Criterion: A peak maps only if a single aligned block of a single chain contains it end to end and the lifted interval stays inside the query chromosome. Peaks straddling an alignment gap are quarantined, never truncated.

## 6. What the overlap produced - and why it is not in the graph

| item | value |
| --- | --- |
| eCLIP files used | 1,431 (GRCh38-native 756, lifted 675) |
| RBP symbols resolved to a protein node | 165 / 168 |
| evidence rows (lncRNA x RBP x file) | 1,745,070 |
| global `binds_protein_eclip` edges built | 428,949 |
| context-specific edges built | 253,544 |
| lncRNAs touched | 6,885 |
| protein nodes touched | 165 |

Biosample to cancer mapping: `{'HepG2': 'LIHC'}`. Biosamples with **no** cancer mapping: `['K562', 'adrenal gland']`. K562 is CML, which is not one of the 33 cancers, so it is never forced onto LAML; HepG2 produces LIHC context edges only and is never broadcast to all 33.

### The layer is gated off

**Decision: GATED OFF - built but not admitted to the main graph.** Switch `include_encode_eclip_overlap` defaults to `False`.

A peak overlapping a lncRNA gene body is not by itself evidence that the RBP bound that lncRNA. Two null models were applied and they disagree:

| null | result |
| --- | --- |
| uniform placement, peak level | GRCh38 0.71x, hg19 0.74x - **depletion** |
| per-pair Poisson | 89.6% of pairs at >=2x, median 28.3x |

Two nulls disagree: peaks land in lncRNA gene bodies less often than uniform placement predicts (0.71-0.74x), while a per-pair Poisson test calls 89.6% of pairs enriched because it is conditioned on the pair existing and assumes peaks are uniform. Neither null preserves the clustering of eCLIP peaks, and until one does, gene-body overlap cannot be told apart from position.

Merging would have taken the typed binding table from 918,266 to 1,347,215 rows. That is exactly why the measurement was made before merging.

Criterion before this layer may be admitted:

1. a null model that preserves the genome-wide clustering of eCLIP peaks (for example a per-RBP peak-density background computed from the same files, or a dinucleotide-matched shuffle)
1. an enrichment measured against that null that exceeds 1 with a stated confidence interval
1. a peak-level criterion rather than a gene-body one, ideally requiring the peak to fall in an annotated exon of the lncRNA

### Merge

Not performed. `phase3b_merge_encode_binding.py` is the step that would merge the layer, and the gate above is why it was not run.

### Lineage

1,721,605 lineage rows cover 1,410 source files (native 744, lifted 666), 6,885 lncRNAs and 165 RBPs. Checks: `manifest_and_receipt_status_agree`, `all_cited_files_verified`, `every_lifted_file_cites_the_pinned_chain`.

## 7. Every ENCODE layer, and its gate

| layer | built | admitted to the main graph | switch |
| --- | --- | --- | --- |
| eCLIP gene-body overlap | 428,949 edges | **no** - see section 6 | `include_encode_eclip_overlap = False` |
| eCLIP coordinates and lineage | 1721605 rows | yes (provenance only) | - |
| RBP knockdown RNA-seq | 7,072 files inventoried | **no** | `include_knockdown_evidence = False` |
| RBNS | 827 files inventoried | **no** | `include_predicted_evidence = False` |

The reason each is gated off:

* **RBP knockdown RNA-seq** - A knockdown measures abundance change after depletion. Treating it as lncRNA functional truth would present a perturbation response as a physical interaction, which the plan forbids.
* **RBNS** - RBNS reports an in-vitro k-mer preference. Any lncRNA-level statement built from it is a prediction, and predictions are excluded from the main graph by the conservative default.
* **eCLIP gene-body overlap** - the peak-level enrichment measurement came out below 1 (see section 6).

Inventoried but not downloaded: the knockdown resource is 10148.6 GiB. The RBNS small products (347 files, 79.8 MiB) were fetched and verified, because at that size the layer can be present rather than merely described.

## 8. Four-mode ablation

| mode | binding layer | binding rows | active binding edges | status |
| --- | --- | --- | --- | --- |
| A `legacy_generic_binding` | frozen flat binds_protein | 623207 | PENDING | not materialised |
| B `typed_binding_only` | typed assay-class relations | PENDING | 98873 | materialised |
| C `typed_binding_plus_encode_eclip` | typed assay-class relations + ENCODE eCLIP | PENDING | PENDING | gated_off |
| D `full_rbp_evidence` | identical to C; RBP knockdown enters the Evidence layer only | PENDING | PENDING | not materialised |

Mode C is `GATED OFF - built but not admitted to the main graph`. It built 428949 global and 253544 context ENCODE edges, and did not materialise them into a graph generation. Until it does, modes B, C and D produce the same graph, which is stated here rather than hidden behind a table of identical numbers.

Mode D is graph-identical to mode C by construction: knockdown evidence is declared `evidence_only` and `assert_fair_comparison` forbids it from the primary graph.

Resolved on CPU: each mode's binding layer contribution, per relation; the four modes' identical fairness axes; that RBP knockdown never becomes a primary-graph edge in any mode.

**Not** resolved on CPU: AUPRC per mode - requires the GPU that the CPU gate holds shut; any statement about which mode ranks better.

## 9. CPU gate

| gate | status |
| --- | --- |
| encode_manifest | PASS |
| genome_assembly | PASS |
| encode_download | PASS |
| liftover_quarantine | PASS |
| encode_adds_no_nodes | PASS |


## 10. Prohibitions, and how each is held

| prohibition | mechanism |
| --- | --- |
| no new RBP node type | ENCODE RBPs map onto the existing `protein` nodes; `encode_adds_no_nodes` fails if any new node appears |
| knockdown is not lncRNA functional truth | declared `evidence_only`; `rbp_kd_is_lncrna_function_truth` is a hard `False` in every mode |
| no context-specific broadcast | only HepG2 -> LIHC is mapped; K562 is CML and is never forced onto LAML |
| no silent liftOver | the chain is pinned by SHA256 and every mapped peak records its chain id, strand and original coordinates |
| no overwriting of formal artifacts | every output goes to a new path; the typed authority refuses to write into a populated directory |
| no GPU while the CPU gate is open | no training is launched by any script here |

