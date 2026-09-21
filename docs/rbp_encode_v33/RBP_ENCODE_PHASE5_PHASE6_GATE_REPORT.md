# RBP_ENCODE_PHASE5_PHASE6_GATE_REPORT.md

**Task**: CancerLncAtlas V3.2 — RBP evidence repair + ENCODE RBP integration
**Phases covered**: 5 (ENCODE resource audit / download manifest) and 6 gate (eCLIP peak mapping)
**Date**: 2026-09-21
**Server**: `149`
**Verdict**: **FAIL-CLOSED — no ENCODE file was downloaded**

---

## 1. Phase 5 requirement: check project directories before downloading

The plan forbids a full-disk search and restricts the check to the project's own
data directories. All six requested accessions were searched under

```
${PRIVATE_WORK_ROOT}/CancerLncAtlas/{input,raw,processed,staging}
${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/inputs
${PRIVATE_WORK_ROOT}/CancerLncAtlas/{input,raw}
```

| accession | present in project directories? |
|---|---|
| ENCSR456FVU | **NO** |
| ENCSR369TWP | **NO** |
| ENCSR795JHH | **NO** |
| ENCSR413YAF | **NO** |
| ENCSR870OLK | **NO** |
| ENCSR876DCD | **NO** |

None is present, so per the plan the download branch applies — subject to the
Phase 6 assembly gate below.

---

## 2. Phase 6 prerequisite: the project's genome assembly

The plan states: *"下载前必须先确定当前 CancerLncAtlas lncRNA annotation 的 genome
assembly. 不能假设是 hg19 或 hg38."*

Determined from the frozen annotation authority **and verified at coordinate level**
rather than trusted from a label:

`dim_lncRNA.parquet` (36,016 rows) declares `annotation_release ∈ {GENCODE_v50,
GENCODE_v36_GDC_DR45}` and uses `chr1…chrY` naming.

Coordinate spot-check against known GRCh38 loci:

| gene | observed | GRCh38 reference | verdict |
|---|---|---|---|
| MALAT1 | chr11:65,497,606-65,508,073 | chr11:65,497,688-65,506,516 | **GRCh38 MATCH** |
| NEAT1 | chr11:65,422,774-65,445,540 | chr11:65,422,794-65,445,540 | **GRCh38 MATCH** |
| XIST | chrX:73,820,649-73,852,714 | chrX:73,820,649-73,852,753 | **GRCh38 MATCH** |

Under hg19, MALAT1 would sit at chr11:65,238,240-65,246,883 — roughly **270 kb away**
from what the project actually stores. The project annotation is therefore
**GRCh38**, consistent with the GDC DR45 lineage.

**Assembly gate result: PASS — the project is GRCh38.**

---

## 3. Phase 6 gate: every requested resource is hg19 → FAIL-CLOSED

Queried live from `www.encodeproject.org` through the working SOCKS5 proxy on 149.

### 3.1 ENCSR456FVU (the plan's first-priority eCLIP resource)

Sampled 24 released files directly:

| output_type | file_format | assembly | files |
|---|---|---|---|
| reads | fastq | *(assembly-independent)* | 12 |
| alignments | bam | **hg19** | 6 |
| **peaks** | **bed** | **hg19** | 6 |

Experiment-level `assembly` field: **`['hg19']`**. Description: *"A Large-Scale
Binding and Functional Map of Human RNA Binding Proteins: eCLIP data"*, Yeo lab,
released 2018-09-12.

**There is no GRCh38 peak file for ENCSR456FVU.**

### 3.2 All six requested accessions

| accession | experiment assembly | observed file assemblies |
|---|---|---|
| ENCSR456FVU | `['hg19']` | `['hg19']` |
| ENCSR369TWP | `['hg19']` | `['hg19']` |
| ENCSR795JHH | `['hg19']` | `['hg19']` |
| ENCSR413YAF | `['hg19']` | `['hg19']` |
| ENCSR870OLK | `['hg19']` | `['hg19']` |
| ENCSR876DCD | `['hg19']` | `[]` |

**Every resource named by the plan is hg19. The project is GRCh38.**

### 3.3 Why this stops the phase

The plan's own rules leave no latitude:

* *"如果项目是 GRCh38：优先使用 ENCODE hg38 processed peaks."* — no hg38 peaks exist
  for these accessions.
* *"禁止静默 liftOver."* — lifting hg19 peaks to GRCh38 is the only way to use them,
  and it is forbidden.

Proceeding would require either silently lifting coordinates (forbidden) or
attaching hg19 intervals to GRCh38 lncRNA coordinates (silently wrong: the two
annotations disagree on ~270 kb for MALAT1 alone).

**No file was downloaded. No peak mapping was attempted.** The gate did its job
before any GPU spend.

---

## 4. Deliverable produced

`manifests/ENCODE_DOWNLOAD_MANIFEST.tsv` — the Phase 5 schema, with every row
carrying an explicit `status` and `decision_reason`:

```
resource_type  publication_set_accession  experiment_accession  file_accession
RBP  cell_line  assembly  file_format  output_type  source_url
download_date  sha256  bytes  status  decision_reason
```

Six rows, all `status = NOT_DOWNLOADED`,
`decision_reason = ASSEMBLY_MISMATCH: project annotation is GRCh38; resource
provides ['hg19']; silent liftOver is forbidden`.

A companion `ENCODE_DOWNLOAD_MANIFEST.json` records the full audit (experiment
assembly, sampled files, observed file assemblies, target, biosample, status,
description) for each accession, so the decision is reproducible.

---

## 5. A viable hg38-native path does exist

A live ENCODE search for GRCh38 eCLIP experiments returns:

* **252 experiments**
* **168 distinct RBP targets**

with multiple replicates for common targets (TARDBP, HNRNPK, SRSF1, FXR2, DGCR8,
HNRNPU … three each).

So the plan's *methodology* — hg38 processed reproducible peaks, context-scoped,
never globalised — is achievable; only the *accession list* is unusable. The same
search should be run for the knockdown, secondary-analysis and RBNS resources
before Phase 9/10.

---

## 6. Decision required

Three mutually exclusive paths, none of which should be chosen silently:

**Path A — substitute hg38-native accessions (recommended).**
Re-select eCLIP experiments from the 252 GRCh38 experiments, restricted to RBPs
that intersect the project's lncRNA–RBP partner set, in the cell lines the context
authority already maps (HepG2→LIHC etc.). Keeps the plan's methodology and its
context rules intact; changes only which accessions are cited. The download
manifest would be regenerated against the new list.

**Path B — explicit, audited liftOver.**
The plan forbids *silent* liftOver. A non-silent variant would require: a pinned
UCSC chain file with its own SHA256, per-peak mapping status recorded, unmapped
peaks quarantined rather than dropped, a new authority variant, and the lifting
declared in every downstream lineage record. This is materially more work and
introduces a coordinate-provenance layer the current graph has no place for.

**Path C — abort the ENCODE integration** and deliver Phases 0–4 only, keeping the
RBP repair (the 348,364 dropped RBP rows, the 84 % prediction discovery, and the
116,803 mislabelled predictions) as the substantive result.

Path A preserves the plan's intent with the least new risk and no forbidden
operation. I have not started it because it changes the resource set the plan
names explicitly, and that is your call rather than mine.

---

## 7. Status

| phase | status |
|---|---|
| 0 — real-code baseline audit | COMPLETE |
| 1 — assay taxonomy | COMPLETE |
| 2 — Evidence Transformer | COMPLETE |
| 3 — typed main-graph relations | COMPLETE at binding-table layer |
| 4 — cross-database duplicate audit | COMPLETE |
| **5 — ENCODE resource audit + manifest** | **COMPLETE** |
| **6 — eCLIP peak mapping** | **FAIL-CLOSED (assembly mismatch), documented** |
| 7–15 | blocked on the §6 decision for the ENCODE half; the non-ENCODE half (typed relations → G012 authority variant) is not blocked |
| Paid GPU | NOT STARTED |
| Formal V3.2 artifacts overwritten | NO |
| Origin capsule modified | NO |

### Work that is not blocked

The typed-relation wiring (Phase 7's `static_context_lnc_rbp_eclip` edge role and
the typed relation map in `formal_graph`) depends on the `binds_protein_predicted`
placement decision from the Phase 3 report §8, not on ENCODE. That, plus resolving
the frozen `623,207 rows / 2,567 lncRNAs` invariant, is the natural next step
whichever ENCODE path is chosen.
