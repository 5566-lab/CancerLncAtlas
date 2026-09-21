# RBP_ENCODE_INPUT_AUDIT.md

**Deliverable**: Phase 15 item 4 (input audit)
**Date**: 2026-09-21
**Server**: `149`
**GitHub `main` at task start**: `ab982cf43b34900f63309b1eca250e21c6384eea`

---

## 1. Evidence sources actually present

| source | rows | PMID coverage |
|---|---|---|
| RNAInter | 5,510,363 | **0.0 %** |
| NPInter | 635,758 | 100.0 % |
| LncTarD | 8,343 | 100.0 % |
| LncACTdb | 3,955 | 100.0 % |
| LncRNA2Target | 2,288 | 100.0 % |
| **total** | **6,160,707** | **7.8 %** |

RNAInter contributes 89.4 % of all interaction rows and **carries no PMID column in its source table at all** (`RNAInterID, Interactor1.Symbol, Category1, …, score, strong, weak, predict`). Nothing downstream can recover a publication anchor for it.

## 2. What the pipeline did with those inputs

| stage | in | out | note |
|---|---|---|---|
| protein-layer partner filter | 6,160,707 | 813,653 | `partner_type.str.contains("protein")` — **all 348,364 `rbp` rows dropped** |
| strict cancer context | 813,653 | 762,635 | 51,018 context rows excluded (61.6 %); lost-to-global = 0 |
| protein mapping | — | — | 8,940 unmappable in both modes |
| binding table | — | 755,346 | single flat `binds_protein` relation |

## 3. Assay information that was available and discarded

NPInter5's `methods` column is **non-empty for 100 % of its 635,758 rows** — 500 distinct assay strings. The historical pipeline collapsed them into one token:

| legacy family | rows |
|---|---|
| `physical_binding` | 626,756 |
| `other_experimental` | 8,908 |
| `reporter_assay` | 92 |
| `expression_or_abundance` | 2 |

Of the 626,756 rows called `physical_binding`, **509,928 now receive a finer subtype**.

### The most serious input-side defect

```
"Conserved miRNAs target sites predicted by TargetScan and miRanda
 overlap with the AGO CLIP dataset"                              116,499 rows
```

This is a **computational target-site prediction** that merely references the AGO CLIP dataset. The historical regex matched the substring `clip` and recorded every row as `physical_binding`.

Including its variants, **773,283 rows are sequence-based predictions** that were presented as measured binding.

## 4. Assay classification after the repair

| graph_assay_class | rows |
|---|---|
| `predicted` | 773,283 |
| `other_clip` | 56,245 |
| `eclip` | 46,629 |
| `experimental_unspecified` | 38,688 |
| `rna_capture` | 3,174 |
| `rip` | 209 |
| `other_physical` | 23 |
| `unknown` | 15 |

Uninformative rows (`unknown` + `experimental_unspecified`): **38,703 (4.21 %)**, so **95.79 % receive a concrete assay class**.

## 5. Cross-database duplication

* canonical experiments (PMID-anchored): **96,093** from 480,781 rows
* seen in more than one database: **14**
* rows without a PMID, never merged: **5,679,926**

**The RBP path cannot be de-duplicated across databases at all**: every RBP row comes from RNAInter (0 % PMIDs), while NPInter (100 % PMIDs) never labels partners `rbp`. Any claim that RBP evidence has been cross-database de-duplicated would be false.

## 6. Effect on the main graph

| | legacy | typed, physical only |
|---|---|---|
| global binding edges | 724,502 | **102,425** |
| rows that are predictions | — | **622,077** (85.9 % of legacy) |
| relation types | 1 | 7 |
| context leakage | 0 | **0** |

## 7. ENCODE inputs

| accession | resource | assembly | status |
|---|---|---|---|
| `ENCSR456FVU` | eCLIP reproducible peaks | `['hg19']` | **NOT_DOWNLOADED** |
| `ENCSR369TWP` | RBP knockdown RNA-seq (HepG2) | `['hg19']` | **NOT_DOWNLOADED** |
| `ENCSR795JHH` | RBP knockdown RNA-seq (K562) | `['hg19']` | **NOT_DOWNLOADED** |
| `ENCSR413YAF` | processed DESeq/rMATS/MISO/Cuffdiff | `['hg19']` | **NOT_DOWNLOADED** |
| `ENCSR870OLK` | batch-corrected expression/splicing | `['hg19']` | **NOT_DOWNLOADED** |
| `ENCSR876DCD` | RBNS sequence preference | `['hg19']` | **NOT_DOWNLOADED** |

All six are hg19; the project annotation is GRCh38 (verified at coordinate level: MALAT1, NEAT1 and XIST all match GRCh38, while hg19 would place MALAT1 roughly 270 kb away). **No file was downloaded and no peak mapping was attempted.**

A live ENCODE search finds **252 GRCh38 eCLIP experiments across 168 RBP targets**, so an hg38-native path exists — just not through the accessions the plan names.

## 8. Inputs deliberately NOT touched

* the frozen G012 binding artifact and its `623,207 rows / 2,567 lncRNAs` invariant;
* the existing LASSO baseline (reused by digest, never re-materialised);
* every formal V3.2 artifact — all new output goes to a new work root.
