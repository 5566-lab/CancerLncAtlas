# The reproducible-peak and exon correction

**Date**: 2026-09-22
**Scope**: the ENCODE eCLIP layer, the interaction-context gate, and the generation
that follows from both.

---

## 1. What was wrong with the first ENCODE attempt

The first attempt downloaded 1,431 eCLIP narrowPeak files and built 428,949
lncRNA–RBP edges from them. Three defects made that layer unusable, and only one of
them was visible at the time.

### 1.1 The peak files were the wrong ones

Each eCLIP experiment ships **three** peak calls per assembly:

| file class | files | peaks | median peaks/file |
|---|---|---|---|
| replicate 1 | 477 | 63,728,160 | 103,577 |
| replicate 2 | 477 | 60,432,268 | 105,856 |
| **combined (reproducible)** | **477** | **1,784,367** | **1,747** |

The combined call requires both replicates and is **1.5 %** the size of either
single-replicate call (252 of 252 experiments). The first attempt used all three
classes: **125.9 M peaks, 98.6 % of them single-replicate calls.**

`preferred_default` is **not** the reproducible flag. It picks one of the two
single-replicate files — 240 times replicate 1, 210 times replicate 2, and only 27
times the combined call. Building the layer on `preferred_default` would have been
wrong in a different way.

### 1.2 The interval was the gene body, not the exons

| interval | bp | share of the covered genome |
|---|---|---|
| gene body (sum of loci) | 449,981,173 | 14.57 % |
| gene body (union) | 413,985,126 | 13.41 % |
| **exon union** | **32,868,991** | **1.064 %** |

The gene body is 13.7× larger than the exon union. Using it diluted any signal and
also made the test strand-blind.

### 1.3 One experiment was counted up to three times

Every experiment contributes a GRCh38 and an hg19 copy of the same measurement, so
counting files double-counts experiments, and counting all three peak classes
triples it again.

---

## 2. The enrichment test, before and after

A falsifiable prediction was recorded before the correction was run:

> switching from gene body to exon should flip the depletion into an enrichment;
> if it does not, the depletion was not an artefact of the interval.

| build | interval | peaks | observed in locus | null | result |
|---|---|---|---|---|---|
| first attempt | gene body | 125.9 M (all classes) | 9.55 % / 9.86 % | 13.41 % | **0.71× / 0.74× — depletion** |
| corrected | exon union | 1.78 M (reproducible) | 3.29 % / 3.27 % | 1.064 % | **3.09× / 3.08× — enrichment** |

**The prediction held.** The depletion was an artefact of the interval and the peak
class, not a property of eCLIP.

---

## 3. The corrected layer

```
reproducible peak set   477 files (252 GRCh38 + 225 hg19), 1.78 M peaks
                        251 experiments, 168 RBPs, 3 biosamples
liftOver                845,230 peaks -> 844,849 mapped (99.95%)
                        381 quarantined, 0 dropped
evidence rows           25,784          (first attempt: 1,745,070)
global edges            13,007
context edges            6,785          HepG2 -> LIHC only
lncRNAs                  2,309
RBPs                       165          3 symbols unmapped: AARS, GARS, TROVE2
```

The exon authority was built from the GENCODE releases the project already names:

```
source              GENCODE v50 + GENCODE v36 (GDC DR45), both already on the server
external annotation none introduced
exon records        346,913
gene agreement      8,541 / 8,541 chromosome, 8,541 / 8,541 strand,
                    8,541 / 8,541 gene start, 8,541 / 8,541 gene end
exon union          56,465 intervals, 32,868,991 bp
nodes without exons 0
```

---

## 4. The interaction-context gate

`cc_hhgt/interaction_context.py` states its rule in code:

> Map contextual interaction rows or exclude them; never globalize them.

A row carrying `disease_raw`, `tissue` or `cell_line` is either mapped to exactly one
of the 33 cancers or **removed**. `CELL_LINE_CANCER_HINTS` covers 29 cell lines and
12 cancers; K562 is deliberately absent because it is CML, which is not one of the 33.

### 4.1 What the rule removes

Measured on the protein layer — the part that feeds the binding graph, not the raw
table:

```
protein-partner rows entering the gate   813,653
kept                                     762,635
removed                                   51,018     (matches the library's own audit)
```

Across the whole interaction table the removal is 316,546 rows, of which **98.1 % is
NPInter**. RNAInter, 89 % of all rows, loses none.

### 4.2 Why the rule misfires

The `tissue` column of the removed rows contains:

```
HEK293                     93,362
K562                       56,831
HEK293T                    46,687
human brain                27,926
Flp-In 293                  9,098
DMEM                        1,058     <- a cell-culture medium
iPSC-derived motoneurons    1,103
```

**DMEM is a medium.** NPInter wrote material descriptions into both `tissue` and
`cell_line`, so almost no row is removed by `cell_line` alone:

```
cell_line only              17
tissue only                  3
cell_line AND tissue   310,576
```

Removing `cell_line` from the context axes therefore recovers **17 rows**. The rule
is in practice a `tissue` rule, and the field it fires on is not a disease claim.

### 4.3 The ruling

> A protein–lncRNA interaction is not cancer-type specific. A global edge being
> reachable from all 33 cancers is not a problem; the requirement is only that the
> interaction exists in some cell line.

Implemented as **ruling B**: the rows the gate removes become global instead of being
dropped, and the context edges are left exactly as they are.

```
branch                        global binding rows
current chain (reproduces the frozen 623,207)      623,207
after ruling B                                     650,571
delta                                              +27,364
```

The chain was validated by reproducing the frozen 623,207 exactly before any change
was reported. Every new pair lies inside the `validated` superset that
`protein_layer.py` already checks against, so its own invariant still holds:

```
new (lncRNA, protein) pairs              27,364
of which outside the validated superset       0
context rows before / after              30,844 / 30,844  (identical)
```

---

## 5. The one generation that follows

`outputs/phase3_typed_binding_v2/lnc_protein_binding_typed_v2.parquet`

```
frozen typed rows          918,266
+ ruling B rows             35,955
+ ENCODE v2 rows            19,792
= v2 rows                  974,013

binds_protein_predicted              773,283
binds_protein_experimental_unspecified 74,643   (+35,955)
binds_protein_eclip                    66,421   (+19,792)
binds_protein_other_clip               56,245
binds_protein_rna_capture               3,174
binds_protein_rip                         209
binds_protein_other_physical               23
binds_protein_unknown                      15
```

Materialised once, into `outputs/phase7_authority_v2`, folds 0–4. The static
artifacts `detection`, `signed_membership`, `pathway_hierarchy`,
`protein_gene_encoding` and `ppi` are **bit-identical to the frozen generation**;
only the binding table differs.

**No frozen artifact was modified and no frozen function was changed.** Ruling B is
implemented in the new-generation script against the same declarations the library
uses, so `apply_strict_cancer_context` keeps its published contract.

---

## 6. What is still open

* **The `predicted` relation.** 649,084 edges after the candidate filter, 86.78 % of
  the binding layer. Still excluded by the conservative default. Coverage is
  asymmetric: +7,469 proteins but only +136 lncRNAs (+1.6 %). The architecture has
  per-relation attention but no readable per-relation gate, so "let the model decide"
  cannot currently be measured.
* **The gene/mRNA context path.** `graph_build.py` applies the same gate to gene
  partners. The ruling was about interactions; whether it extends to the
  lncRNA→gene relation is not settled.
* **C vs B vs C-shuffle.** The shuffle arms are designed (shuffle_A permutes RBP
  labels, shuffle_B permutes peak positions) but require GPU to evaluate.
