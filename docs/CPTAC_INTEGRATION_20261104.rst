CPTAC measured assays and native-score integration (2026-11-04)
==============================================================

This extension adds the ten harmonized CPTAC Pan-Cancer cohorts distributed by
LinkedOmicsKB Release 1. It aligns measured RNA, protein, phosphorylation,
promoter-related methylation and clinical observations to the current gene,
cancer, pathway and State identifiers. It does not train HHGT, restore MOGSA,
or combine uncalibrated assays into one Y. Pathway=1.0 and State=0.5 are unchanged.

Authority and coverage
----------------------

The frozen 106-file download manifest records original public URLs, byte sizes,
ETags and SHA256 hashes. Downloads validate these before materialization.
Published tumor case lists define the patient authority; normal samples remain
separate. A patient is CPTAC:<original cohort>:<case>; tumor and normal samples
append ::T or ::N. Numeric-looking case IDs remain strings. Cases outside the
published authority are retained in raw files and reported as exclusions.

The 1,043 unique tumor cases include 1,037 with RNA, 1,030 with protein,
1,030 with phosphorylation and 716 with gene methylation. Native-to-existing
cancer aliases are CCRCC->KIRC, HNSCC->HNSC, LSCC->LUSC and PDAC->PAAD.
Other cohort names are unchanged. This scope does not include later independent
CPTAC cohorts absent from this harmonized release.

RNA starts as GENCODE v34 log2(TPM+1); stable Ensembl identifiers are mapped to
the current v36 gene authority without replacing unrecognized genes by guesses.
Symbol-level RNA inputs sum duplicate symbols in linear TPM. Small invalid
negative source values and nonfinite assay values become NA with explicit
audit records. RNA methods requiring complete finite matrices omit affected
genes and record them; there is no zero fill or 95% gene-coverage rule.

Scoring contract
----------------

* RNA Pathway: current 2,547 MSigDB IDs, GSVA ssGSEA alpha=0.25,
  normalize=TRUE, at least 10 observed genes, at most 500 genes in a set.
  The input scale is CPTAC log2(TPM+1), not TCGA logCPM.
* Protein Pathway: the same 2,547 IDs, NA-aware ssGSEA raw ES
  (normalize=FALSE), alpha=0.25 and at least 10 measured genes per patient.
  Protein log2 reference-intensity abundance is not an activation measurement.
* Methylation Pathway: measured-gene mean beta, at least 10 observed genes,
  same pathway IDs and maximum set size. This measures pathway-associated
  promoter methylation, not signed pathway activity.
* Phosphorylation: original site IDs, protein isoforms and 15-aa flanking
  sequences are preserved. PTM-SEA uses the 624 PATH/KINASE signatures from
  PTMsigDB 2.0.0. A signature without sufficient observed support is NA.
  These signatures have their own identifiers and are not renamed as MSigDB
  pathways. The site/gene bridge makes the relationship explicit.
* Each phosphosite is centered on its own frozen CPTAC tumor-cohort median.
  Only identical sequences within the same gene are collapsed for the PTM
  input; sequences mapping to multiple genes are excluded and reported.
  Scores are not normalized by total protein and are not causal activation.
* PTM-SEA uses pinned Broad ssGSEA2.0 commit
  cae7bed2c357826c0d887ca528c0c9b272705006, weight=1, rank correlation,
  area-under-RES ES, minimum overlap=5, and no permutations. A narrowly scoped
  compatibility copy returns NA for an undefined weighted walk and skips the
  original ES branch's unused single permutation. The upstream checkout is
  untouched. The wrapper suppresses printing the returned random-walk arrays.
  This run recovered already written outputs from that upstream printing step;
  per-cohort POST_WRITE_RECOVERY receipts record full-file validation before
  terminating only the completed computation's verbose output. A numerical test compares ordinary ES with the original and
  confirms that zero weights do not abort other samples. Placeholder
  p-value/FDR files emitted by the upstream writer are not inferential results
  and are excluded from the integrated score matrices.
* State uses the existing 485-column catalog. RNA-derived original recipes,
  current weighted PROGENY/DoRothEA resource scores, xCell, MCP-counter,
  ESTIMATE, IPS, CYT, hallmark, signifinder, SenMayo and EXTEND are recomputed.
  The original Malta OCLR weights yield Spearman rho plus explicitly named
  CPTAC-reference minmax variants. Raw variants are retained.
* Published CPTAC ABSOLUTE purity/ploidy and maftools TMB retain source labels.
  Relative CIBERSORT composition is derived from all 22 published absolute
  components by dividing by their complete, finite, nonnegative positive sum.
  Original absolute outputs remain available. This is relative composition
  within these 22 components, not a fraction of all tissue cells.
* DNA stemness requires original CpG probes. Gene methylation cannot replace
  them. Missing HRD, aneuploidy, continuous MSI, the original TLS12 PCA score
  and incompatible SBS endpoints remain NA with target-specific reasons.

Gene methylation definition is verified against the original methods:
mean beta of CpG-island probes upstream within 1 kb of the TSS or in the 5' UTR
for coding genes; upstream only for noncoding genes, using the stated EPIC
annotation and GENCODE v34. This is promoter-related, but not automatically
identical to the existing TCGA TSS1500/TSS200/first-exon aggregation.
The verification receipt supersedes the provisional annotation in the raw
materialization receipt. No unavailable probe-level scores are manufactured.

Clinical files preserve published metadata, phenotype and survival tables.
A survival endpoint is available only when both a nonnegative follow-up time
and a valid event indicator exist. An event=0 without a time is not a valid
censored observation. Field-specific coverage is reported. The optional GDC
API was unavailable during this run; published cohort clinical data were used.

Execution and readers
---------------------

All workloads run on server 149. The source specification pins installed
references, R library paths and the clean Broad checkout. Install dependencies
in an isolated environment before launching; no tool starts paid compute.
The observed R package versions are recorded in
requirements/cptac_R_versions.json; Python additionally requires openpyxl.

::

  python -m cla_engine.cptac_pipeline \
    --source-spec config/cptac_149_sources_20261104.json \
    --output /NEW/VERSIONED/CPTAC_RUN --cancer-type all --workers 3

A small demo selects published CPTAC cohort names or existing cancer aliases:

::

  python -m cla_engine.cptac_pipeline \
    --source-spec config/cptac_149_sources_20261104.json \
    --output /NEW/VERSIONED/CPTAC_DEMO --cancer-type list \
    --cancer-list BRCA KIRC --workers 2

A nonempty output directory is rejected. Remove the optional
pre_downloaded_complete field from a COPY of the source spec to download the
hash-pinned public files again. Keep existing immutable products.

The cptac_input binding is verified by cla.py preflight and the prepare input
contract. The existing replacement-required Pathway policy still blocks
training until a separate target design is implemented. Readers require an
explicit cohort and assay; they do not silently append patients to TCGA:

::

  from cla_engine.common import read_json
  from cla_engine.cptac_binding import load_scores, load_assay
  binding = read_json("BOUND_149_CONFIG.json")["cptac_input"]
  scores = load_scores(binding, cohort="KIRC", layer="RNA_PATHWAY")
  state = load_scores(binding, cohort="KIRC", layer="STATE")
  rna = load_assay(binding, cohort="KIRC", layer="rna", mapped_genes_only=True)

Other score layers are PROTEIN_PATHWAY, METHYLATION_PATHWAY and PHOSPHO_PTMSEA;
other measured layers are protein, methylation_gene and phosphorylation.
Each reader returns values and an explicit observed mask. All score matrices
within a cohort share the complete canonical tumor-patient axis; an unmeasured
assay profile remains an all-NA column. Measured-assay readers retain the
original measured sample subset and explicit tumor/normal sample IDs. Gene-level collapsing
of phosphorylation is rejected. Normal assays require tissue="Normal".

Principal products
-------------------

COMPLETE.json and BOUND_SOURCES.json bind products and source hashes.
PATIENTS.tsv, CPTAC_SAMPLE_ASSAY_MAP.tsv and FEATURE_ALIGNMENT.parquet retain
identity and gene/site mapping. Clinical and State tables use the same patient
axis. CLINICAL_SOURCE.tsv.gz also preserves all 64 original metadata and
survival columns, alongside normalized clinical fields. SCORE_INDEX.tsv and SCORE_COVERAGE.tsv.gz describe per-cohort matrices.
STATE_ALIGNMENT.tsv records each of the 485 targets, its method and missing
reason. STATE_GENE_PROTEIN_BRIDGE.parquet preserves the complete existing
mapping including source weights and directions. PATHWAY_GENE_PROTEIN_BRIDGE
and PTM_SIGNATURE_SITE_GENE_BRIDGE retain gene/protein and exact-site links.
MEASURED_FEATURE_COVERAGE.parquet reports actual assay support.

RNA_PROTEIN_PATHWAY_CONCORDANCE.tsv.gz uses matched patients, reports Spearman
rho and within-cohort BH FDR; it is cross-assay concordance, not activation
ground truth. Methods are cohort-native and not calibrated to TCGA. Future
inductive modeling must fit and freeze cohort-derived normalization on training
or reference samples before held-out application. Native scores from separate
assays must not be pooled as a single calibrated target.

Sources
-------

* https://kb.linkedomics.org/download
* https://pmc.ncbi.nlm.nih.gov/articles/PMC10530292/
  (LinkedOmicsKB original processing methods, Cell Systems 2023)
* https://github.com/broadinstitute/ssGSEA2.0
* https://doi.org/10.1016/j.cell.2018.03.034
  (Malta et al. original stemness model)

Verified run results
--------------------

The final product retains all 485 State columns; 413 have observed
values and 72 remain unavailable. The coverage workbook reports every target
and cohort separately. All four score layers use the full cohort patient axis.

* METHYLATION_PATHWAY: 1607 of 2547 IDs have an observed
  score in at least one cohort; 1116753 finite patient-score entries.
* PHOSPHO_PTMSEA: 444 of 624 IDs have an observed
  score in at least one cohort; 317984 finite patient-score entries.
* PROTEIN_PATHWAY: 1697 of 2547 IDs have an observed
  score in at least one cohort; 1488462 finite patient-score entries.
* RNA_PATHWAY: 1901 of 2547 IDs have an observed
  score in at least one cohort; 1971337 finite patient-score entries.

Validation scope
----------------

The repository-wide pytest collection was attempted and reported 27 collection
errors from unavailable optional h5py/fastapi dependencies and historical
script modules absent from this checkout. This is not a full-repository pass.
The supported regression selection from the previous release is rerun with
tests/test_cptac.py added; real 149 input preflight, all 87 cohort/assay/score
reader checks, 217 measured methylation spot checks, and the original-versus-
patched PTM numerical comparison are separate validation gates.

The final current-mainline/CPTAC selection passed all 154 tests. The real
149 preflight returned INPUT_CHECK_PASS_NOT_TRAINED and validated the new
CPTAC binding alongside unchanged TCGA inputs. No full HHGT training ran.
