State score additions and protein mapping (2026-11-04)
=======================================================

Current State has 485 targets: 484 primary and auxiliary ESTIMATEScore.
The active registry is CLA_STATE_AUGMENTED_20261104_V3. Existing 483 definitions,
patient values and missing masks are preserved. Pathway=1.0 and State=0.5 remain
joint losses on the live encoder. MOGSA remains retired; this change does not
select a replacement Pathway target or launch real-patient training.

What was actually missing
-------------------------

The earlier checklist was stale. CIN70, SenMayo, Ayers IFN-gamma, Ayers
T-cell-inflamed GEP, Wang APM, Kang DNA repair, glycolysis, lipid metabolism,
OXPHOS, LRRC15-CAF and ECM-up/down were already registered and cover all 10,432
frozen RNA patients. Cabrita TLS was also present, but its nine-gene signature
is different from the requested 12-chemokine signature.

Two previously downloaded/computed columns existed in the historical source
matrix but had not entered State. They are now appended without rescaling:

* TLS12_Chemokine_Published: the published PanImmune/Wolf Chemokine12_score,
  9,060/10,432 patients (86.85%). Original published values are used. This is
  not a new PCA fit, singscore calculation, or histological TLS observation.
* DNARepair_Hallmark: existing HALLMARK_DNA_REPAIR GSVA ssGSEA,
  10,432/10,432 patients (100%). The 150-gene definition matches the exact
  historical expression-network cache. It remains distinct from Kang DNArep
  and genomic HRD, and does not measure DNA-repair efficiency.

The TLS signature is CCL2, CCL3, CCL4, CCL5, CCL8, CCL18, CCL19, CCL21, CXCL9,
CXCL10, CXCL11 and CXCL13. All twelve map to verified UniProt accessions.
The existing published GDC score archive and gene-set workbook were checked
against their original download SHA256 receipts. Failed fresh downloads were
not used as inputs. New downloads recovered the original LM22 publication
workbook, the MCP-counter reference source and the original EXTEND package.

These are native-scale observational RNA scores, not absolute or independently
validated pathway activation. Missing published TLS values stay NA. No 95%
gene-coverage rule, cross-patient imputation, score mixing, or new genomic
Pathway-label input was introduced.

Complete mapping table
----------------------

STATE_PROTEIN_MAP.tsv.gz accounts for all 485 State identities in 336,474 rows.
411 States have verified protein mappings. The remaining 74 consist of four
methylation-probe scores and 70 measurements without a fixed gene/protein
signature, including 60 SBS exposures, TMB, HRD components, MSI, aneuploidy,
ABSOLUTE purity/ploidy and methylation-derived leukocyte fraction.

Each row records State ID, family, original member name, Ensembl gene,
reference biotype, HGNC-curated UniProt accessions, existing graph protein IDs,
original weight where defined, signature subgroup, mapping evidence/status
and annotated RPPA500 targets. The table also marks which RPPA500 features
actually have measurements; an antibody match does not imply measured values
for every patient or a gene-specific measurement for a multi-gene antibody.

Mapping rules preserve meaningful distinctions:

* Expression signatures retain actual component membership and signed source
  coefficients. T-cell GEP housekeeping references are separate from predictors.
* PROGENy rows preserve the full resource used by the existing weighted scorer.
  They are response-footprint features, not physical pathway membership.
* DoRothEA rows are TF target priors with the existing mean-MOR aggregation.
* CIBERSORT rows retain the original 547-by-22 reference expression matrix and
  distinguish cell-specific DEG indicators from other joint-reference entries.
  Reference expression is not a regression coefficient or a regulatory weight.
* xCell retains all 489 original subsignatures and their cell-type identities.
  Calibration and spillover correction mean a member list alone is not a full
  reconstruction of the final cell score.
* MCP-counter, ESTIMATE, IPS, CYT, SenMayo and the original stemness signatures
  retain their source-specific roles. Derived purity components are not
  interpreted as a direct purity-protein interaction.
* EXTEND's source-code tokens include aliases and noncoding markers. They are
  not asserted to be sixteen independent genes or proteins. Duplicate aliases,
  noncoding genes, ambiguous symbols and unresolved members remain explicit.
* The four methylation signatures preserve all 450 original probe/weight
  entries. No direct protein is fabricated from a CpG name. Genomic aggregate
  scores do not receive a substituted list of biologically related genes.

Exact reference symbols and unambiguous HGNC previous/alias symbols are used.
LOC identifiers may resolve through the exact HGNC Entrez identifier. No fuzzy
name matching is used. GENCODE v36 presence is distinguished from HGNC-only loci;
the latter do not acquire invented model nodes. Existing graph identifiers and
authoritative UniProt links are separate columns.

The aggregate map is reference material, not automatically inserted model
edges. Current preflight validates its State-catalog identity and file/source
hashes, and compatible future preparation records the binding in lineage.

Reproduction and artifacts
--------------------------

All preparation, audits and tests run on host 149. Install the optional
requirements/state_signatures.txt dependencies in an isolated environment.
Export exact installed package definitions with
scripts/export_state_signature_definitions.R, and extract the original LM22
workbook with scripts/extract_lm22_reference_149.py. Bind verified source paths
in config/state_protein_map_sources_20261104.json without changing hashes.

::

  python -m cla_engine.state_augment \
    --config VERIFIED_PRIOR_483_STATE_CONFIG.json \
    --historical TCGA_sample_score_matrix.tsv.gz \
    --output NEW_STATE_DIRECTORY --cancer-type all

  python -m cla_engine.state_protein_map \
    --source-spec BOUND_STATE_MAP_SOURCES.json --output NEW_MAPPING_DIRECTORY

A State demo uses --cancer-type list --cancer-list BRCA LUAD.
The public repository contains code, source bindings, definitions and aggregate
verification; patient score matrices stay on 149. The delivered Excel workbook
contains full State/protein mappings, coverage and unresolved-member summaries;
the compressed TSV is the complete machine-readable table.

Active products are in
/dell_2/DSC/CancerLncAtlas/inputs/state_augmented_20261104_r1/ and
/dell_2/DSC/CancerLncAtlas/inputs/state_protein_map_20261104_r1/.
STATE_CURRENT.json records the bound current configuration and code commit.

Primary sources
---------------

* Published PanImmune scores: https://api.gdc.cancer.gov/data/80a82092-161d-4615-9d96-e858f113618d
* PanImmune gene definitions: https://api.gdc.cancer.gov/data/9b174979-fe97-48bc-9e97-9384b0519f03
* Original 12-chemokine study: https://www.nature.com/articles/srep00765
* Hallmark DNA repair: https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_DNA_REPAIR
* CIBERSORT/LM22 publication: https://www.nature.com/articles/nmeth.3337
* MCP-counter: https://github.com/ebecht/MCPcounter
* xCell: https://github.com/dviraran/xCell
* EXTEND original release: https://github.com/NNoureen/EXTEND/tree/v.1.0.0
* HGNC reference: https://www.genenames.org/download/
