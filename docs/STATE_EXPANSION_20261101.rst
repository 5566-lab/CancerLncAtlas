Expanded patient State scores (2026-11-01)
==========================================

Current target status (2026-11-04): MOGSA and its derived lavaan parent
targets are retired. The public configuration requires a replacement design;
preparation and training do not automatically fall back to RNA activity_score.
See PATHWAY_TARGET_POLICY_20261104.rst. State remains
CLA_STATE_ROUTED_PROGRAMS_20261101_V2 with 482 primary targets plus ESTIMATEScore:
the 198-dimension base and 285 existing PROGENY/DoRothEA programs are unchanged.


The base expansion uses CLA_STATE_EXPANDED_20261101_V1: 198 measured or
estimated State dimensions, including 197 primary training targets and one
auxiliary ESTIMATEScore. Pathway=1.0 and State=0.5 optimize the same live HHGT
encoder. Expanding the State catalog does not change the task into patient-score
regression: Y remains a fold-separated cancer/lncRNA/State association label.
Missing, constant or otherwise unestimable targets remain in the catalog with
explicit availability masks; they are never manufactured as negative labels.

The authoritative, ordered definitions are config/state_catalog_20261101.json.
Each target records its original column, method, source, measurement layer,
family, unit, interpretation and training role. Changing those definitions
requires a reviewed catalog revision and new preparation/checkpoints. Old
15-target bound configurations retain their recorded roles; the public default
and the expanded preparation use the new catalog. The compact synthetic fixture
remains small by default; the expanded integration test explicitly exercises all
198 dimensions, all 197 primary identities and shared-encoder gradients.

Measurement inventory
---------------------

* Seven published stemness/telomerase dimensions: EXTEND, RNAss, EREG.EXPss,
  DNAss, DMPss, ENHss and EREG-METHss. EXTEND estimates telomerase activity,
  not telomere length.
* Eighteen newly computed RNA dimensions: Buffa hypoxia; the native epithelial
  and mesenchymal Miow EMT components; Lundberg cell cycle; Yang mitotic index;
  Carter CIN70; Kang DNA repair; Zhang glycolysis; Zheng lipid metabolism;
  Hu VEGF; Ayers IFN-gamma and T-cell-inflamed GEP; Wang APM; Cabrita TLS;
  Dominguez LRRC15-CAF; native Chakravarthy ECM-up/ECM-down; and SenMayo ssGSEA.
  No subtraction of native EMT or ECM components is invented.
* OXPHOS reuses the existing Hallmark ssGSEA bulk score. The inspected
  signifinder Barkley method expects malignant-cell labels in single-cell data;
  it is not relabelled as a bulk method. This method substitution is explicit.
* Historical CYT, ESTIMATE ImmuneScore/StromalScore/TumorPurity and the published
  leukocyte fraction are reused. IPS reuses the historical hacksig 0.2.0 native
  immunophenoscore, not a newly substituted RNA mean or a cohort z score.
* CIBERSORT contributes 22 relative cell fractions, xCell 64 cell enrichment
  scores, and MCP-counter 10 cell abundance scores. These 96 dimensions are in
  cell_composition, separate from immune function. xCell and MCP-counter values
  are not percentages. CIBERSORT P.value, Correlation and RMSE, and xCell's
  three aggregate microenvironment scores, are not extra biological targets.
* Genomic measurements reuse published non-silent mutations/Mb (TMB), HRD
  total and its original hrd-loh/ai1/lst1 columns, MSIsensor2 score, ABSOLUTE
  purity/ploidy and the altered-arm aneuploidy score. The HRD total is verified
  against the sum of its three components. These are published results, not
  newly claimed maftools/scarHRD/ASCAT runs. TIDE's RNA MSI model is not used.
* Sixty COSMIC v3.2 GRCh37 non-artifact SBS signatures are estimated from MC3
  PASS SNVs. SBS2 and SBS13 are already included; a duplicate APOBEC aggregate
  is not added. The primary value is relative mutation contribution. Native
  fitted counts and reconstruction QC remain separate output artifacts.

CIN70 and DNA-repair signatures are expression proxies, not direct CIN or repair
efficiency measurements. SenMayo is senescence/SASP-related transcription, not
a senescent-cell fraction. Metabolic signatures do not measure metabolic flux;
TLS RNA scores do not establish histological TLS. No score is interpreted as
proof of pathway activation, suppression or causal lncRNA regulation.

Patient identity and source selection
------------------------------------

All CPU work runs on host 149. The scoring wrappers record that host before
processing. RNA uses the full measured gene-symbol TPM universe, not only the
signature union. Original signifinder transformations and output columns are
retained; Miow EMT, Wang APM and Chakravarthy ECM receive log2(TPM+1), while the
other functions receive unlogged TPM and perform their documented transforms.
SenMayo uses the original 124-gene SAUL_SEN_MAYO set with normalized GSVA ssGSEA.
Native cohort-normalized RNA methods are explicitly cohort-dependent labels;
they do not provide independent functional validation of RNA associations.

Only TCGA tumor sample types 01-09 enter new RNA scoring. The upstream adapter
restores dots to hyphens in full RDS barcodes and parses validated TCGA barcode
components. Generic prepared joins still require exact patient/sample keys.
The materializer excludes known normal samples, checks cancer/patient conflicts
against the frozen patient authority, and selects a measured record separately
for every target: exact authority aliquot, parsed authority sample type, then
the documented primary-tumor preference, with explicit patient-level records
as fallback and lexical tie-breaking. Different tumor values are audited.
They are not averaged, preserving discrete published IPS and genomic values.
Missing measurements remain NA; no cross-patient imputation or score rescaling
is introduced. This replaces the old normal-plus-tumor patient averaging for
expanded production State inputs.

SBS96 construction checks the MC3 GRCh37 reference allele against the centered
context, orients contexts to pyrimidines and deduplicates participant/chromosome/
position/ref/alt. Counts aggregate observed tumor SNVs per participant; they do
not imply sequencing callability for patients absent from MC3. A separate
BSgenome hg19 check validates 1,000 contexts. MutationalPatterns 3.7.1 refits the
frozen COSMIC v3.2 non-artifact reference using NNLS. At least 50 SNVs and a
reconstruction cosine of 0.8 are required for State availability. This is a
declared quality filter, not a guarantee of signature identifiability or a
bootstrap confidence interval. Unsequenced and below-threshold patients remain
missing across all SBS columns, rather than receiving zero APOBEC exposure.

Preparation and training contracts
----------------------------------

The materializer writes STATE_EXPANDED_PATIENT.tsv.gz (one row per frozen
patient), STATE_COVERAGE.tsv, STATE_VALUE_SOURCES.tsv.gz,
STATE_EXCLUDED_ROWS.tsv, STATE_MULTIPLE_TUMOR_VALUES.tsv.gz,
STATE_MATERIALIZATION.json and STATE_SOURCE_BINDING.json. Source hashes,
catalog definitions, patient selection and matrix identity are sealed together.
Bind the contents of STATE_SOURCE_BINDING.json as config.sources.state.
Preflight and prepared validation reject mismatched catalogs, duplicate
patients, source changes and incorrect cancer identities. The preparation
copies the materialization receipt into the sealed capsule.

All 197 primary IDs enter SPEC.states and the trainable State embedding table.
Purity and ploidy now have primary roles as requested. ESTIMATEScore alone
remains auxiliary because its components already participate in State training.
State families distinguish transcriptional function, cell composition and
genomic measurements. The existing family-balanced State loss/metrics are
retained, preventing the 60 SBS or 96 composition dimensions from becoming
60 or 96 independently weighted task families.

The target-specific adjustment contract covers every new target. For each
CIBERSORT macrophage M0/M1/M2 target, its identical source covariate is excluded;
other composition measures stay requested. Source-dependent ESTIMATE/ABSOLUTE
purity self/derivative exclusions continue to apply. No required clinical
covariate admission threshold is relaxed for the expanded catalog.

Commands on 149
---------------

The pinned package commits/archive hashes are stored in the catalog. Install
them into an isolated R library with their dependencies. SenMayo is downloaded
from the original MSigDB entry, with its resource checksum retained.

::

  Rscript scripts/score_expanded_rna_states.R \
    --input-root RNA_RDS_DIRECTORY --output-root RNA_OUTPUT \
    --senmayo-json SAUL_SEN_MAYO.json --r-library R_LIBRARY \
    --cancer-type list --cancer-list BRCA,LUAD

  python scripts/build_state_sbs96.py \
    --maf mc3.v0.2.8.PUBLIC.maf.gz --patient-map SAMPLE_PATIENT_FOLD_MAP.tsv \
    --output-root SBS_OUTPUT --cancer-type all

  Rscript scripts/refit_state_sbs.R SBS_OUTPUT R_LIBRARY

  python -m cla_engine.state_materialize \
    --historical TCGA_sample_score_matrix.tsv.gz \
    --rna-root RNA_OUTPUT/rna_scores --sbs-root SBS_OUTPUT \
    --patient-map SAMPLE_PATIENT_FOLD_MAP.tsv \
    --output STATE_OUTPUT --cancer-type all

For a materialization demo, replace all with list and add
--cancer-list BRCA LUAD. Use a new output directory. Training cancer selection
is independently recorded during cla.py prepare using the same all/list
contract. A demo cannot be claimed as a completed pan-cancer training run.

Primary method references
-------------------------

* signifinder: https://github.com/CaluraLab/signifinder
* SenMayo: https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/SAUL_SEN_MAYO
* MutationalPatterns: https://github.com/UMCUGenetics/MutationalPatterns
* Published TCGA DNA-damage methods (including HRD components and aneuploidy):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5982584/
* Original HRD table: https://api.gdc.cancer.gov/data/66dd07d7-6366-4774-83c3-5ad1e22b177e

No raw patient matrices are committed to the public repository. The source
wrappers, reviewed definitions and aggregate verification receipts are public;
private/source-controlled measurements remain on 149.


Real-input integration repair
-----------------------------

The real BRCA/LUAD preparation exposed 955 graph genes outside the protein-coding
assay matrix axis. Every one is explicitly listed in the original, hash-verified
product's unresolved/non-coding exclusion artifact. The adapter now preserves
these graph nodes with both assay availability masks false and binds that
exclusion artifact into prepared provenance. It still rejects undocumented
unknown genes and changed exclusion files. This keeps gene CNV and promoter
methylation enabled without inventing measurements or removing graph nodes.


Verification of the delivered revision
--------------------------------------

The materialized matrix contains 10,432 frozen patients from all 33 cancers and
198 score columns. All 18 newly computed RNA dimensions cover those patients;
5,641 patients satisfy the declared SBS quality thresholds. More than 1.2 million
reused historical values and all 187,776 new RNA values were compared directly
with their selected source records. All SBS values and missing masks match the
quality-controlled refit output. The portable R wrapper reproduces the native
CHOL scores. See the aggregate coverage and verification receipts under
provenance/joint_mainline_20261101/ for the regression results and real BRCA/LUAD
prepared coverage. This is input and integration verification, not a completed
33-cancer training run or evidence of biological predictive accuracy.
