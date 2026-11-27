# Patient RNA quantification

TCGA and CPTAC use GDC STAR Counts, GENCODE v36 and the unstranded TPM column. The common patient matrices store `log2(TPM+1)`. TCGA calls the assay `tpm_unstrand` in the exported R object. The original GDC TSV names it `tpm_unstranded`.

The source covers 33 TCGA cancer types and 10 CPTAC cohorts. Each CPTAC RNA sample column retains the dataset, case, tissue type, GDC file and aliquot. A GDC aliquot can have several source tissue pieces. The specimen comparison uses the complete set of pieces and records support for total protein and phosphoproteomics separately. Consult `GDC_SAMPLE_MATCH.tsv` for tumor and normal samples or `RNA_SPECIMEN_SUPPORT.tsv` for a tumor expression matrix. For a same-specimen experiment require the matching flag for each assay involved. Case-level RNA remains usable with a partial or unverified tissue match.

The full CPTAC assay retains the 60,660 genes from the GDC files, including distinct pseudoautosomal Y records. The patient lncRNA matrix uses the existing TCGA axis of 16,889 distinct lncRNAs. Common gene and symbol score inputs keep the prior 59,257-gene background. This allows a comparison across quantifications without simultaneously changing genes used to calculate a score. Member IDs and association estimates for a lncRNA gene set are downstream of these expression matrices.

## Reproducing the input replacement

The commands use an immutable copy of the previous CPTAC input (`$old_raw`), the previous common RNA input (`$old_common`), and a new release directory (`$release`). After activation, the previous paths are recorded in `RNA_QUANTIFICATION_CURRENT.json`. Use the archived directories for a byte-for-byte replay. Use the frozen GDC and PDC metadata to reproduce this release's specimen choices.

```sh
python scripts/fetch_cptac_gdc_star.py "$release/intake" \
  --axis "$old_raw/CPTAC_SAMPLE_ASSAY_MAP.tsv"
python scripts/match_cptac_gdc_star.py \
  --assay-map "$old_raw/CPTAC_SAMPLE_ASSAY_MAP.tsv" \
  --metadata "$release/intake/gdc_files.json" \
  --pdc-metadata "$release/matching/pdc_discovery_all.json" \
  --output "$release/matching"
python -m cla_engine.cptac_gdc_rna \
  --baseline "$old_raw" --matches "$release/matching/GDC_SAMPLE_MATCH.tsv" \
  --intake "$release/intake" --output "$release/source/aligned_r4" --published-root "$raw_live"
python scripts/materialize_patient_star_rna.py \
  --baseline "$old_common" --cptac-aligned "$release/source/aligned_r4" \
  --output "$release/common_next"
python scripts/activate_patient_star_rna.py \
  --raw "$release/source/aligned_r4" --common "$release/common_next" \
  --raw-live "$raw_live" --common-live "$common_live" \
  --record "$functional_root/RNA_QUANTIFICATION_CURRENT.json"
```

The materializer validates the TCGA expression exports and previously computed scores, rebuilds CPTAC RNA views and common expression scores, and preserves sample order. The installer verifies the source and score manifests before replacing the original RNA input paths. Previously published model weights, figures and patient endpoint snapshots retain their original release provenance.
