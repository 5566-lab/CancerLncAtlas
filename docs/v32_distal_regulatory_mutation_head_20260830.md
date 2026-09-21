# V3.2 distal-regulatory mutation head — training log and contract

## Purpose

This is a fresh V3.2 independent functional head. It tests whether somatic
mutations in distal ATAC peaks linked to an lncRNA predict an expression
component for that lncRNA and whether that held-out component predicts
continuous pathway activity in the same cancer. It is supporting association
evidence, not a causal effect and not a replacement for HHGT.

## Authoritative inputs

- TCGA MC3 v0.2.8 mutations are GRCh37. Their pinned coordinates are lifted to
  GRCh38 before any overlap with the GRCh38 ATAC peak set. Direct mixed-build
  overlap is forbidden.
- DataS7 supplies the distal peak-to-lncRNA topology: 7,476 links and 971
  linked lncRNAs.
- TCGA ATAC supplies patient-level distal accessibility where available.
- GDC harmonized Illumina HumanMethylation450 SeSAMe Beta files supply
  promoter and distal-peak methylation. The selected manifest has 8,236
  primary-tumour files for 8,200 patients. Every file must pass manifest size
  and MD5 verification before feature construction.
- Full-33 TCGA masked CNV segments supply local lncRNA CNV; tumour purity is a
  nuisance covariate. Formal V3.2 lncRNA expression and continuous pathway
  activity are the two supervised targets.
- Missing assays remain typed unavailable. An MC3 zero means no reported WES
  event in the linked peak, not proven locus-level wild type.

## Leakage-safe model

The formal r2 head uses nested patient five-fold cross-fitting.

1. For each outer fold, all expression from that fold is removed before
   constructing expression components for the four training folds.
2. Training-fold components are themselves inner-OOF predictions.
3. The outer-heldout expression component is predicted from the four outer
   training folds only.
4. A cancer- and candidate-specific ridge model is fitted from the nested
   mutation-regulatory expression component to pathway activity on the outer
   training patients and evaluated on the outer-heldout patients.
5. The five outer-fold predictions are pooled for OOF RMSE, baseline RMSE,
   delta MSE, R-squared, prediction correlation and mean slope.

This replaces the preliminary r1 ordinary-OOF design, whose first-stage
training components could indirectly use expression from the second-stage
outer test fold. r1 must not be published or used for model selection.

## Formal paths

- Inputs: `./data/CancerLncAtlas/inputs/v32_distal_regulatory_mutation_20260830_r1`
- Head: `./data/CancerLncAtlas/results/v32_distal_regulatory_mutation_20260830_r1/head_full33_nested_patient_oof_r2`
- Closure: `./data/CancerLncAtlas/results/v32_distal_regulatory_mutation_20260830_r1/closure_full33_nested_patient_oof_r2`
- Pipeline log: `./data/CancerLncAtlas/results/v32_distal_regulatory_mutation_20260830_r1/logs/05_distal_regulatory_pipeline_nested_oof_r2.log`

## Raw-data deletion gate

The 107.84 GB methylation download may be deleted only after all of the
following are true: the manifest-wide size and MD5 gate passes, retained
patient-lncRNA methylation features pass their audit, all 33 candidate
partitions and nested components pass the independent closure audit, and the
head `SUCCESS.json` hash matches its audit. Deletion is restricted to the exact
pinned raw download root and writes a non-recoverable deletion receipt.

## Current execution status (2026-08-30)

- Distal mutation features: PASS.
- Distal ATAC topology/features: PASS with typed cancer-level gaps.
- HM450 probe-to-lncRNA promoter/distal map: PASS.
- Methylation download: resumed with 16 GDC client connections; completion
  must be rechecked after COMPUTE_HOST network access is restored.
- Nested-head implementation tests: 10/10 PASS locally.
- G0/G1/G2 materialization: waiting for the local PyTorch/PyG runtime staging
  receipt and a clean r3 restart. The previous r2 attempt failed solely because
  `torch_geometric` was absent and is not reusable.
