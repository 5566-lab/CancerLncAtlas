#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(GenomicRanges)
  library(IRanges)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
  stop(paste(
    "usage: materialize_v32_atac_gene_accessibility.R",
    "<matrix.rds> <mapping.tsv> <peak_set.tsv> <promoters.bed.gz>",
    "<required_genes.tsv.gz> <patient_folds.tsv.gz> <output_dir>"
  ))
}

matrix_path <- normalizePath(args[[1L]], mustWork = TRUE)
mapping_path <- normalizePath(args[[2L]], mustWork = TRUE)
peak_path <- normalizePath(args[[3L]], mustWork = TRUE)
promoter_path <- normalizePath(args[[4L]], mustWork = TRUE)
required_gene_path <- normalizePath(args[[5L]], mustWork = TRUE)
fold_path <- normalizePath(args[[6L]], mustWork = TRUE)
output_dir <- normalizePath(args[[7L]], mustWork = TRUE)

# The eighth argument is intentionally reserved for forward-compatible runner
# metadata.  V1 requires the literal contract marker and refuses anything else.
if (!identical(args[[8L]], "CC_HHGT_V3_2_ATAC_MATERIALIZE_V1")) {
  stop("missing exact V3.2 ATAC materialization contract marker")
}

formal_cancers <- c(
  "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
  "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
  "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
  "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
  "UVM"
)
expected_covered <- c(
  "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "ESCA", "GBM",
  "HNSC", "KIRC", "KIRP", "LGG", "LIHC", "LUAD", "LUSC", "MESO",
  "PCPG", "PRAD", "SKCM", "STAD", "TGCT", "THCA", "UCEC"
)
expected_gaps <- setdiff(formal_cancers, expected_covered)

atomic_fwrite <- function(value, path) {
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  fwrite(value, partial, sep = "\t", quote = FALSE, na = "NA", compress = "gzip")
  if (!file.exists(partial) || file.info(partial)$size <= 0L) {
    stop(paste("empty ATAC output", partial))
  }
  if (file.exists(path)) stop(paste("refusing ATAC output overwrite", path))
  if (!file.rename(partial, path)) stop(paste("atomic rename failed", path))
}

mapping <- fread(mapping_path)
mapping_required <- c("bam_prefix", "aliquot_id", "Case_ID")
if (!all(mapping_required %in% names(mapping))) {
  stop("official ATAC mapping lacks bam_prefix/aliquot_id/Case_ID")
}
mapping[, cancer_id := sub("-.*$", "", as.character(bam_prefix))]
mapping[, cancer_id := sub("x$", "", cancer_id)]
mapping[, aliquot_id := as.character(aliquot_id)]
mapping[, patient_id := substr(aliquot_id, 1L, 12L)]
mapping[, expected_matrix_column := gsub("-", "_", bam_prefix, fixed = TRUE)]
if (!identical(sort(unique(mapping$cancer_id)), sort(expected_covered))) {
  stop("official ATAC mapping coverage differs from frozen 23-cancer scope")
}

folds <- fread(fold_path)
if (!all(c("cancer_id", "patient_id", "patient_fold_id") %in% names(folds))) {
  stop("canonical ATAC fold input is malformed")
}
folds[, cancer_id := toupper(as.character(cancer_id))]
folds[, patient_id := substr(as.character(patient_id), 1L, 12L)]
folds[, patient_fold_id := as.integer(patient_fold_id)]
if (!identical(sort(unique(folds$cancer_id)), sort(formal_cancers)) ||
    !identical(sort(unique(folds$patient_fold_id)), 0:4)) {
  stop("canonical fold input is not exact 33 cancers x folds 0..4")
}
if (folds[, anyDuplicated(paste(cancer_id, patient_id, sep = "\t"))] > 0L) {
  stop("canonical fold input duplicates a cancer-patient")
}
mapping <- merge(
  mapping,
  folds[, .(cancer_id, patient_id, patient_fold_id)],
  by = c("cancer_id", "patient_id"),
  all.x = TRUE,
  sort = FALSE
)
unaligned <- unique(mapping[is.na(patient_fold_id), .(cancer_id, patient_id)])
aligned_mapping <- mapping[!is.na(patient_fold_id)]
if (!identical(sort(unique(aligned_mapping$cancer_id)), sort(expected_covered))) {
  stop("one or more raw-covered cancers have zero canonical fold-aligned patients")
}

required_genes <- sort(unique(as.character(fread(required_gene_path)$gene_id)))
required_genes <- sub("\\..*$", "", required_genes)
if (length(required_genes) < 1L) stop("required ATAC gene list is empty")

promoters <- fread(cmd = paste("gzip -cd", shQuote(promoter_path)), fill = TRUE)
setnames(promoters, sub("^#", "", names(promoters)))
if (!all(c("chrom", "start", "end", "gene_id") %in% names(promoters))) {
  stop("GENCODE promoter BED is malformed")
}
promoters[, gene_id_bare := sub("\\..*$", "", as.character(gene_id))]
promoters <- unique(promoters[
  gene_id_bare %in% required_genes,
  .(chrom = as.character(chrom), start = as.integer(start),
    end = as.integer(end), gene_id_bare)
])
if (nrow(promoters) == 0L) stop("no required genes map to GENCODE promoters")

atac <- readRDS(matrix_path)
if (!is.data.frame(atac) || nrow(atac) != 562709L || ncol(atac) < 8L) {
  stop("official TCGA ATAC matrix has unexpected class or shape")
}
sample_columns <- colnames(atac)[8L:ncol(atac)]
mapping[, matrix_column_index := match(expected_matrix_column, colnames(atac))]
if (anyNA(mapping$matrix_column_index)) {
  stop(paste(
    "official mapping columns missing from ATAC matrix",
    paste(head(mapping[is.na(matrix_column_index), bam_prefix], 5L), collapse = ";")
  ))
}
if (anyDuplicated(mapping$matrix_column_index)) {
  stop("official mapping is not one-to-one with ATAC matrix columns")
}
if (!setequal(mapping$matrix_column_index, match(sample_columns, colnames(atac)))) {
  stop("official mapping does not cover every ATAC matrix sample column")
}
# `aligned_mapping` was created before the matrix-column index existed.
# Re-materialize it after the by-reference data.table assignment; otherwise
# the earlier subset correctly remains without this new column and `split()`
# receives NULL at the first cancer.
aligned_mapping <- mapping[!is.na(patient_fold_id)]

peak_set <- fread(peak_path)
if (nrow(peak_set) != nrow(atac) || anyDuplicated(peak_set$name) ||
    anyDuplicated(atac$name) ||
    !setequal(as.character(peak_set$name), as.character(atac$name))) {
  stop("official peak set and ATAC matrix peak names differ")
}
peak_index <- match(as.character(atac$name), as.character(peak_set$name))
if (anyNA(peak_index) ||
    !identical(as.integer(peak_set$start[peak_index]), as.integer(atac$start)) ||
    !identical(as.integer(peak_set$end[peak_index]), as.integer(atac$end)) ||
    !identical(as.character(peak_set$seqnames[peak_index]), as.character(atac$seqnames))) {
  stop("official peak set coordinates differ from the ATAC matrix")
}

peak_gr <- GRanges(
  seqnames = as.character(atac$seqnames),
  ranges = IRanges(start = as.integer(atac$start) + 1L, end = as.integer(atac$end))
)
promoter_gr <- GRanges(
  seqnames = as.character(promoters$chrom),
  ranges = IRanges(
    start = as.integer(promoters$start) + 1L,
    end = as.integer(promoters$end)
  ),
  gene_id = promoters$gene_id_bare
)
overlap <- findOverlaps(promoter_gr, peak_gr, ignore.strand = TRUE)
pairs <- unique(data.table(
  gene_id = as.character(mcols(promoter_gr)$gene_id[queryHits(overlap)]),
  peak_index = as.integer(subjectHits(overlap))
))
gene_levels <- required_genes
pairs[, gene_index := match(gene_id, gene_levels)]
pairs <- pairs[!is.na(gene_index)]
promoter_counts <- tabulate(pairs$gene_index, nbins = length(gene_levels))
covered_gene_index <- which(promoter_counts > 0L)
if (length(covered_gene_index) < 1L) stop("no required gene promoter overlaps ATAC peaks")
weights <- sparseMatrix(
  i = pairs$gene_index,
  j = pairs$peak_index,
  x = 1 / promoter_counts[pairs$gene_index],
  dims = c(length(gene_levels), nrow(atac))
)

cancer_audit <- list()
for (cancer in expected_covered) {
  map_cancer_all <- mapping[cancer_id == cancer]
  map_cancer <- aligned_mapping[cancer_id == cancer]
  if (nrow(map_cancer) == 0L) stop(paste("no aligned ATAC mapping for", cancer))

  # First average technical sequencing/library replicates within the exact
  # aliquot ID.  This avoids the historical substr(Case_ID, 1, 16) collapse.
  aliquot_groups <- split(map_cancer$matrix_column_index, map_cancer$aliquot_id)
  peak_by_aliquot <- vapply(
    aliquot_groups,
    function(indices) rowMeans(as.matrix(atac[, indices, drop = FALSE])),
    numeric(nrow(atac))
  )
  if (is.null(dim(peak_by_aliquot))) {
    peak_by_aliquot <- matrix(peak_by_aliquot, ncol = 1L)
  }
  colnames(peak_by_aliquot) <- names(aliquot_groups)

  # Then explicitly average aliquot means with equal aliquot weight inside a
  # patient.  This policy is registered in both the R and Python lineages.
  aliquot_patient <- unique(map_cancer[, .(aliquot_id, patient_id)])
  aliquot_patient[, column_index := match(aliquot_id, colnames(peak_by_aliquot))]
  if (anyNA(aliquot_patient$column_index)) stop(paste("aliquot grouping failed", cancer))
  patient_groups <- split(aliquot_patient$column_index, aliquot_patient$patient_id)
  peak_by_patient <- vapply(
    patient_groups,
    function(indices) rowMeans(peak_by_aliquot[, indices, drop = FALSE]),
    numeric(nrow(atac))
  )
  if (is.null(dim(peak_by_patient))) {
    peak_by_patient <- matrix(peak_by_patient, ncol = 1L)
  }
  colnames(peak_by_patient) <- names(patient_groups)
  patient_order <- order(colnames(peak_by_patient))
  peak_by_patient <- peak_by_patient[, patient_order, drop = FALSE]

  gene_by_patient <- as.matrix(weights[covered_gene_index, , drop = FALSE] %*% peak_by_patient)
  if (!all(is.finite(gene_by_patient))) stop(paste("non-finite gene accessibility", cancer))
  result <- data.table(gene_id = gene_levels[covered_gene_index])
  result <- cbind(result, as.data.table(gene_by_patient))
  setnames(result, c("gene_id", colnames(peak_by_patient)))
  setorder(result, gene_id)
  output_path <- file.path(output_dir, paste0(cancer, ".gene_accessibility.tsv.gz"))
  atomic_fwrite(result, output_path)

  patient_aliquot_counts <- aliquot_patient[, .N, by = patient_id]
  cancer_audit[[cancer]] <- list(
    technical_replicates_all = nrow(map_cancer_all),
    technical_replicates_aligned = nrow(map_cancer),
    full_aliquots_all = uniqueN(map_cancer_all$aliquot_id),
    full_aliquots_aligned = uniqueN(map_cancer$aliquot_id),
    aligned_patients = ncol(peak_by_patient),
    patients_with_multiple_aliquots = sum(patient_aliquot_counts$N > 1L),
    promoter_mapped_genes = nrow(result),
    output_rows = nrow(result),
    output_columns = ncol(result)
  )
  rm(peak_by_aliquot, peak_by_patient, gene_by_patient, result)
  gc(verbose = FALSE)
}

audit <- list(
  status = "PASS",
  analysis_version = "CancerLncAtlas_V3.2_FULL_MULTITASK",
  module_id = "atac_gene_accessibility_materialization",
  covered_cancers = expected_covered,
  raw_gap_cancers = expected_gaps,
  matrix_rows = nrow(atac),
  matrix_sample_columns = length(sample_columns),
  official_mapping_rows = nrow(mapping),
  full_aliquots = uniqueN(mapping$aliquot_id),
  unique_patients = uniqueN(mapping$patient_id),
  aligned_patients = uniqueN(aligned_mapping[, paste(cancer_id, patient_id)]),
  unaligned_patients = nrow(unaligned),
  required_genes = length(required_genes),
  promoter_mapped_genes = length(covered_gene_index),
  technical_replicate_policy = "MEAN_WITHIN_EXACT_ALIQUOT_ID",
  multiple_aliquot_policy = paste0(
    "EXPLICIT_EQUAL_ALIQUOT_WEIGHT_MEAN_WITHIN_PATIENT_",
    "AFTER_TECHNICAL_REPLICATE_MEAN"
  ),
  historical_sample16_aggregation_used = FALSE,
  trailing_x_cancer_normalization_applied = TRUE,
  outcome_files_read = list(),
  old_predictions_read = FALSE,
  old_checkpoints_read = FALSE,
  family_pathway_membership_read = FALSE,
  missing_assumed_zero = FALSE,
  cancers = cancer_audit
)
audit_path <- file.path(output_dir, "ATAC_GENE_ACCESSIBILITY_AUDIT.json")
partial <- paste0(audit_path, ".partial")
writeLines(toJSON(audit, auto_unbox = TRUE, pretty = TRUE), partial, useBytes = TRUE)
if (file.exists(audit_path)) stop("refusing ATAC audit overwrite")
if (!file.rename(partial, audit_path)) stop("ATAC audit atomic rename failed")
cat(toJSON(audit, auto_unbox = TRUE), "\n")
