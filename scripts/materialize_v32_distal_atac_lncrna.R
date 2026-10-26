#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (!identical(unname(Sys.info()["nodename"]), "149")) {
  stop("Distal ATAC preparation must execute on target host 149")
}
if (!length(args) %in% c(8L, 9L)) {
  stop(paste(
    "usage: materialize_v32_distal_atac_lncrna.R",
    "<matrix.rds> <mapping.tsv> <peak_set.tsv> <topology_links.tsv.gz>",
    "<required_genes.tsv.gz> <patient_folds.tsv> <output_dir> <contract> [declared_cohort.json]"
  ))
}
matrix_path <- normalizePath(args[[1L]], mustWork = TRUE)
mapping_path <- normalizePath(args[[2L]], mustWork = TRUE)
peak_path <- normalizePath(args[[3L]], mustWork = TRUE)
link_path <- normalizePath(args[[4L]], mustWork = TRUE)
required_gene_path <- normalizePath(args[[5L]], mustWork = TRUE)
fold_path <- normalizePath(args[[6L]], mustWork = TRUE)
output_dir <- args[[7L]]
if (!identical(args[[8L]], "CANCERLNCATLAS_V32_DISTAL_ATAC_LNCRNA_V1")) {
  stop("distal ATAC materialization contract marker is missing")
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE)) > 2L) {
  stop("distal ATAC materializer refuses output reuse")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(toJSON(list(target_host = "149", training_started = FALSE), auto_unbox = TRUE),
           file.path(output_dir, "LAUNCH_PREFLIGHT.json"))
topology_lineage <- list(policy = "UNVERIFIED_OR_COHORT_SELECTED", predictive_evaluation_eligible = FALSE)
topology_audit_path <- paste0(link_path, ".audit.json")
if (file.exists(topology_audit_path)) {
  topology_audit <- fromJSON(topology_audit_path)
  if (!identical(unname(tools::md5sum(link_path)), topology_audit$output_md5)) {
    stop("Topology file differs from its bound audit")
  }
  topology_lineage <- list(
    policy = topology_audit$policy, links_sha256 = topology_audit$output_sha256,
    coordinate_build = topology_audit$coordinate_build,
    predictive_evaluation_eligible = identical(topology_audit$status, "PASS") &&
      identical(topology_audit$policy, "GRCH38_PROMOTER_WINDOW_CANDIDATES_V1") &&
      identical(topology_audit$coordinate_build, "GRCh38") &&
      identical(topology_audit$cohort_expression_used, FALSE) &&
      identical(topology_audit$cohort_pathway_activity_used, FALSE))
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
raw_covered <- expected_covered
if (length(args) == 9L) {
  cohort <- fromJSON(normalizePath(args[[9L]], mustWork = TRUE))
  if (!identical(cohort$format, "DISTAL_ATAC_COHORT_V1") ||
      length(cohort$cancers) == 0L || anyDuplicated(cohort$cancers) ||
      !all(cohort$cancers %in% formal_cancers)) stop("invalid declared distal ATAC cohort")
  formal_cancers <- sort(cohort$cancers)
  expected_covered <- intersect(formal_cancers, raw_covered)
  if (!setequal(cohort$covered_cancers, expected_covered) ||
      !setequal(cohort$raw_gap_cancers, setdiff(formal_cancers, raw_covered))) {
    stop("distal ATAC cohort availability differs from official asset")
  }
}
expected_gaps <- setdiff(formal_cancers, expected_covered)

atomic_fwrite <- function(value, path) {
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  fwrite(value, partial, sep = "\t", quote = FALSE, na = "NA", compress = "gzip")
  if (!file.exists(partial) || file.info(partial)$size <= 0L) stop("empty distal ATAC output")
  if (file.exists(path)) stop("distal ATAC output overwrite refused")
  if (!file.rename(partial, path)) stop("distal ATAC atomic rename failed")
}

folds <- fread(fold_path)
if (!all(c("cancer_id", "patient_id", "patient_fold_id") %in% names(folds))) {
  stop("canonical patient folds are malformed")
}
folds[, cancer_id := toupper(as.character(cancer_id))]
folds[, patient_id := substr(as.character(patient_id), 1L, 12L)]
if (anyNA(folds$patient_fold_id) || !all(folds$patient_fold_id %in% 0:4)) stop("invalid patient fold ID")
folds[, patient_fold_id := as.integer(patient_fold_id)]
if (!identical(sort(unique(folds$cancer_id)), sort(formal_cancers)) ||
    !identical(sort(unique(folds$patient_fold_id)), 0:4) ||
    folds[, anyDuplicated(paste(cancer_id, patient_id, sep = "\t"))] > 0L) {
  stop("canonical patient folds differ from declared cancers x folds 0..4")
}
if (!all(folds[, .(ok = setequal(patient_fold_id, 0:4)), by = cancer_id]$ok)) {
  stop("each declared cancer must contain all five folds")
}

mapping <- fread(mapping_path)
if (!all(c("bam_prefix", "aliquot_id", "Case_ID") %in% names(mapping))) {
  stop("official ATAC mapping lacks required identifiers")
}
mapping[, cancer_id := sub("x$", "", sub("-.*$", "", as.character(bam_prefix)))]
mapping[, aliquot_id := as.character(aliquot_id)]
mapping[, patient_id := substr(aliquot_id, 1L, 12L)]
mapping[, expected_matrix_column := gsub("-", "_", bam_prefix, fixed = TRUE)]
if (!identical(sort(unique(mapping$cancer_id)), sort(raw_covered))) {
  stop("official ATAC cancer coverage differs from exact 23")
}
mapping <- merge(
  mapping,
  folds[, .(cancer_id, patient_id, patient_fold_id)],
  by = c("cancer_id", "patient_id"), all.x = TRUE, sort = FALSE
)
unaligned <- unique(mapping[is.na(patient_fold_id), .(cancer_id, patient_id)])

required_genes <- sort(unique(sub("\\..*$", "", sub("^LNC:", "", as.character(
  fread(required_gene_path)$gene_id
)))))
if (length(required_genes) < 1L) stop("required distal lncRNA list is empty")
links <- fread(link_path)
if (!all(c("lncrna_id", "peak_name") %in% names(links))) {
  stop("DataS7 static topology is malformed")
}
links[, gene_id := sub("\\..*$", "", sub("^LNC:", "", as.character(lncrna_id)))]
coordinate_links <- NULL
if (isTRUE(topology_lineage$predictive_evaluation_eligible)) {
  if (!all(c("chromosome", "peak_start0", "peak_end0") %in% names(links))) {
    stop("Predictive topology lacks explicit peak coordinates")
  }
  coordinate_links <- unique(links[gene_id %in% required_genes,
    .(peak_name, chromosome, peak_start0, peak_end0)])
}
links <- unique(links[gene_id %in% required_genes, .(gene_id, peak_name = as.character(peak_name))])
if (nrow(links) < 1L) stop("DataS7 has no topology for required lncRNAs")

atac <- readRDS(matrix_path)
if (!is.data.frame(atac) || nrow(atac) != 562709L || ncol(atac) < 8L) {
  stop("official TCGA ATAC matrix has unexpected shape")
}
mapping[, matrix_column_index := match(expected_matrix_column, colnames(atac))]
if (anyNA(mapping$matrix_column_index) || anyDuplicated(mapping$matrix_column_index)) {
  stop("official ATAC sample mapping does not uniquely map matrix columns")
}
sample_columns <- colnames(atac)[8L:ncol(atac)]
if (!setequal(mapping$matrix_column_index, match(sample_columns, colnames(atac)))) {
  stop("official mapping does not cover every ATAC sample column")
}
aligned_mapping <- mapping[!is.na(patient_fold_id)]

peaks <- fread(peak_path)
if (nrow(peaks) != nrow(atac) || anyDuplicated(peaks$name) || anyDuplicated(atac$name) ||
    !setequal(as.character(peaks$name), as.character(atac$name))) {
  stop("official peak set and ATAC matrix peak identities differ")
}
peak_order <- match(as.character(atac$name), as.character(peaks$name))
if (anyNA(peak_order) ||
    !identical(as.integer(peaks$start[peak_order]), as.integer(atac$start)) ||
    !identical(as.integer(peaks$end[peak_order]), as.integer(atac$end)) ||
    !identical(as.character(peaks$seqnames[peak_order]), as.character(atac$seqnames))) {
  stop("official peak coordinates differ from the ATAC matrix")
}
if (!is.null(coordinate_links)) {
  linked_order <- match(coordinate_links$peak_name, as.character(peaks$name))
  if (anyNA(linked_order) ||
      !identical(sub("^chr", "", as.character(coordinate_links$chromosome)),
                 sub("^chr", "", as.character(peaks$seqnames[linked_order]))) ||
      !identical(as.numeric(coordinate_links$peak_start0), as.numeric(peaks$start[linked_order]) - 1) ||
      !identical(as.numeric(coordinate_links$peak_end0), as.numeric(peaks$end[linked_order]))) {
    stop("Topology peak coordinates differ from the official ATAC matrix")
  }
}

peak_name_index <- setNames(seq_len(nrow(atac)), as.character(atac$name))
links[, peak_index := unname(peak_name_index[peak_name])]
links <- unique(links[!is.na(peak_index), .(gene_id, peak_name, peak_index = as.integer(peak_index))])
gene_levels <- required_genes
links[, gene_index := match(gene_id, gene_levels)]
distal_counts <- tabulate(links$gene_index, nbins = length(gene_levels))
covered_gene_index <- which(distal_counts > 0L)
weights <- sparseMatrix(
  i = links$gene_index,
  j = links$peak_index,
  x = 1 / distal_counts[links$gene_index],
  dims = c(length(gene_levels), nrow(atac))
)
atomic_fwrite(
  links[, .(gene_id, peak_name, peak_index, linked_peak_count = distal_counts[gene_index])],
  file.path(output_dir, "DISTAL_LNCRNA_PEAK_TOPOLOGY.tsv.gz")
)

cancer_audit <- list()
for (cancer in expected_covered) {
  map_cancer_all <- mapping[cancer_id == cancer]
  map_cancer <- aligned_mapping[cancer_id == cancer]
  if (nrow(map_cancer) < 1L) stop(paste("no canonical patients for", cancer))
  aliquot_groups <- split(map_cancer$matrix_column_index, map_cancer$aliquot_id)
  peak_by_aliquot <- vapply(
    aliquot_groups,
    function(indices) rowMeans(as.matrix(atac[, indices, drop = FALSE])),
    numeric(nrow(atac))
  )
  if (is.null(dim(peak_by_aliquot))) peak_by_aliquot <- matrix(peak_by_aliquot, ncol = 1L)
  colnames(peak_by_aliquot) <- names(aliquot_groups)
  aliquot_patient <- unique(map_cancer[, .(aliquot_id, patient_id)])
  aliquot_patient[, column_index := match(aliquot_id, colnames(peak_by_aliquot))]
  patient_groups <- split(aliquot_patient$column_index, aliquot_patient$patient_id)
  peak_by_patient <- vapply(
    patient_groups,
    function(indices) rowMeans(peak_by_aliquot[, indices, drop = FALSE]),
    numeric(nrow(atac))
  )
  if (is.null(dim(peak_by_patient))) peak_by_patient <- matrix(peak_by_patient, ncol = 1L)
  colnames(peak_by_patient) <- names(patient_groups)
  peak_by_patient <- peak_by_patient[, order(colnames(peak_by_patient)), drop = FALSE]
  distal <- as.matrix(weights[covered_gene_index, , drop = FALSE] %*% peak_by_patient)
  if (!all(is.finite(distal))) stop(paste("non-finite distal accessibility", cancer))
  result <- data.table(gene_id = gene_levels[covered_gene_index])
  result <- cbind(result, as.data.table(distal))
  setnames(result, c("gene_id", colnames(peak_by_patient)))
  setorder(result, gene_id)
  atomic_fwrite(result, file.path(output_dir, paste0(cancer, ".distal_accessibility.tsv.gz")))
  cancer_audit[[cancer]] <- list(
    technical_replicates_all = nrow(map_cancer_all),
    technical_replicates_aligned = nrow(map_cancer),
    aligned_patients = ncol(peak_by_patient),
    distal_linked_lncrnas = nrow(result)
  )
  rm(peak_by_aliquot, peak_by_patient, distal, result)
  gc(verbose = FALSE)
}

audit <- list(
  status = "PASS",
  format = "CANCERLNCATLAS_V32_DISTAL_ATAC_LNCRNA_MATERIALIZATION_V1",
  covered_cancers = expected_covered,
  raw_gap_cancers = expected_gaps,
  requested_cancers = formal_cancers,
  raw_covered_cancers = raw_covered,
  scope_policy = if (length(args) == 9L) "DECLARED_CANDIDATE_SCOPE" else "LEGACY_FULL33",
  required_lncrnas = length(required_genes),
  distal_linked_lncrnas = length(covered_gene_index),
  unique_distal_peak_links = nrow(links),
  source_policy = "CANDIDATE_LINKED_PEAK_MEAN_ACCESSIBILITY",
  target_host = "149",
  training_started = FALSE,
  matrix_rows = nrow(atac),
  r_version = as.character(getRversion()),
  package_versions = setNames(lapply(c("data.table", "Matrix", "jsonlite"), function(p) as.character(packageVersion(p))),
                              c("data.table", "Matrix", "jsonlite")),
  topology_lineage = topology_lineage,
  output_md5 = as.list(tools::md5sum(list.files(output_dir, pattern = "distal_accessibility.tsv.gz$", full.names = TRUE))),
  patient_fold_aligned = TRUE,
  technical_replicate_policy = "MEAN_WITHIN_EXACT_ALIQUOT_ID",
  multiple_aliquot_policy = "EQUAL_ALIQUOT_WEIGHT_MEAN_WITHIN_PATIENT",
  unaligned_patients = nrow(unaligned),
  missing_raw_cancer_assumed_zero = FALSE,
  outcome_files_read = list(),
  cancers = cancer_audit
)
audit_path <- file.path(output_dir, "DISTAL_ATAC_AUDIT.json")
partial <- paste0(audit_path, ".partial")
writeLines(toJSON(audit, auto_unbox = TRUE, pretty = TRUE), partial, useBytes = TRUE)
if (file.exists(audit_path)) stop("distal ATAC audit overwrite refused")
if (!file.rename(partial, audit_path)) stop("distal ATAC audit atomic rename failed")
cat(toJSON(audit, auto_unbox = TRUE), "\n")
