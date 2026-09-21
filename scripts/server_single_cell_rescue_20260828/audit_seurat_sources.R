#!/usr/bin/env Rscript

# Read-only source audit for the CancerLncAtlas single-cell rescue workspace.
# Source RDS/H5 files under /public8 are never modified.  All outputs are
# materialised below ${DATA_ROOT}/CancerLncAtlas.

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: audit_seurat_sources.R ${DATA_ROOT}/CancerLncAtlas")
}

root <- normalizePath(args[[1]], mustWork = TRUE)
expected_root <- "${DATA_ROOT}/CancerLncAtlas"
if (!identical(root, expected_root)) {
  stop("Refusing output root outside ", expected_root, ": ", root)
}

run_id <- "single_cell_rescue_20260828"
out_dir <- file.path(root, "metadata", run_id)
manifest_dir <- file.path(root, "manifests", run_id)
log_dir <- file.path(root, "logs", run_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(manifest_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

legacy_root <- "${DATA_ROOT}/CancerLncAtlas/raw/single_cell"
h5_root <- "${DATA_ROOT}/CancerLncAtlas/processed/sc_tool_input"
legacy_cancers <- c(
  "BLCA", "BRCA", "COAD", "LIHC", "LUAD",
  "OV", "PAAD", "PRAD", "STAD", "THCA"
)
cesc_path <- paste0(
  "${DATA_ROOT}/CancerLncAtlas/raw/single_cell_23/GSE297041/",
  "GSE297041_CESC_18_scRNA_rmdoublet_0.2_cluster_ident_without_anchor.rds"
)

source_rows <- rbindlist(c(
  lapply(legacy_cancers, function(cancer) {
    data.table(
      cancer_id = cancer,
      source_class = "legacy_seurat",
      source_path = file.path(legacy_root, paste0("01_", cancer, "_processed_seurat.rds")),
      h5_path = file.path(h5_root, cancer, "raw_feature_bc_matrix.h5")
    )
  }),
  list(data.table(
    cancer_id = "CESC",
    source_class = "replacement_candidate_GSE297041",
    source_path = cesc_path,
    h5_path = file.path(h5_root, "CESC", "raw_feature_bc_matrix.h5")
  ))
), use.names = TRUE)

first_present <- function(columns, candidates) {
  hit <- candidates[candidates %in% columns]
  if (length(hit)) hit[[1]] else NA_character_
}

normalise_text <- function(x) {
  y <- trimws(as.character(x))
  y[y %in% c("", "NA", "NaN", "NULL", "None")] <- NA_character_
  y
}

compartment_from_label <- function(x) {
  y <- tolower(normalise_text(x))
  out <- rep(NA_character_, length(y))
  out[grepl("malignan|tumou?r|cancer|epithelial", y)] <- "malignant_or_epithelial"
  out[grepl("t[ _-]?cell|b[ _-]?cell|nk|myeloid|macroph|monocyt|dendrit|neutro|mast|immune|lymph", y)] <- "immune"
  out[grepl("fibro|strom|endothel|pericyte|smooth[ _-]?muscle|mesench", y)] <- "stromal"
  out
}

get_counts <- function(object) {
  assay_name <- if ("RNA" %in% names(object@assays)) "RNA" else DefaultAssay(object)
  assay <- object[[assay_name]]
  layers <- tryCatch(Layers(assay), error = function(e) character())
  count_layers <- layers[grepl("^counts($|\\.)", layers)]
  if (length(count_layers) == 1L) {
    return(list(matrix = LayerData(object, assay = assay_name, layer = count_layers[[1]]),
                assay = assay_name, layers = count_layers))
  }
  if (length(count_layers) > 1L) {
    joined <- JoinLayers(object, assay = assay_name, layers = count_layers, new = "counts")
    return(list(matrix = LayerData(joined, assay = assay_name, layer = "counts"),
                assay = assay_name, layers = count_layers))
  }
  mat <- tryCatch(
    GetAssayData(object, assay = assay_name, slot = "counts"),
    error = function(e) NULL
  )
  if (is.null(mat)) stop("No count layer in assay ", assay_name)
  list(matrix = mat, assay = assay_name, layers = "counts_slot")
}

safe_sha256 <- function(path) {
  out <- suppressWarnings(system2("sha256sum", path, stdout = TRUE, stderr = TRUE))
  if (!length(out)) return(NA_character_)
  strsplit(out[[1]], "[[:space:]]+")[[1]][[1]]
}

audit_one <- function(row) {
  cancer <- row$cancer_id[[1]]
  source_class <- row$source_class[[1]]
  source_path <- row$source_path[[1]]
  h5_path <- row$h5_path[[1]]
  started <- format(Sys.time(), tz = "UTC", usetz = TRUE)
  message("AUDIT_START ", cancer, " ", source_path)

  if (!file.exists(source_path)) {
    return(data.table(
      cancer_id = cancer, source_class = source_class, source_path = source_path,
      h5_path = h5_path, status = "SOURCE_RDS_MISSING", started_utc = started,
      finished_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
    ))
  }

  object <- readRDS(source_path)
  meta <- as.data.table(object[[]], keep.rownames = "cell_id")
  object_cells <- colnames(object)
  if (is.null(object_cells)) stop(cancer, ": object has no cell names")
  if (nrow(meta) != length(object_cells)) stop(cancer, ": metadata/cell count mismatch")
  if (!identical(meta$cell_id, object_cells)) {
    meta <- meta[match(object_cells, cell_id)]
  }

  columns <- names(meta)
  donor_col <- first_present(columns, c(
    "patient_id", "patient", "Patient", "donor_id", "donor", "Donor",
    "caseID", "case_id", "sampleID_v2", "sampleID", "sample_id", "Sample",
    "orig.ident"
  ))
  celltype_col <- first_present(columns, c(
    "cell_type_major", "cell_type", "Celltype", "CellType", "celltype",
    "cell.type", "cell_annotation", "annotation", "cluster_ident", "ident"
  ))
  malignant_col <- first_present(columns, c(
    "author_malignant", "malignant", "Malignant", "malignancy",
    "cell_type_major", "cell_type", "Celltype", "CellType", "celltype"
  ))

  donor <- if (!is.na(donor_col)) normalise_text(meta[[donor_col]]) else rep(NA_character_, nrow(meta))
  celltype <- if (!is.na(celltype_col)) normalise_text(meta[[celltype_col]]) else rep(NA_character_, nrow(meta))
  malignant_raw <- if (!is.na(malignant_col)) normalise_text(meta[[malignant_col]]) else rep(NA_character_, nrow(meta))
  compartment <- compartment_from_label(celltype)

  counts_info <- get_counts(object)
  counts <- counts_info$matrix
  if (!inherits(counts, "sparseMatrix")) counts <- as(counts, "sparseMatrix")
  if (!identical(colnames(counts), object_cells)) {
    idx <- match(object_cells, colnames(counts))
    if (anyNA(idx)) stop(cancer, ": count matrix lacks object cells")
    counts <- counts[, idx, drop = FALSE]
  }
  cell_n_count <- as.numeric(Matrix::colSums(counts))
  cell_n_feature <- as.integer(Matrix::colSums(counts != 0))
  feature_n_count <- as.numeric(Matrix::rowSums(counts))
  feature_n_cell <- as.integer(Matrix::rowSums(counts != 0))

  standard <- data.table(
    cell_id = object_cells,
    raw_cell_id = object_cells,
    dataset_id = if (source_class == "legacy_seurat") paste0("LEGACY_SEURAT_", cancer) else "SC_GSE297041_CESC_RESCUE",
    cancer_id = cancer,
    patient_id_raw = donor,
    patient_id_candidate = donor,
    patient_source_column = donor_col,
    patient_id_verified = FALSE,
    cell_type_raw = celltype,
    cell_type_major_candidate = celltype,
    celltype_source_column = celltype_col,
    compartment_candidate = compartment,
    author_malignant_raw = malignant_raw,
    n_count_rna = cell_n_count,
    n_feature_rna_from_counts = cell_n_feature,
    source_object = source_path,
    source_class = source_class,
    analysis_role = "RESCUE_CANDIDATE_NOT_FORMAL"
  )
  metadata_path <- file.path(out_dir, paste0(cancer, "_cell_metadata_rescue_candidate.tsv.gz"))
  fwrite(standard, metadata_path, sep = "\t", na = "NA", compress = "gzip")

  feature <- data.table(
    feature_id = rownames(counts),
    n_count_rna = feature_n_count,
    n_cells_detected = feature_n_cell
  )
  feature_path <- file.path(out_dir, paste0(cancer, "_feature_count_signature.tsv.gz"))
  fwrite(feature, feature_path, sep = "\t", na = "NA", compress = "gzip")

  result <- data.table(
    cancer_id = cancer,
    source_class = source_class,
    source_path = source_path,
    source_bytes = file.info(source_path)$size,
    source_sha256 = NA_character_,
    h5_path = h5_path,
    h5_exists = file.exists(h5_path),
    object_class = paste(class(object), collapse = ";"),
    assay = counts_info$assay,
    count_layers = paste(counts_info$layers, collapse = ";"),
    features = nrow(counts),
    cells = ncol(counts),
    count_nnz = length(counts@x),
    total_counts = sum(counts@x),
    donor_source_column = donor_col,
    donor_nonmissing_cells = sum(!is.na(donor)),
    donor_candidates = uniqueN(donor[!is.na(donor)]),
    celltype_source_column = celltype_col,
    celltype_nonmissing_cells = sum(!is.na(celltype)),
    celltype_candidates = uniqueN(celltype[!is.na(celltype)]),
    compartment_resolved_cells = sum(!is.na(compartment)),
    metadata_columns = paste(columns, collapse = ";"),
    metadata_path = metadata_path,
    metadata_sha256 = safe_sha256(metadata_path),
    feature_signature_path = feature_path,
    feature_signature_sha256 = safe_sha256(feature_path),
    status = "RDS_AUDIT_COMPLETE_PATIENT_IDS_UNVERIFIED",
    started_utc = started,
    finished_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
  rm(feature, standard, counts, counts_info, meta, object)
  invisible(gc())
  message("AUDIT_DONE ", cancer)
  result
}

results <- vector("list", nrow(source_rows))
for (i in seq_len(nrow(source_rows))) {
  results[[i]] <- tryCatch(
    audit_one(source_rows[i]),
    error = function(e) data.table(
      cancer_id = source_rows$cancer_id[[i]],
      source_class = source_rows$source_class[[i]],
      source_path = source_rows$source_path[[i]],
      h5_path = source_rows$h5_path[[i]],
      status = paste0("AUDIT_ERROR:", conditionMessage(e)),
      started_utc = NA_character_,
      finished_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
    )
  )
  fwrite(
    rbindlist(results[seq_len(i)], fill = TRUE),
    file.path(manifest_dir, "seurat_source_audit.partial.tsv"),
    sep = "\t", na = "NA"
  )
}

final <- rbindlist(results, fill = TRUE)
final_path <- file.path(manifest_dir, "seurat_source_audit.tsv")
fwrite(final, final_path, sep = "\t", na = "NA")
cat("RESULT_PATH\t", final_path, "\n", sep = "")
cat("RESULT_SHA256\t", safe_sha256(final_path), "\n", sep = "")
cat("COMPLETE\t", sum(grepl("^RDS_AUDIT_COMPLETE", final$status)), "/", nrow(final), "\n", sep = "")
