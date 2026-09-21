#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(Seurat)
  library(rhdf5)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: materialize_v32_cesc_gse297041_rescue.R SOURCE_RDS OUTPUT_ROOT")
source_path <- normalizePath(args[[1]], mustWork = TRUE)
output <- args[[2]]
if (dir.exists(output) && length(list.files(output, all.files = TRUE, no.. = TRUE))) stop("immutable output exists: ", output)
dir.create(output, recursive = TRUE, showWarnings = FALSE)
building <- file.path(output, paste0(".building.", Sys.getpid()))
dir.create(building, showWarnings = FALSE)

sha256 <- function(path) strsplit(system2("sha256sum", path, stdout = TRUE)[[1]], "[[:space:]]+")[[1]][[1]]

get_counts <- function(object) {
  assay_name <- if ("RNA" %in% names(object@assays)) "RNA" else DefaultAssay(object)
  assay <- object[[assay_name]]
  layers <- tryCatch(Layers(assay), error = function(e) character())
  count_layers <- layers[grepl("^counts($|\\.)", layers)]
  if (length(count_layers) == 1L) return(LayerData(object, assay = assay_name, layer = count_layers[[1]]))
  if (length(count_layers) > 1L) {
    joined <- JoinLayers(object, assay = assay_name, layers = count_layers, new = "counts")
    return(LayerData(joined, assay = assay_name, layer = "counts"))
  }
  value <- tryCatch(GetAssayData(object, assay = assay_name, slot = "counts"), error = function(e) NULL)
  if (is.null(value)) stop("no RNA count layer")
  value
}

markers <- list(
  T_cell = c("CD3D", "CD3E", "TRAC", "IL7R", "CD8A"),
  B_cell = c("CD79A", "MS4A1", "CD74", "CD37", "CD19"),
  Myeloid = c("LST1", "TYROBP", "FCER1G", "CTSS", "AIF1"),
  Endothelial = c("PECAM1", "VWF", "EMCN", "KDR", "ENG"),
  Fibroblast_stromal = c("COL1A1", "COL1A2", "DCN", "COL3A1", "LUM"),
  Malignant_candidate = c("EPCAM", "KRT8", "KRT18", "KRT19", "KRT5", "KRT14", "MKI67")
)

object <- readRDS(source_path)
counts <- get_counts(object)
if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")
cells <- colnames(counts); features <- rownames(counts)
meta <- as.data.table(object[[]], keep.rownames = "object_cell_id")
meta <- meta[match(cells, object_cell_id)]
if (anyNA(meta$object_cell_id)) stop("metadata/count cell mismatch")
if (!"caseID" %in% names(meta) || uniqueN(meta$caseID) != 10L) stop("GSE297041 ten-patient caseID closure failed")
library_size <- Matrix::colSums(counts)
score <- matrix(0, nrow = length(markers), ncol = ncol(counts), dimnames = list(names(markers), cells))
for (label in names(markers)) {
  rows <- which(toupper(features) %in% toupper(markers[[label]]))
  if (length(rows)) score[label, ] <- log1p((Matrix::colSums(counts[rows, , drop = FALSE]) / pmax(library_size, 1)) * 10000 / length(rows))
}
best <- max.col(t(score), ties.method = "first")
ordered <- apply(score, 2L, sort, decreasing = TRUE)
maximum <- ordered[1L, ]; second <- ordered[2L, ]
labels <- rownames(score)[best]
confident <- maximum >= 0.08 & (maximum - second) >= 0.015
labels[!confident] <- "Other"
confidence <- ifelse(maximum > 0, (maximum - second) / (maximum + 1e-6), 0)
sample_column <- if ("sampleID_v2" %in% names(meta)) "sampleID_v2" else if ("sampleID" %in% names(meta)) "sampleID" else "orig.ident"
patient_raw <- as.character(meta$caseID)
sample_raw <- as.character(meta[[sample_column]])
standard <- data.table(
  patient_id_raw = patient_raw, sample_id_raw = sample_raw, sample_class = "Tumor",
  raw_cell_id = cells, cell_type_raw = labels, cell_type_major = labels,
  author_malignant = labels == "Malignant_candidate",
  lineage_support = labels == "Malignant_candidate", tumor_origin_support = TRUE,
  scevan_tumor = FALSE, scevan_call = "not_assessed", doublet_flag = FALSE,
  doublet_score = NA_real_, cell_id = paste0("GSE297041:", cells),
  dataset_id = "SC_GSE297041_CESC", cancer_id = "CESC",
  patient_id = paste0("SC_GSE297041_CESC:", patient_raw),
  sample_id = paste0("SC_GSE297041_CESC:", sample_raw),
  source_object = source_path,
  source_annotation_source = "computed_broad_marker_scores_from_raw_counts",
  source_candidate_method = "cervical_tumor_and_microenvironment_marker_lineage",
  cell_type_score_confidence = confidence,
  analysis_role = "rescue_candidate_source_series_verified_ten_patients",
  expression_scale = "raw_counts"
)

h5_path <- file.path(building, "raw_feature_bc_matrix.h5")
h5createFile(h5_path)
h5createGroup(h5_path, "matrix")
h5createGroup(h5_path, "matrix/features")
h5write(standard$cell_id, h5_path, "matrix/barcodes")
h5createDataset(h5_path, "matrix/data", dims = length(counts@x),
                storage.mode = storage.mode(counts@x), chunk = min(length(counts@x), 1000000L), level = 4L)
h5write(counts@x, h5_path, "matrix/data")
h5createDataset(h5_path, "matrix/indices", dims = length(counts@i),
                storage.mode = "integer", chunk = min(length(counts@i), 1000000L), level = 4L)
h5write(as.integer(counts@i), h5_path, "matrix/indices")
h5write(as.numeric(counts@p), h5_path, "matrix/indptr")
h5write(as.numeric(dim(counts)), h5_path, "matrix/shape")
h5write(features, h5_path, "matrix/features/id")
h5write(features, h5_path, "matrix/features/name")
h5write(rep("Gene Expression", length(features)), h5_path, "matrix/features/feature_type")
h5write(rep("GRCh38", length(features)), h5_path, "matrix/features/genome")
h5write("genome", h5_path, "matrix/features/_all_tag_keys")
h5closeAll()
metadata_path <- file.path(building, "cell_metadata.tsv.gz")
fwrite(standard, metadata_path, sep = "\t", compress = "gzip", na = "")
context_donors <- standard[, .(donors = uniqueN(patient_id), cells = .N), by = cell_type_major][order(-donors, -cells)]
audit <- list(
  format = "CANCERLNCATLAS_V32_CESC_GSE297041_RESCUE_V1", status = "PASS",
  source_sha256 = sha256(source_path), cells = ncol(counts), features = nrow(counts),
  nonzero = length(counts@x), patients = uniqueN(standard$patient_id),
  formal_lncrna_features_from_prior_authority_audit = 3098L,
  detected_lncrna_features_from_prior_authority_audit = 2460L,
  source_series_claim = "10 cervical cancer patients; 18 longitudinal tumor samples",
  patient_mapping = "RDS caseID retained; ten unique caseIDs agree with GEO series design",
  context_donor_counts = context_donors,
  annotation = "computed broad marker scores; independent audit required"
)
audit_path <- file.path(building, "AUDIT.json")
write_json(audit, audit_path, pretty = TRUE, auto_unbox = TRUE)
success <- list(
  format = audit$format, status = "SUCCESS_RESCUE_SOURCE_MATERIALIZED",
  audit_sha256 = sha256(audit_path), h5_sha256 = sha256(h5_path),
  metadata_sha256 = sha256(metadata_path), cells = ncol(counts), features = nrow(counts),
  patients = uniqueN(standard$patient_id), lncrna_features = 3098L
)
success_path <- file.path(building, "SUCCESS.json")
write_json(success, success_path, pretty = TRUE, auto_unbox = TRUE)
for (path in list.files(building, full.names = TRUE)) file.rename(path, file.path(output, basename(path)))
unlink(building, recursive = FALSE)
cat(toJSON(success, auto_unbox = TRUE), "\n")
