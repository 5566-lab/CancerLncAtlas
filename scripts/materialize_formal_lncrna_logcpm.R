#!/usr/bin/env Rscript
if (!identical(unname(Sys.info()[['nodename']]), '149')) stop('Expression execution requires target host 149')
suppressPackageStartupMessages({library(SummarizedExperiment); library(edgeR); library(data.table); library(jsonlite); library(digest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop('usage: materialize_formal_lncrna_logcpm.R INPUT_ROOT THRESHOLD_DIR OUTPUT_DIR')
input_root <- normalizePath(args[[1]], mustWork=TRUE)
threshold_dir <- normalizePath(args[[2]], mustWork=TRUE)
output_dir <- args[[3]]
if (file.exists(output_dir)) stop('Use a new expression normalization output directory')
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
output_dir <- normalizePath(output_dir, mustWork=TRUE)
write_json(list(target_host='149', training_started=FALSE, operation='FORMAL_TMM_LOGCPM'), file.path(output_dir,'LAUNCH_PREFLIGHT.json'), auto_unbox=TRUE, pretty=TRUE)
record <- function(path) list(path=normalizePath(path,mustWork=TRUE), sha256=digest(file=path,algo='sha256',serialize=FALSE))
unchanged <- function(item) {if (!identical(record(item$path)$sha256,item$sha256)) stop('Expression source changed during processing')}
runner <- sub('^--file=', '', commandArgs()[grepl('^--file=',commandArgs())])
implementation <- record(runner)
detection <- record(file.path(threshold_dir, 'full_gdc_lncrna_detection_rates.tsv.gz'))
rates <- fread(detection$path)
unchanged(detection)
if (!all(c('cancer_id','lncrna_id','detection_rate_logcpm_gt0') %in% names(rates))) stop('Incomplete detection table')
if (anyDuplicated(rates[,.(cancer_id,lncrna_id)]) || any(!is.finite(rates$detection_rate_logcpm_gt0)) || any(rates$detection_rate_logcpm_gt0 < 0 | rates$detection_rate_logcpm_gt0 > 1)) stop('Invalid detection table')
rates[, cancer_id := as.character(cancer_id)]
rates[, lncrna_id := as.character(lncrna_id)]
keep_by_cancer <- rates[detection_rate_logcpm_gt0 >= 0.10, .(lncrna_id), by=cancer_id]
cancers <- sort(unique(keep_by_cancer$cancer_id))
if (!length(cancers) || any(!grepl('^[A-Z0-9]+$',cancers))) stop('Invalid cancer scope')
sample_rows <- list()
counts <- list()
for (cancer in cancers) {
  message(sprintf('%s', cancer))
  raw_source <- record(file.path(input_root, sprintf('TCGA-%s.RDS', cancer)))
  x <- readRDS(raw_source$path)
  counts_assembly <- NULL
  count_receipt <- paste0(raw_source$path,'.sources.json')
  if (!is.null(metadata(x)$gdc_star_counts) && !file.exists(count_receipt)) stop('STAR count source receipt is missing')
  if (file.exists(count_receipt)) {
    assembly <- fromJSON(count_receipt,simplifyVector=FALSE)
    completed <- fromJSON(file.path(dirname(count_receipt),'SUCCESS.json'),simplifyVector=FALSE)
    if (completed$status!='PASS' || completed$target_host!='149' || !identical(completed$training_started,FALSE) ||
        !identical(completed$source,record(count_receipt))) stop('STAR count assembly is not complete')
    if (assembly$format!='GDC_STAR_COUNT_ASSEMBLY_V1' || assembly$cancer_id!=cancer ||
        !identical(assembly$output,raw_source) || !identical(assembly$inputs,metadata(x)$gdc_star_counts)) stop('STAR count source binding differs')
    unchanged(assembly$inputs);counts_assembly<-record(count_receipt)
  }
  unchanged(raw_source)
  rd <- as.data.frame(rowData(x), stringsAsFactors=FALSE)
  versioned_id <- as.character(rd$gene_id)
  stable_id <- sub('\\..*$', '', versioned_id)
  keep <- as.character(rd$gene_type) == 'lncRNA'
  keep <- keep & !grepl('_PAR_Y$', versioned_id)
  keep <- keep & !is.na(stable_id) & nzchar(stable_id)
  selected <- which(keep)
  selected <- selected[!duplicated(stable_id[selected])]
  lnc_ids <- paste0('LNC:', stable_id[selected])
  sample_ids <- colnames(x)
  counts_all <- assay(x, 'unstranded')
  if (anyDuplicated(sample_ids) || anyNA(sample_ids) || any(!grepl('^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-[0-9]{2}',sample_ids))) stop('Invalid or duplicated TCGA sample identities')
  if (any(!is.finite(counts_all)) || any(counts_all < 0)) stop('Invalid unstranded counts')
  library_size <- colSums(counts_all)
  if (any(library_size <= 0)) stop('Nonpositive library size')
  sample_code <- suppressWarnings(as.integer(substr(sample_ids, 14L, 15L)))
  sample_class <- ifelse(sample_code >= 1L & sample_code <= 9L, 'tumor', ifelse(sample_code >= 10L & sample_code <= 19L, 'normal', 'other'))
  sample_priority <- ifelse(sample_class == 'tumor' & sample_code == 1L, 1L, ifelse(sample_class == 'tumor', 2L, ifelse(sample_class == 'normal' & sample_code == 11L, 3L, 4L)))
  sample_qc <- data.table(cancer_id=cancer, sample_id=sample_ids, patient_id=substr(sample_ids,1L,12L), sample_type=sample_class, sample_priority=sample_priority, library_size=as.numeric(library_size))
  setorder(sample_qc, patient_id, sample_type, sample_priority, -library_size, sample_id)
  sample_qc[, selected_within_patient_type := seq_len(.N) == 1L, by=.(patient_id, sample_type)]
  sample_qc[, include_expression := selected_within_patient_type & sample_type %in% c('tumor','normal')]
  sample_qc[, include_association := include_expression & sample_type == 'tumor']
  selected_samples <- sample_qc[include_expression == TRUE, sample_id]
  tumor_samples <- sample_qc[include_association == TRUE, sample_id]
  selected_columns <- match(selected_samples, sample_ids)
  tumor_in_selected <- match(tumor_samples, selected_samples)
  if (!length(tumor_samples)) stop('No selected tumor samples')
  y <- DGEList(counts=counts_all[, selected_columns, drop=FALSE])
  y <- calcNormFactors(y, method='TMM')
  logcpm <- cpm(y, log=TRUE, prior.count=0.25)
  sample_qc[, normalization_factor := as.numeric(y$samples$norm.factors[match(sample_id,selected_samples)])]
  # These columns come from the actual RDS metadata, never from path names.
  metadata_keys <- c('pmid','doi','source_dataset','dataset_id','dataset_accession','study_accession','study_dataset_id','experiment_id','experiment_accession')
  cd <- as.data.frame(colData(x),stringsAsFactors=FALSE)
  for (key in intersect(metadata_keys,names(cd))) sample_qc[[key]] <- as.character(cd[[key]][match(sample_qc$sample_id,sample_ids)])
  declared <- metadata(x)$study_metadata
  if (is.list(declared)) for (key in intersect(metadata_keys,names(declared))) {
    if (!key %in% names(sample_qc) && length(declared[[key]])==1L && !is.na(declared[[key]])) sample_qc[[key]] <- as.character(declared[[key]])
  }
  wanted <- keep_by_cancer[cancer_id == cancer, lncrna_id]
  row_index <- match(wanted, lnc_ids)
  if (anyNA(row_index)) stop(sprintf('%s has threshold IDs absent from source: %d', cancer, sum(is.na(row_index))))
  matrix_out <- logcpm[selected[row_index], tumor_in_selected, drop=FALSE]
  rownames(matrix_out) <- wanted
  colnames(matrix_out) <- tumor_samples
  if (any(!is.finite(matrix_out))) stop('Nonfinite normalized expression')
  matrix_path <- file.path(output_dir, sprintf('TCGA-%s_formal_logcpm.RDS', cancer))
  sample_path <- file.path(output_dir, sprintf('TCGA-%s_tumor_samples.tsv', cancer))
  exposure_path <- file.path(output_dir, sprintf('TCGA-%s_normalization_samples.tsv', cancer))
  saveRDS(matrix_out, matrix_path, compress=TRUE)
  fwrite(sample_qc[include_association == TRUE, .(cancer_id, sample_id, patient_id)], sample_path, sep='\t')
  fwrite(sample_qc, exposure_path, sep='\t')
  unchanged(raw_source); unchanged(detection); unchanged(implementation)
  receipt <- list(format='FORMAL_TMM_LOGCPM_SOURCES_V1', target_host='149', training_started=FALSE,
    cancer_id=cancer, inputs=list(counts=raw_source,detection=detection), implementation=implementation,
    outputs=list(matrix=record(matrix_path),tumor_samples=record(sample_path),normalization_samples=record(exposure_path)),
    normalization=list(method='edgeR_TMM',prior_count=0.25,counts_assay='unstranded',normalization_scope='SELECTED_TUMOR_AND_NORMAL_SAMPLES_WITHIN_CANCER',fold_local=FALSE),
    versions=list(R=R.version.string,edgeR=as.character(packageVersion('edgeR'))),
    raw_source_ancestry_complete=FALSE)
  if (!is.null(counts_assembly)) {unchanged(counts_assembly);receipt$counts_assembly<-counts_assembly}
  write_json(receipt,file.path(output_dir,sprintf('TCGA-%s_SOURCE_RECEIPT.json',cancer)),auto_unbox=TRUE,pretty=TRUE)
  counts[[cancer]] <- list(cancer_id=cancer, n_lnc=nrow(matrix_out), n_samples=ncol(matrix_out))
  rm(x, rd, counts_all, y, logcpm, matrix_out); gc(verbose=FALSE)
}
write_json(list(status='PASS', source_universe='DECLARED_DETECTION_TABLE', target_host='149', training_started=FALSE, cancer_count=length(cancers), total_matrix_files=length(cancers), counts=counts), file.path(output_dir, 'SUCCESS.json'), auto_unbox=TRUE, pretty=TRUE)
