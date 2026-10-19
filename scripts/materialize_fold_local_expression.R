#!/usr/bin/env Rscript
if (!identical(unname(Sys.info()[['nodename']]),'149')) stop('Expression execution requires target host 149')
suppressPackageStartupMessages({library(SummarizedExperiment);library(edgeR);library(data.table);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=4L) stop('usage: materialize_fold_local_expression.R COUNT_RDS PATIENT_MAP_TSV CANCER NEW_OUTPUT')
record <- function(path) list(path=normalizePath(path,mustWork=TRUE),sha256=digest(file=path,algo='sha256',serialize=FALSE))
unchanged <- function(item) {if (!identical(record(item$path)$sha256,item$sha256)) stop('Expression source changed during processing')}
runner <- sub('^--file=','',commandArgs()[grepl('^--file=',commandArgs())])
helper <- file.path(dirname(normalizePath(runner)),'frozen_tmm.R')
implementations <- list(record(runner),record(helper));source(helper)
inputs <- list(counts=record(args[[1]]),patient_map=record(args[[2]]))
cancer <- args[[3]];output <- args[[4]]
if (!grepl('^[A-Z0-9]+$',cancer)) stop('Invalid cancer identity')
if (file.exists(output)) stop('Use a new fold-local normalization output directory')
dir.create(output,recursive=TRUE);output <- normalizePath(output)
write_json(list(target_host='149',training_started=FALSE,operation='FOLD_LOCAL_TMM_LOGCPM',cancer_id=cancer),file.path(output,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE,pretty=TRUE)
folds <- fread(inputs$patient_map$path)
if (!all(c('cancer_id','patient_id','patient_fold_id') %in% names(folds))) stop('Incomplete patient fold map')
folds <- folds[cancer_id==cancer]
if (!nrow(folds) || anyDuplicated(folds$patient_id) || anyNA(folds$patient_id) ||
    any(!grepl('^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}$',folds$patient_id)) ||
    anyNA(folds$patient_fold_id) || any(!folds$patient_fold_id %in% 0:4) || !setequal(folds$patient_fold_id,0:4))
  stop('Invalid patient folds; require unique canonical patients and all five folds')
x <- readRDS(inputs$counts$path)
rd <- as.data.frame(rowData(x),stringsAsFactors=FALSE)
if (!all(c('gene_id','gene_type') %in% names(rd))) stop('Missing gene annotation')
versioned <- as.character(rd$gene_id);stable <- sub('\\..*$','',versioned)
samples <- colnames(x);counts <- as.matrix(assay(x,'unstranded'))
rownames(counts) <- paste(seq_len(nrow(counts)),versioned,sep=':')
check_count_matrix(counts)
if (anyNA(samples) || any(!grepl('^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-[0-9]{2}',samples))) stop('Invalid TCGA sample identities')
code <- as.integer(substr(samples,14L,15L))
type <- ifelse(code %in% 1:9,'tumor',ifelse(code %in% 10:19,'normal','other'))
priority <- ifelse(type=='tumor' & code==1,1L,ifelse(type=='tumor',2L,ifelse(type=='normal' & code==11,3L,4L)))
qc <- data.table(cancer_id=cancer,sample_id=samples,patient_id=substr(samples,1L,12L),sample_type=type,
                 sample_priority=priority,library_size=as.numeric(colSums(counts)))
setorder(qc,patient_id,sample_type,sample_priority,-library_size,sample_id)
qc[,selected_within_patient_type:=seq_len(.N)==1L,by=.(patient_id,sample_type)]
qc[,patient_fold_id:=folds$patient_fold_id[match(patient_id,folds$patient_id)]]
metadata_keys <- c('pmid','doi','source_dataset','dataset_id','dataset_accession','study_accession','study_dataset_id','experiment_id','experiment_accession')
cd <- as.data.frame(colData(x),stringsAsFactors=FALSE)
for (key in intersect(metadata_keys,names(cd))) qc[[key]] <- as.character(cd[[key]][match(qc$sample_id,samples)])
declared <- metadata(x)$study_metadata
if (is.list(declared)) for (key in intersect(metadata_keys,names(declared))) {
  if (!key %in% names(qc) && length(declared[[key]])==1L && !is.na(declared[[key]])) qc[[key]] <- as.character(declared[[key]])
}
selected_rows <- function(kind) {
  ids <- which(as.character(rd$gene_type)==kind & !is.na(stable) & nzchar(stable) & !grepl('_PAR_Y$',versioned))
  ids[!duplicated(stable[ids])]
}
lnc_rows <- selected_rows('lncRNA');gene_rows <- selected_rows('protein_coding')
if (!length(lnc_rows) || !length(gene_rows)) stop('Need annotated lncRNA and protein-coding genes')
normalization <- list(method='edgeR_TMM',prior_count=0.25,counts_assay='unstranded',fold_local=TRUE,
  normalization_scope='TRAIN_PATIENT_SELECTED_TUMOR_AND_NORMAL',prior_scale='MEAN_TRAIN_EFFECTIVE_LIBRARY_SIZE',
  detection_scope='TRAIN_SELECTED_TUMOR_ONLY',detection_threshold=0.10,emitted_roles=c('train','validation'))
outputs <- list()
for (fold in 0:4) {
  local <- copy(qc)
  local[,split_role:=ifelse(is.na(patient_fold_id),'unassigned',ifelse(patient_fold_id==fold,'test',ifelse(patient_fold_id==(fold+1L)%%5L,'validation','train')))]
  local[,used_for_fit:=selected_within_patient_type & sample_type %in% c('tumor','normal') & split_role=='train']
  local[,exported:=selected_within_patient_type & sample_type=='tumor' & split_role %in% c('train','validation')]
  train <- local[used_for_fit==TRUE,sample_id]
  tumors <- local[used_for_fit==TRUE & sample_type=='tumor',sample_id]
  emit <- local[exported==TRUE,sample_id]
  if (!length(tumors) || !length(emit)) stop('No selected training tumors')
  state <- fit_frozen_tmm(counts[,train,drop=FALSE])
  transformed <- apply_frozen_tmm(counts[,emit,drop=FALSE],state)
  detection <- rowMeans(transformed$logcpm[lnc_rows,tumors,drop=FALSE]>0)
  detected <- data.table(lncrna_id=paste0('LNC:',stable[lnc_rows]),detection_rate_logcpm_gt0=detection,
    analysis_eligible=detection>=.10,unavailable_reason=ifelse(detection>=.10,'','BELOW_TRAIN_DETECTION_THRESHOLD'))
  local[,normalization_factor:=as.numeric(state$train_factors[match(sample_id,names(state$train_factors))])]
  at <- match(emit,local$sample_id);local$normalization_factor[at] <- transformed$factors
  target <- file.path(output,paste0('fold=',fold));dir.create(target)
  lnc <- transformed$logcpm[lnc_rows,,drop=FALSE];rownames(lnc)<-detected$lncrna_id
  gene <- transformed$logcpm[gene_rows,,drop=FALSE];rownames(gene)<-paste0('GENE:',stable[gene_rows])
  saveRDS(lnc,file.path(target,'lncrna.RDS'));saveRDS(gene,file.path(target,'gene.RDS'))
  saveRDS(state,file.path(target,'normalization_state.RDS'))
  fwrite(detected,file.path(target,'detection.tsv'),sep='\t')
  fwrite(local,file.path(target,'normalization_samples.tsv'),sep='\t')
  files <- setNames(lapply(c('lncrna.RDS','gene.RDS','normalization_state.RDS','detection.tsv','normalization_samples.tsv'),function(p) record(file.path(target,p))),
                    c('lncrna','gene','state','detection','normalization_samples'))
  for (item in c(inputs,implementations)) unchanged(item)
  receipt <- list(format='FOLD_LOCAL_TMM_SOURCES_V1',target_host='149',training_started=FALSE,cancer_id=cancer,fold=fold,
    inputs=inputs,implementations=implementations,outputs=files,normalization=normalization,
    reference_sample=state$reference_sample,mean_train_effective_library_size=state$mean_train_effective_library_size,
    raw_file_exposure='WHOLE_COUNT_RDS_INCLUDING_HELDOUT_ROWS',raw_source_ancestry_complete=FALSE,
    versions=list(R=R.version.string,edgeR=as.character(packageVersion('edgeR'))))
  path <- file.path(target,'NORMALIZATION_SOURCES.json');write_json(receipt,path,auto_unbox=TRUE,pretty=TRUE,digits=NA)
  outputs[[as.character(fold)]] <- record(path)
}
write_json(list(status='PASS',target_host='149',training_started=FALSE,cancer_id=cancer,folds=outputs),file.path(output,'SUCCESS.json'),auto_unbox=TRUE,pretty=TRUE)
