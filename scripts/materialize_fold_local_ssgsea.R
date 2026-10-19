#!/usr/bin/env Rscript
if (!identical(unname(Sys.info()[['nodename']]),'149')) stop('ssGSEA execution requires target host 149')
suppressPackageStartupMessages({library(GSVA);library(BiocParallel);library(data.table);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=4L) stop('usage: materialize_fold_local_ssgsea.R NORMALIZATION_ROOT ANNOTATION_TSV GMT_MANIFEST_JSON NEW_OUTPUT')
record <- function(path) list(path=normalizePath(path,mustWork=TRUE),sha256=digest(file=path,algo='sha256',serialize=FALSE))
verify <- function(item) {if (!identical(record(item$path)$sha256,item$sha256)) stop('ssGSEA source hash mismatch');item$path}
runner <- sub('^--file=','',commandArgs()[grepl('^--file=',commandArgs())])
helper <- file.path(dirname(normalizePath(runner)),'frozen_ssgsea.R')
implementations <- list(record(runner),record(helper));source(helper)
root <- normalizePath(args[[1]],mustWork=TRUE)
inputs <- list(normalization_success=record(file.path(root,'SUCCESS.json')),annotation=record(args[[2]]),gmt_manifest=record(args[[3]]))
success <- read_json(verify(inputs$normalization_success),simplifyVector=FALSE)
if (success$status!='PASS' || success$target_host!='149' || !identical(success$training_started,FALSE) ||
    !setequal(names(success$folds),as.character(0:4))) stop('Incomplete five-fold normalization')
manifest <- read_json(verify(inputs$gmt_manifest),simplifyVector=FALSE)
if (!identical(manifest$format,'SSGSEA_GMT_INPUT_V1') || !length(manifest$collections)) stop('Invalid GMT manifest')
sets <- list();collections <- character();gmt_sources <- list()
for (entry in manifest$collections) {
  collection <- entry$collection
  if (length(collection)!=1L || !grepl('^[A-Z][A-Z0-9_]*$',collection) || collection %in% collections) stop('Invalid or duplicate GMT collection')
  collections <- c(collections,collection);gmt_sources[[collection]] <- entry$file
  lines <- readLines(verify(entry$file),warn=FALSE)
  for (line in lines[nzchar(lines)]) {
    fields <- strsplit(line,'\t',fixed=TRUE)[[1]]
    if (length(fields)<3L || !nzchar(fields[1])) stop('Invalid GMT record')
    id <- paste('MSIGDB',collection,fields[1],sep=':')
    if (!is.null(sets[[id]])) stop('Duplicate canonical pathway identity')
    genes <- unique(fields[-c(1,2)]);genes <- genes[nzchar(genes)]
    if (!length(genes)) stop('Empty GMT gene set')
    sets[[id]] <- genes
  }
}
if (!length(sets)) stop('No GMT gene sets')
annotation <- as.data.frame(fread(verify(inputs$annotation)))
output <- args[[4]]
if (file.exists(output)) stop('Use a new fold-local ssGSEA output')
dir.create(output,recursive=TRUE);output<-normalizePath(output)
write_json(list(target_host='149',training_started=FALSE,operation='FOLD_LOCAL_SSGSEA'),file.path(output,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE,pretty=TRUE)
outputs <- list()
for (fold in 0:4) {
  upstream <- success$folds[[as.character(fold)]]
  norm <- read_json(verify(upstream),simplifyVector=TRUE)
  if (norm$format!='FOLD_LOCAL_TMM_SOURCES_V1' || norm$fold!=fold || norm$cancer_id!=success$cancer_id ||
      !identical(norm$normalization$fold_local,TRUE) || norm$normalization$prior_count!=.25 ||
      norm$normalization$counts_assay!='unstranded') stop('Incompatible fold-local expression source')
  x <- readRDS(verify(norm$outputs$gene))
  samples <- fread(verify(norm$outputs$normalization_samples))
  train <- samples[exported==TRUE & split_role=='train',sample_id]
  emit <- samples[exported==TRUE,sample_id]
  if (!identical(colnames(x),emit)) stop('ssGSEA sample order differs from expression')
  state <- fit_frozen_ssgsea(x,train,annotation,sets,norm$mean_train_effective_library_size)
  raw <- apply_frozen_ssgsea(x,state,scale=FALSE);scaled <- raw
  if (is.finite(state$training_score_range) && state$training_score_range>0) scaled <- raw/state$training_score_range
  else scaled[,] <- NA_real_
  scaled[!state$coverage$analysis_eligible,] <- NA_real_
  target <- file.path(output,paste0('fold=',fold));dir.create(target)
  saveRDS(raw,file.path(target,'raw.RDS'));saveRDS(scaled,file.path(target,'activity.RDS'));saveRDS(state,file.path(target,'state.RDS'))
  fwrite(state$coverage,file.path(target,'coverage.tsv'),sep='\t',na='')
  fwrite(state$gene_detection,file.path(target,'gene_detection.tsv'),sep='\t',na='')
  files <- setNames(lapply(c('raw.RDS','activity.RDS','state.RDS','coverage.tsv','gene_detection.tsv'),function(p) record(file.path(target,p))),
    c('raw','activity','state','coverage','gene_detection'))
  for (item in c(inputs,gmt_sources,implementations,list(upstream),norm$outputs[c('gene','normalization_samples')])) verify(item)
  receipt <- list(format='FOLD_LOCAL_SSGSEA_SOURCES_V1',target_host='149',training_started=FALSE,cancer_id=success$cancer_id,
    fold=fold,normalization=upstream,inputs=inputs,gmt_sources=gmt_sources,implementations=implementations,outputs=files,
    method=list(name='ssGSEA',alpha=.25,min_size=10L,max_size=500L,original_set_size_max=500L,
      cpm_threshold=1,detection_fraction=.20,min_training_detection_n=3L,symbol_collapse='MEAN_LOGCPM',
      scaling='RAW_SCORE_DIVIDED_BY_TRAIN_MAX_MINUS_MIN',emitted_roles=c('train','validation')),
    training_score_range=state$training_score_range,training_detection_required_n=state$training_detection_required_n,
    versions=list(R=R.version.string,GSVA=as.character(packageVersion('GSVA'))))
  path <- file.path(target,'ACTIVITY_NATIVE_SOURCES.json');write_json(receipt,path,auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null')
  outputs[[as.character(fold)]] <- record(path)
}
write_json(list(status='PASS',target_host='149',training_started=FALSE,cancer_id=success$cancer_id,folds=outputs),
  file.path(output,'SUCCESS.json'),auto_unbox=TRUE,pretty=TRUE)
