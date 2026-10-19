#!/usr/bin/env Rscript
if (!identical(unname(Sys.info()[['nodename']]),'149')) stop('STAR count assembly requires host 149')
suppressPackageStartupMessages({library(SummarizedExperiment);library(data.table);library(jsonlite);library(digest)})
args<-commandArgs(trailingOnly=TRUE);verify_mode<-length(args)==3L && args[[1]]=='--verify'
if (verify_mode) args<-args[-1]
if (length(args)!=2L) stop('usage: assemble_gdc_star_counts.R [--verify] COUNT_INPUTS_JSON COUNT_RDS')
spec_path<-normalizePath(args[[1]],mustWork=TRUE);output<-args[[2]]
record<-function(path) list(path=normalizePath(path,mustWork=TRUE),sha256=digest(file=path,algo='sha256',serialize=FALSE))
verify<-function(item) {if (!identical(record(item$path)$sha256,item$sha256)) stop('STAR source hash differs');item$path}
spec<-fromJSON(spec_path,simplifyVector=FALSE)
if (spec$format!='GDC_STAR_COUNT_ASSEMBLY_INPUT_V1' || spec$target_host!='149' || !identical(spec$training_started,FALSE)) stop('Invalid STAR assembly declaration')
if (!verify_mode && file.exists(output)) stop('Count RDS is immutable')
assays<-c('unstranded','stranded_first','stranded_second')
special<-c('N_unmapped','N_multimapping','N_noFeature','N_ambiguous')
annotation<-NULL;matrices<-NULL;sample_rows<-list();summaries<-list()
for (i in seq_along(spec$files)) {
  item<-spec$files[[i]];path<-verify(item$raw)
  if (file.info(path)$size!=item$file_size || digest(file=path,algo='md5',serialize=FALSE)!=item$md5sum) stop('STAR GDC MD5/size mismatch')
  table<-fread(path,skip='gene_id\t',colClasses='character',na.strings=c('NA',''),check.names=FALSE)
  required<-c('gene_id','gene_name','gene_type',assays)
  if (!all(required %in% names(table)) || anyDuplicated(table$gene_id) || anyNA(table$gene_id)) stop('Invalid STAR gene schema/identity')
  if (!setequal(table$gene_id[startsWith(table$gene_id,'N_')],special)) stop('STAR summary rows are incomplete or unknown')
  for (name in assays) {
    if (anyNA(table[[name]]) || any(!grepl('^[0-9]+$',table[[name]]))) stop('STAR counts must be nonnegative integers')
    values<-as.numeric(table[[name]])
    if (any(!is.finite(values)) || any(values>=2^53)) stop('STAR count precision is invalid')
    set(table,j=name,value=values)
  }
  summaries[[i]]<-as.data.frame(table[gene_id %in% special,c('gene_id',assays),with=FALSE])
  genes<-table[!gene_id %in% special]
  if (!nrow(genes) || anyNA(genes[,.(gene_name,gene_type)]) || any(!grepl('^ENSG[0-9]+(\\.[0-9]+)?(_PAR_Y)?$',genes$gene_id))) stop('Invalid STAR gene annotations')
  setorder(genes,gene_id)
  observed<-as.data.frame(genes[,.(gene_id,gene_name,gene_type)],stringsAsFactors=FALSE)
  if (is.null(annotation)) {
    annotation<-observed
    matrices<-setNames(lapply(assays,function(name) matrix(NA_real_,nrow(genes),length(spec$files))),assays)
  } else if (!identical(annotation,observed)) stop('STAR annotation/universe differs across samples')
  for (name in assays) matrices[[name]][,i]<-genes[[name]]
  sample_rows[[i]]<-as.data.frame(item[setdiff(names(item),'raw')],stringsAsFactors=FALSE)
}
samples<-rbindlist(sample_rows);ids<-samples$sample_id
if (!length(ids) || anyDuplicated(ids) || any(samples$cancer_id!=spec$cancer_id)) stop('STAR sample closure differs')
for (name in assays) dimnames(matrices[[name]])<-list(annotation$gene_id,ids)
rownames(annotation)<-annotation$gene_id
samples<-as.data.frame(samples);rownames(samples)<-ids
assembled<-SummarizedExperiment(assays=matrices,rowData=DataFrame(annotation),colData=DataFrame(samples),
  metadata=list(gdc_star_counts=record(spec_path),star_summary_rows=summaries))
for (item in spec$files) verify(item$raw)
for (item in spec$implementations) verify(item)
verify(spec$query)
if (verify_mode) {
  actual<-readRDS(output)
  if (!identical(assayNames(actual),assayNames(assembled)) || !identical(as.data.frame(rowData(actual)),as.data.frame(rowData(assembled))) ||
      !identical(as.data.frame(colData(actual)),as.data.frame(colData(assembled))) || !identical(metadata(actual),metadata(assembled))) stop('Count RDS metadata differs from raw STAR assembly')
  for (name in assays) if (!identical(as.matrix(assay(actual,name)),as.matrix(assay(assembled,name)))) stop('Count RDS values differ from raw STAR files')
} else saveRDS(assembled,output)
