#!/usr/bin/env Rscript
if (!identical(unname(Sys.info()[['nodename']]),'149')) stop('Native fold comparison requires target host 149')
suppressPackageStartupMessages({library(jsonlite);library(data.table)})
args<-commandArgs(trailingOnly=TRUE)
if (length(args)!=1L) stop('usage: compare_fold_local_normalization.R COMPARISON_MANIFEST')
spec<-fromJSON(args[[1]],simplifyVector=FALSE)
format<-if (is.null(spec$format)) 'NATIVE_TMM_NUMERIC_REPLAY_V1' else spec$format
if (!format %in% c('NATIVE_TMM_NUMERIC_REPLAY_V1','NATIVE_SSGSEA_NUMERIC_REPLAY_V1')) stop('Unknown native numerical replay format')
if (length(spec$pairs)!=25L || length(spec$metadata)!=5L) stop('Native comparison must cover all five folds and outputs')
equal<-function(a,b,label) {
  result<-all.equal(a,b,tolerance=1e-12,check.attributes=TRUE)
  if (!isTRUE(result)) stop(paste('Native numerical mismatch:',label,paste(result,collapse='; ')))
}
for (item in spec$pairs) {
  if (identical(item$kind,'rds')) {
    actual<-readRDS(item$actual);expected<-readRDS(item$expected)
  } else if (identical(item$kind,'table')) {
    actual<-as.data.frame(fread(item$actual,sep='\t',na.strings=c('NA')))
    expected<-as.data.frame(fread(item$expected,sep='\t',na.strings=c('NA')))
  } else stop('Unknown native comparison kind')
  equal(actual,expected,item$role)
}
for (i in seq_along(spec$metadata)) equal(spec$metadata[[i]]$actual,spec$metadata[[i]]$expected,paste0('metadata.',i))
cat(paste0('PASS_',format,'\n'))
