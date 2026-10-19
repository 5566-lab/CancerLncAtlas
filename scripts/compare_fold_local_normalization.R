#!/usr/bin/env Rscript
if (!identical(unname(Sys.info()[['nodename']]),'149')) stop('Normalization comparison requires target host 149')
suppressPackageStartupMessages({library(jsonlite);library(data.table)})
args<-commandArgs(trailingOnly=TRUE)
if (length(args)!=1L) stop('usage: compare_fold_local_normalization.R COMPARISON_MANIFEST')
spec<-fromJSON(args[[1]],simplifyVector=FALSE)
if (length(spec$pairs)!=25L || length(spec$metadata)!=5L) stop('Normalization comparison must cover all five folds and outputs')
equal<-function(a,b,label) {
  result<-all.equal(a,b,tolerance=1e-12,check.attributes=TRUE)
  if (!isTRUE(result)) stop(paste('Normalization numerical mismatch:',label,paste(result,collapse='; ')))
}
for (item in spec$pairs) {
  if (identical(item$kind,'rds')) {
    actual<-readRDS(item$actual);expected<-readRDS(item$expected)
  } else if (identical(item$kind,'table')) {
    actual<-as.data.frame(fread(item$actual,sep='\t',na.strings=c('NA')))
    expected<-as.data.frame(fread(item$expected,sep='\t',na.strings=c('NA')))
  } else stop('Unknown normalization comparison kind')
  equal(actual,expected,item$role)
}
for (i in seq_along(spec$metadata)) equal(spec$metadata[[i]]$actual,spec$metadata[[i]]$expected,paste0('metadata.',i))
cat('PASS_NATIVE_TMM_NUMERIC_REPLAY\n')
