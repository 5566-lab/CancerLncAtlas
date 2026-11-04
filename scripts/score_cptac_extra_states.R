options(stringsAsFactors=FALSE)
stopifnot(Sys.info()[['nodename']]=='149')
a<-commandArgs(trailingOnly=TRUE);opt<-function(k){i<-match(k,a);if(is.na(i)||i==length(a))stop(k);a[i+1]}
.libPaths(c(strsplit(opt('--r-library'),':',fixed=TRUE)[[1]],.libPaths()))
suppressPackageStartupMessages({library(data.table);library(jsonlite);library(xCell);library(EXTEND)})
setDTthreads(2);BiocParallel::register(BiocParallel::SerialParam());set.seed(20261104)
root<-opt('--input-root');out<-opt('--output')
if(dir.exists(out)&&length(list.files(out)))stop('Use a new output directory')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_json(list(target_host='149',action='CPTAC_XCELL_EXTEND'),file.path(out,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE)
folders<-sort(list.dirs(file.path(root,'matrices'),recursive=FALSE))
pieces<-list();genes<-NULL
for(d in folders){
  z<-fread(file.path(d,'RNA_TUMOR_SYMBOL_TPM.tsv.gz'),check.names=FALSE)
  m<-as.matrix(z[,-1,with=FALSE]);rownames(m)<-z[[1]]
  pc<-fread(file.path(d,'PROTEIN_CODING_SYMBOLS.tsv'))[[1]]
  m<-log2(m[intersect(rownames(m),pc),,drop=FALSE]+1)
  genes<-if(is.null(genes))rownames(m) else intersect(genes,rownames(m))
  pieces[[basename(d)]]<-m
}
pooled<-do.call(cbind,lapply(pieces,function(m)m[genes,,drop=FALSE]))
cat('XCELL_START',nrow(pooled),ncol(pooled),'\n');flush.console()
xs<-xCell::xCellAnalysis(pooled,parallel.sz=2)
stopifnot(is.matrix(xs)||is.data.frame(xs),nrow(xs)>=64L,identical(colnames(xs),colnames(pooled)))
fwrite(as.data.table(t(as.matrix(xs)),keep.rownames='sample_id'),file.path(out,'XCELL.tsv.gz'),sep='\t',compress='gzip')
rm(pooled,pieces);gc()
# Upstream EXTEND v1 contains an unbound nrow(data) reference in this function.
# Its documented Datasize argument equals the input gene count. Repair only
# that variable reference; preserve original marker ranks and penalty formula.
original<-getFromNamespace('ComponentAndMarkerFunction','EXTEND')
patched<-original
body(patched)<-substitute(B,list(B=body(original)))
txt<-paste(deparse(body(original)),collapse='\n')
stopifnot(grepl('nrow\\(data\\)',txt))
body(patched)<-parse(text=gsub('nrow\\(data\\)','Datasize',txt))[[1]]
ns<-asNamespace('EXTEND');unlockBinding('ComponentAndMarkerFunction',ns);assign('ComponentAndMarkerFunction',patched,envir=ns);lockBinding('ComponentAndMarkerFunction',ns)
ext<-list();marker_records<-list()
for(d in folders){
 z<-fread(file.path(d,'RNA_TUMOR_SYMBOL_TPM.tsv.gz'),check.names=FALSE)
 m<-as.matrix(z[,-1,with=FALSE]);rownames(m)<-z[[1]];m<-log2(m+1)
 original_ids<-colnames(m);colnames(m)<-paste0('S',seq_len(ncol(m)))
 td<-file.path(out,paste0('EXTEND_',basename(d)));dir.create(td)
 oldwd<-getwd();setwd(td)
 tryCatch(EXTEND::RunEXTEND(m),finally=setwd(oldwd))
 e<-fread(file.path(td,'TelomeraseScores.txt'))
 idx<-match(e$SampleID,colnames(m));stopifnot(!anyNA(idx))
 e[,sample_id:=original_ids[idx]];e[,SampleID:=NULL];e[,source_cohort:=basename(d)]
 ext[[basename(d)]]<-e
 markers<-c('C18orf54','CCT6P1','CEP72','POLE2','MCM4','HELLS','TERC','CCT6AP1','C21orf81','LOC81691','AC004381.6','ANKRD20A11P','EHMT2','LIN9','PAXIP1','TERT')
 marker_records[[basename(d)]]<-list(present=intersect(markers,rownames(m)),absent=setdiff(markers,rownames(m)),genes=nrow(m))
}
fwrite(rbindlist(ext),file.path(out,'EXTEND_NATIVE.tsv.gz'),sep='\t',compress='gzip')
write_json(list(status='COMPLETE',target_host='149',xcell_version=as.character(packageVersion('xCell')),
 extend_version=as.character(packageVersion('EXTEND')),xcell_reference='Pooled_CPTAC_tumors_of_selected_cohorts',
 extend_reference='Each_CPTAC_cancer_native_rank_penalty_and_minmax',
 extend_patch='Only_ComponentAndMarkerFunction_nrow(data)_to_Datasize;fix_unbound_variable_in_upstream_v1',
 extend_markers=marker_records,calibration_to_tcga='NOT_ESTABLISHED',original_sample_ids_preserved=TRUE),
 file.path(out,'COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
