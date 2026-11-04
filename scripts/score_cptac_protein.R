options(stringsAsFactors=FALSE)
stopifnot(Sys.info()[['nodename']]=='149')
a<-commandArgs(trailingOnly=TRUE);opt<-function(k){i<-match(k,a);if(is.na(i)||i==length(a))stop(k);a[i+1]}
.libPaths(c(strsplit(opt('--r-library'),':',fixed=TRUE)[[1]],.libPaths()))
suppressPackageStartupMessages({library(GSVA);library(data.table);library(jsonlite);library(Matrix)})
setDTthreads(2);BiocParallel::register(BiocParallel::SerialParam());set.seed(20261104)
input<-opt('--input');out<-opt('--output');gmt<-opt('--gmt')
if(dir.exists(out)&&length(list.files(out)))stop('Use a new output directory')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_json(list(target_host='149',action='CPTAC_PROTEIN_PATHWAY_SCORING'),file.path(out,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE)
d<-fread(input,check.names=FALSE);m<-as.matrix(d[,-1,with=FALSE]);rownames(m)<-d[[1]];rm(d)
stopifnot(!anyDuplicated(rownames(m)),!anyDuplicated(colnames(m)),!any(is.infinite(m)))
f<-strsplit(readLines(gmt,warn=FALSE),'\t',fixed=TRUE)
gs<-setNames(lapply(f,function(x)unique(x[-c(1,2)])),vapply(f,'[[',character(1),1L))
coverage<-data.table(pathway_id=names(gs),gene_count=lengths(gs),
 observed_in_assay=vapply(gs,function(g)sum(g %in% rownames(m)),integer(1)))
selected<-coverage$observed_in_assay>=10L & coverage$gene_count<=500L
param<-ssgseaParam(m,gs[selected],minSize=10,maxSize=500,alpha=.25,normalize=FALSE,checkNA='yes',use='na.rm')
scores<-gsva(param,verbose=FALSE,BPPARAM=BiocParallel::SerialParam())
# Explicit sample-level eligibility independently enforces measured overlap.
counts<-t(vapply(gs,function(g)colSums(is.finite(m[intersect(g,rownames(m)),,drop=FALSE])),numeric(ncol(m))))
rownames(counts)<-names(gs);colnames(counts)<-colnames(m)
scores[counts[rownames(scores),,drop=FALSE]<10L]<-NA_real_
stopifnot(!any(is.infinite(scores)))
fwrite(as.data.table(scores,keep.rownames='pathway_id'),file.path(out,'PROTEIN_PATHWAY_ES.tsv.gz'),sep='\t',compress='gzip',na='NA')
fwrite(as.data.table(counts,keep.rownames='pathway_id'),file.path(out,'PROTEIN_PATHWAY_OBSERVED_GENES.tsv.gz'),sep='\t',compress='gzip')
fwrite(coverage,file.path(out,'PROTEIN_PATHWAY_COVERAGE.tsv'),sep='\t')
write_json(list(status='COMPLETE',target_host='149',method='GSVA_ssGSEA_ES',GSVA_version=as.character(packageVersion('GSVA')),
 input_md5=unname(tools::md5sum(input)),gmt_md5=unname(tools::md5sum(gmt)),normalize=FALSE,alpha=.25,min_observed=10,
 max_set_size=500,missing='NA_preserved_and_removed_per_sample_no_imputation',
 interpretation='Relative_protein_gene_set_abundance_not_absolute_pathway_activation',
 pathway_count=nrow(scores),samples=ncol(scores)),file.path(out,'COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
