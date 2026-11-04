stopifnot(Sys.info()[['nodename']]=='149')
suppressPackageStartupMessages({library(SummarizedExperiment);library(data.table);library(jsonlite);library(digest)})
a<-commandArgs(trailingOnly=TRUE)
if(length(a)<4L)stop('RAW_ROOT PATIENT_MAP OUTPUT CANCER...')
rawroot<-a[1];patient_map<-a[2];out<-a[3];cancers<-a[-(1:3)]
if(dir.exists(out)&&length(list.files(out)))stop('Use a new export directory')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_json(list(target_host='149',action='EXPORT_UNFILTERED_TCGA_TPM',cancers=cancers,training_started=FALSE),
 file.path(out,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE,pretty=TRUE)
setDTthreads(2L)
authority<-fread(patient_map)
sha<-function(p)digest(file=p,algo='sha256',serialize=FALSE)
receipts<-list()
for(cancer in cancers){
 p<-file.path(rawroot,paste0('TCGA-',cancer,'.RDS'));before<-sha(p)
 x<-readRDS(p);stopifnot('tpm_unstrand'%in%assayNames(x))
 rd<-as.data.frame(rowData(x));ids<-sub('\\..*$','',as.character(rd$gene_id))
 keep<-!grepl('_PAR_Y$',as.character(rd$gene_id))&!duplicated(ids)
 auth<-authority[cancer_id==cancer]
 stopifnot(!anyDuplicated(auth$patient_id),!anyDuplicated(auth$sample_id))
 take<-match(auth$sample_id,colnames(x));stopifnot(!anyNA(take))
 m<-assay(x,'tpm_unstrand')[keep,take,drop=FALSE]
 stopifnot(all(is.finite(m)),all(m>=0),identical(before,sha(p)))
 colnames(m)<-auth$patient_id;rownames(m)<-ids[keep]
 d<-file.path(out,cancer);dir.create(d)
 fwrite(as.data.table(m,keep.rownames='gene_id'),file.path(d,'GENE_TPM.tsv.gz'),sep='\t',compress='gzip')
 fwrite(data.table(gene_id=ids[keep],gene_symbol=as.character(rd$gene_name[keep]),gene_type=as.character(rd$gene_type[keep])),
 file.path(d,'GENE_ANNOTATION.tsv'),sep='\t')
 receipts[[cancer]]<-list(raw_path=p,raw_sha256=before,assay='tpm_unstrand',patients=ncol(m),genes=nrow(m),
 source_lncRNAs=sum(rd$gene_type[keep]=='lncRNA'),expression_filter='NONE',patient_rule='EXACT_EXISTING_AUTHORITY_SAMPLE',
 outputs=list(tpm_sha256=sha(file.path(d,'GENE_TPM.tsv.gz')),annotation_sha256=sha(file.path(d,'GENE_ANNOTATION.tsv'))))
 cat('EXPORTED',cancer,ncol(m),nrow(m),receipts[[cancer]]$source_lncRNAs,'\n');flush.console()
 rm(x,m);gc()
}
write_json(list(status='COMPLETE',target_host='149',patient_map=patient_map,patient_map_sha256=sha(patient_map),cancers=receipts),
 file.path(out,'COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
