options(stringsAsFactors=FALSE)
stopifnot(Sys.info()[['nodename']]=='149')
args <- commandArgs(trailingOnly=TRUE)
read_option <- function(name,default=NULL) {
  i <- match(name,args)
  if(is.na(i))return(default)
  if(i==length(args))stop(paste('Missing value:',name))
  args[[i+1L]]
}
out <- read_option('--output-root');input_root <- read_option('--input-root')
sen_path <- read_option('--senmayo-json');r_library <- read_option('--r-library')
if(any(vapply(list(out,input_root,sen_path,r_library),is.null,logical(1))))stop('Required: --output-root --input-root --senmayo-json --r-library')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(r_library,.libPaths()))
suppressPackageStartupMessages({library(signifinder);library(GSVA);library(data.table);library(jsonlite);library(SummarizedExperiment)})
stopifnot(as.character(packageVersion('signifinder'))=='1.9.5')
data.table::setDTthreads(2L);BiocParallel::register(BiocParallel::SerialParam());set.seed(20261101)
mode <- read_option('--cancer-type',read_option('--cancer_type','all'))
available <- sub('TCGA-(.*)_data.RDS','\\1',basename(Sys.glob(file.path(input_root,'TCGA-*_data.RDS'))))
if(mode=='all') cancers <- available else if(mode=='list') {
  requested <- read_option('--cancer-list',read_option('--cancer_list',''))
  cancers <- sort(unique(toupper(trimws(strsplit(requested,',',fixed=TRUE)[[1L]]))))
  if(!length(cancers)||any(!nzchar(cancers))||any(!cancers %in% available))stop('Invalid/empty cancer list; use comma-separated IDs present in input-root')
} else stop('cancer-type must be all or list')
stopifnot(length(cancers)>0L,file.exists(sen_path))
jsonlite::write_json(list(target_host='149',action='RNA_STATE_SCORING',cancer_type=mode,cancers=cancers,input_root=input_root),file.path(out,'RNA_SCORING_PREFLIGHT.json'),auto_unbox=TRUE,pretty=TRUE)
recipes <- list(
  Hypoxia_Buffa=list(fun='hypoxiaSign',args=list(inputType='rnaseq',author='Buffa')),
  EMT_Miow=list(fun='EMTSign',args=list(inputType='rnaseq',author='Miow')),
  CellCycle_Lundberg=list(fun='cellCycleSign',args=list(inputType='rnaseq',author='Lundberg')),
  MitoticIndex_Yang=list(fun='mitoticIndexSign',args=list()),
  CIN70=list(fun='CINSign',args=list(inputType='rnaseq')),
  DNARepair_Kang=list(fun='DNArepSign',args=list(inputType='rnaseq')),
  Glycolysis_Zhang=list(fun='glycolysisSign',args=list(author='Zhang')),
  LipidMetabolism_Zheng=list(fun='lipidMetabolismSign',args=list()),
  VEGF_Hu=list(fun='VEGFSign',args=list()),
  IFNG_Ayers=list(fun='IFNSign',args=list()),
  TCellInflamed_Ayers=list(fun='TinflamSign',args=list(author='Ayers')),
  APM_Wang=list(fun='APMSign',args=list(author='Wang',inputType='rnaseq',kcdf='Gaussian')),
  TLS_Cabrita=list(fun='TLSSign',args=list(inputType='rnaseq')),
  LRRC15_CAF_Dominguez=list(fun='LRRC15CAFSign',args=list()),
  ECM_Chakravarthy=list(fun='ECMSign',args=list())
)
metadata <- signifinder::availableSignatures()
write.table(metadata,file.path(out,'SIGNIFINDER_SIGNATURE_TABLE.tsv'),sep='\t',row.names=FALSE,quote=TRUE)
sen <- jsonlite::read_json(sen_path,simplifyVector=TRUE)$SAUL_SEN_MAYO$geneSymbols
stopifnot(length(sen)>=120L)
folder <- file.path(out,'rna_scores');dir.create(folder,showWarnings=FALSE)
for(cancer in sort(unique(cancers))) {
  final <- file.path(folder,paste0(cancer,'.tsv.gz'));receipt <- file.path(folder,paste0(cancer,'.json'))
  if(file.exists(receipt) || file.exists(final)) {
    if(!file.exists(receipt)||!file.exists(final))stop(paste('Incomplete prior output; use a new output-root:',cancer))
    previous <- jsonlite::read_json(receipt,simplifyVector=TRUE)
    if(previous$state!='COMPLETE'||length(previous$failures)||previous$package_version!='1.9.5')stop(paste('Prior output requires repair in a new output-root:',cancer))
    if(!identical(previous$output_md5,as.character(tools::md5sum(final))))stop('Prior output checksum differs')
    if(!identical(previous$input_md5,as.character(tools::md5sum(file.path(input_root,paste0('TCGA-',cancer,'_data.RDS'))))))stop('Prior input checksum differs')
    cat('ALREADY_COMMITTED',cancer,'\n');next
  }
  input <- file.path(input_root,paste0('TCGA-',cancer,'_data.RDS'))
  cat('START_CANCER',cancer,'\n');flush.console()
  m <- as.matrix(readRDS(input));storage.mode(m)<-'double'
  original_sample_ids <- colnames(m)
  # These RDS inputs use R's dot-separated rendering of full TCGA barcodes.
  # Restore separators only; retain every barcode component without truncation.
  colnames(m) <- toupper(gsub('.', '-', colnames(m), fixed=TRUE))
  stopifnot(!anyDuplicated(rownames(m)),!anyDuplicated(colnames(m)),all(is.finite(m)),all(m>=0))
  # Classify complete TCGA sample barcodes. Normal tissues never enter tumor scoring cohorts.
  valid <- grepl('^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]',colnames(m))
  types <- as.integer(sub('^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-([0-9]{2}).*','\\1',colnames(m)))
  take <- valid & types>=1L & types<=9L
  stopifnot(any(take));excluded <- sum(!take);m <- m[,take,drop=FALSE]
  result <- data.table(sample_id=colnames(m),cancer_id=cancer)
  method_records <- list();failures <- list()
  for(name in names(recipes)) {
    r <- recipes[[name]];warnings <- character()
    cat('SCORE',cancer,name,'\n');flush.console()
    outcome <- tryCatch(withCallingHandlers({
      # Rank/enrichment methods use log2(TPM+1); other functions retain their
      # own documented transformations of normalized unlogged expression.
      expr <- if(name %in% c('EMT_Miow','APM_Wang','ECM_Chakravarthy')) log2(m+1) else m
      scored <- do.call(getExportedValue('signifinder',r$fun),c(list(dataset=expr,nametype='SYMBOL'),r$args))
      scores <- as.data.frame(SummarizedExperiment::colData(scored))
      stopifnot(identical(rownames(scores),colnames(m)))
      if(name=='CIN70')scores <- scores[,'CIN_Carter_70',drop=FALSE]
      stopifnot(ncol(scores)>0,all(vapply(scores,is.numeric,logical(1))))
      for(col in names(scores)) {
        values <- scores[[col]]
        stopifnot(!any(is.infinite(values)),sum(is.finite(values))>=2)
        result[[paste0('signifinder::',col)]] <- values
      }
      list(function_name=r$fun,arguments=r$args,columns=paste0('signifinder::',names(scores)),
           input_scale=if(name %in% c('EMT_Miow','APM_Wang','ECM_Chakravarthy')) 'log2(TPM+1)' else 'unlogged_TPM',
           scoring_cohort='ALL_TUMOR_SAMPLES_OF_THIS_CANCER_NOT_NORMAL_TISSUE')
    },warning=function(w) {warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)e)
    if(inherits(outcome,'error')) {
      failures[[name]] <- conditionMessage(outcome);cat('SCORE_ERROR',cancer,name,conditionMessage(outcome),'\n')
    } else {outcome$warnings<-warnings;method_records[[name]]<-outcome}
    gc(verbose=FALSE)
  }
  cat('SCORE',cancer,'SenMayo','\n');flush.console()
  param <- GSVA::ssgseaParam(log2(m+1),list(SenMayo=sen),normalize=TRUE,minSize=1,maxSize=Inf)
  score <- GSVA::gsva(param,verbose=FALSE,BPPARAM=BiocParallel::SerialParam())
  result[['SenMayo::GSVA_ssGSEA']] <- as.numeric(score['SenMayo',])
  method_records[['SenMayo']] <- list(method='GSVA_ssGSEA',normalize=TRUE,
      gene_set='SAUL_SEN_MAYO',gene_set_size=length(sen),genes_measured=sum(sen %in% rownames(m)),
      input_scale='log2(TPM+1)',scoring_cohort='ALL_TUMOR_SAMPLES_OF_THIS_CANCER_NOT_NORMAL_TISSUE')
  temp <- paste0(final,'.partial.gz');data.table::fwrite(result,temp,sep='\t',compress='gzip');stopifnot(file.rename(temp,final))
  rec <- list(target_host='149',cancer_id=cancer,input=input,input_md5=as.character(tools::md5sum(input)),output_md5=as.character(tools::md5sum(final)),senmayo_md5=as.character(tools::md5sum(sen_path)),
      samples=ncol(m),normal_or_untyped_samples_excluded=excluded,genes=nrow(m),score_columns=ncol(result)-2L,
      identity_rule='Restore dot-separated full TCGA barcode separators; no truncation; tumor sample type 01-09',
      package_version=as.character(packageVersion('signifinder')),GSVA_version=as.character(packageVersion('GSVA')),
      method_records=method_records,failures=failures,state=if(length(failures)) 'PARTIAL_REQUIRES_REPAIR' else 'COMPLETE')
  jsonlite::write_json(rec,receipt,auto_unbox=TRUE,pretty=TRUE,null='null')
  cat('COMMITTED',cancer,rec$state,rec$samples,rec$score_columns,'\n');flush.console()
  rm(m,result);gc(verbose=FALSE)
}
