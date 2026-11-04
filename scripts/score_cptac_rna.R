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
sen_path <- read_option('--senmayo-json');r_library <- read_option('--r-library');resource_path <- read_option('--resources')
if(any(vapply(list(out,input_root,sen_path,r_library),is.null,logical(1))))stop('Required: --output-root --input-root --senmayo-json --r-library')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(strsplit(r_library,':',fixed=TRUE)[[1]],.libPaths()))
suppressPackageStartupMessages({library(signifinder);library(GSVA);library(data.table);library(jsonlite);library(SummarizedExperiment)})
stopifnot(as.character(packageVersion('signifinder'))=='1.9.5')
data.table::setDTthreads(2L);BiocParallel::register(BiocParallel::SerialParam());set.seed(20261101)
mode <- read_option('--cancer-type',read_option('--cancer_type','all'))
available <- basename(list.dirs(file.path(input_root,'matrices'),recursive=FALSE))
if(mode=='all') cancers <- available else if(mode=='list') {
  requested <- read_option('--cancer-list',read_option('--cancer_list',''))
  cancers <- sort(unique(toupper(trimws(strsplit(requested,',',fixed=TRUE)[[1L]]))))
  if(!length(cancers)||any(!nzchar(cancers))||any(!cancers %in% available))stop('Invalid/empty cancer list; use comma-separated IDs present in input-root')
} else stop('cancer-type must be all or list')
stopifnot(length(cancers)>0L,file.exists(sen_path))
jsonlite::write_json(list(target_host='149',action='CPTAC_RNA_STATE_AND_PATHWAY_SCORING',cancer_type=mode,cancers=cancers,input_root=input_root),file.path(out,paste0('RNA_SCORING_PREFLIGHT_',paste(sort(cancers),collapse='_'),'.json')),auto_unbox=TRUE,pretty=TRUE)
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
write.table(metadata,file.path(out,paste0('SIGNIFINDER_SIGNATURE_TABLE_',paste(sort(cancers),collapse='_'),'.tsv')),sep='\t',row.names=FALSE,quote=TRUE)
sen <- jsonlite::read_json(sen_path,simplifyVector=TRUE)$SAUL_SEN_MAYO$geneSymbols
stopifnot(length(sen)>=120L)
resources <- jsonlite::read_json(resource_path,simplifyVector=TRUE)
for(k in names(resources$md5))stopifnot(identical(unname(tools::md5sum(resources[[k]])),resources$md5[[k]]))
read_sets <- function(p) {
 f <- strsplit(readLines(p,warn=FALSE),'\t',fixed=TRUE)
 setNames(lapply(f,function(x)unique(x[-c(1,2)])),vapply(f,'[[',character(1),1L))
}
gene_sets <- read_sets(resources$pathway_gmt)
historical <- jsonlite::read_json(resources$historical_networks,simplifyVector=TRUE)
folder <- file.path(out,'rna_scores');dir.create(folder,showWarnings=FALSE)
pathway_folder <- file.path(out,'pathway_scores');dir.create(pathway_folder,showWarnings=FALSE)

for(cancer in sort(unique(cancers))) {
  final <- file.path(folder,paste0(cancer,'.tsv.gz'));receipt <- file.path(folder,paste0(cancer,'.json'))
  if(file.exists(receipt)||file.exists(final))stop('Use a new output-root; existing cohort output')
  input <- file.path(input_root,'matrices',cancer,'RNA_TUMOR_SYMBOL_TPM.tsv.gz')
  cat('START_CANCER',cancer,'\n');flush.console()
  dat <- data.table::fread(input,check.names=FALSE);m <- as.matrix(dat[,-1,with=FALSE]);rownames(m)<-dat[[1]];rm(dat)
  storage.mode(m)<-'double'
  stopifnot(!anyDuplicated(rownames(m)),!anyDuplicated(colnames(m)),all(is.finite(m)),all(m>=0),
            all(startsWith(colnames(m),paste0('CPTAC:',cancer,':'))),all(endsWith(colnames(m),'::T')))
  excluded<-0L
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

  pc <- data.table::fread(file.path(input_root,'matrices',cancer,'PROTEIN_CODING_SYMBOLS.tsv'))[[1]]
  expr <- log2(m[intersect(rownames(m),pc),,drop=FALSE]+1)
  cat('SCORE',cancer,'Pathway_ssGSEA','\n');flush.console()
  coverage <- data.table(pathway_id=names(gene_sets),gene_count=lengths(gene_sets),
       observed=vapply(gene_sets,function(g)sum(g %in% rownames(expr)),integer(1)))
  coverage[,usable:=observed>=10L & gene_count<=500L]
  gs <- gene_sets[coverage$usable]
  pw <- GSVA::gsva(GSVA::ssgseaParam(expr,gs,minSize=10,maxSize=500,alpha=.25,normalize=TRUE),
                   verbose=FALSE,BPPARAM=BiocParallel::SerialParam())
  data.table::fwrite(data.table::as.data.table(pw,keep.rownames='pathway_id'),
     file.path(pathway_folder,paste0(cancer,'.tsv.gz')),sep='\t',compress='gzip')
  data.table::fwrite(coverage,file.path(pathway_folder,paste0(cancer,'_coverage.tsv')),sep='\t')
  method_records[['RNA_Pathway']] <- list(method='GSVA_ssGSEA',input_scale='log2(TPM+1)',alpha=.25,
      normalization='Within_this_CPTAC_cancer_score_range;not_calibrated_to_TCGA_logCPM',min_detected=10,max_set_size=500,
      pathway_gmt_md5=unname(tools::md5sum(resources$pathway_gmt)))
  # Hallmark State labels use the historical 50-set scoring context.
  hs <- historical$hallmark
  if(is.null(hs))stop('Missing frozen Hallmark State sets')
  hs <- lapply(hs,as.character)
  hm <- GSVA::gsva(GSVA::ssgseaParam(expr,hs,minSize=1,maxSize=Inf,alpha=.25,normalize=TRUE),
                   verbose=FALSE,BPPARAM=BiocParallel::SerialParam())
  for(k in c('HALLMARK_OXIDATIVE_PHOSPHORYLATION','HALLMARK_DNA_REPAIR'))
      result[[paste0('HALLMARK::',k)]] <- as.numeric(hm[k,])
  method_records[['Hallmark_State']] <- list(method='GSVA_ssGSEA',scoring_context='Historical_50_Hallmark_sets',normalize=TRUE)
  result[['CYT::mean_log_GZMA_PRF1']] <- colMeans(expr[c('GZMA','PRF1'),,drop=FALSE])
  method_records[['CYT']]<-list(method='Mean_log2_TPM_plus1_GZMA_PRF1')
  cat('SCORE',cancer,'MCPcounter_IPS_ESTIMATE','\n');flush.console()
  mc <- MCPcounter::MCPcounter.estimate(expr,featuresType='HUGO_symbols',
       probesets=NULL,genes=read.table(resources$mcp,header=TRUE,sep='\t',stringsAsFactors=FALSE,check.names=FALSE))
  for(k in rownames(mc))result[[paste0('MCPcounter::',k)]]<-as.numeric(mc[k,])
  ips <- hacksig::hack_immunophenoscore(expr,extract='all')
  stopifnot(identical(as.character(ips$sample_id),colnames(expr)))
  for(k in setdiff(names(ips),'sample_id'))result[[paste0('IPS::',k)]]<-ips[[k]]
  est <- hacksig::hack_estimate(expr)
  stopifnot(identical(as.character(est$sample_id),colnames(expr)))
  estimate_names <- c(immune_score='ImmuneScore',stroma_score='StromalScore',estimate_score='ESTIMATEScore',purity_score='TumorPurity')
  for(k in names(estimate_names)) {
     if(!k %in% names(est))stop(paste('Unexpected ESTIMATE columns',paste(names(est),collapse=',')))
     result[[paste0('ESTIMATE::',estimate_names[[k]])]]<-est[[k]]
  }
  method_records[['MCPcounter']]<-list(method='MCPcounter.estimate',version=as.character(packageVersion('MCPcounter')),
       input_scale='log2(TPM+1)',source_genes_md5=unname(tools::md5sum(resources$mcp)))
  method_records[['IPS_ESTIMATE']]<-list(method='hacksig_original',version=as.character(packageVersion('hacksig')),
       input_scale='log2(TPM+1)',source_semantics='Recalculated_native_scores;not_TCGA_published_calibration')
  # The routed State TF/pathway programs use the existing full-resource weighted-z implementation.
  cat('SCORE',cancer,'Routed_programs','\n');flush.console()
  net <- data.table::fread(resources$progeny)
  net <- net[,.(gene=as.character(gene),state_id=paste0('PROGENY:',pathway),weight=as.numeric(weight))]
  dn <- data.table::fread(resources$dorothea)
  net <- rbind(net,dn[,.(gene=as.character(target),state_id=paste0('DOROTHEA:',tf),weight=as.numeric(mor))])
  net <- net[is.finite(weight)&gene!='',.(weight=mean(weight)),by=.(gene,state_id)]
  # Existing routed programs include all RNA genes, not only protein-coding RNA.
  e <- log2(m+1);rs<-apply(e,1,sd);keep<-is.finite(rs)&rs>0
  e<-(e[keep,,drop=FALSE]-rowMeans(e[keep,,drop=FALSE]))/rs[keep]
  net<-net[gene %in% rownames(e)];counts<-net[,.(detected=.N),by=state_id]
  ids<-sort(counts[detected>=5,state_id]);net<-net[state_id %in% ids]
  wt<-Matrix::sparseMatrix(i=match(net$gene,rownames(e)),j=match(net$state_id,ids),x=net$weight,dims=c(nrow(e),length(ids)))
  scores<-as.matrix(Matrix::crossprod(wt,e))/sqrt(Matrix::colSums(wt^2))
  sdv<-apply(scores,1,sd);sdv[!is.finite(sdv)|sdv==0]<-1
  scores<-(scores-rowMeans(scores))/sdv
  for(i in seq_along(ids))result[[paste0('native_program::',ids[i])]]<-as.numeric(scores[i,])
  method_records[['Routed_programs']]<-list(method='Existing_R07_full_resource_weighted_gene_z_then_program_z',
       gene_sd_ddof=1,program_sd_ddof=1,min_detected=5,input_scale='log2(TPM+1)',
       reference_cohort=paste0('CPTAC:',cancer),progeny_md5=unname(tools::md5sum(resources$progeny)),
       dorothea_md5=unname(tools::md5sum(resources$dorothea)))

  temp <- paste0(final,'.partial.gz');data.table::fwrite(result,temp,sep='\t',compress='gzip');stopifnot(file.rename(temp,final))
  rec <- list(target_host='149',cancer_id=cancer,input=input,input_md5=as.character(tools::md5sum(input)),output_md5=as.character(tools::md5sum(final)),senmayo_md5=as.character(tools::md5sum(sen_path)),
      samples=ncol(m),normal_or_untyped_samples_excluded=excluded,genes=nrow(m),score_columns=ncol(result)-2L,
      identity_rule='Published CPTAC tumor case authority; cohort-qualified patient and sample IDs',
      package_version=as.character(packageVersion('signifinder')),GSVA_version=as.character(packageVersion('GSVA')),
      method_records=method_records,failures=failures,state=if(length(failures)) 'PARTIAL_REQUIRES_REPAIR' else 'COMPLETE')
  jsonlite::write_json(rec,receipt,auto_unbox=TRUE,pretty=TRUE,null='null')
  cat('COMMITTED',cancer,rec$state,rec$samples,rec$score_columns,'\n');flush.console()
  rm(m,result);gc(verbose=FALSE)
}
