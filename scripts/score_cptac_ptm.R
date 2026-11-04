options(stringsAsFactors=FALSE)
stopifnot(Sys.info()[['nodename']]=='149')
a<-commandArgs(trailingOnly=TRUE);opt<-function(k){i<-match(k,a);if(is.na(i)||i==length(a))stop(k);a[i+1]}
.libPaths(c(strsplit(opt('--r-library'),':',fixed=TRUE)[[1]],.libPaths()))
input<-opt('--input');out<-opt('--output');gmt<-opt('--gmt');script.dir<-opt('--broad-source')
if(dir.exists(out)&&length(list.files(out)))stop('Use a new output directory')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
for(p in c('pacman','gtools','verification','doParallel','foreach','magrittr','jsonlite'))
 if(!requireNamespace(p,quietly=TRUE))stop(paste('Install pinned dependencies before scoring:',p))
jsonlite::write_json(list(target_host='149',action='CPTAC_SITE_LEVEL_PTM_SEA'),file.path(out,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE)
script_file<-sub('^--file=','',commandArgs()[grepl('^--file=',commandArgs())][1])
source(file.path(dirname(script_file),'cptac_ptm_compat.R'))
numerical_patch<-load_cptac_ptm_core(script.dir,out)
set.seed(20261104)
invisible(ssGSEA2(input.ds=input,output.prefix=file.path(out,'PTMSEA'),gene.set.databases=gmt,
 sample.norm.type='none',weight=1,statistic='area.under.RES',output.score.type='ES',
 nperm=0,combine.mode='combine.off',min.overlap=5,correl.type='rank',global.fdr=FALSE,
 extended.output=TRUE,par=FALSE,export.signat.gct=FALSE,param.file=TRUE,log.file=file.path(out,'ptmsea.log')))
jsonlite::write_json(list(status='COMPLETE',target_host='149',method='Broad_PTM_SEA_ES',
 source_commit='cae7bed2c357826c0d887ca528c0c9b272705006',PTMsigDB='2.0.0_PATH_KINASE',
 input_md5=unname(tools::md5sum(input)),gmt_md5=unname(tools::md5sum(gmt)),
 weight=1,sample_norm='none',correl_type='rank',statistic='area.under.RES',min_overlap=5,
 input_scale='Per_site_log2_abundance_minus_frozen_tumor_cohort_site_median',
 nperm=0,numerical_patch=numerical_patch,permutation_note='ES_only;no_inferential_pvalues_or_NES_are_claimed',
 interpretation='Directional_site_signature_relative_to_CPTAC_cohort;not_absolute_or_causal_activation'),
 file.path(out,'COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
