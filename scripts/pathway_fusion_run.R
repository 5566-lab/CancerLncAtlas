# Run one descriptive cohort or one explicitly source-fitted prepared view.
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L)stop('Expected one verified JSON request')
request_path<-normalizePath(args[[1]],mustWork=TRUE)
request<-jsonlite::fromJSON(request_path,simplifyVector=TRUE)
if(!identical(request$target_host,'149'))stop('REQUEST_HOST')
.libPaths(c(unlist(request$r_libraries),.libPaths()))
source(request$core_script)
assert_fusion_host();fusion_packages()
meta<-jsonlite::fromJSON(file.path(request$input_dir,'COMPLETE.json'),simplifyVector=TRUE)
catalog<-jsonlite::fromJSON(request$catalog,simplifyVector=TRUE)
stopifnot(meta$status=='COMPLETE',meta$target_host=='149',meta$cancer_id==request$cancer_id)
blocks<-lapply(meta$blocks,function(b){
 con<-file(file.path(request$input_dir,b$file),'rb');on.exit(close(con))
 x<-readBin(con,what='double',n=prod(b$shape),size=8L,endian='little')
 stopifnot(length(x)==prod(b$shape));matrix(x,nrow=b$shape[[1]],ncol=b$shape[[2]],dimnames=list(b$feature_ids,meta$patient_ids))
})
stopifnot(setequal(names(blocks),c('RNA','CNV','METHYLATION','PROTEIN','PHOSPHORYLATION')))
fit_ids<-unlist(request$fit_patients)
stopifnot(length(fit_ids)>0,!anyDuplicated(fit_ids),all(fit_ids%in%meta$patient_ids))
out<-request$output_dir;dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_matrix<-function(x,name) {
 df<-data.frame(patient_id=colnames(x),t(x),check.names=FALSE)
 data.table::fwrite(df,file.path(out,paste0(name,'.tsv.gz')),sep='\t',na='NA')
}
summary<-list(target_host='149',cancer_id=request$cancer_id,scope=request$scope,fit_patients=fit_ids,
 patient_ids=meta$patient_ids,request=request,packages=as.list(fusion_packages()),
 biological_claim='Integrated RNA/protein/phosphorylation/gene-CNV/promoter-methylation state; not measured causal pathway activation')
if(identical(request$scope,'DESCRIPTIVE_COHORT_ONLY') && is.null(request$rna_reuse)) {
 cat('RNA_SCORING',request$cancer_id,'\n');flush.console()
 rna<-rna_consensus(blocks$RNA,catalog$gene_sets)
 write_matrix(rna$score,'RNA_CONSENSUS')
 for(name in names(rna$methods))write_matrix(rna$methods[[name]],paste0('RNA_',name))
 pairs<-combn(names(rna$methods),2,simplify=FALSE)
 agreement<-data.frame(pathway_id=rownames(rna$score),measured_genes=lengths(rna$genes),check.names=FALSE)
 for(pair in pairs)agreement[[paste(pair,collapse='_vs_')]]<-vapply(seq_len(nrow(rna$score)),function(i)
  suppressWarnings(cor(rna$methods[[pair[1]]][i,],rna$methods[[pair[2]]][i,],method='spearman',use='pairwise.complete.obs')),numeric(1))
 data.table::fwrite(agreement,file.path(out,'RNA_METHOD_AGREEMENT.tsv'),sep='\t',na='NA')
 summary$rna_formula<-rna$formula;summary$rna_pathways<-nrow(rna$score)
 rm(rna);gc()
}
if(!is.null(request$rna_reuse)) {
 stopifnot(identical(request$scope,'DESCRIPTIVE_COHORT_ONLY'))
 summary$rna_reuse<-request$rna_reuse
 summary$rna_formula<-request$rna_reuse$rna_formula;summary$rna_pathways<-request$rna_reuse$rna_pathways
}
summary$assays<-names(blocks)
summary$assay_measurement_types<-catalog$assay_measurement_types
summary$raw_values_imputed_by_pipeline<-FALSE
cat('MOGSA_FIT',request$cancer_id,length(fit_ids),'\n');flush.console()
fit<-fit_mogsa(blocks,catalog$gene_sets,fit_ids,nf=request$nf,assay_feature_genes=catalog$assay_feature_genes)
summary$mogsa_status<-fit$status
all_ids<-names(catalog$gene_sets)
child<-matrix(NA_real_,length(all_ids),length(meta$patient_ids),dimnames=list(all_ids,meta$patient_ids))
if(fit$status=='OK') {
 project<-project_mogsa(fit,blocks);child[rownames(project$score),]<-project$score
 summary$mogsa_fit_patients<-fit$fit_patients;summary$mogsa_nf<-fit$nf
 summary$mogsa_explained_variance<-fit$explained_variance
 summary$mogsa_native_projection_max_error<-fit$native_reconstruction_max_error
 summary$mogsa_features<-lapply(fit$features,length)
 summary$mogsa_available_patients<-sum(project$available)
 summary$projection_policy<-list(name='FROZEN_OBSERVED_FEATURE_WEIGHTED_LEAST_SQUARES',
  fixed_coverage_percentage_applied=FALSE,raw_measurements_imputed=FALSE,
  require_each_assay_observed=TRUE,require_numerical_full_rank=TRUE,
  method_counts=as.list(table(project$projection_method)),native_package_missing_data_predict=FALSE)
 projection_audit<-data.frame(patient_id=meta$patient_ids,available=project$available,
  method=unname(project$projection_method),reciprocal_condition=unname(project$reciprocal_condition),
  observed_weighted_residual_norm=unname(project$observed_weighted_residual_norm),
  t(project$observed_count),check.names=FALSE)
 data.table::fwrite(projection_audit,file.path(out,'MOGSA_PROJECTION_AUDIT.tsv'),sep='\t',na='NA')
 annotation_coverage<-data.frame(pathway_id=all_ids,check.names=FALSE)
 for(n in names(fit$annotation_by_block)) {
  counts<-colSums(fit$annotation_by_block[[n]])
  annotation_coverage[[paste0(n,'_annotated_measurements')]]<-as.integer(counts[match(all_ids,names(counts))])
 }
 data.table::fwrite(annotation_coverage,file.path(out,'PATHWAY_ASSAY_COVERAGE.tsv'),sep='\t',na='NA')
 for(name in names(project$contribution))write_matrix(project$contribution[[name]],paste0('MOGSA_CONTRIBUTION_',name))
 saveRDS(fit,file.path(out,'MOGSA_FIT.rds'),compress=FALSE)
} else {summary$mogsa_reason<-fit;summary$mogsa_available_patients<-0L}
write_matrix(child,'MOGSA_CHILDREN')
cat('LAVAAN_PARENTS',request$cancer_id,length(catalog$parent_members),'\n');flush.console()
parent<-fit_parent_factors(child,catalog$parent_members,fit_ids)
parent_scores<-project_parent_factors(parent,child)
write_matrix(parent_scores,'LAVAAN_PARENTS')
final<-child
# Registered parent targets use CFA only, with an explicit unavailable result
# when their one-factor model is unsupported. No silent substitute or average.
final[rownames(parent_scores),]<-parent_scores
write_matrix(final,'FINAL_PATHWAY_SCORES')
saveRDS(parent,file.path(out,'LAVAAN_FITS.rds'),compress=FALSE)
jsonlite::write_json(parent$audit,file.path(out,'LAVAAN_AUDIT.json'),auto_unbox=TRUE,pretty=TRUE,null='null',na='null')
summary$parent_policy<-'REGISTERED_PARENT_REPLACED_BY_SOURCE_FITTED_CFA; UNSUPPORTED_PARENT_IS_NA; OTHER_TARGETS_NATIVE_MOGSA'
summary$parents_requested<-length(parent$audit);summary$parents_supported<-length(parent$fits)
summary$parent_status_counts<-as.list(table(vapply(parent$audit,function(x)x$status,character(1))))
summary$final_measured_pathways<-sum(rowSums(is.finite(final))>0)
summary$status<-'COMPLETE'
jsonlite::write_json(summary,file.path(out,'SCORES_RECEIPT.json'),auto_unbox=TRUE,pretty=TRUE,null='null',na='null')
cat('SCORING_COMPLETE',request$cancer_id,'MEASURED',summary$final_measured_pathways,'PARENTS',summary$parents_supported,'\n')
