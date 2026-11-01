# Paired raw-omics MOGSA with a frozen projection, and lavaan parent factors.
# Package computation is authoritative; the projection is checked against it.
assert_fusion_host <- function() stopifnot(Sys.info()[['nodename']]=='149')

fusion_packages <- function() {
  required <- c(mogsa='1.44.0',lavaan='0.7.3',GSVA='2.4.9',singscore='1.30.0')
  for (name in names(required)) {
    if (!requireNamespace(name,quietly=TRUE) || as.character(utils::packageVersion(name))!=required[[name]]) stop(paste('PACKAGE_VERSION',name))
  }
  invisible(required)
}

check_raw_blocks <- function(blocks) {
  stopifnot(is.list(blocks),length(blocks)>=2L,'RNA'%in%names(blocks),!is.null(names(blocks)),!anyDuplicated(names(blocks)))
  ids<-colnames(blocks[[1]])
  stopifnot(!is.null(ids),!anyDuplicated(ids))
  for (x in blocks) stopifnot(is.matrix(x),is.numeric(x),identical(colnames(x),ids),!is.null(rownames(x)),!anyDuplicated(rownames(x)),!any(is.infinite(x)))
  ids
}

rna_consensus <- function(rna,sets,min_genes=10L,max_genes=500L) {
  # Cohort-wide descriptive scores only. These never replace fold-local MOGSA labels.
  keep<-rowSums(!is.finite(rna))==0L & apply(rna,1L,stats::sd)>0
  rna<-rna[keep,,drop=FALSE]
  used<-lapply(sets,intersect,y=rownames(rna));size<-lengths(used)
  used<-used[size>=min_genes & lengths(sets)<=max_genes]
  if (!length(used)) stop('NO_RNA_GENE_SETS')
  ss<-GSVA::gsva(GSVA::ssgseaParam(rna,used,normalize=FALSE,minSize=min_genes,maxSize=max_genes),verbose=FALSE)
  gs<-GSVA::gsva(GSVA::gsvaParam(rna,used,kcdf='Gaussian',minSize=min_genes,maxSize=max_genes),verbose=FALSE)
  ranks<-singscore::rankGenes(rna)
  si<-t(vapply(used,function(g) singscore::simpleScore(ranks,upSet=g,centerScore=TRUE,knownDirection=TRUE)$TotalScore,numeric(ncol(rna))))
  colnames(si)<-colnames(rna)
  methods<-list(ssGSEA=ss,singscore=si,GSVA=gs)
  ordered<-names(used)
  methods<-lapply(methods,function(x)x[ordered,colnames(rna),drop=FALSE])
  standardized<-lapply(methods,function(x){s<-apply(x,1L,stats::sd);s[!is.finite(s)|s<=0]<-NA_real_;(x-rowMeans(x))/s})
  score<-Reduce('+',standardized)/length(standardized)
  list(score=score,methods=methods,genes=used,fit_scope='DESCRIPTIVE_COHORT_ONLY_NOT_TRAINING_LABEL',
       formula='RNA_consensus=(z_ssGSEA+z_singscore+z_GSVA)/3',ssgsea_normalize=FALSE)
}

fit_mogsa <- function(blocks,sets,fit_patients,nf=10L,min_genes=10L,max_genes=500L,min_patients=20L,assay_feature_genes=list()) {
  ids<-check_raw_blocks(blocks)
  stopifnot(!anyDuplicated(fit_patients),all(fit_patients%in%ids))
  present<-Reduce('&',lapply(blocks,function(x)colSums(is.finite(x))>=min_genes))
  fit_ids<-ids[ids%in%fit_patients & present]
  if(length(fit_ids)<min_patients)return(list(status='INSUFFICIENT_PAIRED_FIT_PATIENTS',fit_patients=fit_ids))
  means<-norms<-train<-list()
  for (name in names(blocks)) {
    x<-blocks[[name]][,fit_ids,drop=FALSE]
    keep<-rowSums(!is.finite(x))==0L
    x<-x[keep,,drop=FALSE];mu<-rowMeans(x);norm<-sqrt(rowSums((x-mu)^2))
    keep<-is.finite(norm)&norm>1e-12
    x<-x[keep,,drop=FALSE];mu<-mu[keep];norm<-norm[keep]
    if(nrow(x)<min_genes)return(list(status='INSUFFICIENT_COMPLETE_FEATURES',block=name,fit_patients=fit_ids))
    train[[name]]<-x;means[[name]]<-mu;norms[[name]]<-norm
  }
  # Keep the same reviewed gene-set universe; feature intersection is explicit.
  feature_genes<-lapply(names(train),function(n){
    ids<-rownames(train[[n]])
    if(is.null(assay_feature_genes[[n]]))return(setNames(lapply(ids,function(id)id),ids))
    m<-assay_feature_genes[[n]];stopifnot(all(ids%in%names(m)))
    setNames(lapply(ids,function(id)unique(unlist(m[[id]]))),ids)
  });names(feature_genes)<-names(train)
  measured<-unique(unlist(feature_genes,use.names=FALSE))
  eligible<-lengths(lapply(sets,intersect,y=measured))>=min_genes & lengths(sets)<=max_genes
  selected<-sets[eligible]
  if(!length(selected))return(list(status='NO_MOGSA_GENE_SETS',fit_patients=fit_ids))
  # prepSupMoa counts matches across all omics, including repeated gene IDs.
  # The biological gene-set size filter was already applied to unique genes.
  expanded<-lapply(selected,function(g){
    extra<-unlist(lapply(feature_genes[intersect(names(feature_genes),names(assay_feature_genes))],function(m)names(m)[vapply(m,function(ids)length(ids)>0L && all(ids%in%g),logical(1))]),use.names=FALSE)
    unique(c(g,extra))
  })
  sup<-mogsa::prepSupMoa(train,geneSets=expanded,minMatch=1L,maxMatch=Inf)
  for(n in names(sup))rownames(sup[[n]])<-rownames(train[[n]])
  moa<-mogsa::moa(train,proc.row='center_ssq1',w.data='lambda1',statis=FALSE)
  rank<-sum(moa@eig>max(moa@eig)*1e-12)
  k<-min(as.integer(nf),rank,length(fit_ids)-1L)
  if(k<1L)return(list(status='NO_MOGSA_COMPONENT',fit_patients=fit_ids))
  fitted<-mogsa::mogsa(x=moa,sup=sup,nf=k,statis=FALSE,ks.stat=FALSE)
  q<-as.matrix(moa@loading[,seq_len(k),drop=FALSE]);a<-moa@w.data
  coords<-fitted@sup@coord.comb
  core<-list(status='OK',fit_patients=fit_ids,fit_requested=fit_patients,means=means,norms=norms,
    features=lapply(train,rownames),loading=q,weights=a,coordinates=coords,coordinates_by_block=fitted@sup@coord.sep,
    feature_genes=feature_genes,annotation_by_block=sup,nf=k,explained_variance=sum(moa@tau[seq_len(k)]),sets=selected,package_version=as.character(utils::packageVersion('mogsa')),
    policy='SOURCE_PATIENT_FITTED_COMPLETE_FEATURES_NO_IMPUTATION',original_scores=fitted@sup@score)
  projected<-project_mogsa(core,train)
  err<-max(abs(projected$score[rownames(fitted@sup@score),fit_ids,drop=FALSE]-fitted@sup@score))
  if(!is.finite(err)||err>1e-7*max(1,max(abs(fitted@sup@score))))stop('MOGSA_NATIVE_PROJECTION_MISMATCH')
  core$native_reconstruction_max_error<-err
  core
}

# A frozen latent estimate from actually observed measurements, with no fixed
# coverage percentage. Every requested assay must be represented. The weighted
# observed design must have numerical full rank. Raw missing values stay missing.
project_mogsa <- function(fit,blocks) {
 ids<-check_raw_blocks(blocks)
 stopifnot(identical(fit$status,'OK'),identical(names(blocks),names(fit$features)))
 processed<-lapply(names(blocks),function(name){
  ix<-match(fit$features[[name]],rownames(blocks[[name]]));stopifnot(!anyNA(ix))
  (blocks[[name]][ix,,drop=FALSE]-fit$means[[name]])/fit$norms[[name]]
 })
 observed_count<-do.call(rbind,lapply(processed,function(x)colSums(is.finite(x))))
 dimnames(observed_count)<-list(names(blocks),ids)
 observed_fraction<-observed_count/vapply(processed,nrow,integer(1))
 x<-do.call(rbind,processed);full<-colSums(!is.finite(x))==0L
 each_assay_present<-apply(observed_count,2L,min)>0L
 factors<-matrix(NA_real_,length(ids),fit$nf,dimnames=list(ids,NULL))
 method<-setNames(rep('UNAVAILABLE_ENTIRE_ASSAY_MISSING',length(ids)),ids)
 condition<-setNames(rep(NA_real_,length(ids)),ids)
 residual_norm<-setNames(rep(NA_real_,length(ids)),ids)
 if(any(full)){
  factors[full,]<-t(x[,full,drop=FALSE]*fit$weights)%*%fit$loading
  method[full]<-'NATIVE_COMPLETE_PROJECTION';condition[full]<-1
 }
 gram_full<-crossprod(fit$loading*sqrt(fit$weights))
 for(i in which(each_assay_present & !full)){
  observed<-is.finite(x[,i]);miss<-!observed
  gram<-gram_full-crossprod(fit$loading[miss,,drop=FALSE]*sqrt(fit$weights[miss]))
  ev<-eigen(gram,symmetric=TRUE,only.values=TRUE)$values
  condition[i]<-min(ev)/max(ev)
  # Numerical rank, not a biological coverage or post-hoc accuracy threshold.
  tol<-max(sum(observed),fit$nf)*.Machine$double.eps*max(ev)
  if(any(!is.finite(ev))||min(ev)<=tol){method[i]<-'UNAVAILABLE_NUMERICALLY_RANK_DEFICIENT';next}
  rhs<-crossprod(fit$loading[observed,,drop=FALSE],fit$weights[observed]*x[observed,i])
  f<-tryCatch(as.numeric(solve(gram,rhs)),error=function(e)rep(NA_real_,fit$nf))
  if(any(!is.finite(f))){method[i]<-'UNAVAILABLE_LINEAR_SOLVE';next}
  factors[i,]<-f;method[i]<-'OBSERVED_FEATURE_WEIGHTED_LEAST_SQUARES'
  residual_norm[i]<-sqrt(sum(fit$weights[observed]*(x[observed,i]-as.numeric(fit$loading[observed,,drop=FALSE]%*%f))^2))
 }
 available<-rowSums(!is.finite(factors))==0L
 scores<-matrix(NA_real_,nrow(fit$coordinates),length(ids),dimnames=list(rownames(fit$coordinates),ids))
 contribution<-lapply(fit$coordinates_by_block,function(co)scores)
 if(any(available)){
  scores[,available]<-fit$coordinates%*%t(factors[available,,drop=FALSE])
  for(n in names(contribution))contribution[[n]][,available]<-fit$coordinates_by_block[[n]]%*%t(factors[available,,drop=FALSE])
 }
 list(score=scores,available=available,factors=factors,contribution=contribution,
  observed_count=observed_count,observed_fraction=observed_fraction,projection_method=method,
  reciprocal_condition=condition,observed_weighted_residual_norm=residual_norm,
  fixed_coverage_percentage_applied=FALSE,raw_measurements_imputed=FALSE,
  interpretation='Frozen latent projection; partial observations yield estimated scores, not newly measured assay values')
}

fit_parent_factors <- function(child_scores,parent_members,fit_patients,min_patients=30L,min_loading=.3,max_srmr=.08) {
  fits<-list();audit<-list()
  for (parent in names(parent_members)) {
    children<-sort(intersect(parent_members[[parent]],rownames(child_scores)))
    record<-list(parent_id=parent,requested_children=parent_members[[parent]],children=children,status='UNESTIMABLE')
    if(length(children)<3L){record$status<-'FEWER_THAN_THREE_CHILDREN';audit[[parent]]<-record;next}
    x<-t(child_scores[children,intersect(fit_patients,colnames(child_scores)),drop=FALSE]);x<-x[rowSums(!is.finite(x))==0L,,drop=FALSE]
    record$paired_fit_patients<-nrow(x)
    if(nrow(x)<max(min_patients,3L*ncol(x))){record$status<-'INSUFFICIENT_PARENT_FIT_PATIENTS';audit[[parent]]<-record;next}
    mu<-colMeans(x);sd<-apply(x,2L,stats::sd)
    if(any(!is.finite(sd)|sd<=1e-12)){record$status<-'CONSTANT_CHILD';audit[[parent]]<-record;next}
    x<-sweep(sweep(x,2L,mu,'-'),2L,sd,'/')
    if(qr(x)$rank<ncol(x)){record$status<-'RANK_DEFICIENT_CHILDREN';audit[[parent]]<-record;next}
    names_safe<-paste0('child_',seq_along(children));colnames(x)<-names_safe
    syntax<-paste('Parent =~',paste(names_safe,collapse=' + '))
    warnings<-character()
    fit<-tryCatch(withCallingHandlers(lavaan::cfa(syntax,data=as.data.frame(x),std.lv=TRUE,meanstructure=TRUE,estimator='MLR'),
      warning=function(e){warnings<<-c(warnings,conditionMessage(e));invokeRestart('muffleWarning')}),error=function(e)e)
    record$warnings<-unique(warnings)
    if(inherits(fit,'error')){record$status<-'CFA_ERROR';record$error<-conditionMessage(fit);audit[[parent]]<-record;next}
    if(!lavaan::lavInspect(fit,'converged')){record$status<-'CFA_NOT_CONVERGED';audit[[parent]]<-record;next}
    pe<-lavaan::parameterEstimates(fit,standardized=TRUE)
    load<-pe$std.all[pe$op=='=~'];variance<-pe$est[pe$op=='~~'&pe$lhs==pe$rhs]
    fm<-lavaan::fitMeasures(fit,c('df','cfi','rmsea','srmr'))
    record$fit_measures<-as.list(fm);record$standardized_loadings<-as.list(setNames(load,children))
    # Fix only factor sign, never edit measured child scores or truncate loadings.
    orientation<-if(load[1]<0)-1 else 1
    good<-all(is.finite(load))&&all(load*orientation>=min_loading)&&all(is.finite(variance)&variance>=0)&&is.finite(fm[['srmr']])&&fm[['srmr']]<=max_srmr
    if(!good){record$status<-'CFA_MODEL_NOT_SUPPORTED';audit[[parent]]<-record;next}
    record$status<-if(fm[['df']]==0)'OK_JUST_IDENTIFIED_NOT_GLOBAL_FIT_VALIDATED' else 'OK'
    record$fit_patients<-rownames(x);audit[[parent]]<-record
    fits[[parent]]<-list(fit=fit,children=children,mean=mu,sd=sd,names=names_safe,orientation=orientation,source_patients=rownames(x))
  }
  list(fits=fits,audit=audit,fit_requested=fit_patients,policy='SOURCE_PATIENT_FITTED_CFA_REGRESSION_SCORES')
}

project_parent_factors <- function(object,child_scores) {
  result<-matrix(NA_real_,length(object$audit),ncol(child_scores),dimnames=list(names(object$audit),colnames(child_scores)))
  for(parent in names(object$fits)) {
    m<-object$fits[[parent]];x<-t(child_scores[m$children,,drop=FALSE]);ok<-rowSums(!is.finite(x))==0L
    if(!any(ok))next
    x<-sweep(sweep(x[ok,,drop=FALSE],2L,m$mean,'-'),2L,m$sd,'/');colnames(x)<-m$names
    result[parent,ok]<-as.numeric(lavaan::lavPredict(m$fit,newdata=as.data.frame(x),method='regression',transform=FALSE))*m$orientation
  }
  result
}
