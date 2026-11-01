args<-commandArgs(trailingOnly=TRUE)
repo<-if(length(args)>=1)normalizePath(args[[1]]) else normalizePath('.')
if(length(args)>1L).libPaths(c(args[-1],.libPaths()))
source(file.path(repo,'scripts/pathway_fusion_core.R'));assert_fusion_host();fusion_packages();set.seed(20261102)
ids<-paste0('P',1:90);genes<-paste0('G',1:100);latent<-rnorm(90)
blocks<-lapply(c('RNA','CNV','METHYLATION'),function(n){
 x<-matrix(rnorm(9000),100,90,dimnames=list(genes,ids));x[1:45,]<-x[1:45,]+rep(latent,each=45)*2;x
});names(blocks)<-c('RNA','CNV','METHYLATION')
for(n in c('PROTEIN','PHOSPHORYLATION'))blocks[[n]]<-matrix(rnorm(3600),40,90,dimnames=list(paste0(n,1:40),ids))
annotations<-lapply(c('PROTEIN','PHOSPHORYLATION'),function(n)setNames(lapply(1:40,function(i)genes[i]),paste0(n,1:40)))
names(annotations)<-c('PROTEIN','PHOSPHORYLATION');annotations$PROTEIN$PROTEIN1<-genes[c(1,16)]
sets<-list(OnlyA=genes[1:15],Both=genes[1:30],C=genes[31:60],D=genes[61:80])
fit<-fit_mogsa(blocks,sets,ids[1:60],nf=10,assay_feature_genes=annotations)
stopifnot(fit$status=='OK',fit$native_reconstruction_max_error<1e-8)
stopifnot(fit$annotation_by_block$PROTEIN['PROTEIN1','OnlyA']==0,
 fit$annotation_by_block$PROTEIN['PROTEIN1','Both']==1)
full<-project_mogsa(fit,blocks);stopifnot(all(full$available))
stopifnot(max(abs(Reduce('+',full$contribution)-full$score))<1e-8)
changed<-blocks;for(n in names(changed))changed[[n]][,61:90]<-changed[[n]][,61:90]+100
fit2<-fit_mogsa(changed,sets,ids[1:60],nf=10,assay_feature_genes=annotations)
stopifnot(isTRUE(all.equal(fit$original_scores,fit2$original_scores,tolerance=1e-12)))
masked<-blocks;masked$PROTEIN[1:16,90]<-NA;masked$PHOSPHORYLATION[1:20,90]<-NA
before<-lapply(masked,is.na);partial<-project_mogsa(fit,masked)
stopifnot(partial$available[90],partial$observed_fraction['PROTEIN',90]==.6,
 partial$observed_fraction['PHOSPHORYLATION',90]==.5,identical(before,lapply(masked,is.na)))
x<-unlist(lapply(names(masked),function(n)(masked[[n]][fit$features[[n]],90]-fit$means[[n]])/fit$norms[[n]]),use.names=FALSE)
obs<-is.finite(x);q<-fit$loading[obs,,drop=FALSE];a<-fit$weights[obs]
expected<-solve(crossprod(q*sqrt(a)),crossprod(q,a*x[obs]))
stopifnot(max(abs(partial$factors[90,]-as.numeric(expected)))<1e-9)
stopifnot(max(abs(partial$score[,1:89]-full$score[,1:89]))==0)
masked$PROTEIN[,90]<-NA;stopifnot(!project_mogsa(fit,masked)$available[90])
degenerate<-fit;degenerate$loading[,]<-0
masked<-blocks;masked$RNA[1,90]<-NA
stopifnot(!project_mogsa(degenerate,masked)$available[90])
child<-t(sapply(1:5,function(i)latent+rnorm(90,sd=.25)));dimnames(child)<-list(paste0('C',1:5),ids)
parent<-fit_parent_factors(child,list(Parent=rownames(child)),ids[1:60]);stopifnot('Parent'%in%names(parent$fits))
stopifnot(cor(as.numeric(project_parent_factors(parent,child)),latent)>.9)
cat('FIVE_ASSAY_R_CORE_TEST_PASS\n')
