stopifnot(Sys.info()[['nodename']]=='149')
args<-commandArgs(trailingOnly=TRUE);root<-args[1]
.libPaths(c('/dsk2/user/dengsc/cla_cptac_integration_20261104_r1/R_library','/dsk2/user/dengsc/cla_state_expansion_20261101_r1/R_library','/dsk2/user/dengsc/R/x86_64-pc-linux-gnu-library/4.5',.libPaths()))
suppressPackageStartupMessages(library(e1071))
B<-as.matrix(read.csv(file.path(root,'REFERENCE.csv'),row.names=1,check.names=FALSE))
M<-as.matrix(read.csv(file.path(root,'MIXTURES.csv'),row.names=1,check.names=FALSE))
X<-(B-mean(B))/sd(as.vector(B));out<-list()
for(i in seq_len(ncol(M))){
 y<-(M[,i]-mean(M[,i]))/sd(M[,i]);best<-Inf;ans<-NULL
 for(nu in c(.25,.5,.75)){
  fit<-svm(X,y,type='nu-regression',kernel='linear',nu=nu,cost=1,scale=FALSE,tolerance=.001,shrinking=TRUE)
  w<-as.numeric(t(fit$coefs)%*%fit$SV);w[w<0]<-0
  if(sum(w)<=1e-12)next
  w<-w/sum(w);rmse<-sqrt(mean((as.vector(X%*%w)-y)^2))
  if(rmse<best){best<-rmse;ans<-w}
 }
 out[[i]]<-ans
}
result<-do.call(rbind,out);rownames(result)<-colnames(M);colnames(result)<-colnames(B)
write.csv(result,file.path(root,'R_FRACTIONS.csv'))
writeLines(as.character(packageVersion('e1071')),file.path(root,'E1071_VERSION.txt'))
