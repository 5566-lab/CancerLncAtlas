options(stringsAsFactors=FALSE)
stopifnot(Sys.info()[['nodename']]=='149')
args <- commandArgs(trailingOnly=TRUE)
if(length(args)<2L)stop('Usage: Rscript export_state_signature_definitions.R OUTPUT LIBRARY [LIBRARY ...]')
out<-args[[1L]]
if(dir.exists(out)&&length(list.files(out,all.files=TRUE,no..=TRUE)))stop('Use an empty output directory')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(args[-1L],.libPaths()))
suppressPackageStartupMessages({library(signifinder);library(hacksig);library(xCell);library(GSEABase)})
stopifnot(as.character(packageVersion('signifinder'))=='1.9.5',as.character(packageVersion('hacksig'))=='0.2.0')
jsonlite::write_json(list(target_host='149',action='EXPORT_EXACT_INSTALLED_STATE_DEFINITIONS'),file.path(out,'LAUNCH_PREFLIGHT.json'),auto_unbox=TRUE)
export_frames<-function(package,filename,predicate,pretty=TRUE) {
 ns<-asNamespace(package);result<-list()
 for(name in ls(ns,all.names=TRUE)){obj<-get(name,ns);if(is.data.frame(obj)&&predicate(obj))result[[name]]<-obj}
 if(package=='hacksig')jsonlite::write_json(result,file.path(out,filename),auto_unbox=TRUE,pretty=pretty,digits=NA) else jsonlite::write_json(result,file.path(out,filename),auto_unbox=TRUE,pretty=pretty,na='null',digits=NA)
}
export_frames('signifinder','SIGNIFINDER_EXACT_DEFINITIONS.json',function(x)'SYMBOL'%in%names(x))
export_frames('hacksig','HACKSIG_EXACT_DEFINITIONS.json',function(x)'signature_id'%in%names(x),pretty=FALSE)
e<-new.env();data('xCell.data',package='xCell',envir=e);x<-get('xCell.data',e)
x$signatures<-GSEABase::geneIds(x$signatures)
jsonlite::write_json(x,file.path(out,'XCELL_ORIGINAL_DATA.json'),auto_unbox=TRUE,pretty=FALSE,digits=NA)
jsonlite::write_json(list(target_host='149',packages=lapply(c('signifinder','hacksig','xCell'),function(p)list(package=p,version=as.character(packageVersion(p))))),file.path(out,'DEFINITION_EXPORT_RECEIPT.json'),auto_unbox=TRUE,pretty=TRUE)
