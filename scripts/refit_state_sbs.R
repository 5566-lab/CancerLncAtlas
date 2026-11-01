stopifnot(Sys.info()[['nodename']]=='149')
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L)stop('Usage: Rscript refit_state_sbs.R OUTPUT_ROOT R_LIBRARY')
out <- args[[1L]]
stopifnot(file.exists(file.path(out,'LAUNCH_PREFLIGHT.json')),file.exists(file.path(out,'SBS96_COUNTS_RECEIPT.json')))
if(file.exists(file.path(out,'SBS_STATE_RECEIPT.json')))stop('Existing SBS fit; use a new output root for revisions')
.libPaths(c(args[[2L]],.libPaths()))
stopifnot(as.character(packageVersion('MutationalPatterns'))=='3.7.1')
suppressPackageStartupMessages({library(MutationalPatterns);library(data.table);library(BSgenome.Hsapiens.UCSC.hg19)})
data.table::setDTthreads(2)
check <- fread(file.path(out,'SBS96_REFERENCE_CHECK.tsv'))
chrom <- ifelse(startsWith(check$chromosome,'chr'),check$chromosome,paste0('chr',check$chromosome))
genome <- BSgenome.Hsapiens.UCSC.hg19::Hsapiens
observed <- as.character(Biostrings::getSeq(genome,names=chrom,start=check$position-1L,end=check$position+1L))
stopifnot(identical(unname(observed),check$context))
cat('REFERENCE_CHECK_PASS',length(observed),'\n');flush.console()
raw <- fread(file.path(out,'SBS96_MC3_COUNTS.tsv.gz'))
contexts <- raw[[1]];mat <- as.matrix(raw[,-1,with=FALSE]);rownames(mat)<-contexts;storage.mode(mat)<-'double'
total <- colSums(mat)
sigs <- MutationalPatterns::get_known_signatures(muttype='snv',source='COSMIC_v3.2',genome='GRCh37',incl_poss_artifacts=FALSE)
reference_file <- system.file('extdata/signatures/snv_COSMIC_v3.2_reference_GRCh37.txt',package='MutationalPatterns')
reference_meta <- fread(reference_file)
stopifnot(isTRUE(all.equal(unname(sigs),unname(as.matrix(reference_meta[,colnames(sigs),with=FALSE])),tolerance=1e-10)))
rownames(sigs) <- paste0(substr(reference_meta$Subtype,1,1),'[',reference_meta$Type,']',substr(reference_meta$Subtype,3,3))
stopifnot(setequal(rownames(mat),rownames(sigs)),all(c('SBS2','SBS13') %in% colnames(sigs)))
mat <- mat[rownames(sigs),,drop=FALSE]
take <- total>=50
cat('REFIT',sum(take),'PATIENTS',ncol(sigs),'SIGNATURES','\n');flush.console()
fit <- MutationalPatterns::fit_to_signatures(mat[,take,drop=FALSE],sigs)
contributions <- fit$contribution
reconstructed <- sigs %*% contributions
cosine <- colSums(mat[,take,drop=FALSE]*reconstructed)/sqrt(colSums(mat[,take,drop=FALSE]^2)*colSums(reconstructed^2))
stopifnot(all(is.finite(contributions)),all(contributions>=0),all(is.finite(cosine)))
fraction <- sweep(contributions,2,colSums(contributions),'/')
stopifnot(all(abs(colSums(fraction)-1)<1e-8))
qc <- data.table(patient_id=colnames(mat),snv_count=as.integer(total),fit_eligible=take,reconstruction_cosine=NA_real_,state_available=FALSE)
qc$reconstruction_cosine[take]<-cosine
qc$state_available[take]<-cosine>=0.8
scores <- matrix(NA_real_,nrow=ncol(mat),ncol=ncol(sigs),dimnames=list(colnames(mat),paste0('SBS::',colnames(sigs))))
scores[colnames(fraction),]<-t(fraction)
scores[!qc$state_available,]<-NA_real_
fwrite(as.data.table(scores,keep.rownames='patient_id'),file.path(out,'SBS_STATE_FRACTIONS.tsv.gz'),sep='\t',compress='gzip')
fwrite(as.data.table(t(contributions),keep.rownames='patient_id'),file.path(out,'SBS_FITTED_COUNTS.tsv.gz'),sep='\t',compress='gzip')
fwrite(qc,file.path(out,'SBS_FIT_QC.tsv'),sep='\t')
fwrite(as.data.table(sigs,keep.rownames='mutation_type'),file.path(out,'COSMIC_V3_2_GRCH37_SBS_REFERENCE.tsv'),sep='\t')
record <- list(target_host='149',package='MutationalPatterns',version=as.character(packageVersion('MutationalPatterns')),
  reference='COSMIC_v3.2_GRCh37',artifact_signatures_included=FALSE,method='fit_to_signatures_NNLS',
  primary_value='relative_contribution_fraction',reference_contexts_independently_checked=length(observed),
  minimum_snv_count=50L,minimum_reconstruction_cosine=0.8,quality_policy='Below-threshold or unsequenced patients remain missing, never zero exposure',
  patients_with_counts=ncol(mat),patients_fitted=sum(take),patients_state_available=sum(qc$state_available),
  targets=paste0('SBS::',colnames(sigs)),apobec_components=c('SBS::SBS2','SBS::SBS13'),
  scope='Exome-based signature estimates from filtered MC3 SNVs; not direct APOBEC enzyme activity; no bootstrap confidence interval claimed',state='COMPLETE')
jsonlite::write_json(record,file.path(out,'SBS_STATE_RECEIPT.json'),auto_unbox=TRUE,pretty=TRUE)
cat('COMMITTED_SBS',sum(qc$state_available),'\n')
