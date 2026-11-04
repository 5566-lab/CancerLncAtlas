# Numerical compatibility for the pinned Broad ssGSEA2.0 implementation.
# ES mode is deterministic: the unused single permutation must not cause errors.
load_cptac_ptm_core <- function(broad_source, output_dir) {
  original <- file.path(broad_source,'src','ssGSEA2.0.R')
  lines <- readLines(original,warn=FALSE)
  marker <- 'score <- function(max.ES, min.ES, RES, gaps, valleys, statistic){'
  stopifnot(sum(grepl(marker,lines,fixed=TRUE))==1L)
  guard <- paste0(marker,"\n",
    "                    if (!is.finite(max.ES) || !is.finite(min.ES) || any(!is.finite(RES))) {\n",
    "                        return(list(RES=rep(NA_real_,length(RES)), ES=NA_real_, arg.ES=NA_integer_))\n",
    "                    }")
  modified <- gsub(marker,guard,lines,fixed=TRUE)
  needle <- 'nperm = 1, correl.type'
  stopifnot(sum(grepl(needle,modified,fixed=TRUE))==1L)
  modified <- gsub(needle,'nperm = 0, correl.type',modified,fixed=TRUE)
  patched <- file.path(output_dir,'BROAD_SSGSEA2_ES_NA_GUARD.R')
  writeLines(modified,patched)
  assign('script.dir',broad_source,envir=.GlobalEnv)
  source(patched,local=.GlobalEnv)
  list(original_md5=unname(tools::md5sum(original)),patched_md5=unname(tools::md5sum(patched)),
       changes=c('Nonfinite_weighted_random_walk_returns_NA_ES_not_an_abort_or_imputed_zero',
                 'ES_only_calls_project.geneset_with_nperm_0;observed_ES_formula_unchanged'),
       inferential_pvalues_produced=FALSE)
}
