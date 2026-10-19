# Freeze both the TMM factor scale and the logCPM prior on training samples.
# edgeR calcNormFactors centers factors over its input columns. Ratios from a
# reference/sample pair recover the uncentered relative factor without using
# internal edgeR functions or centering over held-out samples.
check_count_matrix <- function(counts) {
  if (!is.matrix(counts) || !nrow(counts) || !ncol(counts) ||
      is.null(rownames(counts)) || anyDuplicated(rownames(counts)) ||
      is.null(colnames(counts)) || anyDuplicated(colnames(counts)) ||
      any(!is.finite(counts)) || any(counts < 0) || any(colSums(counts) <= 0))
    stop('Invalid count matrix or row/sample identity')
}

fit_frozen_tmm <- function(train, prior_count=0.25) {
  check_count_matrix(train)
  if (length(prior_count)!=1L || !is.finite(prior_count) || prior_count<=0) stop('Invalid prior count')
  # This is edgeR's default TMM reference rule, evaluated on training only.
  nonzero <- rowSums(train)>0
  reduced <- train[nonzero,,drop=FALSE]
  libs <- colSums(train)
  upper <- apply(reduced,2L,quantile,probs=.75)/libs
  ref <- if (median(upper)<1e-20) which.max(colSums(sqrt(reduced))) else which.min(abs(upper-mean(upper)))
  factors <- edgeR::calcNormFactors(train,method='TMM',refColumn=ref)
  if (any(!is.finite(factors)) || any(factors<=0)) stop('Invalid fitted TMM factors')
  list(format='FROZEN_TMM_STATE_V1', row_ids=rownames(train),
       reference_sample=colnames(train)[ref], reference_counts=train[,ref],
       reference_factor=unname(factors[ref]), reference_rule='TRAIN_UPPER_QUARTILE_CLOSEST_TO_MEAN_WITH_SPARSE_FALLBACK',
       train_samples=colnames(train), train_factors=setNames(factors,colnames(train)),
       mean_train_effective_library_size=mean(libs*factors), prior_count=prior_count)
}

apply_frozen_tmm <- function(counts, state) {
  check_count_matrix(counts)
  if (!identical(state$format,'FROZEN_TMM_STATE_V1') || !identical(rownames(counts),state$row_ids))
    stop('Frozen TMM row identity mismatch')
  factors <- vapply(seq_len(ncol(counts)),function(j) {
    pair <- cbind(reference=state$reference_counts,query=counts[,j])
    value <- edgeR::calcNormFactors(pair,method='TMM',refColumn=1L)
    unname(value[2L]/value[1L]*state$reference_factor)
  }, numeric(1))
  effective <- colSums(counts)*factors
  prior <- state$prior_count*effective/state$mean_train_effective_library_size
  values <- log2(sweep(sweep(counts,2L,prior,'+'),2L,effective+2*prior,'/')*1e6)
  if (any(!is.finite(values))) stop('Nonfinite frozen logCPM')
  list(logcpm=values,factors=setNames(factors,colnames(counts)),effective_library_size=effective)
}
