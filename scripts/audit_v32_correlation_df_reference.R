#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: audit_v32_correlation_df_reference.R OUTPUT_JSON")
}
output <- normalizePath(args[[1L]], mustWork = FALSE)
if (file.exists(output)) {
  stop(paste("refusing to overwrite", output))
}
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

evaluate_case <- function(case_id, x, y, design) {
  design <- as.matrix(design)
  qr_design <- qr(design)
  residual_x <- qr.resid(qr_design, x)
  residual_y <- qr.resid(qr_design, y)
  effect <- cor(residual_x, residual_y)
  design_rank <- qr_design$rank
  correlation_df <- length(x) - design_rank - 1L
  if (correlation_df <= 0L) {
    stop(paste("nonpositive correlation df for", case_id))
  }
  statistic <- effect * sqrt(correlation_df / (1 - effect^2))
  p_value <- 2 * pt(-abs(statistic), df = correlation_df)
  list(
    case_id = case_id,
    n_samples = length(x),
    design_columns = ncol(design),
    numerical_design_rank = design_rank,
    correlation_df = correlation_df,
    residual_correlation = effect,
    t_statistic = statistic,
    two_sided_p_value = p_value
  )
}

n <- 12L
index <- seq_len(n)
x <- sin(index / 2) + index / 20
y <- cos(index / 3) - index / 30
z <- index - mean(index)

cases <- list(
  evaluate_case("intercept_only", x, y, cbind(intercept = rep(1, n))),
  evaluate_case(
    "rank_deficient_duplicate_covariate",
    x,
    y,
    cbind(intercept = rep(1, n), z = z, z_duplicate = 2 * z)
  )
)

payload <- list(
  format = "CC_HHGT_V3_2_CORRELATION_DF_R_REFERENCE_V1",
  status = "PASS",
  formula = "n_samples - numerical_design_rank - 1",
  implementation = "base_R_qr_residuals_cor_t_distribution",
  cases = cases
)
write_json(payload, output, auto_unbox = TRUE, pretty = TRUE, digits = 17)
