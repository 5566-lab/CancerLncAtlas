#!/usr/bin/env Rscript
required <- c("Seurat", "Matrix", "rhdf5", "jsonlite", "data.table")
available <- vapply(required, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
print(available)
if (!all(available)) quit(status = 2L)
