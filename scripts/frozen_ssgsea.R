# Preserve the native MSigDB/GSVA ssGSEA definition, with fitted quantities
# learned exclusively from the selected training tumors.
fit_frozen_ssgsea <- function(logcpm, train_samples, annotation, sets, mean_library, prior=.25) {
  if (!is.matrix(logcpm) || any(!is.finite(logcpm)) || anyDuplicated(rownames(logcpm)) ||
      anyDuplicated(colnames(logcpm)) || !length(train_samples) || anyDuplicated(train_samples) ||
      !all(train_samples %in% colnames(logcpm)) || !is.finite(mean_library) || mean_library<=0)
    stop('Invalid ssGSEA expression/training identity or frozen library scale')
  required <- c('ensembl_gene_id','gene_symbol','gene_type')
  if (!all(required %in% names(annotation))) stop('Incomplete protein-coding symbol annotation')
  ann <- as.data.frame(annotation[,required],stringsAsFactors=FALSE)
  ann$ensembl_gene_id <- sub('\\.[0-9]+$','',ann$ensembl_gene_id)
  if (anyNA(ann$ensembl_gene_id) || anyDuplicated(ann$ensembl_gene_id)) stop('Ambiguous stable gene annotation')
  ids <- sub('^GENE:','',rownames(logcpm))
  at <- match(ids,ann$ensembl_gene_id)
  symbols <- ann$gene_symbol[at]
  coding <- !is.na(at) & !is.na(ann$gene_type[at]) & ann$gene_type[at]=='protein_coding' & !is.na(symbols) & nzchar(symbols)
  # Algebraically undo the frozen edgeR logCPM prior. E is the sample's
  # effective library and M is the frozen mean training effective library:
  # 2^logCPM=(count+.25*E/M)/(E+.5*E/M)*1e6.
  cpm <- 2^logcpm[,train_samples,drop=FALSE]*(1+2*prior/mean_library)-prior*1e6/mean_library
  if (any(cpm < -1e-7)) stop('Expression does not match nonnegative frozen CPM')
  threshold <- max(3L,ceiling(.20*length(train_samples)))
  detection <- rowSums(cpm>=1)
  keep <- coding & detection>=threshold
  mapping <- data.frame(gene_id=rownames(logcpm),gene_symbol=ifelse(is.na(symbols),'',symbols),
                        training_cpm_ge1_n=detection,analysis_eligible=keep,stringsAsFactors=FALSE)
  selected <- mapping[keep,c('gene_id','gene_symbol'),drop=FALSE]
  detected_symbols <- sort(unique(selected$gene_symbol))
  mapped_sets <- lapply(sets,function(x) intersect(x,detected_symbols))
  coverage <- data.frame(pathway_id=names(sets),gene_count=lengths(sets),
    detected_gene_count=lengths(mapped_sets),stringsAsFactors=FALSE)
  coverage$coverage <- coverage$detected_gene_count/coverage$gene_count
  coverage$unavailable_reason <- ifelse(coverage$gene_count>500L,'PATHWAY_EXCEEDS_MAX_SIZE',
    ifelse(coverage$detected_gene_count<10L,'INSUFFICIENT_TRAIN_DETECTED_GENES',
      ifelse(coverage$detected_gene_count>=length(detected_symbols),'NO_BACKGROUND_GENES','')))
  coverage$analysis_eligible <- coverage$unavailable_reason==''
  state <- list(format='FROZEN_SSGSEA_STATE_V1',input_gene_ids=rownames(logcpm),symbol_mapping=selected,
    sets=mapped_sets[coverage$analysis_eligible],all_pathways=names(sets),train_samples=train_samples,
    training_detection_required_n=threshold,alpha=.25,min_size=10L,max_size=500L,
    training_score_range=NA_real_,coverage=coverage,gene_detection=mapping)
  raw <- apply_frozen_ssgsea(logcpm[,train_samples,drop=FALSE],state,scale=FALSE)
  eligible <- coverage$analysis_eligible
  if (any(eligible)) {
    observed <- raw[eligible,,drop=FALSE]
    if (any(!is.finite(observed))) stop('GSVA returned nonfinite training scores')
    state$training_score_range <- diff(range(observed))
    if (!is.finite(state$training_score_range) || state$training_score_range<=0) {
      state$coverage$analysis_eligible[eligible] <- FALSE
      state$coverage$unavailable_reason[eligible] <- 'DEGENERATE_TRAIN_SCORE_RANGE'
    }
  }
  state
}

apply_frozen_ssgsea <- function(logcpm,state,scale=TRUE) {
  if (!identical(state$format,'FROZEN_SSGSEA_STATE_V1') ||
      !identical(rownames(logcpm),state$input_gene_ids) || !ncol(logcpm) ||
      anyDuplicated(colnames(logcpm)) || any(!is.finite(logcpm))) stop('Frozen ssGSEA query identity/values differ')
  scores <- matrix(NA_real_,length(state$all_pathways),ncol(logcpm),
                   dimnames=list(state$all_pathways,colnames(logcpm)))
  if (length(state$sets)) {
    mapping <- state$symbol_mapping
    x <- logcpm[mapping$gene_id,,drop=FALSE]
    # Native producer averages logCPM values for duplicate symbols.
    sums <- rowsum(x,group=mapping$gene_symbol,reorder=FALSE)
    means <- sums/as.numeric(table(mapping$gene_symbol)[rownames(sums)])
    parameter <- GSVA::ssgseaParam(exprData=means,geneSets=state$sets,minSize=state$min_size,
      maxSize=state$max_size,alpha=state$alpha,normalize=FALSE)
    raw <- GSVA::gsva(parameter,verbose=FALSE,BPPARAM=BiocParallel::SerialParam())
    if (any(!is.finite(raw))) stop('GSVA returned nonfinite query scores')
    scores[rownames(raw),] <- raw
  }
  if (scale) {
    if (is.finite(state$training_score_range) && state$training_score_range>0) scores <- scores/state$training_score_range
    else scores[,] <- NA_real_
    scores[!state$coverage$analysis_eligible,] <- NA_real_
  }
  scores
}
