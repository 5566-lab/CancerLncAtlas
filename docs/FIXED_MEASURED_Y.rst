Fixed patient Y for pathway and State targets
============================================

The materialized Y package has 524 top-level entries: 476 pathways and 48 States.
Each entry owns a set of unique, source-specific molecular observations. The
State hierarchy retains 91 cellular-composition identities and 60 SBS signatures
under two collection entries. A collection is a vector of distinct observations.

Six cohort files contain 1,910 development patients from TCGA/CPTAC BRCA, LUAD
and COAD. The original reference/query assignments are preserved. Per cohort,
``Y_VALUES.parquet`` and ``Y_OBSERVED.parquet`` have the same patient and
observation axes. ``OBSERVATIONS.tsv.gz`` retains methods, units, molecular
identities and supervision eligibility. ``TARGET_OBSERVATIONS.tsv.gz`` binds
these columns to biological entries. Only bound target observations enter the
portable package; unrelated source-pool readouts remain in the source registry.

``TARGET_APPLICABILITY.parquet`` records patient-specific conditions separately.
The supervision mask is observed AND source-eligible AND applicability-equals-1.
An unknown condition remains unknown. Observed zero remains observed. The
accessor ``cla_engine.fixed_y_bundle.load_fixed_target`` returns original values,
mask, applicability and metadata. Its default view exposes measured data;
``for_supervision=True`` applies eligibility and condition masking.

All 476 pathway entries have some measured observations. Under the frozen event
evidence, 335 have nonempty supervision masks and 141 retain unknown event
conditions. States with measurements number 45 in each TCGA cancer and 33 in
each CPTAC cancer. Three conceptual State entries have no independent patient
readout: oncogenic dedifferentiation, epithelial/mesenchymal programs, and
cancer-associated extracellular-matrix programs.

The task remains cancer x lncRNA x exact pathway. Y uses patient molecular
measurements, without lncRNA association or threshold-based label construction.
The M0 comparison keeps Y dimensions, masks and patient folds identical across
lncRNA, mRNA and combined-input models, with the agreed clinical and independent
purity covariates. Target-derived non-RNA input layers are excluded; RNA-score
tasks are interpreted as reconstruction. Pathway/State weights remain 1.0/0.5.
