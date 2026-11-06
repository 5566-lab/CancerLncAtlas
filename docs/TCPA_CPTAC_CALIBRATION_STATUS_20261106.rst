TCPA--CPTAC common-panel calibration status
=========================================

The formal model still uses assay/cohort-native CPTAC supervision. Separate
cross-platform calibration diagnostics now exist for retrospective BRCA and
both OV centers. They have not been promoted to a universal training Y.

Implemented calculation
-----------------------

Within one cancer and one retrospective center, use a fixed named common
protein panel after unambiguous gene/antibody matching. Keep the five frozen
patient folds. In each fold, calculate each assay's gene mean and standard
deviation using training patients only, then average observed standardized
members of the same pathway panel with equal nonnegative weights. Export each
patient's observed-member count. Missing measurements remain missing.

Fit MS_score = intercept + slope * RPPA_score on training pairs only and evaluate
on the held-out patients. Keep all pathway rows, failed fits and negative R2.
Report the mean of the five fold R2 values per pathway. Do not pool differently
referenced fold scores into one Y. These are common-panel protein-abundance
scores, not absolute pathway activation. Published inputs already include
upstream processing; this is not validation from entirely raw held-out assays.

The PDC audit checks agreement of sample-vial metadata within the specified
study version. Processed matrix column-to-aliquot lineage remains unverified;
same-vial metadata is not proof of the same lysate. BRCA excludes two different-
vial matches, retaining 102 patients. Each eligible patient is held out once.

Actual descriptive results
--------------------------

For pathways with at least three train-usable common proteins in all folds and
finite R2 in all five folds (a reporting stratum, not a new admission threshold):

========== ======== ========== ============= ================ ==================
Cohort     Patients Pathways   Positive mean Positive all five Median mean-fold R2
========== ======== ========== ============= ================ ==================
BRCA-BI    102      584        394           66               0.044583
OV-JHU     89       525        370           99               0.093649
OV-PNNL    62       479        216           33              -0.035699
========== ======== ========== ============= ================ ==================

Common protein panels contain 174, 152 and 136 train-usable genes respectively.
Panels and evaluable pathways differ, so these medians are not a paired test
that one center calibrates better. OV centers share 26 patients (125 unique
patients across the two); they are not independent cohorts and counts must not
be added as independent evidence. Positive R2 in all five folds is descriptive,
not a production acceptance rule or a multiple-testing claim.

Both OV results were independently reconstructed from original assay inputs,
including means/scales, missing masks, usable genes, observed counts, OLS
predictions, held-out R2 and the pathway summaries. All 12,704 available fold
R2 values agree within 1e-9; the maximum observed discrepancy is 9.77e-15.
The frozen patient-fold assignments were checked separately. Seven relevant
regression tests passed, including cancer/center isolation, unresolved-source
rejection and the existing train-only reference/phospho tests.

COAD source contract
--------------------

The downloaded COADREAD LinkedOmics matrix contains 5538 genes by 90 patients,
with 179162 zeros out of 498420 finite entries (35.946%). Its filename identifies
CDAP UnsharedPrecursorArea, whereas the download-page table labels it Log2ratio.
CDAP's official report defines Unshared Area as a label-free precursor-area
measurement, distinct from an iTRAQ reporter-ion ratio.

The distribution is compatible with transformed abundances and unquantified
zeros, but the exact transformation and zero encoding have not been verified
against matching original cells. Do not infer them from the distribution.

PDC000111 version 1 identifies the original precursor-area report and publisher
MD5. Its returned CloudFront URL gave HTTP 403; the official legacy endpoint
refused connection; the version-specific unshared_precursor_area matrix API
reported no matrix. The pipeline rejects COAD calibration until this contract
is resolved. It does not log-transform twice, silently mask zeros or manufacture
a result. COAD and READ also remain distinct cancers in cohort selection.

Phosphorylation and remaining limits
------------------------------------

The earlier BRCA analysis checks five sequence/epitope-reviewed sites, with
separate within-platform training-only total-protein residualization and five
held-out folds. It does not establish broad phospho-pathway calibration.
No common PTMsigDB signature reaches the existing minimum of five reviewed
members. Residual targets depend on both total protein and phosphorylation;
future predictors for those targets must exclude both assay layers.

No retrospective-to-prospective CPTAC transfer validation, absolute activation
scale, externally validated acceptance rule or universal patient Y is created.
Current training inputs are unchanged. The present result is evidence about
where calibration succeeds or fails, not a promise that all pathways can share
one numerical label.

Reproduction
------------

Run on host 149; use new output directories and the already verified RPPA500,
CPTAC bridge, mapping, source-manifest and specimen-audit artifacts.

::

  python scripts/score_tcpa_cptac_protein_cv.py \
    --rppa RPPA500 --bundle CPTAC_BUNDLE --mapping COMMON_PANEL_MAPPING \
    --retrospective DOWNLOAD_COMPLETE.json --specimens SPECIMEN_AUDIT \
    --dataset OV_JHU --cancer-id OV --output NEW_OV_JHU_CV

  python scripts/verify_tcpa_cptac_protein_cv.py \
    --root NEW_OV_JHU_CV --output NEW_INDEPENDENT_REVIEW

Use OV_PNNL for the other center and BRCA_BI / BRCA for the BRCA definition.
The existing BRCA result remains immutable. COADREAD_VU currently fails with an
explicit unresolved-source error.

Aggregate evidence is in provenance/tcpa_cptac_calibration_ov_cv_20261106.
The prior BRCA protein/site evidence is in
provenance/tcpa_cptac_calibration_specimen_cv_20261106.
Patient matrices and specimen identifiers are retained privately on host 149.

Primary documentation
---------------------

* https://linkedomics.org/data_download/TCGA-COADREAD/
* https://proteomic.datacommons.cancer.gov/pdc/assets/CDAP_ProteinReports_description_20160503.pdf
* https://pdc.cancer.gov/pdc/publicapi-documentation
