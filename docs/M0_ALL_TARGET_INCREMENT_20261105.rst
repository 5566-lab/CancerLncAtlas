Three RNA arms across existing CPTAC pathways and State
=======================================================

The BRCA/LUAD/COAD demo extends the completed unscreened three-arm assay
comparison to every existing CPTAC RNA_PATHWAY and STATE endpoint. This
includes TF programs, cell composition, expression programs and available
non-RNA State endpoints. The existing native CPTAC scores are unchanged;
this run does not replace them with the separate harmonized TCGA/CPTAC
score builder or pool native scores across cohorts. Upstream score
preprocessing is retained, not fitted anew within outer folds.

Each cancer is trained separately using the same five outer folds, inner
holdouts, 25 Lasso penalty ratios and complete clinical-covariate policy.
There is no correlation/FDR admission filter. Age, sex, stage and DNA purity
are unpenalized. A DNA purity target excludes purity from its covariates and
from the corresponding eligibility check. A zero-RNA fit in any outer fold
or the final fit makes the complete target-arm training unsuccessful.

All attempted and unavailable targets remain in coverage tables. Matched
performance uses the identical targets and held-out patients successful in
all three arms. Improvements are combined minus mRNA held-out R2. They are
exploratory, conditional on successful training and not multiplicity-adjusted.
RNA-derived endpoints test score reconstruction, not independent protein
measurement or causal regulation. TF and other RNA-derived State results
are displayed separately from non-RNA source endpoints.

The figure exporter writes one figure each for lncRNA, mRNA and their union,
plus a paired comparison and a training-coverage figure, in 300-dpi PNG and
vector PDF. Existing protein, phosphorylation and methylation results are
reused after receipt checks rather than rerun. All computation executes on
host 149. Model files and patient-level outcomes remain outside Git.
