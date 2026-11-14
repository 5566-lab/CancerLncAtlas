Patient measurement definitions
===============================

The biological task is cancer x lncRNA x exact pathway. Pathway identities,
members, measured readouts and missingness are defined before lncRNA modeling.
The output catalog has 477 KEGG NETWORK mechanisms and 197 State identities.
Multiple assays can describe one identity without becoming additional pathway
identities. A member-state readout and a whole-pathway readout retain different
structural roles.

Protein-coding RNA basis
-----------------------

The expression basis selects 19,912 GENCODE protein-coding gene IDs before
aggregating matching symbols in linear TPM, yielding 19,907 symbols. Missing
measurements remain missing. Methods using reference distributions estimate
their parameters in the designated reference patients, then apply the same
parameters to query patients. The sealed outer-test fold is not read.

The PC_GENE_ID_BACKGROUND_R1 method version identifies the recomputed pathway,
TF, PROGENy and State readouts. Restricting an original model's members changes
its measurement definition; original member lists and the applied version
are both retained.

LM22 relative cell composition
-----------------------------

LM22_PC_LINEAR_NUSVR_RELATIVE_V1 uses the fixed external LM22 reference and
linear TPM. Of 547 source symbols, 498 directly match coding genes and 19 are
resolved through approved HGNC-to-GENCODE gene identities. The resulting
reference contains 517 genes and 22 cell categories. Twenty-four directly
noncoding symbols and six unmatched symbols do not enter this version.

For each patient independently, the reference and mixture are centered and
scaled, and linear nu-SVR is evaluated at nu=0.25, 0.5 and 0.75. Coefficients
are truncated at zero and normalized to sum to one. The candidate with the
lowest reference reconstruction RMSE supplies the relative composition.
Missing required measurements yield missing readouts. No cross-patient model
or lncRNA association is fitted. This is a coding-member adaptation of the
published algorithm, not the original published CIBERSORT score.

Method: Newman et al., Nature Methods (2015),
https://pmc.ncbi.nlm.nih.gov/articles/PMC4739640/ .
The implementation matches R e1071 on three synthetic mixtures to a maximum
absolute difference of 6.66e-16. This numerical check does not measure accuracy
against independently measured cell fractions.

Mechanism-specific conditions
-----------------------------

The 334 Reference mechanisms do not require an additional event condition.
The other 143 require matching evidence: 78 variant/expression/receptor
conditions, 46 pathogen/effector conditions and 19 exposures. An unspecified
gene mutation does not establish an activating variant; a continuous copy
number value does not by itself establish an amplification call.

The dictionary records the required evidence separately from measured member
states. Unmeasured conditions are unknown, qualified negatives are absent,
and discordant qualified observations remain unknown. A measured member does
not establish that its upstream mutation, pathogen or exposure is present.

N01392 describes arsenic-to-complex-II chemistry. Its cached source record's
Pathogen category is preserved, while its patient condition is arsenic
exposure, based on the mechanism name and C20296 chemical entry.

Model comparison
----------------

M0 uses the fixed patient readouts, their masks and the same patients as the
graph comparison. It compares lncRNA, mRNA and combined RNA inputs with
unpenalized clinical and independent-purity covariates. The M0 specification
defines the input exclusions and distinguishes RNA score reconstruction from
prediction of independently measured outcomes. TF/kinase/PROGENy display
classification does not create independent losses. Pathway and State weights
are 1.0 and 0.5.
