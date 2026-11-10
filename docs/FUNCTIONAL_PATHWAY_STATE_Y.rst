Functional Pathway and State relationships
=========================================

The target is a cancer-specific relationship between an lncRNA and a canonical
Pathway or State. Patient measurements are denoted by S; relationship evidence
is denoted by Y. Protein abundance, phosphosite signatures, RNA enrichment and
other State measurements retain their own readout definitions and units.

Identity and composition
------------------------

``data/functional_targets_20261109`` contains the stable biological identities,
source bindings, aliases and redirects. It has 3,456 identities with defined
readouts and four structural context nodes. Pathway and State roles overlap for
three identities: 2,595 identities have a Pathway role and 864 have a State role.
The current patient-score inventory contains finite measurements for 2,840 of
these identities; the inventory and the dictionary report coverage separately.

An equivalence mapping shares one canonical identity. Parent-child, branch,
shared-reaction-core and molecular-facet relations preserve distinct identities.
The Reactome reaction mappings describe reaction-set intersections and extensions;
complex, catalyst and regulator relationships preserve the annotated mechanism.
Source protein-list overlap is recorded separately from functional equivalence.
``ID_REDIRECTS.tsv`` resolves prior identities to surviving IDs.

``TARGET_COMPOSITION_COUNTS.tsv`` summarizes each target's gene, probe,
phosphosite, reaction and child-program composition. ``PATHWAY_REACTION_BRANCHES.tsv.gz``
links 21,358 annotated branch occurrences to 1,111 scored identities by explicit
pathway-order membership, including descendants. These are source reaction-order
relations; canonical target IDs remain unchanged.

Patient scores
--------------

RNA pathway scores use a common gene background and ssGSEA definition across
TCGA and CPTAC. State definitions retain their signature composition and units.
RPPA total-protein scores summarize measured pathway members using the source
patient reference. RPPA phosphosite readouts use matched total-protein adjustment,
a source-fitted reference and the signature direction. CPTAC native measurements
retain their source definitions. The dictionary provides the assay, algorithm,
reference, unit, molecular composition and observed patient coverage per readout.

``CPTAC_STATE_METHODS.tsv`` lists the actual method, software, input scale and
reference for 413 State readouts across ten cohorts. For example, RNA stemness
uses frozen Malta OCLR weights with a pooled CPTAC min-max reference, xCell uses
a pooled CPTAC reference, and relative CIBERSORT composition is derived from
22 published absolute components. ``CPTAC_COVARIATE_COVERAGE.tsv`` reports
observed clinical and composition covariates for each cohort.

Each cancer is handled separately. All 16,889 TCGA lncRNA features remain on the
feature axis. Clinical and composition covariates follow the explicit adjustment
contract, with raw signed local CNV acting on the lncRNA residual. Both tumor-cell
fraction readouts exclude purity from their covariates because purity represents
their own biological quantity. Other target adjustment roles are preserved.

Relationship Y
--------------

The released relationship catalog covers TCGA/TCPA in 33 cancers and five
patient contexts per cancer. CPTAC contributes patient scores, molecular
composition, method references and coverage from ten separately indexed cohorts.

Associations use only the label patients for the selected context. The covariate
encoder and applicable score references are fitted on disjoint source patients.
Direct RNA score members are excluded from evidence for their own RNA-derived
readouts. Correlation is defined only for nonconstant rank vectors with variation
remaining after adjustment. The numerical tolerance scales with machine precision
and rank-vector norm, so expression units do not determine feature eligibility.

The per-assay association criteria are:

* Strict support: absolute adjusted Spearman rho >= 0.20 and BH q <= 0.05.
* Additional support: absolute rho >= 0.15 and BH q <= 0.10, excluding strict pairs.

Multiple algorithms on one assay contribute one assay-family bit. Each context
stores these uint8 matrices on the same lncRNA by canonical-identity axes:

* ``membership``: 0 unlabelled; 1 supported by at least one assay family;
  2 strict support from at least two distinct assay families.
* ``direction``: 0 unknown; 1 positive; 2 negative; 3 conflicting supported signs.
* ``observed_assay_bits``: assays in which the association is estimable.
* ``strict_assay_bits`` and ``supported_assay_bits``: assay-specific evidence.

Bit values are RNA=1, protein=2, phospho=4, methylation=8, copy number=16,
DNA variant=32 and other=64. Membership zero is an unlabelled relationship;
these outputs do not define a supervised negative class. Conflicting directions
remain explicit. The Pathway task weight is 1.0 and the State task weight is 0.5.
A shared identity is counted once with the Pathway-primary weight.

Reading a context
-----------------

The public reader checks file hashes, identity axes, evidence codes and patient
separation before returning arrays::

    from cla_engine.functional_y import load_functional_context

    catalog = "/dell_2/DSC/CancerLncAtlas/processed/functional_y_20261109_r1/Y"
    target = load_functional_context(catalog, "BRCA", "sealed_test")
    print(target.counts())
    membership = target.arrays["membership"]

The equivalent command is::

    python scripts/inspect_functional_y.py --catalog /path/to/catalog \
        --cancer BRCA --context sealed_test

``Y_by_cancer`` in the dictionary reports the independent label fold once for
each cancer. ``Y_by_context`` describes the separate training, validation and
sealed-test contexts. Counts across these contexts are not pooled as unique
relationships. The independent label fold contains estimable associations in
22 cancers and supported relationships in 20 cancers.
``Y_SUPERVISION_BY_CANCER.tsv`` and ``Y_SUPERVISION_BY_CANCER_CONTEXT.tsv``
provide the counts and patient denominators.


In that independent label fold, 495 relationships meet strict criteria in at
least two assay families. Of these, 476 concern tumor-cell fraction with RNA and
copy-number-derived evidence. The other 19 concern 17 Pathway identities with
RNA and total-protein evidence. No phospho family appears in this strict
multi-family subset. These counts describe relationship evidence, not predictive
model performance.

The integration combines canonical identity and assay-specific association
evidence. It does not average TCPA and CPTAC patient scores onto a validated
common activity scale. The current formal relationship Y uses TCGA and matched
TCPA; CPTAC scores retain their aligned identities and measurement definitions.
CPTAC formal relationship estimation still needs a covariate design compatible
with its missing methylation-derived leukocyte fraction.

Reproducing the materialization
------------------------------

``provenance/functional_y_20261109/source`` preserves the executed preparation,
curation, scoring, numerical correction, aggregation, validation and reporting
code. ``SOURCE_MANIFEST.json`` records file hashes, the source code baseline and
the filesystem inputs. All scientific preparation runs execute on host 149.
Patient matrices remain in the project data store; the repository includes
biological definitions and code.

The stage order is source inventory and annotation, identity/structure curation,
patient score preparation, fold-specific adjusted associations, direct-member
exclusion, canonical-quantity adjustment, numerical estimability correction,
assay-family aggregation, and independent artifact validation. The final
coverage tables feed the reader dictionary and presentation.

The saved numerical correction leaves retained correlations, raw p-values,
patient counts and design ranks unchanged and recomputes BH q-values over the
estimable pairs. The runtime pair and graph routines apply the same rank-residual
variation criterion when calculating new results.

The project release entry is
``/dell_2/DSC/CancerLncAtlas/processed/functional_y_20261109_r1``.
On server 149 this directory resolves to the immutable dataset under
``/dsk2/user/dengsc/CancerLncAtlas/processed/functional_y_20261109_r1``.
``PATIENT_SCORE_INDEX.tsv`` lists 710 cohort, context and measurement-layer
entries with relative matrix and readout-axis paths. ``COMPLETE.json`` records
the release file hashes. The public reader was checked against the published
BRCA independent-label context.
