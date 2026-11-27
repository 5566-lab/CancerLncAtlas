WikiPathways patient observations
================================

Biological definitions and routing
----------------------------------

The collection contains 193 frozen WikiPathways definitions, each retaining its
original name, WP identifier and complete protein membership. The primary
dictionary contains 476 pathway identities and 48 State identities. WikiPathways
definitions are linked to this dictionary as follows:

* Three definitions share a reviewed biological identity with a primary pathway:
  WP438, Non-homologous end-joining; WP531, Mismatch repair; and WP561, Heme
  biosynthesis. Their source-specific observations are attached to that identity.
* 142 definitions have supported protein-composition links. Their complete
  source scores remain separately accessible, while shared molecular observations
  retain their existing measurement identities. Composition links preserve
  subpathway and broader-pathway scope; they do not establish identity equivalence.
* 33 definitions overlap the backbone but do not meet the composition criterion.
* 15 definitions have no protein overlap with the backbone. The latter two groups
  remain in the source collection without a primary-pathway route.

Composition support requires at least three shared proteins and a hypergeometric
overlap test with Benjamini--Hochberg q <= 0.05. The fixed protein universe contains
16,797 proteins. The 91,868 new comparisons (193 x 476) are corrected jointly with
1,331,784 previous comparisons, including zero-overlap comparisons with p = 1.
These mappings use protein definitions, not patient expression or lncRNA
associations. The original KEGG backbone member lists remain explicit; complete
WikiPathways lists are retained as their own source definitions.

Patient measurements and scores
------------------------------

RNA uses the common background of 19,907 protein-coding gene symbols. Source
scores use the existing ssGSEA calculation with alpha = 0.25, at least 10 mapped
genes, at most 500 definition genes, and raw enrichment scores without fitting a
cohort-wide score range. Each cohort has 167 source definitions with finite RNA
scores. All 26 remaining definitions have fewer than 10 mapped coding genes.
For example, WP561 contains nine mapped genes: its RNA member measurements are
retained, but the ten-gene scoring requirement is not relaxed.

TCPA RPPA500 protein observations retain published antibody values. For source
protein scores, each antibody is standardized using only the frozen reference
patients, requiring at least three finite reference values and positive standard
deviation. Antibodies mapping to the same gene group are averaged once. The
source score is the mean of observed eligible gene groups, requiring at least
three groups per patient. Multi-gene antibody groups must be wholly contained
in the source definition to contribute directly.

CPTAC protein inputs retain the published native scale. Protein-coding rows
mapping to the same reference symbol are averaged over observed values. Protein
scores use GSVA ssGSEA with alpha = 0.25, normalize = FALSE, use = 'na.rm', at
least 10 observed genes per patient and at most 500 definition genes. Missing
proteins and patients remain missing. TCPA and CPTAC scores retain their own
platform-specific measurement definitions and are not averaged across platforms.

Phosphorylation preserves the original site or antibody identity. Ordinary
protein membership does not specify a directional phosphosite signature, so it
does not produce an additional phospho activity score. CNV and methylation
preserve the original gene-level measurements. Partially matched multi-gene
assays remain annotated context rather than direct member observations.

.. csv-table:: Development-cohort coverage
   :header: "Cohort", "Patients", "With protein measurements", "RNA scores", "Protein scores"
   :widths: 24, 12, 26, 18, 20

   "TCGA BRCA", 872, 688, 167, 102
   "CPTAC BRCA", 96, 96, 167, 142
   "TCGA LUAD", 407, 272, 167, 100
   "CPTAC LUAD", 88, 88, 167, 147
   "TCGA COAD", 363, 279, 167, 100
   "CPTAC COAD", 84, 77, 167, 130

Score counts denote source definitions with at least one finite patient score;
coverage can differ between individual scores. All 193 definitions have member
measurements in each cohort. Patient eligibility and each observation's measured
mask are stored separately.

Representation of Y and shared observations
-------------------------------------------

Y is a patient-by-unique-observation matrix with explicit pathway memberships,
source identities and observation masks. It is not one averaged activity scalar
per pathway. A source readout is identified by its WP definition, omics layer and
scoring method. The three equivalent definitions contribute six registered
RNA/protein score dimensions per cohort to the primary matrix; dimensions that
fail their measurement requirements remain masked. Whole-source scores from
composition-only links remain source-level observations and do not become exact
backbone labels.

The same molecular measurement keeps one observation ID and one value even when
multiple pathways reference it. Shared assay keys reuse the prior bundle's IDs;
their raw values and missingness must match exactly. Source membership records
preserve the provenance of these links. An additional source definition does not
automatically create an additional loss or increase a shared measurement's
weight. Existing patient values are retained unchanged.

Validation and access
---------------------

The six cohorts contain 1,910 development patients, with exactly the existing
reference and query partitions. RNA backgrounds and RNA members contain only
annotated protein-coding genes. Independent validation reproduced all RPPA
reference parameters and source scores, compared CPTAC scoring inputs to native
protein aggregation including missing values, and recomputed the observed gene
counts. Every measured mask equals the finite-value mask, and shared molecular
values match the preceding bundle exactly.

The repository reader is ``cla_engine.wiki_patient_y.load_wikipathways_source``.
The portable bundle also includes ``wiki_reader.py``, which can read its bundled
``sources/WikiPathways`` collection directly:

.. code-block:: python

   from pathlib import Path
   from wiki_reader import load_wikipathways_source

   values, observed, metadata = load_wikipathways_source(
       Path("/path/to/Y_bundle"), "TCGA_BRCA", "WP531",
       layer="RNA", kind="source_readout", for_supervision=True,
   )

``for_supervision=True`` intersects the measured mask with the observation's
eligibility flag. It does not promote a composition-context source to an
equivalent primary pathway. Primary routing remains explicit in
``WP_SOURCE_ROUTES.tsv`` and ``WP_EQUIVALENT_ROUTES.tsv``. Known sources without
a requested layer return an empty observation axis; unknown WP identifiers raise
``KeyError``.
