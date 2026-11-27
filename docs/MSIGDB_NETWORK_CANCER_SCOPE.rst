MSigDB to cancer NETWORK definitions
=====================================

This pipeline traces each existing KEGG_MEDICUS source definition to its
original KEGG NETWORK element, reads its official map classification and
retains mechanisms represented in Cancer or Cancer signaling categories.

Data flow
---------

1. build_msigdb_network_mapping.py joins the source registry to verified
   MSigDB exactSource N accessions. Archived collection JSON can fill gaps
   in source-page metadata. Every metadata file is checked against SHA256.
   Original source definitions and their gene memberships are preserved.
2. fetch_kegg_network_scope.py freezes original N records, their nt maps,
   and BRITE br08907. It preserves full Definition and Expanded strings,
   network type, parent human pathways and Entrez members. It checks
   categories against BRITE and N CLASS lists against nt MEMBER records.
3. build_network_cancer_scope.py emits source-level scope decisions and
   typed cross-database links using those frozen inputs.

Admission requires an exact verified N identity with official membership in
a map categorized as Cancer or Cancer signaling. Missing source records
remain unresolved. A source outside these categories is outside this
selection; that designation does not assert that its biology is unrelated
to cancer.

The 28 human KEGG PATHWAY cancer maps and these NETWORK maps are different
objects. An hsa pathway is a broad pathway definition. An N element describes
a particular mechanism; an nt map groups related reference and perturbed
mechanisms.

Cross-database correspondence
-----------------------------

* reviewed_equivalent_mechanism reuses a previously reviewed functional
  identity with explicit biological rationale and evidence. It can transfer
  cancer scope to that counterpart.
* shared_parent_pathway_candidate records that the N element belongs to
  a human KEGG pathway with an external equivalent or part-of mapping.
  It is a candidate, not equivalence of the N mechanism.
* protein_similarity_candidate carries the existing shared-protein count,
  Jaccard index and overlap coefficient for review. It does not transfer
  cancer scope or create a merged identity.

Canonical WNT (N00056), APC-inactivated WNT (N00057) and CTNNB1-activated WNT
(N00058) retain distinct mechanism accessions even when their members
overlap. Their shared parent WNT pathway is insufficient to merge them.
Definition strings retain activation, inhibition, expression and complex
notation; they are not converted into simplified protein-pair edges.

Outputs
-------

MSIGDB_NETWORK_MAPPING.tsv
    Source definition, MSigDB systematic ID, original N, existing canonical
    ID, metadata evidence and mapping outcome.
ARCHIVE_MEMBER_COMPARISON.tsv
    Archived members compared with original and cached memberships.
    The archive does not replace scoring inputs.
network_snapshot/NETWORK_RECORDS.tsv
    Original names, mechanism strings, types, gene IDs and parent pathways.
network_snapshot/NETWORK_MAP_MEMBERSHIP.tsv
    Official nt memberships and categories.
SOURCE_SCOPE_DECISIONS.tsv and ADMITTED_SOURCE_DEFINITIONS.tsv
    Complete decisions and the selected source definitions.
CROSS_DATABASE_ALIGNMENT.tsv
    Typed counterparts, evidence, scope-transfer and identity-merge fields.
COMPLETE.json
    Counts and input/output checksums.

Execution
---------

All preparation runs on host 149. Each command records and checks that host.
Archive URL and version must be supplied explicitly. Downloads use verified
caches, bounded retries and at most three requests per second. Completed
snapshots are immutable; changed inputs require a new output root.

The three command-line interfaces expose their arguments through --help.
A scope materialization with a completed snapshot is::

    python scripts/build_network_cancer_scope.py \
      --mapping /path/to/bridge/MSIGDB_NETWORK_MAPPING.tsv \
      --snapshot-root /path/to/frozen_network_snapshot_root \
      --catalog-root /path/to/canonical_targets_r7 \
      --protein-pairs /path/to/PAIR_SIMILARITY.parquet \
      --out /path/to/new_scope_output

Consumers select explicit admission decisions. Candidate correspondences
must not be treated as equivalent identities. This foundation does not
change patient scores or relation-label values.
