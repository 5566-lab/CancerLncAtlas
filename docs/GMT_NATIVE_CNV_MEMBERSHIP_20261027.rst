GMT / GENCODE membership for native CNV
======================================

Execution and input
-------------------

All commands run on host ``149``. No model fitting is dispatched. The immutable
input manifest contains exactly ``format=NATIVE_PATHWAY_MEMBERSHIP_INPUT_V1``,
``reference_annotation`` and ``gmt_manifest``. Both references contain ``path``
and ``sha256``. The first points to the native GENCODE ``ANNOTATION_SOURCES.json``;
the second uses the existing ``SSGSEA_GMT_INPUT_V1`` schema. Relative GMT file
paths are resolved against the GMT manifest's directory. Collection names,
duplicate pathways and gene symbols follow the existing ssGSEA GMT parser.

::

  python -m scripts.run_native_pathway_membership_workflow \
    --manifest MEMBERS_INPUT.json --output /new/membership --plan
  python -m scripts.run_native_pathway_membership_workflow \
    --manifest MEMBERS_INPUT.json --output /new/membership

Planning verifies all inputs and code without creating the destination. Run
requires a new destination and emits ``LAUNCH_PREFLIGHT.json``,
``pathway_membership.parquet`` and ``NATIVE_PATHWAY_MEMBERSHIP_BINDING.json``.
Failure leaves ``RUN_FAILED.json`` and no valid completion. Validation rehashes
the original GTF, GMT, manifests, code and outputs, then reconstructs the table.
A process-local replay cache is reused only for the same verified bytes.

The chain registry's 33rd operation is ``pathway_membership.run``. Its
``reference_annotation`` can use ``{"stage_ref": "reference"}`` from
``annotation.run``. Downstream native CNV manifests consume
``{"stage_ref": "members", "pointer": "/membership"}`` as ``membership``.
Existing expression candidates can use the same GMT and GENCODE reference;
the CNV loaders check this relationship before materialization. Independently
located, byte-equivalent reference tables and GMT collections are accepted.

Scientific identity and missingness
----------------------------------

``pathway_id`` is the existing ``MSIGDB:<collection>:<GMT set name>``. The
identifier labels a declared GMT collection; it does not authenticate MSigDB
origin. GMT descriptions do not create biological study evidence. Repeated
symbols within one GMT line are deduplicated by the existing parser.

The original ``build_v32_genomic_static_assets.symbol_gene_map`` is reused:
matching is exact and case sensitive, protein-coding mappings are preferred
when present, and every distinct matching Ensembl ID is retained. If a symbol
has no protein-coding mapping, the original noncoding mappings remain. There
is no symbol alias guessing or expression/detection filter. The standalone
legacy builder retains its historical output, including its inner join.

Matched ``gene_id`` values prepend ``GENE:`` to GENCODE stable IDs, retaining
``_PAR_Y``. ``ensembl_gene_id``, ``gene_symbol`` and ``gene_type`` retain the
reference mapping. ``mapping_status`` is ``MAPPED`` or ``UNMAPPED_SYMBOL``.
For unmatched symbols, ``gene_id`` is ``UNMAPPED_GMT:<SHA256 of UTF-8 symbol>``;
the Ensembl ID and gene type are empty. This reserved identifier represents
an unavailable source member, never a fabricated gene or measured zero.
It remains in the actual existing CNV callability logic. An all-neutral
observed subset with an unobserved member cannot establish pathway neutrality.

One symbol may expand to several genes, preserving the original mapping
choice. The CNV consumers deduplicate ``(pathway_id, gene_id)`` and use their
existing unweighted gene estimands, not a new symbol-weighted calculation.
The signed pathway mean and absolute-burden pathway sum remain distinct.
Mapped noncoding fallback IDs may lack native gene intervals: they remain
unavailable and counted, with no invented coordinates. The native genomic
interval producer selects protein-coding genes and lncRNAs by their existing
separate entity types. This entry does not change that selection.

``source_symbol_memberships`` counts unique GMT pathway/symbol pairs;
``mapping_rows`` includes expanded mappings; ``consumer_gene_memberships``
counts deduplicated pathway/gene pairs. ``unmapped_symbol_memberships`` counts
unmatched source pairs, ``one_to_many_symbol_memberships`` counts source pairs
with multiple retained IDs, and ``noncoding_fallback_rows`` counts mapped
noncoding rows. No unmatched member is dropped from the new native output.

Consumers and evidence limits
-----------------------------

Both native directional and compact genomic CNV loaders reconstruct marked
membership tables and compare their annotation bytes and named GMT collection
hashes with the actual candidate definition. Their receipts include
``membership_reference``. Model-use inspection traces the consumed table back
to raw GMT and GTF and records a numerical reconstruction, not patient or
study identities inferred from reference resources.

Unmarked legacy membership remains usable under its existing rules and retains
the generic unverified upstream transformation status. Marked copied tables,
missing bindings, removed native markers in the producer directory and
rehashed forged mappings are rejected. Neither GENCODE nor GMT authenticity
is established by reconstruction. Their explicit acquisition/authenticity
gaps remain. This entry covers the two CNV consumers; it does not claim all
other membership consumers have identical identifier policies or source proof.

Verification uses synthetic GTF/GMT/count/segment files, the actual original
R expression and CNV producers, both CNV CLIs, a resumable four-stage chain
and actual source inspectors. The genomic test dependency uses explicitly
marked frozen core export serialization. No encoder fitting, optimizer,
real-cohort processing, download, paid compute or deployment is performed.
