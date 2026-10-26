GENCODE coordinates in native CNV analysis
=========================================

``genomic_annotation.run`` is operation 32 in the native chain. It takes the
existing ``annotation.run`` reference binding and generates a new, immutable
``entity_intervals.parquet`` for the original directional and genomic CNV
producers. The workflow and all validation run on host ``149``. No reference
acquisition, liftover, model fitting or real-cohort execution is implied.

Input and entry point::

    {
      "format": "NATIVE_GENOMIC_ANNOTATION_INPUT_V1",
      "reference_annotation": {"path": "/reference/ANNOTATION_SOURCES.json", "sha256": "..."}
    }

    python -m scripts.run_native_genomic_annotation_workflow \
      --manifest INPUT.json --output NEW_DIRECTORY

``--plan`` verifies the reference dependency without creating outputs. Its
parent binding reconstructs the original GTF gene table. The new producer
selects exact ``lncRNA`` and ``protein_coding`` biotypes, without an expression
filter or contig filter, and retains one interval for each retained gene.
Other biotypes are counted as excluded. No gene-symbol alias decision is made.

Fields and coordinate contract
------------------------------

* ``entity_type`` is ``lncrna`` or ``gene``; ``entity_id`` is ``LNC:`` or
  ``GENE:`` plus the parent annotation's stable Ensembl ID. Numeric versions
  are removed by the parent parser; PAR-Y remains a distinct stable identity.
* ``chromosome``, ``start`` and ``end`` retain original GTF gene coordinates.
  ``coordinate_system`` is ``1_BASED_CLOSED``. There is no coordinate shift.
* ``ensembl_gene_id``, ``versioned_gene_id``, ``gene_symbol``, ``gene_type``,
  ``genome_build`` and ``gtf_line`` retain the parent gene identity and source
  line. Start/end/line numbers are integers.
* Parquet metadata ``native_genomic_annotation_preflight`` pins the output's
  preflight. The binding records retained/excluded row counts, identifier
  policy, absence of a contig filter and absence of liftover.

Validation reconstructs every interval and all metadata from the bound parent
annotation/GTF. Updating output hashes cannot substitute changed coordinates,
chromosomes, identities, types or missing rows. A marked copy must still have
its matching producer binding; deleting the marker in the original output
does not downgrade it to legacy input. In-process proof reuse rechecks raw,
output, declaration and implementation byte pins.

In the existing ``NATIVE_DIRECTIONAL_CNV_INPUT_V1`` segment manifest, use this
binding's ``intervals`` pin and explicitly declare::

    "segment_reference": {
      "genome_build": "GRCh38",
      "coordinate_system": "1_BASED_CLOSED"
    }

The declared build/convention must match the native reference. Declared
one-based segments with zero starts are rejected, including during numerical
reconstruction. Half-open segments require a separate verified conversion;
this path does not guess a conversion from filenames. Legacy unmarked interval
tables retain the original input behavior when no declaration is supplied.
The declaration itself is not an independent proof of the segment alignment's
assembly. Receipts explicitly retain ``segment_reference_independently_verified=false``.

The coordinate convention comes from the GTF reference contract, described at
https://www.gencodegenes.org/pages/data_format.html. GDC describes its CNV
products at https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/CNV_Pipeline/;
those general descriptions do not replace a specific input's reference evidence.

Actual consumers and membership coverage
---------------------------------------

The original native directional manifest loader reconstructs marked intervals
and checks that candidate definition and intervals use identical annotation
bytes. Separately bound copies with the same annotation content may reside at
different paths. Different coordinate/annotation content is rejected even if
the stable IDs overlap. This contract does not load the global candidate table
or infer membership from symbols.

The existing pathway-membership table remains unchanged. The reference context
counts unique member genes with and without coordinates. Missing-coordinate
members stay in the original pathway denominator/availability logic; they are
not dropped or filled with zero. Membership identifiers, aliases and raw
pathway-resource ancestry remain distinct unfinished source transformations.

Both native directional partition receipts and the native genomic RUN_CONTEXT
carry ``interval_reference`` with the producer, candidate-annotation binding,
segment declaration and membership coverage. Native directional materialization,
its numeric reconstruction and the original compact genomic reader all apply
the same declared coordinate check before computing results.

The scientific estimands remain different and unchanged. Both map segment
values at the original integer midpoint ``(start + end) // 2``. Directional
CNV preserves signed entity means and separate amplification/deletion calls;
its pathway score averages observed gene signed means. Genomic CNV retains
events and absolute burden, with pathway burden summed across observed genes.
The original event-or-all-members callability rule remains in both paths.
An unobserved member with otherwise neutral genes therefore leaves the pathway
unavailable; an observed event follows the original event-supported rule.

Source interpretation and verification
-------------------------------------

The actual native CNV/model-input inspectors discover interval metadata and
follow the producer to its GTF. The transformation inventory records one
interval table reconstructed, with no folds fitted and no optimizer started.
Reference dependencies are not emitted as patient studies. The explicit
remaining reference gap is ``GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED``;
raw segment acquisition, alignment veracity, membership ancestry and full
cross-module identity remain separate limits.

Synthetic integration runs reference -> intervals -> original R expression,
ssGSEA and candidate production, then both original CNV CLIs. It tests a gene
midpoint exactly at a segment boundary, signed versus absolute estimands,
unmapped membership, missing patients and source reconstruction. Genomic core
export metadata in this fixture is explicit frozen serialization, not a trained
model. No neural forward, fitting, real data acquisition or deployment occurs.
C01, C13 and C16 remain open.
