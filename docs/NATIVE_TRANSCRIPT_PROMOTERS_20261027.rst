Recovered transcript promoters in native ATAC and methylation workflows
=====================================================================

Original producer and exact geometry
------------------------------------

``scripts/derive_gencode_promoters.py`` is a byte-for-byte recovery of
``D:/model/exports/v3_1_target_context_assets/provenance/derive_gencode_promoters.py``.
Its original SHA256 is
``b234faafdafa9ae21b48daf563395e6cdb245fb7e1def69275a0787aa1436051``.
The associated historical metadata declares 232,117 transcript promoters for
GENCODE v36. That historical count is provenance context, not a new real-data
recalculation or an independent authentication of the original reference.

The recovered producer is executed unchanged on host ``149``. For a positive
strand transcript beginning at one-based position s, t=s-1 and its BED interval
is ``[max(0,t-1000),t+101)``. For a negative strand transcript ending at e,
t=e-1 and its interval is ``[max(0,t-100),t+1001)``. Thus the unchanged window
contains positions -1000 through +100 inclusive relative to the strand-aware
TSS, with 1,101 bases before left-edge clipping. It does not clip to chromosome
lengths because no chromosome length authority is supplied by this producer.

The original selection retains transcript features on chr1..chr22, chrX,
chrY and chrM. It preserves every selected transcript, including distinct
transcript IDs with identical promoter intervals; it does not select a single
representative TSS or replace absent transcript annotations with a gene TSS.
The native wrapper checks required transcript attributes, unique transcript
identity, exact parent gene version/name/type, chromosome, strand and containment
within its parent gene. Non-reference transcript exclusions are counted.
Malformed or inconsistent transcript declarations fail before output creation.

Manifest, outputs and replay
----------------------------

``NATIVE_PROMOTER_INPUT_V1`` contains exactly ``format`` and
``reference_annotation``. The reference is a path/SHA256 pin to the native
GENCODE ``ANNOTATION_SOURCES.json``. The parent binding supplies the exact raw
GTF, annotation bytes, declared GRCh38 build and GENCODE release. Plain GTF
inputs are deterministically compressed in temporary storage for the original
gzip-only CLI; gzip inputs are consumed directly. No reference is downloaded.

::

  python -m scripts.run_native_promoter_workflow \
    --manifest PROMOTERS.json --output /new/promoters --plan
  python -m scripts.run_native_promoter_workflow \
    --manifest PROMOTERS.json --output /new/promoters

The destination must be new. ``--plan`` verifies pinned parent reference/code
without creating output. Run additionally validates transcript records, invokes
the original CLI, then writes ``LAUNCH_PREFLIGHT.json``,
``original_transcript_promoters.bed.gz``, ``native_transcript_promoters.bed.gz``
and ``NATIVE_PROMOTER_BINDING.json``. The binding's ``promoters`` output is the
native consumer handoff; ``original_promoters`` retains original producer bytes.

The native table preserves the original coordinates, chromosome, strand,
transcript ID, gene name and gene type. Its ``gene_id`` is the parent GENCODE
stable ID, and ``versioned_gene_id`` retains the original versioned identifier.
``gtf_line`` identifies the actual raw transcript row. The gzip TSV header also
contains a native preflight marker in an otherwise empty metadata column. Both
existing R and Python consumers ignore that extra column. No patient values
or study identifiers are introduced into the reference.

This explicit ID adaptation matters for PAR_Y. Existing consumers truncate
the first dot and everything following it, so a raw versioned ID ending in
``.N_PAR_Y`` could be collapsed to its other copy. The native stable gene ID
retains ``_PAR_Y`` before those existing consumers run. The original file and
original version ID remain available for audit. Unmarked legacy inputs retain
their existing identifier behavior.

Validation rehashes the input, raw GTF, parent reference, original producer,
wrapper and both outputs, reruns the original producer in temporary storage,
and compares the exact deterministic bytes and metadata. Rehashed coordinate,
identity, missing-row and metadata substitutions are rejected. Process-local
proof reuse requires identical verified source/output/code bytes. Marked copied
files pointing outside their producer directory or missing binding/markers are
rejected. This reconstructs a transformation, not reference authenticity.

Actual consumers and orchestration
----------------------------------

The chain operation ``promoter.run`` is operation 34. Its reference can be an
``annotation.run`` stage reference; downstream ``promoters`` references use
``{"stage_ref":"promoter","pointer":"/promoters"}``.

The existing ATAC materializer accepts the native BED through its original
``--promoters`` argument. Its input contract must contain the actual candidate
definition. Native promoters and candidates must use identical annotation
bytes. Materialization records ``promoter_reference`` in preflight and success,
checks it again before success, and trainer/completed-result validation compares
the reconstructed proof. Removed proofs and unscoped native inputs cannot
silently fall back to legacy handling. Original promoter-peak aggregation,
replicate/aliquot weighting, measured-member projection and minimum coverage
remain unchanged. Actual expert source inspection follows the native promoter
file back to its original GTF and records the reconstruction separately from
unresolved patient/study ancestry.

``probe.run`` accepts the same native BED, stores its source proof and carries
all raw-reference/producer pins into ``validate_map``. Those pins propagate
into actual ``methylation.run`` preparation and recovery. The original genomic
overlap and outcome-independent distal-window algorithms are unchanged,
including zero-link coverage for requested genes with no promoter/link.
An arbitrary requested gene list is still an input declaration; binding a
promoter producer does not authenticate that list's biological completeness.

Verification boundaries
-----------------------

Hand-computed positive/negative strand, left-edge, multiple-transcript and
PAR_Y expectations check the recovered geometry independently. Tests exercise
the actual CLI, reference/promoter/probe/methylation chain, immutable resume,
source tampering, both ATAC preparation and its original R computation, frozen
auxiliary result collection and actual fusion expert source inspection.
Synthetic beta files yield independently specified promoter/distal values.
The official static HM450 manifest is read under its existing checksum gate;
no real patient cohort is analyzed. Existing full-shape synthetic ATAC inputs
and explicitly frozen model-result containers are retained.

All verification uses the opt-in session fitting guard. No learned fitting,
scientific acquisition, paid compute, real-cohort analysis or deployment is
performed in this change. The earlier ATAC test-selection deviation remains
in the audit history and is not erased by these checks. GENCODE/reference
authenticity, ATAC raw normalization, other upstream branches, full patient
ancestry and final requirement-level acceptance remain unresolved.
