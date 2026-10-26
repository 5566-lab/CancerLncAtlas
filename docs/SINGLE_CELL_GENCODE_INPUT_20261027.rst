Original GENCODE production in the single-cell input path
========================================================

``single_cell_annotation.run`` is operation 31 in the native chain. It takes
the completed ``annotation.run`` reference binding, then calls the existing
``single_cell_partition_builder.prepare_gencode_annotation`` on its original
GTF. Execution and verification require host ``149``. There is no reference
download, cell QC, PCA, grouping, LASSO or neural fitting in this workflow.

Input and command::

    {
      "format": "NATIVE_SINGLE_CELL_ANNOTATION_INPUT_V1",
      "reference_annotation": {"path": "/reference/ANNOTATION_SOURCES.json", "sha256": "..."}
    }

    python -m scripts.run_native_single_cell_annotation_workflow \
      --manifest INPUT.json --output NEW_DIRECTORY

``--plan`` checks and reconstructs the parent native reference as necessary;
it does not create the single-cell output. Output directories are immutable.
Failures after preflight leave ``RUN_FAILED.json`` and no accepted binding.

The scientific transformation remains the original producer
----------------------------------------------------------

``gene_annotation.parquet`` has the original columns ``gene_id``,
``gene_symbol``, ``gene_class``, ``chromosome`` and ``unique_symbol``. Only
primary chromosomes chr1..chr22/chrX/chrY and lncRNA/protein_coding gene rows
are selected. Mitochondrial, alternate-contig and other biotypes are excluded
by the original code. A symbol is unique only if it identifies one retained
gene. Both genes with a shared symbol remain present; symbol fallback cannot
choose ambiguously between them. Exact duplicate removal and conflicting
gene-ID rejection preserve the existing rules.

Single-cell ID normalization removes known type prefixes and a terminal
numeric version. A GENCODE ``.N_PAR_Y`` ID retains that form under this original
policy. This workflow does not silently replace it with the bulk producer's
different stable PAR-Y representation or change feature mapping semantics.

The original producer now accepts an optional native preflight pin. When
provided, it adds ``native_gene_annotation_preflight`` to Parquet's
``PANDAS_ATTRS`` metadata and the original provenance JSON before hashing that
JSON. It does not alter the scientific columns/values. Legacy calls omit it
and retain their original output schema and provenance contract.

The native binding pins ``LAUNCH_PREFLIGHT.json``, the annotation table and
``GENCODE_PROVENANCE.json``. Validation invokes the original GTF producer into
a temporary directory, compares every table value and metadata attribute,
and reconstructs the original provenance fields and contract hash. Only the
temporary output path and its serialization hash are replaced with their
actual-output equivalents. Rehashing modified classes, symbols, uniqueness,
row counts or provenance is insufficient to pass. Cached proofs always
recheck current raw/reference/output/declaration/implementation byte pins.

Actual single-cell consumption
------------------------------

The original ``run_v32_single_cell_verified_streaming._load_contract`` reads
the Parquet metadata before its normal feature mapping. A native marker must
resolve to the matching table/preflight and pass complete GTF reconstruction.
The resulting preparation binding is recorded in the actual mapping contract.
Native mapping preflight includes all reference, producer and source files.

Only verified native annotations may bypass the old requirement to reside in
the fixed r7 authority directory. Raw H5, metadata, exact membership and all
unmarked legacy annotation tables retain the original path contracts. A
marked table copied elsewhere or disconnected from its source fails; it
cannot silently fall back to legacy handling. Hash-pinned raw GTF changes
invalidate the mapper even when its gene table bytes remain unchanged.

A chain can produce reference -> single-cell annotation -> mapping. In its
generated RUN_STATUS JSON, use the annotation stage's ``/annotation/path`` and
``/annotation/sha256`` pointers for the per-cancer annotation fields, and
pass that JSON with ``json_file`` to the mapping stage. All other source,
cell-type, review, donor, scale and exact-pathway contracts remain required.

Model-input and evidence semantics
---------------------------------

Both the generic input inspector and the actual native single-cell model
inspector follow the annotation through its original producer to the GTF.
The latter records ``NATIVE_SINGLE_CELL_ANNOTATION_RESULTS_V1`` in its native
reconstruction inventory, with zero folds recomputed and two outputs compared
(gene table and original provenance). The proof inherits the actual inspected
expert context and does not initialize a neural model or run an optimizer.

Reference files and their preparation receipts are not assigned the raw H5
dataset's study token. They are dependencies, not patient observations or
independent experiments. The explicit remaining gap is
``GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED``. Raw cell
acquisition/QC/cell-type authority, fitted geometry/LASSO, other reference
paths and complete cross-module patient identity remain separate requirements.

Synthetic validation uses actual original annotation and H5 mapping, including
duplicate-feature sums, coding/lncRNA separation and immutable chain resume.
The full model-input fixture continues through original static/fold graph,
nested/outer statistics, episode preparation and completed-model serialization.
Geometry/LASSO and completion telemetry in that existing fixture are explicitly
frozen test substitutes; no fitting is executed. C01/C13/C16 remain open.
