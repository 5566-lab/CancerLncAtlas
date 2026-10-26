GENCODE annotation in the native bulk workflow
=============================================

Scope and execution
-------------------

``annotation.run`` is the thirtieth operation in the native workflow chain.
It consumes an already available, SHA256-pinned GTF on host ``149``. It does
not acquire a reference, read real cohorts implicitly, or fit a model. The
CLI is ``python -m scripts.run_native_annotation_workflow --manifest INPUT.json
--output NEW_DIRECTORY``. ``--plan`` checks declaration/path/code pins only;
the receipt explicitly says ``raw_annotation_parsed=false``. A run parses
the reference before creating its output, then reconstructs the table from
the GTF before returning success. Existing output directories are rejected.

Input schema::

    {
      "format": "NATIVE_ANNOTATION_WORKFLOW_INPUT_V1",
      "gtf": {"path": "/absolute/reference.gtf.gz", "sha256": "..."},
      "gencode_release": 36,
      "genome_build": "GRCh38"
    }

Plain UTF-8 GTF and gzip GTF are supported. This entry targets modern human
GENCODE releases >=25 on GRCh38, including explicit ``GRCh38.pN`` descriptions.
The mandatory description header must match the declared release/build.
Lifted GRCh37 IDs with extra mapping suffixes are deliberately unsupported.
Header declarations and byte identity do not authenticate a supplied file.
The format reference is https://www.gencodegenes.org/pages/data_format.html.

Gene table and reconstruction
----------------------------

``gene_annotation.tsv`` retains every gene row, contig and biotype. It does
not select genes by measured expression or drop genes sharing a symbol.
Transcript/exon rows do not become additional gene rows. The existing
candidate consumer still selects ``lncRNA`` and excludes PAR-Y duplicates.

* ``ensembl_gene_id``: gene ID with only its numeric version removed;
  the ``_PAR_Y`` suffix remains part of the stable identity.
* ``gene_symbol`` and ``gene_type``: exact required gene_name/gene_type values.
  Missing/duplicate required attributes are rejected. Repeated optional tags
  and unquoted numeric attributes such as ``level 2`` are allowed.
* ``versioned_gene_id``: original gene ID, including numeric version and PAR-Y.
* ``chromosome``, ``start``, ``end``, ``strand``: original gene coordinates;
  ``coordinate_system=1_BASED_CLOSED``. Start must be positive, end >= start,
  and strand must be + or -. No liftover or BED conversion is performed.
* ``gencode_release`` and ``genome_build``: declared release and GRCh38 family,
  checked against the header. ``metadata.header_genome_build`` preserves an
  explicit patch suffix; patch biological veracity is not independently checked.
* ``gtf_line``: original one-based physical line number, including header lines.
* ``native_annotation_format``: marks this table as requiring its native binding.

``ANNOTATION_SOURCES.json`` binds the table, metadata and ``LAUNCH_PREFLIGHT.json``.
The preflight binds the original input declaration, GTF and producer code.
Validation reconstructs every output cell and compares deterministic TSV bytes,
so replacing table values and recomputing their hashes is insufficient.
In-process proof reuse always rechecks all raw/output/declaration/code pins.

Production consumers
--------------------

In a chain, use ``{"stage_ref":"annotation","pointer":"/annotation"}``
as the ``annotation`` input of ``expression.run``. The result's normal file pin
is accepted by the existing producers. The fixed sidecar location plus table
marker retains native ancestry even after R/Python consumers normalize pins
and discard optional metadata. A marked copy without its matching bound
source is rejected. An optional ``upstream_annotation`` pointer must resolve
to the same sidecar. Legacy unmarked annotation files remain accepted with
their unresolved source-ancestry status.

Native expression preflight checks every counted gene's original versioned
ID, symbol and biotype against the GTF-derived table, using one raw STAR file
per cancer. The original R assembler already checks identical gene metadata
across every sample in that cancer. Therefore a sample cannot bypass the GTF
check by differing from the first sample. This does not infer a unique GENCODE
release from gene IDs; ``star_release_independently_verified`` remains false.
The receipt reports counted genes, annotation genes, and uncounted annotation
genes. Later candidate/detection universe checks retain their original rules.

The generated table reaches the actual candidate sampler and R ssGSEA
producer/converter. Native candidate and activity model-input inspectors follow
it through the sidecar to the raw GTF. The reconstruction inventory records an
annotation replay; a reference is not represented as a patient study or an
experimental observation. The remaining gap is explicitly
``GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED``.

Verification boundary
---------------------

The integration fixture uses synthetic GTF/archived GDC/count inputs and the
original R count, TMM, ssGSEA, conversion, candidate and detection producers.
It checks chain resume without rerunning completed producers, then invokes
actual candidate/activity model-input inspection. No learned fitting substitute
is needed in this preprocessing route; learned optimizers are forbidden.
Negative cases cover version/name/type mismatches, ambiguous IDs/attributes,
coordinates, changed raw bytes, rehashed output forgery, and missing ancestry.

This closes raw GTF-to-bulk-annotation transformation and consumption only.
It does not close reference acquisition authenticity, the single-cell or
genomic annotation pathways, raw alignment/biospecimen veracity, fitted
metacell/LASSO ancestry, all-module patient coverage, or independent biological
validation. C01, C13 and C16 remain open.
