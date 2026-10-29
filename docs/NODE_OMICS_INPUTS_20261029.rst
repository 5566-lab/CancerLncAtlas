Measured CNV and promoter methylation node inputs
================================================

This continues the DEEPSEEK session exported as session_cae4e657.jsonl.
The previous encoder implementation accepted mutation, signed CNV and ATAC
blocks, but its reader was not called by the production training loader.
Methylation was mentioned in that conversation but was not an accepted channel.

The new preparation entry point is::

    python -m scripts.prepare_v32_node_omics --config inputs.json --output NEW_DIRECTORY

Run on host 149. It checks the hostname, writes LAUNCH_PREFLIGHT.json before
processing, and refuses an existing directory. It does not download data, fit
normalization parameters, construct an optimizer or train a network.

Input contract
--------------

``inputs.json`` has these fields (all file records contain path and SHA256)::

    {
      "format": "CANCERLNCATLAS_NODE_INPUT_PREPARATION_V1",
      "target_host": "149",
      "outer_fold": 0,
      "scope": "cohort",
      "cancer_id": "BRCA",
      "sample_authority": {"path": "samples.parquet", "sha256": "..."},
      "assays": [
        {"channel": "cnv_signed", "unit": "log2_copy_ratio",
         "measurements": {"path": "cnv.parquet", "sha256": "..."},
         "mapping": {"path": "cnv_nodes.parquet", "sha256": "..."}},
        {"channel": "methylation_promoter_beta", "unit": "beta",
         "measurements": {"path": "beta.parquet", "sha256": "..."},
         "mapping": {"path": "promoter_probe_nodes.parquet", "sha256": "..."}}
      ]
    }

Paths are relative to the configuration. TSV, CSV and parquet are supported.
Replace the ellipses with actual digests; they are not accepted placeholders.

Existing native products can be consumed directly by setting an assay's layout:

* ``native_methylation_patient``: measurements points to the completed
  ``NATIVE_METHYLATION_BINDING.json``. The original workflow validator checks
  source/recovery completion before reading patient_lncrna_methylation.parquet.
  It consumes promoter_methylation_beta and its availability flag, not temporary
  batches. Declared training patients in the held-out outer fold are refused.
* ``native_directional_cnv``: measurements points to a cancer partition's
  ``SUCCESS.json``. The reader checks the signed-CNV semantics and the exact
  hashes/shapes/order of patient_ids.json, lncrna_ids.json,
  lncrna_signed_mean.npy and lncrna_callable.npy. Absolute burden is not used.

Both native products are already patient-level. Each patient's value is counted
once regardless of how many aliquots occur in the sample authority. Their
mapping is explicitly to lncRNA nodes; these files do not invent gene-level
measurements. Gene-level inputs can use the sample_feature_long layout above
or the verified raw-to-gene route described below.

* Measurements: ``sample_id, feature_id, raw_value``. One row per sample and
  feature. Missing numeric values remain missing; numeric zero is observed.
* Mapping: ``feature_id, node_type, canonical_id``. Explicit feature-to-node
  relations, including multiple probes per promoter or a declared feature
  mapping to multiple nodes. Node types are gene, lncRNA or protein. Mapping a
  gene assay to a protein node does not turn it into measured protein abundance.
  Promoter annotation and genome builds must be resolved upstream. No implicit
  symbol conversion, barcode truncation or guessed promoter assignment occurs.
* Sample authority: ``sample_id, patient_id, cancer_id, is_tumor, partition``.
  Unique samples; boolean tumor flags; partition is train, validation or test.
  A patient cannot cross partitions or cancer types. Supply a separate, correct
  authority for each outer fold. These declared roles are not an independent
  attestation of the project's historical fold assignment or biological identity.
* Signed CNV accepts declared log2 copy ratio or GISTIC thresholded values
  {-2,-1,0,1,2}. Absolute copy counts require an upstream transformation and are
  refused under these units. No conversion to absolute magnitude is performed.
* Promoter methylation accepts beta in [0,1]; M-values are not beta values.

Aggregation and scale
---------------------

Only declared training tumors enter cohort features. Average observed features
inside a sample, samples inside a patient, then patients. Thus extra probes or
technical samples do not give a patient greater weight. No missing measurement
is replaced with zero before aggregation. The encoder receives a value and an
observed flag for every requested channel; an absent node/channel is (0,0), a
measured zero is (0,1).

Scaling is explicitly identity (center=0, scale=1), followed by the encoder's
existing [-8,8] clipping. No training or held-out statistic is fitted. Units
remain in the manifest; different CNV units must not be silently pooled.

``cancer_id=PANCAN_TRAINING_POOL`` explicitly requests a global aggregate over
the declared training tumors of all cancers. It is a shared node feature, not
a cancer-specific embedding. Otherwise one cancer is selected. The loader
requires the graph's cancer universe to match the declared input universe.
``scope=subject`` additionally requires subject_id and produces that patient's
tumor features for the intervention API; the cohort training loader refuses it.
This does not implement a patient minibatch training objective.

Production consumers
--------------------

In a training configuration, set ``task_contract.node_omics_channels`` to the
ordered channel list. In the matching input-manifest fold row, add::

    "node_omics": {"path": "/absolute/output/node_evidence.json", "sha256": "..."}

Each fold uses its own prepared evidence. The actual graph loader applies it
after graph/binding overlays and before constructing the encoder. The selected
core export loader uses the same reconstruction. The input-authority hashes
include both the evidence manifest and table, so checkpoint/input substitution
is detected by the existing receipt checks. Loading replays the aggregation
and compares the private contributing rows, as well as rechecking source hashes.

For the three binding-evidence arms, declare node_omics_channels in the common
comparison configuration before preparing new sidecars. Their settings digest
is immutable: adding channels to an already generated comparison is correctly
rejected as configuration drift. Rebuild that preparation under a new release.

The historical three-channel default remains unchanged; methylation is opt-in.
Existing degree-only bundles keep two inputs. New features require a newly
constructed compatible model and newly authorized training; historical trained
checkpoints cannot acquire these capabilities by replacing a configuration.
This change neither launches training nor upgrades historical results.

Patient intervention correction
-------------------------------

The temporary intervention context now changes the stored node_omics blocks as
well as the current tensors. Runtime chunk reconstruction therefore uses the
intervened values. All stores are restored even if block installation or the
context body raises; tensors preserve their original device and dtype.

Evidence and remaining limits
-----------------------------

The test suite uses synthetic samples, real aggregation, graph loading, frozen
encoder forwards and the export loader. Optimizer entry points are forbidden.
It checks held-out perturbations, unequal technical replicate counts, signed
CNV, measured zero versus missing values, invalid beta/units/mapping/partitions,
repinned output corruption, graph scope, runtime rebuilding and restoration.
The native integration test also runs the registered weighted all-chunk
embedding exporter, writes and reloads gene and lncRNA embeddings, and separately
perturbs each measured methylation/CNV channel. Each perturbation must change
the corresponding final node vector by more than 1e-7 while leaving model
parameters unchanged. The original vectors must return bit-for-bit after the
temporary intervention. The measured vector deltas are retained in the test XML.

Raw assay normalization, promoter-map correctness, identity/fold authority
authenticity and full model-use patient ancestry are not proved by these tests.
The existing model-use inspector retains unresolved input-authority gaps for
the new source branch until that traversal is implemented. C13 stays open.
The native workflow does not yet generate these optional per-fold declarations
automatically; the documented explicit manifest interface is required.
C01 and the complete public field dictionary (C16) also remain open.

The preceding test results use synthetic cohorts and no newly trained model. No evidence is
provided for improved prediction, biological mediation or a true new binding/PPI.
Existing CNV segments or methylation batches may be upstream sources after the
required assay processing; this change does not establish that new downloads
are necessary or that any particular external dataset is available.

Verified protein-coding gene inputs
----------------------------------

The ``verified_gene_patient`` layout consumes a completed
``GENE_PATIENT_ASSAYS_VERIFIED_V1`` product. It does not reinterpret the existing
lncRNA matrices. The producer is::

    python -m scripts.bind_v32_tcga_gene_sources --sources SOURCE_SPEC.json --output NEW_BINDING
    python -m scripts.prepare_v32_gene_patient_omics --config NEW_BINDING/gene_assays.json --output NEW_MATRICES --workers 12
    python -m scripts.prepare_v32_gene_node_folds --gene-product NEW_MATRICES/COMPLETE.json --prepared-root EXISTING_G2 --output NEW_FOLDS

All three commands require host 149 and a new output directory. The final
command reads existing PTs, writes five new node-input preparations, and calls
the actual training input loader and ``_validate_prepared`` for every fold.
``INPUT_MANIFEST.json`` and the final ``COMPLETE.json`` appear only after all
five checks pass. A pending manifest is not an accepted release. This command
does not train or replace a checkpoint. ``MODEL_CONFIG_PATCH.json`` declares
the ordered channels for a compatible future model configuration; existing
binding-comparison configurations retain their immutable settings contract.

``SOURCE_SPEC.json`` declares target_host=149, genome_build=GRCh38,
methylation_download_root and a ``sources`` dictionary of path/SHA256 records:
gtf, hm450, patient_folds, signed_membership, protein_gene_encoding,
cnv_manifest, cnv_staging, methylation_manifest. The GDC manifest/staging
schemas are checked by the binding script. It joins GDC file, sample and patient
identities, verifies staging metadata against GDC size/MD5, and pins the current
formal patient-fold map. It does not use download-time CNV fold assignments.
Binding alone is explicitly not raw-file verification.

The materializer reads and checks every selected raw file against its GDC MD5
and byte size, records SHA256 of the bytes actually parsed, and retains these
records in ``verified_raw_files.parquet``. The input JSON, reference annotation,
raw-file inventory, source authorities, implementation, matrix axes and outputs
are hash-bound. Completed matrices are verified again when used by the node
loader. This is content/identity/coordinate and implementation verification;
it is not independent biological validation or re-estimation of SeSAMe/CBS.

* Both channels use the same GENCODE GRCh38 GTF. Only genes explicitly annotated
  protein_coding and represented by requested graph IDs enter the matrix.
  Ensembl numeric version suffixes are removed; PAR_Y is preserved. Unresolved
  or non-coding requested genes are listed separately and remain unobserved.
* CNV is signed DNAcopy segment mean (log2 copy ratio), evaluated at the
  GENCODE gene-body midpoint in one-based coordinates. It is not a gene-body
  length-weighted mean, absolute burden, integer copy count or pathway value.
  Adjacent GDC segments may share an endpoint. A gene midpoint exactly at such
  a shared boundary is marked unavailable; true interior overlaps are rejected.
  Per-file receipts count shared boundaries and affected genes.
* Promoters use every protein-coding gene's transcript TSS, strand-aware
  -1000 through +100 bp inclusive, converted to BED0 half-open intervals.
  Official HM450 GRCh38 CpG coordinates are intersected with these windows.
  A probe is counted once per gene even if it overlaps several transcripts;
  one probe can contribute to distinct genes whose promoters overlap.
* Finite SeSAMe beta values must be in [0,1]. Average distinct probes per file,
  files per biological sample, then samples per patient. Missing probes,
  uncovered promoters, missing assays and ambiguous CNV positions stay NaN,
  with an explicit availability matrix. Observed zero remains observed.
* Matrix orientation on disk is patient x gene, with an explicit patient list
  for every cancer and a shared gene table. This is the transposed storage form
  of gene x patient data, not a change in biological granularity.
* Fold aggregation follows the formal three-way split: outer_fold is test,
  (outer_fold+1) modulo 5 is validation, the other three folds are training.
  The reader rejects a relabeled test or validation patient. Aggregation uses
  equal patient weight and identity scaling, without fitting parameters.

The node loader streams these matrices; it does not expand the full cohort to
hundreds of millions of patient/gene rows. Private lineage records identify
each contributing patient's exact matrix row and the hash-bound gene axis.
The original matrices retain patient-specific information. The pooled prepared
node features are training-cohort gene features; they are not one learned
embedding per patient. A future compatible trained encoder is needed for
learned embeddings. Existing frozen tests verify that each channel can alter
gene vectors without changing model weights.

Source definitions: `GDC CNV pipeline <https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/CNV_Pipeline/>`_
and `GENCODE release history <https://www.gencodegenes.org/human/releases.html>`_.
Real-cohort completion and formal validation are recorded separately from
synthetic test results in the versioned server receipts; neither is inferred
from a running process or a partial matrix directory.
