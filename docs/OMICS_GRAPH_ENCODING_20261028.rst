Multi-omics node channels in the typed graph encoder
====================================================

Status: code implemented and fixture-verified on host 149. No optimizer, real
cohort, paid compute or deployment ran. This document describes a graph *input*
contract only; it makes no claim about predictive gain.

The question this code answers
------------------------------

Can measured mutation, signed copy number and promoter accessibility enter the
encoder whose output is propagated by message passing, rather than being
attached to the decoder as a post-hoc expert?

Before this change every node entered the encoder as exactly two columns,
``[log1p(degree), 1]`` (``cc_hhgt/gnn.py``), and the first layer was a fixed
``nn.Linear(2, hidden)``. Mutation, CNV and ATAC reached the model only through
separate arms that are combined after encoding.

What is now implemented
-----------------------

``cc_hhgt/v32/omics_node_encoding.py`` builds a per-node-type block of
``2 * len(channels)`` columns - one ``value`` and one ``observed`` flag per
declared channel - and appends it after the two degree columns:

* ``cc_hhgt/gnn.py`` validates the block (``validate_node_omics``), widens every
  node input to ``2 + width``, records ``GraphBundle.node_input_dim`` and keeps
  the block on the bundle so runtime chunks re-materialise with the same inputs;
* ``build_model`` reads ``bundle.node_input_dim`` for the first layer of both
  the heterogeneous HGT encoder and the homogeneous RGCN encoder;
* ``attach_node_omics`` widens an already materialised bundle and then reads
  every written tensor back, comparing values column by column.

Evidence and standardisation
----------------------------

An evidence pair is a table (``.tsv``, ``.csv`` or ``.parquet``) with exactly
``node_type``, ``canonical_id``, ``channel``, ``raw_value`` plus a JSON manifest
declaring ``format``, ``scope``, ``cancer_id``, optional ``subject_id``,
``channels``, frozen ``standardisation`` parameters per channel,
``table_sha256`` and ``row_count``.

* ``scope = cohort`` is a cancer-level aggregate; ``scope = subject`` is one
  patient, and the identifier is required. A subject table is never silently
  reused as a cohort table, or the reverse.
* Standardisation parameters (``center``, ``scale``) are read from the manifest.
  Nothing is fitted at load time, so a fold cannot inherit a statistic computed
  from its test split.
* Declared channels are restricted to ``mutation_damage``, ``cnv_signed`` and
  ``atac_accessibility``.
* A node and channel with no measurement keeps ``value = 0.0`` and
  ``observed = 0.0``. It is never imputed from another patient, another cancer
  or a cohort mean: the flag is what distinguishes "measured zero" from "not
  measured".

Fail-closed codes
-----------------

Evidence layer: ``OMICS_EVIDENCE_FILE_MISSING``, ``OMICS_EVIDENCE_FORMAT_INVALID``,
``OMICS_EVIDENCE_SCOPE_INVALID``, ``OMICS_EVIDENCE_CANCER_ID_MISSING``,
``OMICS_EVIDENCE_SUBJECT_ID_MISSING``,
``OMICS_EVIDENCE_SUBJECT_ID_NOT_ALLOWED_FOR_COHORT``,
``OMICS_EVIDENCE_CHANNELS_INVALID``, ``OMICS_EVIDENCE_CHANNEL_UNKNOWN``,
``OMICS_EVIDENCE_CENTER_INVALID``, ``OMICS_EVIDENCE_SCALE_INVALID``,
``OMICS_EVIDENCE_CLIP_DECLARATION_INVALID``,
``OMICS_EVIDENCE_TABLE_SHA256_MISSING``, ``OMICS_EVIDENCE_TABLE_SHA256_MISMATCH``,
``OMICS_EVIDENCE_COLUMNS_MISSING``, ``OMICS_EVIDENCE_ROW_COUNT_MISMATCH``,
``OMICS_EVIDENCE_EMPTY_IDENTIFIER``, ``OMICS_EVIDENCE_CHANNEL_NOT_DECLARED``,
``OMICS_EVIDENCE_RAW_VALUE_INVALID``,
``OMICS_EVIDENCE_DUPLICATE_NODE_CHANNEL``, ``OMICS_EVIDENCE_NODE_NOT_IN_GRAPH``.

Alignment layer: ``OMICS_NODES_LACK_INDEX_COLUMN``, ``OMICS_NODE_INDEX_DUPLICATE``,
``OMICS_NODE_CANONICAL_ID_DUPLICATE``.

Graph layer: ``OMICS_NODE_OMICS_NOT_A_MAPPING``, ``OMICS_BLOCK_NOT_2D``,
``OMICS_BLOCK_NOT_FINITE``, ``OMICS_BLOCK_WIDTH_INCONSISTENT``,
``OMICS_NODE_TYPE_BLOCK_MISSING``, ``OMICS_NODE_TYPE_BLOCK_UNKNOWN``,
``OMICS_BLOCK_ROW_MISMATCH``, ``OMICS_BUNDLE_ALREADY_AUGMENTED``,
``OMICS_ATTACH_SHAPE``, ``OMICS_ATTACH_VALUE_MISMATCH``,
``OMICS_ATTACH_HOMOGENEOUS_ROW_MISMATCH``, ``OMICS_ENCODER_NOT_WIDENED``,
``OMICS_ENCODER_INPUT_WIDTH_MISMATCH``.

A missing node-type block is refused rather than zero-filled: the graph layer
must not turn "no signature exists for pathway nodes" into an all-zero
measurement.

What the fixture checks prove, and what they do not
---------------------------------------------------

The checks in ``tests/test_v32_omics_node_encoding.py`` prove, on synthetic
graphs only:

1. the block is row-aligned to ``node_index_within_type`` and independent of the
   input frame row order;
2. unmeasured pairs stay zero with an explicit zero flag;
3. the first encoder layer of both model kinds really reads ``2 + width``
   columns, and a model built before augmentation is refused;
4. a mutation value written on a **gene** node changes the embedding of a
   connected **lncRNA** node without any decoder term, while an isolated node
   and a node outside the perturbed neighbourhood keep bit-identical
   embeddings. This is the operational meaning of "multi-omics participates in
   the graph encoding";
5. stored tensors equal the requested block within ``1e-6``.

They do **not** prove that multi-omics conditioning improves any prediction.
That requires trained arms on real cohorts and is not part of this contract.

Non-claims
----------

* No trained multi-omics graph model exists. Every checkpoint in this repository
  was trained against the two-column contract and still loads unchanged, because
  ``GraphBundle.node_input_dim`` defaults to ``2``.
* No before/after comparison exists yet.
* No patient-level conditioning, mediation test or novel binding discovery is
  implied by this document.
