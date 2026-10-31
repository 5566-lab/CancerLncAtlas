KO, KD, OE and rescue regulatory-network inputs
=============================================

The supplied causal-perturbation collection is consumed through
cc_hhgt.v32.causal_regulatory_inputs and causal_regulatory_network. The existing
patient_conditioning.experimental_regulatory_graph entry point builds the
native GNN GraphBundle from one declared experiment. Its typed edges reach
the current HGT and RGCN graph representation; they are not merely extra
columns in an evidence inventory.

The collection is indexed once and read by experiment. Large source matrices
stay on the server, and every source file is SHA256-bound. The index retains
ineligible experiments and their review reasons. Source files are not modified.
The output directory must be new and outside the source directory.

Preparation and consumption
---------------------------

All preparation and validation commands execute on host149::

    python -m scripts.prepare_v32_causal_regulatory_network index \
        --source-root SOURCE_DIRECTORY \
        --source-manifest-sha256 VERIFIED_INPUT_MANIFEST_SHA256 \
        --output NEW_NETWORK_INPUTS

    python -m scripts.prepare_v32_causal_regulatory_network network \
        --input-manifest NEW_NETWORK_INPUTS/COMPLETE.json \
        --source PerturbAtlas --experiment Perturb_3634 \
        --max-fdr 0.05 --output NEW_KO_NETWORK

    python -m scripts.prepare_v32_causal_regulatory_network network \
        --input-manifest NEW_NETWORK_INPUTS/COMPLETE.json \
        --source GEO_rescue_GSE154812 --experiment circPRE8 \
        --purpose mechanism_validation --output NEW_RESCUE_NETWORK

The network command writes RESPONSES.parquet, EDGES.parquet, NODES.parquet,
GRAPH.pt and a hash-bound completion record. GRAPH.pt contains graph tensors
and provenance, with no trained parameters or optimizer state. It is an
experimental regulatory graph, not a replacement for a formal patient-fold
G2 artifact or an old trained checkpoint.

Python callers can reuse one verified reader to avoid repeatedly loading
L1000 matrices::

    from cc_hhgt.v32.causal_regulatory_network import CausalRegulatoryNetwork
    inputs = CausalRegulatoryNetwork("NEW_NETWORK_INPUTS/COMPLETE.json")
    experiments = inputs.experiments(intervention="OE")
    response = inputs.responses("LINCS_iLINCS_OE", "LINCSOE_1000")
    graph = inputs.graph("LINCS_iLINCS_OE", "LINCSOE_1000")

Alternatively, patient_conditioning.experimental_regulatory_graph accepts
the manifest, source and experiment ID directly. graph.hetero_data and
graph.homogeneous enter the existing encoder. An encoder must be constructed
for the graph's declared relations. Automatically created reverse edges are
computational message-passing routes; they do not assert reverse biological
causality.

Source-specific network meaning
-------------------------------

* PerturbAtlas: native DESeq2 log2FoldChange, lfcSE and adjusted P values.
  Metadata IDs and matrix axes must agree. Complex or unresolved interventions
  remain indexed but do not generate eligible response edges.
* GPSAdb2: raw integer counts, actual treatment/control groups and the supplied
  descriptive difference of mean log2(CPM+1). Group-specific effects are
  replayed. No differential-expression P value is invented.
* iLINCS OE: directly measured landmarks in source MODZ/signature units.
  Gene-by-signature storage is explicitly aligned to signature metadata.
  Flagged lncRNA constructs remain available for review, outside eligible
  molecular networks.
* L2S2: directly measured landmark up/down memberships. Unlisted genes are
  unobserved, not zero-response or negative-interaction labels. Conflicting
  up/down membership is unavailable. Inferred readouts and flagged guide
  assignments do not enter eligible measured-response edges.
* LncRNA2Target3: eligible source direction only. Unverified effect magnitudes
  remain audit fields, not regression targets. Invalid statistics and the
  source-misaligned experiment remain visible and excluded as appropriate.
* GSE154812: NORAD CRISPRi, circPRE2 low-valency comparator, and circPRE8
  functional bypass are distinct. Effects replay from the eight supplied
  normalized-expression samples without rounding. circPRE8 is not WT NORAD
  re-expression, and its study-level role is not a gene-level rescue-success
  label. The absence of unperturbed-construct control arms remains explicit.
* LncTarD2 and the LncRNA2Target3 low-throughput table are available through
  auxiliary(). Their curated relation/assay fields are retained; an absent
  structured intervention cannot become an invented KO/OE/rescue experiment.
* CasRx2024 is an auxiliary family-by-cell fitness input. Its native family
  identities, observed masks and CasRx controls stay separate from molecular
  RNA-response edges. The source observed mask requires finite LFC and FDR;
  a finite LFC with missing FDR remains outside that mask. Raw LFC and separate
  per-field availability masks are retained.

Graph semantics and missingness
------------------------------

Each graph belongs to one source, experiment, cell context and effect scale.
Positive/negative relation types describe the observed response to that
intervention. A KO-induced increase is not automatically an activating basal
edge. No RNA response establishes protein abundance, activity or physical
binding. Missing measurements stay NaN; observed zero remains observed in
RESPONSES, and creates no signed response edge.

Quantitative edges carry absolute native response as weight and observed
response sign as relation_polarity. Direction-only sources carry unit ordinal
membership. Those unit weights are not effect-size measurements. Cross-source
scales are not pooled. --max-fdr filters only valid source adjusted P values;
sources without those values cannot pass that filter.

Canonical Ensembl identities and unambiguous supplied symbols are used for
gene alignment, with the current model's GENE: and LNC: canonical namespaces.
Unresolved or ambiguous readouts remain in the response view,
outside eligible graph edges. --cross-type-only selects lncRNA/coding-gene
cross-type responses; all mapped molecular response types are retained by
default. Synthetic rescue constructs have their own node type and retain
their NORAD-CRISPRi background.

Study grouping and validation
-----------------------------

Known GSE/SRP/GSM/SRX/SRR and plate identifiers form connected study groups
across source metadata and sample tables. Matching PMIDs connect curated
records. A mechanism holdout embargo propagates through each known connected
component. Original source split assignments remain provenance, not an
automatically accepted joint split.

The registry deliberately leaves other studies unassigned and reports
cross_source_split_ready=false. A complete GEO/SRA/PMID crosswalk, repeated
sample review and a declared joint evaluation design are required before
joint training. The graph API currently accepts exploration and reserved
mechanism_validation purposes; it rejects an unreviewed training purpose.

Synthetic tests exercise native H5, NPZ and sparse readers, numerical group
replay, reviewed/unreviewed inputs, missing values, and actual graph encoding
with unchanged model parameters. Real-source receipts report their own
consumption scope separately. Neither preparation nor a frozen encoder check
is evidence of predictive benefit or a new biological mechanism.

Verified run on 2026-11-01
-------------------------

All 230 files in the supplied collection passed independent SHA256 checks.
The source input-manifest SHA256 is
3ab80abc6988d0ef54f2cc5e4e0e890b9b96b756210547f72fd7c086bd06777c.
The completed registry has 167,383 entries across eight sources: these mix
experimental contrasts, signatures, curated records and one fitness handle,
and must not be described as 167,383 independent biological experiments.

Ten real example graphs passed serialization, readback and a frozen HGT
encoder forward on149. They cover PerturbAtlas lncRNA KO/OE and coding-gene
KD, GPSAdb lncRNA KD, L1000 KO/OE, LncRNA2Target direction-only response,
NORAD CRISPRi, circPRE2 and circPRE8. This does not claim every indexed entry
has been materialized or independently validated. Selected edge counts are:

* Perturb_3634 KO: 290 edges, source FDR <= 0.05.
* Perturb_6834 OE: 130 edges, source FDR <= 0.05.
* circPRE8 functional-bypass rescue: 16,579 mapped descriptive-response edges;
  no source FDR filter or gene-level rescue-success label is supplied.

Perturb_9045 had no eligible edges under the same FDR <= 0.05 policy. That
empty result is retained. The next eligible OE comparison was selected only
to exercise graph consumption; this is not an unbiased biological estimate.
CasRx retains 142,440 observed family-by-cell entries out of 24,171 by six;
2,586 additional finite LFC values have missing FDR and stay outside the source
observed mask. Curated LncTarD and low-throughput LncRNA2Target are consumed
as auxiliary evidence, not relabeled as experimental response edges.

The accepted code check has 65 passing tests, including existing patient
conditioning regressions. Versioned source hashes, source verification,
the input registry, real consumption, empty selections and incomplete earlier
attempts are recorded in provenance/code_completion_20261015/causal_*.
The registry receipt locates EXPERIMENTS.parquet and ENTITIES.parquet on149;
the consumption receipt locates the ten completed graph artifact directories.
The immutable source data and graph tensors remain on the server. The source
scale and experimental-response direction are never pooled into an inferred
basal activation/repression graph.
