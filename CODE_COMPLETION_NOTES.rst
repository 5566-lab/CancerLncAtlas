CancerLncAtlas code completion, 2026-10-15
========================================

This branch is an isolated development copy. Production results and pointers
have not been changed. Real-data preparation, training and deployment have not
been executed. The requirement ledger is CODE_COMPLETION_PLAN.json; pending
items there remain part of the requested work.

Execution and validation
------------------------

All executable preparation and validation must run on host 149. The development
copy is /dsk2/user/dengsc/CancerLncAtlas_code_completion_20261015. The available
interpreter is /dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python. Set PYTHONPATH
to the development root. Source edits and coordination occur in the isolated
local worktree. No paid instance is needed.

Methylation recovery
--------------------

scripts/materialize_v32_methylation_lncrna.py now accepts --resume-batches with
the existing sample_feature_batches directory. Supply the original
--selected-files, --probe-map, --patient-folds and --download-root, plus a NEW
--output directory. Recovery does not read the raw download files. It validates
file/sample/patient identity, probe counts, Beta ranges, duplicate rows and
manifest completeness. Zero-output files require their processing records.
It averages technical files within each exact sample, then gives each sample
equal weight within the patient. A measured zero Beta remains available; a
missing value remains null.

Fresh runs retain BATCH_INPUT_AUTHORITY.json before processing and sample
records before final aggregation. The final SQL uses explicit aliases, fixing
the reserved-word parser failure. Original outputs are not overwritten.

Recovery does not claim raw-file MD5 verification. A legacy batch without its
original aggregation authority cannot prove that its Beta averages used the
new probe map. Such a recovery remains useful as a diagnostic artifact but
does not become a leakage-isolated formal predictor. Analysis completion and
eligibility for later raw-file deletion are recorded separately.

Distal prediction inputs
------------------------

scripts/build_v32_independent_distal_topology.py consumes --promoters (GRCh38
BED0 with gene_id), --peak-set (official GRCh38 one-based closed ATAC peaks),
--required-genes (the COMPLETE formal gene universe, not a Data S7 subset),
--reference-build GRCh38, --max-distance-bp and a NEW --output directory.
The default window is 1,000,000 bp from any transcript promoter interval.
Peaks overlapping any promoter of the same gene are excluded. The output is
a coordinate-proximity candidate hypothesis, not proof of regulation.

INDEPENDENT_DISTAL_LINKS.tsv.gz and its adjacent .audit.json are consumed by:

* materialize_v32_distal_mutation_lncrna.py --topology-links
* materialize_v32_hm450_lncrna_probe_map.py --topology-links
* materialize_v32_distal_atac_lncrna.R, the fourth positional argument

The historical --datas7-links spelling is retained for compatibility. Data S7
links remain cohort-selected evidence even if their correlation/FDR columns
are removed. Their features are ineligible for formal within-cohort expression
prediction. Recompute features with the coordinate topology before formal use;
attaching its audit to old aggregates is insufficient.

materialize_v32_distal_regulatory_head_input.py verifies matching topology and
actual feature-artifact hashes for mutation, ATAC and methylation. The optional
--allow-cohort-selected-topology creates diagnostic inputs explicitly rejected
by run_v32_distal_regulatory_mutation_head.py. The latter consumes both ordinary
patient OOF components and the nested inner-OOF components for each outer fold.
No fold's held-out expression is used to fit any of its training components.
Missing assays are null. MC3 WES zero burden means no reported event, not
confirmed locus-level wild type. GRCh37 events still require audited liftover.

Signed single-cell fusion
------------------------

scripts/materialize_v32_signed_single_cell_fusion.py consumes:

* --current-pointer: the activated SC-HHGT pointer, including bound OOF
  predictions, independent validation and selected primary arm;
* --candidates: exact cancer_id / lncrna_id / pathway_id Parquet universe;
* --donor-crosswalk: Parquet dataset_id / query_donor_id / canonical_donor_id,
  reconciling repeated patients across datasets;
* --min-donors (default 3) and a new --output.

The adapter verifies canonical donors do not cross outer folds. It retains
DONOR_CELL_CONTEXT.parquet and CELL_CONTEXT_SUMMARY.parquet. For each candidate,
it averages cell contexts within a donor and then gives donors equal weight.
Positive and negative strengths are separate measurements; they are not
probabilities. The signed mean and between-donor SD are retained for inspection.
The SD is descriptive variability, not model or population confidence bounds.
Observed rho targets, residuals and the ablation arm are not used as features.

ExpertSpec.value_transform now distinguishes probability_logit, signed_fisher
and identity_unit. Existing checkpoints default to probability_logit. New
checkpoint metadata and public native-value columns preserve the chosen
measurement semantics. Missing experts fall back exactly to the primary score.

Current-source fusion runner
-----------------------------

scripts/run_v32_current_multimodal_fusion.py --manifest MANIFEST.json
--expected-core-run-id RUN_ID --output NEW_DIRECTORY prepares the fusion inputs
without fitting. PREPARED.json says PREPARED_NOT_TRAINED. It does not write a
training success marker. The separate --fit option exists for a future
authorized run and has NOT been used in this task.

Manifest format: CURRENT_MULTIMODAL_FUSION_MANIFEST_V1. Required fields are
core_run_id, primary_prediction {path,sha256}, primary_binding_audit {path,sha256},
primary_folds [{fold_id,artifact:{path,sha256}}] and experts. Each expert declares
spec (ExpertSpec fields), prediction {path,sha256}, binding_audit {path,sha256},
run_id and estimand. The producer binding must contain status PASS, the actual
prediction output hash and its provenance. The primary binding includes run_id,
prediction {path,sha256} and fold_prediction_sha256. Future fitting additionally
requires fusion_run_id matching V32-FUSION-[A-Z0-9_-]{8,96}; training_config is
optional. The retired neural Evidence head is excluded. Rule strength must not
be declared as a calibrated probability.

This removes hard-coded historical input paths. Remaining native-head adapters,
the selected-score release binding and downstream GeneSet/web integration are
tracked separately; this runner alone does not complete those requirements.

Patient and mutation-subgroup heterogeneity
------------------------------------------

Run ``python -m scripts.analyze_v32_patient_heterogeneity`` on host 149, with
--expression, --activity, --groups, --covariates, --folds, --candidates,
--group-id and a NEW --output directory. Each input is a Parquet table/dataset
(TSV is also accepted for the per-cancer source tables). Exact patient IDs are
joined without truncation. All patients in the frozen fold map are retained.

Expression columns are cancer_id, patient_id, lncrna_id and lncrna_expression
(logcpm is accepted). Activity uses pathway_id and activity_score. Groups use
group_id, group_value (0/1/null), group_available and group_unavailable_reason;
lncrna_id is optional for a general subgroup. Covariates use measurement columns
and corresponding <column>__available booleans. The default nuisance covariates
are local_cnv_log2 and tumor_purity; override with --nuisance-columns.

--groups-are-mutation-calls accepts reported_variant_count,
mutation_assay_available and locus_callable, with optional column-name overrides.
A positive mutation report identifies group 1. Group 0 requires an assayed,
callable locus and zero reported events. Zero WES events without locus coverage
remain unknown. No missing assay is converted into a wild-type patient.

The five patient folds provide held-out pathway-activity predictions from
expression, subgroup, their interaction and nuisance covariates. Transformations
and ridge coefficients are estimated on the other four folds. Separate full-cohort
OLS interaction estimates use HC3 covariance and asymptotic Wald intervals.
These describe association heterogeneity, not individual causal effects. BH
correction includes every declared candidate within cancer x subgroup, including
unavailable tests in the family size. --candidate-selection is required:
prespecified_independent or exploratory_current_core. Current-core selection
does not become independent external validation by using this command.

Outputs include patient_oof/, INTERACTION_SUMMARY.parquet,
SUBGROUP_OOF_METRICS.parquet and FIT_AUDIT.jsonl, with hash-bound AUDIT/SUCCESS
records. Synthetic tests verify HC3 against statsmodels, held-out isolation,
unknown mutation status, sparse groups and the actual CLI materialization.

Current V4 core export and native-head readers
--------------------------------------------

``python -m scripts.export_v32_current_core_embeddings --manifest INPUT.json
--expected-core-run-id RUN_ID --output NEW_DIRECTORY`` performs frozen inference
on host 149. CPU is the default; --device cuda is explicit. This task has NOT
executed it on real checkpoints. It has no training/optimizer entry point.

INPUT.json uses CURRENT_CORE_EXPORT_INPUT_V1, core_run_id, graph_variant and
exactly five fold_inputs. Each fold has integer fold plus config, input_manifest,
prepared, checkpoint and training_receipt, each a {path, sha256} object.
graph_overlay is required when the original training input authorizes it;
global_binding_source can locate a relocated C-global binding source. Relative
paths resolve against INPUT.json. Original config/input hashes remain bound to
the selected checkpoint. Relocating files must not rewrite those original files.

The exporter reconstructs the original C overlay and data authority, verifies
the selected checkpoint against its SUCCESS receipt and checks the complete
registered runtime chunk weights. It computes sum_c w_c * encoder(chunk_c),
streaming one chunk at a time with compensated float64 accumulation. This is
an encoder-space feature definition; it is not encoder(union_graph) and is not
claimed to reproduce the core's decoder probabilities. No partial-chunk export
can produce a complete manifest. Parameter hashes are checked before/after.

The V2 CORE_EMBEDDING_MANIFEST.json records core_run_id, copied training receipts,
all exported artifact hashes, exact weights, constant-embedding diagnostics and
inference precision. It attests to a selected completed training run; it does
not invent a from-scratch declaration absent from that run's receipts.

Genomic/CNV, Drug and State readers now accept this V2 manifest while retaining
their old V1 reader checks. The clinical runner also consumes the manifest's
actual export paths/hashes and reconciles registered graph type prefixes while
preserving biological namespaces. For clinical V2 runs, provide both
--expected-core-run-id and --training-run-id. Resume verifies the same core,
raw-input lineage, training configuration, implementation and output hashes.
No clinical training has been started by this work.

Fusion producer identity
------------------------

Every expert binding audit must now declare run_id, estimand, expert_specs and
core_dependency. The declared spec must exactly match the producer's approved
columns, transform and endpoint roles. CURRENT_CORE_EMBEDDINGS additionally
requires core_run_id and core_manifest {path,sha256}, verified against the
selected run. INDEPENDENT_MODEL and CORE_INDEPENDENT_EVIDENCE retain their own
source identity without pretending to have used bulk-core embeddings.

The signed SC adapter writes a deterministic adapter run_id derived from its
actual inputs and aggregation policy. Use that audit's run_id in the fusion
manifest. A freely supplied name no longer substitutes for producer identity.
Native Mutation/CNV/ATAC fold adapters, native State/Clinical context consumption
and selected-score release integration are still tracked as unfinished C05 work.

Native drug identity, cancer context and response direction
----------------------------------------------------------

The formal build_v32_drug_raw_inputs command now requires --drug-identifiers,
a curated TSV/Parquet table with drug_id, identifier_type, identifier_value and
source. drug_id must be a namespaced stable identity. identifier_type=name
covers explicitly supplied synonyms; source:<dataset_id> covers native IDs.
Names are normalized for case, Unicode and whitespace only. Salt, stereo and
punctuation distinctions are preserved. Conflicting identities remain separate
source-scoped ASSAY identifiers; they are never merged by fuzzy name matching.
These unresolved measurements, including unnamed PRISM compounds, are retained.

Optional --model-annotations uses canonical_model_id, cancer_id, source and
optional paired subtype_system/subtype_id. Native GDSC diagnosis propagates to
the same canonical model in PRISM. Tissue supplies a cancer only when the mapping
is unique. Conflicting explicit diagnoses are unavailable. Tissue is never used
to invent PAM50 or another molecular subtype. MODEL_CONTEXT_AUDIT.parquet and
MODEL_SUBTYPES.parquet retain the provenance and unavailable reasons. Legacy
multi-cancer duplication is rejected by both drug training readers.

Both dense and streaming sampling now preserve the eligible population's class
proportions through uniform sampling. The streaming reservoir is independent of
chunk boundaries. Unknown/gray labels remain excluded, so the documented prior
is the non-gray association-label prior, not prevalence of clinical response.
Both checkpoint formats record population and sampled class counts.

``python -m scripts.analyze_v32_native_drug_response --exact EXACT.parquet
--response RAW_RESPONSE --expression RAW_EXPRESSION --mapping CELL_LINE_MAP
--subtypes MODEL_SUBTYPES --targets DRUG_TARGETS --output NEW_DIRECTORY``
runs a descriptive scan on host 149. Within each cancer and declared subtype,
every cancer-assayed drug is paired with each exact lncRNA, including drugs
without annotated targets. Dataset response orientation is normalized to higher
sensitivity before rank correlation. Repeated assays do not inflate the number
of unique models. Whole-cohort and five disjoint model-block correlations,
support counts, direction agreement and context coverage are materialized.
Insufficient support yields null, not zero. These are descriptive associations;
no p-values, causal effects, clinical benefit or fitted OOF predictions are claimed.

Frozen drug target and pathway ablation
---------------------------------------

``python -m scripts.analyze_v32_drug_model_attribution --manifest INPUT.json
--output NEW_DIRECTORY`` loads all five frozen current-core drug heads on 149.
INPUT.json is DRUG_MODEL_ATTRIBUTION_INPUT_V1 with core_run_id, drug_run_id and
artifacts {path,sha256}: core_embedding_manifest, drug_lineage,
checkpoint_manifest, config, exact_candidates, exact_pathway_membership,
raw_drug_response, raw_lncrna_expression, cell_line_map, drug_gene_target, queries.
queries contains cancer_id, lncrna_id, drug_id. Optional checkpoint_paths permits
relocation by fold while preserving the original checkpoint hashes.

Actual model forward passes compare the baseline with each removed target and
with all targets in one exact pathway removed jointly. Every removal recomputes
the mean target embedding and target count/availability. The script binds the
original trained inputs, model/config hashes, selected current core and fold.
It verifies parameter immutability and source hashes; it contains no optimizer.
Outputs preserve signed logit/probability differences and extrapolation to a
targetless input. These effects explain an association-strength head; they do
not turn its abs(rho) training label into a sensitivity-direction model. Pathway
effects are not sums of target effects. The optional pharmacology extension
described below now supplies action, activity, structure and protocol covariates
to the current head. Measured dose-conditioned analysis is described below;
protocol mixture covariates alone do not provide that estimand.

The synthetic end-to-end test constructs five tiny serialized heads solely to
test loading and inference. Its emulated training receipt is test-only and does
not constitute a real trained drug model. No real-data scan, checkpoint
attribution or drug training has been executed in this code-change task.

ENCODE RBP depletion and differential splicing
----------------------------------------------

``python -m scripts.analyze_v32_encode_functional --manifest INPUT.json
--cache EXISTING_NATIVE_CACHE --output NEW_DIRECTORY`` runs on host 149.
It verifies cached native files against the existing ENCODE inventory's byte
count and MD5. Optional --download retrieves only selected released products,
uses a temporary file and verifies it before renaming. Optional --proxy passes
an explicitly configured proxy to curl. No downloads were performed by this
code-change task; tests use cached synthetic native tables and archives.

INPUT.json is ENCODE_FUNCTIONAL_ANALYSIS_V1 with reference_annotation_id and
artifacts {path,sha256}: inventory, gene_crosswalk, rbp_crosswalk,
pathway_membership, physical_binding, exact_candidates. Gene crosswalk columns
are source_gene_id and gene_id; source versions are matched exactly. RBP crosswalk
columns are source_rbp_label and rbp_id in the existing graph namespace. Physical
binding columns are lncrna_id, rbp_id, biosample, graph_assay_class and
binding_source_id. Exact candidates use cancer_id, lncrna_id, pathway_id.

Each contrasts entry has contrast_id, file_accession, experiment_accession,
rbp_id, biosample, schema, effect_orientation and orientation_source. Supported
schemas are deseq2, cuffdiff, rmats and mapped_de. The last requires explicit
gene_column, effect_column, pvalue_column and qvalue_column; these column overrides
also work for the named schemas. effect_orientation is KD_MINUS_CONTROL or
CONTROL_MINUS_KD and is verified as a declared direction, never guessed from
effect signs. For a splicing archive, member selects one exact native table;
event_column defaults to ID; event_type declares SE/MXE/A3SS/A5SS/RI and
quantification selects JC or JCEC. The archive is read without extracting paths.
Selecting all JC/JCEC variants as independent evidence is not permitted by the
single-table selector. Biological experiment accessions remain separate from
file and contrast counts; no cross-file replicate pooling is performed.

DE keeps the source's full-family adjusted p-values. Splicing uses within-gene
Simes p-values and BH over all declared gene tests, including unavailable and
unmapped members in the family size. A changed gene additionally needs an event
meeting the native FDR and absolute-effect cutoffs. Simes/BH rely on their usual
dependence assumptions; these exploratory summaries are not external validation.
Opposite splice events in the same gene remain visible. Defaults are alpha=.05,
absolute log2 fold change >=1 and absolute delta PSI >=.1. analysis can override
these and min_tested_genes (default 3) before analysis.

Pathway over-representation uses actually tested genes as the background and
retains all declared pathways in each multiple-testing family. A fully unmapped
contrast still has unavailable pathway rows. Native tests, per-gene summaries,
RBP_PATHWAY_SUPPORT.parquet and COVERAGE.parquet are emitted. The functional
consumer joins significant pathway support to physical lncRNA--RBP binding in
the SAME biosample and exact lncRNA/pathway candidates, writing
FUNCTIONAL_EVIDENCE.parquet. Predicted binding is excluded from this bridge.
This is a traceable RBP-perturbation hypothesis bridge, not a measured lncRNA
perturbation or evidence that a cell line reproduces a TCGA patient context.
Final rule-summary/release consumption remains C12 work.

RBNS transcript sequence scoring and predicted relations
---------------------------------------------------------

``python -m scripts.analyze_v32_rbns_sequences --manifest INPUT.json
--cache EXISTING_NATIVE_CACHE --output NEW_DIRECTORY`` consumes verified native
RBNS enrichment tables and a matching transcript FASTA on 149. It also supports
explicit --download and --proxy for a future run. This task did not download
sequences or score real project transcripts.

INPUT.json is RBNS_SEQUENCE_ANALYSIS_V1 with annotation_id, artifacts
{path,sha256}: inventory, rbp_crosswalk, transcript_fasta, transcript_crosswalk,
lncrnas and protein_gene_map; and profiles entries with profile_id,
file_accession, rbp_id, concentration (the exact native column, e.g. '5 nM').
Transcript crosswalk columns are transcript_id, source_gene_id, lncrna_id and
annotation_id. FASTA transcript and gene versions are checked against this
crosswalk, and annotation_id must agree. The protein mapping is the existing
typed-binding map: gene_id, protein_id, uniprot_accession, mapping_multiplicity,
mapping_weight. lncrnas is the full declared lncRNA universe, including genes
whose sequences are unavailable.

One prespecified native 4-7-mer table/concentration defines each profile. Missing
k-mers are rejected rather than imputed as neutral. Measured zero enrichment is
retained through the declared log floor (default 1e-6). Scores use transcript
orientation without reverse complementing RNA. Only unambiguous A/C/G/T windows
are scored; U is normalized to T. The average window log2 enrichment is compared
with the expectation under independent bases at that transcript's nucleotide
frequencies. This adjusts a simple composition baseline; it does not model RNA
structure, dinucleotide dependence, binding affinity or cellular occupancy.

Transcript scores stream into Parquet, then aggregate with equal weight per
isoform. Defaults require 50 valid windows, at least 90% valid windows per
transcript and 100% registered-transcript coverage per gene. Missing sequences
retain null prediction scores and explicit reasons. The outputs preserve
isoform extrema and coverage. The default predicted relation threshold is
composition excess >=.25 log2 units with mean log2 enrichment >=0; these are
prespecified operational cutoffs, not calibrated probabilities or significance
thresholds. scoring in the input manifest can override these settings.

GENE_SEQUENCE_PREFERENCE.parquet feeds PREDICTED_INTERACTION_RELATION.parquet
and the ACTUAL existing build_typed_lnc_protein_binding consumer. The resulting
RBNS_TYPED_BINDING.parquet contains only binds_protein_predicted relations.
Its weights retain the existing fixed predicted prior and protein-mapping
weight, not a claim that sequence scores estimate binding probability. The
separate relation artifact does not alter the currently trained C graph; the C11
entry below admits it only in the declared predicted-expansion comparison arm.

Both ENCODE/RBNS analyses bind inputs, source files, implementation, run identity
and outputs in AUDIT/SUCCESS records. Neither starts an optimizer or changes the
primary graph. The current combined synthetic suite has 14 checks, including
both complete commands, direction reversal, native archive selection, missing
gene mappings, sequence/version mismatches and the actual typed-edge consumer.

Primary format/semantic references reviewed for these adapters:
https://www.encodeproject.org/files/ENCFF949TLK/ (Cuffdiff contrast description),
https://github.com/Xinglab/rmats-turbo (event fields and inclusion-difference
orientation), https://www.encodeproject.org/rbns/ and
https://www.encodeproject.org/data-standards/terms/ (RBNS enrichment definition).

Current-C predicted-binding runtime compatibility (C11)
------------------------------------------------------

Selected GNN changes were imported from the audited October expansion snapshot;
predicted_runtime_source_import.json records its exact source hash and the
adaptations. The current typed formal-graph modules were preserved. Unrelated
R4 context injection was not imported.

HGT/RGCN now route predicted binding through the weighted message channel only.
The denominator counts predicted edges rather than summing their magnitudes,
so a solitary .25-weight prediction contributes .25 of its source message
instead of being normalized back to full strength. Measured relations retain
their existing normalization and topology behavior. Empty active relations
retain the full declared schema. Old RGCN bundles containing predicted edges
must be rematerialized with the new masks; they cannot silently fall back to
unweighted predicted topology.

GraphBundle.relation_edge_dropout can declare only binds_protein_predicted.
Runtime masks are deterministic in seed, optimizer step and canonical edge ID,
independent of row order or runtime replay. A relation may lose all its edges;
the old expansion code's forced minimum-one retention was removed. Reverse
edges inherit the same decision when tensors are built. Node features are held
fixed. The actual training graph closure requests dropout; validation and
frozen-export calls retain the full graph by default. New predicted-arm
checkpoints bind this policy in their runtime contract. Existing measured-only
runtime contracts remain unchanged.

Seven new checks execute real tiny HGT/RGCN forwards and the runtime bridge,
without an optimizer. Together with current-core export, C-overlay, typed-graph
and index-invariance regressions, the selected suite has 60 passing cases.
The measured-only HGT result matches the historical topology path exactly in
the tested fixture. This does not claim a real-data graph equivalence audit.

C11 preparation and consumers are completed by the current-C sidecar integration
below. The historical graph module was not substituted for the current typed
version; current eCLIP/RBP relations are retained.

Current-C three-arm preparation and actual consumers (C11)
--------------------------------------------------------

The entry is scripts/prepare_v32_current_binding_comparison.py. It runs only on
host 149, writes a target-host receipt, and prepares current_c, experimental,
and experimental_predicted for the same five patient folds. It never dispatches
training. The original prepared files are reused, including all candidate,
label, patient, modality and non-graph inputs. Source admission rows, comparison
receipts, three configs/input manifests and 15 blocked task commands are emitted.

Example invocation on 149::

  python scripts/prepare_v32_current_binding_comparison.py \
    --manifest /path/to/COMPARISON_INPUT.json \
    --output /path/to/new-comparison-directory \
    --release current-c-binding-r1 --edge-chunk-size 250000

The input format is CURRENT_C_BINDING_COMPARISON_INPUT_V1. Its config is a
path/sha256 record for the current typed C config, with graph_variant=G2. Its
fold_inputs contains exactly five integer fold records, each with hash-bound
prepared, graph_overlay and input_manifest artifacts and an optional relocated
global_binding_source. Prepared files remain under G2 and both graph/prepared
filenames use a consistent FOLD_{fold} pattern. Existing size-and-semantics C
overlays are supported, while their source artifacts in this preparation input
are independently SHA256-bound. This procedure does not overwrite the large
original prepared files or their old authorization hashes.

Each sources entry has a unique source_id and one supported kind:

* native_october: edges and lineage Parquet path/sha256 records. The adapter
  recomputes ordered event-lineage hashes, event counts and endpoint-bound edge
  identifiers from the native October schema. Unique protein mapping, physical
  evidence, absence of family broadcast and experimental/computational flags
  are checked. Raw methods recover the current assay classes; contradictions
  such as knockdown labelled physical binding are rejected.
* rbns_sequence: binding and audit path/sha256 records. The binding hash must
  match RBNS_TYPED_BINDING.parquet in the completed, versioned sequence-analysis
  audit. These rows are admitted only as predicted binding, never as measured
  lncRNA occupancy. In-vitro RBP preference remains distinct from this prediction.

Current C edges are retained with their original metadata. A pair already in
current C is excluded from both expansions. A new physical pair takes precedence
over predictions for that same pair; multiple physical assay classes remain
typed. Predicted weights are capped at .25, with raw source weights retained in
the admission audit. Predictions carry observed=false; the safe graph rejects
predictions labelled observed or unobserved edges in other roles.

All arms share an extended node universe, G0-derived features and full relation
schema. New lncRNA/protein nodes append to their existing type-specific indices.
Old candidate tensors therefore keep the same biological endpoints. The
coexpression schedule, chunk count and weights remain common; new binding edges
are resident. A comparison without both new physical and predicted pairs is
rejected rather than producing an empty contrast.

The expansion sidecar is loaded AFTER the existing C overlay by the actual
cc_hhgt.v32.training.run_authorized_task entry. Its manifest, full SHA256, parent
authority, complete non-graph digest, graph tables, node features, runtime
schedule, arm and dropout policy are checked. The common config digest covers
runtime budgets, optimizer/batching/scientific settings and seed. Only arm/output
locations and the separately guarded execution switch differ. The actual task
seed must agree with the common config. Unknown new non-graph inputs participate
in the digest instead of escaping a fixed field allowlist.

Generated tasks target server149/SERVER149_CPU with explicit CPU/FP32 settings.
The existing trainer now accepts that declared host/device contract and skips
CUDA memory calls on CPU; it never turns missing CUDA into implicit CPU fallback.
The existing training authorization remains required for a future run; no
approval artifact is generated and no optimizer is created during preparation.
The code-only command records deliberately omit --allow-training and are blocked.

scripts/export_v32_current_core_embeddings.py consumes the same expansion after
the current C overlay, binds it to the selected training config/receipt, and
restores the actual residual model strictly. Its export input includes an
additional binding_expansion path/sha256 artifact for every expanded fold.
The resulting current-core manifest retains those source identities for the
existing Genomic/CNV/Drug/State/Clinical readers. Predictions are not dropped
during frozen export.

Verification uses synthetic files only on 149. It exercises native lineage,
pair exclusion, node-index preservation, real HGT forwards, five-fold preparation,
all 15 actual training-loader paths, the frozen-export payload/model entry and
the residual candidate forward. Adam/AdamW construction is forbidden in the
integration test. Corrupted sidecars, changed labels/candidates, newly introduced
non-graph fields, budget/seed drift and wrong arms are rejected. Existing graph,
overlay, export and training-guard checks are included in the recorded receipt.
No real-data expansion was materialized, no model was fitted and no new arm's
accuracy was measured. Selection/validation metrics and downstream release
integration remain separate outstanding work under C12-C16.

Observed rule evidence, independent aggregation audit and actual consumers (C09/C12)
-----------------------------------------------------------------------------------

The new command consumes generated native exact events, ENCODE functional bridges
and RBNS gene sequence preferences. It prepares an evidence annotation release,
without a learned Evidence head, optimizer, model probability or ranking change.
Run from this checkout on server 149::

  python -m scripts.materialize_v32_current_rule_evidence \
    --manifest /path/to/RULE_INPUT.json --output /path/to/new-rule-release
  python -m scripts.audit_v32_current_rule_evidence \
    --binding /path/to/new-rule-release/RULE_EVIDENCE_BINDING.json \
    --expected-sha256 <binding-sha256> --output /path/to/new-reaudit.json

CURRENT_RULE_EVIDENCE_INPUT_V1 contains candidates and exact_membership as
path/sha256 records. Candidates have unique cancer_id/lncrna_id/pathway_id keys;
members have pathway_id/member_id, never just a pathway family. Each sources
entry has a unique source_id and one of these kinds:

* native_exact_events supplies events and lineage path/sha256 records. Every
  event must retain complete source-row/file hashes and exact routing. Partner
  routes are checked against exact members; family broadcasts are rejected.
* encode_functional supplies audit and table records. The table hash must match
  FUNCTIONAL_EVIDENCE.parquet in the completed upstream analysis audit. The rule
  reader retains contrast, RBP, biosample, assay, affected genes, tested universe,
  p/q values and direction of the RBP-depletion readout. It does not convert that
  direction into a direct lncRNA effect or claim binding inferred from knockdown.
* rbns_sequence supplies audit and table records bound to the upstream
  GENE_SEQUENCE_PREFERENCE.parquet. Only available scores meeting the registered
  producer thresholds enter a gene-RBP to exact-member prediction bridge. These
  are predicted sequence preferences, excluded from measured study counts and
  direct direction/review. Physical occupancy and affinity are not estimated.

PMID, DOI, registered study dataset and ENCODE experiment tokens form connected
provenance groups. Thus overlapping multi-PMID database copies are not counted
as independent replications. The historical independent_studies column name is
retained, with the explicit semantics CONNECTED_PROVENANCE_GROUPS_NOT_PROOF_OF_
STATISTICALLY_INDEPENDENT_REPLICATION. Provenance-free records remain inspectable
but cannot contribute to verified study counts, direct direction or review.

The event ledger retains each source row and full hash lineage. Context summaries
retain cell line, biosample, tissue, species, assay, intervention, readout, mediator
and source cancer. Opposing directions and review statuses remain conflicts.
Global/cell-line evidence can be displayed beside multiple cancer candidates;
source_cancer_id remains GLOBAL and cancer_specific_studies stays zero for it.
This presentation does not turn a cell-line experiment into a TCGA patient result.

Review uses only explicit raw manual-review status. For example, validated is
accepted, invalid is rejected, and not validated is unknown. Legacy learned
confidence_target values are ignored. The fraction accepted among explicitly
reviewed provenance groups is descriptive, not an evidence or causal probability.
The upstream native builder now preserves raw review, DOI and experiment/context
fields and includes context/review/typed provenance in new event signatures.
Old artifacts cannot recover fields previously discarded by deduplication; they
must be rematerialized from raw inputs to obtain this additional information.

The full candidate universe is retained. No registered support is unavailable
with NO_REGISTERED_EVIDENCE_NOT_A_NEGATIVE, never a zero confidence prediction.
SOURCE_COVERAGE.parquet reports normalized events, distinct events actually
matched to candidates, attached candidate rows, and unmatched source events.
Version suffixes and pathway families are never guessed away to improve coverage.

The output binds SOURCE_EVENTS, per-cancer EVENTS/CONTEXTS, RULE_SUMMARY and
SOURCE_COVERAGE by SHA256. The independent audit reloads the original sources,
reconstructs the attached ledger and recomputes summary counts, directions and
review fractions with DuckDB SQL instead of the pandas aggregation. It also
compares context and source-coverage outputs. SUCCESS is emitted only after this
audit passes. This is a computational consistency audit; independent biological
validation remains C13 and is explicitly false in these evidence summaries.

Actual consumption is implemented in ensemble_fold_predictions and
materialize_exact_release, via an optional rule_evidence_binding path/sha256
record; the exact-release CLI accepts matching --rule-evidence-binding and
--rule-evidence-binding-sha256 options. run_v32_current_multimodal_fusion accepts
the same record in its manifest, writes rule annotations to its prepared input,
and carries them through materialize_multimodal_fusion_release. Explicit expert
specifications control fitting features, so rule annotations never become fitted
expert inputs. All existing score values are checked unchanged during attachment.

The retired evidence_transformer is removed from the default fusion policy.
The historical run_v32_multimodal_fusion_formal entry now delegates to the current
manifest runner: use --manifest, --expected-core-run-id and --output; preparation
is the default and fitting still requires explicit --fit. It no longer launches
the hard-coded August workflow. Historical binding readers remain available for
provenance inspection. RuleEvidenceQuery provides parameter-bound exact queries
over the same summary, event and context files; website mounting remains C15.

Synthetic verification uses the actual cached-native ENCODE and RBNS commands,
rule materializer/audit, current fusion preparation, public serialization and
exact ensemble consumer. Fixed synthetic coefficients exercise serialization;
no fit or optimizer is invoked. Tests cover duplicate research, conflicting
reviews/directions, empty-source releases, nonmatching versions/families, source
and rehashed-output corruption, preserved scores and literal query filters.
Real-data preparation, evidence production, biological validation, training and
deployment have not been run as part of this code-only task.

External source isolation and observed metrics (C13, integration still open)
-------------------------------------------------------------------------

scripts/analyze_v32_current_external_validation.py is an executable analysis
entry, not a source inventory or a training command. It reads existing bound
exact/fusion producer outputs, reconstructs their scores, parses raw external
observations and computes new metrics. It runs only on 149::

  python -m scripts.analyze_v32_current_external_validation \
    --manifest /path/to/EXTERNAL_INPUT.json --output /path/to/new-external-run

The manifest format is CURRENT_EXTERNAL_VALIDATION_INPUT_V1. It contains:

* identity_crosswalk: a path/sha256 table with namespace, raw_id, canonical_id.
  Namespaces are lncrna, cancer and pathway. Mapping is explicit and unambiguous;
  versions are not removed and pathway families are not expanded. For cell-line
  cohorts the registered cancer crosswalk must explicitly map the cell line.
* models: unique model_id, run_id, producer_kind, producer_binding path/sha256,
  and score_column. exact_release accepts association_membership_probability,
  l1_probability or ridge_probability and independently recomputes the five-fold
  arithmetic ensemble from bound source predictions. fusion_release accepts
  discovery_adjusted_probability or fused_confidence_probability and replays the
  frozen coefficients on the published native inputs. No optimizer is created.
  Baseline and primary models must have the same full exact candidate universe.
* model_use_sources: unique source_id, kind (canonical_usage or native_exact_events),
  phase (TRAINING, MODEL_SELECTION, FEATURE_CONSTRUCTION or CALIBRATION), model_ids
  and a table path/sha256 record. The tables preserve lncrna_id/pathway_id plus
  cancer and study provenance when present. Their union across all compared models
  is excluded for each model, avoiding an asymmetric comparison.
* external_sources: unique source_id, kind and table path/sha256. exact_assay
  requires record_id, lncrna, cancer, pathway, label, is_experimental, is_predicted
  and assay. label is measured 0/1 or unknown; any zero requires a registered
  measured_negative_definition. Provenance may include pmid, doi, dataset_id,
  dataset_accession, source_dataset, experiment_id or study_accession.
* evaluation: exactly primary_model, cutoffs, lncrna_aggregation,
  bootstrap_replicates and seed. The aggregation is MAX, MEAN_TOP5 or the explicitly
  named LEGACY_0.7_MAX_0.3_TOP5 policy. The primary model, aggregation and cutoffs are
  supplied before evaluation; external results never select a model or threshold.

Native known-positive readers accept lnc2cancer, lncrnadisease,
rnadisease_experimental, rnadisease_predicted and gse85011_followup. Mixed
LncRNADisease methods require typed source flags or an explicit method_classes
mapping of exact method strings to experimental/predicted/unknown; unrecognized
methods remain unknown. Predicted database records cannot become measured truth.
GSE85011 uses typed control status and the registered published follow-up
selection interpretation PUBLISHED_GROWTH_MODIFIER_HITS. Selected noncontrol
RNA-seq follow-up targets support only known-positive lncRNA ranking; controls and
other metadata rows are not invented as measured negatives, and these records
do not validate individual pathways. No screen effect size is inferred from
sample inclusion.

PMID/DOI/dataset/experiment/study-token connected components exclude overlap,
including transitive multi-PMID bridges across database copies. Same-file overlap
is recorded separately. Cancer-agnostic lncRNA/pathway pairs and lncRNA identities
are compared against all registered model-use phases. Four scopes are reported:
descriptive, provenance_held_out, pair_held_out and lncrna_held_out. Missing model-use
study provenance prevents provenance-held-out eligibility; missing entity
identities prevents pair/lncRNA-held-out eligibility. Unknown mappings stay in the
isolation ledger. A cell-line label is not silently promoted to patient evidence.

Within each source/cancer/endpoint and the pooled source cohort, identical target
annotations are deduplicated. Conflicting measured labels are excluded from
metrics and counted in coverage. Their raw records remain in the isolation table.
Exact-pathway recovery ranks exact pairs within cancer; disease-only recovery
ranks lncRNAs using the registered aggregation. Rankings use the full eligible
candidate universe, with separate novel-entity universes for the relevant scopes.
Equal-score ties contribute their expected hit fraction under uniform tie breaking,
so alphabetical ordering cannot create apparent enrichment.

Observed AUROC, average precision, Brier score and log loss require both measured
classes. Known-positive sources produce recovery metrics rather than synthetic
classification labels. Recovery reports both all-eligible-positive and matched-
positive denominators. Coverage reports unmatched targets, conflicting labels,
available candidate universe and resampling block count. Missing results are null
with a reason, not zero performance.

Confidence intervals and primary-minus-baseline differences use paired resampling
blocks connected by shared lncRNA or study provenance. All models use identical
block draws. Bootstrap multiplicities are handled as observation weights instead
of expanding repeated large tables. Fewer than two blocks or insufficient valid
replicates leaves uncertainty unavailable. Intervals are explicitly exploratory
and unadjusted across the reported comparisons; they do not establish multiplicity-
controlled superiority or component-specific causal effects.

The release binds ISOLATION, MODEL_USE, METRICS, DETAILS and COVERAGE tables, model
producer provenance and the exact evaluation policy. It replays the raw-source
analysis before SUCCESS. This replay uses the same statistical implementation;
the separate numerical tests compare AUROC/AP/ties/weighted resampling against
analytic examples. CurrentExternalValidationQuery serves these same hash-bound
tables with parameterized filters and pagination.

C13 is NOT marked complete. These calculations isolate against REGISTERED_MODEL_
USE_ONLY: completeness of the inventory has not yet been derived from all actual
training, feature-construction, calibration and model-selection dependencies.
The selected-score producer is now connected through C14 below. Complete
usage-closure integration remains required alongside C16. The output records
model_use_inventory_completeness_verified=false and
independent_external_validation_attested=false. A passing replay or a supplied
inventory must not be substituted for that missing completeness proof.

Synthetic checks run the actual exact producer and current external command,
reconstruct a frozen synthetic fusion release, verify analytic metrics and paired
intervals, and reject altered producer/metric outputs. No real external cohort
was analyzed, no actual scientific gain was estimated and no model was trained.

Selected score, Gene Sets and downstream consumers
------------------------------------------------

The existing scripts/materialize_v32_release.py now accepts --selected-manifest
as an alternative to the historical --prediction-pattern. It invokes the real
selected-score materializer; it does not send the new policy through the old
primary-only ranking path. The standalone entry point is
scripts/materialize_v32_selected_analysis.py. Both use new output directories
and require host 149. Neither entry trains or chooses a model from test results::

  python scripts/materialize_v32_release.py \
    --selected-manifest /absolute/path/SELECTED_INPUT.json \
    --output /absolute/path/new_selected_analysis

  python scripts/audit_v32_selected_analysis.py \
    --binding /absolute/path/new_selected_analysis/SELECTED_ANALYSIS_BINDING.json \
    --expected-sha256 <binding-sha256> \
    --output /absolute/path/new_selected_analysis_reaudit.json

The input format is CURRENT_SELECTED_ANALYSIS_INPUT_V1. It requires:

* reference_exact_model: model_id, run_id, producer_kind=exact_release or
  current_core_release,
  producer_binding={path, sha256}, and
  score_column=association_membership_probability. This supplies the complete
  candidate universe, exact pathway/family, shared/local scope and primary model
  direction. Candidate keys and annotations are checked against the actual five
  fold files and bound candidate authority; merely rehashing changed annotations
  does not satisfy that check.
* selected_model: the same producer declaration shape. exact_release may select
  association_membership_probability, l1_probability or ridge_probability from
  the same historical exact producer. current_core_release supplies the actual
  V4 primary score and L1 only; it has no bound Ridge producer. fusion_release may select
  discovery_adjusted_probability or fused_confidence_probability. Fusion must
  bind the very same primary input SHA/run and retain its real private input
  snapshot. Coefficients are applied again to those inputs, without fitting.
* selection: either kind=EXPLICIT_SOURCE_REGISTRATION with a nonempty rationale,
  or kind=G012_VALIDATION_WINNER with a path/sha256 binding to the existing locked
  G0/G1/G2 comparison. Explicit registration records a source choice and claims
  no validated winner. A G012 lock is restricted to its exact primary score;
  all five losses, the deterministic winner, candidate authority, prepared files,
  checkpoint hashes and training receipts must agree. No new winner is selected.
* geneset: min_members (positive integer), max_members (at least min_members),
  and score_min in [0,1]. Example: 10, 200 and 0.5.
* subtype: RankedSubtypeConfig fields, explicitly including
  score_column=selected_score. Other defaults retain the existing rank/composition
  analysis; bootstrap_replicates defaults to 200. These are Gene Set composition
  subtypes across cancers, not newly inferred patient molecular subtypes.
* optional rule_evidence_binding={path, sha256}: actual observed rule annotations
  covering precisely the same candidate universe. They reach member outputs but
  never become ranking or similarity weights.

The selected_score snapshot preserves the primary score separately, as well as
the selected model/run/column and source scientific status. Selecting a diagnostic
fusion result does not upgrade its performance status or establish a new gain.
Fusion confidence remains labelled FUSED_CONFIDENCE. Association direction comes
from the bound primary model; it is explicitly not a causal regulation label or
a direction learned by the selected fusion. An exactly neutral 0.5 direction is
unknown and may form an unknown-direction Gene Set, but is excluded from the
directional subtype comparison.

Gene Sets group by cancer, exact pathway and primary association direction. They
filter and rank selected_score descending, break score ties by stable lncRNA ID,
cap max_members and require min_members. Master and member tables keep exact
pathway IDs; family is an annotation, never a substituted target. Coverage lists
all input candidate groups, including groups producing no Gene Set, threshold
counts, exported member counts, cap-boundary ties and explicit missing reasons.

Ranked subtypes now actually use selected_score for weighted Jaccard and the
selected member ordering for rank-biased overlap. Cancer-local members remain
excluded from cross-cancer profiles. The full input cancer/pathway universe is
retained even if no directional Gene Sets survive. Unavailable subtype IDs and
K values remain typed nulls, with reasons for insufficient members, directions,
pathways or cancer coverage. No subtype result feeds back into model scores.

Exact-producer fold statistics use the actual five prediction files. The sample
SD uses ddof=1. Selection frequency applies the same per-fold direction, score
threshold, minimum count, ranking cap and tie policy as Gene Sets. Mean rank
percentile compares scores within cancer/exact-pathway. Primary direction
agreement is separate. Fusion has no matched per-fold estimates for its final
candidate scores: these statistics stay null, their count is zero and the missing
reason is retained. The final fusion score is never copied five times to invent
stability. These quantities are descriptive stability summaries, not independent
confidence intervals.

Fusion materialization now records FUSION_INPUT.PRIVATE.parquet, with public=false
in its artifact declaration. This includes the actual native inputs and may
contain private labels; it is not a downloadable public table. Existing legacy
fusion readers can still reconstruct their public scores, but a selected current
release requires rematerializing its private input snapshot from authentic bound
sources. No synthetic substitute is acceptable for a real release. Availability,
nonzero contribution and exact-primary fallback flags are recomputed from the
chosen frozen endpoint and carried into selected scores and coverage. Available
experts with zero coefficients remain visible without claiming a contribution.

Outputs include SELECTED_SCORES.parquet and its binding in selected_score/, plus
GENESET_MASTER, GENESET_MEMBERS, GENESET_COVERAGE, PATHWAY_CONTEXT_SUBTYPE,
PATHWAY_CONSERVATION, PATHWAY_PAIRWISE_SIMILARITY, CANCER_PROGRAM_SUBTYPE,
CANCER_PROGRAM_SIMILARITY parquet files and SELECTED_GENESETS.gmt. Every analysis
table carries selected_score_run_id and selected_score_binding_sha256; GMT's
description carries the same identifiers. A changed policy/producer or relevant
implementation produces a different selected score identity. SUCCESS is written
only after upstream score and downstream table/GMT replay agree. The replay is
explicitly the same implementation, not an independent statistical validation.

SelectedAnalysisQuery offers exact parameterized table filters and pagination;
download(name) returns the verified same-version file declaration. It does not
expose the fusion private snapshot. C15 still needs the actual website/API routes
and frontend consumption. This query helper alone does not complete that task.

Current external evaluation additionally accepts producer_kind=selected_release,
score_column=selected_score, with producer_binding pointing to
SELECTED_SCORE_BINDING.json. Its run_id is the underlying training_run_id recorded
there; selected_score_run_id is a distinct analysis identity. The reader replays
the bound actual producer before computing external metrics. This connection does
not prove model-use inventory completeness or biological external independence.

Synthetic validation exercises the existing CLI, actual exact/fusion serializers,
both fusion endpoints, selected score swaps, analytical weighted similarities,
five-fold selection caps, unknown directions, empty Gene Sets, source-rule
annotations, same-version queries/GMT, external metric consumption and G012 lock
failures. It also rejects rehashed score, direction, membership and GMT changes.
Frozen synthetic coefficients and serialization-only checkpoint fixtures are
used; optimizers and fusion fitting functions are forbidden during these tests.
No real cohort, current scientific checkpoint inference, training or deployment
has been performed. The native current-V4 full-candidate producer is now described
below. C05 still needs the remaining auxiliary adapters and the authentic
post-selection label connection for fusion; C01/C08/C13/C15/C16 remain unfinished.

Current V4 full-candidate predictions
------------------------------------

The actual current V4 loader, residual decoder and complete runtime chunk
estimator are used by scripts/infer_v32_current_candidates.py. On host 149::

  python scripts/infer_v32_current_candidates.py \
    --manifest /absolute/path/current_prediction_input.json \
    --expected-core-run-id <selected-training-run-id> \
    --output /absolute/path/new_current_prediction --device cpu

The existing scripts/infer_v32_release_fold.py also accepts --current-manifest,
--expected-core-run-id, --output and --device for this mode. Historical per-fold
arguments cannot be mixed with it. The dedicated command avoids the historical
CUDA-only, single-graph and test-label evaluation path.

CURRENT_CORE_PREDICTION_INPUT_V1 declares core_run_id, graph_variant (G0/G1/G2),
candidate_authority={path,sha256}, lncrna_scope={path,sha256}, and fold_inputs with
exactly folds 0..4. Each fold binds config, input_manifest, prepared, checkpoint
and training_receipt in the same path/sha256 form. Supply graph_overlay,
global_binding_source and binding_expansion when used by the selected current
training inputs. The native loader verifies those graph authorities; a different
input file cannot be substituted for the selected checkpoint's source. This
command does not train, compare graph variants or select a winner.

The candidate authority contains cancer_id, lncrna_id, pathway_id and
pathway_family_id. Scope contains cancer_id, lncrna_id and
shared_or_local_scope (shared/cancer_local). All prepared candidates must match
the full candidate authority exactly. Stable node maps restore canonical row
order even when batches are ordered differently. The trained candidate file
hash is additionally checked when present in the training receipt.

Only candidate node indices, base_logit, conservation_context and graph_available
are forwarded from prepared train/validation batches. Their shapes, finiteness
and exact feature parity are checked without evaluating either label set.
Sealed test payloads are rejected. Original prepared files still carry their
training metadata; accepting them does not mean labels are prediction inputs.

Each model is frozen and encoded once per registered graph chunk. The actual
decoder is applied to every candidate batch. Chunk logits are accumulated in
float64 with the original training weights, then transformed to probabilities.
The five-fold ensemble averages these probabilities. Averaging chunk
probabilities or first averaging embeddings would change this estimator and is
not used. Unavailable graph contributions retain the exact L1 fallback. No
Ridge scores, held-out metrics or per-patient predictions are fabricated.

Outputs are FOLD_0..4.parquet, CURRENT_CORE_PREDICTIONS.parquet,
CURRENT_PREDICTION_BINDING.json and SUCCESS.json. Fold tables retain bounded
residuals/gates and graph availability; the binding retains their sources and
actual frozen parameter fingerprints. The ensemble carries exact pathway/family
and scope annotations.
The prediction scope is FULL_CANDIDATE_DISCOVERY_NOT_HELDOUT_EVALUATION. The
binding records all five original inputs, checkpoint/training receipts, runtime
weights and relevant code hashes. Its run_id identifies the prediction release;
training_run_id remains the selected training identity.

Selected-score/GeneSet and current external readers accept
producer_kind=current_core_release, producer_binding pointing to this binding,
run_id equal to its training_run_id, and score_column set to the primary or L1
probability. They verify the original sources and recompute the fold ensemble.
This default verification does not rerun the frozen checkpoint forwards.
To additionally reproduce all five model outputs on host 149::

  python scripts/audit_v32_current_candidates.py \
    --binding /absolute/path/new_current_prediction/CURRENT_PREDICTION_BINDING.json \
    --expected-sha256 <binding-sha256> --replay-models --device cpu \
    --output /absolute/path/new_frozen_prediction_audit.json

Full replay requires the recorded implementation and compares native model
outputs at absolute tolerance 2e-6. It rejects consistently rehashed fold and
ensemble scores that disagree with the frozen model. Both levels state their
scope explicitly; neither constitutes independent biological validation.

Synthetic verification exercises real V4 serialization/loading and residual
forwards across five folds and two chunks, analytical weighted-logit semantics,
candidate ordering, L1 fallback, source mismatches, label exclusion, the existing
CLI, selected Gene Sets and frozen-checkpoint replay. Query records now preserve
Python floating-point precision rather than rounding through pandas JSON's
default ten decimal places. The related 46-case suite passed on host 149 with
all optimizers forbidden; none of these fixtures is a trained scientific model.

This closes the current full-candidate primary producer in C05. The native
Mutation/CNV/ATAC fold adapters are described below. State/Clinical context
consumers and the authentic held-out label authority needed by the current
fusion trainer remain unfinished. Full-candidate discovery predictions are not
relabelled as OOF evaluation. No real candidate release has been generated.

Native Mutation, CNV and ATAC fusion consumers
---------------------------------------------

scripts/materialize_v32_native_auxiliary_fusion.py reads native model outputs
without entering any trainer. On host 149::

  python scripts/materialize_v32_native_auxiliary_fusion.py \
    --manifest /absolute/path/native_auxiliary_input.json \
    --expected-core-run-id <selected-current-core-run-id> \
    --output /absolute/path/new_native_auxiliary_adapter

NATIVE_AUXILIARY_FUSION_INPUT_V1 supplies core_run_id, native_run_id,
producer_kind, modalities, candidate_authority, patient_folds and native_receipt.
Every artifact declaration is {path, sha256}; relative paths resolve against
their owning manifest. Mutation/CNV also require core_manifest pointing to the
actual current V4 embedding export. A historical core is not promoted by
renaming its run. ATAC is a core-independent model and does not require that
export. Supported native formats and their meanings are:

* producer_kind=genomic, modalities=[mutation] or [mutation,cnv]: native
  genomic_training.py SUCCESS/LINEAGE/CHECKPOINT_MANIFEST and the five
  patient_fold_oof_predictions partitions. The target is positive event-phi
  association within each modality, not expression direction or patient risk.
  A declared NULL_WITH_REASON training fold remains unavailable and has no
  invented checkpoint.
* producer_kind=cnv_event, modalities=[cnv]: the original cnv_only_training.py
  event-cooccurrence head. Its successful output is now
  CC_HHGT_V3_2_CNV_HEAD_OOF_V2, adding the actual run, target, candidate/patient
  sources, precheck and partition hashes. The production serializer is also
  exercised by the synthetic checks without invoking training. Previously
  emitted V1 manifests lack those bindings and are not silently upgraded.
* producer_kind=cnv, modalities=[cnv]: cnv_only_training_v2.py's
  CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_V2, with fold-by-cancer partitions and
  current fold-local unadjusted expression-association proxy supervision.
  READY_FOR_INDEPENDENT_AUDIT is retained as the native status; adapter success
  does not upgrade it to an independently validated scientific release.
* producer_kind=atac, modalities=[atac]: atac_training.py's native
  SUCCESS/LINEAGE/CHECKPOINT_MANIFEST, five checkpoint-bound OOF partitions and
  typed average. The score describes promoter coaccessibility persistence,
  never a causal, directional or perturbational effect.

The two CNV estimands are kept separate: event-phi outputs register expert_id
cnv_event, while the expression-association proxy head registers cnv. Mutation
and ATAC register mutation and atac. They can be consumed together. Selecting
both the genomic CNV event producer and a separate CNV event producer would
duplicate cnv_event and is rejected by the existing fusion runner.

The adapter inspects actual private checkpoint files and their current-core
fingerprints, native source receipts, canonical patient-fold authority and
full candidate coverage. It rejects a substituted core, mismatched run or fold,
duplicate or absent candidates, probabilities outside [0,1], non-boolean
availability, unavailable values filled with zero, and contradictory native
patient counts. Canonical TCGA patient aliases cannot occupy different folds.
Zero remains a valid measured probability when availability is true.

Original fold partitions are scanned through DuckDB. Aggregation can spill to
disk under its memory limit; native wide average tables and all fold summaries
are not simultaneously loaded into pandas. Each modality's final score is the
equal mean of its available fold predictions. No available fold means null;
one available fold has unknown sample SD, not zero SD. Native aggregate scores,
availability and fold counts must agree with the actual fold files, allowing
1e-6 absolute tolerance for the native float32 averages. Original aggregate
missing reasons are retained, along with every missing fold's known reason.

Outputs are AUXILIARY_FUSION.parquet, NATIVE_FOLD_CONTEXT.parquet,
FOLD_COVERAGE.parquet, AUDIT.json, FUSION_EXPERTS.json and SUCCESS.json. The fold
context retains only keys, modality, fold, predictions, availability, reasons
and applicable patient-count columns. Observed labels and unrelated private
source columns are not copied. Predictions retain native estimands and run
identity. Fold SDs describe variation across overlapping fitted folds; they
are not confidence intervals or external-validation estimates.

Add the entries in FUSION_EXPERTS.json to the current fusion manifest's experts
list. scripts/run_v32_current_multimodal_fusion.py now invokes this native
reader, reconstructs fold aggregation and checks the native producer identity
before preparing its actual private feature table. Rehashed modifications to
the aggregate, retained fold context or coverage are rejected by reconstruction.
Aggregate replay allows absolute tolerance 1e-12 for floating-point reductions.
This reconstructs native predictions' aggregation; it does not rerun the native
models from raw omics data. AUDIT.json explicitly records that distinction.

Synthetic checks cover all four native formats, actual CLI-to-fusion preparation,
joint Mutation/CNV-event/CNV-association/ATAC consumption, declared untrainable
folds, partial coverage, observed zeros, typed nulls, patient aliases and corrupt
source/output identities. A frozen fusion model verifies that observed inputs
can change its final score and that wholly missing inputs preserve the exact
primary probability. Optimizers and the native fitting functions are forbidden.
These checks do not estimate scientific gain or execute any real cohort.

C05 remains in progress: State/Clinical native context still needs its actual
consumer, and current full-candidate primary predictions still need the authentic
post-selection label authority for fusion. The preparation tests above use
explicit synthetic primary-label fixtures; they do not establish that unfinished
connection or independent pair-held-out performance for native patient-OOF heads.

Current native State results and selected GeneSet context
-------------------------------------------------------

The existing scripts/materialize_v32_state_gene_sets.py command now accepts
``--current-manifest INPUT.json --output NEW_DIRECTORY`` on host 149. Legacy
inputs remain supported; current manifests cannot use ``--fixture-authority``
or unrecorded command-line threshold overrides. This path only consumes
completed sources and computes descriptive summaries. It cannot train a head.

The input JSON has format CURRENT_STATE_CONTEXT_INPUT_V1, core_run_id,
state_run_id, and three {path, sha256} records: core_manifest (the current V2
core export), state_training_lineage (the native STATE_TRAINING_LINEAGE.json),
and selected_analysis (SELECTED_ANALYSIS_BINDING.json). Optional state_sets
contains cancer_max_members=100, pan_cancer_max_members=200,
minimum_set_members=3, pan_cancer_min_cancers=5 and
pan_cancer_min_direction_consistency=0.60. The last threshold must exceed 0.5;
a tied positive/negative cancer count cannot establish a consistent direction.

The adapter reads actual native private-head checkpoints, verifies their tensor
fingerprints and architecture, deterministic initialization receipts and changed
parameters, and binds all five frozen core checkpoints to the selected primary
reference. It checks the native input hashes, patient-fold consistency, complete
five-fold predictions per native candidate and original aggregate. The native
producer's core run must match the selected analysis's current V4 reference,
including when the selected scores themselves came from residual fusion.

DuckDB independently reconstructs native State aggregation: available fold
predictions are weighted by max(test_n_pairs, 1), with the producer's original
minimum available-fold count. Missing folds retain their native reasons.
Predicted membership probability, observed held-out Pearson correlation and
effect direction remain separate. Measured zero is a valid neutral effect.
Unavailable values must be null, and availability must be Boolean. The adapter
does not estimate a causal effect or treat a State probability as an exact
pathway membership probability.

Two actual downstream computations share this bound source:

* Native cancer-specific and direction-consistent pan-cancer State GeneSets
  use the existing ranking algorithms and recorded set thresholds. Neutral
  effects remain in the aggregate and selected-member context, and are excluded
  from both positive and negative State sets. GMT members preserve the native
  gene-ID convention and include source-run and selected-analysis identity.
* Every selected pathway GeneSet member is joined to all seven State programs.
  Its actual selected score weights the available members' predicted membership
  and observed correlations. Profiles report member coverage and available score
  mass divided by total selected score mass. Missing members never enter the
  available-weight denominator as zero effects. No available weight produces a
  null summary with reason. All seven states also appear in coverage for groups
  that failed the selected GeneSet threshold and therefore exported no set.

The weighted correlation is explicitly a descriptive mean over selected
members. It is not a correlation of joint GeneSet activity, a posterior
probability, a p-value or an independent validation result. Pathway association
direction and State correlation direction are different columns. State context
does not change the already selected primary/fusion ranking.

Outputs include native_state_aggregate.parquet, state_geneset_catalog.parquet,
state_geneset_members.parquet, state_genesets.gmt,
selected_member_states.parquet, selected_geneset_state_profiles.parquet and
selected_state_coverage.parquet. CURRENT_STATE_CONTEXT_BINDING.json binds the
input manifest, source artifacts, selected version, policy, implementation and
outputs. A successful source recomputation writes RECOMPUTATION_AUDIT.json and
SUCCESS.json. Rehashing a modified profile cannot bypass recomputation.

CurrentStateContextQuery(binding_path, expected_sha=..., selected_analysis_sha=...)
provides exact, parameterized filters and version-bound downloads. The default
query table is selected_geneset_state_profiles; member, coverage and native
State tables are separately queryable. A different selected-analysis identity
is rejected. This query class is not yet a website/API route.

Synthetic verification uses real tiny V4 checkpoint loaders, all-chunk frozen
inference/export, the selected-analysis materializer, native-format State
checkpoint serialization, and the existing State command. Optimizers and
native State fitting are forbidden. The tests exercise observed zeros, neutral
effects, missing measurements, absent members, empty sets, invalid source
identity, five-fold incompleteness and changed selected-score weights. Raw
measurement statistics and private State model predictions are not replayed;
the binding and audit explicitly record both limits. No real cohort was run.

C05 remains in progress: Clinical native context and authentic post-selection
fusion-label authority still require implementation. C15 also still needs the
actual API/frontend and full multi-module download integration. This change
completes the native State-to-selected-context subpath, not those requirements
as a whole.

Current native Clinical results and selected analysis
----------------------------------------------------

scripts/materialize_v32_current_clinical_context.py accepts
``--manifest INPUT.json --output NEW_DIRECTORY`` and requires host 149. It reads
completed Clinical results, replays recorded computations and materializes the
selected analysis's clinical context. It never calls a neural-head trainer or
an optimizer. Existing training commands only gained provenance fields; they
were not run as part of this code change.

INPUT.json has format CURRENT_CLINICAL_CONTEXT_INPUT_V1, core_run_id,
patient_risk_run_id, entity_run_id and optional fdr_threshold (default 0.05).
The required {path, sha256} records are core_manifest, selected_analysis,
patient_risk_lineage, patient_risk_predictions, entity_lineage and
entity_calculation_manifest. These identify the current V2 frozen-core export,
completed selected analysis, native clinical MODULE_LINEAGE.json and root
patient risk parquet, ENTITY_CLINICAL_LINEAGE.json and CALCULATION_MANIFEST.json.

The selected primary reference must be the actual current V4 producer. Its
five core checkpoint and parameter identities must match those used by the
clinical private heads. All five native checkpoint files are inspected and
loaded with their recorded architecture and initialization. The original
input audit and hashes are checked, and patient identity retains the native
explicit patient_id contract rather than truncating TCGA-like identifiers.
Patient risk and entity associations must use the same outcome workbook and
patient-fold assignments. Entity annotations must bind the selected candidate
authority; incomplete typed universes and duplicate folds fail closed.

New patient-risk producer receipts include feature_artifact (path and SHA256)
for patient_features.fold_local.parquet, and the root lineage records the
public prediction artifact. When these feature records exist, the consumer
replays the frozen private head on the held-out rows and compares predictions
at absolute tolerance 1e-6 to accommodate FP32 device arithmetic. It does not
recompute the projection from raw measurements. Older native receipts without
feature hashes remain readable but explicitly record an empty or partial
patient_risk_frozen_replay_folds list; merely loading a checkpoint is not
reported as replaying its predictions.

The entity producer now stores calculation_config in its native calculation
manifest and lineage. This preserves the original minimum eligible-fold
threshold and its config hash for downstream replay. For an older native
result that only stored config_sha256, the context manifest must additionally
bind an entity_config JSON whose canonical hash matches that original receipt.
There is no default threshold substitution or fixture-authority bypass.

Entity summaries are reconstructed from each native cancer/subject partition's
five held-out beta/standard-error records using the existing random-effects
Cox score summary. The original full native subject-type/cancer/endpoint FDR
family is preserved. It is not recomputed over the smaller selected GeneSet.
This is a replay of the same statistical implementation, not independent
biological validation or a rerun of raw patient-level Cox statistics.

The two Clinical units remain distinct in the public context:

* Native entity results retain beta, standard error, hazard ratio, confidence
  interval, p-value, FDR, heterogeneity, fold support and typed missing reasons.
  The native field named clinical_relevance_probability is exposed as
  one_minus_fdr_score with an explicit non-posterior definition. A measured
  zero beta has a neutral effect direction; the original native direction is
  retained separately for traceability. DFS remains unavailable with
  NO_DISTINCT_DFS_SOURCE. PFS, PFI and DFI are not silently renamed or pooled.
* Patient-risk distributions are summarized separately for each cancer,
  endpoint and held-out fold, with that fold's recorded training time bins.
  Coverage counts every authorized patient, including those absent from the
  native molecular-feature intersection. Missing risks remain unavailable,
  while an observed probability of zero is valid. No shared-time calibration
  or cross-fold pooled risk distribution is claimed. These context tables
  contain no patient identifiers or private outcome labels.

Selected GeneSet members keep their actual rank and selected score alongside
the native lncRNA Clinical association. The consumer calculates the selected
score mass of risk/protective/neutral members meeting the recorded FDR
threshold, divided by available selected score mass. It also reports member
coverage and selected weight coverage. Missing members do not count as negative
clinical evidence, and zero available weight yields null support fractions.
Six-endpoint coverage persists for candidate groups that exported no GeneSet.
The corresponding exact pathway's native survival association is a separate
table and is explicitly associated with measured pathway activity.

These support fractions are descriptive member summaries. They do not average
member hazard ratios into a GeneSet hazard ratio, estimate joint GeneSet survival
effects, establish causality, provide a new GeneSet p-value or alter the selected
primary/fusion ranking. Joint GeneSet survival analysis is not claimed by this
context adapter.

The six public outputs are native_entity_clinical.parquet,
patient_risk_fold_coverage.parquet, selected_member_clinical.parquet,
selected_geneset_clinical_support.parquet, selected_clinical_coverage.parquet and
selected_pathway_clinical.parquet. CURRENT_CLINICAL_CONTEXT_BINDING.json binds
sources, policy, implementation and outputs. RECOMPUTATION_AUDIT.json and
SUCCESS.json record source replay. CurrentClinicalContextQuery requires the
expected context hash and selected_analysis_sha, supports exact parameterized
filters and offers same-version downloads. It is a query implementation, not
yet an API/frontend route or deployment.

Synthetic checks use actual tiny current V4 loaders/exporters and selected
analysis, serialized native Clinical heads with frozen forward passes, and
native vectorized Cox statistics on synthetic patients. Optimizers and private
head training are forbidden. Tests include changed checkpoints/features/risk,
patient-fold drift, missing folds, invalid standard errors, rewritten summaries,
private-label leakage, DFS, empty sets and changed selected-score weights.
No real patient data, model training or production deployment was executed.

C05's native Clinical-to-selected-context connection is implemented; authentic
post-selection labels for current fusion remain unfinished. Global completeness
of cross-module patient/use inventories remains C16, and actual public routes
and frontend consumption remain C15.

Native primary validation labels for current fusion development
--------------------------------------------------------------

scripts/materialize_v32_current_primary_fusion.py accepts
``--binding CURRENT_PREDICTION_BINDING.json --expected-sha256 SHA256`` plus
``--expected-core-run-id RUN_ID --output NEW_DIRECTORY``. It requires host 149
and consumes a completed current V4 prediction release. The command validates
the five native checkpoints and their exact authorized prepared payloads using
the existing current loader, including graph overlays and binding expansions.
It constructs no model, optimizer or replacement labels. Native training and
validation features must match and cover the exact candidate authority.

The adapter extracts validation_batches.proxy_label and the explicit boolean
association_available tensor. It never assumes that a missing availability
mask means all labels are observed. Prepared input_scope must identify only
validation replication labels, train-patient features, and no sealed-test
access. Sealed test payloads remain forbidden. Model outer fold f and actual
validation patient fold (f + 1) modulo 5 are recorded separately.

An unavailable native loss entry may contain a zero filler. The exported private
validation_proxy_label is null in that case, with a boolean availability column
and a reason noting that the original detailed reason was not serialized.
Observed zeros remain valid negative labels. Neither direction/effect labels
nor test labels are used as fusion inputs. Label extraction restores exact
candidate order through the native node maps; it does not join on guessed IDs.

The output includes five FOLD_n.PRIVATE.parquet files,
CURRENT_PRIMARY_FUSION_BINDING.json, FUSION_PRIMARY_INPUTS.json and SUCCESS.json.
The JSON fragment supplies core_run_id, primary_prediction,
primary_binding_audit and primary_folds for the existing
CURRENT_MULTIMODAL_FUSION_MANIFEST_V1. Add the expert entries emitted by the
native auxiliary or signed single-cell adapter, then call
scripts/run_v32_current_multimodal_fusion.py. Its default remains prepare-only.
The consumer re-extracts every label from the checkpoint-authorized source;
editing a label file and recomputing its output hash cannot bypass this replay.

All five native validation views must cover the complete candidate universe.
The fusion target is the mean of available validation labels, with the actual
number of available folds retained. A wholly unavailable target remains null.
Fitting, model selection and metrics use only eligible targets, while frozen
application still predicts every candidate. All five pair folds must contain
eligible targets before any optimizer can start. A small preparation with fewer
pair folds is allowed but is explicitly marked ineligible for fitting.

These are development labels already used by the primary model's validation
and checkpoint selection. The source, private frame, metrics, public scores
and lineage retain VALIDATION_REUSED_FOR_FUSION_DEVELOPMENT. Any resulting
increment is DEVELOPMENT_INCREMENT or DEVELOPMENT_NO_INCREMENT; scientific
status is DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED. Pair cross-fitting is
conditional on the already fitted primary and is not independent validation
of the full system. Existing selected-score consumers carry this status into
GeneSet context. A release cannot upgrade this status to SECONDARY_VALIDATED.
Independent performance assessment still requires the separate frozen-winner
evaluation/external-validation workflows and complete model-use inventories.

Older bare primary hash lists remain usable only for compatibility preparation.
They are marked UNVERIFIED_LEGACY_HASH_LIST and ineligible for current fitting;
--fit rejects them before training starts. This compatibility path does not
count as verification of authentic primary labels.

Synthetic checks cover actual tiny V4 producers, native checkpoint/source
replay, current preparation CLI consumption, masked and partial-fold labels,
real zeros, full-candidate frozen predictions, rejection of rewritten labels,
and propagation of development-only status. Cross-fit orchestration tests
substitute explicit frozen parameters for fitting; no optimizer or real cohort
analysis is executed. A successful code test is not a new trained fusion result.

This completes C05's remaining code path for native primary supervision.
C05 is code_implemented_fixture_verified, with real-data execution recorded as
NOT_RUN_AS_REQUESTED. C01, C08, C13, C15 and C16 remain open; the overall code
completion goal is not yet complete.

Pharmacology features in the actual drug head and frozen ablation
---------------------------------------------------------------

scripts/run_v32_drug_training.py now accepts --pharmacology-manifest for both
the existing formal streaming runner and the development dense runner. The
optional source is DRUG_PHARMACOLOGY_INPUT_V1 with three {path, sha256} records:
target_annotations, structures and assay_conditions. They are verified against
the actual target authority and native response/model identities; no fuzzy
drug/target matching is performed in this encoder. The input tables are:

* target_annotations: drug_id, gene_id, action_type, affinity_type,
  affinity_value, affinity_unit, affinity_relation and source_id. Missing
  measurements use nulls for the complete affinity tuple. The supported
  quantitative endpoints are Ki, Kd, IC50 and EC50, kept separate. Supported
  concentration units are M, mM, uM, micro-symbol M, nM and pM; p_molar explicitly
  denotes an already negative-log10-molar value. Concentration inequalities
  reverse under the logarithmic conversion. Censored values remain bounds and
  never enter exact-point medians. Native p-scale measurements are not treated
  as statistical p-values or as automatically equivalent endpoints.
* structures: drug_id, smiles and source_id. RDKit parses the actual molecule,
  preserves stereo and disconnected fragments, and computes canonical SMILES,
  six molecular descriptors and a 256-bit radius-2 chiral Morgan fingerprint.
  Conflicting structures, invalid SMILES and absent structures have separate
  unavailable reasons. No identifier hash is substituted for molecular data.
* assay_conditions: dataset_id, cell_line_id, drug_id, assay_id, dose_value,
  dose_unit, exposure_hours and source_id. Each protocol must match an observed
  native assay key and an unambiguous canonical model/cancer. Response-derived
  fields are rejected. Conflicting conditions for one native assay ID fail.

The DrugCentral staging reader also retains native STRUCT_ID, ACCESSION,
ACTION_TYPE, ACT_TYPE, ACT_VALUE, ACT_UNIT, RELATION, ACT_SOURCE, ACT_COMMENT and
MOA fields when present, under native_ column names. It does not infer a unit or
convert ACT_VALUE. Automated conversion of these retained raw fields into the
canonical pharmacology input still requires its explicit native-source adapter.

The new input is consumed by actual candidate feature construction and the
existing private head. It adds action-specific target embedding means and, for
each quantitative endpoint separately, the mean of the target embedding times
that target's median exact p-molar value. Availability is explicit for every
additional target pool. These are interaction covariates for a learned model,
not signed biological effects or estimates of occupancy. Swapping which gene
is inhibited or swapping target potencies changes this representation even
when the old mean target embedding and drug-level summaries remain identical.

Domain covariates include action fractions, separate exact/censored endpoint
coverage and exact-point medians, real structure descriptors/fingerprints, and
the observed assay protocol distribution within the requested cell-model split.
Dose range/median, vehicle fraction, exposure time and protocol count are never
calculated from a different fold's cell models. Zero concentration is an
observed vehicle, with a separate positive-dose flag for logarithmic features;
zero exposure is also observed. Missing public values stay null. Only the
model tensor uses zero fillers with explicit availability indicators. Existing
domain scaling is fitted on training inputs only.

Protocol-distribution covariates do not convert the existing cross-cell-line
association-strength label into a dose-specific efficacy or patient-treatment
label. Public predictions carry pharmacology_features_consumed and explicitly
set dose_specific_response_claimed=false. The separate measured dose-conditioned
analysis is described below. The legacy abs(rho)-based strength estimand is not
renamed as sensitivity/resistance merely because pharmacology features exist.

PHARMACOLOGY_FEATURE_RECEIPT.json, native head checkpoints and run lineage bind
feature order, the source hashes, implementation, RDKit version and target-pool
definitions. The formal runner snapshots the extra source files, accounts for
the extra domain/target-pool arrays and cache in its memory preflight, and stops
before feature reservoirs or optimization if the verified budget is insufficient.
Both public runner functions require host 149 and write LAUNCH_PREFLIGHT.json.
Legacy six-domain-feature heads remain readable and explicitly lack the new
feature receipt; they are not reported as having consumed pharmacology.

scripts/analyze_v32_drug_model_attribution.py accepts the same optional
pharmacology_manifest artifact, but it must match the selected native head's
recorded source/schema. Target or exact-pathway deletion recomputes the original
target mean, the action-specific pools, endpoint-specific interaction pools,
and all target pharmacology summaries before a new frozen forward. Structure
and assay design are retained. Outputs record target_pharmacology_recomputed.
The effects remain model-input ablations without a biological causal claim.

scripts/materialize_v32_drug_pharmacology.py provides a preparation-only command
on 149. Its DRUG_PHARMACOLOGY_PREPARATION_V1 manifest binds pharmacology_manifest,
drug_gene_target, raw_drug_response and cell_line_map, plus the cell-fold seed.
It writes normalized target measurements, actual drug structure/feature rows,
and explicit train/validation/test protocol context for all five folds. This
output is not a new trained model or a dose-response fit.

Verification forbids optimizers, stops both existing runners at the first fit
boundary after inspecting their real feature tensors, and loads five serialized
native private heads for frozen attribution. It checks unit conversion,
censoring, invalid/ambiguous structures, observed zeros, model-fold isolation,
changed target-action/target-potency assignments, actual forward sensitivity,
source tampering and the original signed association paths. Project-local RDKit
2025.9.3 was installed on 149 for these checks; the shared environment was not
modified. The dependency is in requirements/requirements-drug-pharmacology.txt.
The fingerprint API follows the official documentation:
https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html .
DrugCentral's source-specific activity scale must be handled explicitly; its
native pharmacology description is documented by its authors at
https://pmc.ncbi.nlm.nih.gov/articles/PMC10692006/ .

C08 remains in progress. Native pharmacology extraction and identity integration
are described in the next section. Measured dose-conditioned analysis follows;
actual selected-analysis/public Drug consumption remains unfinished.
No real compound/cohort processing,
optimization, paid compute or deployment was performed for this change.

Native DrugCentral and assay-protocol adaptation
------------------------------------------------

``python -m scripts.materialize_v32_native_drug_pharmacology --manifest INPUT.json
--output NEW_DIRECTORY`` runs only on host 149. INPUT.json has format
NATIVE_DRUG_PHARMACOLOGY_V1, staging_manifest and drugcentral_structures as
{path,sha256} records, drugcentral_activity, and an assay_protocols list. The
staging manifest must be the successful native stable-ID producer's actual
STAGING_MANIFEST.json. Its original DrugCentral/HGNC/identity-crosswalk files
and its target, response, model-map and native-drug-map outputs are verified.
The existing native target mapper is replayed; target pairs must equal that
staging's authority. Original native row numbers are now retained, allowing
non-human and missing/ambiguous mapping exclusions to be recorded separately.

drugcentral_activity declares value_scale and relation_scale independently as
NEGATIVE_LOG10_MOLAR or CONCENTRATION, and includes schema_reference. These
settings are part of the hashed manifest. No default scale is inferred from
ACT_VALUE or an empty ACT_UNIT. A p-molar value whose inequality is expressed
on the concentration scale reverses its relation before normalization. A
concentration requires its native molar unit; conflicting scale/unit declarations
and invalid values fail. Unsupported endpoints or absent measurement metadata
retain the target/action row with unavailable quantitative fields and an explicit
adaptation reason. Raw values, units, endpoints, relations and row identities
remain in the adaptation audit; censored values never become exact estimates.

The native DrugCentral structures.smiles.tsv reader uses its ID, INN, SMILES
and optional InChIKey fields. It invokes the same stable identity resolver used
for the actual assays and targets. Conflicts and out-of-universe structures
remain in the disposition table. RDKit computes the actual molecular features
through the existing encoder. The source formats were checked against the
official download descriptions and table headers:
https://www.drugcentral.org/download .

Each assay_protocols entry contains a source {path,sha256}, condition_kind,
columns and constants. Native column names are explicitly mapped to dataset_id,
cell_line_id, source_drug_id, assay_id, dose_value, dose_unit and exposure_hours;
constants can supply explicitly declared dataset/unit/protocol metadata. A field
cannot be supplied by both mechanisms. The four identity fields are required.
POINT_CONCENTRATION rows join the exact staged native drug map and an observed
response identity, including source-scoped targetless compounds. Replicates with
conflicting conditions fail the downstream canonical validator. Zero dose and
zero exposure remain observed. No response column is copied to the features.
FITTED_CONCENTRATION_RANGE rows receive RANGE_IS_NOT_POINT_DOSE; their endpoints
are not expanded into imaginary dose measurements. An empty protocol list keeps
dose/exposure features unavailable rather than manufacturing a standard protocol.

The command emits canonical target_annotations, structures and assay_conditions,
native disposition tables, NATIVE_ADAPTATION_AUDIT.json and PHARMACOLOGY_INPUT.json.
The latter is passed unchanged to the existing training CLI's
``--pharmacology-manifest`` option or the feature-preparation command. The actual
encoder verifies the native audit against the three canonical outputs, retains
all upstream origin hashes in its checkpoint receipt, and rechecks raw files and
staged directory trees at consumption. Origin records remain separate from the
three model feature tables: raw response values are not labelled static features.
Streaming resource/snapshot checks include this complete origin set.

Synthetic checks execute the existing native staging producer and this adapter,
then load the emitted manifest through the actual encoder. They cover quantitative
scale/relation conversion, unavailable metadata, conflicting structures/identities,
targetless protocols, observed vehicle, unsupported endpoints, raw-source tampering,
and the native target mapping disposition. The existing dense/streaming pre-fit
checks and five-head frozen attribution checks also run; no optimizer starts.
C08 is still open for current selected/public Drug consumption. Measured dose
analysis is implemented below. No real biological data were processed.

Measured point-dose response and matched-model dose contrasts
-------------------------------------------------------------

The existing ``python -m scripts.analyze_v32_native_drug_response`` command now
accepts ``--dose-manifest POINTS.json`` in addition to its existing exact,
response, expression, mapping, subtypes and targets arguments. It still emits
the original assay-summary analysis, and additionally consumes actual native
point measurements through cc_hhgt/v32/drug_dose_response.py. Execution requires
host 149; the command never constructs an optimizer or trains a model.

POINTS.json has format NATIVE_DRUG_POINT_RESPONSE_V1, staging_manifest as a
{path,sha256} record, and a nonempty measurements list. Each measurement entry
contains a source record, measurement_level=OBSERVED_POINT_RESPONSE,
condition_kind=POINT_CONCENTRATION, columns and constants. The same native
protocol identity adapter maps dataset_id, cell_line_id, source_drug_id and
assay_id to the actual staging authority. The supplied analysis arguments must
hash-match that producer's exact candidates, raw response, expression, model
map, targets and subtype annotations. Unmatched native identities remain in
the private disposition table, rather than being silently mapped by names.

Explicit native fields additionally provide dose_value, dose_unit,
exposure_hours, response_value, higher_is_sensitive, response_metric and
response_unit. response_value must be an actual measurement column, not a
constant. Supported response_metric values are POINT_VIABILITY, POINT_GROWTH,
POINT_LOG_FOLD_CHANGE and POINT_INHIBITION. The response unit and orientation
are declared, never inferred. Different endpoint/unit combinations are not
pooled. AUC, IC50 and other fitted-summary fields cannot be relabelled as point
measurements; a concentration range is not a series of observed doses.

Concentrations are converted to exact decimal molar keys: 0.1 uM and 100 nM
refer to one condition. Exposure retains its exact decimal key as well as the
numeric display value. Observed dose zero and exposure zero remain measured.
Missing dose, exposure or response produces an explicit disposition; a known
unit with no dose value remains missing. Invalid units/negative values and
conflicting records for one assay identity fail. Replicated copies of the same
native assay do not change its weight. Technical assays are averaged once per
dataset/canonical model/condition before either point or contrast analysis.

Point associations preserve concentration, exposure, endpoint and unit. They
use the existing signed rank-association engine, explicit cancer/subtype
contexts, full exact-lncRNA universe, minimum matched-model support and five
fixed model-fold summaries. Targetless assayed drugs remain eligible. Candidate
rows with missing expression or no finite response remain null with reasons.
The source model map is unchanged across doses, so folds do not change when
one concentration has fewer observations.

Adjacent-dose contrasts pair the same canonical model within the same dataset,
cancer, drug, endpoint, unit and exposure. The response is higher-dose minus
lower-dose sensitivity-oriented measurement, after technical-assay averaging.
Doses observed only in different datasets or disjoint models cannot create a
paired response. Intervals are the adjacent declared doses for that cancer/drug
context, selected without looking at the response or association. They include
vehicle-to-dose comparisons and retain an unavailable candidate when no models
are paired. Signed expression/change associations and subtype directions are
computed from those actual paired measurements, not differences between two
unmatched cohort correlations.

Descriptive mean/median/SD, measured unique-model count, contributing datasets
and within-dataset variation are emitted alongside association availability.
Each model contributes once to descriptive summaries after averaging its
dataset measurements. These measurement-support fields are separate from the
number of models with matched expression used by the association. An observed
constant/zero paired change stays observed (including a mean of zero), while its
undefined rank correlation remains null. No fitted dose-response curve, causal
effect, clinical efficacy, p-value or independent performance validation is
claimed. Cell-fold statistics remain descriptive replication summaries, not
out-of-fold model predictions. The original strength head is unchanged.

Outputs include dose_associations/, dose_contrasts/ and
DOSE_CONTEXT_COVERAGE.parquet. Public aggregate rows contain no individual cell
IDs or response values. PRIVATE_DOSE_IDENTITY_DISPOSITION.parquet and
PRIVATE_DOSE_MEASUREMENT_DISPOSITION.parquet retain native audit identities and
are explicitly ineligible for public downloads. The existing AUDIT.json binds
all output hashes, native measurement/staging origins, code and settings, with
dose_response_analysis describing the actual measured-data analysis performed.
Sources are rechecked before SUCCESS.json is written.

Synthetic verification covers opposite subtype directions at fixed dose,
equivalent dose units, exposure/endpoint isolation, targetless compounds,
technical duplicates, observed zero, missing measurements, pairing failures
across datasets or models, and the actual native-staging-to-existing-CLI path.
The single-dataset correlation expectation preserves the existing Fisher
transform's documented-in-code clipping at 0.999999. The real native sources
and trained models were not executed. C08 remains open for current selected
Drug context and public query/download consumption.

Current selected Drug context and native replay (2026-10-16)
---------------------------------------------------------

``scripts/materialize_v32_current_drug_context.py`` consumes the actual selected
score/GeneSet release together with the native sparse Drug release, signed
response scan and five-fold frozen ablation. Its input format is
CURRENT_DRUG_CONTEXT_INPUT_V1. It requires core_run_id, drug_run_id and
``{path, sha256}`` records named core_manifest, selected_analysis,
sparse_query_manifest, attribution_manifest, attribution_audit and
response_scan_audit. Paths are resolved against the input manifest. Execution
and independent replay auditing require hostname 149; no fitting is invoked.

The selected exact-reference producer and Drug heads must use the same five
current core checkpoints and candidate universe. Native configuration, input
hashes, private-head initialization, finite tensors, changed parameter hashes,
optimizer receipts and frozen-core identities are checked. These checks consume
existing completed training artifacts; they do not train or fabricate a new
real model. Optional pharmacology must match the actual trained source/schema.

The existing signed-response CLI is replayed from its bound native inputs,
including point-dose analysis when present. The actual attribution CLI is
replayed with all five frozen heads. Bounded multiset comparisons check the
published tables against the replay, including subtype/condition coverage.
Ablation baseline means and fold masks must also match the native public
ensemble. Public predictions outside explicitly requested attribution queries
are source-bound but are not claimed as independently forward-replayed.

The materializer emits selected members and available drug predictions without
expanding every member against every drug. It keeps association-strength model
probabilities separate from measured signed associations and model-input
ablation deltas. Response summaries report positive/negative/neutral available
member weight fractions, with member and selected-weight coverage; absent
correlations do not become neutral observations. Dose/exposure/endpoint/unit
identities stay separate, and observed zero changes remain measured even when
the correlation is undefined. These are descriptive member summaries, not
joint GeneSet response estimates, causal effects or clinical efficacy.

Target ablation rows remain drug/lncRNA-specific. Only native
EXACT_PATHWAY_TARGET_SET experiments contribute to a selected pathway's
attribution profile. Fold deltas are averaged within member before selected
score weighting. The result remains nonadditive. Chemical structure descriptors
and action/potency annotations are available when consumed by the native head;
potency's negative-log-molar scale is explicitly named to avoid confusing it
with a statistical p-value. No raw cell measurements, model IDs, private
disposition tables or fingerprint arrays are exposed as public downloads.

``CurrentDrugContextQuery`` provides bounded, parameterized same-version table
queries and hash-verified downloads. ``member_predictions(geneset_id=...,
drug_id=...)`` resolves missing predictions through the actual native sparse
factors. It revalidates those factors for each request and never fills absent
predictions with zero. DC:, CHEMBL:, ASSAY: and legacy DRUG: identifiers now
survive the actual sparse writer/reader; identity case is preserved, and
malformed namespaces remain rejected. This fixes the previous DRUG:-only
publication validator that contradicted the stable-ID staging implementation.

The context binding records source hashes, semantics, selected-score identity,
native-replay scope and implementation hashes. A second bounded SQL comparison
recomputes the derived tables before SUCCESS. Independent audit can rerun both
native analyses and context aggregation. C15 still owns real website/API route
and frontend integration, and C16 owns final orchestration/model-use inventory.
No real cohort analysis, optimizer, paid compute or deployment is performed by
the code-verification work.

Current selected website, HTTP routes and public downloads (2026-10-16)
-------------------------------------------------------------------

``python -m scripts.materialize_v32_current_analysis_web --manifest INPUT.json
--output NEW_DIRECTORY`` registers actual outputs on host 149. The input format
is CURRENT_ANALYSIS_WEB_INPUT_V3. It requires a release_id, a selected_analysis
``{path, sha256}`` record, and components with exactly these nine keys: state,
clinical, drug, evidence, external, single_cell, patient, genomic and distal.
Each component supplies either
``{"status":"AVAILABLE","binding":{"path":"...","sha256":"..."}}``
or an explicit NOT_PRODUCED, SOURCE_UNAVAILABLE or NOT_APPLICABLE status and a
nonempty reason. Omitted components and broken configured sources fail.

Every selected context must bind the same selected-analysis hash. Rule
evidence must cover the actual reference producer's candidate universe and,
when selected scores already carry rule annotations, use that exact binding.
External metrics must evaluate the selected score snapshot or its exact
selected producer as their primary model; merely including that model as a
comparator is insufficient. Full external model-use completeness remains open.

The output CURRENT_WEB_BINDING.json records the selected run, component source
bindings, public files, source files, schemas, row counts, sizes, hashes and
materializer implementation hash. Publication does not rerun optimization.
Evidence provenance is streamed into new public Parquet files with machine
paths removed, while source hashes and experiment lineage remain. Native
evidence files are unchanged. Other verified public outputs are referenced
directly; moving a release requires retaining those pinned files. Raw patient
and cell-model measurements, private Drug dispositions and checkpoints are not
download entries. Source bindings and files are rechecked before completion.

The existing ``website.backend.app:app`` activates this mode only when both
CANCERLNCATLAS_CURRENT_WEB_BINDING and CANCERLNCATLAS_CURRENT_WEB_SHA256 are set.
A mismatching binding prevents startup. This mode serves the new page at the
site root, uses the actual selected readers and excludes historical API/page
routes. It cannot be combined with the old V3.2 staging flag. Without the new
configuration, the existing historical mode and its V3.1 requirement remain.
No production process or pointer was changed during this work.

The HTTP contract is:

* GET /api/current/catalog: available components, explicit missing statuses,
  public tables, schemas, exact filter names and immutable file hashes.
* POST /api/current/tables/{table_id}/query: version (the web binding SHA256),
  filters, limit (1--1000) and offset (0--1000000). Filters are parameterized
  exact strings; unknown fields and invalid values are rejected. The response
  retains nulls and zeros and includes has_more.
* GET /api/current/downloads/{table_id}/{part}?version=...: a registered file,
  with its content hash, web version and selected-analysis hash in headers.
* GET /api/current/drug-members?geneset_id=...&drug_id=...&version=...: native
  sparse member predictions, including typed missingness, with pagination.

The catalog supplies the version used by every query and download. Stale
versions receive 409; bad queries receive 422; unregistered downloads receive
404. Changed artifacts receive a generic 503 without their filesystem paths.
The online table reader uses bounded DuckDB rather than loading a full graph.
The Drug missingness endpoint also revalidates its native source factors.

The Chinese frontend provides module/table selection, exact identity filters,
member drill-through, pagination, detailed fields, version details, all public
download parts and native Drug missingness lookup. A saved link retains its
analysis version and filters. Numeric zero stays zero; unavailable estimates
remain explicit. Descriptive member summaries, measured response direction,
model probability and nonadditive ablation retain their separate meanings.
The external page explicitly discloses the incomplete model-use inventory.

Verification uses native synthetic producers, HTTP calls to every table and
download, and a temporary loopback instance of the actual main application on
149. Chromium exercises filters, module navigation, missingness lookup, GMT
download, stale links and mobile layout. Its process is stopped after testing
and also has an independent 90-second stop timer. Optimizer constructors are
forbidden. Test/browser dependencies live in project-specific verification
directories; the shared scientific environment is unchanged. Runtime version
pins are in requirements/requirements-current-web.txt. Screenshots and browser
receipt are under provenance/code_completion_20261015/current_web_*.

These routes cover selected scores/GeneSets, State, Clinical, Drug, rule
evidence, registered external evaluation, signed single-cell details, patient
subgroup statistics, native genomic details and distal regulatory statistics.
C13 still requires complete model-use provenance; C16 owns the final
orchestration, data dictionary and acceptance pass. This is code and synthetic
verification, not a new real analysis, independent biological validation or
deployment. The requirement ledger records the current acceptance status.

Current signed single-cell and patient subgroup views (2026-10-16)
----------------------------------------------------------------

``python -m scripts.materialize_v32_current_single_cell_context --manifest
INPUT.json --output NEW_DIRECTORY`` consumes CURRENT_SINGLE_CELL_CONTEXT_INPUT_V1
with selected_analysis and native_binding hash records. The latter is the actual
AUDIT.json emitted by materialize_v32_signed_single_cell_fusion. The native
candidate universe must match the selected reference producer exactly. The
activated pointer, OOF predictions, donor crosswalk, independent-validation
receipt and every native output are hash checked. Preparation replays the
existing donor aggregation from those sources and compares all private and
public native tables with bounded SQL. It does not run model forwards or fit
an SC model. Floating aggregates use an absolute comparison tolerance of 1e-12.

The release preserves pooled positive/negative strength, per-cell context,
per-dataset context, selected-member support, selected-member cell contexts and
selected-GeneSet context summaries. Dataset context averages repeated aliases
within one canonical donor before summarizing donors. Pooled native context
continues to reconcile donors reused across studies. Dataset-level donor
counts are not additive across studies. Both donor counts and outer-fold
coverage are retained. Donor standard deviation describes observed spread in
the donor predictions; it is not a confidence interval or model uncertainty.

Context availability applies the native minimum-independent-donor threshold.
Insufficient contexts retain their descriptive native measurements with an
explicit support status; they do not enter selected-GeneSet weighted summaries.
Missing contexts and zero association are separate. Summary denominators use
the complete selected GeneSet membership and score weight, including members
without a supported context. The resulting mean is a descriptive member
profile, not a joint GeneSet activity estimate. If the selected fusion actually
contains signed single-cell inputs, its source inventory must contain this
same native binding. Otherwise used_in_selected_score is explicitly false.
Raw donor identifiers stay in bound private source files.

``python -m scripts.materialize_v32_current_patient_context --manifest INPUT.json
--output NEW_DIRECTORY`` consumes CURRENT_PATIENT_CONTEXT_INPUT_V1 with
selected_analysis and a nonempty analyses list. Each list entry has a unique
source_id and a native_binding hash record pointing to the actual patient
heterogeneity AUDIT.json. Every analysis must cover the same complete candidate
universe as the selected reference producer. Separate subgroup definitions and
analysis sources remain separate, even when they concern the same candidate.

The existing patient analysis command now records its original grouping mode
(declared groups or mutation calls), mutation source column names and exact
implementation hashes. It rechecks raw source files/dataset manifests before
writing completion. Older receipts lacking this information are rejected for
the current context rather than guessed; they need rematerialization. Current
preparation replays the original statistical command with its recorded sources,
covariates, fold settings and grouping policy. It compares interaction tables,
subgroup OOF metrics, per-patient predictions and fit receipts. This is a replay
of the same statistical implementation, not independent external validation.

Public tables retain group-specific slopes, their difference, HC3 uncertainty,
native family-wide P/FDR values, sample counts, missingness and the original
candidate-selection scope. Patient OOF errors remain in separate metric tables.
Selected views do not recompute FDR after selecting GeneSet members. GeneSet
coverage reports available members and weight, opposite group-direction counts
and counts supported at native family FDR <= 0.05; no mean member slope or joint
GeneSet interaction is fabricated. Neither the context nor its display changes
the selected ranking. Individual patient records remain bound private sources.

Both context releases write a source-bound manifest, a public-table replay audit
and a final SUCCESS receipt. Their query readers require the same selected
analysis hash as the website. The web input V3 requires explicit single_cell
and patient component statuses, so absent modules cannot silently disappear.
The frontend offers dataset/cell and subgroup/source filters, all public table
downloads and readable core statistics. Public HTTP download registration
excludes the raw donor and patient tables. No live site was deployed.

The 29-case synthetic check on 149 covers these actual native commands and
current context consumers, existing patient-statistics regression checks and
the real main-site/browser workflow. Optimizer constructors are forbidden;
only small synthetic statistical fits and frozen core fixture forwards occur.
These checks do not establish real cohort results or a complete global
cross-module patient/model-use inventory. The genomic/distal additions below
extend this verification; the C01/C13/C16 reconciliation remains unfinished.

Current native genomic and distal regulatory views (2026-10-16)
------------------------------------------------------------

``python -m scripts.materialize_v32_current_omics_context --manifest INPUT.json
--output NEW_DIRECTORY`` consumes CURRENT_OMICS_CONTEXT_INPUT_V1 on host 149.
Both kinds require a selected_analysis ``{path, sha256}`` record. Genomic input
has ``kind: genomic`` and native_bindings with all four producer keys: genomic,
cnv_event, cnv and atac. Each key supplies an AVAILABLE binding or an explicit
NOT_PRODUCED, SOURCE_UNAVAILABLE or NOT_APPLICABLE status with a reason. At least
one native producer must be available. Distal input has ``kind: distal`` and a
native_binding record for the original nested statistical command's AUDIT.json.

Genomic publication uses the actual native auxiliary loader. It checks native
checkpoints and five-fold provenance, replays the original fold aggregation and
matches the full candidate universe. Core-dependent heads must use the same
five checkpoint and core-parameter hashes as the selected reference model.
Active genomic experts in the selected fusion require matching source bindings;
other contextual predictions remain explicitly outside the selected score.
Mutation/CNV event co-occurrence, CNV expression association and ATAC
co-accessibility keep distinct producer/modality/expert identities and native
estimands. CNV event heads and CNV expression association are not interchangeable.

Public tables include the producer catalog, native aggregate, all native fold
predictions and counts, selected-member predictions and selected-GeneSet member
summaries. Available zero probabilities stay zero; missing folds, unavailable
models and their reasons stay explicit. The weighted member mean is descriptive,
not a jointly fitted GeneSet probability. Publication inspects checkpoint
provenance and replays aggregation; it does not replay genomic neural forwards
or reconstruct original raw-omics features.

The actual distal command now records its prepared-input audit, candidate
authority, each cancer's input and pathway-activity files, full expression and
pathway configurations, and exact implementation hashes. It rejects duplicate
input partitions and invalid candidate keys, and rechecks source hashes before
completion. Old receipts without this provenance cannot become current context
releases by adding an unverified declaration.

Distal publication re-executes that original statistical CLI in a temporary
directory with the recorded inputs and settings. It compares both stages'
audits, patient OOF components, inner-OOF components within every outer fold,
and candidate pathway results. The prepared-input topology must be eligible
for prediction and consistent across assays. This replay starts at the bound
prepared inputs; it does not regenerate mutation, ATAC or methylation features
from raw experimental files. The upstream feature preparation has its separate
contracts and tests. Only synthetic prepared inputs were used in this check.

Public distal tables retain slopes, OOF errors, R-squared, prediction correlation,
availability, assay coverage and per-fold expression-component coverage. GeneSet
views report supported membership and slope-direction counts; they do not
invent joint effects, confidence intervals or causal claims. Patient identifiers
and nested patient predictions remain private. Public aggregation is independently
recomputed by bounded SQL before the release SUCCESS receipt. Neither context
changes the selected ranking.

The actual website registers both modules under the same selected-analysis
version, including their public table queries and hash-pinned downloads. The
frontend offers producer/modality/expert and assay-feature filters. V3 requires
explicit genomic and distal statuses, including when results are unavailable.
The browser verification exercises a missing CNV context and zero measured
distal-methylation coverage. Versioned screenshots and receipts are preserved
under ``provenance/code_completion_20261015/current_omics_web_*``; older browser
evidence is retained. No model optimizer, real cohort analysis or deployment
is performed by these checks.

Unified current publication workflow (2026-10-16)
------------------------------------------------

``python -m scripts.run_v32_current_analysis_workflow --manifest INPUT.json
--output NEW_DIRECTORY`` calls the existing publishers on host 149 in this
order: rule evidence, selected scores/GeneSets, State, Clinical, Drug, signed
single-cell, patient subgroups, genomic, distal, external metrics, and web
registration. It consumes completed native producers. It does not launch
upstream trainers, rent compute, change a production pointer or start a site.
Context publishers retain their existing native statistical/frozen replay
behavior. Real executions of those analyses remain outside this code-only work.

CURRENT_ANALYSIS_WORKFLOW_INPUT_V1 contains exactly format, release_id, selected
and components. Selected and each component can declare either:

* ``{"status":"BUILD","input_manifest":{"path":"...","sha256":"..."}}``
  to invoke that module's real publisher from its existing input format; or
* ``{"status":"REUSE","binding":{"path":"...","sha256":"..."}}``
  to consume an existing pinned release without rebuilding it.

The components map must include all nine current web component keys. Each may
instead declare NOT_PRODUCED, SOURCE_UNAVAILABLE or NOT_APPLICABLE with a reason
and no hidden input. Selected cannot be absent. There is no TRAIN status or
arbitrary command/function field.

A build input may use ``{"workflow_ref":"selected_analysis"}`` in place of
its selected-analysis artifact, or ``{"workflow_ref":"selected_score"}`` for
the selected score binding in an external-evaluation model specification.
``{"workflow_ref":"evidence"}`` can supply the selected analysis's rule
evidence binding. Only completed earlier stages are available. Unknown, missing
or forward references fail. Other named component bindings are available after
their stage completes. Ordinary path/SHA256 records are checked and resolved
relative to the original input template. Generated resolved inputs are written
separately and bound in the workflow receipt; original templates are unchanged.

The selected BUILD stage executes the existing score selection, GeneSet,
coverage, subtype and GMT implementation. Downstream contexts all receive its
actual new binding through references. REUSE preserves the supplied analysis
identity. If a selected score already contains evidence annotations, the web
publisher still requires that exact evidence release; rebuilding different
evidence cannot silently substitute for it.

Each completed stage is written to STAGE_STATUS.json as BUILT, REUSED or its
explicit absence status. A failure records its stage and exception, preserves
earlier results and does not create the workflow SUCCESS receipt. Output reuse
is rejected. To retry, provide a new output directory; valid completed stages
can be supplied as REUSE records. The final website validates all same-version
relationships using its actual native readers before the workflow can pass.
Input hashes and publisher implementation hashes are checked again at the end.

CURRENT_WORKFLOW_BINDING.json records the actual stage bindings and web binding.
PUBLIC_TABLE_CATALOG.json derives every published table's row count, Arrow field
types/nullability, table interpretation and per-part hashes from the actual
artifacts. It has no private filesystem paths. This is a public schema catalog;
it does not yet supply a complete scientific definition and unit for every
field. SUCCESS pins the workflow binding and stage-status receipt.

The workflow explicitly retains model_use_inventory_completeness_verified=false
and independent_external_validation_attested=false. Its stage/input inventory
is not a complete raw biological-source inventory. C16 remains in progress:
complete source-use/patient reconciliation, field definitions, upstream command
integration and final per-requirement acceptance still need implementation.
