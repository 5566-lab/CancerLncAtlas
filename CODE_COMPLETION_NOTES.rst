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

Native model input exposure and external evaluation V2 (2026-10-16)
-----------------------------------------------------------------

``python -m scripts.materialize_v32_current_model_use --manifest INPUT.json
--output NEW_DIRECTORY`` inspects existing selected/core/fusion producers on
host 149. CURRENT_MODEL_USE_INPUT_V1 contains models with the same model_id,
producer_kind, producer_binding, run_id and score_column specifications used
by current external evaluation. Optional artifact_locations are path/SHA256
records for relocated native authorities. A locator must match a hash already
carried by an actual authority and be used; it cannot register unrelated data
as a model dependency.

For current V4 predictions, the implementation follows all five authorized
config/input/prepared/checkpoint/training-receipt records and graph sidecars.
It uses the real native payload validator and candidate-alignment reader,
including their sealed-test firewall. The training and validation availability
masks determine available-supervision exposure separately. Candidate features
and graph lncRNA nodes retain their own views. It does not read sealed test
payloads, construct a neural model, perform neural forwards or fit parameters.

The entity table groups native candidate exposure by lncRNA within model,
outer fold, use phase and input view. candidate_rows counts candidate records,
not patients, effective gradients or independent observations. The exclusion
scope is explicitly LNCRNA_EXPOSURE_ALL_PATHWAYS: a lncRNA exposed through a
model input is conservatively excluded from entity-novel evaluation for every
pathway. This avoids expanding millions of candidate rows across all five
folds merely to recover the same exposed lncRNA universe. It is not a pair-
specific claim about which learned coefficient is nonzero.

The L1 score is treated separately: its published values come from base_logit.
The inventory retains candidate-feature exposure and reports unresolved
baseline-fitting ancestry. It does not attribute the reference neural graph
or neural checkpoint-selection labels to L1. A selected score follows its
actual selected producer; reference directions and post-hoc rule annotations
do not become model-fitting dependencies. A validation winner lock remains
explicitly incomplete until all competitors' training ancestry is resolved.

For fusion, the actual private feature snapshot and available calibration
targets supply exposure. A native primary-label binding is replayed and traced
to the actual current core when present. Every declared fitted expert is
retained in the unresolved ancestry table, including a zero final coefficient:
zero contribution is not proof that its source was absent during fitting.
Historical fusion without native primary ancestry remains explicitly incomplete.

Graph static/fold authorities, patient authorities and unlocated input hashes
are recorded separately from the actual exposure. Present files with a wrong
hash fail. Directory inputs use the existing native artifact-tree hash and
detect changed partitions. Missing authorities remain unresolved rather than
being asserted complete. File integrity does not reconstruct original study
identity or the upstream raw feature/label transformations. The current release
therefore records raw_study_provenance_reconstructed=false and
model_use_inventory_completeness_verified=false. No study ID is invented from
a prepared filename, checkpoint or input checksum.

MODEL_USE_BINDING.json pins USAGE.parquet, private DEPENDENCIES.parquet and
GAPS.parquet, actual model identities, source inputs and implementation hashes.
An audit independently reruns the native input inspection and compares all
three tables before SUCCESS. Consumers replay this inspection and require the
same evaluated model, run, producer hash and numeric score. Rehashing an edited
usage table cannot hide omitted exposure. All individual patient identities
remain outside the public inventory.

The existing external-analysis CLI now also accepts
CURRENT_EXTERNAL_VALIDATION_INPUT_V2, requiring model_use_inventory as a bound
MODEL_USE_BINDING.json record. Optional model_use_sources supplement that native
inventory; they cannot replace it. V1 remains explicitly limited to declared
sources. V2 combines native exposure and registered evidence, checks source-file
overlap against native dependency hashes, and reports
NATIVE_DEPENDENCIES_INCOMPLETE when ancestry is unresolved. Descriptive metrics
continue to use the actual measured outcomes; held-out eligibility is not
fabricated. Coverage and query results retain the native inventory's actual
scope. Public model_dependencies omits paths; model_dependency_gaps explains
the unresolved roles. Completeness and biological independence remain false.

In an external BUILD template for the unified publication workflow, use V2
and ``"model_use_inventory":{"derive_from_native_models":true}`` to derive
the inventory automatically from the resolved evaluated models. The workflow
records a real model_use stage before external evaluation and substitutes its
actual binding. The standalone external-analysis CLI takes an already built
binding instead of this workflow directive. The actual website serves the new
public tables and provides role/phase/use-view filters. Browser verification
checks the unresolved feature/label lineage for all five folds.

Synthetic validation covers native masks, graph exposure, L1 separation,
directory integrity, model identity substitution, rehashed omission, fusion
ancestry and actual external V2 consumption. A deliberately incomplete manual
list misses an exposed lncRNA; V2 correctly identifies it through the native
inputs. Complete raw-study reconstruction, all auxiliary/competitor ancestry,
cross-module patient reconciliation and final C01/C13/C16 acceptance remain open.

2026-10-16: actual formal preparation reads retained through external evaluation
-----------------------------------------------------------------------------

Formal preparation now records its immediate consumed inputs, including the
actual local-CNV training and validation association partitions for each cancer
and fold. A SUCCESS receipt alone previously did not identify those table
bytes. SourceCapture wraps the native reads, hashes before and after reading,
and checks again before emitting a fold. A changed association partition fails
even when the enclosing SUCCESS receipt is unchanged. Legacy preparation also
records actual expression partitions, the activity tree and its individual
Parquet files. Detection, pathway hierarchy, candidate authority, patient
authority and authorization controls retain their separate roles. No input
path is inferred from an unexplained upstream checksum.

SOURCE_READS/PATIENT_FOLD_<fold>.json is copied into each arm's native payload
as preparation_sources. Its canonical digest is part of the input authority,
so the existing C overlays, guarded loader, checkpoints and export lineage
retain it. Training checks receipt structure, source-role/cancer/fold coverage,
mode and digest without requiring the original source paths on a training
machine. Historical payloads without this new receipt remain readable and
explicitly unresolved; this change does not retroactively attest their sources.
Inventory replay checks the original files, or exact-hash relocated files,
including the entire recorded partition list and source metadata.

The native reader extracts study identifiers only from explicit PMID, DOI,
dataset/accession and experiment columns in consumed tables. It does not infer
a study from a filename or claim that a standardized table is the raw study.
Source metadata uses WHOLE_CONSUMED_FILE_BEFORE_ROW_FILTERING semantics. It is
conservative file-level evidence, not proof that every row or patient entered
a loss. Usage rows have no asserted entity identity and candidate_rows=0;
their use_view is PREPARATION_SOURCE_METADATA. Existing lncRNA exposure rows
remain separate. Absent metadata and unresolved upstream transformations are
reported explicitly. Full raw-study ancestry and inventory completeness remain
false. External V2 now detects study overlap for an otherwise unseen lncRNA
through these actual preparation inputs, and the website displays study tokens.

The fixed-C L1 baseline is fitted on available training rows. Its validation
metrics are diagnostic and do not fit its numeric score. When new native
preparation sources exist, L1 therefore retains the actual training sources
and available-supervision exposure, while validation association sources and
neural graph dependencies remain excluded from its numeric-score ancestry.
Core neural selection retains validation sources separately. Legacy L1 inputs
without this receipt still report unresolved baseline fitting provenance.

The formal CLI now enforces host149 and writes a preflight before preparation.
It DOES include L1 baseline fitting; it is not a no-training verification
command. The restored historical server wrapper explicitly says so and checks
host149. The wrapper was imported from the original ranked-subtypes checkout;
its provenance is recorded in preparation_sources_source_adaptation.json. It
was not launched. The previously omitted frozen patient authority fixtures
were restored to the ignored artifacts directory after checking their exact
canonical hashes. Regression tests use those only for authority/identity
checks; no real expression cohort or real model is fitted.

New tests exercise actual adjusted readers, the legacy CLI's native readers
and all three payload writers, digest/role/fold/cancer rejection, changed
partitions, metadata alteration, relocation, native five-fold current inputs,
L1 separation and actual external V2 consumption. The legacy CLI test uses a
small synthetic scope, fixed baseline output and a graph stand-in; it does not
claim to validate the full real-data preparation calculation. The independent
current-producer tests use real native loaders and frozen synthetic forwards,
then forbid model construction during inventory inspection. Optimizer and
logistic fitting calls are forbidden. Existing graph authority tests and the
actual website/API/browser checks provide additional regression coverage.

C13 still requires complete upstream biological transformations and raw
sources, auxiliary and comparison-model ancestry, and cross-module patient
closure. C01/C16 consolidation, upstream orchestration and final acceptance
remain unfinished. This is a source-capture and downstream-consumption change,
not a completed real-data analysis or an attestation of independent validation.

2026-10-16: all G0/G1/G2 selection inputs reach native source isolation
-------------------------------------------------------------------

The native G012 winner-lock producer previously inspected all fifteen folds
but serialized only the winner's prepared/checkpoint files. The other two
arms survived only as scalar validation summaries. New locks retain all three
compared_runs, including per-fold prepared files, checkpoints, exact SUCCESS
receipts and each arm's config, input manifest and task manifest. They carry
comparison_sources_format=G012_COMPARISON_SOURCES_V1. Existing top-level winner
aliases and sealed-inference compatibility are preserved. The producer enforces
host149 and writes a preflight before code hashing or source inspection.

validate_comparison_sources reopens the pinned native records for all fifteen
folds. It checks complete arm/fold coverage, task/authorization identity,
candidate authority, validation-only flags, recorded checkpoint/prepared hashes,
each fold loss, mean losses, deterministic winner ordering and winner aliases.
The selected-score consumer invokes this replay and reports whether all
comparison receipts were replayed. This reconstructs the selection calculation
from native training receipts; it does not independently recompute neural
validation losses or attest optimizer execution.

Model-use inspection now traverses the comparison inputs through the same
authorized V4 payload/checkpoint loader and candidate-alignment firewall used
for current core prediction. It records actual candidate/graph exposure,
available training and model-selection supervision, preparation sources and
unresolved upstream ancestry for every arm. Authorized C/expansion sidecars
are retained when present; their native loader still checks their contract.
No comparator model is constructed and no neural forward or optimizer is run
during inventory inspection. This does not average losing models' scores into
the published score or change which producer supplies that score.

The new dependency_context column applies to usage, dependency and gap rows.
SELECTED_SCORE means a dependency of the selected numeric producer;
G012_COMPARISON:G0/G1/G2 identifies an arm involved in choosing that producer.
These contexts retain the evaluated model_id, allowing source isolation to
exclude the union of all comparison inputs while displaying their distinct
roles. The native external query and actual website support exact context
filters. Public dependency tables continue to omit file paths and patient IDs.

Old winner-only locks remain readable and report
ALL_COMPETITOR_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED. An unsupported historical
checkpoint or unresolved native run identity remains an explicit gap. New
locks cannot omit an arm/fold or substitute losses without being checked
against their bound receipts. Reading all fifteen native inputs still does
not reconstruct missing raw studies or upstream transformations; overall
model-use completeness and independent-validation attestation remain false.

Synthetic verification stages actual three-arm graph masks, native V4 payloads
and fixed random checkpoints with optimizer calls forbidden. Only the selected
G1 producer performs frozen forwards to create its test score publication.
The actual native winner-lock writer, selected-score consumer, model-use
materializer and external V2 evaluator are then exercised. A study present
only in losing G0 preparation inputs correctly excludes an otherwise unseen
external lncRNA. Tests also reject losing-arm/fold omission, altered losses,
receipt replacement and winner-alias drift. Historical frozen patient files
serve only as identity authorities in these fixtures; no real biological
cohort analysis or model fitting occurs.

C13 still requires raw biological ancestry, auxiliary-model dependencies,
cross-module patient closure and handling historical comparison gaps. The
current V4 all-arm traversal is implemented; C01/C13/C16 final acceptance is
not complete.

Native auxiliary dependencies consumed by fusion
------------------------------------------------

The current fusion preparation command retains each expert's exact specification,
run, estimand, native binding and consumed prediction in INPUT_AUDIT.json. The
model-use inspector follows these records for every expert used by the evaluated
endpoint, including experts with a final zero coefficient. It first replays the
native aggregation and compares the exact candidate keys, values and availability
against the private fusion snapshot. Unconsumed source declarations, changed
run identities, specifications or snapshot values are rejected.

Mutation/CNV-event, directional CNV and ATAC native bindings now contribute their
actual checkpoints, receipts, patient-fold authorities and candidate exposure.
Genomic lineage input_artifacts and ATAC input-audit/materialization sources are
hash-verified and inspected for explicit study metadata. Native genomic/CNV heads'
two consumed lncRNA/pathway embedding exports are retained, together with core
training receipts and any recorded source authorities. Whole input-file metadata
is exposure evidence, not a claim that each row fitted a model. Patient IDs are
not exported. Non-table inputs, unlocated authorities and unrecorded ancestors
remain explicit gaps. Historical CNV hashes do not reconstruct their consumed
matrix partitions; old fusion audits without expert mappings remain incomplete.

These rows carry dependency_context=FUSION_EXPERT:<expert_id>. External V2 uses
their file/study exposure in isolation and supports the same context filter in
public dependency queries, with paths removed. Native fusion audit/source/lineage
artifacts containing filesystem references are private; public scores and metrics
remain public. Source replay and numeric aggregation are not neural forward
validation or an assertion of complete patient-level training/selection isolation.

Synthetic verification uses fixed auxiliary checkpoints and fixed zero-weight
fusion parameters with optimization forbidden. The actual prepare-only runner,
native adapter, fusion serializer, model-use replay and external V2 consumer are
exercised. Bare legacy primary-label lists remain ineligible for current fitting;
the fixed-parameter fixtures do not claim training readiness. Signed SC auxiliary
ancestry, complete native core/raw ancestry, CNV partition lineage and cross-module
patient usage remain open under C13/C16. No real data analysis or training is run.

Signed SC-HHGT training inputs and upstream sources
--------------------------------------------------

The current fusion runner now replays the native signed SC adapter rather than
accepting its PASS marker alone. The shared loader checks the activated OOF
prediction alias, aggregation policy, all donor/cell-context tables and full
candidate summaries. The model-use consumer also compares each consumed signed
feature against the actual private fusion snapshot before following its source.

When a native training SUCCESS is bound, the inspector opens both five-fold
SC-HHGT arms, native checkpoint tensors and digests, reports, source implementation
bindings, input lineage and episode manifests. It reconciles the activated OOF
table with all per-model prediction files. Actual native load_episode calls then
rebuild each arm's graph/features and enforce its donor isolation contract;
their audits must equal the bound training reports. Query identities and observed
targets must match the consumed query episode. No model is constructed and no
model prediction, optimizer or LASSO fit occurs during this inspection.

Explicit episode/static/matrix/source receipts are followed recursively through
bound JSON records and paired path/hash fields. Source tables contribute only
explicit study metadata. Native lncRNA node identities and fit/validation/query
exposure reach the current inventory, including data absent from the fusion
candidate summary. Usage identifiers bind graph exposure to each actual episode,
not only a shared static graph. Unbound paths, unsupported array ancestry and
missing historical training bindings retain explicit gaps.

The adapter's donor crosswalk is also applied privately to native source and
excluded donor sets. Canonical aliases crossing a native training/query partition
or outer folds are rejected. Missing or ambiguous mappings remain gaps; this does
not establish cross-module patient independence. No raw donor identifiers, raw
receipt mapping keys or filesystem paths enter public dependency tables. Dynamic
upstream mapping keys use structural indices in public role labels.

FUSION_EXPERT:single_cell_positive and FUSION_EXPERT:single_cell_negative preserve
the evaluated model identity even for final zero coefficients. Synthetic tests
verify exclusion of an otherwise unseen external entity whose study appears only
in SC upstream sources. They also reject changed files, repinned checkpoint state,
omitted ablation-arm episodes, changed query donors and canonical-alias leakage.
Fixtures initialize the actual native architecture with fixed parameters and
prebuilt episode tensors; their native-shaped receipts are explicitly synthetic,
not evidence of optimization or scientifically validated predictions.

C13/C16 still require complete raw transformations/study ancestry, CNV consumed
partitions, full core ancestry and cross-module patient closure. Signed SC native
producer traversal is now implemented; complete biological independence and final
C01/C13/C16 acceptance remain unproven.

Native CNV consumed matrices, labels and segment sources
------------------------------------------------------

Both existing CNV runners now capture hashes at their actual reads and check
them again before publishing CONSUMED_SOURCES.json. Event CNV records its six
matrices, identifiers and materialization receipts. Directional CNV records its
eight matrices plus every cancer/fold train and validation association-label
partition and the exact columns read. Repeated reads must retain the same hash.
The compact readers reject disagreement with recorded materializer file hashes
before fitting; this change does not start any fitting. Runner execution requires
host 149. The event OOF serializer and both downstream native adapters consume
the new receipt. Historical receipts without it retain the explicit missing-
partition gap and are not retroactively certified.

Native source verification checks candidate/fold/core identities, complete
matrix and label partitions, matrix shapes against unique identifier arrays,
materializer-to-partition hashes, label roots, source run and label values.
Existing aggregation and fusion-snapshot replay remains in place; no neural
forward is added. A consumed-file receipt describes file exposure, not that all
rows or patients were used in optimizer updates.

The model-use inventory follows these verified reads into hash-bound JSON
records, materialized member maps and explicitly bound upstream tables. The two
segment materializers now hash each raw segment before and after its read and
bind SOURCE_FILES.json into the partition SUCCESS. Directional receipts also
record the path accompanying their existing old-partition SUCCESS hash. Public
dependency roles use structural indices rather than arbitrary upstream keys;
public dependency results omit filesystem paths. Explicit study metadata from
CNV upstream sources reaches external V2 isolation, including when the external
lncRNA is unseen and the fusion coefficient is zero.

Review also found that directional materialization recorded the configured event
threshold but did not pass it into either segment mapping call. Both lncRNA and
gene mappings now use that threshold. Synthetic segment values of 0.5 produce an
amplification event at threshold 0.30 and no event at 0.70, matching the receipt;
the default behavior is preserved.

Synthetic verification exercises actual compact/label readers, both native CNV
adapters, fusion and external consumers, plus tiny actual segment materialization.
It rejects changed reads, changed arrays, missing or misassigned partitions,
repinned wrong array shapes, and upstream source changes. Native-shaped model
fixtures use fixed parameters; no optimizer, real-cohort analysis, paid compute
or deployment is performed. The initial combined run exposed a legacy test
helper's event-format branch and a process-memory-sensitive test; the helper was
corrected and memory-gate tests were run separately without relaxing their gate.

CNV consumed-partition tracing is implemented for newly generated receipts.
Complete raw transformation ancestry, absent historical source bindings, full
frozen-core ancestry and cross-module patient closure remain C13/C16 work.
Scientific independence and final C01/C13/C16 acceptance are still unverified.

Frozen-core training ancestry consumed through auxiliary exports
----------------------------------------------------------------

The native current-core exporter now binds its input manifest and writes resolved
source paths. Previously it copied relative input paths into a different output
directory without changing their base. Existing consumers still accept historical
exports, while new exports retain the exact prepared/config/checkpoint/training
receipt and optional graph-overlay/expansion records that were actually read.

Auxiliary model-use inspection follows those sources through the same V4 native
prepared loader used by core inference. Source checkpoints and copied training
receipts must match the exported fold, and new export records must match the
bound export-input manifest. Exact-hash artifact locators support relocated
source files. Missing historical source records remain gaps; a partially declared
or mismatched new source record is rejected.

Without constructing a model, the inspector verifies checkpoint tensor state
against the exported parameter digest, prepared runtime chunks/weights against
the export contract, and exported lncRNA/pathway node order against the graph.
It reconstructs the complete core candidate keys from native node indices,
independently of any smaller downstream candidate table, and uses the existing
availability masks to retain training and model-selection exposure. Formal
preparation source metadata and graph source tables now propagate into the
auxiliary FUSION_EXPERT context. Graph inspection also follows the base authority
inside a current-C binding expansion and retains fold-specific source events.

Synthetic checks execute the actual frozen five-fold exporter on tiny native
graphs and fixed checkpoint parameters; inventory inspection itself performs no
model construction or forward. Actual genomic, directional CNV and event CNV
adapters/fusion/external consumers then verify exclusion of unseen entities whose
studies appeared only in core training, validation or graph sources. Tests reject
repinned parameter identities, changed node IDs, omitted sources, inconsistent
copied receipts, wrong export input identities and changed graph-source files.
No optimizer, LASSO fitting, real cohort analysis or deployment is performed.

This closes native frozen-core export-to-training-input traversal for present
sources, not reconstruction of every raw transformation. Complete upstream study
ancestry, missing historical bindings, cross-module patient closure and final
C01/C13/C16 acceptance remain open. Source integrity is not a biological
independence attestation, and exported embedding values are not recomputed by
the source inspector.

File-scoped patient identity and cross-module exposure
----------------------------------------------------

Native model-use inputs and external V2 inputs now accept an optional explicit
patient identity crosswalk. The unified workflow passes the same declaration
into its derived native inventory. A declaration has the following shape::

    "patient_identity_crosswalk": {
      "path": "patient_identity.parquet",
      "sha256": "<SHA256 of that crosswalk file>",
      "namespace": "<shared patient identity authority>"
    }

The crosswalk has four string columns: source_file_sha256, identity_column,
raw_id and canonical_patient_id. Each source hash identifies the actual consumed
file, not its directory or cohort name. Two aliases identify the same patient
only when explicitly mapped to the same canonical identity in the same namespace.
Identical raw IDs from different files do not automatically identify one person.
Conflicting mappings of one file/column/raw-ID tuple and external/native namespace
disagreement are rejected. The crosswalk is an asserted authority, not an inferred
biological identity check. Its own digest is retained in the private provenance.

Actual formal-preparation and auxiliary input-table reads contribute patient_id,
sample_id, donor_id, query_donor_id and canonical_donor_id when present. Readers
load only these columns in bounded batches. CSV/TSV identity and crosswalk readers
retain textual leading zeros. For native CNV compact inputs, identity_column is
patient_ids and source_file_sha256 is the actual patient_ids.json hash; only IDs
referenced by the native fold authority and matched by its row lookup are included.
For SC-HHGT episode inputs, identity_column is source_donors or query_donor_id and
the hash identifies the actual episode file. Actual loaded feature donors and
target rows are recorded; unused excluded-donor declarations are not exposures.

The private patient_usage_private table stores namespaced patient tokens with
model/context, source hash/role, native fold, phase and exposure scope. It does not
store raw or canonical identifiers. Public patient_mapping rows report counts of
distinct observed, mapped and unmapped aliases (column plus raw ID), not counts of
people. Public patient_context_overlap reports unique mapped people shared between
model/context pairs. These are lower bounds under the supplied mapping; zero does
not prove disjointness. Numeric fold IDs from different split schemes are not
assumed comparable, so this table does not certify cross-module fold leakage or
its absence. Whole-file exposure is conservative and does not mean every row was
used in an optimizer update.

External V2 compares observation identities against the union of all mapped
native model/context identities. Known patient overlap and incomplete external or
registered mappings receive explicit isolation reasons and cannot qualify as
held out. Registered mapping completeness covers only observed registered IDs;
files without identity columns and missing ancestors remain outside that claim.
Full model-use completeness and independent biological validation remain false.
Legacy V1 external behavior is unchanged and cannot accept a patient crosswalk.

Only aggregate mapping and overlap tables reach the public query/download and
Chinese frontend. Native patient usage must remain marked private. External
loaders reject unapproved table names, private output declarations and patient
identity columns even if a private table is renamed to an allowed public name.
Public source roles continue to use structural names, without raw mapping keys.

Synthetic checks exercise the actual formal preparation reader, native genomic
and signed-SC consumers, compact CNV source inspection, fusion, external evaluation
and unified web workflow. They distinguish matching aliases, unrelated identical
raw IDs, unmapped identities and textual leading zeros. The initial failures
exposed external-only all-null annotations in model-use replay and a synthetic
CNV fixture whose IDs disagreed with the native canonical row lookup; serialization
and the fixture were corrected without relaxing source validation. No optimizer,
real cohort analysis, paid compute or deployment was started.

Explicit cross-module identity inspection is implemented for present bound
sources. Complete raw transformations, row-level usage and patient coverage,
missing historical source bindings, upstream command records and final C01/C13/C16
acceptance remain open. Existing preparation-only datasets have not been analysed
by this code change.

Native local-CNV transformations and recovered statistical corrections
--------------------------------------------------------------------

Source inspection found that this development copy lacked the native local-CNV
association, validation-label, corrected-E1-graph and closeout scripts present in
the original October checkout. Eleven original source/test files were recovered
or reconciled; local_cnv_ancestry_recovered_sources.json preserves their original
paths, hashes and the previous development-file hashes. Subsequent adaptations
are recorded separately. The original checkout and existing results are unchanged.

The recovered numerical corrections keep constant or design-collinear CNV as a
zero adjustment, rather than multiplying by a NaN coefficient and discarding an
otherwise observed association. Effective residual rank and correlation degrees
of freedom are retained per candidate. Finite correlation alone does not make a
row estimable when its p value or residual degrees of freedom is unavailable.
The preparation adapter now requires the actual rank/df columns and rejects
nonpositive df for an available row; it no longer invents rank=2 and df=n-3.
Historical association tables lacking those columns require regeneration.

The shared covariate contract includes every declared covariate. Missing declared
columns fail, and high-cardinality categorical covariates are retained for this
contract rather than silently dropped. The generic prepare_design default still
retains its historical category cap; strict callers explicitly disable it. The
existing bulk-coexpression test fixture was updated to declare all covariates,
but its clustering suite was not executed because it fits KMeans. Focused design
and actual native association calculations were verified without model training.

Both native per-cancer producers now hash their actual table reads and all eight
directional CNV matrices/three identifier lists, before and after reads and before
publication. Available materializer hashes are checked. The produced five-fold
partitions bind these reads, their columns, the statistical/read implementations
and their own exact output hashes in source_reads/<cancer>.json. Reuse validates
both prior content and the requested candidate, patient-map, covariate, compact
directory and run identity. Old receipts without captured reads are not upgraded
by reuse; a new output directory is required. Entry points check host 149 and
write preflight records before scientific work.

Each output binds the roles it actually depends on. A1/pair tables do not inherit
the gene-expression and graph-annotation inputs used only by G0 construction.
Corrected G0 graphs do not inherit pathway-activity inputs used only for pair
statistics. Captured files describe conservative file exposure, not an assertion
that every patient or every row contributed to a particular correlation.

The actual formal preparation reader attaches the native transformation receipt
to each consumed train/validation association partition. Current model-use replay
follows the bound source tables and compact materialization records, retaining
explicit study metadata and file-scoped patient mapping. Validation-only upstream
exposure retains MODEL_SELECTION semantics. L1 continues to exclude diagnostic
validation dependencies. The E1 graph materializer also retains each consumed
edge-partition transformation. Its inspector replays the native present_E1/rho_E1
selection and compares actual corrected graph rows before following upstream
gene-expression/CNV sources. Repinning a changed graph output does not bypass
this replay. New provenance reaches direct core, auxiliary-core and comparison
contexts through the existing inventory paths.

The recovered commands are runnable on 149 with PYTHONPATH set to the development
root; they were not launched on real inputs. The required ordering is:

* run_v32_local_cnv_confounding_audit_r2.py, once per cancer, with --candidates,
  --patient-map, --covariates, --source-config, --graph-root, --directional-root,
  --output-root and --cancer. Use --run-id V32_LOCAL_CNV_CONFOUNDING_20260902_R3
  when using the recovered R3 closeout/downstream contract. The candidate scope
  remains 100,000 pairs per cancer across the 33 declared cancers.
* finalize_and_audit_v32_local_cnv_r3.py, with --root, --candidates,
  --directional-root, --runner-code, --patient-map, --covariates and
  --source-config. It requires all 33 cancers/five folds; restoration and import
  checks do not constitute execution of that full closeout.
* materialize_v32_local_cnv_validation_associations.py, once per cancer, with
  --cancer, --candidates, --patient-map, --source-config, --directional-root,
  --local-cnv-audit-root, --local-cnv-success-sha256 and --output-root. The
  subsequent audit_v32_local_cnv_validation_associations.py consumes --root,
  --candidates, --patient-map and --local-cnv-audit-success.
* materialize_v32_local_cnv_corrected_graph_authority.py uses --phase initialize,
  then materialize-cancer for each --cancer, then finalize. All phases bind
  --source-graph-receipt, --source-graph-receipt-sha256, --local-cnv-audit-root,
  --local-cnv-audit-success-sha256 and --output-root. This precedes the existing
  formal preparation/current comparison path; it does not start a trainer.

Synthetic verification uses actual per-cancer calculations on 120 patients and
one candidate, reducing only fixture dimensions. Actual five-fold association,
validation and E1 graph materialization, native preparation, frozen core forwards,
inventory replay and external evaluation are exercised. The full 33-cancer
statistical closeout is not run; its aggregate authorization in the fixture is
explicitly synthetic. No neural optimizer, LASSO/KMeans fit, real-cohort analysis,
paid compute or deployment occurred. A transient SSH banner error interrupted
receipt downloads after successful tests; read-only host probes returned 149 and
the existing receipts were retrieved without repeating completed jobs.

This implements local-CNV transformation ancestry for newly bound native sources.
Earlier raw expression/activity normalization, complete per-row patient usage,
absent historical bindings and final C01/C13/C16 acceptance remain unresolved.

Native TMM expression normalization and conversion ancestry
---------------------------------------------------------

The original standalone materialize_formal_lncrna_logcpm.R was recovered into
scripts/ with its original path and SHA256 recorded in
expression_ancestry_recovered_sources.json. Its existing scientific calculation
is retained: use the unstranded count assay, select one sample per patient and
sample type with the original priority/library-size rules, calculate edgeR TMM
on all genes in selected tumor AND normal samples, use log CPM prior.count=0.25,
then export the declared eligible lncRNAs in selected tumor samples. Duplicate
stable gene IDs retain the first source row and PAR_Y rows are excluded from the
lncRNA export. They are not removed from the count-library normalization.

Both the R producer and Python converter require host 149, record a preflight,
and require a new output directory. The producer records the actual count RDS,
detection table, its implementation and edgeR/R versions. It checks input hashes
around reads and before publication. Each cancer has an immutable source receipt
binding its output matrix, tumor-sample table and complete read-sample metadata.
The latter retains which samples contributed to normalization and which were
exported, including normal samples absent from the tumor output. Study metadata
is copied only from explicit source RDS colData or metadata(x)$study_metadata
fields. Study identifiers are not inferred from TCGA filenames or cancer names.

convert_formal_lncrna_rds.py now has --source, --output and repeatable --cancer
arguments. Omitting --cancer still requires 33 matrices. An explicit subset is
recorded as that subset, not a full release. The converter rejects duplicated or
invalid lncRNA/sample identifiers and nonfinite values; it preserves exact RDS
values and TCGA patient identity. EXPRESSION_SOURCES.json in each output partition
binds the actual RDS read, optional native normalization receipt, output Parquet
and converter implementations. Legacy RDS files can be converted, but absent
normalization provenance remains explicitly unresolved.

The normalization sample table has these scientific field meanings:

* library_size: sum of unstranded raw counts across all input rows for that
  aliquot, before the tumor/normal selection; unit is read counts.
* normalization_factor: dimensionless edgeR TMM factor for a selected aliquot;
  missing for excluded aliquots. The effective library size is library_size
  multiplied by this factor.
* selected_within_patient_type: first aliquot under the recorded sample-type,
  priority, descending library-size and sample-ID ordering.
* include_expression: selected tumor or normal aliquot actually used by TMM.
* include_association: selected tumor aliquot exported to the expression
  matrix. It does not assert use in every later correlation or training batch.
* logcpm: edgeR log2 counts per million with TMM normalization and the specified
  prior count; it is not TPM, a probability, or an effect size.
* patient_id: the first 12 characters of a validated TCGA sample identifier;
  cross-study identity still requires the separate explicit patient crosswalk.

Formal preparation and native local-CNV readers capture this receipt during the
actual Parquet read. Later inspection checks the previously captured binding;
it does not silently attach provenance to old payloads. Inventory replay compares
the converted table against the actual bound RDS, so repinning an edited output
does not bypass the transformation check. Native normalization input hashes and
explicit study/sample metadata then reach direct core, auxiliary and comparison
consumers through the existing source paths. The shared workflow and public
dependency outputs retain their existing private-identity restrictions.

Run on 149 with the development root on PYTHONPATH, using new destinations:

* Rscript scripts/materialize_formal_lncrna_logcpm.R COUNT_RDS_DIRECTORY
  DETECTION_TABLE_DIRECTORY NEW_NORMALIZATION_DIRECTORY
* python scripts/convert_formal_lncrna_rds.py --source NEW_NORMALIZATION_DIRECTORY
  --output NEW_PARQUET_DIRECTORY
* Bind NEW_PARQUET_DIRECTORY as the actual lnc_expression_root for local-CNV or
  --formal-expression-root for legacy formal preparation. Existing pinned
  authorities must be regenerated and rebound; old results are not overwritten.

Synthetic verification uses 80 count rows and 123 sample aliquots, selects 121
normalization samples and exports 120 tumor samples. It executes real edgeR TMM,
the actual RDS conversion, native five-fold local-CNV calculations, current-core
frozen forwards and external isolation. An independent direct edgeR calculation
uses explicitly expected sample/row indices to check values and sample selection.
This is stronger than an R parse check. No optimizer or real cohort run occurs.

This recovers one earlier transformation, not complete raw ancestry. The
historical TMM procedure is cohort-level and is explicitly reported as NOT
fold-local; this change does not attest independent predictive validation or
retroactively correct historical scores. GDC acquisition before the count RDS,
pathway-activity/gene-expression preprocessing, complete per-row exposure,
missing historical bindings and final C01/C13/C16 acceptance remain open.

The expression_ancestry_r2 receipt records 40 passing synthetic cases on 149.
Broader regression initially failed during SCP upload before the remote test
command was sent. A subsequent SSH probe returned an invalid banner, and iOA
could not be made readable through the available desktop controls. That incident
is preserved in expression_ancestry_network_pending.json; no credentials were
submitted. A later read-only probe returned 149 without an observed recovery
mechanism. Remote inspection confirmed no matching test process or receipt before
resumption. The 44-case regression and 11-case browser suite subsequently passed,
bringing this change's current scoped validation to 95 distinct cases. A separate
transient web-launch handshake failure was also checked for an absent process and
receipt before retry. New desktop/mobile screenshots and the new browser receipt
are retained; historical screenshots are not reused as current evidence.

The ledger now distinguishes actual synthetic edgeR execution for this recovered
R producer from the unchanged parse-only scope of other R scripts. Passing these
checks does not close the outstanding normalization, provenance or overall
acceptance requirements above.

Frozen training-patient expression normalization
------------------------------------------------

The new scripts/materialize_fold_local_expression.R fits a separate TMM state
for each outer patient fold. Fold f is held out, fold (f+1) modulo 5 is validation,
and the other three folds supply training patients. The original aliquot priority
is retained. Selected tumor AND normal aliquots belonging to training patients
determine the reference library, centered normalization factors and mean
effective library size. Unassigned patients do not contribute to fitting.

scripts/frozen_tmm.R provides fit_frozen_tmm and apply_frozen_tmm. Fitting follows
edgeR's default training-only upper-quartile reference rule, with its sparse
fallback. Applying uses public edgeR reference/sample pair factors: their ratio
recovers the relative factor, which is multiplied by the frozen reference
factor. Applied samples are never recentered together. The logCPM prior is also
frozen to the training mean effective library size. For a sample with effective
library E and frozen training mean M, the added count is a=0.25*E/M and the value
is log2((count+a)/(E+2*a)*1e6). Fixing only the TMM reference would leave the
usual all-input factor centering and logCPM prior dependent on held-out samples.
The edgeR definitions are documented in the official manual:
https://bioconductor.org/packages/release/bioc/manuals/edgeR/man/edgeR.pdf

The producer writes five separate normalization states, full read-sample tables,
training-tumor detection tables, lncRNA matrices and protein-coding gene matrices.
The latter use GENE: stable identifiers; lncRNAs use LNC: stable identifiers.
Duplicate stable identifiers keep the first source row and PAR_Y rows are
excluded from export, while all count rows remain in library normalization.
Each matrix emits selected training and validation tumors only. Outer-test
tumors are not emitted. The original count RDS is nevertheless read as a whole;
the receipt explicitly records that raw-file exposure, rather than claiming
test patients were never read anywhere upstream.

lncRNA eligibility is computed as the fraction of selected TRAINING TUMORS
whose frozen logCPM exceeds zero, with threshold 0.10. Validation expression
cannot change eligibility. All exported lncRNA measurements remain in the
Parquet output, with analysis_eligible and unavailable_reason columns. Native
association readers exclude ineligible measurements from statistical fitting,
but retain their candidate rows as typed_unavailable with reason
BELOW_TRAIN_DETECTION_THRESHOLD. A0 statistics no longer turn an entirely absent
feature into a measured zero association through imputation. Gene measurements
are exported without this lncRNA detection filter; the existing training-only
G0 variance filter remains responsible for usable graph features.

Run the following only on 149, with new output directories:

* Rscript scripts/materialize_fold_local_expression.R COUNT_RDS PATIENT_MAP_TSV
  CANCER NEW_CANCER_NORMALIZATION_DIRECTORY
* python scripts/convert_fold_local_expression.py
  --source NEW_CANCER_NORMALIZATION_DIRECTORY --output NEW_EXPRESSION_DIRECTORY
  (repeat --source for additional explicit cancer directories)
* Merge the generated SOURCE_CONFIG_FRAGMENT.json sources into a new local-CNV
  source config, retaining the actual covariate and pathway-activity authorities.
  Both lnc_expression_root and gene_expression_root use layout=fold_local.
* Run the previously documented native local-CNV association, validation,
  independent closeout and corrected graph materialization commands with the
  SAME patient-map path. Then use their actual partitions in adjusted formal
  preparation. These instructions do not launch a trainer or replay real data.

The converter requires all five folds for each explicitly supplied cancer,
preserves finite RDS values, checks emitted patient roles and replays detection
against training-tumor values. Counts, patient map, fit state, reference sample,
sample roles/factors, detection, matrices and implementations are bound in native
receipts. Source layouts are explicit: missing fold-local data never silently
fall back to cohort files. The association and validation producers read the
correct fold partition; the association graph also reads the corresponding
coding-gene partition. Whole-cohort legacy files are cached, but fold matrices
are released between folds rather than retaining five full copies in memory.

Local-CNV source receipts now distinguish fold-specific input reads, record
consumed eligibility columns and check per-output expression coverage. Core
inventory traversal follows only the expression partitions used by that output's
fold. Converted values are replayed from RDS before normalization sources are
followed. The complete raw-read sample table still participates conservatively
in external study/patient exclusion; this is not a fitted-row independence claim.

Synthetic verification executes actual edgeR calculations on 83 count rows and
123 aliquots. Independent direct edgeR training values, drastic held-out count
perturbations, training-state invariance, query-batch invariance, training-only
detection, five-fold native statistics with excluded-candidate closure, adjusted
preparation, frozen core forwards and actual external study exclusion are checked.
Only fixture dimensions are reduced. Full 33-cancer closeout is not executed.

This new path addresses lncRNA and coding-gene count normalization for native
local-CNV/adjusted preparation. Historical cohort-normalized outputs remain
historical; legacy direct preparation still uses its declared cohort input.
Candidate-universe selection and pathway-activity preprocessing can still carry
whole-cohort dependencies. GDC acquisition before count RDS, complete per-row
patient exposure and unsupported historical ancestry also remain unresolved.
No real cohort analysis, optimizer, paid compute or deployment was started.
Overall C01/C13/C16 acceptance remains open.

The fold_local_expression_r3 receipt records 31 passing cases, including the
new no-eligible-lncRNA case: candidate rows remain unavailable and empty G0/E1
outputs retain the complete downstream schema with a floating-point rho column.
The fold_local_expression_regression_r1 receipt records 40 passing historical
expression/preparation/CNV-source cases. These 71 distinct scoped checks ran on
149 without optimization. Earlier r1/r2 receipts are retained as intermediate
verification, not added again to the distinct latest count. The summary's old
top-level R parse-only label is corrected to scoped synthetic execution; the
scope for other, unexecuted R scripts is unchanged.

Training-fitted native ssGSEA activity (2026-10-19)
------------------------------------------------

The original 01_tcga_preprocess.R and 02_bulk_pathway_ssgsea.R were recovered
from /public8/DSC/CancerLncAtlas/R on host 149 and preserved as inactive text in
provenance/code_completion_20261015/ssgsea_original_sources, with source paths
and hashes. Their hardcoded real-cohort commands were not executed. The original
pathway method is actual GSVA ssGSEA, not the rank-mean fixture utility.

scripts/materialize_fold_local_ssgsea.R consumes each cancer's five-fold frozen
TMM normalization root, an explicit stable Ensembl/symbol annotation TSV and an
explicit GMT manifest. scripts/frozen_ssgsea.R fits training-tumor detection and
the training raw-score range. Coding genes require CPM >= 1 in at least
max(3, ceiling(0.20 * n_training_tumors)) samples. CPM is recovered from frozen
logCPM as 2^logCPM * (1 + 0.5/M) - 0.25*1e6/M, where M is the recorded mean
training effective library size. Protein-coding symbols must be present in the
annotation; duplicate symbols retain the native arithmetic mean of logCPM in
first-occurrence order. The ssGSEA rank background includes all retained coding
symbols, including those outside the supplied pathway sets.

The native minSize=10, maxSize=500, alpha=0.25 and original GMT set size <=500
are retained. Canonical IDs are MSIGDB:COLLECTION:GENE_SET_NAME. Actual GSVA
2.4.9 was inspected on 149: ssGSEA does not remove constant-expression rows;
normalize=TRUE divides raw scores by their global max-minus-min. The new path
uses normalize=FALSE and divides by the TRAINING raw-score range, with no
minimum subtraction or added z-score. Held-out patients cannot choose genes,
change symbol mappings or set the normalization denominator. The callable
apply function accepts independent query batches under that frozen state.

This is a new, versioned input path. Historical 01_tcga_preprocess.R used the
stranded_second count assay and summed duplicate stable rows. The current
fold-local expression source uses unstranded counts and first stable rows.
The new pipeline is not represented as numerical reproduction of historical
cohort-normalized scores. Raw count RDS exposure is still recorded as a whole;
only training/validation tumor matrices are exported, not outer-test matrices.

Manifest schema (paths and hashes must refer to actual server files)::

  {"format":"SSGSEA_GMT_INPUT_V1","collections":[
    {"collection":"HALLMARK","file":{"path":"/actual/hallmark.gmt","sha256":"..."},
     "study_metadata":{"dataset_id":"explicit_source_accession"}},
    {"collection":"REACTOME","file":{"path":"/actual/reactome.gmt","sha256":"..."}},
    {"collection":"KEGG_MEDICUS","file":{"path":"/actual/kegg_medicus.gmt","sha256":"..."}}
  ]}

Study metadata is optional but missing metadata remains an explicit inventory
gap; no study identifier is inferred from GMT descriptions or filenames. The
annotation requires ensembl_gene_id, gene_symbol and gene_type. Explicit study
columns in that annotation are also followed into external exclusion.

Command order, server 149 only, new outputs:

* Run the fold-local expression producer and converter documented above.
* Rscript scripts/materialize_fold_local_ssgsea.R NORMALIZATION_ROOT
  ANNOTATION_TSV GMT_MANIFEST_JSON NEW_NATIVE_ACTIVITY_ROOT
* python scripts/convert_fold_local_activity.py --source NEW_NATIVE_ACTIVITY_ROOT
  --output NEW_ACTIVITY_ROOT (repeat --source for additional cancers)
* Merge NEW_ACTIVITY_ROOT/SOURCE_CONFIG_FRAGMENT.json into the new local-CNV
  source config. pathway_activity_root uses layout=fold_local. Keep the same
  patient map and normalization sources as the fold-local expression inputs.
* Run the existing local-CNV association/validation and adjusted-preparation
  commands in the documented order. This does not authorize or launch training.

Every supplied pathway and every emitted sample retains a row. activity_score
is dimensionless training-range-normalized ssGSEA; raw_score is unnormalized
GSVA ssGSEA. gene_count counts unique input symbols; detected_gene_count counts
symbols after the training filter; coverage is their fraction. analysis_eligible
and unavailable_reason retain insufficient coverage, excessive original size,
no background genes, or a degenerate training range. Missing scores stay NaN.
Native train/validation association rows remain typed_unavailable for excluded
pathways; they are not silently dropped or interpreted as measured negatives.

ACTIVITY_SOURCES.json binds the R output, converter, native source receipt,
fold and cancer. Native receipts bind raw/scaled matrices, coverage, gene
filter, state, annotation, GMT files/manifest and expression normalization.
Conversion inspection replays gene filtering, coverage, training normalization
and Parquet values. It does not independently rerun the GSVA algorithm.
Actual input capture carries these dependencies through native local-CNV
statistics, adjusted core preparation, current model-use inventory and external
study exclusion. Inventory follows the consumed fold only and conservatively
retains raw-file patient exposure. Historical inputs without bindings remain
unverified rather than inheriting this implementation's guarantees.

The remaining C01/C13/C16 scope includes global candidate-universe selection,
global detection-derived scope/features, legacy direct-preparation input semantics, acquisition before count RDS,
complete row/patient ancestry, unsupported historical comparisons, consolidated
upstream command/schema records and final acceptance. No real cohort analysis,
optimizer, paid compute or deployment was started by these changes.

The next concrete upstream dependency is scripts/prepare_v32_formal.py:
load_scope still reads whole-cohort detection, asserts 16,889 lncRNAs and 4,712
shared lncRNAs, and uses detected_cancers in candidate metadata and model
features even in adjusted mode. The new fold-local measurements do not repair
that separate feature/universe selection. It must be addressed before claiming
fully training-local core preparation.

Verification receipts: fold_local_activity_r2 contains 14 passing new cases;
fold_local_activity_native_regression_r1 contains 31 passing expression/native
CNV cases; fold_local_activity_consumers_regression_r1 contains 68 passing
preparation, CNV-input, graph-expansion, core-prediction and model-use cases.
These are 113 distinct scoped checks on host 149, not a current whole-suite
run. The first activity iteration's 11 passing cases are preserved as an
intermediate source version and are not added to the latest distinct count.
The r2 test process finished successfully; its first SCP receipt download
failed transiently, and the existing completed receipts were then retrieved
without rerunning the test process. Only synthetic statistics and frozen
forwards ran; no optimizer was started.

Annotation-defined candidates and fold-local detection (2026-10-19)
-----------------------------------------------------------------

This section supersedes the global detection TODO above for the explicit
local-CNV-adjusted preparation path. It does not change historical results or
the legacy direct-preparation path. C01/C13/C16 remain in progress.

scripts/materialize_fold_local_scope.py constructs the candidate universe from
the supplied lncRNA annotation and all explicit GMT definitions. It never uses
measured expression, pathway coverage, correlation or outcome labels to select
candidates. It reuses the deterministic affine sampler with the declared seed
and budget. Every annotated lncRNA must fit in that budget. Unexpressed lncRNAs
and unusable pathways retain candidate rows; lack of measurements is not a
negative association. Candidate membership is the same in all five folds.

Each fold separately computes detection_rate as the fraction of selected
training tumors with frozen logCPM > 0. within_cancer_eligible means a rate of
at least 0.10. detected_cancers counts eligible cancers among the explicitly
declared set, and assessed_cancers is its size. shared requires eligibility in
at least three cancers; otherwise an eligible cancer/lncRNA is cancer_local.
Ineligible pairs are not_expression_eligible. All five scopes retain the full
annotation-by-cancer grid. There must be exactly one unfiltered normalization
row for every annotated lncRNA, with finite measurements and a common patient
map. Missing coverage is an error, not a zero detection rate.

Manifest schema, with actual server paths and SHA256 values::

  {"format":"FOLD_LOCAL_SCOPE_INPUT_V1",
   "annotation":{"path":"/actual/annotation.tsv","sha256":"..."},
   "gmt_manifest":{"path":"/actual/gmt_manifest.json","sha256":"..."},
   "normalizations":{
     "BRCA":{"path":"/actual/normalization/BRCA/SUCCESS.json","sha256":"..."},
     "LUAD":{"path":"/actual/normalization/LUAD/SUCCESS.json","sha256":"..."}},
   "budget_per_cancer":100000,"seed":20260726}

The annotation requires ensembl_gene_id and gene_type; gene_type=lncRNA rows
are used, _PAR_Y entries excluded, and stable Ensembl IDs version-stripped.
Stable duplicates are rejected. The GMT manifest follows the preceding
ssGSEA schema. No collection, study or hierarchy is guessed from names.

Command order, server 149 only:

* Produce all five frozen expression normalizations for every declared cancer.
* python scripts/materialize_fold_local_scope.py --manifest SCOPE_INPUT_JSON
  --output NEW_SCOPE_ROOT
* Use NEW_SCOPE_ROOT/candidates.parquet as the identical candidate authority
  for native local-CNV train and validation transformations. Use the same fold
  expression normalizations and patient map; produce fold-local ssGSEA as above.
* Complete the native association, validation and corrected graph authorities.
* Supply the usual explicit patient/graph/CNV authorities to
  scripts/prepare_v32_formal.py, with --association-mode local-cnv-adjusted,
  --formal-candidates NEW_SCOPE_ROOT/candidates.parquet, its SHA256,
  --fold-local-scope NEW_SCOPE_ROOT/SCOPE_RELEASE.json, its
  --fold-local-scope-sha256, and the same --budget-per-cancer as the manifest.

These commands are documented for subsequent authorized execution. This code
change did not run real preparation or L1 fitting. The actual preparation entry
point fits its baseline during a real run; synthetic checks replace that fit
with a fixed zero-logit object and prohibit optimizers.

CANDIDATE_SOURCES.json binds annotation, GMT and sampler implementations.
SCOPE_SOURCES.json binds each fold's normalization sources and scope values.
Source inspection replays candidate selection and detection from those actual
inputs, including nested GMT hashes. Explicit annotation/GMT study metadata
and raw normalization sample exposure reach model-use inventory. Missing GDC
acquisition ancestry before count RDS remains an explicit gap.

Adjusted preparation checks that each consumed train/validation association
uses the same native normalization as its fold's scope. The actual feature
tensors use that fold's detection and detected_cancers / assessed_cancers.
expressed_in edges use role=fold_train_expression_eligibility, split=train and
the actual outer fold. Candidates without eligible expression remain nodes
without expression edges. The graph rejects reuse of those edges in another
fold. G0/G1/G2 share their node and relation schema within each fold.

Outputs include FORMAL_LNCRNA_SCOPE/fold=F/scope.parquet and scope_by_fold in
PREP_INPUT_SUMMARY.json. No global FORMAL_LNCRNA_SCOPE.parquet or single shared
count is manufactured for this new mode. Legacy preparation retains its
existing global file and explicitly records all_samples_pre_outcome detection.
The frozen graph control receipt still verifies the old static detection file
as part of its historical authority; in the new mode its values are not read
for expression edges, whose binding instead points to the actual fold scope.

Integration limits that still require code work:

* Native per-cancer producers/finalizers retain the fixed 33-cancer,
  100,000-candidate production contract. Scope materialization supports other
  declared sizes, but those sizes do not yet close through all native commands.
  Synthetic tests reduce those dimensions explicitly and do not certify the
  full production closeout or an arbitrary subset workflow.
* The global prediction-scope boundary identified here is addressed by the
  subsequent section. Do not substitute one fold or a cohort-derived table
  for the native five-fold release.
* Matching association normalization is enforced. The separate corrected-graph
  normalization gap identified here is addressed by the subsequent graph
  consistency section. Activity, static pathway hierarchy and legacy
  preparation retain their own source constraints.
* Pre-RDS acquisition, complete per-row patient exposure, unsupported
  historical sources, final upstream orchestration/schema and acceptance remain
  open. Passing these checks does not prove full biological independence.

Synthetic checks cover actual edgeR-derived training detection, held-out
perturbation invariance, cross-cancer classification, rejected altered sources,
unexpressed candidate retention, native association normalization identity,
actual model feature tensors and graph construction. The actual adjusted main
entry writes all three graph arms for one synthetic fold; frozen cohort control
checks and baseline fitting are explicit substitutes. Five-fold training or
new real predictions were not executed.

fold_local_scope_r5 records 50 passing scope/preparation/graph/sampler cases;
fold_local_scope_native_regression_r1 records 31 passing frozen-expression and
native local-CNV ancestry cases; fold_local_scope_consumers_regression_r1 records
54 passing CNV-input, graph-expansion, core-prediction and model-use cases.
These 135 distinct scoped checks ran on host 149, without optimizer fitting.
The initial scope run exposed the legacy requirement that every candidate
must have an expression edge (8 passed, 1 failed); it was corrected to retain
unexpressed nodes. Intermediate passing runs are retained but not added again
to the latest unique count. This is not a whole-suite or real-cohort result.

Fold-local scope through current prediction and GeneSets (2026-10-19)
-------------------------------------------------------------------

The current-core inference manifest now accepts exactly one scope mode. Legacy
manifests retain candidate_authority and lncrna_scope. A native manifest instead
requires these four source records, each with an explicit path and SHA256::

  {"format":"CURRENT_CORE_PREDICTION_INPUT_V1",
   "core_run_id":"ACTUAL_AUTHORIZED_RUN", "graph_variant":"G2",
   "candidate_authority":{"path":"/actual/scope/candidates.parquet","sha256":"..."},
   "fold_local_scope":{"path":"/actual/scope/SCOPE_RELEASE.json","sha256":"..."},
   "patient_map":{"path":"/actual/patient_map.tsv","sha256":"..."},
   "pathway_hierarchy":{"path":"/actual/hierarchy.parquet","sha256":"..."},
   "fold_inputs":["the existing five explicit prepared/config/checkpoint/receipt records"]}

The fold_inputs illustration is a placeholder, not an executable manifest;
its existing object schema is unchanged. Candidate selection and each fold's
detection values replay from the bound scope producer. Exact pathway families
come from the explicit hierarchy. Each model's preparation sources must bind
the same candidate file, patient map and hierarchy and its own detection scope.
The actual inference feature tensors must match that fold's shared indicator,
detection_rate and detected_cancers / assessed_cancers. A native prepared model
cannot be annotated with a legacy global scope table.
Conversely, a historical checkpoint prepared with cohort detection cannot be
retroactively declared to have consumed these new fold-local inputs. Native
scope predictions require matching native preparation and checkpoint lineage.

Per-fold prediction files contain the actual fold classification,
scope_basis=OUTER_TRAIN_PATIENT_DETECTION and scope_outer_fold. The ensemble and
selected-score tables retain scope_fold_0 through scope_fold_4 and their counts:
scope_shared_fold_count, scope_local_fold_count, scope_ineligible_fold_count and
scope_eligible_fold_count. Each count has units of folds (0 to 5).
scope_basis=FIVE_TRAINING_FOLD_AGREEMENT describes these aggregate annotations.
If all five labels agree, shared_or_local_scope is their common value; otherwise
it is fold_variable. No majority vote, global expression classification or
single-fold substitute is used. These are descriptive ensemble annotations,
not inputs from other folds to an individual model.

All candidates, including not_expression_eligible rows, keep their predictions.
The numeric estimand remains the arithmetic mean of the five original fold
probabilities. Missing eligibility is not a zero probability or a negative
biological association. This change does not make an ensemble prediction an
out-of-fold patient effect or establish whole-pipeline independence.

Native selected analysis requires an explicit geneset.min_expression_folds
integer from 1 through 5. No default scientific threshold is imposed. A member
must reach the training detection threshold in at least that many folds and
meet the existing score, direction, size and tie policies. The chosen threshold
is part of the selected-analysis manifest and version; it is also exported as
geneset_min_expression_folds. Legacy inputs do not accept this threshold because
they lack fold evidence. Candidate scores and coverage retain all rows, while
GeneSet/GMT members exclude rows failing this explicit policy.

Per-fold membership frequency uses the actual expression eligibility and the
same directional score/size policy in that fold, with denominator five. Fold
score SD still uses all five score estimates. Rank-percentile means use only
eligible folds; zero eligible folds produces missing rank, not zero rank.
selected_fold_rank_eligible_count records the denominator. For producers without
matched fold estimates, fold statistics remain unavailable. Fold-variable
members may enter GeneSets under the chosen policy and are counted separately
from unanimous shared/local members.

The existing cross-cancer subtype estimand uses shared members. With native
scope it therefore uses only members classified shared in all five training
folds, labelled subtype_scope_policy=UNANIMOUS_TRAINING_FOLD_SHARED. Other
members remain in GeneSets and coverage; insufficient shared profiles retain
explicit unavailable subtype results. Legacy subtype behavior is unchanged.
The current HTTP tables preserve all fold fields, and the browser labels
fold-variable and ineligible states explicitly.

This closes the downstream representation boundary only. Fixed native cohort
size/budget closeout, pre-count acquisition ancestry, complete patient exposure
and final orchestration remain open under C01/C13/C16. The separate corrected-
coexpression normalization gap is addressed in the following section. No real
cohort preparation, fitting, training, paid compute or deployment is authorized
or performed by this change.

Verification: prediction_scope_r4 records 72 passing native-scope, current-core,
selected-analysis and typed-binding-expansion checks. prediction_scope_consumers_r1
records 38 passing model-use, current-primary-fusion and current-web checks.
These are 110 distinct scoped checks on host 149, not a full-suite run. The
native test executes real edgeR-derived scopes, native synthetic association
statistics, typed graphs, five frozen checkpoint forwards and replay, selected
scores, GeneSets, GMT, source inventory, actual HTTP handlers and Chromium.
Its frozen-cohort control metadata, untrained checkpoint serialization, fixed
L1 logits, pathway activity and coexpression are explicit synthetic fixtures.
The browser uses the actual page assets and API handlers with intercepted
transport; the regression suite also exercises the actual application server.

Initial failed checks are preserved: r1/r2 exposed missing typed-graph and
concrete-variant metadata in the new synthetic fixture; the fixture now uses
the actual graph materializer and arm binder. r3 completed native inference,
GeneSet and source-inventory assertions but lacked HTTP dependencies. r4 used
the existing isolated web verification dependencies. Production validation
was not weakened to accommodate these failures. No optimizer was run.

Corrected graph and detection use the same fold normalization (2026-10-19)
-----------------------------------------------------------------------

Training-patient identity checks alone allowed a corrected coexpression graph
and the detection/association features to use different expression versions.
The native fold-local graph path now follows the actual E1 producer's source
receipts before constructing the graph. It verifies that every declared cancer
has exactly one consumed corrected partition for the current fold, including
an explicit empty parquet when no edge passes. Extra, missing or duplicated
partitions and a missing transformation ancestry are errors.

Each corrected parquet must equal the actual present_E1/rho_E1 selection from
its bound native G0 statistics. The transformation must be the matching cancer,
fold and graph output, not a pair association or validation partition. Both
lncrna_expression and gene_expression must be native fold-local conversions
from the same normalization receipt used by detection. Their table values are
replayed from the native RDS conversion, not accepted solely because hashes
were updated. The transformation patient map must also match the native
normalization patient map. Changing only a coding-gene normalization is enough
to fail this check, even if lncRNA detection and patient identities still match.

The existing native command order is unchanged: generate fold-local expression,
candidate/scope and native adjusted associations; run the corrected-graph
materializer; then pass --fold-local-scope to formal preparation. The graph
producer's explicit local_cnv_adjustment.source_receipts supplies the E1
ancestry. No new source-path inference or global-expression fallback is added.
Historical graph preparation without the explicit native scope retains its
existing behavior and does not receive the new consistency claim.

Native graph bindings now contain fold_normalization_match with format
FOLD_LOCAL_GRAPH_NORMALIZATION_MATCH_V1, the outer fold, detection-scope record,
graph receipt, coexpression artifact and a record for every cancer. Each cancer
record includes the normalization and patient-map files, the actual lncRNA and
gene conversion/table pairs, E1 producer, native transformation, corrected
partition and its edge_rows count. This count describes that whole partition
before candidate filtering; it is not optimizer usage or a biological score.
The matching_fold_expression_normalization gate records the completed check.

Portable payload validation requires the complete native binding. Current
prediction and model-use inspection also replay its actual sources; a binding
claim alone is insufficient. Experimental/predicted expansion arms retain and
check their base graph's evidence. Older native prepared files without this
evidence need to be prepared again; their historical checkpoints cannot be
relabeled as having consumed a verified replacement graph.

This addresses expression-version consistency for the existing native adjusted
path. Full native cohort/budget closeout, fresh typed-authority/sidecar command
orchestration, pre-count acquisition, complete patient exposure and final
acceptance remain open. It does not establish full biological independence.

Verification: fold_graph_normalization_r2 records 28 passing native-source,
repinned-corruption, empty-partition and actual G0/G1/G2 preparation checks.
fold_graph_consumers_r1 records 14 passing current-scope checks, including five
frozen model forwards/replay, source inventory, selected scores, GeneSet/GMT,
actual HTTP handlers and browser consumption. fold_graph_regression_r1 records
38 passing native local-CNV ancestry, formal graph authority, C-overlay and
model-use regressions. These are 80 distinct scoped checks on host 149; no
optimizer or real cohort job ran. Native E1 calculations and corrected output
writers replace the earlier direct synthetic coexpression table. Frozen cohort
control metadata, untrained checkpoint serialization, fixed L1 logits, reduced
dimensions and synthetic pathway activities remain explicit test substitutes.

The first graph-normalization run passed 27 cases and failed while constructing
an empty-directory test fixture. The corrected fixture contains metadata but
no parquet, reaching the intended missing-partition check. The actual native
fixture has zero E1 edges in folds 0 and 4 and positive edge counts in folds
1 through 3; all five retain their cancer and normalization evidence. The
consumer run passed but its initial SCP receipt transfer failed during SSH
banner exchange. A read-only probe returned host 149 and the existing completed
receipts were downloaded without rerunning the tests or changing the VPN login.

Fresh typed global graph authority and formal preparation (2026-10-19)
--------------------------------------------------------------------

The formal preparation CLI previously always requested GENERIC_GLOBAL and
called the flat binding materializer. The older phase7 typed-authority script
depends on historical absolute directories and existing prepared files. A new
explicit command now creates the typed authority from a pinned source receipt
without needing historical prepared payloads::

    python -m scripts.prepare_v32_typed_graph_authority \
      --manifest /absolute/typed_inputs.json --output /absolute/new_typed_authority

The input format is TYPED_GLOBAL_GRAPH_AUTHORITY_INPUT_V1. source_receipt,
patient_map and patient_receipt are path/SHA256 records. binding_declaration
contains path, sha256, source_role=static_annotation, outcome_derived=false,
historical_model_output=false and toy_or_synthetic=false. These are explicit
input lineage declarations, not independently established biological facts.
fold_expression_pattern and fold_coexpression_pattern are absolute paths with
one literal {fold} placeholder each. The existing strict authority loader
validates every static input and all five fold declarations before publication.

The command runs only on host 149, records its target in LAUNCH_PREFLIGHT.json,
refuses output-directory reuse, preserves the source receipt and publishes
GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json plus SUCCESS.json. The new receipt
retains the original fold-local E1 ancestry and uses TYPED_ASSAY_CLASS_V1 with
binding_context_policy=GLOBAL_PHYSICAL_BINDING. Its preparation record pins
the source receipt, manifest and binding file. Old receipt counts are retained
as source_receipt_counts rather than presented as new graph totals.

Run the existing scripts.prepare_v32_formal command with all its explicit
patient, candidate, adjusted-association and graph arguments, adding
--graph-relation-schema-generation TYPED_ASSAY_CLASS_V1. Point the graph receipt
and its SHA256 at the new receipt, and use its binding artifact path/SHA256.
Continue supplying --fold-local-scope and its hash for the native training-only
detection/normalization path. Selecting a typed receipt without the explicit
generation fails. Typed preparation requires the global physical-binding
policy; it never silently globalizes cancer-restricted input rows.

Fresh typed G0/G1/G2 payloads use one common node inventory and full registered
relation schema. G0 has the base graph, G1 adds typed binding and protein-gene
encoding, and G2 additionally includes PPI. All arms use the existing feature,
candidate, label and source writers. Predicted binding is excluded, retaining
its source-row count in the preparation record. Unknown assay evidence, empty
identifiers, nonpositive/nonfinite weights and cancer/context restrictions are
rejected. Existing generic and historical typed materializers remain available
under their prior contracts. Portable validation also checks a fresh global
payload for flat, predicted, unknown or contextual binding edges.

The authority command never fits a model. Formal preparation retains its
existing L1 baseline fitting step when executed normally; code-only verification
substitutes fixed zero L1 logits and forbids optimizers and LogisticRegression
fitting. No real cohort preparation or training was launched for this change.

This closes fresh typed authority creation and direct formal payload writing.
The historical C-overlay/current-C comparison launcher still uses its existing
sidecar contract; it has not been replaced by the new direct payload route.
Native full-cohort closeout, complete acquisition/patient ancestry and final
orchestration/acceptance remain open. Existing historical checkpoints must not
be relabeled as trained on these new inputs.

Verification: typed_graph_preparation_r2 records 38 passing checks, including
26 fresh typed-preparation cases and the existing formal-authority/C-overlay
regressions. Each of the fifteen written payloads passes the actual prepared
validator and frozen encoder plus residual/direction heads with unchanged model
parameters. typed_graph_consumers_r1 records 52 passing binding-expansion,
native corrected-graph and current-prediction checks, including existing
GeneSet/GMT, source inventory, HTTP and browser consumers. These are 90 distinct
scoped checks on 149, not a full-suite or real-cohort run.

The first run's five G1 failures were incorrect test expectations: it expected
binding edges only in G2. The assertions now preserve the existing G1 binding
and encoding edges, with PPI added only in G2. Production arm semantics were
not changed. The initial failed receipt is retained alongside the passing
rerun. Unknown assays, invalid identifiers and invalid weights were also added
to the fresh global policy checks before the passing verification.

Native typed G2 enters the three-arm comparison directly (2026-10-19)
-------------------------------------------------------------------

This supersedes the preceding boundary that limited current-C comparison to
historical C graph sidecars. The formal preparation command now writes an
INPUT_MANIFEST.json in every G0/G1/G2 directory, pinning each actual serialized
fold file by path, SHA256 and byte size. A native scoped typed G2 row additionally
declares graph_input_mode=NATIVE_GLOBAL_PHYSICAL_BINDING. Other graph variants
and historical preparation modes do not receive that declaration.

To compare the newly prepared native G2, use the existing
scripts.prepare_v32_current_binding_comparison command and
CURRENT_C_BINDING_COMPARISON_INPUT_V1 manifest. Set config.task_contract fields
graph_variant=G2 and rbp_evidence_mode=NATIVE_GLOBAL_PHYSICAL_BINDING. Every
fold_inputs record supplies fold, graph_input_mode with that same value,
prepared={path,sha256}, and input_manifest={path,sha256} pointing to the actual
G2/INPUT_MANIFEST.json. The existing named native_october and rbns_sequence
source declarations are unchanged. Do not include graph_overlay,
global_binding_source or training_io.c_graph_overlay_pattern in this mode.

The comparison command validates actual native payloads and replays their
fold-normalization/E1 sources before constructing the arms. Its base must be
a fresh untrained typed global G2 with training-patient detection, a matching
normalization proof and explicit typed binding preparation ancestry. Mixed
native/historical folds, stale source modes and a historical overlay declared
alongside native direct inputs are rejected. Historical C_TYPED_ENCODE_V2 and
C_GLOBAL_PHYSICAL_BINDING configurations retain their sidecar path.

The expansion builder now preserves the baseline's expression eligibility
scope from its complete registered relation schema. This is necessary because
the previous cohort-default reconstruction rejected native training-patient
expressed_in edges. Reading the schema also preserves the intended scope when
an outer fold has no active expression edges. Expanded nodes remain appended
after existing indices; original candidate tensors, labels, non-graph inputs,
G0-derived features and coexpression runtime assignments are retained.

The actual trainer now calls native_graph_input.load_training_graph_inputs
before model or optimizer construction. It selects the declared direct or
historical source route, applies an authorized binding expansion and returns
the loaded payload. Frozen core export checks the same native source-mode
contract before applying the expansion. Both routes require a verified
expansion arm binding even when no historical graph overlay is present.
This shared loading stage can be exercised without calling the optimizer.

Comparison output remains the three arms current_c, experimental and
experimental_predicted, with five binding-expansion sidecars per arm. These
sidecars are the actual comparison differences; they do not pretend to be
historical C graph sidecars. Generated configurations remain CODE_ONLY with
training_authorized=false, CPU host 149 and zero paid budget. Fifteen command
records remain BLOCKED_CODE_ONLY. No training authorization or result is
created by successful preparation.

This change connects the newly generated native inputs to existing comparison,
training-input loading and frozen export code. It does not establish that a
real model trained on those inputs, or that all source/patient exposure,
native cohort closeout and final project acceptance requirements are complete.

Native prepared inputs initially carry their original source hashes in
artifact_hashes. The expansion writer/loader now uses the trainer's existing
input-authority accessor, preserving that mapping when input_authority_hashes
has not yet been created. It then appends expansion hashes without dropping
the original preparation source receipt. Historical overlaid inputs continue
using their explicit input_authority_hashes. This also binds the expansion's
parent hash to the actual native source mapping, rather than an empty mapping.

Verification: native_binding_comparison_r3 records 47 passing native-direct
and historical-expansion cases. Twenty-five new native cases include all
three arms across five folds through actual prepared serialization, authorized
trainer loading, frozen export restoration and forward calculations. Original
candidate/label tensor identity, source hashes, node indices, training-only
expression schema and unchanged model parameters are checked. Generated task
commands are exercised only to confirm they remain blocked before training.
native_binding_regression_r1 records 34 passing typed-preparation, training
objective and patient-authority checks. native_binding_consumers_r1 records
47 passing current export, source-inventory and scope prediction checks,
including GeneSet/GMT and HTTP/browser consumption. These are 128 distinct
scoped checks on host 149, not a full current-suite or real-cohort analysis.

The first failed run exposed the cohort-default expansion schema and a test
helper's import-time constant contamination. The second reached all fifteen
comparison outputs but failed source-receipt validation because the expansion
loader discarded artifact_hashes. Those implementation and test-isolation
issues were corrected; source validation was retained. Both failed receipts
remain available. Synthetic frozen-cohort controls, reduced closeout size,
fixed L1 logits, synthetic pathway/expansion evidence and untrained checkpoint
serialization are explicit test substitutes, not evidence of model training.

Declared local-CNV cohort and independent closeout (2026-10-19)
------------------------------------------------------------

The source configuration shared by the association and validation producers
now accepts candidate_definition={path,sha256} at its top level. This points to
the actual CANDIDATE_SOURCES.json from scripts.materialize_fold_local_scope.
The annotation/GMT inputs and deterministic candidate sampling are replayed;
the declared cancers, exact candidate keys, requested budget, observed counts
and all five patient folds are bound into LOCAL_CNV_ANNOTATION_COHORT_V1.
The current sampler requires the exact budget and rejects a budget larger than
the available no-replacement universe. It never silently shrinks the budget.
Without this declaration the historical 33-cancer, 100,000-per-cancer closeout
contract remains in force; a small observed file cannot redefine that contract.

Run the existing association command once per declared cancer, using one
explicit --run-id. Then call scripts.finalize_and_audit_v32_local_cnv_r3 with
that same --run-id and the existing explicit input paths. The finalizer's
historical default remains the R3 identifier. The independently selected run
identity is checked against both worker receipts and pair/edge rows. Candidate,
patient, source configuration and covariate inputs must match actual captured
reads. Missing, additional or mixed cancer/fold files prevent publication.
The existing scientific decision thresholds are retained unchanged.

Only a LOCAL_CNV_CORE_RETRAIN_REQUIRED decision authorizes the corrected
preparation route. A genuinely small-effect synthetic audit still produces
LOCAL_CNV_IMPACT_SMALL and remains ineligible for that route. On a qualifying
audit, the validation producer consumes the same candidate definition and
covariate authority. Its independent finalizer uses the declared cohort, checks
five partitions per cancer, candidate closure and validation patient roles,
and pins the actual parent audit SUCCESS. The corrected-E1 graph materializer
uses the same declared cancer scope and exact five-fold partition layout.

Formal preparation checks matching training/validation cohorts, the validation
parent SUCCESS, the actual requested candidate/patient files and, when supplied,
the matching native expression-scope candidate definition. This closes the
fixed-size closeout mismatch for explicitly declared native candidates. It
does not authorize fitting or declare real cohort analyses complete.

Actual synthetic closeout exposed a numerical defect: constant expression or
an outcome fully projected out by covariates could leave tiny floating-point
residuals and appear estimable. Scalar and vector local-CNV correlations now
apply the existing effective-variance tolerance to each residual separately;
the full A0 comparator also checks estimability before reporting a correlation.
Constant CNV remains a no-op adjustment when expression/activity are variable.
Unestimable candidate rows remain present with unavailable effects and labels.

The new closeout fixture runs real two-cancer, five-fold association workers,
independent train/validation finalizers and corrected-E1 materialization. It
does not patch cancer/budget constants or fabricate aggregate success/decision
receipts. Expression/activity tables, compact CNV arrays, annotation/GMT and
minimal parent graph metadata are synthetic inputs. It does not independently
validate raw acquisition, native count normalization or complete graph metadata;
those have separately scoped checks. Overall C01/C13/C16 acceptance remains open.

Verification: local_cnv_cohort_r5 records 65 passing closeout, source-ancestry,
core-adapter and statistical checks. Three actual scenarios retain the original
core-required, small-effect and insufficient-rank-review outcomes. Corrupted
cohort counts, budget, patient folds, worker identity, partitions, audit hashes
and cohort-removal attempts are rejected. local_cnv_cohort_preparation_r1 records
38 passing fold-local scope and typed-preparation checks, including actual
native statistical preparation and G0/G1/G2 payload writing.
local_cnv_cohort_consumers_r1 records 33 passing native binding-comparison and
model-use checks, including actual graph loading and frozen export restoration.
These are 136 distinct scoped checks on 149, not a full current-suite run.

Failed r1-r4 receipts are retained. They exposed a fixture budget beyond its
universe, an implementation-pin count mismatch, the hardcoded closeout run ID,
projection roundoff and an incorrect small-effect expectation when only two
rank-comparable candidates remained. The last scenario now explicitly expects
review; the small-effect scenario uses three variable outcomes. Scientific
decision thresholds were not relaxed to make a test pass.

GDC STAR files to native count RDS and downstream source inspection (2026-10-19)
----------------------------------------------------------------------------

scripts.materialize_gdc_star_counts provides two explicit host-149 commands.
The query command archives original GDC API response pages, requested filters,
pagination and retrieval times. It requires released open RNA-Seq STAR count
files and complete declared cancer coverage. Each file must map through one
associated aliquot and the biospecimen tree to one sample and patient. UUIDs,
barcodes, projects, workflow, filename, size and MD5 are checked. Duplicate
files/aliquots, ambiguous mappings and incomplete pages fail before publication.
Normal and additional tumor aliquots remain in the count input; the existing
normalization stage makes the patient/sample selection.

Example command order, using explicit new directories on 149::

  python -m scripts.materialize_gdc_star_counts query --cancers BRCA,LUAD --output /new/star-query
  python -m scripts.materialize_gdc_star_counts materialize --manifest /new/star-query/QUERY.json --cancer BRCA --download-root /verified/gdc-cache --output /new/BRCA-counts
  Rscript scripts/materialize_fold_local_expression.R /new/BRCA-counts/TCGA-BRCA.RDS /frozen/patients.tsv BRCA /new/BRCA-normalization

Repeat assembly/normalization for each declared cancer, then use the existing
convert_fold_local_expression.py command and fold-local ssGSEA/scope consumers.
The materializer reads the standard file-UUID/filename GDC cache layout. It
requires existing files by default; --download explicitly enables fetching
missing open files. Downloads use temporary files and must match the archived
GDC byte size and MD5 before being published. Existing invalid files are not
silently overwritten. Successful reads also receive SHA256 bindings.

The actual R assembler produces a SummarizedExperiment with unstranded,
stranded_first and stranded_second integer count assays; no normalization,
feature filtering or sample aggregation occurs here. Versioned gene IDs,
symbols and gene types must agree across files. The four STAR N_* summary rows
are preserved as metadata and excluded from gene matrices. Counts must be
finite nonnegative integers below 2^53, and all gene/sample identities remain
explicit. Units are raw read counts. Original source tables and API pages stay
unchanged; output goes to a new directory.

COUNT_INPUTS.json pins the actual raw files, query and assembly implementations.
The RDS embeds its hash; TCGA-<cancer>.RDS.sources.json binds that declaration,
the resulting RDS and SAMPLE_SOURCES.tsv. Both native normalization commands
now carry this optional assembly binding forward. A native RDS that declares
this ancestry cannot silently lose its sidecar during normalization. Existing
unbound count RDS inputs retain their explicit unresolved-ancestry gap.
The count SUCCESS must pin the completed sidecar before downstream consumption;
an in-progress assembly is not an eligible normalization input.

Inspection of fold-local expression, ssGSEA and training detection, as well as
legacy expression, follows the pinned assembly binding. First inspection
reassembles the three assays from the raw files and compares actual RDS values,
gene/sample metadata and STAR summary rows. Hash-only repinning of altered
count values is insufficient. Query pages, raw files, sample exposure and code
versions reach the existing model-use inventory. This is whole-file exposure,
including held-out samples read during input preparation, not an assertion
that every sample was fitted or that patient mapping is globally complete.

Explicit source_dataset=TCGA-<project> now produces a GDC_PROJECT provenance
token. This conservatively flags observations from the same registered cohort
in external evaluation; it does not count projects as independently verified
publications. The token is limited to external/model-use isolation metadata;
shared evidence-rule study counting retains its previous behavior.
No identifier is inferred from a filename. GDC alignment/BAM
processing and the biological truth of submitted biospecimen identities remain
unverified. The old acquisition-before-count-RDS gap is replaced only when the
actual bound raw-to-count transformation replays successfully.

Format references: GDC documents the three STAR count channels at
https://gdc.cancer.gov/node/1661 and file/biospecimen API retrieval at
https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/ .
Verification uses archived synthetic API responses, generated STAR-shaped
files and actual R/SummarizedExperiment/edgeR execution. No live assay download,
real cohort preparation, optimization or training is authorized by these checks.

Verification: gdc_star_counts_r3 records 19 passing assembly, replay, raw-format,
metadata, checksum, completion and native-source checks. The actual fixture has
122 raw files for 120 synthetic patients, 83 gene rows and three distinct count
assays. gdc_star_transforms_r1 records three passing actual native scope/ssGSEA,
legacy-TMM and missing-sidecar checks. gdc_star_consumers_r3 records 34 passing
frozen core/external-consumer, cross-file identity and unchanged evidence-rule
checks. gdc_star_regression_r1 records 28 passing existing-expression ancestry
and fold-local normalization checks. These are 84 distinct scoped checks on
149, not a complete current-suite run or a real-cohort validation.

The initial counts-r1 failure exposed the missing TCGA project overlap token;
the failed receipt is retained. Its correction is limited to external/model-use
isolation and is explicitly tested not to add independent rule evidence.
Frozen downstream fixtures retain fixed L1 controls and untrained checkpoint
serialization; their purpose is consumption and isolation, not estimated gain.
Two transient SSH interruptions affected receipt return. Read-only probes
confirmed hostname 149, completed original receipts were recovered, and the
running regression check was not duplicated. No iOA login or reconnection was
performed. Remaining C01/C13/C16 consolidation and acceptance are still open.

Native expression preprocessing workflow (2026-10-19)
----------------------------------------------------

scripts.run_native_expression_workflow joins existing cached-STAR assembly,
five-fold TMM, five-fold ssGSEA, both native conversions and annotation-defined
candidate/detection scope. It runs on 149 and ends before CNV associations,
model fitting, frozen prediction and publication. It does not download files.
The existing GDC metadata/acquisition entry remains a separate prerequisite.

The input JSON has exactly these fields::

  {
    "format": "NATIVE_EXPRESSION_WORKFLOW_INPUT_V1",
    "cancers": ["BRCA", "LUAD"],
    "gdc_query": {"path": "/input/query/QUERY.json", "sha256": "<actual hash>"},
    "download_root": "/input/gdc-cache",
    "patient_map": {"path": "/input/patients.tsv", "sha256": "<actual hash>"},
    "annotation": {"path": "/input/annotation.tsv", "sha256": "<actual hash>"},
    "gmt_manifest": {"path": "/input/gmt.json", "sha256": "<actual hash>"},
    "budget_per_cancer": 100000,
    "seed": 20260726
  }

Cancer coverage must exactly match the archived query and patient map. Patient
IDs are canonical TCGA case barcodes; every cancer must have all five folds,
and every declared patient needs a primary-tumor aliquot. Patient IDs cannot
appear in two cancer rows. The annotation/GMT formats are the existing native
formats documented above. GMT collection files use absolute pinned paths,
as required by the current native R readers. Candidate budgets remain bounded
by the existing annotation-only sampler and are checked before assembly.

Commands, to execute on 149 with a new output directory::

  python -m scripts.run_native_expression_workflow --manifest /input/workflow.json --output /new/preprocessing --plan
  python -m scripts.run_native_expression_workflow --manifest /input/workflow.json --output /new/preprocessing

--plan verifies the archived query, actual cached byte hashes, patient map and
annotation/GMT candidate declaration, then prints exact argument arrays. It
does not create the output directory or invoke scientific producers. Planning
reads the declared input files and performs candidate validation; it is still
a server-149 operation. The execution path invokes only the listed existing
preprocessing commands; arbitrary commands and training stages are not inputs.

LAUNCH_PREFLIGHT.json pins the manifest, actual raw files, external metadata,
implementation files and command arguments. STAGE_STATUS.json records each
running/passed/failed stage and its output log. A failed producer stops the
workflow, preserves earlier results for inspection and leaves no workflow
SUCCESS. Existing output directories are never reused or overwritten.

Native reader acceptance checks every cancer and all five folds using the same
expression_path, activity_path and scope.release readers as downstream native
analysis. This reconciles fold identities, the patient map, common expression
normalization, candidates and source bindings. NATIVE_EXPRESSION_BINDING.json
records the resulting partitions and scope. SOURCE_CONFIG_FRAGMENT.json merges
the three fold-local source roots and candidate_definition for the existing
local-CNV source config. It is a fragment: compact CNV, covariates and other
required inputs must still be supplied to the downstream command. The binding
also supplies candidate and scope paths for formal preparation.

The workflow validator rechecks recorded inputs, command-stage closure, logs
and actual native output readers. A workflow PASS means only preprocessing
completed for its declared data. It does not certify complete biological
ancestry, independent validation, training, CNV analysis or final publication.

Verification: native_expression_workflow_r1 records ten passing checks on 149.
The actual CLI --plan leaves its destination absent. The actual execution then
processes 40 synthetic raw STAR files for BRCA/LUAD, 20 synthetic patients per
cancer, all five folds and 30 expression/activity partitions. Native consumers
retain unavailable candidates, matching normalization, scope and original raw
file dependencies. Missing primary-tumor patients, corrupt caches, wrong cancer
scope, invalid budgets, relative GMT members and extra command instructions
fail before output. A malformed raw count reaches the real R parser and leaves
a failed counts stage with no success receipt or later normalization stage.
Changed original bytes invalidate the completed workflow. No real cohort or
optimizer ran, and no source acquisition or production deployment occurred.

Native local-CNV command workflow and preprocessing handoff (2026-10-19)
------------------------------------------------------------------------

scripts.run_native_cnv_workflow now executes the existing cancer workers,
independent association closeout, validation-A1 workers/audit and corrected-E1
graph phases in their required order. It requires an explicit annotation/GMT
candidate_definition and all five patient folds. It never fits an L1 baseline
or neural model. Formal core preparation includes L1 fitting and remains a
separate, unexecuted stage; typed binding preparation and the three-arm
comparison are also explicitly outside this workflow's completion claim.

The input JSON fields are::

  {
    "format": "NATIVE_LOCAL_CNV_WORKFLOW_INPUT_V1",
    "run_id": "<new explicit run identity>",
    "candidates": {"path": "/input/candidates.parquet", "sha256": "<actual hash>"},
    "patient_map": {"path": "/input/patients.tsv", "sha256": "<actual hash>"},
    "source_config": {"path": "/input/sources.json", "sha256": "<actual hash>"},
    "directional_success": {"path": "/input/cnv/SUCCESS.json", "sha256": "<actual hash>"},
    "graph_root": "/input/graph",
    "graph_receipt": {"path": "/input/graph_authority.json", "sha256": "<actual hash>"},
    "expression_workflow": {"path": "/input/preprocessing/NATIVE_EXPRESSION_BINDING.json", "sha256": "<actual hash>"}
  }

expression_workflow is optional for independently prepared native source
configs. When supplied, candidates, patient map, candidate_definition and all
three expression/activity source roots must exactly match that completed
workflow. Start with its SOURCE_CONFIG_FRAGMENT.json and add the independently
prepared covariates source to construct the complete source config. The input
config is never silently rewritten. All native source paths must be absolute.
The directional aggregate must have passed and cover exactly the declared
cancers. The two static tables actually read by CNV statistics must match the
source graph receipt. Other formal graph requirements remain enforced by the
existing downstream typed/formal graph loader.

Commands on server 149::

  python -m scripts.run_native_cnv_workflow --manifest /input/cnv_workflow.json --output /new/cnv-analysis --plan
  python -m scripts.run_native_cnv_workflow --manifest /input/cnv_workflow.json --output /new/cnv-analysis

--plan checks actual inputs and prints command argument arrays without creating
the destination or running producers. A later-stage SHA256 dependency is an
explicit sha256_of=association_success reference; execution resolves it only
after successful independent association closeout. Every worker and finalizer
receives the declared run ID, candidates, patient map, config and relevant
source paths. No historical path defaults or arbitrary commands are accepted.

The original decision gates control execution:

* LOCAL_CNV_CORE_RETRAIN_REQUIRED runs validation workers, their independent
  audit, corrected graph initialization, per-cancer materialization and final
  graph closeout.
* LOCAL_CNV_IMPACT_SMALL records subsequent stages as NOT_RUN_BY_DECISION and
  does not produce corrected validation/graph outputs.
* LOCAL_CNV_SENSITIVITY_ONLY_REVIEW likewise stops corrected production and
  retains the need for scientific review. It is not relabelled as small impact.

validate_local_cnv_audit keeps require_correction=True by default for existing
corrected-core consumers. The workflow's explicit require_correction=False
accepts only the three known, independently audited decisions, preserving
their cohort/hash/covariate checks. Declared patient identity is now unique
across cancers as well as within each cancer; duplicated canonical patients
cannot enter a second cancer under a different fold assignment.

The workflow records preflight hashes, resolved per-stage commands, logs and
failure stages. SUCCESS pins NATIVE_CNV_BINDING.json. That binding distinguishes
audit completion from corrected_inputs_available and explicitly records that
core preparation, typed binding preparation, comparison and training did not
run. An actual producer failure preserves diagnostics and leaves no workflow
SUCCESS. A new destination is required for each execution.

Completion validation checks original sources, implementations, every worker's
consumed-source receipt, native independent audits and corrected graph hashes.
When an expression workflow is supplied, it is revalidated at closeout and on
subsequent validation, including its original cached GDC files. The resulting
association and graph receipts retain their existing native source paths so
model-use inspection can continue through normalization to original raw files.
This does not resolve missing historical ancestry, full per-row patient use,
the remaining typed/formal/comparison command orchestration or final acceptance.

Verification: native_cnv_workflow_r2 records 37 passing checks on 149: thirteen
workflow/decision/error checks, three actual GDC-to-CNV source-consumption checks
and twenty-one declared-cohort/closeout regressions. Two-cancer fixtures with
120 synthetic patients per cancer execute all three original decision paths;
the correction-required path runs all ten commands, creates thirty validation
rows and ten cancer/fold corrected graph partitions. A separate actual GDC
preprocessing workflow uses forty synthetic STAR files and twenty patients per
cancer, then feeds the real association workers and independent closeout.
Native model-use inspection reaches all forty original raw files.

The intermediate integration-r1 failure was an incorrect test assumption:
twenty patients per cancer leave twelve training patients in each outer fold,
which meets the existing MIN_MATCHED_PATIENTS=12 threshold. The final test
checks finite estimable effects for variable expression and typed unavailable
rows for constant expression. No scientific threshold or calculation was
changed to satisfy that assertion. The failed receipt remains available.
The synthetic static graph receipt is a limited source control, not a fully
audited frozen core graph authority. No L1 fitting, neural optimization, real
cohort analysis, source download or production publication was executed.

Native typed/core/comparison workflow (2026-10-19)
-------------------------------------------------

scripts.run_native_core_workflow connects fresh typed graph authority, native
formal G0/G1/G2 preparation and the current/experimental/experimental+predicted
binding comparison. All three stages use their existing scientific producers.
The completed release contains fifteen prepared files, fifteen comparison
overlays and thirty blocked training commands. It does not run neural training
or create trained neural checkpoints. The core preparation stage DOES include
the existing five-fold L1 baseline fitting; execution is refused unless
--allow-baseline-fitting is explicitly supplied. That flag does not enable
any of the generated neural training tasks. This code-only project has not
authorized or performed real baseline fitting.

The input manifest has these fields::

  {
    "format": "NATIVE_CORE_WORKFLOW_INPUT_V1",
    "release": "<new-lowercase-release-token>",
    "typed_graph_manifest": {"path": "/input/typed.json", "sha256": "<hash>"},
    "candidates": {"path": "/input/candidates.parquet", "sha256": "<hash>"},
    "fold_local_scope": {"path": "/input/SCOPE_RELEASE.json", "sha256": "<hash>"},
    "local_cnv_success": {"path": "/input/cnv/association/SUCCESS.json", "sha256": "<hash>"},
    "validation_success": {"path": "/input/cnv/validation/SUCCESS.json", "sha256": "<hash>"},
    "authorization_hashes": {"path": "/input/hashes.json", "sha256": "<hash>"},
    "comparison_config": {"path": "/input/native-g2.yaml", "sha256": "<hash>"},
    "native_cnv_workflow": {"path": "/input/cnv/NATIVE_CNV_BINDING.json", "sha256": "<hash>"},
    "expansion_sources": [
      {"source_id": "october", "kind": "native_october",
       "edges": {"path": "/input/october_edges.parquet", "sha256": "<hash>"},
       "lineage": {"path": "/input/october_lineage.parquet", "sha256": "<hash>"}},
      {"source_id": "rbns", "kind": "rbns_sequence",
       "audit": {"path": "/input/rbns/AUDIT.json", "sha256": "<hash>"},
       "binding": {"path": "/input/rbns/RBNS_TYPED_BINDING.parquet", "sha256": "<hash>"}}
    ]
  }

The expansion_sources entries use the existing comparison producer's native
source schemas. native_cnv_workflow is optional for independently prepared
native inputs. When supplied, its exact corrected graph, association and
validation outputs must match this request, and its original sources are
rechecked at closeout and on subsequent validation. A small-impact or review
CNV outcome cannot be passed to this corrected-core workflow.

typed_graph_manifest uses TYPED_GLOBAL_GRAPH_AUTHORITY_INPUT_V1 and explicit
five-fold expression/coexpression patterns. Its source_receipt must be the
matching corrected graph. The frozen patient authority, annotation/GMT
candidate definition, fold-local scope and train/validation cohorts must
agree. All graph declarations retain the existing source-role and hash
requirements. comparison_config must describe unexpanded native G2 using
NATIVE_GLOBAL_PHYSICAL_BINDING. The frozen seed is 20260726; a different
crossfit seed is rejected before output creation. Historical C overlays and
already expanded configurations are rejected. authorization_hashes is the
existing formal preparation's source-hash mapping, not permission to train.
Its optional formal_graph_authority_receipt_sha256 must match the newly
generated typed receipt; an old receipt hash cannot authorize a new graph.

Non-executing plan command on server 149::

  python -m scripts.run_native_core_workflow --manifest /input/core.json --output /new/core --plan

The plan checks actual input authorities and sources without creating the
destination or fitting a model. A sha256_of=typed_receipt dependency resolves
after typed publication. Future execution, only when L1 fitting is authorized,
uses the same command without --plan and with --allow-baseline-fitting. The
workflow calls the existing Python command entry points in-process with
explicit argument lists; it does not accept arbitrary executable commands.
prepare_v32_formal.main now accepts optional argv and preserves default CLI
argument parsing.

The output layout is typed/, prepared/G0/, prepared/G1/, prepared/G2/,
comparison/<arm>/ and training/<G variant>/. TRAINING_COMMANDS.json covers
G0, G1, G2, current_c, experimental and experimental_predicted, each with
folds 0--4 exactly once. The prepared payloads are shared by comparison arms;
only their binding overlays differ. Generated configurations use server149,
CPU/FP32 and CODE_ONLY with training_authorized=false. No training approval
file is created. The actual run-shard entry retains its authorization guard.

Before SUCCESS, the workflow loads all thirty commands through the real
trainer's authorized prepared-input and graph-loading functions, validates
prepared payload semantics and checks their graph generation. It also checks
task/config/fold/seed correspondence and records accepted artifact hashes.
This acceptance does not fit a model or execute a forward pass. STAGE_STATUS
and individual logs preserve producer failures; failed work has no workflow
SUCCESS. A new output directory is required. NATIVE_CORE_BINDING.json links
preflight inputs, producer receipts, comparison inputs and blocked commands.
Validation rechecks these files, original sources and actual input acceptance.

The corrected-E1 finalizer now accepts source authorities without optional
counts, including the new typed generation that deliberately drops stale
parent binding totals. It calculates the two new coexpression totals from
actual verified output partitions; it does not reconstruct old binding counts.

This closes the native corrected-graph to formal/comparison command handoff.
It is not the final raw-to-publication workflow, a scientific performance
comparison, a whole-pipeline patient-independence certificate or completion of
historical data analyses. Those separate scopes remain visible in the plan.

The native CNV workflow now also verifies and pins each original segment file
listed in the directional materializer's SOURCE_FILES.json records array.
Previously it pinned that manifest and the compact matrices but did not check
changes in the underlying segment bytes during workflow validation. The new
pins are checked before execution, at closeout and during later validation,
including through a bound core workflow. This is direct recorded segment
provenance, not a claim about upstream assay generation or all historical data.

Verification: native_core_workflow_r3 records 62 passing checks on host 149:
43 new core workflow/forward/error checks, 15 native CNV workflow regressions
and four formal graph-authority regressions. The core fixture uses one cancer,
120 synthetic patients, three candidates and actual native CNV association,
decision closeout, validation-A1 and corrected-E1 commands. It then executes
all three preparation entry points in-process. All thirty generated training
commands are rejected by the actual run-shard guard; the thirty actual model
forward cases preserve parameter hashes. Duplicate/mismatched fold tasks,
different seeds, mixed authorities, producer failure and changed original CNV
bytes are checked. The separate native CNV regressions retain the correction,
small-impact and review branches with unchanged scientific thresholds.

Substitutes are explicit: synthetic biological inputs, reduced frozen-cohort
control identities and five fixed zero-logit L1 calls. LogisticRegression.fit
and neural optimizer construction are forbidden in the core test process.
No native CNV decision/aggregate audit is replaced in this workflow fixture.
Actual L1 fitting, real cohorts, paid compute and production deployment were
not executed. The failed r1/r2 receipts remain recorded: r1 exposed missing
optional graph counts, and r2 exposed missing original segment-byte pins.

Completed native runs to frozen results and publication (2026-10-19)
-------------------------------------------------------------------

scripts.run_native_results_workflow consumes completed runs from the preceding
native core preparation. It never invokes training, auxiliary-head fitting or
fusion fitting. A single explicit request validates all six groups, locks the
existing G0/G1/G2 validation winner, produces six full-candidate prediction
releases, exports the selected core, prepares its private fusion labels, and
calls the existing selected-analysis/context/publication workflow. Production
deployment is not performed.

The required JSON fields are format=NATIVE_CORE_RESULTS_INPUT_V1, release,
core_preparation, runs, geneset, subtype and publication_components. release
is a lowercase token. core_preparation pins NATIVE_CORE_BINDING.json. runs
must contain exactly G0, G1, G2, current_c, experimental and
experimental_predicted. Each group has this schema, with five distinct fold
records for fold values 0 through 4::

  {
    "config": {"path": "/completed/G0/config.yaml", "sha256": "<hash>"},
    "input_manifest": {"path": "/completed/G0/INPUT_MANIFEST.json", "sha256": "<hash>"},
    "task_manifest": {"path": "/completed/G0/TASK_MANIFEST.tsv", "sha256": "<hash>"},
    "folds": [
      {"fold": 0,
       "checkpoint": {"path": "/completed/G0/fold0/best_model_state.pt", "sha256": "<hash>"},
       "training_receipt": {"path": "/completed/G0/fold0/SUCCESS.json", "sha256": "<hash>"}}
    ]
  }

The example shows one fold only to describe its fields. All five are required.
Each receipt must describe completed optimizer work and full validation chunk
coverage, match its actual V4 checkpoint, and bind the current code tree,
configuration, input manifest and task manifest. This is a check of recorded
completed artifacts, not independent observation of a training process.
Missing results, a changed scientific configuration, mismatched checkpoint,
duplicate fold, changed input or incomplete task inventory stop before output.

The original preparation remains immutable. A future training run uses its
own config/task copies: execution_control may change from CODE_ONLY to an
authorized TRAINING configuration, and training_io.output_root may move.
All other settings must remain identical, including model, seed, runtime,
budget, prepared paths and binding overlay. Input manifests retain the exact
prepared authority contents. Tasks retain their run/fold/seed identity and
server149 CPU owner and have PENDING status as in the existing authorized
training plan. This workflow neither changes these files nor grants training
authorization. It does not require historical authorization/training folder
names. The existing G012 selector now accepts explicit pinned run declarations
in addition to its unchanged legacy directory interface.

geneset and subtype use SelectedGeneSetConfig and RankedSubtypeConfig fields.
For native fold-local inputs, geneset.min_expression_folds is explicit (1--5).
It governs GeneSet eligibility; it does not remove full prediction rows or
change the five-fold probability mean. Subtype settings retain their existing
rank/composition semantics and do not define patient molecular subtypes.

publication_components contains the existing nine component names: state,
clinical, drug, evidence, external, single_cell, patient, genomic and distal.
Each uses BUILD plus a pinned input_manifest, REUSE plus a pinned binding, or
an explicit NOT_PRODUCED/SOURCE_UNAVAILABLE/NOT_APPLICABLE status and reason.
Configured native contexts are validated against the selected analysis by
their existing consumers; missing contexts remain visibly missing. A PASS
does not imply that all nine analyses exist.

BUILD templates may reference results with exactly {"result_ref": "name"}.
The allowed names are selected_model (the direct primary model declaration),
core_prediction (its prediction binding), core_run_id (its training run ID),
core_export (its embedding manifest), and primary_fusion (private label
binding). Referenced input paths are resolved relative to their original
template and pinned before the template is copied. Existing workflow_ref
references to selected_analysis, selected_score and earlier published stages
are left for the existing publication workflow to resolve.

For external evaluation of the selected score, use producer_kind=
selected_release, score_column=selected_score, producer_binding=
{"workflow_ref":"selected_score"} and run_id={"result_ref":"core_run_id"}.
With model_use_inventory={"derive_from_native_models":true}, the existing
publisher reconstructs the selected score's native model-use inventory,
including all three G012 comparison arms. This preserves losing-model
selection exposure; the direct primary model declaration alone does not
represent the G012 selection decision. Incomplete study or patient mappings
remain explicitly unverified.

Commands, on server 149 and with a new output directory::

  python -m scripts.run_native_results_workflow --manifest /input/results.json --output /new/results --plan
  python -m scripts.run_native_results_workflow --manifest /input/results.json --output /new/results

--plan checks completed checkpoints, sources and real input loaders without
constructing models, fitting, forwarding or creating the destination. The
execution command uses actual frozen model forwards. LAUNCH_PREFLIGHT and
STAGE_STATUS record inputs and stage logs. Failures preserve the failed stage
without a workflow SUCCESS. NATIVE_RESULTS_BINDING.json links all six
prediction releases, selection, binding comparison, selected core export,
private fusion input and current publication. Validation rechecks original
sources, all prediction ensembles, comparison losses, native label replay,
export file hashes and registered public consumers.

G012 selection still minimizes the arithmetic mean of the five recorded best
validation losses, with the original G0/G1/G2 tie ordering. It does not read
sealed test data. BINDING_COMPARISON.json separately reports the same five
recorded losses and their arithmetic mean for each binding arm, retaining
all sources and predictions. It does not replace the G012 winner or claim
independent scientific improvement. Full-candidate predictions retain the
existing weighted-chunk-logit then sigmoid estimand and arithmetic mean of
five fold probabilities. Fusion labels retain their actual availability masks
and VALIDATION_REUSED_FOR_FUSION_DEVELOPMENT status; unavailable labels are
NaN and are not negatives.

This workflow consumes completed checkpoints and any declared native auxiliary
results. It does not acquire missing data or fit their models. The overall
code acceptance and complete source/patient coverage audits remain open.

Verification: native_results_workflow_r2 records twelve passing checks on 149:
nine native results integration/error cases and three legacy G012 selector
regressions. The actual --plan and execution entry points consume thirty
synthetic V4 checkpoint containers derived from the real native preparation
fixture. All thirty frozen prediction folds and five selected embedding
exports run. The original selector chooses G1 from the declared synthetic
validation losses. Native fusion label views exactly match the prepared masks,
including fold-varying eligibility and NaN values for unavailable labels.
The actual current publisher builds selected scores/GeneSets, native model-use
inventory, external V2 isolation and web registration. Real HTTP queries return
all three candidates and the expected lncRNA-overlap exclusion. G0's losing
comparison context remains in the model-use dependency table.

Checks reject incomplete groups, duplicate folds, altered scientific config,
wrong tasks, missing optimizer-completion evidence and mismatched checkpoints
before output. A hash-consistent but structurally damaged model state reaches
the real model loader and leaves a failed prediction_G0 stage without workflow
SUCCESS or publication. Changing a losing-arm checkpoint invalidates the
completed workflow. Checkpoint files remain byte-identical after successful
frozen inference/export.

The test containers are serialized untrained random model states, with
explicit synthetic_serialization_fixture and no_optimizer_actually_run flags.
Their optimizer-success fields exist to exercise strict downstream artifact
contracts; they are not evidence that an optimizer ran. The upstream fixture
also substitutes frozen full-cohort control identities and fixed L1 logits.
Actual optimizer construction and LogisticRegression.fit remain forbidden.
The failed r1 receipt records an incorrect test assumption of exactly one
available label per fold; the final test compares actual aligned masks. No
scientific threshold or availability calculation was changed for that test.

That workflow check verified traversal and exclusion. At commit 53a4ff6 the
native inventory still hard-coded completeness=false and mixed study-free
entity exposure with study-source coverage. The following change addresses
that decision policy; complete-core ancestry remains a separate open boundary.

Evidence-derived registered-source completeness
----------------------------------------------

The native model-use producer now writes COMPLETENESS.parquet, computed from
the actual usage, dependency and gap tables, separately for every model and
dependency_context. This includes losing G012 comparison arms and zero-weight
fusion experts. A context requires at least one verified dependency and one
study source, no unresolved dependency, no unverified study source, and no
blocking gap. Every context of every compared model must pass. An absent
model or an otherwise empty context fails. Unknown future gap reasons fail
closed. No input manifest accepts a completeness assertion.

usage_record_kind separates ENTITY_EXPOSURE from STUDY_SOURCE. Entity rows
must retain their typed native assay, source ID, model ID, lncRNA scope and
empty study tokens. They continue to exclude exposed lncRNAs and all their
pathway pairs, but are not themselves supposed to carry study citations.
Manual model-use declarations remain study sources. They cannot supply this
exemption through their raw input columns or specifications.

model_use_inventory_completeness_verified is the computed conjunction, not
a constant. Native binding loading recomputes it from verified tables and
full auditing reconstructs those tables from actual producers. External V2
publishes the public model_use_completeness table and carries the same value
through its binding, query response and current publication workflow. The
scope is NATIVE_REGISTERED_INPUTS_WITH_EXPLICIT_PROVENANCE_LIMITS.

The only nonblocking gap reason is the explicit GDC upstream alignment and
biospecimen-veracity limitation, appended after archived API metadata, raw
STAR bytes, count RDS values and sample-table identity have been replayed.
The GDC inspector now records sample exposure after that validation without
also adding the generic unknown-table-transformation gap. Generic tables
still retain that gap even if their hashes and study identifiers are present.
No other unresolved ancestry gap is waived. This distinction does not verify
upstream alignment, experimental truth or absence of unregistered studies.
raw_study_provenance_reconstructed and independent_external_validation_attested
remain false.

provenance_held_out additionally requires measured, mapped external outcomes,
verified external provenance, no connected-study or source-file overlap,
complete registered and external patient mapping, and no mapped patient
overlap. pair_held_out and lncrna_held_out add their existing entity exclusions.
Completeness alone never overrides these exclusions. Missing labels are never
converted to negatives, and descriptive metrics retain their separate scope.
The positive check exposed a previously masked patient-isolation bug: joining
annotated observations with native usage could promote booleans to object
dtype, for which bitwise negation of True is the truthy integer -2. Patient
flags are now normalized to boolean before exclusion. The shared-patient alias
case is verified with otherwise complete source coverage, so an unrelated
provenance failure cannot hide this regression.

The positive test boundary is an actual GDC INPUT SLICE: two synthetic raw
STAR files and archived API responses pass real R assembly and replay, then
typed entity exposure, explicit patient mappings and synthetic measured
outcomes enter the actual isolation/metric consumers. Scores are explicitly
declared fixtures, not model predictions. This proves the covered slice can
produce nonempty strict metrics while rejecting overlapping and incomplete
records. It does not claim a full core/fusion release is provenance-complete.

C13 remains open: core/graph authorities, some preparation and auxiliary/CNV/SC
inspectors still retain unresolved or unconditional ancestry gaps. Each must
be reconciled with verified actual transforms, reference/control roles and
patient coverage, without removing unknown lineage. A complete native
core/selected/fusion release must then pass the positive external V2 path;
the GDC-slice metric check is not a substitute for that integration.

The consumer regression pass also found a legacy formal-preparation crash:
output-manifest hashing used sha256_file, whose import existed only inside
the adjusted-CNV branch. Importing the same helper in the common path restores
the legacy writer without changing statistics or fitting behavior. The legacy
synthetic graph-authority fixture now supplies its consumed receipt field.
The ordinary G012 comparison fixture no longer incorrectly declares the
historical C-overlay mode; production overlay/native-mode guards are unchanged.

Verification: model_use_completeness_final_r1 records 116 passing checks on
149 against the final source version. These cover the new positive and
negative source-slice cases, actual native model-use and external releases,
patient identity, legacy preparation, native auxiliary and signed SC sources,
all G012 competitors, GDC assembly, publication/API and browser consumers.
Earlier failed receipts are retained, including the positive-coverage patient
bug and legacy preparation import failure. No optimizer, real cohort analysis,
paid compute or deployment was started. C01, C13 and C16 remain in progress.

Raw-count numerical replay through fold-local expression
-------------------------------------------------------

Native expression conversion now records normalization_release, the pinned
five-fold SUCCESS consumed by conversion. Detection scope likewise records
normalization_releases per cancer. Historical receipts remain readable but
cannot establish numeric replay without this authority. The ssGSEA receipt
already has an explicit normalization_success input and uses the same check.

normalization_source_replay requires host 149 and all five normalization folds
with matching cancer, raw count input and patient map. It verifies the current
producer/helper hashes, reruns materialize_fold_local_expression.R in a new
temporary directory, and compares all 25 native outputs. These include both
lncRNA and coding-gene logCPM matrices, the fitted TMM state, detection table,
and complete sample/normalization-factor table. The comparator also checks
reference sample, mean effective training library size, scientific parameters
and recorded R/edgeR versions. R object attributes and table order are retained;
numeric comparison uses all.equal tolerance 1e-12. This is a replay of the same
implementation, not an independent statistical implementation.

The temporary work records target_host=149 and normalization_fitting_replayed,
then is removed. Original counts, normalized outputs and receipts are not
rewritten. No neural model, optimizer or L1 baseline is fitted. Within one
inventory, the verified five-fold release is replayed once; subsequent views
still recheck the consumed receipt, input/output bytes and implementations.

TRANSFORMATION_REPLAYS.parquet records the model, dependency context, consumed
fold, release/normalization hashes, replay format, five folds and 25 outputs
compared. External V2 exposes it as model_transformation_replays; public queries
do not include raw patient IDs or paths in that table.

Study identifiers can now pass from inspected native upstream sources to a
derived expression/preparation record that has no study columns of its own.
Such records are marked VERIFIED_UPSTREAM_STUDY_METADATA. This uses only the
source records emitted during that specific upstream inspection, never all
other records already in the inventory. Missing-source and unknown-transform
gaps from any ancestor remain blocking. A recognized transform avoids the
generic unknown-transform gap only because its own inspector supplies the
specific missing or unsupported ancestry checks. Tables without a native
transform continue to retain the generic gap.

The source boundary now reaches GDC raw counts -> actual five-fold TMM ->
lncRNA/coding-gene conversion -> formal preparation exposure. It is still not
a provenance-complete core/selected/fusion release. At this checkpoint, ssGSEA
coverage and range scaling were previously replayed from stored raw scores,
not from the GSVA computation. This was explicitly blocking as
SSGSEA_RAW_SCORE_TRANSFORMATION_NOT_REPLAYED. Annotation/reference ancestry,
core/graph controls, CNV and other auxiliary lineage still need reconciliation.

Verification: normalization_source_replay_r1 (21 checks),
normalization_source_consumers_r1 (73 checks), and
normalization_source_public_r1 (97 checks) all pass on 149: 191 related checks
on this source version. These include real synthetic edgeR/GDC calculations,
native expression/detection/activity/CNV consumers, frozen-core input and
prediction consumers, current model-use/external publication, and actual web
API/browser queries. The positive source boundary and inherited metadata are
verified; repinned gene matrices, fitted states, library sizes and declared
scales fail numerical comparison. The original output files stay unchanged.
No intermediate failing receipt occurred in this change. C01/C13/C16 remain
in progress; this is not a complete native-core independent-validation claim.

Numerical replay from expression to native ssGSEA activity
--------------------------------------------------------

Native activity conversion now includes activity_release, the hash-bound
five-fold SUCCESS.json read by conversion. Validation checks its fold/cancer
mapping against the consumed ACTIVITY_NATIVE_SOURCES.json. Historical conversion
receipts without this field remain readable, but model-use inspection records
SSGSEA_REPLAY_RELEASE_NOT_BOUND and SSGSEA_RAW_SCORE_TRANSFORMATION_NOT_REPLAYED.
Neither the existence of raw scores nor a matching conversion hash is enough
to remove these gaps.

activity_source_replay verifies that all five native folds use the same
normalization release, annotation and GMT inputs. On host 149 only, it executes
the pinned materialize_fold_local_ssgsea.R and frozen_ssgsea.R producer/helper
in a temporary directory. It compares all 25 native outputs: raw GSVA scores,
scaled activity, fitted ssGSEA state, pathway coverage and gene detection.
The comparison also includes method parameters, training score range, training
detection threshold, cancer/fold identity and recorded R/GSVA versions. The
recomputed receipts must name the same actual input files as the originals.
The shared R comparator preserves object attributes, NA values and table order,
with all.equal tolerance 1e-12. No producer calculation or scientific threshold
was changed by adding this verifier.

The release is numerically replayed once per inventory, while every subsequent
consumer still rechecks its source files and code. Original files are rehashed
after computation and never rewritten. Temporary output is removed on success
or failure. The preflight records target_host=149, training_started=false and
ssgsea_training_scale_replayed=true. These checks fit the synthetic normalization
and ssGSEA scaling state only; they never construct a neural or L1 optimizer.

Successful replay removes only the missing ssGSEA numerical-transformation gap.
The existing TMM/raw-count inspector is still required, and annotation/reference
ancestry remains unresolved where it was unresolved before. Inherited study
identifiers do not waive those gaps. Full core, selected-model and fusion source
completeness, patient closure and a complete native positive external release
are still unproved; C01, C13 and C16 remain in progress.

The public transformation-replay table now includes activity_sha256. TMM rows
leave that field empty; ssGSEA rows identify both the consumed activity receipt
and its normalization receipt, alongside the native release hash, replay format,
model/context/fold, five folds and 25 compared outputs. It contains no patient
identifiers or file paths. This is consistency with the same pinned scientific
implementation, not independent biological validation.

Verification: activity_source_replay_r1 (15 checks),
activity_source_consumers_r1 (36 checks), and activity_source_public_r1
(97 checks) all pass on 149: 148 related checks on this source version.
These include actual synthetic raw-count/TMM/GSVA computations, original
independent ssGSEA oracle and held-out isolation checks, native five-fold
preprocessing, frozen-core/external integration and website/API queries.
The external consumer exposes nonempty ssGSEA proof rows for all five folds
without claiming complete model provenance. Self-consistent altered scores,
frozen states and software versions fail replay; unavailable pathways, legacy
gaps and source bytes are preserved. Reuse of a cached numerical proof rejects
subsequent changes to consumed file bytes. No intermediate failing receipt
occurred. Dependency deprecation warnings in Torch and Starlette are retained
in the logs. C01/C13/C16 remain in progress. No neural/L1 fitting, real cohort
analysis, paid compute or production deployment was executed.

Native control receipts and actual research-source boundaries
------------------------------------------------------------

Native model-use inspection now distinguishes validated control records from
empirical source tables. A frozen patient map and its receipt are located by
their bound hashes, then checked by the existing physical and semantic patient
authority validators. Merely carrying the embedded payload audit is insufficient
when files are unavailable. Missing files retain explicit unresolved-dependency
and PATIENT_CONTROL_FILES_NOT_VERIFIED gaps. Relocation works through the existing
hash locators; changed bytes or incompatible authority contents fail validation.
The control files create dependencies, not biological study rows or evidence
that every patient in the split map was fitted. Controls alone cannot establish
complete study coverage. L1 inspection retains its original preparation-source
boundary and does not inherit reference-neural control dependencies.

The existing graph receipt metadata check is shared between preparation and
model-use inspection. Known G012 receipts must agree on frozen patient authority,
generation, policy gates, six static input declarations, five fold declarations,
the consumed fold's input hashes and training identity hashes. Fold-local
detection retains its existing independently replayed replacement binding.
Unrecognized control formats remain explicitly incomplete. The static and
fold-local scientific tables still pass their own source inspectors; a valid
graph control receipt does not turn those tables into verified raw studies.

A current-C expansion receipt is a canonical hash of its embedded audit and
comparison, not a path to a missing file. The actual expansion loader now checks
that digest and checks audit fields against the loaded base/expanded graphs:
node identities, base/master edge identities, appended node count, selected-arm
edge count, runtime chunks, preservation flags and dropout policy. Rehashing a
false node/edge-count audit therefore does not make it valid. The model-use route
consumes the check performed by that loader and binds the actual sidecar file.
Its verified logical receipt and comparison hashes are recognized as resolved
input authorities; the base graph and experimental/predicted source dependencies
are still inspected separately. No second large sidecar load is needed.

The new positive patient fixture runs the real patient-authority builder and
both semantic validators on twenty synthetic patients. Only the frozen cohort
identity/count constants are replaced for that fixture. Existing native workflow
fixtures retain their separately recorded frozen-control and fixed-L1 substitutes;
they are not evidence of real training. Complete annotation/reference ancestry,
feature/label transformation replay, CNV/auxiliary/SC ancestry and cross-module
patient closure remain open. C01, C13 and C16 are not marked complete.

Verification: model_control_sources_r3 (50 checks), model_control_native_r1
(60 checks), and model_control_public_r2 (89 checks) pass on 149: 199 related
checks against the final code. These cover actual patient builders/validators,
five-fold graph preparation, expansion loading, native completed-result
registration/selection/frozen prediction/publication, study/patient isolation,
and actual API/browser consumers. The first two source-suite receipts are
retained as failures: adding control hashes had incorrectly implied L1 training
source coverage, then inspecting reference-neural controls added dependencies
outside the original L1 boundary. Production code was corrected; the original
L1 regression assertions were preserved. The earlier public_r1 pass belongs
to the superseded source version and is not counted as final validation.
Only dependency deprecation warnings remain. No neural/L1 fitting, real cohort
analysis, paid compute or deployment was started.


Partitioned genomic results: training, reassembly and fusion handoff
------------------------------------------------------------------

The resource-bounded Mutation/CNV trainer has a different completion contract
from the older in-memory trainer. Its TRAINING_SUCCESS.json deliberately says
TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED. The resource stage SUCCESS.json
and the separate reassembly SUCCESS.json are not substitutes for that receipt.
Previously the native auxiliary adapter accepted only the older producer, so
completed partitioned runs could not enter the current fusion preparation path.

The partitioned trainer now seals its input audit, split policy, checkpoint
manifest and resource audit as hash-bound completion_records inside LINEAGE.json.
The top-level training receipt binds that lineage and the checkpoint manifest.
This changes serialization only; the optimizer, sampling and model objectives
are unchanged. Existing partitioned outputs without these recorded bindings
remain unsupported; do not retrospectively manufacture the missing records.

For a future completed run on 149, the handoff order is:

1. Run the existing partitioned trainer only under separate fitting authorization.
   Its completion receipt points to the sealed PREDICTIONS_READY.json.
2. Run scripts/manage_v32_genomic_resource_memmaps.py reassemble with that exact
   --predictions-ready, the same --training-run-id, and a new --output-root.
3. Supply a NATIVE_AUXILIARY_FUSION_INPUT_V1 manifest to
   scripts/materialize_v32_native_auxiliary_fusion.py. Keep producer_kind=genomic
   and modalities=[mutation, cnv]; bind the current core_manifest,
   candidate_authority and patient_folds as before. native_receipt must point
   to TRAINING_SUCCESS.json. Add reassembly_receipt with the path and SHA-256
   of the reassembly SUCCESS.json. Every record contains actual path/sha256
   values; neither a prospective filename nor a resource-stage receipt is enough.
4. Use the adapter's FUSION_EXPERTS.json entries with the selected core's native
   FUSION_PRIMARY_INPUTS.json to assemble CURRENT_MULTIMODAL_FUSION_MANIFEST_V1.
   The current fusion CLI without --fit writes PREPARED_NOT_TRAINED. Actual
   fusion fitting remains a separate explicit --fit invocation and still
   requires replayed native validation labels covering all five pair folds.

The new bridge recomputes patient split identities with the trainer's original
normalizer, inspects all ten frozen private checkpoints, checks their parent
core and split policy, verifies the actually consumed frozen export hashes,
and validates the stage/candidate/seal identities. It runs the original bounded
memmap reassembler in a temporary directory on 149 and compares all five fold
tables and their aggregate in bounded batches. Repinning consistently edited
fold predictions and their average does not bypass that numerical replay.
Original training status is preserved in the auxiliary audit. No training
receipt is rewritten as a new fit or an independently validated release.

Model-use retains the inspected partition inputs, memmaps, controls and frozen
checkpoints even when fusion assigns the experts zero weight. The additional
implementation hashes participate in model-use validation. This does not prove
auxiliary model forwards, original biological source correctness or complete
cross-module patient ancestry; existing blocking gaps remain in force.

The synthetic checks execute the actual stage/write/seal/reassembly serializers,
adapter, fusion preparation and model-use consumer. Model states/history and
the compact-CNV aggregate gate are explicit serialization fixtures, not evidence
that a trainer or raw CNV analysis ran. The first bridge check identified a real
normalization mismatch: the general adapter canonicalizes patient IDs, whereas
the trainer preserves their spelling. Replay now uses the actual producer
normalizer on its bound patient file. No test expectations were relaxed.

Final verification on 149: partitioned_genomic_bridge_r3 (15 checks),
partitioned_genomic_consumers_r2 (56 checks) and
partitioned_genomic_model_use_r2 (35 checks), for 106 related checks on the final
source. The bridge covers absolute and manifest-relative input records. Earlier
passed receipts predate the relative-path correction and are retained only as
development history. The failing r1 receipt is also retained. Remaining warnings
are Torch deprecations. No optimizer, real cohort run or paid compute started.

Native candidate scope through the directional CNV head
------------------------------------------------------

The directional CNV private-head trainer and independent auditor previously
required 33 cancers, 100,000 candidates per cancer and historical run IDs. The
upstream native association workflow already produced annotation-defined
candidate sets and explicit new run IDs, so its completed outputs were rejected
before head fitting. This mismatch is now removed for native cohort receipts.
Inputs without a native declaration retain the legacy fixed coverage contract.

cnv_head_association.py verifies the completed local-CNV SUCCESS, independent
audit, decision, lineage and cancer worker receipts. The candidate definition
is replayed from its original annotation/GMT; the candidate and patient inputs
must match that authority. Each of the five label partitions is bound through
the completed worker receipt. The head uses that verified run ID, cancer order
and candidate counts. Removing the native cohort, repinning a different worker
run or replacing labels does not silently select the legacy route.

prepare_directional_inputs and collect_directional_examples are the actual
functions used by the trainer. They can also run without fitting: they load
directional matrices, frozen core representations and fold-local train and
validation labels, apply the original availability masks, and perform the
original per-cancer sampling and class balancing. Cancer quotas use the declared
cancer count; legacy cancer ordering and RNG offsets remain unchanged. Repeated
canonical patient IDs, nonintegral fold IDs and missing per-cancer folds are
rejected instead of silently collapsing conflicting assignments. Unavailable
association labels remain NaN and do not become negative training examples.

The existing preflight_v32_directional_cnv_head.py now calls those same functions
for all five folds. Supply --candidates, --patient-folds, --directional-root,
--association-root, --core-manifest and a new --output. Its --seed,
--min-pair-callable, --max-train-rows and --max-validation-rows options mirror
the trainer's sampling controls and are recorded in sampling_config. A
CHECKING_INPUTS receipt names host 149 before work; completion writes PASS_READY
with actual sample/class counts, while failure writes FAILED. This command
does not construct or optimize a private head. Missing association completion
no longer produces a misleading static-ready status.

Future fitting still uses run_v32_directional_cnv_head_oof.py as a separate
training command. Native OOF_MANIFEST/LINEAGE/PACKAGE_READY records carry the
cohort, association run and actual partition count. Native availability is
written to cnv_availability.tsv; the historical route retains its 33c filename.
The native auxiliary consumer verifies those declarations against consumed
association controls rather than accepting a caller-supplied smaller scope.

The independent auditor now accepts the same declared cohort. For native input,
--expected-training-run-id is mandatory; it is not inferred from the observed
head output. The auditor verifies PACKAGE_READY's manifest hash, requires all
declared cancer/fold partitions, and uses their actual sizes in coverage and
completion receipts. OOF targets still come from validation labels of
(prediction_fold-1) mod 5, after checkpoint selection. Unknown or missing label
classes now fail instead of silently becoming negative observations. Neither
this evaluation nor the CNV head changes primary-HHGT rankings or claims a
causal CNV effect or incremental fusion gain.

Synthetic verification executes the actual two-cancer/five-fold local-CNV
workers and independent closeout with a new run ID, then reaches all five CNV
input/sample collectors, the preparation CLI, consumed-source validator, native
auxiliary adapter and independent head audit. It uses six annotation-defined
candidates and 240 synthetic patients. Checkpoint tensors and optimizer telemetry
in the audit check are explicitly marked serialization fixtures; no head fitting
or auxiliary prediction forward is invoked. Metrics are recomputed on those
synthetic predictions only. Raw-study ancestry and complete cross-module patient
closure remain separate open requirements. No real cohort was recomputed.

Final verification on 149: native_cnv_head_scope_r4 (41 checks),
native_cnv_head_consumers_r2 (72 checks) and native_cnv_head_upstream_r2
(54 checks), for 167 related checks. The earlier upstream_r1 failure is retained:
a source-only E1 binding lacks a full consumed patient/graph contract, and the
previous control inspector raised KeyError before replaying its available
sources. It now records GRAPH_CONTROL_CONSUMED_BINDING_INCOMPLETE, which remains
blocking under the existing completeness policy, while continuing the actual
E1/source checks. Full native graph controls still undergo strict validation.
The original edge-replay assertions remain, with an additional explicit gap
assertion. Earlier passing development receipts are superseded. Only Torch
deprecation warnings remain; no training or real-data execution occurred.

Native candidate scope through ATAC materialization and consumers
---------------------------------------------------------------

ATAC materialization, trainer preparation and independent result audit previously
required the historical 33-cancer / 100,000-candidates-per-cancer universe. The
new native path accepts the annotation-defined candidate receipt already emitted
by the expression/scope workflow. The original seven-file ATAC input contract
adds a candidate_definition record containing that receipt's path and SHA256.
The receipt is replayed from annotation/GMT; observed predictions never define
the allowed cancer set or candidate counts. An omitted native declaration retains
the historical fixed-scope contract. Duplicate input-contract roles are rejected.

atac_cohort.py binds the candidate definition, exact candidate and patient files,
declared cancer order, per-cancer candidate counts, and raw-covered/raw-gap sets.
The latter are the declared cancers intersected with the unchanged official
23-covered / 10-gap classification. The R producer still requires the complete
official mapping and 562,709-peak matrix. Its optional ninth argument is a bound
cohort JSON; only the cohort's covered cancers are exported. Technical replicates
are averaged within exact aliquots, followed by equal-weight aliquot means per
patient. Missing raw cancers remain unavailable, never a zero accessibility value.
Native training rechecks the seven raw/static input hashes, original input
contract, materialization manifest and R audit, and rejects different candidate,
patient or membership inputs. Materialization and training rehash inputs before
publishing final SUCCESS. Neither check reconstructs biological source truth or
replays the full original ATAC normalization; those ancestry limits remain open.

prepare_atac_inputs and collect_atac_examples are now shared by the real trainer
and preflight_v32_atac_training.py. They read the actual materialized matrices,
project exact pathway genes, compute inner-training-only standardized features
and adjacent-validation persistence targets, then apply the original deterministic
sampling. The declared covered-cancer count sets the quota. Fractional/null/outside
fold IDs and null/empty candidate IDs fail. A raw-covered cancer unable to produce
any eligible outer-fold prediction also fails before fitting.

The no-fit preflight accepts --candidates, --patient-folds, --pathway-membership,
--materialization-success and a new --output. Sampling options are --seed,
--min-pathway-promoter-genes, --min-inner-train-patients,
--min-outer-train-patients and --max-training-rows. PASS_READY requires all five
actual feature/target collections, at least 100 sampled rows and both persistence
classes per fold, matching the calibrator's existing fitting requirements. It
records host 149 and training_started=false before work and writes FAILED on
errors. It does not construct a calibrator, optimize or run auxiliary predictions.

run_v32_atac_materialization.py and run_v32_atac_training.py preserve the frozen
patient-authority gate for legacy inputs. Native inputs instead validate the
candidate receipt's exact patient-file binding; --patient-fold-authority-receipt
is optional only on that verified native route. Actual fitting remains a separate
explicit training command. audit_v32_atac_readiness.py accepts
--candidate-definition and reports the declared denominator. Its historical
spreadsheet checks remain file inventory checks, not workbook-content validation.

Completed native checkpoints, fold-success files, LINEAGE and SUCCESS carry the
same cohort. The independent result auditor additionally requires an explicit
--expected-training-run-id for native output. It checks candidate keys, fold IDs,
typed availability, checkpoint/fold-success hashes and all five fold tables;
published probabilities and counts must equal their independently recomputed
fold aggregate. It processes one cancer at a time. The native auxiliary consumer
verifies the cohort against the materialization actually consumed and checks
checkpoint/fold-success cohort identity. Existing native legacy fixtures remain
supported. These checks do not establish incremental fusion gain or causal effects.

Synthetic verification runs the real R extraction with the unchanged full raw
shape, two annotation-declared cancers (BRCA and raw-gap OV), 240 candidate rows
and 40 declared synthetic patients. It checks the exact replicate/aliquot mean,
all five 120-by-7 feature collections, the readiness and no-fit preflight CLIs,
independent auditing of 1,200 frozen fold rows, and native fusion consumption.
Frozen probabilities/checkpoint JSON are explicitly marked serialization fixtures;
they are not generated by a fitted head. Historical spreadsheet/source placeholders
exercise only the readiness inventory. Changed raw files, duplicate contract roles,
removed scope, wrong input paths, changed keys/fold IDs and inconsistent aggregates
are rejected. The first development run exposed a test-contract omission (missing
byte sizes); the production hash/size gate correctly rejected it and was retained.
No real cohort analysis, model fitting, auxiliary prediction, paid compute or
deployment was performed. C01, C13 and C16 remain in progress.

Final verification on 149: native_atac_scope_r3 (24 checks) and
native_atac_consumers_r1 (72 checks), for 96 related checks on the final source.
The first failed fixture receipt and superseded r2 pass remain recorded as
development history. Remaining warnings are two Torch deprecations.

Native prepared feature, label and baseline reconstruction
---------------------------------------------------------

Formal preparation previously recorded the input association/detection files
without preserving enough information to independently reconstruct all tensors
consumed by the core. The model-use inventory therefore always retained
RAW_FEATURE_LABEL_STUDY_LINEAGE_NOT_RECONSTRUCTED, even when upstream file and
normalization checks succeeded. New annotation-defined, local-CNV-adjusted
preparation now writes a per-fold construction record and uses the same source
join/label attachment implementation as the independent reconstruction path.

prepared_construction.py records the 4096-column hashed training design,
candidate-key digest, labels, availability-weighted fitting weights and fixed-C
L1 configuration. The fitted binary coefficient vector and intercept live in
SOURCE_READS/PATIENT_FOLD_<fold>_CONSTRUCTION.json. Prepared tensor files contain
only that file's path/SHA256 reference, bound through their input authority;
they do not embed fitted parameters or optimizer state. The original L1 fitting
method, class weights, feature allow-list, seeds and scientific labels are
unchanged. Historical preparation remains supported with its explicit limits.

Reconstruction resolves the actual consumed candidate, training-only detection,
pathway hierarchy and train/validation association files. It replays native
candidate/detection definitions, rebuilds the original joins and hashed designs,
and verifies the recorded training labels and weights. It then recomputes the
frozen L1 linear score and reconstructs every batch tensor, including candidate
indices, context, proxy labels, weak-positive and availability masks, direction
labels/masks and graph availability. Comparisons follow the actual batch sizes.
Validation effects supply validation direction targets; they do not replace
training-discovery features. Baseline-only inspection reconstructs only training
batches and never reads validation association files as baseline dependencies.

The current model-use producer runs this reconstruction for selected native
core inputs, G012 comparison inputs and native core exports used downstream.
Only a successful native reconstruction removes the construction-specific gap;
all independent normalization, association, graph, study and patient gaps remain.
Transformation records report the inspected fold and tensor count. Their frozen
linear score calculation is explicitly reported as model_forwards_executed=true,
while neural_model_forwards_executed and training_started remain false. Historical
inventories with no such linear reconstruction continue reporting false. This
inspection does not prove the optimizer that originally produced the coefficient
values, rerun parameter fitting, or certify biological source independence.

The standalone audit_prepared_construction.py accepts --prepared,
--expected-sha256 and a new --output; --baseline-only narrows inspection to the
actual L1 training dependencies. It requires host 149, records CHECKING_INPUTS
before loading, writes FAILED on mismatch, and refuses a missing native
construction authority rather than emitting a misleading PASS. Its PASS still
reports raw_study_ancestry_complete=false. Missing or altered native references,
fit policy, code identity, labels, weights, keys and tensors fail validation.

Synthetic checks use the actual native CNV association/core preparation route,
with explicitly frozen zero and nonzero L1 coefficients instead of fitting. All fifteen
G0/G1/G2 prepared files are reconstructed and label/direction masks are also
checked directly against the original statistical tables. A separate actual
producer run with nonzero coefficients is checked against an independent scalar
summation oracle for all three graph variants. Existing reduced
cohort controls remain declared fixtures. No real cohort analysis or optimizer
ran. Complete original-study/reference ancestry and complete external evaluation
remain C13; unified auxiliary command handoffs and final acceptance remain C16.

Final verification on 149 covers 195 distinct related checks: 21 construction
checks, 14 historical preparation-source checks, 106 upstream checks and 54
downstream model-use/publication checks. The earlier 20 construction checks are
superseded by the final 21-case suite and are not counted twice. The complete
native publication fixture reaches PASS with the construction gap removed and
other original-study, upstream-transformation and patient-use gaps retained.
Warnings are Torch deprecations and a Starlette deprecation.

Native local-CNV numerical source reconstruction
-----------------------------------------------

Local-CNV source inspection previously verified consumed bytes and the E1 graph
selection step but did not reconstruct the upstream association statistics.
New worker receipts declare NATIVE_LOCAL_CNV_NUMERIC_REPLAY_V1 and bind the
actual producer, statistical routines, matrix readers and replay implementation.
Historical seven/eight-file implementation receipts remain readable and retain
LOCAL_CNV_NUMERIC_REPLAY_NOT_BOUND. Removing the replay declaration from a new
receipt is rejected by its implementation contract.

local_cnv_source_replay.py reconstructs each consumed cancer/fold partition
from the captured expression, activity, covariate, patient-map and compact CNV
files. It calls the original association_fold, g0_fold and validation calculation;
the validation partition assembly is now a shared producer/replay function.
Association and graph functions accept an explicit run ID for reconstruction,
preserving the original default. No scientific threshold, missing-value rule,
patient split, FDR correction, covariate design or graph selection is changed.

All output columns and row order are compared. Integer/categorical decisions
and null positions remain exact; floating statistics use relative tolerance
1e-10 and absolute tolerance 1e-12 with the pinned current implementation.
The audit also reconstructs partition counts, covariate/design metadata,
validation patient hashes and empty/insufficient graph statuses. Actual source
paths must agree with the source configuration. Validation reconstruction checks
its completed association parent. Cached numerical proof is keyed by source,
producer and output hashes; every use rechecks the bound bytes and code.

Model-use inspection runs this reconstruction for consumed training pairs,
validation labels and graph partitions, including G012 comparison contexts.
The transformation table records the actual fold and output kind. Graph replay
does not load pathway activity or candidate tables, and pair/validation replay
does not load gene-expression matrices. Raw file hashing remains conservative
over the source receipt. Numerical inspection constructs no neural model or
learned classifier. Downstream synthetic workflow fixtures execute frozen core
prediction separately, without parameter fitting.
Statistical nuisance residualization is the original preprocessing calculation.

The standalone command is audit_local_cnv_source_replay.py --receipt <worker>
--receipt-sha256 <sha256> --partition <parquet> --output <new-json>. It requires
host 149, writes CHECKING_INPUTS before reading, PASS only after numerical
comparison, and FAILED on mismatch or missing replay authority. It never
rewrites the original partition. Its PASS is explicitly not complete raw-study
ancestry. The earlier segment acquisition/materialization and per-pair/cross-module
patient ancestry gap remains blocking; no source gap is waived by this change.

Synthetic checks use the actual two-cancer workers and independent closeout
across correction-required, small-impact and review decisions. They reconstruct
70 original pair/graph/validation partitions over five folds, including unavailable
effects and absent E1 edges. Compact CNV inputs are declared synthetic arrays;
this check does not establish their earlier segment materialization. An independent
rank-projection and Student-t oracle checks the existing one-sided A1 effect,
patient count and degrees of freedom. Altered values, labels, null availability,
patient hashes, schema, metadata, implementation identity and omitted replay
declarations are rejected even after repinning output/source/producer hashes.

The first development run had two test errors: its independent oracle used
symmetric rather than the original one-sided correction, and its graph tamper
case assumed a finite E1 edge. Only the tests were corrected; the scientific
estimand and production thresholds stayed unchanged. The failed receipt remains
recorded. Final cache/dependency hardening additionally rechecks validation
closeout before and after calculation, including cache hits, and binds SUCCESS,
the independent audit and decision files into the downstream dependency table.

Validation on 149 covers 198 distinct related checks: 72 upstream regression
checks, 70 downstream checks and 56 direct/source/consumer checks. The final
cache/dependency change is verified by the 56-check suite and a repeated actual
native publication workflow (9 checks already included in the 70, not counted
again). Unchanged upstream/legacy regression results precede that last hardening.
The native workflow reaches PASS through source_recheck with numerical replay
for every fold and all G012 comparison contexts; the remaining ancestry gaps
continue to prevent a completeness claim. Only Torch/Starlette deprecations remain.

Native signed CNV materialization from declared segment files (2026-10-19)
-------------------------------------------------------------------------

The existing ``scripts.materialize_v32_directional_cnv`` now accepts a native
manifest as an alternative to the historical old-matrix/staging route. The
native route derives its cancer, patient and candidate scope from the actual
annotation-defined cohort. It needs neither an old compact matrix nor a fixed
33-cancer universe. Both routes require host 149 and record a preflight before
materialization. No model is constructed, fitted or predicted by this producer.

The JSON manifest has ``format: NATIVE_DIRECTIONAL_CNV_INPUT_V1`` and these
required fields: ``candidate_definition``, ``candidates``, ``patient_map``,
``intervals``, ``membership`` and ``segments``. The first five are explicit
``{path, sha256}`` file bindings. Candidate and patient contracts are those of
the native local-CNV workflow; interval and membership tables use the existing
streaming materializer schema. Interval columns are ``entity_type`` (``lncrna``
or ``gene``), ``entity_id``, ``chromosome``, ``start`` and ``end``. Membership
columns are ``pathway_id`` and ``gene_id``. Coordinates must be finite integers,
nonnegative and ordered; this check occurs before integer conversion.

Every declared patient requires exactly one segment record. An available row
contains ``cancer_id``, canonical ``patient_id``, ``status: available`` and
``file: {path, sha256}``. Optional ``study_metadata`` is an explicit metadata
object; filenames are not used to invent study identifiers. An unavailable row
has ``status: typed_unavailable`` and exactly
``unavailable_reason: NO_SELECTED_SEGMENT_FILE`` instead of a file. Missing,
duplicate and out-of-cohort patient records are rejected. The same path cannot
represent two patients. A segment file is tab-delimited with case-insensitive
``Chromosome``, ``Start``, ``End`` and ``Segment_Mean`` columns. If a Sample,
sample_id, patient_id or case_submitter_id column exists, every value must
canonicalize to the declared patient. Overlapping segments within a chromosome
are rejected. Chromosome prefixes and interval mapping follow the existing
mapping implementation; the manifest does not certify genome-build compatibility.

Optional ``event_threshold`` defaults to 0.30 in the input segment-mean units;
it must be positive and finite. ``max_segment_rows`` defaults to 2,000,000 and
is a positive integer. Each file read is bounded to this limit plus one row;
exceeding the limit fails rather than truncating. Patients are processed one
at a time into memory-mapped output arrays. Shared annotation/membership tables
are loaded in memory; this is not a universal memory-budget guarantee.

Run on 149, using new output paths::

  python -m scripts.materialize_v32_directional_cnv --phase all \
    --native-manifest /absolute/path/native_directional.json \
    --output-root /absolute/path/new_directional_release

For partitioned execution, use ``--phase cancer --cancer BRCA`` and then
``--phase finalize`` with the same manifest/root. The all-phase command requires
new partitions; it does not overwrite or silently resume completed partitions.
Native and legacy CLI inputs cannot be mixed. Set the native event threshold
in the manifest, not via the legacy CLI override.

Each ``cancer=<id>`` contains three ordered identity JSON files, SOURCE_FILES.json
and twelve arrays. lncRNA and pathway each have signed_mean (float32), callable,
amplification, deletion and neutral (uint8 flags); pathway additionally has
amplification_fraction and deletion_fraction (float32). Rows are sorted patient
IDs, columns are sorted corresponding candidate entity IDs. Signed means retain
the input segment units, while fractions are dimensionless. Mapping uses the
actual existing ``map_segments_directional``, ``aggregate_pathway_directional``
and streaming ``_mapped_entities`` functions. In particular, the canonical
pathway callability rule is an observed event OR all members observed. Partial
neutral coverage stays unavailable. Opposite amplified/deleted intervals may
cancel in signed mean while both direction flags remain true. Missing patients
have NaN continuous values and zero flags; they are not neutral observations.

Partition success binds the original manifest, cohort, seven implementation
files, all sixteen output files, identities, threshold and counts. Construction
uses a temporary directory and immutable publication; failures retain diagnostic
records without publishing a completed partition. Aggregate success checks all
declared physical cancer directories, child hashes, input identity and counts.

Independent reconstruction is callable on a partition or aggregate::

  python -m scripts.audit_native_directional_cnv \
    --success /absolute/path/new_directional_release/SUCCESS.json \
    --expected-sha256 <actual-receipt-sha256> \
    --output /absolute/path/new_audit.json

The audit records host149 and PASS/FAILED, re-reads the bound raw segments and
compares every output array row exactly, including NaNs. Changing an array and
recomputing its receipt hash cannot bypass reconstruction. Input, code and output
hashes are checked again after replay. A cached source-inspection replay still
checks the current raw/reference files and all output bindings. This is
same-implementation reproducibility; selected scalar examples have a separate
direct arithmetic oracle, not an independent biological validation.

The existing native CNV workflow accepts this aggregate as directional_success,
replays it before association and rechecks its files on completion/validation.
Actual local-CNV and CNV-head source consumers recognize native partition
contracts and run reconstruction. Verified head inputs narrow the old combined
raw-transformation gap to the remaining acquisition/patient-closure gap. They
also retain SEGMENT_ACQUISITION_AND_REFERENCE_ANCESTRY_UNVERIFIED. Explicit raw
patient mapping is recorded privately as declared file ownership, not proof of
per-pair fitted patient usage. No original acquisition, genome-build/reference
ancestry, or cross-module patient closure is inferred from a successful replay.

Direct verification uses BRCA/LUAD synthetic inputs with 240 declared patients,
238 segment files and two explicitly unavailable patients. It reconstructs all
24 cancer/array combinations, exercises cancellation/missingness, rejects
repinned numerical and contract corruption, checks cached-input changes and
immutable/failure behavior, then runs the actual native association/validation/E1
workflow and all five CNV-head input collectors. Core embeddings are frozen
serialization fixtures; auxiliary fitting and prediction are forbidden. The
direct and original mapping/streaming modules pass 43 tests. An old streaming
row-limit test now uses an explicit 1 GB test memory allowance so previously
loaded libraries do not trip its unrelated default memory gate; production
limits and the separate memory-gate tests are unchanged. No real cohort,
learned-model fitting, paid compute or production deployment was executed.

Related regression suites also pass on 149: 69 CNV/current-source/auxiliary
consumer checks, 65 local-CNV ancestry/reconstruction/workflow checks and nine
actual native result-publication checks. Together with the 43 direct/mapping
checks these are 186 distinct tests on the synchronized source revision. The
publication fixture uses frozen synthetic checkpoint containers and declared
synthetic optimizer telemetry; it does not execute an optimizer or establish
performance. Its unchanged biological ancestry gaps remain visible. Only the
existing Torch/Starlette deprecation warnings remain.

Native CNV/ATAC preparation and completed-result handoff (2026-10-26)
------------------------------------------------------------------

``scripts.run_native_auxiliary_workflow`` connects the existing native CNV and
ATAC input collectors, training CLI arguments, independent result auditors and
native fusion adapters. Its prepare/collect phases never call a trainer. This
entry starts from completed directional-CNV/association or ATAC accessibility
inputs; it does not acquire raw data, perform core selection, or orchestrate
the separate genomic, State, Clinical and single-cell producers.

The manifest uses ``format: NATIVE_AUXILIARY_WORKFLOW_INPUT_V1`` with
``producer_kind`` equal to ``cnv`` or ``atac``, an explicit ``native_run_id``,
``core_run_id``, and ``{path, sha256}`` bindings for ``candidate_authority`` and
``patient_folds``. CNV also requires ``directional_success``,
``association_success`` and ``core_manifest``. The two CNV success files must
be aggregate SUCCESS.json files; the completed association must declare its
native annotation-defined cohort. ATAC instead requires
``materialization_success`` and ``pathway_membership``. Its accessibility
success must carry the native candidate/patient/input contract. Existing
legacy producer interfaces remain available separately.

CNV consumes the selected core embeddings and verifies the core run. ATAC is
an independent model: its core_run_id selects the intended comparison/fusion
context and does not assert that core embeddings were used by ATAC fitting.
The adapter retains this distinction in ``core_dependency``.

Optional ``config`` contains overrides of the original GenomicTrainingConfig
or AtacTrainingConfig fields exposed by the corresponding trainer CLI. Unknown
fields, boolean numeric parameters, fractional integer parameters and nonfinite
values are rejected. CNV's unused cnv_event_threshold and ATAC's historical
expected_rows_per_cancer are not override fields here. Native coverage comes
from the actual candidate/cohort contracts. All remaining defaults are expanded
and saved before generating commands; sampling uses exactly the corresponding
subset of this same configuration.

Plan or prepare on 149::

  python -m scripts.run_native_auxiliary_workflow prepare \
    --manifest /absolute/path/auxiliary_input.json \
    --output /absolute/path/new_auxiliary_preparation --plan

  python -m scripts.run_native_auxiliary_workflow prepare \
    --manifest /absolute/path/auxiliary_input.json \
    --output /absolute/path/new_auxiliary_preparation

The plan is non-writing. Actual preparation writes LAUNCH_PREFLIGHT.json, calls
the original no-fit preflight and collects all five folds' features/labels.
It preserves the original CNV two-class sampling and the ATAC >=100-row/two-class
readiness gates. It validates the native cohort, source matrices, actual core
export and captured input bytes, then writes PREPARED.json, HEAD_PREFLIGHT.json
and TRAINING_COMMAND.json. The latter contains the exact trainer argv plus
``execution_mode: CODE_ONLY`` and ``automatically_executed: false``. The actual
trainer parser checks these arguments without invoking its execution function.
The workflow dispatcher accepts only preparation and independent-audit modules;
it rejects training dispatch. No training output directory is created by prepare.

The generated trainer command is a separate future action requiring explicit
execution. It points to ``<preparation>/training`` and uses the declared run ID,
input paths and expanded settings. This code-only change did not execute it.
The directional CNV producer now saves its actual GenomicTrainingConfig in
OOF_MANIFEST.json. ATAC already saves its settings under ``training`` in the
hash-bound RUN_CONFIG.json. A completed result lacking these required settings
cannot be retroactively claimed to match a workflow preparation. Original
standalone consumers of historical results retain their earlier behavior.

After a separately completed training run, collect its explicit receipt into
a new directory::

  python -m scripts.run_native_auxiliary_workflow collect \
    --preparation /absolute/path/new_auxiliary_preparation/PREPARED.json \
    --preparation-sha256 <actual-preparation-sha256> \
    --completed /absolute/path/new_auxiliary_preparation/training/OOF_MANIFEST.json \
    --completed-sha256 <actual-completed-receipt-sha256> \
    --output /absolute/path/new_auxiliary_collection

For ATAC, completed is ``training/SUCCESS.json``. The result must be at the
prepared output location; changing its run ID, input authority, core, consumed
association/materialization or training settings is rejected. Preparation/code
and all captured sources are rechecked. Source hashes alone do not attest that
optimization happened; the original result auditors retain their own checkpoint
and split checks, with the same limitations as those original implementations.

Collection executes the actual independent auditor, writes the resolved native
adapter manifest, calls the existing fusion adapter and replays its five-fold
aggregation. Directional CNV auditing now accepts an optional new output-root,
so this workflow writes audit products separately and leaves the completed
training package unchanged. The original default audit location is preserved.
Failures retain RUN_FAILED.json and stage logs without publishing a completed
workflow binding. Successful collection writes NATIVE_AUXILIARY_BINDING.json.

The final adapter manifest binds both workflow_preparation and workflow_audit.
The actual native fusion loader validates the preparation, completed parameters,
independent audit/success identities and their source receipt every time this
workflow's outputs are consumed. CNV global audit metrics are also hash-bound.
Thus later fusion preparation follows the audited result instead of relying
only on an earlier workflow status. ``fusion_experts`` points to the generated
FUSION_EXPERTS.json entries accepted by the existing current fusion manifest;
the final workflow validator checks their exact specification, run, estimand,
prediction and producer binding. This handoff does not fit the fusion model.

Synthetic verification uses actual two-cancer native CNV associations and actual
R-generated native ATAC accessibility inputs. Preparation/collection and the
independent auditors run through their production functions. Completed model
containers and optimizer telemetry are explicitly marked serialization fixtures;
no optimizer, CNV/ATAC forward or fusion fitting is permitted. The final fusion
preparation check uses explicit synthetic primary probabilities and legacy
hash-listed labels; it must retain unverified label authority and fitting
ineligibility. That check proves auxiliary handoff, not native primary-label
reconstruction or scientific gain. Earlier native-primary tests have their own
separate scope. Raw acquisition/reference ancestry and per-pair/cross-module
patient closure remain open; these workflow bindings keep source_ancestry_complete
and formal_increment_established false.

Final verification on 149: 63 direct/native CNV/native ATAC checks and 71
downstream source, auxiliary-fusion and partitioned-genomic regression checks
pass (134 distinct cases). The initial 55-check pass predates the final audit
dependency hardening and is superseded for those modules. The final direct run
completed remotely with exit 0 after its observing SSH connection closed; its
original XML/log/receipt were recovered after a successful hostname/process
probe. The test job was not restarted. No model fitting or real cohort run was
started during recovery. The broader C01/C13/C16 acceptance remains unfinished.

Variable cancer partitions for Mutation/CNV (2026-10-26)
------------------------------------------------------

The partitioned sampler already supported unequal cancer candidate counts, but
the resource stage, empty-stage check and prediction reassembler still required
a single rows-per-cancer value. The production disk-backed path now accepts an
exact count map in the existing resource budget, for example::

    "scope": {
      "formal_cancers": ["BRCA", "COAD", "LUAD"],
      "candidate_rows": 20,
      "candidate_rows_by_cancer": {"BRCA": 7, "COAD": 4, "LUAD": 9}
    }

Every declared cancer needs a positive integer count. Counts must sum to the
total and the largest partition must fit max_candidate_partition_rows. The old
candidate_rows_per_cancer integer remains valid for uniform scopes. If both
forms are supplied, they must agree. Do not set that integer to the largest
count for an unequal scope. New stage manifests record all counts and the
maximum partition size; their uniform field is null for an unequal scope.
Historical uniform manifests without the new optional fields remain readable.

Use the existing manage_v32_genomic_resource_memmaps.py stage, write, seal and
reassemble commands with this budget. There is no new training command. The
existing partitioned trainer's empty-stage inspection now derives its bounded
scan size from actual partition counts, as does the reassembler. Sampling,
patient splits, optimizer settings and probability aggregation are unchanged.

The stage scans the candidate cancer column in bounded batches and checks the
complete declared count map before creating its staging directory. Undeclared
cancers and omitted rows can no longer disappear through cancer filtering.
Partition key validation rejects null, blank and duplicate exact keys. Every
subsequent loader checks that partition intervals form contiguous, nonoverlapping
coverage of the declared budget, and that manifest totals, count maps and memmap
shape/dtype declarations agree. Repinning a modified manifest does not bypass
these semantic checks. These controls validate candidate storage identity;
they do not certify raw assay ancestry or optimizer execution.

Verification on 149: genomic_variable_scope_r2 passed 74 related tests without
an optimizer. The production CLI and trainer empty-stage inspection handle
7/4/9 and legacy 7/7/7 candidate partitions. Explicit synthetic fold values are
written and sealed; each emitted fold and aggregate is compared to an independent
array calculation, including true zero, partial fold coverage and unavailable
CNV/Mutation values. The existing actual native-auxiliary/fusion preparation
fixture additionally exercises 73/19 candidates, alongside the original 80/80
scope, absolute and relative paths, and zero-weight model-use traversal.
Frozen checkpoint containers and streaming-CNV gate metadata in these fixtures
are explicitly synthetic. No streaming-CNV raw preparation, neural forward,
model fitting or real cohort analysis was performed for this change.
The original 73-case pass is superseded by this run, which also verifies that
unused Pandas categorical cancer levels do not create phantom candidate rows.

This fixes a scope mismatch in the actual partition lifecycle. The existing
partition training CLI still requires its frozen patient-fold authority, and
the original streaming-CNV producer has separate input requirements. A complete
native genomic prepare/collect workflow must reconcile those upstream contracts;
this change does not claim that broader handoff is complete. C01/C13/C16 retain
their outstanding integration and evidence requirements.

Native genomic raw inputs and the no-fit trainer entry (2026-10-26)
-----------------------------------------------------------------

The historical streaming-CNV aggregate and its launcher required the frozen
33-cancer patient authority. An explicit native path now reuses the original
compact six-array producer for the annotation-defined cohort. The existing
partition training CLI accepts that cohort and can collect every train/validation
input without fitting. The legacy frozen-authority entry remains available and
retains its original validation.

The new genomic CNV input JSON has these fields, with every file represented by
an explicit path/SHA256 record::

    format: NATIVE_GENOMIC_CNV_INPUT_V1
    segment_manifest: <NATIVE_DIRECTIONAL_CNV_INPUT_V1 JSON record>
    sample_gene_mutation: <standardized mutation table record>
    sample_lncrna_mutation: <standardized mutation table record>
    mc3: <raw MC3 file record>
    core_manifest: <current frozen core export record>
    core_run_id: <the exact current core run identity>
    max_memory_bytes: <optional positive integer; default 536870912>

The reused segment manifest supplies candidate definition, candidates, patient
map, reference intervals, exact pathway membership, explicit available/unavailable
patient segment files, event threshold and row limit. Reuse concerns source and
patient identity: genomic CNV still uses event presence and absolute burden,
whereas directional local-CNV uses signed means. Opposite signed segments retain
genomic event/burden; they are not substituted with a zero signed mean.

On 149, from the repository directory::

    python -m scripts.materialize_native_genomic_cnv prepare \
      --manifest <native-genomic-input.json> --output <new-streaming-directory>

An optional --cancers list permits resumable subsets of the declared cohort.
The command returns 2/INCOMPLETE until every declared partition is present;
only then does it publish aggregate SUCCESS.json. Context changes prevent
reuse. Every raw file is checked against its explicit patient/file declaration.
Memory/row limits and typed unavailability use the original compact producer.
The output records host149/no-training preflight before materialization.

An explicit audit reruns that producer and compares all six arrays per cancer::

    python -m scripts.materialize_native_genomic_cnv audit \
      --success <streaming-directory/SUCCESS.json> \
      --expected-sha256 <sha256> --output <new-audit.json>

This numerical reconstruction is separate from biological source verification.
Acquisition and reference ancestry remain unverified where no evidence exists.

Use the existing resource staging command with this completed streaming SUCCESS
and the declared candidate budget. Then the actual training entry supports::

    python -m scripts.train_v32_partitioned_genomic_heads \
      --resource-staging-success <resource-stage/SUCCESS.json> \
      --patient-folds <patient-map.tsv> \
      --candidate-definition <CANDIDATE_SOURCES.json> \
      --candidate-definition-sha256 <sha256> \
      --expected-core-run-id <current-core-run-id> \
      --pathway-gene-membership <exact-membership.parquet> \
      --sample-gene-mutation <gene-mutation.parquet> \
      --sample-lncrna-mutation <lncrna-mutation.parquet> \
      --mc3 <raw-mc3-file> --core-embedding-manifest <core-export.json> \
      --cnv-streaming-success <streaming-directory/SUCCESS.json> \
      --output-root <new-preflight-directory> --training-run-id <v32-run-id> \
      --prepare-only

Use actual scientific sampling parameters when overriding defaults. The native
candidate definition and historical --patient-fold-authority-receipt options
are mutually exclusive. Native mode requires the current core identity and
candidate-definition hash; it validates the cohort against both the resource
stage and the streaming source. Its event threshold must match the materializer.
Patients in other cancers of the shared map do not enter this run's split trace.

The --prepare-only mode uses the same source loading and global partition
sampling as training. It checks two-class train and validation rows for both
modalities and all five folds, then writes HEAD_PREFLIGHT.json with counts,
split identities, source hashes, full settings and resource observations.
It creates no checkpoint, optimizer, prediction, write receipt or training
SUCCESS, and leaves resource memmaps unchanged. Failed preparation retains its
staging/preflight evidence and never publishes a PASS_READY output. Fitting is
still a separate execution and was not performed here.

The native completed-result adapter now checks the actual training cohort,
core run, CNV threshold and required mutation/membership/source bindings against
the native streaming context. It retains original source files in the consumer
inventory. Native streaming identity requires the native cohort declaration;
removing it cannot select legacy validation, even after repinning all split
policy and checkpoint records. Actual auxiliary model-use traversal numerically reconstructs native
streaming arrays, follows raw segment study declarations and records declared
file patients. Counts remain file exposure, not per-pair fitted patient usage;
SEGMENT_ACQUISITION_AND_REFERENCE_ANCESTRY_UNVERIFIED remains a blocking gap.

Direct verification uses 40 declared synthetic patients in two cancers plus
five patients in an unused cancer of the shared map. Thirty-eight synthetic
segment files produce six arrays per cancer. An independent parity oracle checks
event/burden and typed missingness; production replay checks all twelve arrays.
The actual training CLI collects ten two-class input groups with optimizers and
prediction calls forbidden. Completed-model checks use explicitly synthetic
checkpoint containers and fold probabilities, not a claimed training run.
Native raw acquisition, unified genomic prepare/collect orchestration and the
complete native-core/fusion/external positive-metric acceptance remain open.
Final verification: native_genomic_inputs_r4 passes 51 direct/partition checks;
native_genomic_consumers_r2 passes 120 related source, directional-CNV, fusion,
resource-scope, independent audit and patient-authority checks (171 distinct
cases). Earlier direct and consumer passes precede the final scope/control and
validation-downgrade checks and are superseded for those modules. No fitting or
real cohort run occurred.

Unified native genomic prepare/collect handoff (2026-10-26)
---------------------------------------------------------

run_native_genomic_workflow.py now connects the existing native raw CNV producer,
its numerical reconstruction, resource partitions, actual no-fit training input
collector, completed partition trainer records, bounded reassembly and native
fusion adapter. It never dispatches fitting. A new workflow input has::

    format: NATIVE_GENOMIC_WORKFLOW_INPUT_V1
    native_run_id: <v32-prefixed unique run ID>
    genomic_input: <path/SHA256 record for NATIVE_GENOMIC_CNV_INPUT_V1>
    resource_limits:
      adapter_budget:
        max_declared_adapter_peak_rss_bytes: <positive integer bytes>
        max_memmap_bytes: <positive integer bytes>
        max_staging_disk_bytes: <positive integer bytes>
      downstream_budget:
        max_declared_process_rss_bytes: <positive integer bytes>
        minimum_host_available_memory_bytes: <positive integer bytes>
        minimum_host_free_disk_bytes: <positive integer bytes>
    config: <optional numeric GenomicTrainingConfig overrides>

The raw genomic input supplies the current core run and exact source bindings.
The workflow derives cancer counts, maximum candidate partition size and the
ordered-key hash; users do not copy those fields between handoffs. It retains
the original training defaults except that the CNV event threshold comes from
the raw segment declaration. An explicit conflicting threshold is rejected.
Resource limits are required; no paid resource is created by either phase.

On 149, from the repository directory::

    python -m scripts.run_native_genomic_workflow prepare \
      --manifest <workflow.json> --output <new-preparation-directory> --plan

    python -m scripts.run_native_genomic_workflow prepare \
      --manifest <workflow.json> --output <new-preparation-directory>

--plan checks the declared inputs and actual CLI argument schema without creating
the output directory. prepare writes the raw streaming release and raw-array
audit, generates RESOURCE_BUDGET.json and the disk-backed resource stage, then
executes the actual partition trainer with --prepare-only. PREPARED.json pins
the checked inputs and source code. TRAINING_COMMAND.json contains the separate
future training command and explicitly says it was not executed. The workflow
dispatcher rejects that command because it lacks --prepare-only.

The resource stage's probability/availability/processed arrays are the trainer's
mutable destination. They are not included as immutable consumed preparation
sources. Candidate partitions, the resource contract, raw compact arrays,
reference/control files and core exports are immutable. Preparation remains a
historical valid input receipt after the prediction arrays are filled; collection
requires their actual completed trainer seal and checks their current hashes.
This distinction prevents ordinary completed training from invalidating its own
preparation receipt, while preserving the empty-stage gate on a fresh trainer.

After a separately executed training run has produced the declared output::

    python -m scripts.run_native_genomic_workflow collect \
      --preparation <preparation-directory/PREPARED.json> \
      --preparation-sha256 <sha256> \
      --completed <preparation-directory/training/TRAINING_SUCCESS.json> \
      --completed-sha256 <sha256> --output <new-collection-directory>

No training command was executed to verify this implementation. Collection
requires the exact prepared output location, run/core/cohort, full settings,
source files, resource stage and per-head train/validation row counts. It calls
the original sealed-memmap reassembler and the actual native five-fold
aggregation. The collection audit checks checkpoint split identities and
reassembly/aggregation contracts; it is not an independent predictive-performance
audit and does not replay neural forwards. The final native adapter rechecks
that audit during subsequent fusion consumption, including rehashed source/run
changes. Changed or missing audit bindings cannot be substituted silently.

NATIVE_GENOMIC_BINDING.json links PREPARED, the completed trainer receipt,
reassembly, COLLECTION_AUDIT.json, native adapter and FUSION_EXPERTS.json. Those
expert declarations are accepted by the actual current fusion preparer. There
are two modalities and five patient folds, so fold_rows is ten times candidate
rows; it is neither a count of patients nor ten distinct patient folds.
The workflow validator also reconciles the expert declarations themselves.

Synthetic verification executes raw CNV preparation, all ten train/validation
collections, the actual write/seal/reassembly functions and actual fusion
preparation. Completed checkpoints/probabilities remain explicit frozen
serialization substitutes. The final fusion test uses synthetic primary scores
and legacy hash-listed labels and remains ineligible for current fitting.
Unknown raw/reference ancestry, per-pair/cross-module patient usage and complete
external acceptance remain open. State/Clinical/single-cell command consolidation
and final requirement-level acceptance also remain outstanding.

Verification on 149: genomic_workflow_r1 passes 69 direct/native-genomic/
partition checks; genomic_workflow_consumers_r1 passes 76 fusion, model-use,
resource-scope and existing CNV/ATAC workflow checks (145 distinct cases).
No failed or skipped case, model fitting or real cohort execution occurred.

Native State producer compatibility and numerical replay (2026-10-26)
===================================================================

The current State context previously rebuilt each completed private head with
the module identity ``state``. The actual native trainer initializes
``state_fold_0`` through ``state_fold_4``. Their initialization receipts differ,
so actual producer results could fail despite passing manually authored
serialization fixtures. Training and loading now share
``state_training.build_state_private_head`` with the original native names,
seeds and architecture. The original training algorithm is unchanged.

The actual State outer driver now exposes its existing input validation,
within-patient alignment and fold-local statistics through
``prepare_state_inputs`` and ``state_fold_statistics``. Current State context
calls the same functions with the recorded standardized measurements,
expression, patient map and five current core exports. It verifies:

* Measured/missing program inventory, aliquot collapse and excluded-row counts.
* Original patient split hashes, train/validation candidate counts, class
  counts, fallback declaration, input dimensions and training pair normalizer.
* Every native OOF row through the recorded frozen private head on CPU, including
  pair counts, measured held-out effects, availability, reasons and identities.
* The existing independent pair-count-weighted five-fold aggregation, selected
  member profiles, State GeneSets and public downloads.

Only neural probabilities permit float32 device roundoff (absolute and relative
tolerance 1e-6). Missingness must match exactly; the existing table comparison
retains zero relative tolerance and 1e-12 absolute tolerance for the other
numeric fields. No optimizer or training entry point is called during replay.

Native training also accepts partitioned Parquet inputs and CSV patient maps.
The State consumer now follows those original formats instead of treating every
input as a single Parquet/TSV file. Directory checks use the producer's original
``path_sha256`` tree hash, including file membership and relative names. Each
member is retained separately in the private source inventory. Adding, deleting
or changing a member invalidates the original completed lineage.

Current State bindings and their recomputation audits now contain
``native_numeric_replay``, and both ``raw_measurement_statistics_replayed`` and
``private_head_predictions_replayed`` are true. Here "raw measurement" means
the standardized measurement table consumed by the native trainer; this does
not verify its original acquisition or reference ancestry. The replay explicitly
declares ``same_implementation_replay`` and keeps
``source_acquisition_ancestry_verified`` and
``independent_predictive_performance_verified`` false.

This supersedes the earlier aggregate-only State-context limitations above.
Existing aggregate-only context bindings must be regenerated from their actual
completed native inputs under the new implementation; no fitting is necessary
to regenerate the context. No old scientific artifacts were changed here.
The existing ``materialize_v32_state_gene_sets.py --current-manifest`` and
``CURRENT_ANALYSIS_WORKFLOW_INPUT_V1`` State stage remain the entry points.
State/Clinical/single-cell producer command consolidation, full cross-module
patient/source closure and final requirement acceptance remain outstanding.

The integration fixture now uses the actual native outer driver, original
association/forward code and original completion serializer. Only its fitting
callback is replaced by explicitly changed frozen parameters and synthetic
telemetry; optimizers are forbidden. It includes 40 patients, 41 aliquots,
seven programs, five folds and 35 OOF rows. Independent hand-constructed values
check positive/negative/unit correlations, a measured zero correlation, eight
held-out pairs and a wholly missing program. Tests reject changed raw values,
split/count/normalizer receipts, repinned frozen parameters and altered OOF
tables even when the aggregate and every declared hash are updated consistently.

Final verification on 149: state_numeric_replay_r2 passes 37 cases;
state_numeric_consumers_r2 passes 41 cases, including the existing complete
publication workflow, HTTP/browser/download consumers, Clinical/patient
contexts and five non-fitting State input/statistic checks. All 78 distinct
cases pass with no failures or skips. Earlier r1 runs also passed and are
superseded by r2 after native directory/CSV support was added. The first source
upload failed before execution; a read-only SSH reprobe returned hostname 149,
with no iOA login/reconnect and no duplicate job. No model fitting, real cohort
analysis, paid compute or deployment was performed.

Native State prepare/collect command handoff (2026-10-26)
=======================================================

The existing State CLI now has an actual ``--prepare-only`` mode. Both fitting
and preparation use ``prepare_state_head_inputs`` for the original training
and validation arrays, masks, normalization and validation fallback. A hash of
the array values, shapes and dtypes is included in the fold description. This
refactoring preserves the original features, labels, sampling and optimizer
behavior. Preparation constructs no State private head, invokes no optimizer
and computes no held-out test association statistic. It writes
``HEAD_PREFLIGHT.json`` with the five fold descriptions and actual input hashes.

The CLI continues to default to the original frozen patient authority. The
explicit ``--patient-authority-mode current`` route validates the original
patient-first authority receipt and requires ``--expected-core-run-id``. Both
the original authority validator and current-core export validator execute;
no fixed count/hash constants are patched. The actual patient authority record
is retained in completed State lineage and checked again by the current State
consumer. The CLI also exposes the previously API-only validation offset.
Numeric configuration rejects booleans, fractional integer fields and nonfinite
values. The path runner now requires host 149, writes a launch receipt before
input processing and refuses a nonempty output directory before execution.

The consolidated entry is ``scripts.run_native_state_workflow``. Its manifest
has format ``NATIVE_STATE_WORKFLOW_INPUT_V1`` and the following fields:

* ``state_run_id``: lowercase ``v32-*`` identifier; ``core_run_id``: the expected
  completed current core identity.
* Hash-bound ``state_measurements``, ``patient_folds``,
  ``patient_authority_receipt``, ``core_manifest``, ``lncrna_expression`` and
  ``selected_analysis`` inputs. Standardized measurement/expression directories
  use the original State dataset-tree hash. The patient authority uses its
  original TSV map and receipt; it is not replaced by inferred barcode IDs.
* Optional ``config`` fields from ``StateTrainingConfig`` and optional
  ``state_sets`` fields from ``StateSetConfig``. Defaults are expanded and bound,
  so the complete configuration is carried into preparation and collection.

All commands below run from the development checkout on server 149::

    python -m scripts.run_native_state_workflow prepare \
      --manifest /path/state_workflow.json --output /new/state_prepared --plan

    python -m scripts.run_native_state_workflow prepare \
      --manifest /path/state_workflow.json --output /new/state_prepared

The plan validates source/core/selected identities and the actual training CLI
arguments without creating output. Preparation invokes the original CLI with
``--prepare-only``, replays its numerical input collection and writes
``PREPARED.json`` plus ``TRAINING_COMMAND.json``. The latter is a complete,
parsed command for ``scripts.run_v32_state_training`` whose output is
``state_prepared/training``. The workflow cannot dispatch that command: its
dispatcher accepts the training module only when ``--prepare-only`` is present.
Fitting remains a separate operation outside this code-only goal.

After a separately completed native State run exists::

    python -m scripts.run_native_state_workflow collect \
      --preparation /new/state_prepared/PREPARED.json \
      --preparation-sha256 <sha256> \
      --completed /new/state_prepared/training/STATE_TRAINING_LINEAGE.json \
      --completed-sha256 <sha256> --output /new/state_collected

Collection revalidates the original preparation and every captured input/code
file. It requires identical run/configuration, patient authority, input records,
five core parents, split hashes, array fingerprints, fold counts and program
coverage. It then invokes the actual current State publisher, including raw
standardized-measurement and frozen-head numerical replay. The collected input
contains ``workflow_preparation``; the State auditor follows this dependency
and repeats preparation/completion matching. Training outputs are immutable
during collection. ``NATIVE_STATE_BINDING.json`` links the resulting
``state_context`` binding, which is accepted by the existing overall publication
workflow as the State component's ``REUSE`` binding.

This handoff validates the State authority itself, its consistency across
preparation/completion and its selected core identity. It does not prove that
all State source patients have the correct ancestry relative to every fitted
core graph or every other module. Original acquisition/reference ancestry,
complete cross-module patient closure and independent predictive performance
remain C13 work; all corresponding completion/performance flags stay false.
The current-mode authority option is explicit and does not relabel a legacy
frozen receipt or silently weaken the default legacy entry point.

The fixture uses the actual patient-authority builder for 40 patients and 41
aliquots, the actual State preparation/training CLI, original outer driver and
completion serializer. Only the private fitting callback is replaced by
explicitly frozen parameters and synthetic telemetry. The preparation test
forbids head construction, forward calls, optimizer construction and test-split
statistics. Collection runs the actual five-head forwards and existing complete
publication workflow with other producer stages explicitly absent in this
scoped fixture. It is not evidence of actual cohort training or biological
increment.

Verification on 149: state_workflow_r1 passes 60 workflow/current-State cases;
state_workflow_consumers_r1 passes 51 publication, Clinical/patient, explicit
patient-identity and non-fitting State input/statistic cases. All 111 distinct
cases pass with no failures or skips. No model fitting, real cohort analysis,
paid compute or deployment ran. State command handoff is implemented; Clinical
and single-cell handoffs, complete source/patient closure, public scientific
field definitions and final requirement acceptance remain outstanding.

Native Clinical raw numerical replay (2026-10-26)
------------------------------------------------

Clinical publication now reconstructs the quantities that the original native
producers actually consumed. Previously, patient risk replay began at a saved
feature table and entity replay began at saved fold statistics. Those checks
could not determine whether the intermediate values came from the declared
patient measurements and clinical workbook.

The risk producer's shared ``collect_fold_inputs`` now constructs the original
fold-local features, core/domain arrays, train/validation/test indices, endpoint
targets and train-only time bins. Preparation and fitting call this same helper.
Completion records retain the complete six-parameter configuration, producer
and implementation hashes, exact source roles, per-fold patient counts/columns,
patient-order digest and array digest including values, shapes and dtypes.
The input tree is hashed before and after execution. The actual fitting
algorithm, loss and endpoint definitions are unchanged.

The risk CLI has an explicit no-fit route. Run on server 149, with the original
paths and parameters declared for the intended run::

    python -m scripts.run_v32_clinical_training \
      --repo-root /path/to/checkout \
      --workbook /path/outcomes.xlsx \
      --fold-manifest /path/SAMPLE_PATIENT_FOLD_MAP.tsv \
      --patient-fold-authority-receipt /path/PATIENT_FOLD_AUTHORITY_RECEIPT.json \
      --patient-authority-mode current \
      --expression-root /path/expression --activity-root /path/activity \
      --core-root /path/current_core_export --expected-core-run-id <core_run_id> \
      --training-run-id <clinical_run_id> --output-root /new/clinical_prepared \
      --seed 20260726 --epochs 100 --patience 12 --n-bins 12 \
      --min-patients 40 --min-events 12 --prepare-only

``HEAD_PREFLIGHT.json`` records the five collected inputs and patient authority.
The directory includes private standardized outcomes/covariates and patient
features; it is not a public download. Preparation constructs no model, runs
no forward and calls no optimizer. It writes no checkpoint, risk prediction,
fold success or completed module lineage. Preparation is incompatible with
``--resume`` and cannot reuse an existing output directory. This is an actual
input-collection entry point; unified Clinical prepare/collect orchestration
and its separate completed-command handoff remain to be implemented.

Both Clinical producers now offer ``--patient-authority-mode current`` for the
existing generic authority validator. The default remains ``frozen`` and keeps
the original frozen-authority validator. Risk current mode additionally requires
an explicitly identified current core export. The entity CLI still supports
scoped subject types without requiring unused measurement files; current full
publication continues to require all lncRNA, exact-pathway and seven State
subjects. Its output directory must be new, its launch receipt names host 149,
and its source-role inventory is checked again after calculation.

``clinical_numeric_replay.py`` reads the actual Excel sheets, expression and
activity partitions, patient authority and core embeddings. It reconstructs
every saved risk feature and target/bin/split digest before the existing five
frozen-head prediction checks. Entity replay reconstructs the original patient
measurement matrices and invokes the original statistical implementation,
comparing every fold-statistic column and missing value before native
meta-analysis and selected-member support. The State lineage remains only the
original generation anchor; this does not establish complete State/core patient
ancestry. Risk and entity sources must share the same workbook and explicit
patient authority. Actual input sources and producer implementations are
included in the current Clinical publication dependency inventory.

Current Clinical bindings now require successful raw feature and entity
statistic replay. Historical outputs lacking the actual feature artifact,
complete original parameters or source-role contract cannot silently receive
these guarantees. Preserve those outputs and recover their original evidence
before creating a new verified binding; do not infer missing training settings
from current defaults. This change does not rewrite old results.

The integration fixture uses the actual patient-authority builder, real XLSX
reader, risk/entity CLIs, native completion serialization and current publisher:
40 synthetic patients, 41 aliquots, five risk folds, 240 typed patient-risk rows,
270 entity fold rows and 54 entity summaries. Only private-head fitting is
replaced by explicitly marked frozen parameters and synthetic training telemetry;
neural optimizers are forbidden. The current core parent is the existing frozen
synthetic fixture. DFS remains unavailable; neither a joint GeneSet hazard ratio
nor independent performance is inferred. Entity calculations are native
synthetic statistical checks, not neural-head optimization.

Verification on 149: ``clinical_numeric_replay_r1`` passes 34 Clinical cases;
``clinical_numeric_consumers_r1`` passes 24 actual publication/browser, patient
context and non-fitting Clinical calculation cases. All 58 distinct cases pass
without failures or skips. Checks reject rehashed expression/activity/workbook
changes, modified training features with unchanged test predictions, rehashed
State measurements and forged fold statistics even after recomputing their
summaries. Preparation is tested with model construction, forward and fitting
forbidden; changed inputs during collection cannot produce a ready receipt.

The initial ``clinical_native_smoke_r1`` failed because the verification
environment lacked the already-declared ``openpyxl`` dependency. Only the
isolated verification directory was supplemented with openpyxl 3.1.5 and
et_xmlfile 2.0.0; ``clinical_native_smoke_r2`` then passed. A transient SSH banner
error during later observation resolved on the authorized read-only probe,
which returned host 149. No login, test restart or duplicate job was needed.
These are same-implementation numerical checks; acquisition/reference ancestry,
complete cross-module patient closure, optimizer history and independent
predictive performance remain unproved. C01/C13/C16 stay in progress. No fitting,
real cohort analysis, paid compute or deployment ran.

Native Clinical preparation/completion workflow (2026-10-26)
----------------------------------------------------------

``scripts.run_native_clinical_workflow`` now connects both actual Clinical
producers to current selected publication. It separates risk-head fitting from
entity statistical calculation and records a distinct command for each. The
workflow itself dispatches only the original CLIs' ``--prepare-only`` modes.

The input JSON has format ``NATIVE_CLINICAL_WORKFLOW_INPUT_V1``. Required fields
are ``patient_risk_run_id``, ``entity_run_id``, ``core_run_id`` and the following
explicit path/SHA256 records: ``workbook``, ``patient_folds``,
``patient_authority_receipt``, ``expression_root``, ``activity_root``,
``state_measurements``, ``state_lineage``, ``core_manifest``,
``selected_analysis`` and ``candidates``. File hashes use SHA256; dataset
directories use the existing ``input_lineage.artifact_sha256`` relative-path
and file-content inventory. All actual files are retained in the dependency
inventory. The candidate record must be the selected analysis's actual
candidate authority, and all five selected/core parents must match.

Optional ``risk_config`` fields are ``seed``, ``epochs``, ``patience``,
``n_bins``, ``min_patients`` and ``min_events``. Optional ``entity_config`` fields
are ``block_size``, ``min_train_patients``, ``min_train_events``,
``min_validation_patients``, ``min_validation_events``, ``min_test_patients``,
``min_test_events``, ``min_train_measurements`` and ``min_folds_available``.
Unspecified fields take the actual original CLI defaults; the expanded complete
configuration is then recorded. Unknown fields, Boolean/fractional counts,
nonpositive counts and more than five available folds are rejected. Optional
``fdr_threshold`` defaults to 0.05 and must be finite in (0, 1]. The workflow
requires the full candidate/subject universe; it offers no smoke cap or cancer
subset. Legacy standalone producer options remain available.

Run these commands from the development checkout on server 149::

    python -m scripts.run_native_clinical_workflow prepare \
      --manifest /path/clinical_workflow.json --output /new/clinical_prepared --plan

    python -m scripts.run_native_clinical_workflow prepare \
      --manifest /path/clinical_workflow.json --output /new/clinical_prepared

The plan validates identities, patient authority, source hashes, configuration
and actual CLI argument parsing without creating output. Preparation calls the
original risk collector for five folds and the original entity collector for
every cancer/subject partition. Entity preparation reads aligned patient
measurements, endpoint and covariate tables, but does not calculate survival
statistics or instantiate a neural head. Its new ``INPUT_PREFLIGHT.json``
records ordered patient/entity digests, exact matrix/fold-array fingerprints,
fold counts, observed-measurement counts and standardized endpoint/covariate
digests. The full input files remain independently hash-bound. Actual entity
calculation and numerical replay use this same patient-matrix collector.

``PREPARED.json`` binds the launch plan, both numerical preflights and
``EXECUTION_COMMANDS.json``. The latter contains parsed original CLI commands
for ``risk`` and ``entity``, targeting ``clinical_prepared/risk_training`` and
``clinical_prepared/entity_calculation`` respectively. Neither is dispatched by
the workflow. Its dispatcher rejects either command without ``--prepare-only``
before opening a log or running a producer. Fitting and real entity analysis
remain separate operations outside this code-only execution. Preparation
directories contain private patient inputs and are not public downloads.

After separately completed native risk and entity runs exist::

    python -m scripts.run_native_clinical_workflow collect \
      --preparation /new/clinical_prepared/PREPARED.json \
      --preparation-sha256 <sha256> \
      --risk-completed /new/clinical_prepared/risk_training/MODULE_LINEAGE.json \
      --risk-completed-sha256 <sha256> \
      --entity-completed /new/clinical_prepared/entity_calculation/ENTITY_CLINICAL_LINEAGE.json \
      --entity-completed-sha256 <sha256> --output /new/clinical_collected

Collection revalidates actual preparation sources and numerical values. Both
completed runs must match the declared output locations, run IDs, selected/core
identity, full configuration, raw source roles and patient authority. Risk
input arrays/targets and every entity patient-matrix description must match
preparation. The actual current Clinical publisher then reconstructs raw
features, frozen-head predictions and entity statistics before its existing
meta-analysis and selected-member context. Collection leaves completed result
directories unchanged and writes a new ``NATIVE_CLINICAL_BINDING.json`` whose
``clinical_context`` is accepted by the existing whole-publication workflow's
Clinical ``REUSE`` stage.

The current Clinical source reader follows ``workflow_preparation`` and repeats
preparation/completion matching during later audits. Preparation code, actual
input files and saved feature tables are bound dependencies. Rehashed but
inconsistent commands, patient counts, input fingerprints, run parameters or
completed source declarations are rejected. Failed preparation/collection
retains diagnostics and cannot publish a ready/success binding. Existing output
directories are refused; an output cannot be nested in an input dataset.

The fixture uses the actual generic patient-authority builder, both original
CLIs, both completion serializers, raw numerical replay and the existing
publication workflow. Its 40 patients/41 aliquots generate 240 typed patient-risk
rows, 270 entity fold rows and 54 summaries. Only private-head fitting is replaced
by marked frozen parameters and synthetic optimizer telemetry. Preparation is
tested with head construction, model calls, optimizers and entity statistics
forbidden. Actual entity statistics run only on synthetic inputs during
completion/replay checks. This verifies within-Clinical consistency, not
source acquisition/reference ancestry, State/core/cross-module patient closure,
optimizer history or independent predictive performance. Clinical command
handoff is implemented; single-cell handoff, complete dependency/external
acceptance, public scientific field definitions and final C01/C13/C16 acceptance
remain open. No actual fitting, real cohort analysis, paid compute or deployment
was started.

Clinical workflow verification on 149: ``clinical_workflow_r1`` passes 62
workflow/current-Clinical/numerical-replay cases in 448.73 seconds;
``clinical_workflow_consumers_r1`` passes 32 publication/browser, patient-context,
explicit patient-identity and non-fitting Clinical calculation cases in 94.63
seconds. All 94 distinct cases pass with no failures or skips. The earlier
``clinical_workflow_smoke_r1`` passed six cases and is not counted a second time.
There was no intermediate failing test receipt in this change.

Native SC-HHGT input collection and frozen numerical replay (2026-10-26)
---------------------------------------------------------------------

The original ``scripts.train_single_cell_hhgt_149`` CLI now supports
``--prepare-only``. Use its ordinary explicit ``--episode-root``,
``--graph-assets-root``, ``--output-root`` and configuration arguments on 149.
Preparation loads the actual fit, validation and query episode graphs for both
arms of all five folds, fingerprints their tensors and ordered rows, and writes
``INPUT_PREFLIGHT.json``. It constructs no neural model and performs no model
forward or optimization. Input assets are fully hash-verified before and after
collection, and only one episode graph is retained at a time. Invalid or
non-finite settings and an epoch budget unable to cover all fit episodes are
rejected. No completion, checkpoint or OOF prediction is produced by preparation.

Ordinary native completion now binds that preflight. The original 136-line
optimizer/model-selection block is extracted into ``fit_fold_model`` with its
15 AST statements unchanged from parent commit ``f87f6c8``; the static comparison
is recorded in ``sc_numeric_fit_loop_static_check.json``. The actual fitting
block was not executed. This extraction allows an explicitly marked frozen
fitting substitute to exercise the original completion serializer in fixtures.

After a separately completed run is available, a new standalone audit runs on
149, outside its immutable output directory::

    python -m scripts.audit_single_cell_numeric_replay_149 \
      --training-root /path/completed \
      --training-success-sha256 <sha256> --output /new/sc_numeric_audit

The audit executes the unchanged independent structural/history validator,
reconstructs the prepared graph/row tensors, loads all ten frozen native model
states and reruns query predictions and best-model inner-validation loss.
Prediction identities, targets, signed rho, absolute strength and residuals must
match; floating values use absolute tolerance 1e-6 with zero relative tolerance.
Checkpoint parameters and completed output files must remain unchanged.
``NUMERIC_REPLAY.json`` binds the actual completion, input preflight, checkpoint
and prediction files, implementation hashes and numerical checks. Failure leaves
a diagnostic receipt and cannot produce a passing numerical receipt.

The fixture uses actual native input loaders, the original CLI and completion
serializer, the independent validator and the new numerical audit. Its 20
synthetic episode files produce 40 arm-specific episode inputs, ten models and
40 OOF rows. Only fitting is substituted with explicitly changed frozen HGT
parameters and marked synthetic optimizer telemetry; neural optimizers are
forbidden. The preparation test additionally forbids model initialization and
model calls. This does not prove optimizer execution. The independent validator
deliberately remains separate from the new same-implementation numerical audit.

Verification on 149: ``sc_numeric_replay_r1`` passes 18 cases in 39.55 seconds;
``sc_numeric_consumers_r1`` passes 43 cases in 57.52 seconds. All 61 distinct
cases pass with no failures or skips. One existing test that updates parameters
through a real optimizer was intentionally deselected. The earlier two-case
smoke run is not counted again. Consistently rehashed changes to predictions,
derived fields, validation loss or input tensor fingerprints still pass the old
structural validator, but are rejected by the new numerical reconstruction.

The new audit is standalone at this stage: signed-SC fusion, model-use inspection
and current publication still retain their explicit no-neural-forward semantics.
Their existing regression checks do not establish integration of this new audit.
Unified SC preparation/completion handoff remains open. Inputs start at prepared
episodes/static graph assets; raw acquisition, reference, metacell and LASSO
transformation ancestry, cross-module patient closure and independent predictive
performance remain unproved. C01/C13/C16 stay in progress. No fitting, real cohort
analysis, paid compute or deployment was started.

Native single-cell preparation, completion and signed publication (2026-10-26)
----------------------------------------------------------------------------

``scripts.run_native_single_cell_workflow`` connects the original SC-HHGT CLI
input collector to completed native serialization, frozen numerical replay,
signed fusion features and the existing selected/publication consumers. It
starts at pre-materialized episodes and static graph manifests. It does not
produce raw-study, metacell or LASSO inputs and never dispatches fitting.

The input format is ``NATIVE_SINGLE_CELL_WORKFLOW_INPUT_V1``. Required fields
are a nonempty safe ``run_id`` and four path/SHA256 records: ``episode_manifest``
(``SC_HHGT_EPISODE_MANIFEST.json``), ``graph_asset_manifest``
(``GRAPH_ASSET_MANIFEST.json``), ``candidates`` and ``donor_crosswalk``.
Candidates have unique ``cancer_id,lncrna_id,pathway_id`` rows. The crosswalk has
unique ``dataset_id,query_donor_id`` rows and nonempty ``canonical_donor_id``.
Both tables must be nonempty. ``min_donors`` defaults to three and must be an
integer of at least two. The full declared candidate universe is preserved;
unsupported candidates retain missing values and explicit reasons.

Optional ``training_config`` accepts the original native CLI parameters:
``seed``, ``epochs``, ``patience``, ``minimum_validation_improvement``,
``prediction_batch_size``, ``max_rows_per_episode_step``,
``max_fit_episodes_per_epoch``, ``training_workers``, ``threads_per_worker``,
``learning_rate``, ``weight_decay``, ``hidden_channels``, ``num_layers``,
``num_heads``, ``dropout`` and ``max_gradient_norm``. Omitted values use the
original producer defaults and the expanded configuration is recorded.
Unknown fields, Boolean/fractional counts and non-finite values are rejected
by the native configuration checks. The five folds and both LASSO arms remain
mandatory; the primary arm is ``with_lasso``.

Native episode donor arrays contain unqualified IDs. This workflow therefore
requires every consumed donor ID to map unambiguously across the crosswalk,
including fit, validation, source and excluded donors, not only query donors.
Canonical outer-training/query and source/excluded partitions must be disjoint,
and each canonical query donor must remain in one outer fold. These checks
verify consistency of the supplied identity map; they do not independently
establish patient identity or close cross-module patient ancestry.

Run from the development checkout on server 149::

    python -m scripts.run_native_single_cell_workflow prepare \
      --manifest /path/sc_workflow.json --output /new/sc_prepared --plan

    python -m scripts.run_native_single_cell_workflow prepare \
      --manifest /path/sc_workflow.json --output /new/sc_prepared

The plan checks hashes, native manifest geometry, canonical donor partitions,
configuration, candidate/crosswalk tables and actual CLI argument parsing.
It creates no output. Preparation dispatches only the original CLI's
``--prepare-only`` command and compares its actual ``INPUT_PREFLIGHT.json``
against a reconstruction of both arms' graph/row inputs. It builds no model,
performs no forward or optimization and produces no trained completion.
``PREPARED.json`` binds the plan, numerical input description and
``EXECUTION_COMMAND.json``. The latter records the parsed original fitting
command targeting ``sc_prepared/training``; it is never executed by this
workflow. A dispatcher rejects a command lacking ``--prepare-only`` before
opening its log. These receipts contain private donor paths and identifiers.

After a separately completed native run exists::

    python -m scripts.run_native_single_cell_workflow collect \
      --preparation /new/sc_prepared/PREPARED.json \
      --preparation-sha256 <sha256> \
      --completed /new/sc_prepared/training/SUCCESS.json \
      --completed-sha256 <sha256> --output /new/sc_collected

Collection requires the exact declared completion directory, configuration and
prepared graph/row fingerprints. It executes the independent structural/history
validator and all ten frozen models' query/inner-validation reconstruction.
The saved numerical receipt is also rechecked when generating signed features.
Output uses a scoped ``PASS_SC_HHGT_WORKFLOW`` pointer, explicitly separate from
historical activation. No global pointer or historical result is rewritten.
``NATIVE_SINGLE_CELL_BINDING.json`` binds preparation, native completion,
numerical replay and ``signed/AUDIT.json``. Failed collection retains diagnostic
receipts without a successful final binding. Output directories must be new and
collection cannot write into completed training results.

The signed binding is accepted by the existing fusion preparer. Its two expert
channels retain positive and negative association strengths, equal context
weight within canonical donor and equal weight across donors. These strengths
are not probabilities. To attach it to an existing selected analysis, use the
existing ``CURRENT_SINGLE_CELL_CONTEXT_INPUT_V1`` manifest with
``selected_analysis`` pointing to that selected binding and ``native_binding``
pointing to ``sc_collected/signed/AUDIT.json``. Then run the existing
``scripts.materialize_v32_current_single_cell_context`` CLI. Its resulting
context binding is accepted by the overall publication workflow's single-cell
``REUSE`` stage. If selected fusion actually uses SC, the consumer requires this
same SC binding; adding a context to an unrelated selected score does not claim
that SC changed its ranking.

New workflow signed bindings carry ``native_numeric_replay_bound=true``.
Current-context generation and full re-audits repeat frozen neural forwards and
report that scope explicitly. Legacy pointers retain their existing aggregate
replay semantics and do not acquire a numerical-model guarantee. Model-use
inspection deliberately passes ``replay_model=false``: it reconstructs signed
aggregation, checks the preparation and every bound numerical dependency, then
inspects actual checkpoints and episode inputs without constructing/calling a
model. Its existing ``model_forwards_executed=false`` remains truthful.
The new receipt does not prove optimizer execution or independent performance.

Verification uses the actual original CLI, actual completion serializers and
real current fusion/context/publication consumers with synthetic prepared graph
inputs. Only the extracted fitting callback uses explicitly marked frozen
parameters and synthetic optimizer telemetry; Adam, AdamW and SGD are forbidden.
The existing fusion serialization fixture uses frozen zero weights, not fitted
weights or evidence of benefit. Source acquisition, reference/metacell/LASSO
ancestry, full patient closure, complete external acceptance, overall command
orchestration and final C01/C13/C16 acceptance remain open. No fitting, real
cohort analysis, paid compute or deployment was started.


Single-cell workflow verification on 149: ``sc_workflow_r1`` passes 68 cases in
222.15 seconds and ``sc_workflow_consumers_r1`` passes 39 cases in 115.31 seconds.
All 107 distinct related cases pass, with no failures or skips. The earlier
three-case smoke run is not counted again. Negative cases cover rehashed
commands/tensor counts/configuration, prediction and numerical receipt
corruption, donor aliases/missing mappings, mismatched source identity/minimums
and output/host boundaries.

The smoke tests passed; their SCP receipt return failed once with an SSH banner
error. The first handoff archive upload later encountered the same error. Each
authorized read-only probe returned hostname 149; the original receipts and
handoff transfer resumed without rerunning tests, changing login or opening UI.

Native fusion preparation, frozen completion and selected publication (2026-10-26)
--------------------------------------------------------------------------------

``scripts.run_native_fusion_workflow`` joins the native primary-label binding and
native auxiliary/signed single-cell bindings through the original current
fusion input collector. Preparation never dispatches the recorded fitting
command. Collection requires an existing completion, reconstructs all native
inputs and twelve frozen residual models, and calls the existing selected
GeneSet and current publication workflow. This closes the fusion command
handoff; it does not establish real optimization, an independently validated
winner or a complete acquisition-to-publication command chain.

The input format is ``NATIVE_FUSION_WORKFLOW_INPUT_V1``. Required fields are:

* ``fusion_run_id``: ``V32-FUSION-`` followed by 8--85 uppercase letters, digits,
  underscores or hyphens, compatible with the publication release identifier.
* ``primary_fusion``: a path/SHA256 record for the native current primary fusion
  binding, including all five native validation-label views.
* ``experts``: nonempty list of path/SHA256 records for native auxiliary or
  signed single-cell bindings. Their actual producer declarations determine
  every expert channel and its estimand. Duplicate expert IDs are rejected.
* ``selected_score_column``: ``discovery_adjusted_probability`` or
  ``fused_confidence_probability``; ``selection_rationale`` must explain this
  explicit source registration. The workflow does not select a winning model.
* ``geneset`` and ``subtype``: the existing selected-analysis configurations.
* ``publication_components``: all nine current publication components, each
  with ``BUILD`` and a pinned ``input_manifest``, ``REUSE`` and a pinned
  ``binding``, or an existing explicit unavailable status and a nonempty reason.

Optional ``training_config`` accepts the original ``ResidualFusionConfig``
fields and defaults: seed, five folds, optimization limits, learning rate,
regularization, sampling limits, evaluation interval, tolerance, clipping and
increment threshold. Counts must be integers, numeric settings finite, and
thresholds nonnegative. Optional ``rule_evidence_binding`` remains an annotation
and never becomes a fitted neural Evidence expert.

BUILD templates support existing ``workflow_ref`` references, including
``selected_analysis``, and the following ``result_ref`` values:
``selected_model``, ``core_run_id``, ``core_prediction`` and ``primary_fusion``.
An auxiliary context needing a core embedding export must pin that existing
export explicitly; this workflow does not create or resolve ``core_export``.
Unsupported result references and invalid GeneSet/subtype configurations are
rejected during planning, before any preparation output is created.

Run from the development checkout on server 149::

    python -m scripts.run_native_fusion_workflow prepare \
      --manifest /path/fusion_workflow.json --output /new/fusion_prepared --plan

    python -m scripts.run_native_fusion_workflow prepare \
      --manifest /path/fusion_workflow.json --output /new/fusion_prepared

Planning validates native bindings, configuration and publication templates
without creating output. Preparation runs the original CLI without ``--fit``
and independently reconstructs its private matrix and input audit. Native
available labels must cover all five pair folds. Missing expert values retain
their masks and typed nulls. Preparation may replay already completed upstream
single-cell models while validating their binding; it performs no new fusion
forward or fitting. The private plan, normalized manifest, matrix, audit, ready
receipt and non-dispatched ``EXECUTION_COMMAND.json`` are pinned by
``PREPARED.json``. A failed preparation has diagnostics but no successful
workflow binding. These artifacts are private, not website payloads.

After the separately recorded native fitting command has completed::

    python -m scripts.run_native_fusion_workflow collect \
      --preparation /new/fusion_prepared/PREPARED.json \
      --preparation-sha256 <sha256> \
      --completed /new/fusion_prepared/training/release/MULTIMODAL_FUSION_BINDING.json \
      --completed-sha256 <sha256> --output /new/fusion_collected

Collection checks the exact prepared output directory, configuration, sources,
matrix and completion hashes. The original crossfit bookkeeping is shared by
training and frozen replay through a model-provider callback; the original
optimizer body and crossfit driver statements are unchanged apart from this
callback. ``fusion_workflow_fit_static_check.json`` records the AST comparison.
Frozen replay verifies five fold models and one final model for each of the
discovery and confidence endpoints. It reconstructs the native bounded samples,
training centres, seeded initial-parameter hashes, final-parameter hashes and
regularized inner-validation loss, then every OOF prediction, fold/aggregate
metric, final score, contribution, availability value and serialized metadata.
It uses the original release serializer in a temporary directory and compares
its full outputs to the completion. It never calls the optimizer.

New current-fusion outputs bind ``CURRENT_FUSION_NATIVE_INPUTS_V1``. Existing
selected-score, external-validation and model-use consumers detect this
contract and repeat input and residual-output reconstruction. Source-only
inspection disables upstream neural forwards; full workflow collection
requests upstream numerical replay. Legacy fusion outputs keep their prior
validation scope. Legacy SC pointers do not acquire a native checkpoint replay
guarantee merely by entering this workflow.

Successful collection creates ``NUMERIC_REPLAY.json``, the actual selected
analysis, configured contexts, current publication/web binding and
``NATIVE_FUSION_BINDING.json``. It does not deploy the website or alter a global
pointer. Collection uses a new directory outside the training results and
leaves completed artifacts unchanged. Primary validation reuse remains labeled
``DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED`` throughout selection and GeneSet
members. Numerical agreement does not independently establish optimizer
execution, provenance completeness or biological predictive performance.

The end-to-end synthetic fixture has 80 candidates, 79 available primary labels,
four native expert channels and both endpoints; actual selected and website
consumers run, and the single-cell context must use the same binding as the
selected fusion. The fixture uses frozen native core checkpoints, a synthetic
genomic auxiliary completion and explicitly synthetic SC OOF correlations with
a legacy pointer. It does not prove those upstream optimizer or raw-data paths.
Only the fusion fitting callback is substituted with declared fixed positive
weights and zero optimizer steps. The native crossfit driver, serializer,
selection and publication execute unchanged. Neural optimizers and the actual
fusion optimizer are forbidden. Two existing tests that execute real fusion
optimization are intentionally deselected in this no-training verification.

Acquisition/reference/metacell/LASSO ancestry, complete per-pair/State-core and
cross-module patient closure, complete external acceptance, overall command
orchestration, public scientific field definitions and final C01/C13/C16
acceptance remain open. No real cohort analysis, fitting, paid compute or
deployment was started.

Fusion workflow verification on 149: ``fusion_workflow_r2``,
``fusion_workflow_consumers_r1`` and ``fusion_workflow_preflight_r1`` pass
137 distinct related cases with no failures or skips. The final workflow
module run supersedes its earlier cases when counting this total. Two
actual-optimizer tests are intentionally deselected. The earlier one-case
smoke pass is not counted again. Failed smoke r1 and main r1 receipts are
retained: their fixture tuple unpacking and expected exception type were
corrected. Negative cases reject rehashed input/configuration, commands,
OOF/metric/contribution/model/lineage changes, invalid publication settings,
unavailable result references and host/output boundaries. The source-only
inventory test forbids every neural module call and retains unresolved ancestry.

Resumable native workflow chain (2026-10-26)
-------------------------------------------

``scripts.run_native_workflow_chain`` connects the current native workflows by
their actual bindings. It is separate from the unused historical forty-task
``full_orchestration`` training DAG, whose all-heads-depend-on-core policy does
not describe independent SC or today's rule Evidence module. The new chain
never dispatches a fitting function or arbitrary shell command.

The input has exactly ``format=NATIVE_WORKFLOW_CHAIN_INPUT_V1``, ``chain_id``
and a nonempty ``stages`` list. Each stage has exactly ``id``, ``operation`` and
``inputs``. IDs start with a lowercase letter and contain at most 64 lowercase
letters, digits, underscores or hyphens. Stages are declared in dependency
order; duplicate, forward, self and unknown references are rejected. Supported
operations are:

* ``expression.run``, ``cnv.run`` and ``results.run`` execute their existing
  native workflows. Inputs are the corresponding complete native manifest.
* ``auxiliary.prepare/collect``, ``genomic.prepare/collect``,
  ``state.prepare/collect``, ``clinical.prepare/collect``,
  ``single_cell.prepare/collect`` and ``fusion.prepare/collect`` call the
  existing native handoffs. Preparation inputs are the native manifest;
  collection inputs are ``preparation`` and ``completed`` pinned records.
  Clinical collection instead takes ``preparation``, ``risk`` and ``entity``.
* ``core.accept`` takes a completed ``binding`` and its ``cnv_workflow`` record.
  It executes the native core and CNV validators and checks their direct link.
  Native core preparation includes L1 fitting, so it must be executed separately
  through its existing explicit gate. The chain does not call ``core.run`` or
  create that preparation on the user's behalf.
* ``publication.run`` executes the actual current publication workflow with its
  existing manifest, including all nine explicit component statuses. This can
  publish a collected fusion selection with subsequently completed State,
  Clinical, genomic, drug, distal, patient, SC and external contexts.

Nested input values may use the following exact objects:

* ``{"stage_ref":"earlier-id"}`` substitutes that stage's actual result
  path/SHA256 record. An optional ``pointer`` selects a JSON value from its
  binding, for example ``/core_export`` or ``/selected_analysis``. JSON pointer
  escaping uses ``~0`` and ``~1``; list entries use numeric indices.
* ``{"external_ref":"completed-run"}`` substitutes a registered external
  path/SHA256 record. Its optional pointer reads a field from that pinned JSON,
  for example the six completed core run declarations. Relative pinned paths
  within the selected payload resolve against the external JSON's directory.
  Payloads cannot inject further undeclared stage/external references.
* ``{"json_file":{...}}`` resolves a nested template and writes a private,
  content-addressed JSON file within the stage attempt, returning its pin.
  Use this when an existing native manifest expects a separate source-config or
  publication-template file.
* ``{"project_path":"relative/path"}`` explicitly resolves a plain path
  against the chain manifest directory. Other plain path strings should be
  absolute because each native input manifest is written in its own attempt
  directory. Existing file path/SHA256 records are checked and made absolute.
  Directory records preserve the native membership-hash declaration; the actual
  native stage preflight validates it using that producer's directory contract.
  Structural chain planning does not claim to have validated those datasets.

A minimal fusion continuation, after supplying the full native preparation
manifest in the first stage, has this shape::

    {
      "format": "NATIVE_WORKFLOW_CHAIN_INPUT_V1",
      "chain_id": "current-analysis",
      "stages": [
        {"id": "fusion-inputs", "operation": "fusion.prepare",
         "inputs": {"external_ref": "fusion-manifest", "pointer": "/manifest"}},
        {"id": "fusion-results", "operation": "fusion.collect",
         "inputs": {"preparation": {"stage_ref": "fusion-inputs"},
                    "completed": {"external_ref": "completed-fusion"}}}
      ]
    }

In this example the external JSON wraps the native input as
``{"manifest": <full native input>}``. An empty pointer always returns the file record itself, not its
JSON body. Alternatively place the native manifest inline as ``inputs``.

Run only on server 149::

    python -m scripts.run_native_workflow_chain plan --manifest /path/chain.json
    python -m scripts.run_native_workflow_chain init --manifest /path/chain.json --output /new/chain
    python -m scripts.run_native_workflow_chain advance --output /new/chain
    python -m scripts.run_native_workflow_chain advance --output /new/chain --external-inputs /path/completions.json

``completions.json`` has exactly ``format=NATIVE_WORKFLOW_CHAIN_EXTERNAL_V1``
and ``inputs``, a mapping from declared external names to pinned files.
Registrations are append-only: an existing external name cannot be replaced,
even by an identical file at a different path. Native validators decide whether
the registered data are legitimate completed results. No completion is inferred
from the external name, an optimizer status label or file existence alone.

``plan`` validates the graph and available direct pins; it does not claim that
every native scientific preflight has run. ``init`` pins the declaration and
implementation and creates a new chain directory. ``advance`` invokes ready
stages serially, continues independent stages across missing inputs or a terminal
stage exception, and records exact failures. Downstream consumers remain blocked
until their own dependencies pass. Each successful stage is revalidated through
the existing native consumer on resume, including its original input manifest
and exact preparation/completion link. Changing source data or implementation
invalidates the existing chain and requires a new declared run.

Attempts use new numbered directories. A terminal failed attempt retains its
diagnostics and a later advance can retry in a new directory. A live OS lock
rejects a concurrent coordinator. An unfinished attempt without a terminal
failure or committed stage receipt produces ``NEEDS_INSPECTION``; absence of a
process observation is never treated as permission to rerun scientific work.
A finished result left by an interrupted coordinator can be adopted explicitly,
with full native validation and no producer invocation::

    python -m scripts.run_native_workflow_chain recover --output /new/chain \
      --stage fusion-results --completed /existing/attempt/output/NATIVE_FUSION_BINDING.json \
      --completed-sha256 <sha256>

Then call ``advance``. Recovery cannot substitute an output outside its original
attempt. If the committed stage receipt already exists, advance reconciles it
directly. Recovery does not decide whether an unfinished external job is alive.

``CHAIN.json`` distinguishes READY, ADVANCING, FAILED, WAITING_EXTERNAL,
NEEDS_INSPECTION and DECLARED_STAGES_COMPLETE. The last means only that the
explicitly declared native stages passed. It does not claim acquisition coverage,
all scientific requirements, real optimizer observation, independent predictive
validation or deployment. ``training_started=false`` refers to this coordinator;
it does not erase the supplied completed producers' own history.

Publication re-audit now reconstructs all declared stage templates, selected
and component stores, model-use derivation, table-source inventory and public
catalog, and compares the actual completion/state records. Fusion collection
has a reusable validator connecting its prepared matrix/configuration,
twelve-model frozen replay, selected input and original publication templates.
These validators make resuming dependent stages meaningful instead of trusting
only a cached PASS flag. Public schemas are still not complete scientific field
definitions, and C01/C13/C16 remain in progress.

Native chain verification on 149: ``workflow_chain_r1``, ``workflow_chain_core_r2``, ``workflow_chain_r3``, ``workflow_chain_branches_r3``, ``workflow_chain_publication_r2``
pass 49 distinct related cases, counting the latest listed suite per
module. All 17 registered operations execute in separate scoped synthetic
chains, including both CNV and ATAC auxiliary branches. This is not one full
real-study execution. The earlier two-case smoke pass is not counted again.
No real optimizer, real cohort analysis, paid compute or deployment was run.

Cached native RBP evidence workflow (2026-10-26)
-----------------------------------------------

``scripts.run_native_rbp_workflow`` executes the existing ENCODE depletion or
RBNS sequence analysis from a verified local cache on host 149. The current
chain registry now has 18 operations, adding ``rbp.run`` to the preceding 17.
RBP analyses have no core dependency; stages depend only on explicitly declared
references. No neural/L1/fusion fitting, downloads, graph activation or global
pointer changes are dispatched. Existing native acquisition commands retain
their separate explicit ``--download`` option for future authorized execution.

The workflow manifest has exactly ``format=NATIVE_RBP_WORKFLOW_INPUT_V1``,
``kind`` (``encode_functional`` or ``rbns_sequence``), ``analysis`` (the pinned
native analysis JSON) and ``cache`` (a directory path). RBNS additionally accepts
positive integer ``chunk_rows`` (default 50000); ENCODE does not use this option.
The native analysis schemas, selected concentrations/contrasts, input artifact
roles and scientific thresholds are unchanged. The original analysis JSON's
relative artifact paths resolve against its own directory. The cache resolves
against the wrapper manifest, and must already contain every selected released
ENCODE file with matching size and MD5. Planning also pins each raw file's SHA256.
An absent cache or selected file fails before creating workflow output.

Run on 149::

    python -m scripts.run_native_rbp_workflow --manifest /path/rbp.json --output /new/rbp --plan
    python -m scripts.run_native_rbp_workflow --manifest /path/rbp.json --output /new/rbp

In a native chain use ``operation=rbp.run`` and the full wrapper manifest as
``inputs``. ``/audit`` selects the original producer audit; ``/outputs/<name>``
selects a native output record. These references can populate an actual
``publication.run`` Evidence BUILD template via ``json_file``. ENCODE supplies
``FUNCTIONAL_EVIDENCE.parquet`` for an ``encode_functional`` rule source; RBNS
supplies ``GENE_SEQUENCE_PREFERENCE.parquet`` for ``rbns_sequence`` rule evidence
and ``RBNS_TYPED_BINDING.parquet`` for an explicit predicted expansion source.
Both retain the original required ``source_id`` and ``audit`` fields.

Each new native audit binds ``source_contract=CACHED_RAW_RBP_ANALYSIS_V1``, its
original input manifest and cache root; RBNS also binds chunk size. The workflow
replays the actual native command into an isolated temporary directory on 149,
checks every metadata field and output inventory, and compares all parquet
columns and rows (floating tolerance rtol=1e-10, atol=1e-12). Source bytes are
rechecked before and after replay. ``raw_numerical_replay_verified`` means this
same-implementation reconstruction, not independent biological validation.
Collection uses a new output directory outside inputs/cache, with launch and
failure receipts. The original cached inputs and completed producer outputs
remain unchanged.

The current rule-source reader, rule-release query loader and RBNS comparison
source loader now perform this replay for the new contract. A saved rule-summary
audit does not exempt a native KD/RBNS source from later raw-file checks. Legacy
audits without raw-manifest binding keep their explicitly narrower historical
scope; merely loading them does not grant raw reconstruction. Fully re-auditing
large native releases can be expensive: this implementation reconstructs the
analysis rather than trusting a cached numerical PASS.

Actual native RBNS output exposed a mixed-source graph issue: RBNS declares a
``relation_type`` column while native October physical tables omit it. Source
normalization now derives the canonical relation from every assay class before
concatenation and rejects conflicting explicit declarations. Thus missing
optional columns cannot create null relationship types. The existing predicted
weight cap and three-arm policy are unchanged; predicted edges remain absent
from the current-C and experimental-only arms.

KD remains an RBP depletion/pathway overrepresentation result bridged through
same-biosample physical binding. It is not an observed lncRNA perturbation.
RBNS remains transcript-sequence preference with complete registered isoform
coverage, an independent-base-frequency comparator and unavailable values for
unscorable transcripts. Its score is neither measured lncRNA binding nor a
probability. The integration fixture uses actual cached synthetic files and
native analysis/aggregation/graph/publication code; only the pre-existing core
checkpoints are explicitly frozen synthetic containers. No real cohort analysis,
parameter fitting, download, paid compute or deployment was executed. Remaining
acquisition/statistical branches, full source/patient ancestry, public field
definitions and final C01/C13/C16 acceptance remain open.

Native RBP verification on 149: ``rbp_workflow_r1``, ``rbp_workflow_native_comparison_r1``, ``rbp_workflow_r2``
pass 119 distinct related checks (latest listed suite per module).
The initial failed smoke receipt is retained with the mixed-source schema fix.
No fitting, downloads, real cohort analysis, paid compute or deployment ran.

Native methylation workflow and distal consumption (2026-10-26)
--------------------------------------------------------------

``scripts.run_native_methylation_workflow`` invokes the existing HM450
materializer or completed-batch recovery on host 149. ``methylation.run`` is
the nineteenth operation in the resumable native chain. This entry point
does not download data, delete raw inputs or fit any learned model. It
preserves the existing sample weighting and beta missingness semantics.

The manifest has these fields::

    {
      "format": "NATIVE_METHYLATION_WORKFLOW_INPUT_V1",
      "mode": "raw",
      "selected_files": {"path": "selected.tsv", "sha256": "<SHA256>"},
      "probe_map": {"path": "probe/probes.parquet", "sha256": "<SHA256>"},
      "patient_folds": {"path": "folds.tsv", "sha256": "<SHA256>"},
      "download_root": "raw",
      "workers": 8,
      "batch_size": 64
    }

Artifact paths and cache/batch directory paths resolve against the manifest.
``workers`` and ``batch_size`` are optional positive integers, with defaults
shown above. Raw mode requires every selected file at
``download_root/file_id/file_name`` with matching MD5 and byte size. Preflight
also pins SHA256 for each file, the probe map, its existing audit, and the
patient-fold authority. Selected TSV columns are ``file_id``, ``file_name``,
``md5sum``, ``file_size``, ``cancer_id``, ``patient_id`` and ``sample_id``.
Probe rows identify ``probe_id``, ``lncrna_id`` and promoter/distal region;
patient rows identify cancer, canonical patient and ``patient_fold_id``.
The original producer remains responsible for validating beta/probe values.

For recovery use ``mode=recovery`` and additionally supply
``batches={"path":".../sample_feature_batches","sha256":"<directory SHA256>"}``.
The directory digest follows ``input_lineage.artifact_sha256``. Existing
``BATCH_INPUT_AUTHORITY.json`` and ``SAMPLE_PROCESSING_RECORDS.parquet`` are
read from the batch directory's parent and pinned separately. Recovery does
not require or read raw beta files; ``download_root`` is retained as a location
record only. A legacy batch without input authority remains ineligible for
predictive topology validation. Neither recovery nor its replay re-verifies
the original raw MD5, reconstructs missing original provenance, or authorizes
raw deletion. Changed raw files therefore invalidate raw-mode validation,
while a separately declared recovery can still validate unchanged batches.

Run on 149::

    python -m scripts.run_native_methylation_workflow --manifest /path/hm450.json --output /new/hm450 --plan
    python -m scripts.run_native_methylation_workflow --manifest /path/hm450.json --output /new/hm450

Planning writes no output. Execution requires a new directory outside raw and
batch inputs. It writes ``LAUNCH_PREFLIGHT.json``, ``stage.log``, the original
``features`` outputs and ``NATIVE_METHYLATION_BINDING.json``; exceptions preserve
``RUN_FAILED.json``. In a chain, use ``operation=methylation.run`` with the full
manifest as ``inputs``. ``/outputs/patient_features`` selects the patient table,
``/audit`` the original audit and ``/batch_root`` the directory binding accepted
by a subsequent recovery stage. Resume revalidates results without rerunning
the completed producer into its existing output directory.

The original command is rerun in isolated temporary storage during validation.
All native metadata and output inventories must match. Patient features,
sample records and, in raw mode, output batches are compared by keys with
exact schemas/missingness and absolute floating tolerance 1e-12 (rtol=0).
Raw mode also compares the generated batch authority. Files are checked again
before workflow acceptance. ``native_numerical_replay_verified`` means this
same-implementation reconstruction, not independent biological validation.
Repeated raw validation incurs the cost of reading and aggregating the full
cache again; recovery incurs the cost of aggregating the existing batches.

Patient beta is the mean within each exact sample across its file replicates,
followed by equal sample weighting within patient. A beta of zero is a measured
value; unavailable beta is null with ``__available=false``. Probe counts retain
their original meaning, mean informative probes per file then per sample.
``raw_checksum_reverified`` is true only in raw mode. Workflow fields
``raw_deletion_executed``, ``source_ancestry_complete``, ``training_started`` and
``production_deployed`` remain false. The original raw audit's conditional
deletion-readiness field does not dispatch deletion or waive its separate
distal-head/independent-closure prerequisites.

The actual distal input assembler now accepts the paired options::

    --methylation-workflow /new/hm450/NATIVE_METHYLATION_BINDING.json
    --methylation-workflow-sha256 <SHA256>

Its existing ``--methylation`` must identify that workflow's exact patient
feature output. The assembler validates the workflow and compares all assembled
promoter/distal beta values, availability and matching patient folds to the
native table. Current distal publication and later context-query validation
repeat this check and retain its original inputs in the source inventory.
Rehashing an altered assembled table therefore does not bypass the comparison.
Legacy assembler calls without the optional workflow retain their former scope;
they are not retroactively granted raw methylation replay coverage.

Verification uses actual raw-file and batch aggregation, actual coordinate
topology construction, the existing distal assembler/statistical CLI and the
current publication/query consumer. Two-patient tests preserve unavailable
results; a 60-patient synthetic example reaches an available negative pathway
association with measured promoter beta and absent distal beta. Statistical
regressions execute only on synthetic data; neural/L1/auxiliary/fusion optimizer
fitting is prohibited. Probe-region assignment is an explicit synthetic fixture,
not a run of the official reference-to-probe mapping producer. Mutation,
expression and covariates are synthetic; the selected core uses existing frozen
synthetic containers. These checks do not recover the real 8,236-file run,
establish methylation causality, improve a real GeneSet, complete upstream
reference or cross-module patient ancestry, or finish C01/C13/C16 acceptance.

Native methylation verification on 149: ``methylation_workflow_r2``, ``methylation_workflow_consumers_r1``
pass 60 distinct related checks. The earlier failed fixture
receipt is retained. No neural/L1/auxiliary/fusion optimizer, real cohort
analysis, raw deletion, download, paid compute or deployment was run.

Native distal statistics and declared-scope closure (2026-10-26)
---------------------------------------------------------------

``scripts.run_native_distal_workflow`` joins the existing patient input
assembler, nested two-stage statistical command and final closure auditor.
The resumable registry now contains 20 operations, including ``distal.run``.
Every operation executes on 149. The new path does not launch neural/L1/
auxiliary/fusion optimizers, acquire missing raw datasets or delete files.
It does execute the original ridge regressions as a declared statistical
analysis; this is not a plan-only preparation command.

The exact manifest fields are::

    {
      "format": "NATIVE_DISTAL_WORKFLOW_INPUT_V1",
      "methylation_workflow": {"path": "hm450/NATIVE_METHYLATION_BINDING.json", "sha256": "<SHA256>"},
      "formal_expression_root": {"path": "expression", "sha256": "<directory SHA256>"},
      "distal_mutation": {"path": "mutation/features.parquet", "sha256": "<SHA256>"},
      "distal_atac_root": {"path": "atac", "sha256": "<directory SHA256>"},
      "local_cnv": {"path": "cnv.parquet", "sha256": "<SHA256>"},
      "purity_covariates": {"path": "purity.parquet", "sha256": "<SHA256>"},
      "pathway_activity_root": {"path": "activity", "sha256": "<directory SHA256>"},
      "candidate_universe": {"path": "candidates.parquet", "sha256": "<SHA256>"},
      "parameters": {
        "ridge_alpha_expression": 1.0,
        "ridge_alpha_pathway": 1.0,
        "min_expression_train_patients": 20,
        "min_pathway_train_patients": 20
      }
    }

Paths resolve against the manifest. Directory hashes use
``input_lineage.artifact_sha256`` and bind all files, including source metadata.
Directories must be nonempty; an ATAC directory without selected cancer
matrices may contain its explicit unavailable-source record. Missing cancer
ATAC files keep the original assembler's unavailable semantics. Present
ATAC files require their topology-bound native audit. Mutation features
require their adjacent native audit, and mutation, methylation and any
present ATAC matrix must share outcome-independent distal topology.
No diagnostic/cohort-selected-topology override is exposed by this workflow.

Candidate keys are cancer, lncRNA and exact pathway. They must be complete
and unique, with canonical cancer IDs. The candidate file determines the
requested cancers; expression and activity require each corresponding
``cancer_id=<ID>/part-0.parquet``. Other table schemas and sample-merging rules
are those of the original assembler. Parameters are optional with defaults
shown above; ridge penalties must be finite and nonnegative, patient-count
thresholds integers of at least five. Unrecognized options are rejected.

Run explicitly on 149::

    python -m scripts.run_native_distal_workflow --manifest /path/distal.json --output /new/distal --plan
    python -m scripts.run_native_distal_workflow --manifest /path/distal.json --output /new/distal

Planning validates sources without writing output or fitting regressions.
Execution requires new output outside input directories and records the host,
inputs, implementations and parameters before analysis. It writes ``head_input``,
``statistics`` and ``closure`` directories, ``stage.log``, and finally
``NATIVE_DISTAL_BINDING.json``. Exceptions retain ``RUN_FAILED.json``. In a
chain use ``operation=distal.run`` and the manifest as ``inputs``. A preceding
``methylation.run`` result can populate ``methylation_workflow`` by stage
reference. The result's ``/audit`` is the native statistical binding accepted
by the existing distal publication template; ``/closure`` identifies its final
auditor record. No neural or fusion fitting stage is implicitly dispatched.

The input audit binds ``native_distal_preparation`` to the immutable workflow
preflight. Validation reconstructs the entire assembled patient table from
the registered expression, mutation, ATAC, methylation, CNV and purity sources
in temporary storage on 149. Metadata, partition identity, every column,
patient keys, availability and numerical values must agree; float tolerance
is absolute 1e-12, with exact missingness and schemas. This extends the earlier
methylation-only projection check to all assembled covariates. Current distal
publication/query consumes this source contract and retains these dependencies.
The original statistical command is also replayed, and its actual input roots,
candidate authority and four exposed parameter values must match preparation.

The closure auditor now accepts ``--candidate-scope-sha256`` for a declared
candidate file. It compares exact output candidate identities as well as row
counts; matching counts with substituted pathway IDs are rejected. Without
that option its historical 33-cancer/3,300,000-candidate constraints remain.
The audit reports the actual scope and ``full33`` truthfully. Declared-scope
closure never authorizes raw methylation deletion. Its original requirement
for at least one available result remains: a statistically uninformative input
may produce completed native statistics but will not receive successful
closure or a completed workflow binding. It is not converted into a positive
scientific result. Successful closure is rerun and compared during workflow
validation, excluding only its timestamp.

The current estimand is the mutation-associated expression component after
adjusting for ATAC, methylation, local CNV and purity, followed by its nested
OOF relationship to pathway activity. ATAC and methylation are covariates here;
their inclusion does not estimate their separate causal effects. Patient
isolation, sign, missingness, ridge defaults and all availability thresholds
are unchanged. Closure verifies internal consistency and coverage; it is not
independent biological validation or proof of improved final GeneSets.

Tests use sixty synthetic patients, actual raw methylation aggregation, native
assembly/statistics/closure and current publication/query. Mutation, expression,
CNV, purity and activity are synthetic prepared inputs; probe assignment is an
explicit fixture. Existing frozen core containers supply the selected result.
No raw MC3/liftover, official distal ATAC or official probe-reference producer
is exercised by this new workflow. Those upstream acquisition/reference
contracts, complete cross-module patient ancestry, remaining branch orchestration
and final C01/C13/C16 acceptance remain open. No real cohort was analyzed.

Native distal verification on 149: ``distal_workflow_r1``, ``distal_workflow_consumers_r1``, ``distal_workflow_r2``
pass 67 distinct related checks. Statistical regression executed
only on synthetic data. No neural/L1/auxiliary/fusion optimizer, real cohort
analysis, raw deletion, download, paid compute or deployment was run.

Native distal ATAC matrix preparation (2026-10-26)
------------------------------------------------

``scripts.run_native_distal_atac_workflow`` now runs the original R distal-peak
aggregation from the official normalized TCGA ATAC matrix and its identifier
and peak tables. The registry adds ``distal_atac.run`` as operation 21. The
workflow does not acquire data, run sequence alignment, fit a model, delete
inputs, or activate global results. It executes on host 149 using
``/usr/bin/Rscript`` and records its executable hash, R version and versions of
``data.table``, ``Matrix`` and ``jsonlite``.

The input manifest has exactly these fields::

    {
      "format": "NATIVE_DISTAL_ATAC_WORKFLOW_INPUT_V1",
      "matrix": {"path": "matrix.rds", "sha256": "<SHA256>"},
      "mapping": {"path": "mapping.tsv", "sha256": "<SHA256>"},
      "peak_set": {"path": "peak_set.tsv", "sha256": "<SHA256>"},
      "topology_links": {"path": "INDEPENDENT_DISTAL_LINKS.tsv.gz", "sha256": "<SHA256>"},
      "patient_folds": {"path": "patient_folds.tsv", "sha256": "<SHA256>"},
      "candidate_universe": {"path": "candidates.parquet", "sha256": "<SHA256>"}
    }

All paths resolve against the manifest. Native candidates require unique,
complete cancer/lncRNA/exact-pathway keys. Their cancer set and union of lncRNAs
define aggregation scope. Candidate gene IDs accept the formal ``LNC:`` prefix
and version suffix; the R producer now normalizes them consistently with the
topology file. The original patient authority is pinned, then restricted to
declared cancers in a separately written input. Every declared cancer must
retain folds 0 through 4; fractional fold IDs and patients assigned to multiple
declared cancers are rejected. Patients outside the declared cancer scope are
not passed into the R aggregation.

The original R command still accepts its eight arguments. A ninth, optional
JSON argument declares ``format=DISTAL_ATAC_COHORT_V1``, requested ``cancers``,
``covered_cancers`` and ``raw_gap_cancers``. Native coverage must be exactly
the intersection of requested cancers and the official raw 23-cancer set.
Without this ninth argument the historical full33 scope remains. The wrapper
generates the declaration, required genes and scoped folds in ``inputs``.

The raw input requirements are unchanged: the matrix is a data frame with
562,709 peaks and seven leading metadata columns; the sample mapping must
contain all official 23 cancers and uniquely account for every sample column.
Every matrix peak must match the accompanying peak table by name, chromosome
and coordinates. Predictive independent-topology links additionally must
carry matching chromosome, BED0 start and end for the actual matrix peaks.
This prevents matching names from concealing coordinate-version drift. The
existing independent-topology policy remains mandatory for the native workflow;
legacy diagnostic topology is not promoted by deleting statistical columns.

Run explicitly on 149::

    python -m scripts.run_native_distal_atac_workflow --manifest /path/atac.json --output /new/atac --plan
    python -m scripts.run_native_distal_atac_workflow --manifest /path/atac.json --output /new/atac

Planning writes no output and performs no matrix aggregation. It verifies
declared inputs, candidates, folds and topology, and queries installed R/package
versions. Execution writes a preflight, generated inputs, ``r.log``, the original
``features`` outputs and ``NATIVE_DISTAL_ATAC_BINDING.json``. Failures retain
``RUN_FAILED.json`` and the R log without a completed workflow binding. All raw
inputs remain unchanged. In a native chain use ``operation=distal_atac.run``;
``/feature_root`` is the directory pin accepted by ``distal.run`` as its
``distal_atac_root``. This connects original matrix aggregation to patient
assembly, nested statistics and final closure without manual path substitution.

The native feature audit binds ``native_distal_atac_preparation``. Validation
reruns the R command into temporary storage and compares its metadata, every
accessibility matrix, the lncRNA/peak mapping table, and all generated inputs.
TSV table values use absolute tolerance 1e-12, preserving types, row/column
identity and missingness. File MD5/SHA records and exact cancer inventory are
also checked. A process-local replay cache can avoid another R aggregation
only after current raw input bytes, generated inputs, output bytes, audit,
implementation hashes and R/package versions reproduce the verified signature.
Even a cache hit rechecks source content; changed mapping bytes or a jointly
rewritten output/audit cannot reuse the cached proof. Full input hashing remains
an I/O cost on every validation; a fresh process must reconstruct the R outputs.

The feature-topology reader, native distal planner and current distal
publication/query now honor this contract. Raw matrix/mapping/peak inputs,
candidate scope, patient authority, topology and implementation records reach
the downstream private source inventory. Legacy feature audits without the
native preparation pointer retain their previous, narrower coverage.

Aggregation semantics remain: technical replicates are averaged within exact
aliquot; aliquots receive equal weight within patient; linked peaks receive
equal weight within lncRNA. A measured zero remains a value. Raw-gap cancers
produce no accessibility matrix, and candidates without linked peaks are absent
from the matrix rather than assigned zero; the distal assembler retains them
as explicitly unavailable. ATAC values remain normalized accessibility signals,
not binary openness, calibrated probabilities or causal effects.

Verification uses a synthetic matrix retaining all 562,709 peaks and all 23 raw
cancer mappings. Sixty BRCA patients plus technical/aliquot repeats test exact
weighting; declared DLBC coverage remains missing and an unlinked candidate is
not filled with zeros. The actual R producer reaches native distal statistics,
closure and public queries with the previous synthetic omics/frozen-core
fixtures. This is not a biological cohort analysis, official-asset download,
or reconstruction from sequencing reads. The new path verifies aggregation
from supplied normalized matrices; upstream normalization/acquisition, official
reference ancestry, MC3/liftover, probe-reference preparation and full cross-module
patient coverage remain separate unfinished requirements. C01/C13/C16 remain open.

Native distal ATAC verification on 149: ``distal_atac_workflow_r1``, ``distal_atac_workflow_consumers_r1``
pass 55 distinct related checks. Actual R aggregation and downstream
statistical regressions ran only on synthetic inputs. No neural/L1/auxiliary/fusion
optimizer, real cohort, raw deletion, download, paid compute or deployment ran.

Official HM450 reference preparation (2026-10-26)
-----------------------------------------------

``scripts.run_native_probe_workflow`` adds ``probe.run`` as operation 22. It
generates coordinate-only distal candidates from supplied GRCh38 promoters,
peaks and the complete required-gene universe, then executes the original
official HM450 interval mapper. It does not consume patients, select links by
expression/activity, fit a model, acquire data or delete source files.

The manifest accepts exactly four file pins and an optional distance::

    {
      "format": "NATIVE_PROBE_WORKFLOW_INPUT_V1",
      "hm450_manifest": {"path": "HM450.hg38.manifest.gencode.v36.tsv.gz", "sha256": "<SHA256>"},
      "promoters": {"path": "promoters.tsv.gz", "sha256": "<SHA256>"},
      "required_genes": {"path": "genes.tsv", "sha256": "<SHA256>"},
      "peak_set": {"path": "peaks.tsv", "sha256": "<SHA256>"},
      "max_distance_bp": 1000000
    }

All paths resolve relative to the manifest. The official reference must retain
MD5 ``e163fc110043abb5a7ef623816383bb9`` and at least 450,000 valid, uniquely
identified probes. These gates are unchanged, including during verification.
Promoters require ``chromosome`` (or ``#chrom``), ``start``, ``end``, ``gene_id``;
coordinates are BED0 half-open. Peak columns are ``seqnames,start,end,name``
with 1-based inclusive coordinates. Genes require ``gene_id``. Compression is
inferred; both plain and gzip TSV are accepted. Gene normalization removes an
optional ``LNC:`` prefix and version suffix. Chromosome prefixes are normalized.
Nonintegral, missing, negative or empty intervals are rejected before casting.

The fixed-distance topology includes every peak within the configured window
of any transcript promoter, excluding peaks intersecting any promoter of the
same gene. Zero-link genes remain in ``GENE_COVERAGE.tsv``. These are proximity
hypotheses, not validated regulatory links. The wrapper labels promoter inputs
``INPUT_BED0_PROMOTERS_SOURCE_BOUND``; supplied BED coordinates alone do not prove
they were derived from the stated GENCODE annotation. The original standalone
mapper retains its historical promoter-definition default. Eligible independent
links must carry coordinates matching the supplied peak table, in addition to
matching names. Probe overlap uses strict half-open comparisons, so touching
boundaries alone do not count as overlap.

Run on host 149 only::

    python -m scripts.run_native_probe_workflow --manifest /path/probes.json --output /new/probes --plan
    python -m scripts.run_native_probe_workflow --manifest /path/probes.json --output /new/probes

Plan verifies source pins, the official checksum and parameter/host contracts
without creating outputs or calculating a map. Coordinate/table validation
occurs in the actual producers. Execution writes ``LAUNCH_PREFLIGHT.json``,
``topology``, ``features``, ``stage.log`` and ``NATIVE_PROBE_BINDING.json``. A
failed run retains diagnostics and does not publish a completed binding.
The map audit binds ``native_probe_preparation``. Validation reconstructs both
producers in temporary storage, compares complete topology and coverage files,
all probe-assignment rows/schema, and native audit metadata. Only generated
output locations and their dependent audit hash are normalized for comparison.

In a chain, pass ``{"stage_ref":"probes","pointer":"/probe_map"}`` to
``methylation.run`` as ``probe_map``. ``/topology_links`` provides the same
coordinate topology for downstream mutation/ATAC producers. Native methylation
plans and downstream distal public-context inventories preserve the official
reference, supplied coordinate inputs, topology, code and preparation records.
A process-local cache avoids repeat reconstruction only after all current
input/output/code pins are rechecked. A fresh process reconstructs again.
Legacy maps without a native preparation pointer retain their earlier limited
validation; they do not acquire source-reconstruction coverage automatically.

Verification uses the cached, checksum-verified official static reference with
synthetic promoter/peak regions, required genes and patient Beta values. It
executes actual topology/mapping, raw methylation/recovery, distal assembly and
public queries. The two-patient statistical example remains explicitly
unavailable; no association or effect is invented. Original mutation, expression,
CNV, purity, activity and frozen-core fixtures remain synthetic substitutes.
This closes reference-to-map reconstruction, not promoter annotation derivation,
original data acquisition, MC3/liftover, full patient ancestry or biological
study completion. C01/C13/C16 remain open; no real cohort analysis, optimizer,
paid compute, raw deletion or deployment is authorized or executed here.

Native official probe verification on 149: ``probe_workflow_r1``, ``probe_workflow_consumers_r1``
pass 56 distinct related checks. Static official HM450 reference
was read; all patient values and coordinate regions were synthetic. No neural/L1/
auxiliary/fusion optimizer, biological cohort, deletion, download, paid compute or
deployment ran.

Raw MC3 distal mutation preparation (2026-10-26)
----------------------------------------------

``scripts.run_native_distal_mutation_workflow`` adds operation 23,
``distal_mutation.run``. It prepares actual supplied MC3 MAF rows, lifts their
coordinates, generates the patient assay-observation table and invokes the
existing distal mutation materializer. It consumes no expression/pathway
outcomes and does not train a learned model. All execution is on host 149.

The manifest contains exactly these fields::

    {
      "format": "NATIVE_DISTAL_MUTATION_WORKFLOW_INPUT_V1",
      "mc3": {"path": "mc3.v0.2.8.PUBLIC.maf.gz", "sha256": "<SHA256>"},
      "chain": {"path": "hg19ToHg38.over.chain.gz", "sha256": "<SHA256>"},
      "patient_folds": {"path": "patient_folds.tsv", "sha256": "<SHA256>"},
      "peak_set": {"path": "peaks.tsv", "sha256": "<SHA256>"},
      "topology_links": {"path": "INDEPENDENT_DISTAL_LINKS.tsv.gz", "sha256": "<SHA256>"}
    }

Original production SHA256 guards are unchanged: MC3
``cb0caa0e3a80a24cc4ccaaacc95c690a4032cc87c8ac78519a5b33f3aa3f6b81``
and the UCSC hg19-to-hg38 chain
``5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d``.
There is no runtime option to bypass these guards or substitute a synthetic
source profile. Planning verifies pins, host, independent topology and patient
authority without writing outputs or parsing patient variants.

Run on 149::

    python -m scripts.run_native_distal_mutation_workflow --manifest /path/mutation.json --output /new/mutation --plan
    python -m scripts.run_native_distal_mutation_workflow --manifest /path/mutation.json --output /new/mutation

The source adapter follows the genomic/allele identity and MAF-observed assay
semantics located in the historical ``CC_HHGT_v2_7_mutation``
``src/cc_hhgt_v27/variants.py`` and ``context.py``. It prepares the fields needed
for distal analysis, not the historical coding/transcript/protein annotations.
Raw MAF rows are streamed, required fields validated and source row numbers
retained. GRCh37, positive integral 1-based inclusive intervals, complete
alleles and canonical TCGA tumor aliquots are mandatory. Only declared
patients enter event outputs; counts of excluded out-of-cohort rows remain in
``SOURCE_AUDIT.json``. No MAF FILTER row selection is introduced. A canonical
patient cannot occur in multiple cancers, and fractional folds are rejected
before casting. The complete authority must contain folds 0 through 4.

Variant IDs retain the v2.7 genomic-coordinate/allele SHA256 prefix convention;
chromosome prefixes and integer formatting are canonicalized. Repeat aliquots
and repeat event rows share a variant identity. The generated event table
retains every selected source row, while coordinates are unique by variant and
the distal materializer counts unique patient/variant/gene observations.

The chain parser uses UCSC zero-based half-open coordinates and reverse-query
conversion as documented at https://genome.ucsc.edu/goldenPath/help/chain.html.
The source interval must fit completely inside exactly one aligned block.
Reverse-strand mappings retain their strand and chain ID. Cross-gap, absent and
ambiguous mappings remain present with null target coordinates and explicit
failure reasons. This replaces historical endpoint pairing, which did not
prove the interior mapped continuously. The stricter policy can quarantine
historically accepted intervals and is recorded as
``UNIQUE_COMPLETE_ALIGNED_BLOCK_GRCH37_TO_GRCH38``. These are positional
overlaps, not allele/reference validation against target FASTA.

Assay availability means a patient was observed in selected MAF records. A
patient absent from those records remains unavailable; an independently assayed
patient with zero MAF records cannot be discovered from this source alone.
All locus callability flags remain false. A zero burden means no reported
event overlapped a linked peak, not confirmed locus-level wild type. Source
quarantine counts remain explicit. The existing ``log1p`` unique-variant burden
and descriptive within-cancer site recurrence are unchanged; the formal
statistical head continues to use burden, not the whole-cohort recurrence
descriptor. Predictive topology coordinates must match the actual peak table.

Execution writes ``inputs`` (events, lifted coordinates, crosswalk and audit),
``features`` (the original four feature tables and audit), preflight/log and
``NATIVE_DISTAL_MUTATION_BINDING.json``. The audit binds
``native_distal_mutation_preparation``. Validation reconstructs source tables
from MAF/chain, reruns the existing materializer and compares complete typed
tables and metadata. A process cache rechecks every current input/output/code
pin before reuse; fresh processes reconstruct again. Failed execution retains
diagnostics without a completed binding, and never deletes source files.

Pass ``/distal_mutation`` from this stage to the matching field of ``distal.run``.
Feature readers and native distal planning validate the source reconstruction;
public-context consumers also compare actual assembled mutation values, flags
and patient folds with the source producer. Raw MAF, chain, patient/peak/topology,
generated inputs and code records reach the private source inventory. Legacy
feature audits without the preparation pointer retain their narrower validation.

Verification uses the unchanged cached official chain with synthetic MAF rows,
regions and 60 patients. The test first proves production rejects the synthetic
MAF, then substitutes only its expected digest inside the test process. This is
an explicit test substitution, not execution against official MC3 patient rows.
Synthetic probe assignments and other omics/frozen-core fixtures remain. Tests
cover actual raw preparation, native distal statistics/closure/public query,
duplicate aliquots, missing assays, strand/gap/ambiguity cases, chunk invariance
and rehashed corruption. No real cohort, optimizer, download, paid compute,
raw deletion or deployment ran. Raw acquisition, complete annotation/metacell/
LASSO branches, patient ancestry and final C01/C13/C16 acceptance remain open.

Native raw MC3 verification on 149: ``distal_mutation_workflow_r1``, ``distal_mutation_workflow_consumers_r1``
pass 55 distinct related checks. Actual static official chain and
synthetic MAF/other omics were used, with the test-only MAF digest substitution
described above. No neural/L1/auxiliary/fusion optimizer, biological cohort,
deletion, download, paid compute or deployment ran.

Single-cell raw matrix mapping (2026-10-26)
-----------------------------------------

``scripts.run_native_single_cell_mapping_workflow`` adds operation 24,
``single_cell_mapping.run``. It calls the original r7/r11 input-contract reader
and original metacell matrix mapper using an explicit upstream root, instead
of looking up a fixed historical active-run directory. It writes measured
coding/lncRNA matrices, coding grouping-scale values, full-gene library totals,
exact pathway signature arrays, identifiers and H5-aligned retained metadata.
It does not fit PCA, LASSO or neural models, infer QC/cell annotations, acquire
scientific data, delete scratch data or promote a current result pointer.

Manifest fields are::

    {
      "format": "NATIVE_SINGLE_CELL_MAPPING_WORKFLOW_INPUT_V1",
      "cancer_id": "BRCA",
      "r7_root": "/path/to/upstream_capsule",
      "run_status": {"path": "/path/RUN_STATUS.json", "sha256": "<SHA256>"},
      "server_preflight": {"path": "/path/SERVER_PREFLIGHT.json", "sha256": "<SHA256>"},
      "max_rank": 1500
    }

``max_rank`` is optional and defaults to 1500. All paths resolve relative to
the manifest. The upstream preflight must name target host 149. The existing
r7/r11 contract remains mandatory: declared formal cancer, source-generation
paths, complete metadata, H5/metadata barcode alignment, explicit tumour/
doublet/source-review/preprocessing masks, annotation and exact membership
hashes, and all 2,135 pathways. The wrapper does not fabricate historical
receipts or relax pathway/source gates. ``TRAINING_HANDOFF.json`` and the
dataset manifest in the supplied root remain bound dependencies. Where a
source-review policy is declared, the reader now retains its path as well as
its checked hash in the actual dependency inventory.

Run on 149::

    python -m scripts.run_native_single_cell_mapping_workflow --manifest /path/mapping.json --output /new/mapping --plan
    python -m scripts.run_native_single_cell_mapping_workflow --manifest /path/mapping.json --output /new/mapping

Planning reads and verifies the upstream contract, metadata and H5 identity
arrays, but does not write output matrices. Execution writes preflight/log,
``mapped/<cancer>/MAPPED_SUCCESS.json`` with its actual native files, and
``NATIVE_SINGLE_CELL_MAPPING_BINDING.json``. Failures retain diagnostics
without a completed binding. The ``/mapped_root`` output is a directory pin;
its ``path`` is the ``--scratch-root`` input for existing metacell/nested
preparation commands. Other prerequisites for those commands remain required.
This stage alone does not produce training episodes or signed predictions.

The original mapper gains an optional explicit upstream bundle while its
historical default stays available. When it encounters a cached mapped result
with ``native_mapping_preparation``, it calls the new source validator before
returning the cache. Thus existing downstream callers of this mapper receive
reconstructed raw-source validation automatically. Legacy cached results
without this pointer retain their earlier hash-only contract.

Validation reopens the actual raw H5 and reconstructs the original mapping in
temporary storage. It compares every sparse/array value, shape and dtype,
identifier order, native metadata and the complete retained cell table in
row order. Order matters because each matrix row represents that cell. A
process cache rechecks current raw inputs, policies, output bytes and code
before reusing an earlier reconstruction. Duplicate raw barcodes are now
rejected even when dimensions and repinned source hashes appear consistent.

For raw counts, duplicate stable gene rows are summed, coding grouping values
use log1p of counts normalized by the library across ALL original features,
and metacell aggregation remains sum-then-normalize/log1p. Unmapped features
still contribute to the library. Provided log-normalized input is preserved
on its supplied scale; duplicate mapped gene rows are rejected rather than
summed, and no exponential inversion is introduced. Tumour, doublet and
source-review exclusions retain exact H5 order; donor/cell-type contexts remain
in the native metadata. Supplied QC and annotation declarations are checked
for consistency but their independent biological correctness is not proved.

Verification uses synthetic raw-count and log-normalized H5 files, shuffled
metadata, 123 cells with 120 retained across ten donors, duplicate raw-count
gene rows, unmapped library features and all 2,135 pathway identities. Actual
mapping reaches the original cached metacell-input reader, aggregation and
UCell scorer. PCA/LASSO/optimizer entry points are disabled during these checks.
This is supplied-H5-to-mapped-matrix reconstruction, not nested episode/LASSO
completion, raw acquisition, annotation inference or independent donor identity.
Those branches and final C01/C13/C16 acceptance remain open.

The existing r7 publication regression now launches the real producer in a
child process because its finalization uses ``os.execve``. The parent pytest
process checks the resulting summaries and independent association reference.
Only an isolated test copy redirects historical authorized output roots to
its private temporary directory; production restrictions are unchanged. This
tests the actual exec, PID, file-hash and atomic publication handoff. The
synthetic contract now explicitly carries its cancer ID for handoff checks.

Native single-cell mapping verification on 149: ``single_cell_mapping_workflow_r2``, ``single_cell_mapping_workflow_consumers_r2``
pass 69 distinct related checks. The initial collection-only failure
was missing h5py; version 3.16.0 was installed only in the isolated verification
directory on 149. Synthetic H5/metadata and existing frozen-consumer fixtures
were used. No LASSO/neural optimizer, biological cohort, raw deletion, scientific
data download, paid compute or deployment ran.

Canonical donor preparation for nested single-cell inputs (2026-10-27)
-------------------------------------------------------------------

``single_cell_donors.run`` is operation 25. Its CLI is
``python -m scripts.run_native_single_cell_donors_workflow --manifest INPUT.json --output NEW_DIRECTORY``;
``--plan`` validates without writing. This stage consumes the raw-H5 mapping
binding, an explicit donor crosswalk, existing static graph assets and current
bulk candidates. It does not execute geometry, PCA, LASSO or neural fitting.

The manifest is::

    {
      "format": "NATIVE_SINGLE_CELL_DONORS_INPUT_V1",
      "mapping": {"path": "/path/NATIVE_SINGLE_CELL_MAPPING_BINDING.json", "sha256": "<SHA256>"},
      "donor_crosswalk": {"path": "/path/donors.parquet", "sha256": "<SHA256>"},
      "graph_preflight": {"path": "/path/graphs/LAUNCH_PREFLIGHT.json", "sha256": "<SHA256>"},
      "candidates": {"path": "/path/candidates.parquet", "sha256": "<SHA256>"},
      "seed": 1729
    }

Paths resolve relative to the manifest. ``seed`` is optional, nonnegative and
less than 2**31. The donor table requires exact nonempty strings in
``dataset_id``, ``query_donor_id`` and ``canonical_donor_id``; dataset/native
pairs must be unique and cover every retained mapped cell. Distinct native
aliases may map to one canonical donor. They collapse BEFORE donor counting,
fold assignment and nested source/query/validation planning. The original
fold algorithm and minimum-source rules remain unchanged. At least three
canonical donors are required for the nested plan; three donors produce only
unavailable episodes because each leaves fewer than two source donors.

``CANONICAL_CELL_METADATA.parquet`` retains exact mapped cell row order and
all original columns, stores the original donor ID as ``native_donor_id`` and
uses the declared canonical ID as ``donor_id``. Original mapped metadata is
unchanged. ``DONOR_FOLDS.parquet`` and ``NESTED_PLANS.json`` contain actual
original-algorithm partitions, including support-limited and unavailable
states. ``CANONICAL_DONOR_CROSSWALK.parquet`` maps the resulting canonical
query IDs to themselves for later signed-SC consumption. It must accompany
episodes produced from this canonical metadata; the input crosswalk is for
the original raw donor names. These declarations do not independently prove
biological donor identity or cross-module patient separation.

Existing graph assets must have the original per-cancer STATIC layout,
hash/size receipts, node identity ordering and pathway-family dimensions.
Candidates require unique cancer/lncRNA/pathway keys and exact mapped IDs
for the selected cancer. Graph construction, biological binding authority
and annotation inference remain outside this stage's evidence boundary.

``NATIVE_SINGLE_CELL_DONORS_BINDING.json`` binds the preparation; the stage
also writes ``nested/LAUNCH_PREFLIGHT.json`` for the original
``materialize_single_cell_nested_hhgt_episodes_149`` program. This explicit
native route replaces its dependency on the historical full-28-cancer
metacell closure. Historical invocations still require the old closure.
The native consumer reconstructs canonical cell rows, folds, all nested
plans and the downstream crosswalk before reading prepared inputs; replacing
outputs and recomputing hashes does not satisfy these checks.

``EXECUTION_COMMAND.json`` records the original nested CLI command but never
executes it. Before separately running it, supply a finite positive Unix
deadline in place of ``<UNIX_DEADLINE>``. Geometry/screening parameters remain
the current original CLI defaults, bound through the materializer source
hash; seed comes from the manifest. Changing those settings, selected folds,
source paths or enabling scratch deletion fails native preflight. A different
parameterized geometry workflow is not implemented by this stage. Executing
that recorded command fits PCA; it was NOT executed for estimable episodes
during this code-only task. LASSO assembly and full episode-to-training
ancestry still require further integration.

Verification uses the actual synthetic H5 mapper and static graph fixture,
120 retained cells with ten original donors collapsed to nine canonical
donors, plus a three-donor unavailable case. The actual original nested
runner reaches the geometry call with canonical rows in the first case
and completes all six unavailable episodes in the second. A test sentinel
stops the first case before geometry. PCA, LASSO and optimizer entry points
are disabled. This verifies the preparation/consumer boundary, not numerical
geometry, fitted models, biological annotations or real cancer coverage.

Native single-cell donor verification on 149: ``single_cell_donors_workflow_r3``, ``single_cell_donors_consumers_r1``
pass 87 distinct related checks. Intermediate writer/path failures
are retained in the source-adaptation record. No geometry/PCA, LASSO, neural
optimizer, real cohort, download, deletion or deployment was executed.

Canonical outer-query metacell handoff (2026-10-27)
------------------------------------------------

The same ``single_cell_donors.run`` preparation now also writes
``METACELL_EXECUTION_COMMAND.json`` and ``metacells/LAUNCH_PREFLIGHT.json``.
The recorded command calls the original
``scripts.materialize_single_cell_metacell_149``; it is not executed by the
preparation workflow. Both this outer-query branch and the nested branch now
read the same reconstructed canonical cell table and ``DONOR_FOLDS.parquet``.
Aliases are no longer canonicalized in the nested branch while remaining
separate donors in the outer-query branch. The original raw mapped arrays
and metadata are retained unchanged.

As with the nested command, replace ``<UNIX_DEADLINE>`` with a finite positive
Unix deadline before separate execution. Geometry, screening and maximum
training donors per pair use the original CLI defaults; the seed comes from
the common preparation. Different settings, paths or scratch-deletion flags
are rejected before execution. Running this command normally fits PCA; no
such fitting was performed during this code-only task.

Native fold/cancer receipts record the common donor preparation and launch
preflight. Completed native caches recheck the source preparation, exact
fold set, canonical donor/cell counts, donor-fold table, fold receipts and
file hashes. Fold validation additionally checks ordered cell identities,
source/query membership, target cancer/dataset/fold identity and row counts.
Repinning a query donor into a source fold or changing its cancer is rejected.
Native partial folds are retained with an explicit error, rather than
automatically deleted; use a new preparation output for that retry. Historical
invocations retain their previous path and behavior.

This cache audit checks source and identity consistency. It does not rerun
PCA/grouping or independently recompute every cached statistic, and is not a
claim of complete numerical or biological ancestry. Static/fold graph
production, native graph consumers, complete estimable nested outputs and
LASSO/episode assembly still need integration and acceptance.

The synthetic integration fixture has 1,203 raw cells, 1,200 retained cells,
ten native donor names collapsed to nine canonical donors, and all 2,135
pathway identities. An explicit fixed coordinate projection and fixed
within-donor three-cell groups replace geometry only. The original mapper,
sum/library normalization, UCell scoring, source-only candidate screen,
signed donor-resolved correlations, feature attachment and all five-fold
writers execute. Every fold produces nonempty finite query targets; query
donors match the nested plan, and cache resume performs no geometry calls.
The geometry substitutes do not validate fitted PCA or scientific benefit.

Canonical outer-query verification on 149: ``canonical_metacell_workflow_r2``, ``canonical_metacell_consumers_r1``
pass 53 distinct related checks. The earlier r1 suite also passed;
r2 adds rehashed cancer-identity rejection and stronger cache metadata checks.
Geometry is an explicit frozen test substitute. No PCA/LASSO/neural fitting,
real cohort, download, real-data deletion or deployment was executed.

Native single-cell static graph production (2026-10-27)
-----------------------------------------------------

``single_cell_static.run`` is operation 26. It calls the original global
external-evidence and per-cancer static-graph producers using the explicit
raw-H5 mapping, pathway-family table and external evidence table. This stage
requires neither metacell geometry nor the historical full-28-cancer closure;
it writes actual static assets before canonical donor/nested preparation.

The manifest is::

    {
      "format": "NATIVE_SINGLE_CELL_STATIC_INPUT_V1",
      "mapping": {"path": "/path/NATIVE_SINGLE_CELL_MAPPING_BINDING.json", "sha256": "<SHA256>"},
      "pathway_family": {"path": "/path/pathway_family.parquet", "sha256": "<SHA256>"},
      "external_evidence": {"path": "/path/external.parquet", "sha256": "<SHA256>"}
    }

Run on 149 with
``python -m scripts.run_native_single_cell_static_workflow --manifest INPUT.json --output NEW_DIRECTORY``;
add ``--plan`` to verify inputs without creating outputs. All input paths
resolve relative to the manifest. Family columns follow the original
``normalize_pathway_families`` contract: ``pathway_id``, ``pathway_family_id``,
``is_representative`` and optional ``membership_weight``. Missing families
retain the original explicit unassigned-family category.

External evidence follows the original columns ``lncrna_id``, ``partner_id``,
``partner_type``, ``relation_type``, ``direction``, ``experiment_family``,
``is_experimental``, ``is_predicted``, ``pmid``, ``source_database`` and
``mapping_status``. The two flags must be Parquet boolean columns; string
flags are rejected before producing assets. The original mapped,
experimental, nonpredicted filter remains unchanged. Physical binding is
collapsed to canonical lncRNA/gene pairs for the primary graph; typed
sensitivity relations and schemas remain separate outputs. Reference and
evidence declarations are not independent biological verification.

The output contains ``STATIC_GLOBAL``, ``cancers/<cancer>/STATIC``, launch
preflight and ``NATIVE_SINGLE_CELL_STATIC_BINDING.json``. Exact gene/pathway
orientation, signature metadata, node ordering, family hierarchy, primary
and typed edges, schemas and source bindings are produced by the original
code. Its preflight pin is the ``graph_preflight`` input to
``single_cell_donors.run``. Thus the raw mapping, real static production and
canonical donor preparation form an executable no-fit sequence.

Before publishing the binding, the workflow reconstructs all static outputs
in temporary storage. Validation compares table values/order/dtypes, sparse
matrix shape/dtype/values, receipt metadata and source-binding JSON. Path,
hash, size, inventory and global-static links are also verified. Runtime
timestamps and location-dependent global receipt hashes are compared through
their actual verified bindings rather than assumed byte-identical replays.
A process-local reconstruction cache is reused only after current inputs,
implementation and output bytes have been rechecked.

The original static cache reader and nested static-asset reader follow
``native_static_preparation`` and enforce this reconstruction. Legacy static
assets without that marker retain their existing checks. Tests verify that
repinned membership values, physical edge weights and node order changes
fail at the original consumer, and changed evidence invalidates warm caches.

Verification uses synthetic H5/metadata, all 2,135 pathways, one deliberately
unassigned family and six declared external records. Only three mapped
experimental/nonpredicted rows enter the original global evidence producer;
two physical-binding records collapse to one primary edge. No family,
membership or source-generation gate is replaced. The generated static files
enter actual canonical donor preparation. Original graph-asset helper tests
and mapped/donor/query consumers also run. No PCA, LASSO or neural fitting,
scientific download, real cohort or deployment is executed. Fold-specific
graph production and full geometry/LASSO/episode ancestry remain open.

Native static SC graph verification on 149: ``single_cell_static_workflow_r1``, ``single_cell_static_consumers_r1``
pass 62 distinct related checks. Original static producers and
source reconstruction ran on synthetic inputs. Existing outer-query tests
retain explicit frozen geometry. No PCA/LASSO/neural fitting, real cohort,
scientific download, real-data deletion or deployment was executed.

Canonical fold graph reaggregation (2026-10-27)
---------------------------------------------

``single_cell_graph.run`` is operation 27. It calls the original graph-asset
materializer on completed canonical metacells, reusing frozen cell membership.
It aligns canonical cell identities, reaggregates coding/lncRNA measurements,
recomputes UCell activity and writes source/query LASSO observation masks,
context scopes and static/target bindings. PCA, grouping and LASSO fitting
are not dispatched.

The manifest is::

    {
      "format": "NATIVE_SINGLE_CELL_GRAPH_INPUT_V1",
      "donor_preparation": {"path": "/path/NATIVE_SINGLE_CELL_DONORS_BINDING.json", "sha256": "<SHA256>"},
      "metacell_success": {"path": "/path/metacells/cancers/BRCA/SUCCESS.json", "sha256": "<SHA256>"},
      "static_workflow": {"path": "/path/NATIVE_SINGLE_CELL_STATIC_BINDING.json", "sha256": "<SHA256>"}
    }

Use ``python -m scripts.run_native_single_cell_graph_workflow --manifest INPUT.json --output NEW_DIRECTORY``
on 149; ``--plan`` verifies without writing. Sources must identify the same
raw mapping, static preparation, canonical donors and completed metacell
cancer. All active folds are required. A read-only completion validator
checks source receipts and artifacts; a missing fold cannot trigger geometry
fitting. The original historical full-28-cancer entry remains separate.

Each original fold materializer receives reconstructed canonical metadata,
so frozen canonical memberships no longer fail against original donor aliases.
An additional original-producer check derives every group's donor, cell type,
state, dataset/cancer where supplied, and cell count from its actual member
cells. Mixed donor/context groups and mislabelled aggregate metadata fail
before matrix aggregation. This check also applies to newly produced
historical graph assets.

The output contains original per-cancer STATIC and five-or-fewer active fold
directories, ``cancers/<cancer>/SUCCESS.json``, ``GRAPH_ASSET_MANIFEST.json``
and ``NATIVE_SINGLE_CELL_GRAPH_BINDING.json``. Static assets are regenerated
from the same explicit references in the new graph directory. The completion
envelopes retain original formats and declared fold coverage. All families
from the original many-to-many normalizer are retained; this native path
does not impose the old batch launcher's one-family-edge-per-pathway check.

Validation reconstructs static assets and every fold from the original raw
mapped matrices and frozen memberships in temporary storage. It compares
all sparse matrices, activity arrays, ordered observation/scope tables,
metadata, target bindings and verified static links. Current sources, code,
file inventories, hashes and sizes are checked before process-local proof
cache reuse. Original fold-cache readers and the original query-episode
consumer enforce the native preparation marker. Repinning coding values,
UCell values or query-allowed LASSO flags cannot bypass reconstruction.
The first validation in a process repeats graph production for every active
fold and needs temporary storage for those outputs. It is a computational
audit, not a cheap receipt-only check; unchanged proof-cache hits still
verify current source and output bytes.

The source membership is a supplied completed geometry result; reaggregation
does not independently establish that PCA/grouping were originally fitted
correctly. It also does not independently recompute the original target
statistics or prove biological donor equivalence. Those limits remain in
the source-completeness declaration.

The positive integration uses synthetic H5, actual static production,
canonical donor preparation and five-fold outer target production. Only
the upstream geometry uses the previously documented fixed projection and
three-cell within-donor groups. The graph stage itself uses the original
algorithms without numerical substitutes: 1,200 retained cells become 400
metacells, nine canonical donors, five coding genes and all 2,135 pathways
in each fold. The original query assembler loads generated static data,
observations and targets and computes source-context features for all five
folds; an explicit test boundary replaces episode fitting/serialization.
Full native episode launch/finalization and LASSO result ancestry remain open.

Native fold graph verification on 149: ``single_cell_graph_workflow_r1``, ``single_cell_graph_consumers_r1``
pass 68 distinct related checks. Synthetic upstream geometry uses
explicit frozen substitutes; graph reaggregation and numerical reconstruction
are original code. No PCA/LASSO/neural fitting, real cohort, scientific download,
real-data deletion or deployment was executed.

Native episode launch and finalization (2026-10-27)
-------------------------------------------------

The original ``scripts.materialize_single_cell_hhgt_training_episodes_149``
now accepts ``--native-graph-binding`` and ``--prepare-only``. This provides
an explicit path from the native canonical graph and completed nested assets
to the original episode assembler. Without the native option, the historical
full-28-cancer materialization prerequisite is unchanged.

On host 149, source-only preparation is::

    python -m scripts.materialize_single_cell_hhgt_training_episodes_149 \
      --run-id NEW_RUN_ID \
      --source-result-root /path/donor-preparation/metacells \
      --graph-asset-root /path/native-fold-graphs \
      --nested-asset-root /path/donor-preparation/nested \
      --native-graph-binding /path/native-fold-graphs/NATIVE_SINGLE_CELL_GRAPH_BINDING.json \
      --output-root /path/new-episodes \
      --cancer-id BRCA --deadline UNIX_DEADLINE --prepare-only

The command validates actual completed inputs and writes only
``LAUNCH_PREFLIGHT.json``. It does not build a model, call LASSO, serialize
episodes or launch geometry. Missing completed nested partitions cause
failure rather than triggering a producer. The graph binding determines the
authorized cancer and canonical donor preparation; both source roots must
match that preparation. The output must be separate from source directories.

Removing ``--prepare-only`` explicitly invokes the original assembler,
including its LASSO fitting step. That command was not executed with a real
fitter in this code-completion task. After a separately executed assembly,
use the same arguments with ``--finalize`` instead of ``--cancer-id BRCA``.
Finalization retains all five folds and the requirement for fit, validation
and query episodes in every fold. The command does not relax sparse-donor
or missing-validation states into a training-ready manifest.

Native preflight verifies the canonical nested plans and their exact outer
and inner receipt coverage, complete output inventories, donor partitions,
retained cell identities, group donor/context/counts, source-only LASSO masks,
target identities and matrix column ordering. Nested and query static node
orders must agree despite residing in separate output directories. It pins
all nested artifacts and implementation bytes; changing them after preparation
invalidates resume. Geometry audit declarations are checked against the plan,
but this does not replay the fitted geometry, nested numeric matrices,
target statistics or LASSO coefficients. Source completeness remains false.

Native episode payloads, their sidecars and cancer completion receipts bind
the episode preflight. Cached native payloads receive content-hash checks,
and native finalization also verifies sidecar and static-asset bindings.
Legacy receipt-only performance behavior remains unchanged.

The positive integration uses actual H5 mapping, static/outer graph production,
canonical donors and the original nested producer, episode serialization,
finalization and both-arm training-input collection. Upstream PCA/grouping
uses fixed projection/manual groups; the LASSO fitter alone returns explicit
frozen synthetic coefficients. No geometry/LASSO fitting, neural model,
optimizer, real cohort, paid compute, data download, deletion or deployment
is dispatched. This verifies command and serialization interoperability;
fitted numerical ancestry and the remaining C01/C13/C16 acceptance are open.

Native episode launch verification on 149: ``single_cell_episode_sources_r1``, ``single_cell_episode_consumers_r1``
pass 63 distinct related checks. PCA/grouping and LASSO use explicit
frozen synthetic substitutes; original nested statistics, serialization,
finalization and both-arm graph input collection execute without a neural model.

Native episode workflow handoff (2026-10-27)
------------------------------------------

``single_cell_episodes.prepare`` and ``single_cell_episodes.collect`` extend
the closed native registry to 29 operations. They wrap the original native
episode assembler without dispatching its LASSO fitting command. Preparation
consumes completed canonical graph and nested assets, not unexecuted geometry
commands. Its manifest is::

    {
      "format": "NATIVE_SINGLE_CELL_EPISODES_INPUT_V1",
      "run_id": "NEW_EPISODE_RUN",
      "graph_workflow": {
        "path": "/path/NATIVE_SINGLE_CELL_GRAPH_BINDING.json",
        "sha256": "<SHA256>"
      },
      "lasso_config": {"alphas": [0.3, 0.1, 0.03, 0.01], "stability_threshold": 0.5}
    }

``lasso_config`` is optional. Allowed overrides are ``alphas``, ``max_genes``,
``min_donors``, ``inner_folds``, ``bootstraps``, ``response_batch_size``,
``max_iter``, ``tolerance`` and ``stability_threshold``. Unknown parameters,
Boolean/fractional integer dimensions and non-finite scalars are rejected;
the original LASSO policy validator retains its scientific bounds. Arguments
are parsed by the original CLI, not interpreted as shell text.

On 149, run::

    python -m scripts.run_native_single_cell_episodes_workflow prepare \
      --manifest INPUT.json --output NEW_PREPARATION

``--plan`` verifies sources and configuration without creating the output.
Preparation calls the original assembler with ``--prepare-only``. It writes
``PREPARED.json``, ``EXECUTION_COMMANDS.json`` and the original launch
preflight under ``episodes/``. The command document contains separate
``assemble`` and ``finalize`` argument arrays, with an explicit
``<UNIX_DEADLINE>`` placeholder. Only assemble calls LASSO; neither command is
automatically executed. No neural fitting command is dispatched.

After separately completing assembly and finalization in that exact
``episodes/`` directory, collect the pinned manifest::

    python -m scripts.run_native_single_cell_episodes_workflow collect \
      --preparation NEW_PREPARATION/PREPARED.json --preparation-sha256 SHA256 \
      --completed NEW_PREPARATION/episodes/SC_HHGT_EPISODE_MANIFEST.json \
      --completed-sha256 SHA256 --output NEW_COLLECTION

Collection repeats the original finalization in read-only mode. It verifies
the preparation, sources, policy, cancer/fold/role counts, ordered episode
inventory and static/sidecar bindings, then loads every actual episode graph
for both training arms without constructing a neural model. Read-only
preflight/finalization cannot create missing manifests or perform compatibility
migrations. Output collection must be separate from the immutable preparation,
canonical donor preparation and graph source directories.

``NATIVE_SINGLE_CELL_EPISODES_BINDING.json`` binds the completed manifest,
``EPISODE_INPUT_PREFLIGHT.json`` and source inventory. Its ``files`` field
provides the four inputs for ``single_cell.prepare``: ``episode_manifest``,
``graph_asset_manifest``, ``candidates`` and ``donor_crosswalk``. Pass the
collection binding as that next stage's optional ``episode_workflow`` field;
the actual SC workflow revalidates it, requires all four inputs to match,
and retains its source inventory in subsequent preparation records.

In a chain, use an external reference for the completed episode manifest.
The coordinator stops at ``WAITING_EXTERNAL`` until the exact completion is
registered, then performs collect and downstream SC input preparation.
Stage references to ``/files/<name>`` select the four input pins. Resuming
revalidates the native results rather than dispatching fitting or rewriting
the immutable episode preparation.

These checks establish serialization, provenance binding and actual graph
input interoperability. The subsequent frozen-group reconstruction described
below checks nested numeric targets; fitted PCA/grouping and LASSO coefficients
still lack independent verification. The
collection explicitly retains ``source_ancestry_complete=false`` and
``fitted_lasso_numerics_verified=false``. The workflow has one declared cancer
from the graph preparation and preserves the original five-fold role gates.

The synthetic positive chain uses actual mapping, static/outer/nested
producers and original assembly/finalization. Geometry and LASSO outputs
remain explicit frozen test substitutes. Original SC input collection runs
without fitting; separate existing SC/fusion regression fixtures may construct
frozen test models and replay forwards, with optimizers forbidden. No real
cohort, live data acquisition, paid compute, deletion or deployment is run.

The H5 mapping/revalidation path imports ``h5py``. Its previously missing
dependency is now declared consistently as ``h5py>=3.16`` in project metadata
and base requirements. The existing 149 verification environment uses 3.16.0
with Python >=3.10 support. A separate metadata receipt checks declaration
consistency against that environment; it does not claim a fresh installation.

Native episode workflow-chain verification on 149: ``single_cell_episodes_workflow_r1``, ``single_cell_episodes_sources_regression_r2``, ``single_cell_episodes_workflow_regression_r2``
pass 78 distinct related checks. The actual native chain reaches
SC input preparation and resumes without fitting; frozen substitutes and
separate frozen-model regression scope are described above.

Nested measured-value reconstruction (2026-10-27)
------------------------------------------------

The scoped field dictionary is
``docs/NESTED_SINGLE_CELL_FIELDS_20261027.rst``. It defines target/source
statistics, feature weighting and shapes, state pooling, unavailable states
and the distinction between code consistency and biological validation.

Native episode preflight now reconstructs nested statistics from the pinned
mapped measurements, canonical donor metadata, supplied frozen cell groups,
bulk candidates and static pathway membership. It uses the original production
statistical functions and compares exact values, dtypes, row order and keys for
coding/lncRNA aggregates, UCell activity, LASSO observation/source masks,
candidate universe, signed donor targets and source effects, source node
features, context metadata/coverage and receipt summaries. Static identity
bindings and unavailable donor partitions are checked as well.

The original nested producer applies the same check before returning a full
native cache, without rewriting receipts or timestamps. A partially completed
cancer must also verify each existing plan before reuse. Failure raises an
error rather than falling back to geometry fitting. Legacy historical paths
retain their existing behavior.

Successful per-plan reconstructions can be reused within the same process
only after checking current source, implementation and output bytes. Inputs
and outputs are checked again after reconstruction before caching the proof.
The episode preflight records ``nested_frozen_statistics_replayed=true`` and
the reconstruction implementation pins; these pins propagate through episode
collection to downstream SC input preparation. It retains
``nested_fitted_numerics_replayed=false`` and
``fitted_geometry_independently_verified=false`` where reported: supplied
membership is not independently proved to be the output of the declared PCA
and radius fitting. This also does not reconstruct fitted LASSO coefficients
or establish raw biological identity. C01/C13/C16 remain open.

Verification uses synthetic cells and references with explicitly frozen
geometry. Intentional changes to aggregate values, UCell, target/source rho,
candidate flags, node features, context coverage and receipt counts must fail
even after all affected receipt hashes are recomputed. No real cohort,
geometry fitting, LASSO, neural training, acquisition, paid compute or
deployment is dispatched by this verification.

Nested numeric reconstruction verification on 149: ``single_cell_nested_numeric_r1``, ``single_cell_nested_numeric_sources_r1``, ``single_cell_nested_numeric_chain_r2``
pass 27 distinct related checks. Fixed grouping and LASSO
substitutes are disclosed above; these checks do not establish fitted geometry.

Native SC reconstruction in model-use inspection (2026-10-27)
------------------------------------------------------------

``native_single_cell_model_sources.inspect`` connects completed native SC
episode sources to the actual ``current_single_cell_model_use`` inspector.
It recognizes the hash-bound native launch preflight in the consumed episode
manifest, requires matching graph/source roots, and repeats original read-only
finalization. A native graph cannot fall back to generic legacy inspection
after its episode marker is removed.

The native validators reconstruct H5 mapping, static graphs, fold graphs and
frozen-group nested statistics. Their implementation pins and raw input files
enter the model dependency inventory. ``transformation_replays`` records five
separate scopes: mapping, static graph, fold graph, nested statistics and
episode-manifest reconstruction. The last scope checks manifest coverage and
serialization bindings; it does not verify fitted LASSO coefficients.

Only the exact path/hash identities handled by these native checks bypass
the generic unknown-array/table transformation warning. Raw supplied files
are recorded as source inputs, with their own acquisition/QC/reference gaps.
JSON source links continue to be traversed. Legacy inputs retain the previous
generic inspection. Unknown provenance is not made complete by renaming or
hashing a file, and public source roles use structural indices rather than
raw donor identifiers.

Explicit unresolved reasons include raw acquisition/QC/reference ancestry,
PCA/grouping correctness, LASSO coefficient reconstruction, bulk-candidate
and external-reference ancestry, and cross-module patient closure. These
remain blocking. No independence attestation or positive real-cohort
external-validation result is inferred from the new reconstruction rows.
At this stage the nested statistic proof covered fit/validation targets;
outer-query numerical reconstruction was still open. The following outer
reconstruction section records the subsequent implementation and its scope.

The integration fixture uses actual native H5/static/donor/outer/nested/episode
production with explicit fixed grouping and LASSO outputs. It then surrounds
those actual episode files with frozen checkpoint/prediction/telemetry
containers. Native model initialization occurs only when building those
synthetic containers. During source inspection, initialization, neural
evaluation and all fitting are forbidden. This verifies source consumption
and provenance records, not neural prediction accuracy or optimizer history.

Native SC model-use source verification on 149: ``single_cell_model_sources_r1``, ``single_cell_model_sources_regression_r1``, ``single_cell_model_sources_core_r1``
pass 51 distinct related checks. Checkpoint containers are explicit
synthetic fixtures; no fitting or independent biological validation was run.

Original outer single-cell target reconstruction (2026-10-27)
------------------------------------------------------------

``single_cell_outer_numeric_replay`` closes the separate outer train/query
numeric-source gap. The original producer's context statistics now share a
deterministic helper with read-only reconstruction; production progress is
an optional callback. The underlying screening, donor targets, source
features and train-row selection retain their original algorithms.

On complete or partial native fold reuse, the checker validates the exact
output inventory, frozen cell/group identities and current measured inputs.
It reaggregates coding/lncRNA values with the original measurement scale and
library, recalculates UCell, and reconstructs the candidate universe plus
every train/query column and row order. Context coverage and summary counts
are compared. Successful proofs are cached by actual input/output/code
hashes and scientific settings; cache hits still recheck current file bytes.
The supplied grouping and geometry-fit declarations are not independent
proof that the original PCA/radius fit was correct.

The native fold-graph validator already calls the original completed-cancer
gate. It now provides a finite current reconstruction deadline and pins the
new implementation. The actual signed model-use inspector records a sixth
scope, ``NATIVE_SC_OUTER_FROZEN_STATISTICS_REPLAY_V1``, and binds the exact
verified outer source tables. Only those identities bypass generic unknown
transformation gaps. Fitted geometry/LASSO, raw acquisition/QC/reference,
candidate ancestry and cross-module patient completeness remain unresolved.

Verification uses synthetic measured cells with fixed geometry. It includes
independent Spearman/Fisher and leave-one-donor-out reference values, query
perturbation isolation, and rehashed target/source/candidate/count changes.
The native model-use integration uses frozen checkpoint/prediction containers
without construction/forward/fitting during inspection. No real-cohort
analysis, neural/LASSO/PCA fitting, download, paid compute or deployment runs.
The separate existing core model-use regression executes frozen forwards;
this is distinct from the SC source inspector's no-forward scope.

Outer SC reconstruction verification on 149: ``single_cell_outer_graph_regression_r2``, ``single_cell_outer_numeric_r2``, ``single_cell_outer_model_sources_r2``, ``single_cell_outer_core_regression_r1``
pass 67 distinct related checks. No fitting or real-data analysis ran.

Local-CNV statistical patient scopes (2026-10-27)
----------------------------------------------

The actual native local-CNV numerical reconstruction now captures the
patient populations used by its statistical functions. Optional observer
hooks preserve the existing calculations and return values. They record
train/validation A0 cohorts, shared covariate-design populations, exact
A0-matched/A1 masks, the separate A1-on-A2/A2 masks, validation A1 and
graph callability/variance/tested Cartesian scopes. Missing-feature matched
calculations retain an explicit empty scope. Eligibility is distinct from a
finite/estimable association. No test/OOF patient enters these scopes.

``local_cnv_patient_scopes_private`` stores shared patient-map row sets and
aligned pair lists or Cartesian feature sets. It binds the actual consumed
partition, source receipt, model/context, patient map and implementation.
The data represent statistical construction of the consumed partition, not
proof of per-patient optimizer updates. Original broad input exposure stays
present because normalization and shared covariate construction can depend
on more patients than one matched pair.

The source inspector follows the actual patient-map rows through the
explicit file-qualified identity crosswalk. Native canonicalization does
not invent biological identities; unresolved aliases stay unresolved. A
numeric-only cached proof cannot stand in for a missing patient trace, and
captured trace reuse still checks current input/output/code bytes and the
validation closeout. Traces are attributed separately to each model context.

Normal model-use serialization and reconstruction include the new private
table. Its private flag is mandatory, and current external evaluation does
not publish it. Existing public mapping/overlap summaries remain available.
The local statistical row-scope gap is replaced only where actual numeric
reconstruction succeeds; historical unsupported receipts retain it. Earlier
raw ancestry, cross-module identity, State-core dependencies and final
external acceptance remain open. See
``docs/LOCAL_CNV_PATIENT_SCOPES_20261027.rst`` for field semantics.

Synthetic native workers and independent closeout feed the scope tests;
direct mask/neutral-zero/missingness checks compare observer and original
numerical outputs. Alias, cache, privacy and existing statistical/core/
external consumers are verified. Existing consumer fixtures use frozen
forwards and explicit frozen containers. No learned-model fitting, real
cohort analysis, acquisition, paid compute or production deployment ran.

Local-CNV patient-scope verification on 149: ``local_cnv_patient_scopes_r2``, ``local_cnv_patient_consumers_r1``, ``local_cnv_patient_statistics_r1``
pass 132 distinct related checks. No fitting or real-data analysis ran.

State statistical patient scopes on 149: ``state_patient_scopes_r1``, ``state_patient_consumers_r1``
pass 90 distinct related checks. Private scopes bind exact paired patients and frozen core parents;
parent-core training ancestry and full cross-module coverage remain open. No fitting or real-cohort analysis ran.

State-core input-source verification on149: ``state_core_sources_r2``, ``state_core_consumers_r2``
pass 100 distinct related checks. Actual parent training sources and explicit
patient mappings are connected; complete ancestry and all-module acceptance remain open.
Native CSV/TSV identities preserve leading zeros. No fitting or real cohort analysis ran.

GENCODE native annotation verification on149: ``annotation_sources_r1``, ``annotation_consumers_r1``, ``annotation_model_use_r1``
pass 97 distinct related checks. Raw GTF feeds actual bulk candidate/activity
consumers and model-use reconstruction; annotation.run is chain operation 30.
Reference authenticity and all-module acceptance remain open. No fitting or real cohorts ran.

Original single-cell GTF annotation verification on149: ``sc_annotation_sources_r2``, ``sc_annotation_model_r1``, ``sc_annotation_consumers_r1``
pass 79 distinct related checks. Actual H5 mapping and completed-model
input inspection follow the original annotation producer to GTF; chain operation 31.
Reference study exposure is separate from the H5 cohort. Raw/QC/geometry/LASSO
and whole-project acceptance remain open; no fitting or real-data analysis ran.

GENCODE interval-to-CNV verification on149: ``genomic_annotation_consumers_r1``, ``genomic_annotation_workflows_r1``, ``genomic_annotation_sources_r3``
pass 130 distinct related checks. Original coordinates now feed both signed
and compact genomic CNV, including their actual source inspectors; chain operation 32.
Segment reference declarations do not establish alignment veracity. Missing membership
coordinates remain unavailable, not zero. No fitting or real-data analysis ran.

GMT/GENCODE native CNV membership verification on149: ``pathway_membership_sources_r1``, ``pathway_membership_consumers_r1``
pass 133 distinct related checks. Both original CNV CLIs now consume
reconstructed native GMT/GENCODE membership and trace it to original resources.
Unknown source symbols remain unavailable; original many-to-many gene estimands
remain unchanged. Registry operation 33 is exercised in an immutable resumable chain.
Resource authenticity and other membership consumers remain open. No fitting ran.

Native ATAC membership guarded verification on149: ``atac_membership_sources_r2``, ``atac_membership_consumers_r2``, ``atac_membership_guard_r1``
pass 80 distinct related checks. Original R, five-fold preparation,
auxiliary collection and actual current fusion source inspection consume reconstructed
GMT/GENCODE membership. Existing observed-gene means and coverage are preserved.
Initial regression selection accidentally executed five synthetic logistic fits; that
interrupted run is excluded, its receipt corrected and evidence retained. Final runs
block fitting at session scope. No real cohort or paid compute ran.

Original transcript promoter guarded verification on149: ``promoter_atac_consumers_r1``, ``promoter_probe_consumers_r1``, ``promoter_sources_r2``
pass 72 distinct related checks. The recovered original producer
now feeds native ATAC and probe/methylation workflows, retaining transcript geometry
and original IDs while preserving stable PAR_Y identities downstream. Operation 34
and actual source inspectors are verified; reference authenticity remains unresolved.
No fitting or real cohort execution ran in this change; earlier incidents stay recorded.

State measurement contribution guarded verification on149: ``state_measurement_consumers_r1``, ``state_measurement_sources_r2``
pass 77 distinct related checks. Actual raw aliquot rows and native
within-patient means now link to each original paired statistical mask in two
private replayed tables. Supplied-measurement ancestry is not acquisition proof.
No fitting or real cohort execution ran; earlier scope deviations remain recorded.

SC frozen geometry guarded verification on149: ``sc_geometry_consumers_r1``, ``sc_geometry_sources_r3``
pass 78 distinct related checks. Original outer and nested producers
now preserve projection parameters and coordinates; original source-only radius
and disjoint membership are replayed through native graph/episode/model consumers.
Fitted PCA/LASSO provenance remains unresolved. No fitting or real cohort ran.

SC LASSO recorded-evidence replay on149: ``sc_lasso_model_sources_r3``
pass 25 distinct related checks. Signed model-use inspection now
re-derives the candidate nodes, LASSO pair features and sparse edges from the
evidence recorded inside each episode plus the pinned static assets. Recorded
coefficients are still not re-solved, so fit provenance and objective
consistency remain open. No fitting or real cohort ran.

Code acceptance harness on149: ``code_acceptance_r2``
pass 21 distinct related checks. The harness replays the requirement
ledger against recorded receipts and fails closed on missing evidence, drifted
pins or suites, counter disagreement, status/open-list mismatch and a dirty
worktree. The handoff report cannot be rebuilt without a PASS record bound to
this plan, ledger and commit. Bookkeeping only; no optimizer or real cohort ran.

Multi-omics node encoding on149: ``omics_node_encoding_r2``, ``code_acceptance_r4``
pass 86 distinct related checks. Measured mutation, signed copy
number and promoter accessibility now enter the encoder as value/observed column pairs; a
fixture shows a gene-level value reaching a connected lncRNA embedding through message
passing. C18-C21 (comparison, patient conditioning, mediation, discovery) are registered
as open. No optimizer or real cohort ran and no gain, patient result, mediation or
discovery claim is made.

Patient-level intervention on149: ``patient_intervention_r4``
pass 89 distinct related checks. A declared patient-level contrast
changes one omics channel of one node with every other input fixed, and the change in each
lncRNA-pathway logit is reported with sign and rank movement. Interaction attribution adds
a rewiring control and a matched placebo node; the record states that biological mediation
is not proven. The fixture model is untrained and no patient result exists.

Paired multi-omics comparison on149: ``omics_ablation_r2``
pass 31 distinct related checks. Two arms must share candidate, label,
fold, budget and seed identity and differ only in the omics node input. The estimator reports
per-cancer average precision with a paired stratified bootstrap, an exact sign test and a
sign-flip permutation test. The reporting gate refuses while either arm is untrained, which is
the current state, so no effect is reported.

Unvalidated binding candidates on149: ``novel_binding_r2``
pass 27 distinct related checks. Known edges are split by a seeded
hash into scored and held-out partitions, the candidate universe excludes every scored
known edge, and an unfitted declared score is evaluated at a fixed budget against a
degree-prior baseline. Every emitted row is an unvalidated candidate. The fixture plants
a latent rule; no real binding evidence was scored.

Encoder regression on149: ``gnn_regression_b``, ``gnn_regression_c``, ``gnn_regression_d``, ``stale_contract_repair_r1``
pass 202 distinct related checks across the graph-contract, fold-local,
model-use, full-model-contract and core-export/current-model suites. The widened node input does
not change behaviour for bundles without omics. Two pre-existing defects were found: one stale
test missing the production topology channel (repaired) and one test depending on a document
absent from this checkout (now skips with a reason). No requirement was promoted.

Public field definitions on149: ``public_fields_r1``, ``public_fields_wiring_r2``
pass 39 distinct related checks. A registry now carries one checkable
sentence, kind, unit and producing stage per public column, with component clarifications where a
name changes meaning. The catalog attaches definitions, reports defined/total coverage and may
only claim completeness at exactly 100%; pending entries never count. 35 columns and 5
clarifications are authored, no archived catalog was found read-only, and C16 stays open.

Measured node inputs on149 (2026-10-29)
-------------------------------------
The DEEPSEEK continuation adds an opt-in promoter methylation beta channel while retaining the historical three-channel default. A host149-only producer consumes hash-bound measured sample/feature values, explicit node mappings and patient/split authority. It preserves signed CNV and measured zero versus missing values, aggregates observed features per sample, samples per patient and training patients per cohort, and uses declared identity scaling without fitting. Formal training and frozen-core export share the source-replayed node-input loader after binding expansion. A real synthetic native prepared payload passes its original validator and frozen residual-model export/forward with the new inputs. Runtime intervention contexts now also preserve the registered all-chunk exported gene and lncRNA vectors: independent methylation and CNV perturbations change the corresponding final vectors while parameters stay fixed, with exact restoration. The contexts update stored blocks as well as tensors and restore both on normal exit or exceptions. These are synthetic checks of native completed methylation bindings and native signed CNV patient matrices as well: the former uses the original raw/recovery producer, and the latter an explicitly frozen synthetic native-format matrix. Both reach actual prepared validation and exported lncRNA embeddings without duplicating patient values by aliquot count. These remain code checks, not trained multi-omics results. Global pooled inputs are not patient-specific embeddings. Raw assay processing, promoter-map correctness, biological/fold identity authenticity, full new-source model-use traversal, automatic native workflow declaration generation and public field completeness remain unresolved. No real data, downloads, optimizer, paid compute or deployment was started.

Validation: node_omics_native_r2, node_omics_embedding_r1, node_omics_products_r2; 209 passing cases.
