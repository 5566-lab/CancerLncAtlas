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
