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
effects are not sums of target effects. Action type, affinity, chemical structure
and dose remain outside the current head and are unfinished C08 work.

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

* reference_exact_model: model_id, run_id, producer_kind=exact_release,
  producer_binding={path, sha256}, and
  score_column=association_membership_probability. This supplies the complete
  candidate universe, exact pathway/family, shared/local scope and primary model
  direction. Candidate keys and annotations are checked against the actual five
  fold files and bound candidate authority; merely rehashing changed annotations
  does not satisfy that check.
* selected_model: the same producer declaration shape. exact_release may select
  association_membership_probability, l1_probability or ridge_probability from
  the same exact producer. fusion_release may select
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
has been performed. C05 still needs the native current-V4 full-candidate producer
and remaining auxiliary adapters; C01/C08/C13/C15/C16 also remain unfinished.
