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
separate relation artifact does not alter the currently trained C graph; C11
must admit it only in the declared predicted-expansion comparison arm.

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
