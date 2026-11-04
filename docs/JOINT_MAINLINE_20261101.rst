Joint Pathway/State mainline
===========================

Current target status (2026-11-04): MOGSA and its derived lavaan parent
targets are retired. The public configuration requires a replacement design;
preparation and training do not automatically fall back to RNA activity_score.
See PATHWAY_TARGET_POLICY_20261104.rst. State remains
CLA_STATE_ROUTED_PROGRAMS_20261101_V2 with 482 primary targets plus ESTIMATEScore:
the 198-dimension base and 285 existing PROGENY/DoRothEA programs are unchanged.


The executable primary workflow is cla.py with config/149.json. This integration
reconciles public commit480ada4 with the later R6.2 executor and its September16
adjustment-validator revision. The older frozen State training entry is removed.
The shared PyG encoder keeps the October29 variable-width node inputs and the
November1 regulatory-network adapters. There is no second vendored cc_hhgt tree.

Model and target contract
-------------------------

Pathway loss has weight1.0 and State loss weight0.5. State's decoder consumes
the live shared encoder representation. It is not detached or loaded from an
exported embedding table. The zero-initialized residual output is preserved;
membership-gradient checks include the required initial warm-up step rather
than changing model initialization just to produce a nonzero first gradient.

The R6.2 execution path uses weighted binary cross-entropy, its direction term
and bounded residual regularization. The older nnPU helper is not its loss.
Pathway positives satisfy abs(r)>=0.20/q<=0.05, or the weaker
abs(r)>=0.15/q<=0.10 rule with reduced sample weight. State positives satisfy
abs(r)>=0.20 without an extra FDR requirement. A zero is a statistical proxy
negative, not an experimentally established absence of association.

The base expanded State catalog is CLA_STATE_EXPANDED_20261101_V1, with 197 primary
targets plus auxiliary ESTIMATEScore. Purity and ploidy are primary targets in
this revision. RNA signatures, composition estimates and genomic measurements
retain separate source definitions and families. See STATE_EXPANSION_20261101.rst
and config/state_catalog_20261101.json for the complete ordered catalog.

The inherited September16 covariate policy allows an explicitly present but
sparsely measured purity column to be omitted when source-patient coverage is
below its threshold. The audit names this omission; it does not manufacture
purity values. This integration fixes an implementation fall-through that had
still median-imputed that omitted column. Missing required columns (including
an entirely absent purity column where required) and inadequate coverage of
other required covariates remain admission failures. No threshold is relaxed.

Single-cell association scores use the independent cell-type fusion expert
(sc-celltype-table / sc-celltype-fusion-prepare / sc-celltype-fusion-train).
The September16 executor explicitly rejects their former primary input_context
route. This integration corrects the stale template to
singlecell_association_role=disabled and an empty primary cell-type list, so
the default template no longer immediately fails SC_PRIMARY_INPUT_RETIRED.
The separate expert and interpretation consumers remain available. This does
not introduce their association scores into either primary label or encoder.

New input integration
---------------------

Gene CNV and promoter methylation must be supplied from their verified original
patient matrices. Joint training uses multiple inner views: only each view's
source patients may contribute to its node inputs. Previously accepted native
outer-fold input aggregates cannot be substituted, because they can include
patients reserved for the inner label view. Missingness remains separate from a
measured zero, and canonical gene identity determines alignment.

The public config enables gene_node_omics with channels [cnv_signed,
methylation_promoter_beta]. Its complete.path is a deployment placeholder;
complete.sha256 binds the verified October29 patient_gene_matrices_r2 product
(21b1d23b90a62c7edefe477fbe4957182c1e17b739ce3a2d659208424a7748ce).
Set the path to that verified COMPLETE.json before preparation. Missing or
changed products fail admission; they do not silently disable the channels.

Every view adds four columns (CNV value/mask, promoter-beta value/mask) to its
degree/constant inputs. The gene rows contain source-patient means; other node
types have an explicit zero-valued, unavailable four-column block. SPEC.json
binds this six-column architecture, and each view's NODE_OMICS_AUDIT.json binds
patient/axis hashes and assay coverage. Raw values enter the trainable encoder;
they become learned embeddings only after actual joint optimization. A prepared
input or a forward check alone is not a trained embedding release.

An explicitly disabled channel block remains available for controlled ablation
and synthetic protocol tests. It is not the checked-in formal default and is
architecture-incompatible with six-column checkpoints.

The regulatory-network readers remain available for explicit experimental
graphs. The current cross-source study/sample split is incomplete. Their graph
availability does not authorize pooling them into the primary association model
or calling its labels causal effects.

Execution and artifact validity
-------------------------------

All scientific commands require host149. CLI command preflights record the
target host before execution. Off-host orchestration commands are disabled.
Source hashing includes the unified cla_engine, cc_hhgt, scripts and config
trees. Source signatures and model specifications prevent an earlier checkpoint
from silently resuming with this changed code or different input width.

Only a newly verified prepared capsule may enter the joint worker. This source
integration does not relabel old State predictions, retrain the full cohort,
open sealed test labels or deploy a new website/model release. Source and test
receipts are recorded in provenance/joint_mainline_20261101/.

Downstream execution
--------------------

Clinical, genomic, single-cell, drug and evidence consumers validate the joint
R6.2 core-export contract before loading vectors. Gene vectors are exported
from the actual encoder alongside lncRNA/pathway/cancer vectors; the drug
consumer refuses a manifest lacking gene exports. Downstream private heads
remain separate from the jointly optimized primary State head.

The Evidence adapter now materializes three distinct splits: the outer fold
for evaluation, the next fold for validation, and the remaining folds for
training. It removes overlap by PMID, dataset accession, source event and
source record across those splits. Complete study-level supervision is
aggregated before truncating model tokens. The memory-bounded fitter receives
only train and validation factories; the separate V2 prediction command opens
evaluation after fitting. Existing October schema, assay and source-integrity
checks remain in this path.

Evidence resume restores actual step checkpoints and accepts completed folds
only after their input, configuration and output bindings pass. checkpoint_steps
is consumed by the streaming fitter. Resume keeps the original absolute budget;
it cannot restart the deadline. The module controller also enforces its
persistent wall-clock deadline. This replaces the old adapter that passed
evaluation examples into model selection and resumed only completed folds.

Formal execution uses prepare-formal with a verified diagnostic run, then the
local run-queue on host149. The queue binds all 15 G0/G1/G2-by-patient-fold jobs
and keeps its original deadline on resume. A manually written lease file or
an --allow-formal flag cannot substitute for signed diagnostic admission.

State publication
-----------------

The removed frozen State entry is historical functionality. It has no supported
training command or private State head in this mainline. Clinical, Drug and
other downstream consumers may still freeze a verified joint core for their
own independent outcomes; that policy does not apply to primary State training.

State publication uses the new adapter::

    python -m scripts.materialize_v32_joint_state_context --manifest JOINT_STATE_INPUT.json --output NEW_OUTPUT

Its CURRENT_JOINT_STATE_CONTEXT_INPUT_V1 input contains exactly format,
release_id, selected_analysis and folds. selected_analysis is a path/sha256
record for the selected Pathway analysis. folds contains five distinct entries,
patient_fold_id 0 through 4, each with path/sha256 records for prepared_manifest
(MANIFEST.json), training_receipt (TRAINING_RECEIPT.json), and evaluation_receipt
(EVALUATION_RECEIPT.json). The adapter checks formal G2 completion, the shared
gradient evidence, weights 1.0/0.5, signed outer evaluation and the formal gate.

It writes joint State OOF tables, selected-member State profiles and descriptive
State GeneSets. Keys unavailable in every fold remain explicitly unavailable;
they are not silently removed or imputed. These descriptive GeneSets are not
an independently validated biological release. An old frozen State release
cannot satisfy this manifest by changing its label or file name.

Reconciled downstream semantics
------------------------------

Later R6.2 source checks are retained in the shared downstream modules. A
computational prediction cannot become physical evidence; ceRNA/sponge records
cannot create a direct pathway edge without the required mediator context; and
conflicting identifier aliases are rejected. Sample-level mutation sentinels
do not establish locus-level wild type, duplicate aggregate rows are not summed
twice, and burden/CNV inputs reject invalid non-finite values.

Single-cell features retain their cell/state context, require explicit metadata
and adjusted significance, and do not replace FDR with a raw p-value. Empty
validation sets fail before optimizer creation. ATAC results without outer-test
patients stay unavailable, and unverified distal topology is rejected at run
time. Clinical endpoint and covariate conflicts fail closed; ExtraEndpoints-only
patients remain in the universe, and entity-specific missing measurements stay
missing throughout residualization and survival statistics.

Current RPPA source (2026-11-04)
------------------------------

RPPA500 replaces the earlier 198-marker source for new measured-input
materialization. See RPPA500_INPUTS_20261104.rst for exact source hashes,
sample-vial quality exclusions, available features and preflight binding.
Protein/phosphorylation prediction heads remain a separate implementation;
this input replacement does not change Y, State weights or MOGSA retirement.
