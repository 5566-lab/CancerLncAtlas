Native single-cell fields and verification boundaries
====================================================

This dictionary describes the current native nested/outer producers and their
measured-value reconstruction. It is a code contract, not a record of a new
real-cohort analysis. All execution belongs on host 149. The verification
fixtures use synthetic inputs and explicitly frozen grouping/LASSO outputs.

Identity, partitions and observation units
-----------------------------------------

``DONOR_PARTITION.json`` binds cancer, outer fold, inner target fold, inner
validation fold, episode role and the exact canonical donor lists. Donor
aliases are collapsed by the supplied dataset-qualified crosswalk before
fold assignment. This proves consistency with that crosswalk, not independent
biological identity. Nested episodes have ``fit`` or ``validation`` roles;
outer-query episodes are assembled from their separate outer-fold assets.

``METACELL_MEMBERSHIP.parquet`` has one row per retained source or inner-target
cell, including unassigned cells with ``metacell_id=-1``. Cell ID, canonical
donor, cell type and state must match the mapped measurements. Each assigned
group must contain one donor/type/state combination. ``observation_role``
identifies its donor role; ``geometry_fit_allowed`` is true for source cells.
That flag states the allowed population; it does not prove how a supplied
group assignment was fitted.

``METACELL_METADATA.parquet`` has one row per assigned group, in matrix row
order, with its donor/context, ``metacell_id``, ``n_cells``, observation role
and ``lasso_input_allowed``. The replay checks these values against actual
group members. LASSO observation rows and ``LASSO_SOURCE_ROWS.parquet`` are
derived from this metadata; source selection excludes target, validation and
outer-query donors according to the plan.

The metacell grouping preserves ``state_id``. The current ``_context_assets``
and ``source_node_features`` functions then select contexts by ``cell_type``
alone, exposed as ``cell_key``. Different states of the same cell type can
therefore contribute to the same context. Dataset and cancer identity are
carried by the containing native preparation and target rows.

Measured matrices and pathway activity
--------------------------------------

``METACELL_CODING.npz`` and ``METACELL_LNCRNA.npz`` are float32 sparse matrices
with rows matching metacell metadata and columns matching the cancer-level
``MATRIX_IDENTITY`` tables. Aggregation uses the pinned measurement scale:

* ``raw_counts``: sum within each supplied group, divide by the summed
  all-original-gene library, multiply by 10,000 and apply log1p. The policy is
  ``RAW_SUM_ALL_GENE_LIBRARY_1E4_LOG1P``.
* ``log_normalized``: arithmetic mean on the provided log scale. The policy is
  ``PROVIDED_LOG_SCALE_ARITHMETIC_MEAN_NO_INVERSION``; the producer does not
  infer an unavailable original normalization or invert it.

``METACELL_PATHWAY_ACTIVITY.npy`` is float32 with metacell rows and exact
pathway columns. It is computed by the original UCell function from coding
aggregates, exact signature membership, signature gene counts and missing
gene counts. Replay uses those same pinned inputs and function.

Candidate and target fields
---------------------------

``CANDIDATE_UNIVERSE.parquet`` contains exact lncRNA/pathway pairs for each
cell-type context and nested partition. It is the union of frozen bulk pairs
and source-donor discovery pairs. ``is_bulk_candidate`` and
``is_sc_discovery_candidate`` record both origins. ``universe_role`` is
``bulk_rescore``, ``sc_discovery`` or ``bulk_and_sc_discovery``. Discovery uses
CountSketch proposals followed by exact source effects under the saved
configuration and deterministic fold/context seed. The query/target/
validation selection-use flags are false and ``held_out_donor_rows_used``
is zero in the native producer.

``SIGNED_TARGET_ROWS.parquet`` has one estimable row per inner-target donor,
cell type and exact lncRNA/pathway pair, within its dataset/cancer/folds.
Despite its inherited name, ``query_donor_id`` here identifies the inner
target donor. Outer-query donor identity belongs to separate query assets.

* ``rho``: within-donor metacell Spearman correlation between lncRNA
  expression and pathway activity, stored as float32. Ties use average ranks.
* ``association_target``: ``abs(rho)``, retained as the magnitude field.
* ``signed_target_unit_interval``: ``(rho+1)/2``, preserving direction on
  the unit interval.
* ``n_metacells``: number of metacells in that donor/context block.
* ``lnc_detection_rate``: fraction of those metacells with lncRNA value >0.
* ``association_method``: ``WITHIN_DONOR_METACELL_SPEARMAN``.
* ``association_observation_unit``: ``within_donor_metacell``.
* ``association_target_level``:
  ``donor_x_celltype_x_lncrna_x_exact_pathway``.
* ``donor_resolved_target``: true. ``episode_role`` and ``row_role`` retain
  the containing fit/validation plan role.

Donor blocks need at least the configured ``min_target_metacells``. Within
each block, lncRNA detection must meet both the default five-metacell minimum
and the default 10% fraction; lncRNA and pathway ranks must both vary.
Non-estimable pairs do not receive a fabricated zero correlation.

The target rows also carry independently computed source-population fields:

* ``source_donor_count``: number of source donors with an estimable effect
  for the pair, which can be smaller than all source donors in the context.
* ``source_mean_rho``: equal-donor arithmetic mean of those effects.
* ``source_meta_rho``: tanh of the equal-donor mean Fisher transform, with
  individual rho clipped to [-0.999999, 0.999999] before transformation.
* ``source_direction_consistency``: the larger of the positive-effect and
  nonpositive-effect donor counts divided by support. The implementation
  counts exactly zero in the nonpositive group.
* ``source_meta_p_value``: two-sided one-sample t probability for the
  source-donor Fisher transforms, with support minus one degrees of freedom.
  A near-zero standard deviation yields the producer's deterministic 0/1
  special case. This field is not a multiple-testing-adjusted q-value.

The native join retains finite target/source effect fields with at least two
supporting source donors. This is a code-level estimability filter; it does
not establish that supplied donor labels are biologically independent.

Node features and coverage
--------------------------

``NODE_FEATURE_CONTEXTS.parquet`` maps ``context_index`` to ``cell_key`` and
records source/target donor counts, source/target metacell counts and target
row count. Only contexts with target rows and sufficient source donors enter
the feature arrays. Its donor counts describe available metadata; pair-level
``source_donor_count`` instead describes estimable effects.

``NODE_FEATURES.npz`` contains float32 arrays in context order:

* ``gene``, ``lncRNA`` and ``pathway``: context x node x 3, with columns
  weighted mean, weighted population standard deviation and weighted
  fraction >0. Source donors have equal total weight; their metacells share
  that donor weight. Only source metacells contribute.
* ``pathway_family``: context x family x 3, obtained by averaging member
  pathway feature vectors using static family membership. Its standard
  deviation component is an average of pathway standard deviations, not a
  standard deviation recomputed from pooled family activity.
* ``cell_context``: context x 1 x 4, with log1p(source donor count),
  log1p(source metacell count), coding nonzero-entry fraction and lncRNA
  nonzero-entry fraction. These last two fractions are computed over the
  source matrices without the donor weighting used by the node summaries.

``CONTEXT_COVERAGE.json`` also includes contexts that failed eligibility. It
records donor/metacell availability, state/reason, and candidate/target/
discovery row counts when those stages were reached. Missing fields in an
early-unavailable context do not mean that a later analysis was performed.

Completion and proof fields
---------------------------

* ``NESTED_EPISODE_UNAVAILABLE``: the plan lacks its minimum source donors;
  it contains partition/unavailability records without measured outputs.
* ``NESTED_EPISODE_NO_ESTIMABLE_CONTEXTS``: the plan was processed but lacks
  target rows or feature contexts; an empty result remains explicitly empty.
* ``NESTED_EPISODE_ASSETS_READY``: both target rows and feature contexts exist.

The reconstruction verifies actual values as well as hashes, counts and
``STATIC_BINDING.json``. It uses the original statistical implementation,
so passing proves consistency with the supplied measurements/groups and that
implementation. It is not an independent test of the scientific estimator.

``nested_frozen_statistics_replayed=true`` declares this reconstruction.
``nested_numeric_implementations`` pins the code used by the check. The
retained ``nested_fitted_numerics_replayed=false``,
``fitted_geometry_independently_verified=false``,
``fitted_lasso_numerics_verified=false`` and
``source_ancestry_complete=false`` distinguish outstanding geometry fitting,
LASSO and biological/source questions. Neither ``ASSETS_READY`` nor a
successful synthetic check declares neural training or a real scientific
result complete.

Implementation references
-------------------------

The actual model-use inspector now follows native launch bindings and writes
``transformation_replays`` rows for mapping, static/fold graphs, outer/nested
statistics and episode-manifest reconstruction. Each row identifies the
evaluated model and dependency context, source release hash, comparison
count and recomputed fold count. A global stage uses patient fold -1; this
does not merge the donor partitions validated inside that stage. The
``model_training_started`` field reports the inspection's execution scope,
not whether a historical supplied model was fitted.

Only exact handled source identities bypass generic unknown-transformation
messages. Acquisition, supplied QC/annotation, geometry, LASSO and
cross-module patient limits remain explicit; the overall model-use inventory
can still be incomplete after all six reconstruction stages pass.

The outer reconstruction separately covers ``TRAIN_ROWS.parquet``,
``QUERY_ROWS.parquet`` and ``UNIVERSE.parquet`` from the original metacell
producer. It rebuilds aggregation and UCell from the mapped measurements,
all-gene library and exact signatures, using supplied frozen membership.
The original context estimator then repeats source-only screening and
donor-specific Spearman targets. It compares every table column and row
order, group metadata, context coverage and completion counts. Both complete
and partial original caches execute this check before reusing a fold. A
cached successful proof still checks current input/output bytes and code.

Outer contexts pool by cell type, preserving state only in the grouping.
The outer ``association_target`` is abs(rho); signed rho remains a separate
column. ``source_meta_rho`` is the equally weighted Fisher mean across source
donors with estimable targets, using rho clipped to +/-0.999999 before
arctanh. A train target excludes its own donor from this mean and its source
count; a query target uses all eligible source donors. At least two source
donors remain after this exclusion. ``source_direction_consistency`` is the
larger of positive and nonpositive donor counts divided by source count.
Train rows are limited per pair by deterministic donor-hash order.

The model-use row ``NATIVE_SC_OUTER_FROZEN_STATISTICS_REPLAY_V1`` records this
scope separately from nested fit/validation targets. The native graph
validator calls the original completed-cancer gate, so downstream source
inspection cannot substitute receipt hashes for the outer numerical check.
The comparison proves consistency with the original statistical code and
supplied measurements/grouping; it does not independently validate that
estimator, fitted geometry, LASSO, acquisition or biological identity.

See ``scripts/materialize_single_cell_nested_hhgt_episodes_149.py``,
``cc_hhgt/v32/single_cell_nested_episodes.py``,
``cc_hhgt/v32/single_cell_metacell_screen.py``,
``cc_hhgt/v32/single_cell_metacells.py`` and
``cc_hhgt/v32/single_cell_nested_numeric_replay.py`` and
``cc_hhgt/v32/single_cell_outer_numeric_replay.py`` and
``cc_hhgt/v32/native_single_cell_model_sources.py``. The end-to-end handoff
and verification scope are recorded in ``CODE_COMPLETION_NOTES.rst``.
