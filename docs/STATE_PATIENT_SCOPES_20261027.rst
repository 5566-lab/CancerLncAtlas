Native State statistical patient scopes
======================================

The current State context now reconstructs private patient scopes while executing
the original statistical calculation and frozen private heads. This is a code
path for completed results; it neither launches training nor computes a new real
cohort during code verification.

The original calculation
------------------------

``prepare_state_inputs`` validates the original patient authority, State
measurements, lncRNA expression and five exported core embedding tables. Multiple
samples are averaged within each explicit patient before association calculation.
The original expression and State averaging, candidate universe, minimum-pair
threshold, variance checks, effect threshold and head arrays are unchanged.

For each outer fold, cancer, State and lncRNA, ``_paired_correlations`` observes
the actual finite-expression AND finite-State mask used for that correlation.
The optional observer receives this mask before the estimator returns. Candidates
sharing a mask share one private record. A wholly absent State produces an
explicit zero-patient record; measured zeros remain measurements. Constant values
can contribute patients while leaving the statistic unestimable.

Three statistical roles remain distinct:

* ``TRAINING_FEATURE_AND_LABEL``: the training correlation, pair count and
  association label are inputs to the original State head collector.
* ``VALIDATION_FEATURE_AND_LABEL``: the validation correlation and label are
  available to the original validation collector. The configured validation
  offset is used, including nondefault offsets.
* ``HELD_OUT_EFFECT``: the test correlation supplies the reported held-out State
  effect. It is not a head input. Frozen OOF membership probabilities are
  calculated from the training statistics by the existing producer.

If no validation candidate is eligible, the existing producer may fall back to
training inputs for validation. ``native_numeric_replay.folds`` retains the
original ``validation_fallback_to_training`` flag and effective candidate counts.
The private scope table records the actual validation statistics even when that
fallback occurs; statistical eligibility does not assert that an optimizer or
model selector consumed a row in a historical run.

Private field definitions
-------------------------

``state_patient_scopes_private.parquet`` is registered under ``private_outputs``
in the State context binding. It is excluded from public ``outputs``, the State
query/download reader and therefore the current website's State table inventory.
The private flag must be the boolean ``true``. Registering the same artifact hash
as a public output is rejected. This boundary does not replace filesystem access
controls for operators who can read the result directory directly.

``format``
  ``NATIVE_STATE_STATISTICAL_PATIENT_SCOPES_V1``.
``patient_fold_id``, ``split_role``, ``statistic_role``
  Outer fold, native split and its statistical purpose, defined above.
``cancer_id``, ``state_id``
  The actual candidate context. The State inventory remains the original seven
  programs, including missing programs.
``patient_map_sha256``
  The hash of the bound original fold-authority input (file or dataset).
``patient_map_row_indices_json``
  Zero-based positions in the table loaded by the native reader, before its
  validation sorting. All authority rows for each paired patient are retained.
  These are authority mappings, not the raw aliquots contributing to each mean.
``native_patient_count``
  Number of distinct patient units in the actual paired mask. It equals the
  original candidate's ``n_pairs``. Several authority rows can represent one
  analysis patient, so authority-row count may be greater.
``cohort_id``
  Canonical SHA-256 of the patient-map hash and original authority row positions.
  It identifies a file-qualified cohort, not a cross-study biological identity.
``lncrna_ids_json``, ``candidate_count``
  Ordered lncRNAs sharing this paired mask and their number.
``statistic_available_json``
  Aligned boolean flags from the original estimator. Pair count alone is not
  sufficient: zero variance can make the correlation unavailable.
``core_embedding_available_json``
  Aligned flags for lncRNA membership in the actual fold's validated finite core
  embedding table.
``private_head_input_eligible_json``
  Estimable correlation AND available core embedding for train/validation.
  Always false for test statistics, which supply held-out effects instead.
``core_embedding_sha256``, ``core_parameter_sha256``, ``v32_core_checkpoint_sha256``
  The already verified core parent of this State fold. These connect the scope
  to the actual frozen representation; they do not certify its upstream patient
  ancestry or training data independence.

Reconstruction and limits
-------------------------

Materialization captures scopes only while replaying the original measurements
and heads against the recorded OOF results. Context auditing repeats that replay
and compares every private table field, schema, order, row count and artifact
hash. Rehashing changed cohort rows cannot replace the original calculation.
The observer is optional in the estimator and does not change its return values.

Remaining work includes raw aliquot contribution ancestry, source acquisition,
cross-module patient identity reconciliation and the patient population used to
fit the parent core. Head input eligibility is not proof of optimizer execution.
Neither source replay nor a matching patient scope demonstrates independent
predictive performance or a causal lncRNA--State mechanism. The full project
requirements C01, C13 and C16 therefore remain open.

Core training inputs and explicit patient identity
-------------------------------------------------

The completed State context now automatically calls the existing native core
input inspector on its actual bound core export. The original prepared loader,
checkpoint/receipt checks, candidate collection, construction replay and graph
source inspection execute for each available parent fold. The inspector itself
does not construct another neural core or run an optimizer. Existing State
statistics and frozen-head forwards remain a separate calculation.

Records distinguish ``STATE_CORE`` from ``STATE_HEAD``. Missing original files,
unknown study ancestry and unsupported historical controls remain explicit
dependencies or gaps. A core export hash alone cannot mark them reconstructed.
The State head's raw expression and State measurement ancestry also remains
unresolved; neither these declarations nor matching statistics waive it.

An optional ``source_inventory`` object is supported by both the current State
input manifest and the native State workflow manifest::

    {
      "source_inventory": {
        "artifact_locations": [
          {"path": "/absolute/relocated_native_source.parquet", "sha256": "..."}
        ],
        "patient_identity_crosswalk": {
          "path": "identity.parquet",
          "sha256": "...",
          "namespace": "explicit-study-identity-authority"
        }
      }
    }

Paths resolve relative to the declaring manifest. The workflow pins resolved
locations and preserves them through preparation, external completion and State
publication. A locator must be consumed by the actual core input traversal;
unrelated or unused files cannot be added as evidence. Dataset hashes are
verified, while individual files enter the State source artifact inventory.

The crosswalk uses the existing columns ``source_file_sha256``,
``identity_column``, ``raw_id`` and ``canonical_patient_id``. The explicit
namespace and canonical identity determine a private patient token. State
statistical cohorts map the original authority ``patient_id`` values at their
recorded row positions. Core inputs use their actual file-qualified identities
through the existing source inspectors. Equal identifiers in different files
are not assumed to identify one person; a supplied alias can explicitly link
different strings. Unmapped values remain unresolved.
The native CSV/TSV reader retains identity columns as strings, preserving leading
zeros such as ``001`` and keeping that patient distinct from ``1``. Numeric
measurement and fold columns retain their existing validation/conversion path.

The additional ``state_model_use_*`` tables are all private. They contain native
entity/study exposure, file dependencies, gaps, transformation replays, mapped
patient tokens, mapping counts, overlap counts and source-completeness results.
The State auditor reconstructs every table from the actual sources. Private
flags, the exact table inventory and content hashes are checked; all private
tables remain outside public State queries, downloads and website registration.

The public ``state_input_coverage`` table contains one row per dependency
context. Its base fields follow the existing model-use completeness calculation:
dependency counts, unresolved dependency count, entity/study source counts,
unverified study sources, blocking gaps and ``inventory_complete``. Additional
fields are:

``observed_identity_context_count``, ``unmapped_identity_context_count``
  Sums of observed/unmapped identities across recorded source/role/fold scopes.
  They are not unique patient totals and can count a patient in several scopes.
``registered_patient_mapping_complete``
  True only when at least one identity is observed and every recorded identity
  in that context maps. This does not imply that all upstream sources or all
  historical patient use have been recovered.
``patient_identity_namespace``
  The supplied authority namespace, or null when no crosswalk was supplied.
``optimizer_patient_use_attested``, ``independent_performance_verified``
  Both remain false. Source reconstruction cannot prove either claim.

Cross-context overlap is a lower bound from mapped identities. It is not itself
a fold leakage finding: it includes statistical, held-out and whole-file source
roles, and does not claim that all recorded rows trained a model. Source ancestry
and identity mapping are separately reported, so complete mapping cannot hide
unknown raw sources. Parent training source traversal is now connected; complete
parent population ancestry and cross-module identity coverage remain open.
