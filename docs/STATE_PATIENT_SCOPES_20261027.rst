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
