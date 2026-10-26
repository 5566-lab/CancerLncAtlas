State supplied-measurement contributions
=======================================

Scope
-----
The native State algorithm is unchanged: it validates supplied long tables,
aligns them to the patient fold authority, takes within-patient arithmetic means
and computes the original split-specific paired statistics. An optional observer
now receives the actual joined rows and resulting means inside
``_align_source_tables``. Ordinary preparation and training do not enable it.
The current-context frozen-result replay enables it and publishes two additional
private, pinned tables. No new producer, fitting step or scientific result is
claimed by this instrumentation.

Measurement semantics
---------------------
State measurements join by cancer and the original canonicalized explicit
patient ID. TCGA sample suffixes in that ID are removed by the original function;
State sample IDs need not be expression aliquot IDs. Expression measurements
join by cancer and sample ID to the actual fold authority's explicit patient ID.
Multiple samples contribute equally to the native patient/feature mean. Missing
values are skipped by the original pandas mean; measured zeros contribute.
An all-missing group remains missing and cannot enter a finite paired mask.
Out-of-authority rows and non-program State metadata never become contributions.

Private data dictionary
-----------------------
``state_measurement_contributions_private`` has one row per joined
role/cancer/patient/feature group. ``source_sha256`` pins the loaded table or
dataset. ``source_row_indices_json`` addresses zero-based positions in the loaded
table before validation sorting/filtering, not text line numbers. Finite and
missing position lists partition those rows. ``finite_value_count`` is the
number of measurements included in the native arithmetic mean, not a patient
count. ``native_patient_mean`` is the value emitted by the original groupby,
not an independently re-estimated quantity. The patient authority SHA and row
positions link the patient group to all its authority samples. For State, these
authority sample rows do not imply that every sample supplied a State assay;
the source-row lists specify the actual assay observations.

``contribution_id`` hashes the input role, source SHA and sorted source positions.
``feature_id`` is the original State ID or lncRNA ID, with its actual
``value_column``. Patient and sample-associated data remain private.

``state_measurement_scope_links_private`` links every existing paired-mask scope
to both input roles. It retains fold, split, cohort, cancer, State and candidate
IDs from the actual statistical observer. Its sorted contribution IDs select
one State group per paired patient and one expression group per paired
patient/candidate. It checks that every selected group has finite measured
contributions. Empty paired masks retain empty link lists; constant-expression
patients still appear even when a correlation or head input is ineligible.
These are statistical exposures, not an optimizer-use attestation.

Integrity and boundaries
------------------------
Current context source hashes include the tracing implementation. Materialization
and audit rerun the original means and paired statistics and compare both private
tables. Rehashing edited row positions or scope links cannot bypass this replay.
Missing private tables and attempts to expose them through public aliases fail.
The native State preparation code inventory also pins the observer implementation.

Existing State raw-measurement ancestry gaps remain blocking: contribution
reconstruction does not prove acquisition, normalization, biospecimen truth,
cross-study identity, fitted core ancestry or external predictive performance.
In particular, the supplied global State expression table is not silently
replaced with a fold-local expression product with different semantics.
Tests use synthetic long tables and explicit frozen completed-head fixtures on
host 149 with parameter fitting prohibited. No real cohort or model fitting is
part of this change.
