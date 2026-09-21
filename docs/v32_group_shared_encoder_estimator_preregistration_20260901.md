# V3.2 group-Latin shared-encoder estimator preregistration

Date: 2026-09-01 (Asia/Hong_Kong)

Status: `TIMING_AND_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING`

This document freezes the scientific and runtime gates for the proposed
`balanced_group_latin_shared_encoder_v1` candidate/chunk estimator before any
real-data comparison result is observed. The legacy `exact_cartesian_v1`
mixed-chunk schedule remains the reference oracle.

## Motivation and invariant scope

The old schedule requires 403 immutable candidate batches crossed with 29
runtime graph chunks in every coverage cycle. Live RTX 4090 telemetry showed
that this requires 2,929 optimizer steps per fold/cycle and cannot finish under
the authorized time or cost cap.

The proposed estimator retains all of the following without change:

- patient folds, OOF modality lineage, labels, sealed-test firewall and seed;
- the immutable order and membership of every candidate batch;
- four consecutive candidate batches per optimizer group, with the final
  partial group unchanged;
- one global nonlinear nnPU reduction, one direction reduction and one
  shrinkage reduction over that same optimizer group;
- one optimizer update per group and the same learning rate/weight decay;
- exact all-chunk weighted validation and test aggregation;
- byte-identical schedules across G0/G1/G2 for a common fold and seed.

It changes the graph-chunk assignment within an optimizer group. All batches
in canonical group `g` share one chunk, allowing a single encoder result to be
used by all decoders in the group. This changes graph-noise correlation and
encoder-dropout consumption. Because nnPU contains a nonlinear clamp, the new
same-chunk group objective is not mathematically identical to, and is not
claimed to be an unbiased estimator of, the old mixed-chunk oracle.

## Frozen schedule

Let `A=4` be gradient accumulation, `B=403` candidate batches, `G=ceil(B/A)=101`
optimizer groups and `K=29` graph chunks. Group `g` contains batch indices
`4g ... min(4g+3, B-1)`. For cycle `c`, every batch in group `g` uses

`chunk = pi[(g + rho[c mod K]) mod K]`,

where `pi` is the existing frozen chunk permutation and `rho` is the frozen
pass-offset permutation, both derived from seed and outer fold without
consuming the Torch RNG stream. Each cycle visits all 403 batches and 101
groups once. Each chunk receives three or four groups. The first K cycles cover
every `(group, chunk)` and `(batch, chunk)` pair exactly once.

## Real-data oracle comparison

The comparison uses only authorized G2/PATIENT_FOLD_0 training inputs. It must
not read test batches, write a checkpoint, write `SUCCESS.json`, or create any
formal prediction. The canonical comparison unit is optimizer group 0: the
first four immutable training batches and all 29 runtime chunks.

Two fixed parameter states are compared:

1. `theta_0`: the exact formal initialization for seed 20260726.
2. `theta_1`: a copy of `theta_0` after exactly one legacy mixed-chunk group-0
   optimizer step at frozen Cartesian pass offset 0, using the registered
   optimizer and learning rate. `theta_1` exists only in memory.

At each state, dropout is disabled for the primary oracle comparison.

- Legacy oracle: enumerate all 29 mixed-chunk pass assignments for group 0.
- Proposed oracle: enumerate all 29 same-chunk assignments for group 0.
- For each assignment, compute membership risk, direction loss, shrinkage,
  total objective, nnPU clamp branch and the complete parameter gradient.
- Average scalar components and gradients over the 29 assignments in
  canonical order using float64/Kahan accumulation on CPU.

The proposed estimator passes only if every condition below passes at both
`theta_0` and `theta_1`:

- all losses and gradients are finite;
- the nnPU clamp branch is constant across all 58 evaluations (29 legacy plus
  29 proposed), and the paired branch-disagreement count is exactly zero; any
  branch non-constancy is a typed `SCIENTIFIC_FAIL`, even when aggregate loss
  means happen to be close;
- the absolute mean difference for membership risk, direction loss,
  shrinkage and total objective is no greater than
  `max(1e-7, 1e-6 * abs(reference_mean))` for that component;
- the paired 95% bootstrap confidence interval for every scalar component is
  wholly contained within plus/minus 1% of the reference mean;
- relative L2 error of the complete mean gradient is at most `1e-3`, its
  cosine similarity is at least `0.9999`, and its norm ratio
  (proposed/reference) is within `[0.999, 1.001]`;
- encoder, residual map/output, gate and direction-head mean-gradient cosine
  similarity is at least `0.999` for every module with nonzero gradients;
- a module with zero reference and proposed gradients is reported as typed NA;
  a one-sided zero/nonzero module mismatch fails.

The audit also runs a fixed-RNG, dropout-enabled synthetic/small-graph test to
verify deterministic replay and records the expected reduction in RNG draws.
This test checks implementation determinism and does not replace the real-data
dropout-disabled oracle comparison.

Thresholds are conjunctive. A failure cannot be waived after observing the
result; changing a threshold requires a new dated protocol and a fresh
comparison root.

The runner emits schema
`CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1`. Its only terminal statuses
are `PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING`,
`SCIENTIFIC_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY`, and
`RUNTIME_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY`. Every terminal receipt and
heartbeat is typed `TIMING_AND_ORACLE_COMPARISON_ONLY`, carries
`formal_artifacts_written=0`, and states that it is not a formal checkpoint,
prediction, failure marker or winner-selection input. A pass is one necessary
gate only; it does not itself authorize formal training.

The guarded callable is
`cc_hhgt.v32.group_shared_encoder_oracle:run_authorized_oracle_comparison`.
It requires the already authorized pending `G2/PATIENT_FOLD_0/seed=20260726`
task, `B=403`, four full 8,192-row group-0 batches and `K=29`; any drift is a
typed runtime failure. The receipt binds the prepared-artifact SHA, group-input
SHA, canonical-assignment SHA, both in-memory theta state/RNG SHAs, runtime
chunk permutation and weights, protocol SHA, per-theta component/gradient
gates and production shared-helper conformance telemetry. It reports exactly
which four training batches were used. The standard prepared-payload safety
validator may inspect validation-batch schema, but the comparison reports
`validation_rows_used=0`, `validation_labels_read=false` and
`test_labels_read=false`; no validation value enters a loss, gradient or gate.

## Shared-encoder implementation gate

For a group, the implementation must assert one unique chunk, materialize one
graph, call `encoder.encode` exactly once, call the decoder once per candidate
batch in original order, call the global group loss exactly once, call backward
exactly once and perform exactly one guarded optimizer step. It must not fall
back to the legacy multi-encoder algorithm after an OOM because that would
change the registered estimator and RNG semantics.

Synthetic FP32 tests must show parameter-by-parameter loss and gradient
equality to an independent conventional reference that also performs one
encoder call followed by four decoder calls and one global loss/backward. The
loss tolerance is `atol<=1e-6, rtol<=1e-6`; every parameter gradient must have
`max_abs_difference<=1e-5`, `relative_L2<=1e-4`, and the flattened gradient
cosine must be at least 0.99999. The nnPU branch must be identical.

## Formal runtime identity and resume gate

The production trainer uses checkpoint schema
`CC_HHGT_V3_2_FULL_TRAINING_STATE_V4_GROUP_SCHEDULE`. It binds the formal
graph variant across the canonical `v32-g012-{g0|g1|g2}-...` run ID,
`task_contract.graph_variant`, the prepared artifact's immediate `G0`/`G1`/`G2`
directory and `formal_graph_variant` payload field before the optimizer loop.
The same variant is written into the runtime training contract, every
checkpoint, resume identity comparison, heartbeat and terminal receipt.

V3 streaming checkpoints are rejected with the typed
`RESUME_CHECKPOINT_V3_TYPED_REJECT_REQUIRES_V4_GROUP_SCHEDULE` outcome; they are
not interpreted under the expanded group-schedule schema. On V4 resume, every
completed history row is checked in cycle order, all optimizer/call counters
must be non-negative and cumulatively exact, OOM state must be monotone, and the
shared-encoder mode must reproduce its closed-form per-cycle call budget. The
checkpoint totals must equal the recomputed history totals before model or
optimizer state is restored.

Both balanced modes require the literal runtime declaration
`allow_partial_candidate_chunk_rotation: true`. Missing or non-Boolean values
fail closed. This declaration only permits a terminal receipt before all `K`
rotations have completed; it does not relabel partial coverage as full coverage.
The legacy exact-Cartesian default remains unchanged and does not require this
key.

Call telemetry in checkpoints and terminal receipts has scope
`OPTIMIZER_GROUPS_ONLY_EXCLUDES_VALIDATION`. `training_optimizer_steps` and the
four `training_*_calls` fields count only optimizer groups; exact all-chunk
validation calls are explicitly excluded. A CUDA formal-loop E2E regression
executes shared cycle, exact validation, V4 checkpoint, resume and `SUCCESS`,
and separately verifies fail-closed training and validation OOM paths. These
runtime gates do not waive the frozen oracle or equal-step G0/G1/G2 pilot.

## Equal-step downstream pilot

The oracle-gradient comparison is followed by a candidate-only downstream
pilot. For G0, G1 and G2 separately, both estimators start from the same
fold-0/seed initialization and run exactly one 403-batch/101-step candidate
pass:

- reference: `balanced_cyclic_single_pass_v1`, which is one unmodified
  mixed-chunk pass from the legacy Cartesian oracle;
- proposed: `balanced_group_latin_shared_encoder_v1`.

Both are then evaluated by the unchanged exact all-29-chunk validation path.
No arm may pass by borrowing another arm's result. Every arm must satisfy:

- absolute validation-logloss difference at most 0.002;
- candidate-level mean-logit Pearson correlation at least 0.995;
- candidate-level mean-logit Spearman correlation at least 0.995;
- identical validation candidate keys, labels, fold roles and chunk weights;
- no test rows or outer metrics read.

Pilot outputs are typed comparison artifacts and cannot be used as formal
checkpoints or winner-selection inputs.

## Paid-GPU memory and time gate

The first real GPU action after no-GPU authorization is an isolated G2/Fold0
one-group probe. It writes only a typed probe receipt with
`formal_artifacts_written=0` and records exact call counts, group/batch hashes,
the selected chunk, loss/gradient/parameter finite checks, positive parameter
delta, peak allocated/reserved bytes, elapsed seconds and RNG hashes.

Hard memory/numerical requirements:

- peak reserved memory is positive and below 22 GiB;
- encoder calls = 1, decoder calls = 4, global-loss calls = 1,
  backward calls = 1 and optimizer steps = 1;
- no OOM, nonfinite value or semantic fallback;
- no checkpoint, formal log, `SUCCESS.json`, `FAILURE.json` or prediction.

After that gate, bounded probes measure representative shared-group step time
for G0, G1 and G2 and one complete full-K G2 validation. The worst-case core
projection is

`upper_seconds = 1.25 * [sum_v 5*6*(101*step_seconds_v + validation_seconds_G2 + 30) + sum_v 5*setup_seconds_v + 1800]`,

where `v in {G0,G1,G2}`, 30 seconds is per-cycle checkpoint allowance and
1,800 seconds is finalization allowance. G2 validation is deliberately used as
the upper bound for all variants.

Formal GPU training is authorized only if all of the following hold:

- the real-data oracle comparison passes every frozen threshold;
- `upper_seconds / 3600 <= 50`;
- projected cost at the conservative GPU-plus-disk rate, plus the already
  incurred gate cost and a CNY 2 reserve, is no greater than the just-in-time
  remaining budget receipt;
- the projected finish precedes the provider auto-stop by at least two hours;
- the r1 run is sealed `ABORTED_RUNTIME_INFEASIBLE` and the r2 formal output
  root is absent.

Failure of any gate leaves the instance stopped and the estimator classified
as experimental only.

## Seed scope

The current 15-task G0/G1/G2 job uses one initialization seed and is a core
candidate/ablation stage. It must not be described as the three-seed final
stability comparison required by the broader design. Any later 45-task stage
requires a separate time/cost authorization and projection.
