# V3.2 equal-step candidate pilot implementation

Date: 2026-09-01 (Asia/Hong_Kong)

Status: `CODE_AND_LOCAL_CUDA_E2E_READY_NOT_YET_RUN_ON_REAL_G012_INPUTS`

This implementation realizes the non-skippable downstream pilot frozen in
`v32_group_shared_encoder_estimator_preregistration_20260901.md`.  It does not
change that preregistration and is not an authorization to start formal G0/G1/G2
training.

## Guarded entry

The public callable is:

`cc_hhgt.v32.equal_step_candidate_pilot:run_authorized_equal_step_candidate_pilot`

That exact import specification is approval-bound as `authorized_trainer` in
all three component contexts.  Reverification passes it back as
`trainer_specification`; a missing, swapped or differently spelled callable is
rejected before Torch import.

It accepts an `EqualStepPilotAuthorizationContext` assembled from exactly three
independently guarded `TrainingAuthorizationContext` values, one each for G0,
G1 and G2.  The callable reruns the complete hash/approval guard for every
context before importing Torch or touching CUDA.  All contexts must bind the
same endpoint and hardware class.

The real-data runner is hard-bound to:

- `PATIENT_FOLD_0` and seed `20260726`;
- variants G0, G1 and G2, with no missing or duplicate variant;
- 403 immutable training batches, 8,192 rows in every full batch, accumulation
  width four and exactly 101 optimizer groups;
- 29 registered runtime graph chunks;
- the core external-router HHGT candidate architecture.

## Two arms

For each variant the model is built once from the authorized production model
builder.  Its initial state and post-construction Torch RNG state are copied in
memory.  Both arms restore those exact states before creating a fresh AdamW
optimizer.

- Reference: `balanced_cyclic_single_pass_v1`, executed by the production
  two-pass streaming global-loss helper.
- Proposed: `balanced_group_latin_shared_encoder_v1`, executed by the
  production shared-encoder conventional group helper.

Both arms make exactly 101 successful guarded optimizer updates.  Production
helpers report runtime encoder, decoder, global-loss and backward call counters.
Any training, optimizer-step or validation OOM is a typed failure with no
fallback or retry under altered semantics.

The runner emits typed stdout-only setup, successful-optimizer-step and
completed-validation-chunk progress events.  Setup events explicitly report
`training_started=false`; only a successful guarded optimizer update is
optimizer progress. Each successful update reports `training_started=true`
and current CUDA allocated/reserved bytes so the paid-run training-start gate
requires both optimizer evidence and GPU telemetry. These events do not write files and do not consume or
inspect any Torch/NumPy RNG state.  Every event binds pilot, authorization
run/task, graph variant, arm, phase, completed/total counters and a unique
event ID; optimizer steps and validation-chunk ordinals are monotonic within
that identity.

## Exact validation and gates

Both arms call the same production exact-all-chunk core.  A detailed read-only
view exposes its already aggregated candidate mean logits and labels; formal
training continues to use the scalar-only view without the extra candidate
concatenation.

The pilot computes ordinary binary validation logloss from the exact mean
logits.  It also records the production validation objective, which includes
the registered direction and shrinkage terms, but does not use that compound
objective as a substitute for the preregistered logloss gate.

Every variant must independently pass all of:

- absolute validation-logloss difference no greater than 0.002;
- candidate mean-logit Pearson correlation at least 0.995;
- candidate mean-logit Spearman correlation at least 0.995;
- exact ordered node-map-bound candidate keys, labels, validation fold role and
  runtime chunk weights;
- exactly 101 optimizer steps in each arm and exact 29-chunk validation.

The runner additionally checks that candidate/label/fold identities,
optimization contracts and each estimator's schedule are byte-identical across
G0/G1/G2, as required by the preregistration.  No variant may borrow another
variant's pass.

The cross-variant gate also explicitly binds the initial model-state SHA,
post-construction RNG SHA, full ordered training identity, training
batch-row-count SHA, runtime chunk permutation SHA and resolved
precision-contract SHA.  G0/G1/G2 differ only in active graph edges:
preparation freezes a common node universe and complete relation schema, and
the formal builder uses the same architecture and seed.
Consequently a different initial parameter/RNG state is architecture or RNG
consumption drift, not an admissible graph-ablation difference.

Validation identity has separate hashes for ordered biological keys, all four
loss labels/row weights, all masks (including `graph_available`), and every
candidate-side model input (`base_logit`, conservation context and any admitted
candidate extras).  A composite validation-loss-payload SHA binds those parts;
the loss-contract SHA additionally binds nnPU defaults, direction weight,
shrinkage and aggregation/reduction semantics.  Runtime chunk weights remain a
separate exact pair/cross-variant gate.

Training identity is independently bound over all 403 batches in optimizer
order.  Separate hashes cover the node-map-bound candidate `l/p/c` triplets,
all candidate extras and model inputs, all four labels/row weights and masks,
plus a composite training-loss-payload and ordered-training-identity SHA.
G0/G1/G2 must match every training-identity hash exactly even when their batch
row counts are unchanged, and each payload is re-hashed after both arms to
fail closed on in-place mutation during execution.

The scalar formal API and detailed comparison API delegate to one
`_evaluate_full_runtime_coverage_core`.  CPU and CUDA regressions execute both
views independently and require bit-identical objective floats, not an
approximate tolerance.

## Output firewall

The only output is typed JSON on stdout under schema
`CC_HHGT_V3_2_EQUAL_STEP_CANDIDATE_PILOT_V1`.  The runner never resolves an
output root and has no file writer.  It cannot write a checkpoint,
`SUCCESS.json`, `FAILURE.json`, prediction, winner-selection input or formal
training log.  Prepared payload validation rejects test keys before execution;
the receipt states zero test rows, no test labels, no outer prediction and no
outer metric were read.

The receipt includes prepared/config/input authority hashes, initial/final
model and RNG hashes, per-arm schedule and assignment hashes, actual call
telemetry, per-step and validation wall timing, exact validation identity
hashes and every threshold decision.  It calculates an informational 50-hour
projection from observed proposed-arm timing (including G2 validation and
per-variant setup time), but explicitly does not replace
the separately preregistered bounded timing/budget receipt.

## Local verification

`tests/test_v32_equal_step_candidate_pilot.py` covers the frozen 403/29/101
geometry, scientific thresholds, tied Spearman ranks, CPU end-to-end execution,
real local CUDA end-to-end execution, typed training/validation OOM failure,
guard revalidation failure and the absence of comparison file artifacts.
