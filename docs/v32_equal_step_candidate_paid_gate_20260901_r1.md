# V3.2 G0/G1/G2 Fold0 equal-step candidate paid gate r1

Status: `IMPLEMENTED_AND_LOCALLY_TESTED_NOT_DEPLOYED_NOT_RUN`

This is a comparison-only paid-GPU gate for
`cc_hhgt.v32.equal_step_candidate_pilot`. It does not authorize formal
G0/G1/G2 training, a budget increase, winner selection, checkpoints,
predictions, or any outer/test read. No cloud API was contacted while the gate
was built or tested.

## Fixed identity and scope

- namespace: `v32_equal_step_candidate_paid_gpu_20260901_r1`
- pilot: `v32-g012-equal-step-candidate-paid-gpu-20260901-r1`
- fixed fresh provider JobId: `v32-equal-step-g012f0-s20260726-r1`
- tasks: exactly G0, G1, and G2 / `PATIENT_FOLD_0` / seed `20260726`
- trainer: exactly
  `cc_hhgt.v32.equal_step_candidate_pilot:run_authorized_equal_step_candidate_pilot`
- hard cap: six paid hours
- requested comparison cap: CNY 15
- effective cap: `min(CNY 15, conservative JIT project budget remaining)`
- prices: compute/GPU CNY 2.05/h + disk CNY 0.04/h = total CNY 2.09/h

The CPU/no-GPU verifier materializes three different config/input/task/approval
sets under the bootstrap authorization root. Each has a distinct run/task ID
and exact graph variant. The no-Torch bootstrap guards all three separately,
checks that Torch has not entered `sys.modules`, builds one composite context,
and only then calls the pilot. The pilot re-runs every component guard before
its internal Torch import.

## Static authority

The independent verifier is deployed outside the overlay at:

`./data/CancerLncAtlas/runtime/equal_step_gates/v32_equal_step_candidate_paid_gpu_20260901_r1/`

It reads the overlay only as data. Python source is parsed with `ast`; every
authority file is opened with `O_NOFOLLOW`, must be a non-empty regular file
with `st_nlink=1`, is hashed/read through the same descriptor, and is bracketed
by stable `fstat` identity checks. Lexical containment and every extant path
component are checked before a file is admitted. The verifier imports nothing
from the overlay.

Static authorization requires all of the following:

1. The server-archived and independently returned r1-r6
   `STATIC_AUTH_READY` files are byte-identical and match immutable SHA-256
   `1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0`.
2. r1 is sealed `ABORTED_RUNTIME_INFEASIBLE`, has removed live
   authorizations, forbids resume/reuse, and binds that exact archived r1-r6
   receipt.
3. All 15 variant/fold hash, size, and canonical path rows in r2
   `INPUT_REUSE_READY.json` exactly match r1-r6. Each current PT is same-fd
   regular-file size checked. No PT re-hash is substituted for this authority.
4. All three Fold0 rows are cryptographically hard-anchored by the immutable
   r1-r6 receipt SHA. G0 and G1 additionally bind fixed byte sizes
   3,748,393,077 and 4,346,233,053; G2 binds fixed size 4,482,708,589 and direct
   SHA-256 `4ebba1d4060f3efaa8c080824386802ca732bedf07681ebd0283deb087b0ca39`.
   The static receipt records the exact three rows and their composite SHA.
5. The code archive/receipt, code-tree SHA, production training helper,
   preregistration, pilot, no-Torch bootstrap, config template, and external
   control deployment manifest all match.
6. A fresh JIT billing receipt matches the fixed component prices, is at most
   30 minutes wide, names the same instance, and proves the conservative
   balance as `210 - round(instance_age_seconds * 2.09 / 3600, 4) - reserve`,
   with at least CNY 5 retained for final cleanup. Provider-snapshot and
   generator hashes are mandatory.
7. A real oracle PASS already exists. The gate does not ship or freeze a
   not-yet-generated PASS. It waits for the returned PASS line and original
   server execution log, requires exactly one matching terminal line, validates
   the actual G2/Fold0 403/B4/K29 execution, both theta scientific results,
   finite one-step update, RTX-GPU memory/timing evidence, and all safe
   comparison-only fields. Oracle runner, preregistration, production training
   helper, and prepared-input hashes must match the current frozen overlay.

The frozen checked-in deployment manifest is
`docs/v32_equal_step_candidate_external_deployment_manifest_20260901_r1.json`.
It is copied as `EXTERNAL_DEPLOYMENT_MANIFEST.json` beside the six external
controls. The manifest and every control copy are re-hashed by the external
verifier; the local dispatch controller independently re-hashes its local
copies, including itself, before any paid start.

## Output firewall and progress

The server launcher retains one stdout JSONL execution log only below the
comparison bootstrap namespace. Verifier output is suppressed in the paid
launcher, stderr is not merged into that JSONL, and the launcher adds no text
to stdout. Thus its stdout consists only of candidate pilot JSON events. The
pilot owns no formal result root and reports zero formal artifacts, no
checkpoint/marker/prediction, zero test rows, and no outer/test values read.

The log guard accepts only exact pilot JSON. Setup is
`EQUAL_STEP_VARIANT_SETUP_START` with `training_started=false`; it is evidence
for one non-renewable 45-minute initial roughly-4-GB load window for that
variant, not optimizer progress. Once an optimizer or validation phase is
active, a new exact monotonic event is required within eight minutes. Optimizer
progress must be 1..101, carry `training_started=true` plus non-empty CUDA
allocation/reservation telemetry, and validation progress must be 1..29 for
the exact variant/arm/estimator identity. Full-line SHA, semantic event ID, and maximum
completed count are all retained, so repeated/reordered old `--tail 1200`
windows cannot reset a deadline or reopen a load window.

A terminal-looking substring is ignored. PASS/FAIL is accepted only for the
exact terminal schema/status and all safe output flags. PASS additionally
requires the observed progress chain for every G0/G1/G2 arm to contain exact
101-step optimizer completion, exact K=29 validation completion, and the
variant-complete heartbeat. Terminal lines themselves are hash-deduplicated.
The exact terminal line is revalidated after the provider confirms `Stopped`.

## Paid lifecycle

The dispatch controller requires a stopped one-RTX-4090 instance and exact
provider prices. From the original JIT query time it computes one absolute
deadline as the earlier of the six-hour cap and effective-cost cap, less a
shutdown reserve. While the instance is still stopped it:

1. sets the provider absolute auto-stop schedule and reads it back;
2. starts the local supervisor and waits for its first fixed-JobId poll;
3. rechecks that the supervisor is still alive immediately before paid start;
4. starts the instance, immediately rechecks the supervisor, and proves the
   schedule was not extended;
5. proves the JobId is absent from the provider job list, writes a
   `SUBMIT_IN_FLIGHT` phase marker, and submits only the fixed launcher;
6. binds the returned `data.job` record to exact JobId/name/cwd/command and
   `CreatedTime >= dispatch_attempt_not_before`, then writes `SUBMITTED`.

The supervisor consumes the provider's real `data.job` and
`data.logs.Stdout|Stderr` shapes. Pre-submit and post-submit missing-job grace
periods are separate. Dispatch and stop receipts are written atomically as
BOM-free UTF-8 so the strict Python verifier reads the same bytes on Windows
PowerShell 5.1 and newer hosts.

Every dispatcher and supervisor path converges on stop plus provider-state
confirmation. The typed stop receipt is
`CANCERLNCATLAS_EQUAL_STEP_CANDIDATE_AUTOSTOP_V1` with
`INSTANCE_STOPPED_STATE_CONFIRMED` and `gpu_billing_active=false`. Dispatch
returns scientific success only for a post-stop-revalidated pilot PASS.

Neither `STATIC_AUTH_READY`, the JIT/dispatch receipt, the pilot terminal
receipt, nor the stop receipt authorizes formal training or a later budget.
Those remain separate fail-closed computational gates; no interactive review
or approval is part of this controller.

## Formal-r2 integration handoff (non-authorizing)

After verified stop, the same frozen external verifier supports
`--materialize-handoff`. It reads the static authority, archived JIT receipt,
pilot stdout JSONL, local auto-stop receipt, checked-in decision, and external
deployment manifest through the hardened reader. It emits schema
`CC_HHGT_V3_2_EQUAL_STEP_FORMAL_GATE_HANDOFF_R1_V1` with status
`EQUAL_STEP_GATE_PASS_HANDOFF_NOT_FORMAL_AUTHORIZATION`.

This operation does not trust terminal `status`. It strictly recomputes:

- static/JIT/terminal-line/terminal-log/auto-stop/decision/deployment hashes;
- exact G0/G1/G2 run, task, fold, seed, endpoint, hardware, prepared artifact,
  approval artifact hashes, ordered training identity, ordered validation
  identity, and runtime-chunk permutation identity;
- same initial model and RNG for both arms and across all three variants;
- exactly 101 optimizer steps and K=29 validation chunks for each arm of each
  variant;
- every execution gate, every per-variant and cross-variant gate, exact
  identity field gates, proxy-label identity, and the numerical logloss,
  Pearson, and Spearman thresholds.

Recommended r2 validator patch: extend
`CC_HHGT_V3_2_G012_R2_FORMAL_PAID_GATE_BINDINGS_V2` with an `equal_step`
mapping containing exact `{decision, static_auth, jit_budget, terminal,
provider_auto_stop, external_deployment_manifest, handoff}` path/SHA pairs.
The r2 validator should duplicate the handoff recomputation implemented in
`materialize_formal_gate_handoff()` / `_scientific_pass_identity()` rather
than accept the handoff status alone. It must then separately validate the
bounded <=50-hour formal runtime projection, a new formal-training JIT budget,
and the formal provider auto-stop margin. Until all three later gates and their
real hashes exist, the current
`FORMAL_EQUAL_STEP_RUNTIME_BUDGET_GATE_NOT_YET_LOCKED` behavior should remain
unchanged. In particular, the equal-step PASS/handoff must not replace that
blocker automatically.
