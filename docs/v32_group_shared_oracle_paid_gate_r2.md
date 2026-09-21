# V3.2 group-shared oracle-only paid gate r2

Status: `IMPLEMENTED_AND_LOCALLY_TESTED_NOT_DEPLOYED_NOT_RUN`

This gate authorizes only the preregistered real-data comparison callable. It
does not authorize G0/G1/G2 training, checkpoint creation, prediction export,
winner selection, or use of a formal results directory. A PASS still requires
the separately authorized equal-step G0/G1/G2 pilot before any formal-training
decision.

## Fixed scope

- namespace: `v32_group_shared_oracle_paid_gpu_20260901_r2`
- run: `v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r2`
- task: exactly `G2 / PATIENT_FOLD_0 / seed 20260726`
- callable: `cc_hhgt.v32.group_shared_encoder_oracle:run_authorized_oracle_comparison`
- maximum elapsed paid time: 3 hours
- comparison cost ceiling: CNY 8
- effective cost ceiling: `min(CNY 8, conservative JIT project budget remaining)`
- only receipt sink: JSON lines on stdout; the launcher may retain one
  comparison execution log below its bootstrap namespace

## Formal post-Oracle blocker

The separate formal-15 validator is deliberately outside this CPU/Oracle
bundle and still carries the preceding Oracle verifier/deployment pins. Formal
work remains blocked after the comparison until that validator is updated to
the new Oracle verifier SHA
`5395798726e64ccb405cf3acdc560d6c95606ab91c940bbbdd5ee991d7f6512c`
and external deployment manifest SHA
`bda800013f71808f3b83bba226d2ed3b527ac3db2790aee5681f8ce77d295665`,
then its formal archive, lock, and receipt are rebuilt and independently
verified. This deferral does not affect the comparison-only execution path.

## Trust separation

The bootstrap verifier must be copied to:

`./data/CancerLncAtlas/runtime/oracle_gates/v32_group_shared_oracle_paid_gpu_20260901_r2/`

That external control directory must also contain the exact launcher,
CPU/no-GPU authorizer, local supervisor, local dispatch controller, log-guard
module, and `EXTERNAL_DEPLOYMENT_MANIFEST.json`. The checked-in manifest is
`docs/v32_group_shared_oracle_external_deployment_manifest_20260901_r2.json`
with SHA-256
`bda800013f71808f3b83bba226d2ed3b527ac3db2790aee5681f8ce77d295665`.
It freezes all six control artifacts. The external verifier itself is frozen
at SHA-256
`5395798726e64ccb405cf3acdc560d6c95606ab91c940bbbdd5ee991d7f6512c`;
the shell callers compute this hash independently before every invocation.

The staged, executable overlay must be extracted separately to:

`./data/CancerLncAtlas/runtime/tools/v32_group_shared_oracle_paid_gpu_20260901_r2/code/`

The external verifier does not import or execute the overlay. It parses the
runner with `ast`, reads YAML/JSON/text as data, opens regular files with
`O_NOFOLLOW`, and hashes through the same file descriptor bracketed by two
`fstat` identity checks. Files with `st_nlink != 1` are rejected, preventing
an untrusted hardlink alias from mutating an authority inode. The no-GPU
authorizer and paid launcher use the same non-blocking `flock` file.

The static authority is accepted only when all of the following bind:

1. The returned and server-archived r1 r6 `STATIC_AUTH_READY` receipts are
   byte-identical with SHA-256
   `1f5bc54b2a8acc037373bfa380c95c8a717e76fde956852ec481b16a0d8716f0`.
2. The r1 `ABORTED.json` says `ABORTED_RUNTIME_INFEASIBLE`, removes the live
   authorizations, and forbids resume or reuse of formal artifacts.
3. All 15 G0/G1/G2 fold hash/size rows in the existing r2
   `INPUT_REUSE_READY.json` match the anchored r1 r6 authority. The current PT
   files are same-fd size checked but are not re-hashed during this gate.
4. G2/Fold0 is explicitly anchored to SHA-256
   `4ebba1d4060f3efaa8c080824386802ca732bedf07681ebd0283deb087b0ca39`
   and 4,482,708,589 bytes.
5. The code archive receipt, extracted code tree, comparison config, one-row
   task manifest, one-row input manifest, generated approval, oracle runner,
   preregistration, and formal decision receipt all match their hashes.
6. The JIT billing receipt is unexpired, no more than 45 minutes wide, names
   the same instance, and binds the fixed `default` profile and project scope.
   The local dispatcher and supervisor require at least 1,800 seconds before
   any provider API or paid start; the frozen remote validator requires at
   least 900 seconds at the full static gate and immediately before CUDA.
7. The external deployment manifest and every server copy of the six control
   artifacts match their frozen hashes and sizes. `STATIC_AUTH_READY` carries
   the manifest hash and complete record list; the dispatch controller
   re-hashes the local copies, including itself, before any start.

## JIT billing receipt

Before the CPU-only static gate, the external billing controller writes
`JIT_BUDGET_READY.json` under the comparison bootstrap namespace. Its minimum
contract is:

```json
{
  "format": "CANCERLNCATLAS_COMPSHARE_JIT_BUDGET_V1",
  "status": "JIT_BUDGET_READY_CONSERVATIVE",
  "currency": "CNY",
  "source": "COMPSHARE_API_BILLING_SNAPSHOT",
  "project_budget_cap_cny": 210.0,
  "compute_hourly_cny": 2.05,
  "gpu_hourly_cny": 2.05,
  "disk_hourly_cny": 0.04,
  "total_hourly_cny": 2.09,
  "instance_id": "<exact-instance-id>",
  "profile_name_sha256": "37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f",
  "project_scope_sha256": "726631660639f60035b7774f8be7e090618bf083bc400a1580f4050f02a861fd",
  "queried_at": "<timezone-aware ISO-8601>",
  "valid_until": "<timezone-aware ISO-8601, at most 45 minutes later>",
  "conservative_remaining_cny": "<positive number no greater than 210>"
}
```

The external supervisor independently reads both this receipt and the returned
static authority. It stops the instance upon the first observed PASS receipt,
the first typed failure/exit 42, the effective cost deadline, the 3-hour hard
deadline, a fatal log signature, no initial heartbeat within 15 minutes, or no
new assignment progress for 5 minutes. Provider-status failures have a
12-minute grace only while the dispatch phase is `PRE_SUBMIT`; the controller
writes `SUBMIT_IN_FLIGHT` and then `SUBMITTED`, after which the grace is exactly
3 minutes. The supervisor resets the status-failure clock at that phase
transition. A cloud stop request is attempted before local error logging.
After the provider reports `Stopped`, the supervisor writes the local typed
receipt `CANCERLNCATLAS_GROUP_SHARED_ORACLE_AUTOSTOP_V1` with status
`INSTANCE_STOPPED_STATE_CONFIRMED`, the stop trigger, instance/job IDs, static
and JIT receipt hashes, and `gpu_billing_active=false`.

The supervisor parses complete JSON lines. A PASS/FAIL-looking substring is
not terminal. A terminal receipt is accepted only when its exact format,
status, run ID, task ID, `G2`, fold 0, seed 20260726, `scientific_pass`, zero
formal-artifact count, and all false formal-output flags match. The exact line
is revalidated after the provider confirms `Stopped`. Heartbeats are keyed by
full-line SHA, semantic event ID, phase, and monotonic completed-assignment
count, so repeated `--tail 1200` windows cannot reset the stall timer.

## One-click dispatch controller

`scripts/local_dispatch_compshare_group_shared_oracle_paid_gpu_20260901_r2.ps1`
is the only paid-start entry point. Before GPU start it:

1. validates the returned CPU-only static receipt and fresh JIT receipt;
2. validates every frozen local control artifact against the static-bound
   deployment manifest;
3. queries the provider's current stopped state, RTX 4090 spec, and billing,
   requires the fixed JobId to be absent, then writes a separate local
   dispatch-JIT receipt;
4. computes one absolute stop time from the original JIT query time and the
   earlier of the 3-hour and conservative-cost deadlines;
5. sets and reads back the provider shutdown schedule while still stopped;
6. starts the supervisor in a background runspace in the same PowerShell
   process and waits for its first fixed-JobId poll;
7. starts the instance, confirms the deadline was not extended, and submits
   only JobId `v32-oracle-g2f0-s20260726-r2` and the fixed launcher; and
8. accepts job logs only after structured provider metadata proves the exact
   JobId, job name, working directory, launcher command, and a creation time no
   earlier than this dispatch attempt. A retained JobId from any earlier
   attempt is rejected before `instance start` or `job submit`.

Every start, submit, monitor, or controller failure converges on an
unconditional stop-and-provider-confirmation `finally` block. The provider
schedule remains an independent insurance path if local control fails. The
controller contains no interactive prompt.

## Local verification

The focused suite includes a compiled CompShare stub proving the command order
`schedule -> supervisor poll -> start -> idempotent submit -> stop`, plus
an immediate second-attempt rejection when the first attempt's JobId is
retained, and tamper fixtures for every control artifact, hardlinks, repeated
old heartbeats, stale terminal identity, PASS substrings, and unsafe terminal
flags. The focused paid gate, oracle, generic training-guard, and GPU
backward-probe regression set currently reports `61 passed` under the complete
project Python 3.11 conda runtime.

No cloud API was contacted, no instance was started, and the formal r2 config
was not modified while implementing or testing this gate.
