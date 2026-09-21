# V3.2 corrected G0/G1/G2 GPU gate design (read-only check, 2026-09-05)

## Decision

The corrected run is **not yet launchable**. This is a technical gate, not an
approval request. No paid instance was started, no input was transferred, and
no sealed-test data was read during this check.

The only authoritative prepared input remains:

```text
./data/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1
```

It contains 15 local-CNV-corrected payloads (G0/G1/G2 × five patient folds),
totalling 182,363,721,367 bytes (169.839 GiB). The deleted pre-local-CNV
payload and its transfer archive are not valid fallbacks.

## Read-only gate evidence

| Gate | Evidence | Result |
|---|---|---|
| Authority host | `COMPUTE_HOST` reports short hostname `149` | PASS |
| Corrected input rebind | `g012_rebind_20260904_r5/SUCCESS.json`, SHA-256 `bba5951ccae1eb493de7805abed357661f93a41c8609485e2169361d79c1e153`; manifest SHA-256 `cc3f02c6f8e77f5079de1130d853ebb389f62097faf3dc45fc46c0e6adbcb793` | PASS |
| Current code bundle | code-tree SHA-256 `012955c029dc00987b89840874fbffb57cf10411d343367814d7b86762088dae`; archive SHA-256 `d9a08098e396dc95a2a6c0631f6e06d9ce5833b21d29440e55ca1b90c80d08f9` | PASS |
| Launcher | host-149 launcher SHA-256 `440f21f7de06781d360c545fefedd14896b06a922a052ef74484c12c92ca458e` | PASS (static) |
| Static authorization | `STATIC_AUTH_READY.json`, SHA-256 `08ed7192d4ae7236670d50c5605bf74fe419b09b5ad5cfcc70509c3c859f0346`; `training_started=false`, `gpu_started=false` | PASS (CPU-only) |
| Runtime launch preflight | `LAUNCH_PREFLIGHT.json` is absent; current static marker says `launch_preflight_present=false` | **BLOCKED** |
| Host CUDA | 149 has no `nvidia-smi`/CUDA device | **BLOCKED on 149** |
| CompShare endpoint | `uhost-1utsjo3ep1jz` is `Stopped`; one RTX 4090, 16 CPU, 64 GiB RAM, 250 GiB boot disk | **Runtime unverified** |
| Cloud capacity | 182.364 GB input leaves too little margin on a 250 GB system disk for OS, code, logs, checkpoints, and atomic transfer | **BLOCKED for full-copy plan** |

The provider's scheduled stop is `2026-09-07T09:19:49+08:00`; the stopped
disk rate is about 0.07 CNY/hour and compute is about 2.05 CNY/hour. The
endpoint is intentionally left stopped.

## Why the cloud endpoint cannot be used automatically

The current launcher has a hard execution guard:

```text
expected_host=149
```

and exits before CUDA or training if the actual short hostname differs. A
stopped CompShare container does not expose an authoritative runtime hostname,
mount table, Python environment, or CUDA probe. Starting it merely to discover
these values would create paid compute without a ready launch contract. The
hostname must not be spoofed and the guard must not be bypassed.

The existing server-side compatibility receipt records this same fail-closed
state:

```text
./data/CancerLncAtlas/runtime/compatibility_blocks/v32_g012_gpu_host_20260905_r1/v32_g012_gpu_host_compatibility_block_20260905.json
status: BLOCKED_CLOUD_ENDPOINT_HOST_CONTRACT_AND_RUNTIME_UNVERIFIED
```

## Safe preparation that does not start paid compute

The following can be done on host 149 or as small local coordination files:

1. Keep the corrected 15 payloads and r5 manifest immutable; do not rerun the
   rebind or regenerate payloads.
2. Keep the current code/config/launcher hashes and per-variant static
   authorization manifests together as the candidate package.
3. Choose one execution contract before any provider call:
   - attach a real GPU to host `149` (current launcher works after a fresh
     runtime preflight), or
   - explicitly revise the project contract to
     `authority_host=149`, `runtime_host=<verified CompShare hostname>` and
     issue a new launcher/static authorization hash. This is a contract
     change, not a hostname alias.
4. For the 250-GB endpoint, do not copy the complete 182-GB tree. A future
   fold-streaming or server-mounted design would need a new loader/launcher,
   a bounded transfer manifest, and semantic/hash checks. The existing
   monolithic payload launcher cannot silently switch to that layout.
5. Once the contract and capacity plan are closed, start the instance once,
   immediately run hostname/mount/disk/`nvidia-smi`/CUDA/PyG/Python checks,
   emit a fresh `LAUNCH_PREFLIGHT.json`, and require non-empty optimizer-step
   output plus GPU telemetry. On any failure or idle bootstrap, stop it.

No paid start is justified before steps 3–4 are complete. No sealed-test read
is allowed before the hierarchical-gating versus external-router fair
comparison and winner lock.

## Current status marker

```text
status=BLOCKED_RUNTIME_HOST_CONTRACT_AND_CAPACITY
training_started=false
gpu_started=false
input_transfer_started=false
sealed_test_read=false
```

