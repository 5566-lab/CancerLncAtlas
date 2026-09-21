# V3.2 corrected G0/G1/G2 GPU readiness (2026-09-05)

## Current state

The corrected local-CNV input lineage is closed and ready for a runtime
preflight.  The one-time r5 rebind covers all 15 payloads (G0/G1/G2 × five
patient folds), and the host-149 static authorization records
`old_baseline_used=false`.  No corrected primary optimizer run has started.

The superseded pre-local-CNV payload and its transfer archive were deleted;
neither is an allowed fallback or baseline.

## Why the paid instance is still stopped

The only CompShare endpoint, `uhost-1utsjo3ep1jz`, is `Stopped`.  Its
read-only metadata reports one RTX 4090 and a 250-GB system disk, but a stopped
container cannot provide authoritative hostname, mount, Python, or CUDA
checks.  Host 149 has no `nvidia-smi`.  The current launcher intentionally
requires the actual execution host to be `149` and rejects a different host
before any CUDA call.  The corrected payloads total about 182 GB, so the
250-GB disk is not a safe full-input target after system/code/log overhead.

The compatibility receipt is:

```text
./data/CancerLncAtlas/runtime/compatibility_blocks/v32_g012_gpu_host_20260905_r1/v32_g012_gpu_host_compatibility_block_20260905.json
SHA-256: 78c36b7ec41491f02c84a426dd25b65cfba438408a79ca468f5a67a644e2cbb2
status: BLOCKED_CLOUD_ENDPOINT_HOST_CONTRACT_AND_RUNTIME_UNVERIFIED
```

This is a technical gate, not an approval request.  Do not spoof the hostname,
bypass the launcher, or start a paid instance merely to discover whether it
works.

## Conditions for a real run

Before enabling paid compute, create a fresh launch preflight that binds:

1. the actual runtime host/instance identity (or an explicitly reviewed
   `authority_host=149` / `runtime_host=<cloud-host>` contract);
2. the corrected input manifest and all 15 payload paths;
3. the current code/config/launcher hashes;
4. a verified storage/transfer plan with sufficient capacity; and
5. working `nvidia-smi`, CUDA, PyG, and Python environment checks.

After launch, the gate is not satisfied by a live bootstrap process.  It must
show non-empty optimizer-step output and GPU telemetry.  On failure or absent
progress, stop the instance automatically.  After verified results, stop and
delete the instance and its disk.

No sealed-test data may be read before the hierarchical-gating versus external
router comparison and winner lock.
