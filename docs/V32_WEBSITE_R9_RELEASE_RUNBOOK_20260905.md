# CancerLncAtlas V3.2 website r9 staging runbook

This runbook describes the isolated, hash-pinned staging candidate on host
`149`. It is not a production promotion and must not be used to modify the
existing listener on port `8261` (or legacy ports `8260` and `8262`).

## Candidate and integrity pins

```text
root      ./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9
manifest  code/config/v32_server_unified_staging_bindings_20260905_r9.json
catalog   catalog_candidate_20260905_r9/v32-capability-catalog_r9.json
scope     capability_scope_20260905_r9.json
```

| artifact | SHA-256 |
|---|---|
| manifest | `56c8228e04a051df4371eae43fc2eab45a9e3575ba5cbbed226a2ec2a553e485` |
| catalog | `f02bc1ba1b064a81dbd07a554fe94eef7d1b28c81f83b386ffcef24b902d62c1` |
| scope | `9d80c4f4c9bcd58bdbb77b95a2255ea5269c3bf6169dabd790acc3e7100d913c` |
| pair audit | `af02063f37fb890ad9560d6285dfa3a6671f346fe0a92d5737912dbdfa5bf31d` |
| independent audit r2 | `caedb47cc521283fff78b30a9a128eae2d58bf29b64e4faba888a2c94e0c01a6` |
| launcher | `53be24b859e77042fba94283980e93d6380709fb44985409910a4f8528232f0a` |

## Finite staging start

Run only on `149`, with the `${PRIVATE_WORK_ROOT}/miniconda3/bin/python3.13`
environment (the `${PRIVATE_WORK_ROOT}` SSHFS virtualenv is not the preferred runtime for
SciPy extension loading):

```bash
cd ./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9/code
${PRIVATE_WORK_ROOT}/miniconda3/bin/python3.13 \
  scripts/run_v32_website_staging_candidate_20260905_r9.py --port 8303
```

The launcher first reruns the scope/catalog pair audit, requires host `149`,
allows loopback only, and rejects ports `8260`–`8262`. It never starts CUDA or
touches the production listener. A finite smoke must follow startup, and the
process must be stopped afterward. A live listener is not evidence of a
successful release.

## Verified state

- `create_app`: 108 routes, PASS.
- Route smoke: 18/18 strict semantic probes and 7/7 overlay/download probes,
  PASS.
- Catalog status: 2 `QUERYABLE_STAGING`, 1 `PARTIAL_STAGING`, 22
  `PENDING_FORMAL_BINDING`.
- `release_ready=false`, `production_deployed=false`.
- Formal single-cell scope is 23 cancers plus 10 typed-unavailable; unavailable
  values are explicit nulls and do not alter the primary score.
- Directional CNV is an independent overlay; it does not change the primary
  score.

## Release blockers that must remain visible

The 22 pending canonical bindings are genuine missing formal-success/hash
closures, not a single-cell rescue queue. Do not promote a mounted sidecar to
`/api/site/*` merely because an overlay route works. Production publication
also waits for corrected primary G0/G1/G2 training, fair hierarchical-gating
versus external-router comparison, winner lock, sealed-test inference, and
final acceptance. No paid GPU may be started until its host/runtime and launch
preflight are independently verified and it can immediately show optimizer
progress plus GPU telemetry.

The superseded r8 candidate is retained only as immutable diagnostic history;
its nested scope/catalog hash was stale. The pre-local-CNV baseline payload and
transfer archive were deleted and are not fallback inputs.
