# V3.2 website r7 deep-closure audit (2026-09-05)

This note records an independent check of the isolated server-native metadata
merge at:

`./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r7`

It does not promote r7, alter production service `8261`, or change any
authority result.

## What passed

- The r7 merge receipt is present on host `149` and has SHA-256
  `6312b954150002c88e4c02cffde3dcaae0d41a3820091c7d584a644feef0629b`.
- The manifest has 39 bindings: 7 `MOUNTED_HASH_PINNED` and 32
  `PENDING_FORMAL_SUCCESS`.
- All seven mounted sidecars, the copied registry, and both catalog copies
  exist, are not symlinks, and match their declared hashes.  The two catalog
  files are byte-identical (`cdb62039024e94b18b10910a97617554cce8b255e1268cc0a39716ace8c2088d`).
- The receipt correctly keeps `release_ready=false`, `production_deployed=false`,
  `training_started=false`, and `gpu_started=false`.

## What does not pass

The copied `RELEASE_REGISTRY.json` cannot be loaded from the r7 code root.  An
independent recursive check found 19 path references in the registry and 19
missing files under the paths resolved from that root, including the exact
pathway lineage, clinical/evidence/state/single-cell lineages, genomic
lineage, and mixed-query assets.  For example, the first loader failure is:

```text
module exact_pathway lineage does not exist:
./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r7/code/artifacts/v32_full_multitask/exact_pathway_release_r2/MODULE_LINEAGE.json
```

The r7 `api_route_smoke` is therefore correctly `NOT_RUN`.  A bounded attempt
to start the existing staging API implementation with the r7 manifest failed
before opening the isolated socket for this missing registry artifact; the
test process was stopped and temporary dependencies were removed.  No route
200 result is claimed for r7.

The r7 catalog also retains four absolute references into the older r6
`/dsk2` candidate, and `supersedes_manifest.path` is a non-existent
cross-mount relative path.  These are portability/lineage defects even though
the shallow sidecar hash check passes.

## Production-port separation check

A read-only socket check on host `149` observed an existing listener on
`127.0.0.1:8261` (Python PID `1133791`, working directory
`./data/CancerLncAtlas/runtime/cancerlncatlas_web_deployments/20260829T014500Z_web_remediation_candidate_r3`).
No action was taken against that process.  The r6 smoke and the r7 attempted
loader check were isolated from this port; the presence of this legacy service
must not be interpreted as an r7 route PASS or as a production promotion.

## Decision

r7 is a metadata merge with explicit gaps, not a runnable or publishable web
candidate.  Keep the previously audited r6 staging result as the authoritative
staging evidence until a new candidate includes a complete registry/artifact
closure (or a formally approved server-native registry rebinding).  Do not
start paid compute or modify production to work around this gap.
