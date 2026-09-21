# V3.2 website r9 scope/catalog closure (2026-09-05)

The r8 candidate had a metadata-only lineage defect: the scope sidecar's
`catalog_sha256` still pointed to the r6 catalog (`cdb620…`) although the
served r8 catalog and frontend copy were `e94894…`. The r8 candidate and all
of its receipts remain immutable.

An isolated successor was built on server `149` at:

```text
./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9
```

The r9 scope now declares the full SHA-256 of the final catalog, while the
catalog records a canonical scope digest computed after removing the forward
`catalog_sha256` field. This avoids an impossible full-file hash cycle. The
full file hashes are bound by the external pair receipt, which is the
authoritative bidirectional link.

| artifact | path | SHA-256 |
|---|---|---|
| scope | `capability_scope_20260905_r9.json` | `9d80c4f4c9bcd58bdbb77b95a2255ea5269c3bf6169dabd790acc3e7100d913c` |
| catalog | `catalog_candidate_20260905_r9/v32-capability-catalog_r9.json` | `f02bc1ba1b064a81dbd07a554fe94eef7d1b28c81f83b386ffcef24b902d62c1` |
| catalog canonical payload | — | `dcdf116dd39b9e50bb53f89af3a4a6d7a1a4d9ef798acaa26ce7f71adeed5a90` |
| frontend catalog | `code/website/frontend/v32-capability-catalog.json` | `f02bc1ba1b064a81dbd07a554fe94eef7d1b28c81f83b386ffcef24b902d62c1` |
| unified manifest | `code/config/v32_server_unified_staging_bindings_20260905_r9.json` | `56c8228e04a051df4371eae43fc2eab45a9e3575ba5cbbed226a2ec2a553e485` |
| external pair | `receipts/R9_SCOPE_CATALOG_PAIR_20260905.json` | `213cb63c6616d43897e832e61693641d865507808ebe131fa16df47c0baa6e4b` |
| pair audit | `receipts/R9_SCOPE_CATALOG_AUDIT_20260905.json` | `af02063f37fb890ad9560d6285dfa3a6671f346fe0a92d5737912dbdfa5bf31d` |
| independent release audit (r1) | `receipts/R9_INDEPENDENT_RELEASE_AUDIT_20260905.json` | `4f48511e3197e8cdaebfe0cf8c4fb23fd959d2bd307e80477af4b2b80d874f32` |
| independent release audit (r2, launcher amendment) | `receipts/R9_INDEPENDENT_RELEASE_AUDIT_20260905_r2.json` | `caedb47cc521283fff78b30a9a128eae2d58bf29b64e4faba888a2c94e0c01a6` |

The independent audit passed the manifest loader, all seven mounted hash
checks, scope/catalog pair checks, frontend byte equality, and the unchanged
fail-closed capability counts (22 `PENDING_FORMAL_BINDING`, 2
`QUERYABLE_STAGING`, 1 `PARTIAL_STAGING`). The r9 app constructed with 108
routes and the isolated route smoke passed 18/18 semantic probes. The smoke
used `allow-stale-catalog` only to avoid the old reverse-hash cycle; the new
pair audit independently verifies the canonical reverse hash and full-file
forward hash.

This is a staging closure, not a production release. `release_ready=false`,
`production_deployed=false`, and all formal-pending capabilities remain
pending. No GPU or training process was started, and production port 8261 was
not modified.

The reproducible checker is
`scripts/audit_v32_r9_scope_catalog_pair.py` (server copy SHA-256
`ae2d5af5e0d445545c2e6e7d91d2c31de0b90091ea15f6c2f2c0ecf8f316d179`).
The gated r9 server entry point is
`scripts/run_v32_website_staging_candidate_20260905_r9.py` (SHA-256
`53be24b859e77042fba94283980e93d6380709fb44985409910a4f8528232f0a`).

## Amendment (2026-09-05)

The hashes above were independently recomputed on host `149`.  The r9
independent audit was amended with the launcher/checker guard in
`R9_INDEPENDENT_RELEASE_AUDIT_20260905_r2.json` (SHA-256
`caedb47cc521283fff78b30a9a128eae2d58bf29b64e4faba888a2c94e0c01a6`).
The r9 candidate remains staging-only (`release_ready=false` and
`production_deployed=false`); no pending binding is promoted by this
metadata repair.
