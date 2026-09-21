# CancerLncAtlas V3.2 objective-completion audit (2026-09-05)

This is a read-only completion audit of the three persistent objectives.  The
authoritative checks were performed on server `149` (`COMPUTE_HOST`); no GPU,
training process, data rescue, or production service was started or changed.
Immutable receipts were not rewritten.

## 1. Single-cell scope: closed at 23 + 10

**Status: PASS for the accepted partial scope; full 33-cancer coverage is not
claimed.**

Authoritative result:

```text
./data/CancerLncAtlas/results/single_cell_r11_formal23_20260903_r1/SUCCESS.json
```

- `formal_eligible_cancer_count = 23` and `typed_unavailable_cancer_count = 10`.
- The two sets are disjoint and their union is exactly the 33-cancer universe.
- `typed_unavailable_rows_are_null = true` and
  `typed_unavailable_changes_primary_score = false`.
- `total_cells = 1,428,374`; `total_association_evidence_rows = 15,563,738`.
- The formal binding hash recomputes to
  `b43cc5ac723f18c3c63e41831b8619b4ae634584e51c1961708a651ccd311099`.
- The independent audit hash recomputes to
  `be746678952e3598f639f9b6258ec7ceccf9d4c5a2b68a04035451d89472142b` and its
  status is `PASS_23_OF_23_INDEPENDENTLY_VERIFIED`.

The ten typed-unavailable reasons are contract/data-authority failures, not a
missing retry:

- BLCA, LIHC, LUAD, PAAD, PRAD, STAD and THCA: no auditable accession → sample
  → individual mapping.
- KICH and KIRP: only three independent donors, below the five-donor gate.
- UCS: only 38 formal lncRNA features after the accepted release filtering.

The scope is frozen by
`docs/V32_SINGLE_CELL_SCOPE_CLOSED_20260904.md`.  No additional download,
donor search, or rescue should be scheduled unless a new independently
verifiable authority is supplied.  This overlay is independent of the primary
HHGT score and does not invalidate primary-score coverage.

## 2. Local-CNV confounding audit: corrected lineage is authoritative

**Status: PASS for the audit and independent CNV overlay; primary retraining is
required and has not started.**

Authoritative audit root:

```text
./data/CancerLncAtlas/results/v32_local_cnv_confounding_audit_20260902_r3/
```

The `LOCAL_CNV_INDEPENDENT_AUDIT.json` and `LOCAL_CNV_DECISION.json` hashes
recompute against the `SUCCESS.json` declarations.  The independent audit is
`PASS` and the frozen decision is:

```text
LOCAL_CNV_CORE_RETRAIN_REQUIRED
```

Key observed effects:

- baseline positive-label flip rate: `0.18273194012582256`;
- direction flip rate: `0.049638150218087616`;
- median pathway-rank Spearman: `0.977797585284823`;
- median G0 edge Jaccard: `0.6230302453332099`.

The separate CNV-only five-fold OOF is also successful:

```text
./data/CancerLncAtlas/results/v32_cnv_head_oof_20260902_r6/
```

Its manifest explicitly records `cnv_only=true`,
`mutation_features_used=false`, `old_checkpoint_loaded=false`, and
`old_predictions_used=false`; global log loss is `0.18120417074813497` over
`6,361,208` available rows.  This is an independent overlay and is not evidence
of an improved primary score.

The corrected G0/G1/G2 input lineage is the 15-payload root:

```text
./data/CancerLncAtlas/inputs/v32_g012_local_cnv_formal_prepared_20260903_r1
```

The one-time r5 rebind/hash closure is complete; no payload was copied or
rewritten.  The former pre-local-CNV baseline subtree and matching transfer
archive were deleted at the user's direction and verified absent.  The parent
authority directory remains only for static graph/fold metadata referenced by
the corrected lineage.

The historical reconciliation table contains earlier “rebind pending” wording;
its 2026-09-05 amendment supersedes that snapshot.  Current state is
`corrected input/rebind complete`, while `primary G0/G1/G2 optimizer training`
is `not started`.

## 3. Website: isolated self-contained staging, not production release

**Status: PASS as an isolated staging candidate with explicit gaps; not yet a
production release.**

Current candidate on `149`:

```text
./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9
```

Evidence:

- independent release-audit status: `PASS_R9_STAGING_WITH_EXPLICIT_GAPS`;
- independent release-audit SHA-256:
  `4f48511e3197e8cdaebfe0cf8c4fb23fd959d2bd307e80477af4b2b80d874f32`;
- unified manifest SHA-256:
  `56c8228e04a051df4371eae43fc2eab45a9e3575ba5cbbed226a2ec2a553e485`;
- catalog SHA-256:
  `f02bc1ba1b064a81dbd07a554fe94eef7d1b28c81f83b386ffcef24b902d62c1`;
- capability-scope SHA-256:
  `9d80c4f4c9bcd58bdbb77b95a2255ea5269c3bf6169dabd790acc3e7100d913c`;
- scope/catalog pair-audit SHA-256:
  `af02063f37fb890ad9560d6285dfa3a6671f346fe0a92d5737912dbdfa5bf31d`;
- strict route-smoke SHA-256:
  `d764edbf6b1cd844fa9c90c368334a7335761ba6cd6493a572931a9dc3852e10`.

The route-smoke receipt itself is authoritative for the probe result: 18/18
strict probes and 7/7 overlay/download probes returned HTTP 200 with semantic
success.  `create_app` constructed 108 routes.  The candidate is self-contained
under its own root, and the seven mounted hashes plus the no-cycle
scope/catalog pair were independently recomputed on `149`.  The existing
service on port 8261 was observed only and was not touched.  The dedicated r9
launcher (host/port guard and finite shutdown) has SHA-256
`53be24b859e77042fba94283980e93d6380709fb44985409910a4f8528232f0a`.

### Audit finding: stale catalog hash inside the scope sidecar

The r8 scope sidecar itself recomputes correctly to
`e39ff3d5ed05cad91b9d11c3897d7a2735a2215c103c6a68267133280823f944`, but its
internal `catalog_sha256` still declares the historical r6 catalog
`cdb62039024e94b18b10910a97617554cce8b255e1268cc0a39716ace8c2088d`.  The
actual r8 catalog and byte-identical frontend catalog both recompute to
`e94894cfcda479cc812f42ffccb8d5355bc77a2c8813d6fb87d15d6da94d9759`.
The existing independent r8 audit checked the scope file hash and status but
did not check this nested scope-to-catalog declaration.  This is a metadata
lineage defect, not evidence that the route probes or scientific payloads are
wrong.  Do not edit the immutable r8 candidate or its receipts in place; the
website workstream repaired the field in the isolated r9 candidate and reran
the scope/catalog/manifest audit pair.  The r9 pair audit is PASS; r8 remains
immutable diagnostic history.

The manifest deliberately remains fail-closed:

- 39 total bindings;
- 7 `MOUNTED_HASH_PINNED`;
- 32 `PENDING_FORMAL_SUCCESS`;
- catalog-level status: 2 `QUERYABLE_STAGING`, 1 `PARTIAL_STAGING`, 22
  `PENDING_FORMAL_BINDING`.

A mounted sidecar is not silently promoted to a canonical `/api/site/*`
capability.  The pending entries are therefore real publication gaps, not a
single-cell rescue queue.  `release_ready=false` and `production_deployed=false`
are intentional.

The reproducible operator command and production-port/loopback guards are
recorded in `docs/V32_WEBSITE_R9_RELEASE_RUNBOOK_20260905.md`.

A bounded server-149 inventory confirmed that all 32 pending manifest entries
have no current path/SHA closure; only the seven mounted sidecars are
authoritatively queryable in this candidate.  No old baseline payload is used
by the r9 runtime, and the deleted pre-local-CNV payload/archive are absent.

The corresponding server-side inventory receipt is
`./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9/receipts/R9_PENDING_BINDING_INVENTORY_20260905.json`
(SHA-256
`d7becfc03143c2ad4a43c6100be19443acb22d489bba6f2e96c4eecaa0ab9617`).
It records 32 pending entries and zero safely closable entries.  This was a
bounded manifest-only check and did not inspect production or sealed-test data.

## Remaining gates before production publication

1. Execute corrected G0/G1/G2 primary training on a verified CUDA endpoint;
   149 currently has no visible GPU and no G012 optimizer process.
2. Run the predeclared fair comparison of hierarchical gating versus external
   router, then lock the winner.
3. Run the sealed-test and final release acceptance only after the winner lock.
4. For any website capability intended for production, produce a fresh,
   self-contained binding plus independent audit and route-smoke pair; do not
   infer canonical availability from a mounted overlay.

The paid GPU instance remains stopped.  No cost-bearing compute should be
enabled until the active instance, corrected input manifest, code hashes,
runtime preflight, and launch receipt are all verified and the job can
immediately demonstrate optimizer progress plus GPU telemetry.
