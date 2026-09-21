# V3.2 Single-Cell Scope Closure

Date: 2026-09-04

## Frozen decision

The formal single-cell scope is frozen at 23 eligible cancers plus 10
typed-unavailable cancers. Full 33-cancer coverage is not claimed, is not a
release gate, and must not be represented by zeros. The independent formal-23
overlay is an optional evidence module with zero fusion weight; it does not
alter the primary HHGT score.

Authoritative server receipt:

`./data/CancerLncAtlas/results/single_cell_r11_formal23_20260903_r1/SUCCESS.json`

It reports 1,428,374 cells and 15,563,738 association-evidence rows. The
formal binding and independent audit are hash-pinned in that receipt.

The receipt above is the server-side derived formal-23 overlay. It must not
be confused with the earlier R11 input-contract receipt
(`runtime/single_cell_cell_level_r11_rescued_20260903_r1/SUCCESS.json`), which
records the accepted cohort/source contract only. The input contract is not
 evidence that another rescue or derivation is still required.

The release-scope policy used to make this decision is
`config/v32_single_cell_release_scope_20260903_r2.json` (local SHA-256
`4817BA6D692D1B4700B3D17DDCD16145A85E148AA1B84352F42B1D1A5FD86C55`). The
server SUCCESS file is pinned by the staging audit at SHA-256
`daa8d05e61a57b87a3ddf2da3542cf78c97c345483c4311002635eed9656ee1e`; its
internal formal-binding SHA is
`b43cc5ac723f18c3c63e41831b8619b4ae634584e51c1961708a651ccd311099` and its
independent-audit SHA is
`be746678952e3598f639f9b6258ec7ceccf9d4c5a2b68a04035451d89472142b`. These
are evidence/overlay receipts, not a claim that the primary HHGT model has
been retrained.

The frozen cancer lists are:

- Formal eligible (23): `ACC, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM,
  HNSC, KIRC, LAML, LGG, LUSC, MESO, OV, PCPG, READ, SARC, SKCM, TGCT,
  THYM, UCEC, UVM`.
- Typed unavailable (10): `BLCA, KICH, KIRP, LIHC, LUAD, PAAD, PRAD, STAD,
  THCA, UCS`.

## Cross-file closure audit

This closure was checked against the current mainline status, the rescue
decision, the release-gate audit, and the checked-in website catalog on
2026-09-04.

The status and decision documents agree on the frozen `23 + 10` scope and on
the three typed-unavailable reason classes. The checked-in
`website/frontend/v32-capability-catalog.json`, however, is an older local
catalog snapshot. Its single-cell section still contains
`current_fresh_derived_cancers=0`, `derived_assets_bound=false`,
`UNBOUND_FRESH_RECOMPUTE_REQUIRED`, and `fresh outputs remain 0/23`. Those
strings describe the pre-binding local checkout, not the server formal-23
result, and must not be read as an open rescue queue. Until a server-generated
catalog with the formal-23 sidecar is mounted, the older catalog is explicitly
**non-authoritative for current single-cell availability**.

The corresponding release-gate action is a staging binding/catalog refresh,
not another download, donor search, or scientific recomputation. A future
catalog may report the formal-23 overlay as hash-bound while retaining
`release_ready=false`; it must retain the ten typed-unavailable values as
explicit null/unavailable responses.

## Why the rescue stopped

The failed rescue attempts were contract failures, not an absence of all
expression data:

- BLCA, LIHC, LUAD, PAAD, PRAD, STAD and THCA lack an auditable source
  accession-to-individual mapping.
- KICH and KIRP have only three independent donors, below the five-donor
  formal replication gate.
- UCS (GSE299623) exposes only 38 formal lncRNA features after the accepted
  release filtering.

The six promoted datasets and their donor/sample corrections are recorded in
`docs/V32_SINGLE_CELL_RESCUE_DECISION_20260903.md`. No additional download or
rescue should be scheduled unless a new, independently verifiable authority is
provided. Future status reports should cite this closure instead of reopening
the same rescue as a blocker.

## No-loop policy

For this frozen scope, `rescue pending`, `fresh 0/23`, or `download more
single-cell data` is not a valid next-step recommendation. Reopening the
rescue is allowed only when a new, independently verifiable accession,
sample-to-individual authority, or feature-complete source is supplied and
its provenance is independently audited. Otherwise the next permitted work is
limited to binding/query smoke and documentation of the existing 23-eligible
 overlay; it does not change the primary HHGT score.

This is an operational closure, not a request to keep polling the server. A
single non-mutating reachability check on 2026-09-04 timed out before the
server receipt could be reread; the already recorded, hash-pinned server
receipts and independent audit remain the authority. No repeated connection
attempt, download, donor search, or rescue job is warranted merely because
149 is temporarily unreachable.

## Runtime semantics

Formal associations, activity and UCell assets may be bound in staging for the
23 eligible cancers. Pseudotime and figures remain typed-unavailable. Queries
for the ten excluded cancers return typed-unavailable/null; they never reduce
primary-score coverage or ranking.

`release_ready=false` and `production_deployed=false` remain until the primary
model and final website release gates are complete.

## Binding amendment (2026-09-05)

The server-side r9 isolated staging candidate now contains the hash-bound
formal-23 overlay and has passed its finite route and overlay/download smoke.
The older checkout/catalog warning above remains historical context only; it
is not an open rescue task. The r9 candidate still reports
`release_ready=false` and `production_deployed=false`, and the accepted 23+10
scope is unchanged. The r9 independent release audit is:

- `runtime/web_candidates/v32_20260905_r9/receipts/R9_INDEPENDENT_RELEASE_AUDIT_20260905_r2.json`
  (SHA-256 `caedb47cc521283fff78b30a9a128eae2d58bf29b64e4faba888a2c94e0c01a6`)

The prior r8 candidate is retained only as immutable diagnostic history; its
nested scope/catalog hash was stale and it is not the active staging root.

No further single-cell download, donor search, or rescue should be scheduled
unless a new independently verifiable authority is supplied.
