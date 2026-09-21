# CancerLncAtlas V3.2 release-gate matrix (2026-09-05)

This is the current handoff-level release check.  It separates a route-smoked
staging candidate from a production release and does not alter the existing
service on port 8261.

| Gate | Current evidence | Status |
|---|---|---|
| Single-cell scope | R11 formal overlay: 23 eligible cancers + 10 explicit typed-unavailable; rescue is closed | PASS (accepted partial scope) |
| Local-CNV audit | Independent audit and five-fold directional CNV OOF are hash-bound; overlay is independent of the primary score | PASS (staging) |
| Corrected G0/G1/G2 inputs | 15 local-CNV-corrected payloads, r5 rebind and no-GPU static authorization | PASS (static only) |
| Website route smoke | Isolated r9 candidate: 18/18 strict semantic probes plus 7/7 overlay/download probes; r8 retained as immutable diagnostic history | PASS (staging only) |
| Website metadata merge | r9 has 7 hash-pinned mounted bindings + 32 explicit pending bindings; scope/catalog pair and active paths are self-contained under the r9 root | PASS (staging closure; not production) |
| Production service | Existing 149 loopback listener on 8261 was observed only and left untouched | NOT PROMOTED |
| Corrected primary training | No G0/G1/G2 optimizer or GPU telemetry receipt exists | PENDING |
| Fair candidate comparison | Hierarchical gating vs external router, winner lock | PENDING |
| Sealed test and final acceptance | Must follow winner lock; no sealed-test read is recorded | PENDING |

## Authoritative staging evidence

- r9 candidate root on host 149:
  `./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9`.
- Independent r9 release audit:
  `./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9/receipts/R9_INDEPENDENT_RELEASE_AUDIT_20260905.json`
  (SHA-256 `4f48511e3197e8cdaebfe0cf8c4fb23fd959d2bd307e80477af4b2b80d874f32`),
  status `PASS_R9_STAGING_WITH_EXPLICIT_GAPS`.
- r9 scope/catalog pair audit:
  `./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r9/receipts/R9_SCOPE_CATALOG_AUDIT_20260905.json`
  (SHA-256 `af02063f37fb890ad9560d6285dfa3a6671f346fe0a92d5737912dbdfa5bf31d`),
  with external pair receipt SHA-256
  `213cb63c6616d43897e832e61693641d865507808ebe131fa16df47c0baa6e4b`.
- r9 unified manifest SHA-256:
  `56c8228e04a051df4371eae43fc2eab45a9e3575ba5cbbed226a2ec2a553e485`;
  catalog SHA-256 `f02bc1ba1b064a81dbd07a554fe94eef7d1b28c81f83b386ffcef24b902d62c1`;
  capability-scope SHA-256
  `9d80c4f4c9bcd58bdbb77b95a2255ea5269c3bf6169dabd790acc3e7100d913c`;
  route-smoke SHA-256
  `d764edbf6b1cd844fa9c90c368334a7335761ba6cd6493a572931a9dc3852e10`.
- The r9 audit recomputed all seven mounted binding hashes, verified the
  no-cycle scope/catalog hash contract, confirmed destinations are inside the
  r9 root, and found no active external path dereference.
- Detailed r9 scope/catalog closure note:
  `docs/V32_WEBSITE_R9_SCOPE_CATALOG_PAIR_20260905.md` (SHA-256
  `00359a6585c52adb85dbaac55cfcc712e77a68af7fe8b52a87852ba9e13dd119`).
- Operator runbook:
  `docs/V32_WEBSITE_R9_RELEASE_RUNBOOK_20260905.md`.
- The catalog deliberately remains conservative at the capability level:
  2 canonical routes are `QUERYABLE_STAGING`, 1 is `PARTIAL_STAGING`, and 22
  remain `PENDING_FORMAL_BINDING`.  A mounted sidecar or a successful
  overlay/download probe does not silently promote an unverified canonical
  route.
- Historical r6 candidate receipt on host 149:
  `${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260905_r6/receipts/WEBSITE_STAGING_CANDIDATE_20260905_r6.json`
  (SHA-256 `db57ff8b159a820ae71745f6765fd62687e45372283f01b6c5955fbd2730e76d`).
- Historical r6 catalog SHA-256: `cdb62039024e94b18b10910a97617554cce8b255e1268cc0a39716ace8c2088d`.
- r7 deep-closure audit remains available as historical diagnosis:
  `V32_WEBSITE_R7_DEEP_CLOSURE_AUDIT_20260905.md`.
- Corrected static authorization SHA-256:
  `08ed7192d4ae7236670d50c5605bf74fe419b09b5ad5cfcc70509c3c859f0346`.

- Requirement-by-requirement completion audit:
  `docs/V32_OBJECTIVE_COMPLETION_AUDIT_20260905.md` (SHA-256
  `6ddeb583e2893ac807d6e37c8fbe09f47413756e30de46b9ccc074ed51627717`).

## Release decision

The website is now a self-contained, isolated r9 staging candidate for the
seven mounted modules and their proven routes, with explicit unavailable
responses for the 32 pending bindings.  It is not yet a production release:
the corrected primary model has not completed, the two main candidates have
not undergone a fair comparison, and winner-lock/sealed-test/final acceptance
remain pending.

The paid GPU remains stopped.  It may be enabled only after a fresh runtime
preflight binds the active instance, code, corrected input manifest and launch
receipt and then immediately demonstrates optimizer progress plus GPU
telemetry.  No additional single-cell rescue or input rebind is required.  The
prior r8 candidate is retained only as immutable diagnostic history because
its nested scope/catalog hash was stale.
