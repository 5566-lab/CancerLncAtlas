# V3.2 website r8 self-contained candidate design (2026-09-05)

This is a read-only design/audit note.  No r8 directory was created, no large
payload was copied, and no service was started.

## Finding

An r8 candidate is feasible, but copying only the approximately 144 MB r6
`code` tree and the approximately 0.35 MB r7 metadata tree is **not** a
self-contained runtime.  The r6 registry is internally relative and can be
reused, whereas the seven r7 mounted sidecars contain absolute references to
payloads outside both candidates.  The r7 API attempt failed earlier because
its copied registry had no r6 `v32_full_multitask` artifact tree.

The honest choices are:

1. **Strict self-contained r8 (recommended for a portable release
   candidate):** copy or same-filesystem hard-link the runtime payloads below,
   rewrite their binding path declarations to r8-relative paths, and rehash
   the resulting binding chain.  This does not require any pending Drug,
   legacy single-cell, or other unmounted module payloads.
2. **Mounted r8 (smallest/no-data-copy option):** copy r6 code and r7
   sidecars, retain server-native absolute payload paths, and record an
   explicit mount manifest.  This can be a host-149 staging instance, but it
   must not be called self-contained or portable; every mount must be checked
   at startup.

## Audited source sizes

Read-only checks on host `149` (`COMPUTE_HOST`) found:

| source | observed bytes | notes |
|---|---:|---|
| r6 `code` tree | 144,387,558 | 425 files; includes about 3.4 MB `__pycache__`, which can be omitted |
| r7 metadata tree | 350,924 | seven mounted sidecars plus registry/catalog/scope/receipts |
| formal-23 single-cell 23 output roots | 170,024,569 | 23 roots, seven query files per root plus manifests/markers |
| directional CNV predictions | 60,718,177 | plus coverage, lineage and transformation/audit JSON |
| clinical KM curves | 31,364,076 | directory payload |
| clinical KM statistics | 17,440,850 | parquet |
| clinical standardized expression | 155,090,389 | directory payload |
| clinical formal candidate universe | 8,429,901 | parquet |
| clinical TCGA-CDR workbook | 2,945,129 | xlsx |
| state gene-set outputs/metadata | under 1 MB | state prediction (11,156,466 B) is already in r6 |
| Evidence direction probability table | 19,410,241 | five-fold aggregate parquet |

Thus a strict closure needs roughly 0.47 GB of additional payload blocks if
all listed files are physically copied (less if immutable `${PRIVATE_WORK_ROOT}` files can
be hard-linked).  It does **not** require the multi-GB Drug mechanism table
or any of the 32 pending bindings.

## Proposed r8 layout

Use an isolated root on `${PRIVATE_WORK_ROOT}` (which has space), for example:

```text
./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r8/
  code/                         # r6 source code + r6 core artifacts
    cc_hhgt/                    # omit __pycache__
    website/
    scripts/
    config/
    artifacts/
      v32_full_multitask/       # unchanged r6 relative registry closure
      v32_staging/              # unchanged r6 mixed-query closure
      v32_server_bindings/      # r8-localized seven mounted bindings
      r8_payloads/              # copied/hard-linked runtime data
  catalog_candidate_20260905_r8/
  capability_scope_20260905_r8.json
  receipts/
```

The r6 `RELEASE_REGISTRY.json` can stay byte-identical if its relative
`v32_full_multitask` and `mixed_query_assets_exact2135_r12` subtrees stay in
the same positions under `code/artifacts`.  Its observed SHA is
`b20fc54ea38438693608f4fe33a19d629885e4d32573ded328349eb67656fef1`.

The unified r8 manifest must point only to files inside `r8/code` (the loader
rejects paths escaping `repo_root`).  Its seven mounted entries are:

```text
single_cell_formal23
directional_cnv
directional_cnv_independent_audit
clinical_kaplan_meier
state_gene_sets
evidence_direction_probabilities
evidence_direction_probabilities_independent_audit
```

The remaining 32 entries stay `PENDING_FORMAL_SUCCESS` with their explicit
reasons.  They must not be silently inferred from old candidates.

## Runtime closure per mounted sidecar

### Formal-23 single cell

Copy/hard-link the 23 audited output roots (the seven required files are
`RESOURCE_ESTIMATE.json`, `association_context_availability.parquet`,
`association_evidence.parquet`, `association_lncrna_testability.parquet`,
`lncrna_donor_celltype_summary.parquet`, `pathway_availability.parquet`, and
`pathway_donor_celltype_summary.parquet`) and their `FILE_MANIFEST.parquet`
files.  Copy the `COHORT_BINDING.json` and independent `AUDIT.json` authority
files.  The query loader dereferences the audit's `output_root` and manifest
paths, so those declarations must be rewritten to r8-local paths.  The
per-file SHA values remain unchanged when the files are copied byte-for-byte.

`SINGLE_CELL_FORMAL23_SUCCESS.json` then needs its binding/audit paths and
inner SHA values updated; the localized `COHORT_BINDING.json` and `AUDIT.json`
form a new hash chain.  Keep the accepted `23 + 10 typed-unavailable` scope;
do not regenerate or claim 33-cancer single-cell coverage.

### Directional CNV

Copy the 60,718,177-byte prediction parquet, coverage parquet, lineage and
transformation-audit JSON, and the independent-audit report.  Rewrite the
release and independent-audit binding paths to local files.  The artifact
hashes and audit content hashes remain the same; the two binding JSON hashes
must change because their path fields change.  The query code itself can be
the r6 byte-pinned `directional_cnv_query.py` (SHA
`355b9c552b1827e3752f80f3a56c0ef6ea905159201c5f8b93ba046cbab51a38`).

### Clinical KM

This is the largest closure.  Copy/hard-link the statistics parquet, curves
directory, standardized expression directory, formal candidate universe and
TCGA-CDR workbook, plus `SOURCE_INPUTS.json`, module lineage and the sibling
`SUCCESS.json`.  Rewrite all runtime artifact and authority paths in the
clinical binding.  The copied data hashes can remain unchanged; the clinical
binding hash and any receipt that names it must be recomputed.  The query
loader recursively validates the curves/expression directories, so a
metadata-only copy is insufficient for this route.

### State gene sets

Copy the small catalog/GMT/members/report/report-manifest files and source
metadata (`SOURCE_CHECKPOINT_MANIFEST.json`, source state lineage,
materialization module/runner).  Reuse the r6 state prediction parquet and
state module lineage where their bytes match the declared hashes.  Rewrite
binding source/output paths and recompute the binding hash; do not alter the
underlying state predictions.

### Evidence direction probabilities

Copy the 19,410,241-byte aggregate probability parquet and the independent
audit report.  Rewrite release/audit binding and report paths; recompute the
release binding, audit binding, and unified manifest hashes.  Nested source
checkpoint/path strings in the historical provenance are not dereferenced by
the query loader; retain them as provenance-only fields and record that fact
in the r8 closure receipt.  If a future loader begins dereferencing them, the
candidate must fail closed until those sources are separately localized.

## Hashes that must and must not change

**Must change:**

- r8 unified staging manifest (all binding paths and revision change);
- r8 catalog and frontend catalog if their absolute r6 paths are rewritten;
- each localized binding JSON whose path fields change;
- each independent-audit binding that names a localized release binding;
- formal-23 success/audit chain when output roots are localized;
- new r8 capability scope and all r8 receipts.

The current r7 catalog has four stale absolute r6 references that must be
removed by the catalog rebuild: the formal23 success path, the two directional
CNV binding paths, and the capability-scope path.  A raw string replacement is
not sufficient because the rebuilt catalog's own SHA is part of the frontend
pinning receipt.

**Should remain unchanged (byte-for-byte copies):**

- r6 core registry and its registered relative artifacts;
- parquet/GMT/XLSX payloads;
- immutable lineage/audit reports, unless a path-rebinding contract explicitly
  requires a new wrapper;
- r6 source code and frontend assets.

Never edit an old r6/r7 receipt merely to make its path look current.  Create
new r8 receipts and retain old receipts as immutable history.

The r8 `supersedes_manifest` should point to a copied r7 manifest under the
r8 history directory (or to a verified server-native immutable history path),
not to r7's current malformed cross-mount relative path.  This is a lineage
repair in the new manifest; it does not change the old r7 receipt.

## Minimal next step

Do not copy or start anything in this bounded audit.  The next implementation
step is a server-149-only, no-GPU `materialize_r8_runtime_closure` operation
that:

1. verifies the r6 registry and source sidecar hashes without changing them;
2. creates the isolated r8 directory on `${PRIVATE_WORK_ROOT}` using hard-links only where
   source and destination are on the same filesystem, otherwise byte copies;
3. localizes the five runtime data closures above and emits a path-rebinding
   manifest before writing any binding;
4. rewrites/re-hashes the binding chain and catalog atomically (`.partial`
   then rename);
5. runs a bounded validator and only then a loopback route smoke on a new
   non-production port; and
6. writes `release_ready=false`, `production_deployed=false`, and explicit
   pending-binding counts.

Until that operation and route smoke pass, r6 remains the only route-smoked
staging evidence and r7 remains metadata-only.  No paid GPU is involved in
this website closure.

## Implementation amendment (2026-09-05)

The design-time statement above that no r8 directory had been created is now
historical.  The operation completed on host 149 at:

```text
./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r8
```

The resulting candidate has 39 manifest entries (7
`MOUNTED_HASH_PINNED`, 32 `PENDING_FORMAL_SUCCESS`), passed 18/18 strict
semantic route probes and 7/7 overlay/download probes, and is recorded as
`PASS_STAGING_WITH_EXPLICIT_GAPS`.  Its receipt SHA-256 is
`9c9b9aa93cb6b26ad105a237fa00d5742785ecb6e90acd379ab831e29eeed42a`.

An independent host-149 check recomputed all seven mounted binding hashes and
confirmed that their active destinations are regular files inside the r8 code
root.  A small number of absolute `D:/model` or historical source strings
remain in provenance fields and are explicitly non-dereferenced; the runtime
manifest/catalog/API binding has no external path values.  The r8 service was
stopped after the smoke, production port 8261 was untouched, and no GPU was
started.  `release_ready=false` and `production_deployed=false` remain the
authoritative release state.
