# V3.2 G0/G1/G2 r2 external archive-lock contract

Status: `IMPLEMENTED_AND_LOCALLY_TESTED_DRAFT_NOT_DEPLOYED_NOT_AUTHORIZED`

The formal r2 archive must not contain its own archive or manifest SHA-256.
The archive contains the config and patcher, so embedding either digest in
those files creates a recursive hash dependency with no constructible fixed
artifact. The former placeholder-injection design is therefore retired.

The replacement root is
`runtime/bootstrap/v32_g012_paid_gpu_20260901_r2/CODE_ARCHIVE.LOCK.json`, which
is outside the code archive. It binds the fixed archive, external manifest and
bootstrap verifier paths and SHA-256 values. It explicitly states that it is
not in the archive and does not authorize formal training.

The CPU-only materializer is
`scripts/materialize_v32_g012_external_archive_lock_no_gpu_r1.py`. It hashes
the archive, manifest and verifier through no-follow, stable file handles,
checks the verifier is bound by the manifest, rejects a lock member inside the
archive, and commits the external lock exclusively with crash recovery.

`scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r2.sh` accepts the
lock only when an archive-external bootstrap controller supplies its exact
SHA-256 in `V32_R2_EXTERNAL_LOCK_SHA256`. This out-of-band value is the trust
anchor; the archive must never be allowed to choose it. The patcher then reads
the lock through a held no-follow descriptor chain, verifies/extracts the
archive with the separately installed verifier, and writes both the lock path
and lock SHA into `PATCH_READY_V3`.

The static r2 validator independently reloads the fixed external lock,
recomputes its SHA, binds all lock hashes to `PATCH_READY_V3`, and requires the
entire installed code-root file set to match the manifest. Extra root startup
hooks, unknown prefixes, links and special files are rejected.

This change does not remove any paid-training gate. The checked-in estimator,
Formal V2 evidence bridge, zero budget and paid launcher body remain blocked.
No formal archive, external lock, cloud API call or GPU action was produced by
this implementation pass.
