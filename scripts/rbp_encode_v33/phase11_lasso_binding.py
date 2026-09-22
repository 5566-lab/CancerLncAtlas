"""Bind the existing (reused) LASSO baseline into the fairness axis.

The user's instruction is explicit: existing LASSO results are reusable and must
NOT be re-materialised.  So rather than recomputing a baseline, we fingerprint the
frozen artifact set and pin that digest as ``lasso_base_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
BASE = Path(
    "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
    "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/posttraining_lasso_full"
)
OUT = WORK / "manifests"
OUT.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = sorted(p for p in BASE.iterdir() if p.is_file())
rows = []
for path in files:
    rows.append((path.name, path.stat().st_size, file_sha256(path)))

composite = hashlib.sha256()
for name, size, digest in rows:
    composite.update(f"{digest}  {name}\n".encode("utf-8"))
composite_digest = composite.hexdigest()

manifest = OUT / "LASSO_BASE_MANIFEST.tsv"
with manifest.open("w", encoding="utf-8") as handle:
    handle.write("sha256\tsize\tfile\n")
    for name, size, digest in rows:
        handle.write(f"{digest}\t{size}\t{name}\n")

print("=== reused LASSO baseline (NOT re-materialised) ===")
print(f"    source : {BASE}")
for name, size, digest in rows:
    print(f"    {digest[:16]}  {size:>10,}  {name}")
print()
print(f"    composite lasso_base_sha256 = {composite_digest}")
print(f"    written: {manifest}")

(OUT / "LASSO_BASE_BINDING.json").write_text(
    json.dumps(
        {
            "source": str(BASE),
            "reused_not_rematerialised": True,
            "files": [
                {"file": name, "bytes": size, "sha256": digest}
                for name, size, digest in rows
            ],
            "composite_sha256": composite_digest,
            "composite_rule": "sha256 of sorted '<sha256>  <name>\\n' lines",
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"    written: {OUT / 'LASSO_BASE_BINDING.json'}")
print()
print("PASTE THIS INTO rbp_evidence_config.py:")
print(f'    lasso_base_sha256: str = "{composite_digest}"')
