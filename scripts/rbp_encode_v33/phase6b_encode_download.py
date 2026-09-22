"""Phase 6B: download the ENCODE eCLIP narrowPeak files and verify every byte.

Only the files Phase 6A marked as the consumption surface are fetched: released
eCLIP narrowPeak calls.  Each download is verified against the md5sum and the
byte count ENCODE published, so a truncated or substituted payload cannot reach
the liftOver step.  A file that fails verification is deleted and marked FAILED
rather than left on disk to be picked up later.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
MAN = WORK / "manifests"
FILES = WORK / "inputs" / "encode" / "files"
FILES.mkdir(parents=True, exist_ok=True)

#: The SOCKS relay address is environment-overridable: the historical
#: 127.0.0.1:1080 relay on 149 died mid-run, and the replacement binds 1081.
PROXY = os.environ.get("ENCODE_PROXY", "socks5h://127.0.0.1:1080")
BASE = "https://www.encodeproject.org"
#: The SOCKS proxy on 149 saturates: eight concurrent transfers stalled it
#: completely, after which even a single request timed out.  Keep the pool small
#: and abort any transfer that stops making progress, so one bad file cannot hold
#: a worker for the full --max-time budget.
WORKERS = int(os.environ.get("ENCODE_DOWNLOAD_WORKERS", "3"))
MAX_TIME = 240
SPEED_LIMIT = 8000      # bytes/s
SPEED_TIME = 60         # abort if below SPEED_LIMIT for this many seconds
MANIFEST = MAN / "ENCODE_FILE_MANIFEST.tsv"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str).fillna("")
print(f"=== manifest rows: {len(manifest):,} ===", flush=True)

surface = manifest.loc[
    manifest.in_eclip_narrowpeak_search.eq("True")
    & manifest.file_format.eq("bed")
    & manifest.status.eq("released")
].copy()
print(f"    eCLIP narrowPeak surface: {len(surface):,} files", flush=True)
print(surface.groupby(["assembly", "biosample"]).size().to_string(), flush=True)

if LIMIT:
    surface = surface.head(LIMIT)
    print(f"    LIMIT applied: {len(surface):,} files", flush=True)

surface = surface.sort_values("file_accession").reset_index(drop=True)
total_bytes = int(pd.to_numeric(surface.file_size, errors="coerce").fillna(0).sum())
print(f"    declared size: {total_bytes / 1048576:.1f} MiB\n", flush=True)


def fetch_one(row: pd.Series) -> dict:
    accession = row.file_accession
    target = FILES / f"{accession}.bed.gz"
    expected_md5 = row.md5sum
    expected_size = int(float(row.file_size or 0))
    url = BASE + row.href

    if target.exists():
        size = target.stat().st_size
        if size == expected_size and md5_file(target) == expected_md5:
            return {"file_accession": accession, "status": "CACHED_VERIFIED",
                    "bytes": size, "md5": expected_md5, "assembly": row.assembly}
        target.unlink()

    result = subprocess.run(
        ["curl", "-sSL", "--max-time", str(MAX_TIME),
         "--speed-limit", str(SPEED_LIMIT), "--speed-time", str(SPEED_TIME),
         "--retry", "3", "--retry-delay", "2", "--retry-connrefused",
         "--proxy", PROXY, "-o", str(target), "-w", "%{http_code}", url],
        capture_output=True, text=True,
    )
    code = result.stdout.strip()
    if result.returncode != 0 or code != "200":
        if target.exists():
            target.unlink()
        return {"file_accession": accession, "status": "FAILED_HTTP",
                "http": code, "error": result.stderr[:200], "assembly": row.assembly}

    size = target.stat().st_size
    if size != expected_size:
        target.unlink()
        return {"file_accession": accession, "status": "FAILED_SIZE",
                "bytes": size, "expected_bytes": expected_size, "assembly": row.assembly}
    observed_md5 = md5_file(target)
    if observed_md5 != expected_md5:
        target.unlink()
        return {"file_accession": accession, "status": "FAILED_MD5",
                "md5": observed_md5, "expected_md5": expected_md5, "assembly": row.assembly}
    return {"file_accession": accession, "status": "DOWNLOADED",
            "bytes": size, "md5": observed_md5, "assembly": row.assembly}


print("=== downloading ===", flush=True)
results: list[dict] = []
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(fetch_one, row): row.file_accession
               for _, row in surface.iterrows()}
    for done, future in enumerate(as_completed(futures), 1):
        try:
            results.append(future.result())
        except Exception as exc:  # a worker crash must not lose the other files
            results.append({"file_accession": futures[future], "status": "FAILED_EXCEPTION",
                            "error": repr(exc)[:200]})
        if done % 100 == 0 or done == len(futures):
            ok = sum(1 for r in results if r["status"] in ("DOWNLOADED", "CACHED_VERIFIED"))
            print(f"    {done:>5}/{len(futures)}  verified={ok}", flush=True)

results.sort(key=lambda r: r["file_accession"])
counts: dict[str, int] = {}
for r in results:
    counts[r["status"]] = counts.get(r["status"], 0) + 1
print(f"\n=== status: {counts} ===")
failures = [r for r in results if r["status"].startswith("FAILED")]
for r in failures[:10]:
    print(f"    {r['file_accession']}  {r['status']}  {r.get('error', '')}")

status_by_accession = {r["file_accession"]: r["status"] for r in results}
manifest["download_status"] = [
    status_by_accession.get(acc, row_status)
    for acc, row_status in zip(manifest.file_accession, manifest.download_status)
]
manifest.to_csv(MANIFEST, sep="\t", index=False)
print(f"\nupdated manifest: {MANIFEST}")

verified_bytes = sum(r.get("bytes", 0) for r in results
                     if r["status"] in ("DOWNLOADED", "CACHED_VERIFIED"))
receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "manifest": str(MANIFEST),
    "manifest_sha256_after": sha256_file(MANIFEST),
    "surface_files": len(surface),
    "surface_bytes_declared": total_bytes,
    "verified_files": sum(1 for r in results if r["status"] in ("DOWNLOADED", "CACHED_VERIFIED")),
    "verified_bytes": verified_bytes,
    "status_counts": counts,
    "failures": failures,
    "files": results,
    "verification": "md5sum and byte count must both equal the values published by ENCODE",
    "directory": str(FILES),
    "note": (
        "Files failing verification are deleted before the receipt is written, so "
        "the directory contains only payloads whose md5 and size match ENCODE."
    ),
}
receipt_path = MAN / "ENCODE_DOWNLOAD_RECEIPT.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"written: {receipt_path}")
print(f"verified: {receipt['verified_files']:,} files / {verified_bytes / 1048576:.1f} MiB")
if failures:
    raise SystemExit(f"FAIL-CLOSED: {len(failures)} files failed download verification")
