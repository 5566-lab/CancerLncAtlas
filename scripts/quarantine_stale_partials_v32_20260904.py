#!/usr/bin/env python3
"""Move stale upload-partial files out of the old web candidate tree.

The source roots are deliberately hard-coded to two superseded candidate
trees.  This is a *recoverable relocation*, not a recursive delete: every
source file is copied to a separate /dsk2 quarantine and removed only after a
successful rsync transfer and size check.  Canonical siblings are never
modified.  Files that are recent, open, symlinked, missing a canonical peer,
or otherwise fail a check are retained and recorded.

The ${PRIVATE_WORK_ROOT} mount is a slow SSHFS/NFS path, so this operation does not perform
a second full cryptographic read before moving.  The 64-hex suffix and
same-size canonical peer are recorded as provenance; the quarantine remains
available for a later full hash audit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

ROOTS = (
    Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32"),
    Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3_p0_delta_r1/v32"),
)
QUARANTINE_ROOT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_quarantine_v32_partials_20260904")
RECEIPT_DIR = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r3/receipts")
PARTIAL_RE = re.compile(r"^(?P<base>.+)\.partial\.(?P<digest>[0-9a-fA-F]{64})$")


def under_root(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in ROOTS)


def is_open(path: Path) -> bool:
    try:
        return subprocess.run(
            ["fuser", "-s", "--", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return True


def discover(min_age_hours: float) -> list[dict[str, object]]:
    now = time.time()
    rows: list[dict[str, object]] = []
    for root in ROOTS:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.rglob("*.partial.*"):
            if not path.is_file() or path.is_symlink() or not under_root(path):
                continue
            match = PARTIAL_RE.match(path.name)
            if match is None:
                continue
            canonical = path.with_name(match.group("base"))
            row: dict[str, object] = {
                "source": str(path),
                "canonical": str(canonical),
                "suffix_sha256": match.group("digest").lower(),
                "status": "RETAINED",
            }
            try:
                source_stat = path.stat()
                row["bytes"] = source_stat.st_size
                row["mtime"] = source_stat.st_mtime
                age_hours = (now - source_stat.st_mtime) / 3600.0
                row["age_hours"] = round(age_hours, 3)
                if age_hours < min_age_hours:
                    row["reason"] = "TOO_RECENT"
                    rows.append(row)
                    continue
                if not canonical.is_file() or canonical.is_symlink():
                    row["reason"] = "CANONICAL_SIBLING_MISSING_OR_UNSAFE"
                    rows.append(row)
                    continue
                row["canonical_bytes"] = canonical.stat().st_size
                if source_stat.st_size != row["canonical_bytes"]:
                    row["reason"] = "SIZE_MISMATCH"
                    rows.append(row)
                    continue
                if is_open(path) or is_open(canonical):
                    row["reason"] = "FILE_OPEN"
                    rows.append(row)
                    continue
                row["status"] = "ELIGIBLE_RELOCATION"
                row["reason"] = "STALE_TYPED_PARTIAL_CANONICAL_PRESENT"
            except (OSError, ValueError) as exc:
                row["reason"] = f"CHECK_ERROR:{type(exc).__name__}"
            rows.append(row)
    return sorted(rows, key=lambda r: (-int(r.get("bytes", 0)), str(r["source"])))


def write_receipt(rows: list[dict[str, object]], action: str, rsync_timeout: int) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    eligible = [r for r in rows if r.get("status") == "ELIGIBLE_RELOCATION"]
    moved: list[str] = []
    moved_bytes = 0
    failures: list[str] = []
    if action == "relocate":
        if QUARANTINE_ROOT.exists():
            raise RuntimeError(f"Refusing to reuse quarantine root: {QUARANTINE_ROOT}")
        QUARANTINE_ROOT.mkdir(parents=True, exist_ok=False)
        for row in eligible:
            source = Path(str(row["source"]))
            canonical = Path(str(row["canonical"]))
            source_root = next(root for root in ROOTS if root in source.parents)
            relative = source.relative_to(source_root)
            # Keep the two source trees disjoint in quarantine even if they
            # contain the same relative filename.
            destination = QUARANTINE_ROOT / source_root.parent.name / source_root.name / relative
            row["destination"] = str(destination)
            try:
                if destination.exists() or not source.is_file() or source.is_symlink():
                    row["status"] = "RETAINED"
                    row["reason"] = "REVALIDATION_FAILED_BEFORE_COPY"
                    failures.append(str(source))
                    continue
                if is_open(source) or is_open(canonical):
                    row["status"] = "RETAINED"
                    row["reason"] = "FILE_OPEN_AT_COPY"
                    failures.append(str(source))
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                # rsync copies to a same-named destination and removes the
                # source only after a successful transfer.  --timeout bounds
                # slow SSHFS reads; a failure leaves source data intact.
                result = subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--protect-args",
                        f"--timeout={rsync_timeout}",
                        "--remove-source-files",
                        str(source),
                        str(destination),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=rsync_timeout + 30,
                    check=False,
                )
                row["rsync_returncode"] = result.returncode
                if result.returncode != 0:
                    row["status"] = "RETAINED"
                    row["reason"] = "RSYNC_FAILED"
                    row["rsync_stderr_tail"] = result.stderr[-500:]
                    failures.append(str(source))
                    continue
                if source.exists() or not destination.is_file():
                    row["status"] = "RETAINED"
                    row["reason"] = "POST_COPY_SOURCE_OR_DESTINATION_CHECK_FAILED"
                    failures.append(str(source))
                    continue
                if destination.stat().st_size != int(row["bytes"]):
                    row["status"] = "RETAINED"
                    row["reason"] = "DESTINATION_SIZE_MISMATCH"
                    failures.append(str(source))
                    continue
                row["status"] = "RELOCATED"
                row["reason"] = "RECOVERABLE_QUARANTINE_COPY_COMPLETE"
                moved.append(str(source))
                moved_bytes += int(row["bytes"])
            except (OSError, subprocess.SubprocessError) as exc:
                row["status"] = "RETAINED"
                row["reason"] = f"RELOCATION_ERROR:{type(exc).__name__}"
                failures.append(str(source))
    tsv = RECEIPT_DIR / f"STALE_PARTIAL_QUARANTINE_{stamp}.tsv"
    fields = sorted({key for row in rows for key in row})
    with tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(fields) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(key, "")) for key in fields) + "\n")
    summary = {
        "format": "CANCERLNCATLAS_V32_STALE_PARTIAL_RECOVERABLE_QUARANTINE_V1",
        "host": socket.gethostname(),
        "action": action,
        "source_roots": [str(root) for root in ROOTS],
        "quarantine_root": str(QUARANTINE_ROOT),
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "eligible_bytes": sum(int(r.get("bytes", 0)) for r in eligible),
        "relocated_count": len(moved),
        "relocated_bytes": moved_bytes,
        "failed_or_retained_count": len(failures),
        "rsync_timeout_seconds": rsync_timeout,
        "direct_full_hash_audit": "NOT_PERFORMED_BEFORE_RELOCATION_NFS_READ_GUARD",
        "content_address_suffix_recorded": True,
        "canonical_sibling_and_size_checked": True,
        "age_and_open_handle_checked": True,
        "recoverable": True,
        "training_process_started": False,
        "paid_gpu_started": False,
        "tsv": str(tsv),
        "relocated_paths": moved,
    }
    out = RECEIPT_DIR / f"STALE_PARTIAL_QUARANTINE_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"manifest={tsv}", flush=True)
    print(f"receipt={out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relocate", action="store_true")
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--rsync-timeout", type=int, default=120)
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise SystemExit(f"hard stop: expected host 149, got {socket.gethostname()}")
    if args.relocate and QUARANTINE_ROOT.exists():
        raise SystemExit(f"hard stop: quarantine root already exists: {QUARANTINE_ROOT}")
    rows = discover(args.min_age_hours)
    write_receipt(rows, "relocate" if args.relocate else "dry_run", args.rsync_timeout)
    if not args.relocate and any(r.get("reason") == "FILE_OPEN" for r in rows):
        raise SystemExit("hard stop: an eligible candidate is open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
