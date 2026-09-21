#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cc_hhgt.v32.gdc_segment_cnv import download_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Recoverable MD5-verified GDC segment downloader")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cancers", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--lock-ttl-seconds", type=int, default=21600)
    args = parser.parse_args()
    results = download_manifest(
        args.manifest,
        args.output_root,
        cancers=args.cancers,
        workers=args.workers,
        retries=args.retries,
        max_files=args.max_files,
        lock_ttl_seconds=args.lock_ttl_seconds,
    )
    payload = {
        "format": "CC_HHGT_V3_2_SEGMENT_DOWNLOAD_STATUS_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "formal_v32_unchanged": True,
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    status = args.output_root / "DOWNLOAD_STATUS.json"
    status.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status_path": str(status), "files": len(results)}, indent=2))
    successful = {"ALREADY_VERIFIED", "DOWNLOADED_VERIFIED"}
    return 1 if any(row["status"] not in successful for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
