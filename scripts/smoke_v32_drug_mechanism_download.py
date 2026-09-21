#!/usr/bin/env python3
"""Real-asset smoke for hash-verified V3.2 Drug mechanism download."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cc_hhgt.v32.drug_mechanism_query import DrugMechanismReleaseQuery
from cc_hhgt.v32.release_registry import artifact_sha256


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query = DrugMechanismReleaseQuery(
        args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
    )
    manifest = query.download_manifest()
    assert manifest["artifact_count"] == 1
    assert manifest["mechanism_semantics"] == "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION"
    assert manifest["causal_mechanism_claimed"] is False
    assert manifest["target_contribution_claimed"] is False
    resolved = query.resolve_download()
    assert artifact_sha256(resolved["path"]) == resolved["sha256"]
    assert resolved["rows"] == 123_537_897

    payload: dict[str, object] = {
        "format": "CC_HHGT_V3_2_DRUG_MECHANISM_DOWNLOAD_REAL_SMOKE_V1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": query.manifest_sha256,
        "artifact_path": resolved["path"],
        "artifact_sha256": resolved["sha256"],
        "artifact_bytes": resolved["bytes"],
        "artifact_rows": resolved["rows"],
        "request_time_payload_rehash_exercised": True,
        "query_code_sha256": artifact_sha256(
            Path(__file__).resolve().parents[1]
            / "cc_hhgt"
            / "v32"
            / "drug_mechanism_query.py"
        ),
        "smoke_code_sha256": artifact_sha256(Path(__file__).resolve()),
        "mechanism_semantics": "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION",
        "causal_mechanism_claimed": False,
        "target_contribution_claimed": False,
        "does_not_change_primary_pathway_ranking": True,
        "production_deployed": False,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite smoke output: {output}")
    atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
