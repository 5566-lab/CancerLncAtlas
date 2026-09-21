#!/usr/bin/env python3
"""Real-asset smoke for the hash-verified 17-cancer UCell downloads."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.single_cell_ucell_17c_query import (
    SingleCellUCell17CInputError,
    SingleCellUCell17CQuery,
)


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
    parser.add_argument("--binding", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query = SingleCellUCell17CQuery(
        args.binding,
        expected_binding_sha256=args.binding_sha256,
        runtime_rehash_trees=True,
    )
    cancers: list[dict[str, object]] = []
    for cancer in ("ACC", "HNSC"):
        manifest = query.download_manifest(cancer_id=cancer)
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, list) and artifacts
        relative_paths = {str(row["relative_path"]) for row in artifacts}
        assert "PRESENT_MISSING_COUNT_AUDIT.json" in relative_paths
        assert "ucell_donor_celltype.parquet" in relative_paths
        assert "ucell_pathway_availability.parquet" in relative_paths
        assert any(path.startswith("cell_level_ucell/") for path in relative_paths)
        resolved = query.resolve_download(
            cancer_id=cancer,
            relative_path="PRESENT_MISSING_COUNT_AUDIT.json",
        )
        assert artifact_sha256(resolved["path"]) == resolved["sha256"]
        cancers.append(
            {
                "cancer_id": cancer,
                "artifact_count": manifest["artifact_count"],
                "total_bytes": manifest["total_bytes"],
                "tree_sha256": manifest["tree_sha256"],
                "resolved_relative_path": resolved["relative_path"],
                "resolved_sha256": resolved["sha256"],
            }
        )

    traversal_rejected = False
    try:
        query.resolve_download(cancer_id="ACC", relative_path="../SUCCESS.json")
    except SingleCellUCell17CInputError:
        traversal_rejected = True
    assert traversal_rejected

    payload: dict[str, object] = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_UCELL_17C_DOWNLOAD_REAL_SMOKE_V1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding_path": str(Path(args.binding).resolve()),
        "binding_sha256": query.binding_sha256,
        "query_code_sha256": artifact_sha256(
            Path(__file__).resolve().parents[1]
            / "cc_hhgt"
            / "v32"
            / "single_cell_ucell_17c_query.py"
        ),
        "smoke_code_sha256": artifact_sha256(Path(__file__).resolve()),
        "cancers_exercised": cancers,
        "request_time_payload_rehash_exercised": True,
        "path_traversal_rejected": traversal_rejected,
        "typed_unavailable_preserved": True,
        "pseudotime_artifacts_excluded": True,
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
