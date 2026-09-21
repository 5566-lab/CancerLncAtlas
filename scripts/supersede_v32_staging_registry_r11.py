#!/usr/bin/env python3
"""Supersede the stale staging registry with the R11 single-cell contract.

Only registry paths and the audited-unavailable single-cell lineage change.
All mounted result artifacts remain byte-identical and the registry stays
staging-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.release_registry import (  # noqa: E402
    artifact_sha256,
    validate_release_registry,
)


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def build(
    *,
    base_registry: Path,
    single_cell_lineage: Path,
    mixed_asset_manifest: Path,
    mixed_membership: Path,
    output: Path,
) -> dict[str, Any]:
    base_registry = base_registry.resolve()
    single_cell_lineage = single_cell_lineage.resolve()
    mixed_asset_manifest = mixed_asset_manifest.resolve()
    mixed_membership = mixed_membership.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite registry: {output}")
    base_root = base_registry.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(json.loads(base_registry.read_text(encoding="utf-8")))
    manifest["release_id"] = "CancerLncAtlas_V3.2_AUXILIARY_STAGING_R11_20260903"
    supersedes = list(manifest.get("supersedes") or [])
    if str(base_registry) not in supersedes:
        supersedes.append(str(base_registry))
    manifest["supersedes"] = supersedes
    manifest["supersession_reason"] = (
        "Rebind staging metadata to the V3.2 R11 single-cell typed-unavailable "
        "contract; mounted independent-head payloads and primary scores are unchanged."
    )

    for module_id, entry in manifest["modules"].items():
        if module_id == "single_cell":
            lineage_path = single_cell_lineage
            entry["reason_code"] = "R11_NO_FORMAL_SINGLE_CELL_DATASETS_AFTER_DONOR_AUDIT"
        else:
            lineage_path = (base_root / entry["lineage_path"]).resolve()
        entry["lineage_path"] = _relative(lineage_path, output.parent)
        entry["lineage_sha256"] = artifact_sha256(lineage_path)

    for artifact in manifest["website_artifacts"]:
        role = artifact["role"]
        if role == "mixed_query_asset_manifest":
            artifact_path = mixed_asset_manifest
        elif role == "mixed_query_exact_pathway_membership":
            artifact_path = mixed_membership
        elif role in {
            "mixed_query_identifier_map",
            "mixed_query_v32_lnc_exact_association",
            "mixed_query_pathway_metadata",
        }:
            declarations = json.loads(mixed_asset_manifest.read_text(encoding="utf-8")).get("artifacts", {})
            declaration = declarations.get(role)
            if not isinstance(declaration, dict) or not str(declaration.get("filename", "")).strip():
                raise RuntimeError(f"Mixed manifest does not declare filename for {role}")
            artifact_path = mixed_asset_manifest.parent / str(declaration["filename"])
        else:
            artifact_path = (base_root / artifact["path"]).resolve()
        artifact["path"] = _relative(artifact_path, output.parent)
        artifact["sha256"] = artifact_sha256(artifact_path)

    # Validate against the future registry directory before writing it.  This
    # catches broken relative paths without changing any existing authority.
    validate_release_registry(manifest, base_dir=output.parent, require_staging=True)
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    return {
        "status": "PASS",
        "registry_path": str(output),
        "registry_sha256": artifact_sha256(output),
        "release_id": manifest["release_id"],
        "release_ready": False,
        "production_deployed": False,
        "single_cell_lineage_sha256": manifest["modules"]["single_cell"]["lineage_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--single-cell-lineage", type=Path, required=True)
    parser.add_argument("--mixed-asset-manifest", type=Path, required=True)
    parser.add_argument("--mixed-membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(base_registry=args.base_registry, single_cell_lineage=args.single_cell_lineage, mixed_asset_manifest=args.mixed_asset_manifest, mixed_membership=args.mixed_membership, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
