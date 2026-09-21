#!/usr/bin/env python
"""Bind the audited V3.2 module lineages and web assets into a staging registry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.release_registry import (  # noqa: E402
    REGISTRY_SCHEMA_VERSION,
    artifact_sha256,
    validate_release_registry,
)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _registry_relative(path: Path, output_parent: Path) -> str:
    return Path(os.path.relpath(path, output_parent)).as_posix()


def build_registry(*, repo_root: Path, spec_path: Path, output_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing staging registry: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    modules: dict[str, dict[str, Any]] = {}
    for module_id, module_spec in spec["modules"].items():
        lineage_path = _resolve(repo_root, module_spec["lineage_path"])
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {
            "status": module_spec["status"],
            "analysis_version": lineage["analysis_version"],
            "lineage_path": _registry_relative(lineage_path, output.parent),
            "lineage_sha256": artifact_sha256(lineage_path),
        }
        if module_spec["status"] == "SUCCESS_NEWLY_TRAINED":
            entry.update(
                {
                    "training_run_id": lineage["training_run_id"],
                    "new_training_attestation": True,
                    "old_checkpoint_loaded": lineage["old_checkpoint_loaded"],
                    "old_predictions_used_as_features": lineage[
                        "old_predictions_used_as_features"
                    ],
                    "old_rankings_used_as_outputs": lineage["old_rankings_used_as_outputs"],
                }
            )
        else:
            entry["reason_code"] = module_spec["reason_code"]
        modules[module_id] = entry

    website_artifacts: list[dict[str, Any]] = []
    for artifact_spec in spec["website_artifacts"]:
        path = _resolve(repo_root, artifact_spec["path"])
        website_artifacts.append(
            {
                "role": artifact_spec["role"],
                "path": _registry_relative(path, output.parent),
                "sha256": artifact_sha256(path),
                "artifact_kind": artifact_spec["artifact_kind"],
                "generation": spec["analysis_version"],
                "source_module": artifact_spec["source_module"],
            }
        )
    manifest = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "release_id": spec["release_id"],
        "analysis_version": spec["analysis_version"],
        "environment": "staging",
        "production_deployed": False,
        "release_ready": False,
        "all_non_null_predictions_newly_trained_v32": True,
        "all_published_results_generated_in_v32": True,
        "modules": modules,
        "website_artifacts": website_artifacts,
        "capabilities": spec["capabilities"],
    }
    if spec.get("supersedes"):
        manifest["supersedes"] = list(spec["supersedes"])
        manifest["supersession_reason"] = str(spec["supersession_reason"])
    validate_release_registry(manifest, base_dir=output.parent, require_staging=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "status": "PASS",
        "registry_path": str(output),
        "registry_sha256": artifact_sha256(output),
        "release_ready": False,
        "production_deployed": False,
        "module_statuses": {module: value["status"] for module, value in modules.items()},
        "website_artifact_count": len(website_artifacts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_registry(
        repo_root=args.repo_root.resolve(),
        spec_path=args.spec.resolve(),
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
