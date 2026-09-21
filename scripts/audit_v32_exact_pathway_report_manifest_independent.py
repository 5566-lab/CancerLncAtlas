#!/usr/bin/env python
"""Independent, source-rehashing audit of the exact-pathway report binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_BINDING_V1"
MANIFEST_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_MANIFEST_V1"
AUDIT_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_INDEPENDENT_AUDIT_V1"
AUDIT_BINDING_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_INDEPENDENT_AUDIT_BINDING_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    binding_path = (root / args.binding).resolve() if not args.binding.is_absolute() else args.binding.resolve()
    binding_path.relative_to(root)
    output = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    output.relative_to(root)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing non-empty output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": observed, "expected": expected, "pass": observed == expected})

    binding_sha = sha256(binding_path)
    check("release_binding_sha256", binding_sha, args.expected_binding_sha256.lower())
    binding = load(binding_path)
    check("binding_format", binding.get("format"), BINDING_FORMAT)
    check("binding_analysis_version", binding.get("analysis_version"), ANALYSIS_VERSION)
    check("binding_status", binding.get("status"), "PASS_HASH_BOUND")
    check("binding_prediction_rows", binding.get("prediction_rows"), 3_300_000)
    check("binding_fold_count", binding.get("fold_count"), 5)
    for key in ("old_checkpoint_loaded", "old_predictions_used_as_features", "old_rankings_used_as_outputs", "production_deployed", "release_ready"):
        check(f"binding_{key}", binding.get(key), False)
    check("binding_primary_authoritative", binding.get("primary_ranking_authoritative"), True)

    declaration = binding.get("v32_exact_pathway_report_manifest", {})
    manifest_path = (binding_path.parent / str(declaration.get("path", ""))).resolve()
    manifest_path.relative_to(root)
    check("manifest_exists", manifest_path.is_file(), True)
    check("manifest_sha256", sha256(manifest_path), declaration.get("sha256"))
    check("manifest_bytes", manifest_path.stat().st_size, declaration.get("bytes"))
    manifest = load(manifest_path)
    check("manifest_format", manifest.get("format"), MANIFEST_FORMAT)
    check("manifest_analysis_version", manifest.get("analysis_version"), ANALYSIS_VERSION)
    check("manifest_artifact_id", manifest.get("artifact_id"), "v32_exact_pathway_report_manifest")
    check("manifest_status", manifest.get("status"), "PASS_HASH_BOUND")
    check("manifest_prediction_rows", manifest.get("prediction_rows"), 3_300_000)
    check("manifest_fold_count", manifest.get("fold_count"), 5)
    check("manifest_trained_fresh", manifest.get("trained_from_scratch"), True)
    check("manifest_primary_authoritative", manifest.get("primary_ranking_authoritative"), True)
    for key in ("old_checkpoint_loaded", "old_predictions_used_as_features", "old_rankings_used_as_outputs", "historical_predictions_used", "production_deployed"):
        check(f"manifest_{key}", manifest.get(key), False)

    sources = manifest.get("sources", {})
    check("source_count", len(sources) if isinstance(sources, dict) else -1, 8)
    resolved: dict[str, Path] = {}
    if isinstance(sources, dict):
        for source_id, record in sorted(sources.items()):
            source = (manifest_path.parent / str(record.get("path", ""))).resolve()
            source.relative_to(root)
            resolved[source_id] = source
            check(f"{source_id}_exists", source.is_file(), True)
            check(f"{source_id}_sha256", sha256(source), record.get("sha256"))
            check(f"{source_id}_bytes", source.stat().st_size, record.get("bytes"))

    lineage = load(resolved["module_lineage"])
    check("lineage_training_status", lineage.get("training_status"), "SUCCESS")
    check("lineage_trained_fresh", lineage.get("trained_from_scratch"), True)
    check("lineage_five_fold", lineage.get("five_fold_ensemble"), True)
    check("lineage_folds", lineage.get("folds"), 5)
    check("lineage_prediction_rows", lineage.get("prediction_rows"), 3_300_000)
    check("lineage_prediction_sha", sha256(resolved["exact_pathway_predictions"]), lineage.get("prediction_sha256"))
    check("lineage_checkpoint_manifest_sha", sha256(resolved["checkpoint_manifest"]), lineage.get("checkpoint_manifest_sha256"))
    check("lineage_ensemble_manifest_sha", sha256(resolved["ensemble_source_manifest"]), lineage.get("ensemble_source_manifest_sha256"))
    check("parquet_metadata_rows", pq.ParquetFile(resolved["exact_pathway_predictions"]).metadata.num_rows, 3_300_000)
    check("ensemble_source_records", len(lineage.get("ensemble_source_records", [])), 5)
    for key in ("old_checkpoint_loaded", "old_predictions_used_as_features", "old_rankings_used_as_outputs"):
        check(f"lineage_{key}", lineage.get(key), False)

    failed = [row for row in checks if not row["pass"]]
    report = {
        "format": AUDIT_FORMAT,
        "status": "PASS" if not failed else "FAIL",
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "release_binding_sha256": binding_sha,
        "report_manifest_sha256": sha256(manifest_path),
        "checks": checks,
    }
    report_path = output / "INDEPENDENT_AUDIT_REPORT.json"
    atomic_json(report_path, report)
    audit_binding = {
        "format": AUDIT_BINDING_FORMAT,
        "status": report["status"],
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "release_binding_sha256": binding_sha,
        "report_manifest_sha256": sha256(manifest_path),
        "report": {"path": report_path.name, "sha256": sha256(report_path), "bytes": report_path.stat().st_size},
        "production_deployed": False,
    }
    atomic_json(output / "INDEPENDENT_AUDIT_BINDING.json", audit_binding)
    print(json.dumps(audit_binding, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
