#!/usr/bin/env python
"""Materialize the missing hash-bound V3.2 exact-pathway report contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MANIFEST_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_MANIFEST_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_EXACT_PATHWAY_REPORT_BINDING_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": Path(os.path.relpath(path, relative_to)).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    output.relative_to(root)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing non-empty output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)

    exact = root / "artifacts/v32_full_multitask/exact_pathway_release_r2"
    formal = root / "artifacts/formal_release_report_1seed"
    sources = {
        "module_lineage": exact / "MODULE_LINEAGE.json",
        "checkpoint_manifest": exact / "CHECKPOINT_MANIFEST.json",
        "ensemble_source_manifest": exact / "ENSEMBLE_SOURCE_MANIFEST.json",
        "exact_pathway_predictions": exact / "exact_pathway_five_fold_ensemble.parquet",
        "formal_release_audit": formal / "FORMAL_RELEASE_AUDIT.json",
        "model_report": formal / "MODEL_REPORT.md",
        "fold_metrics": formal / "FOLD_METRICS.tsv",
        "cross_hardware_replay_audit": formal / "CROSS_HARDWARE_REPLAY_AUDIT.json",
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing exact-pathway report sources: {missing}")
    lineage = read_json(sources["module_lineage"])
    if (
        lineage.get("analysis_version") != ANALYSIS_VERSION
        or lineage.get("module_id") != "exact_pathway"
        or lineage.get("training_status") != "SUCCESS"
        or lineage.get("trained_from_scratch") is not True
        or lineage.get("five_fold_ensemble") is not True
        or lineage.get("folds") != 5
        or lineage.get("prediction_rows") != 3_300_000
        or lineage.get("old_checkpoint_loaded") is not False
        or lineage.get("old_predictions_used_as_features") is not False
        or lineage.get("old_rankings_used_as_outputs") is not False
    ):
        raise RuntimeError("Exact-pathway lineage semantics are invalid")
    if sha256(sources["exact_pathway_predictions"]) != lineage.get("prediction_sha256"):
        raise RuntimeError("Exact-pathway prediction hash does not match lineage")
    if sha256(sources["checkpoint_manifest"]) != lineage.get("checkpoint_manifest_sha256"):
        raise RuntimeError("Checkpoint manifest hash does not match lineage")
    if sha256(sources["ensemble_source_manifest"]) != lineage.get("ensemble_source_manifest_sha256"):
        raise RuntimeError("Ensemble source manifest hash does not match lineage")
    if pq.ParquetFile(sources["exact_pathway_predictions"]).metadata.num_rows != 3_300_000:
        raise RuntimeError("Exact-pathway prediction row count is not 3.3M")

    manifest_path = output / "V32_EXACT_PATHWAY_REPORT_MANIFEST.json"
    manifest = {
        "format": MANIFEST_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "module_id": "exact_pathway",
        "artifact_id": "v32_exact_pathway_report_manifest",
        "status": "PASS_HASH_BOUND",
        "training_run_id": lineage["training_run_id"],
        "prediction_rows": 3_300_000,
        "fold_count": 5,
        "trained_from_scratch": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "primary_ranking_authoritative": True,
        "historical_predictions_used": False,
        "production_deployed": False,
        "sources": {
            name: record(path, relative_to=output)
            for name, path in sorted(sources.items())
        },
    }
    atomic_json(manifest_path, manifest)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "model_version": "V3.2",
        "module_id": "exact_pathway",
        "status": "PASS_HASH_BOUND",
        "v32_exact_pathway_report_manifest": record(manifest_path, relative_to=output),
        "prediction_rows": 3_300_000,
        "fold_count": 5,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "primary_ranking_authoritative": True,
        "production_deployed": False,
        "release_ready": False,
    }
    atomic_json(output / "EXACT_PATHWAY_REPORT_BINDING.json", binding)
    atomic_json(
        output / "SUCCESS.json",
        {
            "status": "SUCCESS",
            "binding_sha256": sha256(output / "EXACT_PATHWAY_REPORT_BINDING.json"),
            "report_manifest_sha256": sha256(manifest_path),
        },
    )
    print(json.dumps(binding, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
