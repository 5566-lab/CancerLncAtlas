#!/usr/bin/env python3
"""Validate every file in a frozen V3.2 core-embedding bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    manifest_path = root / "CORE_EMBEDDING_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[dict[str, object]] = []
    checked: list[dict[str, object]] = []
    for fold_text, fold in sorted(manifest["folds"].items(), key=lambda item: int(item[0])):
        fold_root = root / f"patient_fold={fold_text}"
        for node_type, record in sorted(fold["exports"].items()):
            path = fold_root / Path(record["path"]).name
            observed = file_sha256(path) if path.is_file() else None
            row = {
                "patient_fold": int(fold_text),
                "node_type": node_type,
                "path": str(path),
                "expected_sha256": record["sha256"],
                "observed_sha256": observed,
                "match": observed == record["sha256"],
            }
            checked.append(row)
            if not row["match"]:
                failures.append(row)
        lineage_path = fold_root / "LINEAGE.json"
        if not lineage_path.is_file():
            failures.append(
                {
                    "patient_fold": int(fold_text),
                    "node_type": "LINEAGE",
                    "path": str(lineage_path),
                    "failure": "MISSING",
                }
            )
            continue
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if (
            lineage.get("patient_fold") != int(fold_text)
            or lineage.get("old_checkpoint_loaded") is not False
            or lineage.get("trained_from_scratch") is not True
        ):
            failures.append(
                {
                    "patient_fold": int(fold_text),
                    "node_type": "LINEAGE",
                    "path": str(lineage_path),
                    "failure": "INVALID_FOLD_PROVENANCE",
                }
            )

    result = {
        "status": "PASS" if not failures else "FAIL",
        "analysis_version": manifest.get("analysis_version"),
        "export_format": manifest.get("export_format"),
        "manifest_sha256": file_sha256(manifest_path),
        "checked_embedding_files": len(checked),
        "checked_fold_lineages": len(manifest.get("folds", {})),
        "all_embeddings_from_newly_trained_v32_core": manifest.get(
            "all_embeddings_from_newly_trained_v32_core"
        ),
        "historical_checkpoint_loaded": manifest.get("historical_checkpoint_loaded"),
        "historical_prediction_loaded": manifest.get("historical_prediction_loaded"),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
