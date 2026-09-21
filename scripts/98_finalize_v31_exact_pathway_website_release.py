#!/usr/bin/env python3
"""Atomically finalize an audited V3.1 exact-pathway website release."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, canonical_json_sha256


REQUIRED_FORMAL_CANCERS = {"HNSC", "LGG"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"Unsafe release manifest path: {value}")
    return relative


def _filter_is_frozen(payload: dict) -> bool:
    contract = payload.get("pancancer_lncrna_filter", {})
    return (
        bool(contract.get("enabled"))
        and bool(contract.get("evidence_cannot_bypass_filter"))
        and int(contract.get("minimum_detected_cancers", -1)) == 3
        and float(contract.get("within_cancer_min_sample_detection_rate", -1))
        == 0.10
    )


def finalize_release(
    *,
    run_id: str,
    release_id: str,
    materialized_root: Path,
    release_audit_root: Path,
    selection_path: Path,
    site_source_success_path: Path,
    release_base: Path,
) -> dict:
    materialized_root = materialized_root.resolve()
    release_audit_root = release_audit_root.resolve()
    selection_path = selection_path.resolve()
    site_source_success_path = site_source_success_path.resolve()
    release_base = release_base.resolve()

    materialized_success_path = materialized_root / "SUCCESS.json"
    audit_success_path = release_audit_root / "SUCCESS.json"
    manifest_path = release_audit_root / "release_file_manifest.tsv"
    for required in (
        materialized_success_path,
        audit_success_path,
        manifest_path,
        selection_path,
        site_source_success_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    materialized = _json(materialized_success_path)
    audit = _json(audit_success_path)
    selection = _json(selection_path)
    site_source = _json(site_source_success_path)
    selected_model = str(selection.get("selected_model", ""))
    if (
        materialized.get("status") != "PASS"
        or materialized.get("run_id") != run_id
        or materialized.get("selected_model") != selected_model
        or materialized.get("pathway_target_level") != "exact_pathway"
        or materialized.get("pathway_family_role") != "auxiliary_hierarchy_only"
        or int(materialized.get("cancers", -1)) != 33
        or list(materialized.get("reference_only_cancers", ["missing"])) != []
        or bool(materialized.get("heldout_label_columns_in_public_scores"))
        or not _filter_is_frozen(materialized)
    ):
        raise RuntimeError("Materialized V3.1 website contract is not release eligible")
    fold_audits = materialized.get("fold_audits", [])
    fold_cancers = {str(row.get("cancer_id")) for row in fold_audits}
    if (
        len(fold_audits) != 33
        or len(fold_cancers) != 33
        or not REQUIRED_FORMAL_CANCERS.issubset(fold_cancers)
    ):
        raise RuntimeError("Materialized release lacks 33 folds including HNSC/LGG")
    if (
        selection.get("status") != "PASS"
        or selection.get("run_id") != run_id
        or selection.get("selection_split") != "val_only"
        or bool(selection.get("test_metrics_used"))
        or selected_model not in {"rgcn", "hgt", "cc_hhgt"}
    ):
        raise RuntimeError("Final release requires validation-only model selection")
    if (
        audit.get("status") != "PASS"
        or not bool(audit.get("release_eligible"))
        or audit.get("run_id") != run_id
        or audit.get("selected_model") != selected_model
        or list(audit.get("reference_only_cancers", ["missing"])) != []
        or not _filter_is_frozen(audit)
        or audit.get("materialization_success_sha256")
        != file_sha256(materialized_success_path)
        or audit.get("selection_sha256") != file_sha256(selection_path)
    ):
        raise RuntimeError("V3.1 website release audit is not release eligible")
    if (
        site_source.get("status") != "PASS"
        or not bool(site_source.get("production_requires_v31_exact_pathway"))
        or bool(site_source.get("production_deployed"))
    ):
        raise RuntimeError("Website source snapshot is not the production-fail-closed release")

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    required_columns = {"relative_path", "size_bytes", "sha256"}
    if manifest.empty or not required_columns.issubset(manifest.columns):
        raise RuntimeError("Website release audit file manifest is empty or malformed")
    if manifest.relative_path.astype(str).duplicated().any():
        raise RuntimeError("Website release audit file manifest contains duplicate paths")
    files: list[dict] = []
    for row in manifest.itertuples(index=False):
        relative = _safe_relative(str(row.relative_path))
        source = materialized_root / relative
        if (
            not source.is_file()
            or source.stat().st_size != int(row.size_bytes)
            or file_sha256(source) != str(row.sha256)
        ):
            raise RuntimeError(f"Materialized release file failed manifest verification: {relative}")
        files.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "size_bytes": int(row.size_bytes),
                "sha256": str(row.sha256),
            }
        )
    if canonical_json_sha256(files) != audit.get("release_file_merkle_sha256"):
        raise RuntimeError("Materialized website release file Merkle drifted")

    destination = release_base / release_id
    temporary = release_base / f".{release_id}.tmp"
    if destination.exists() or temporary.exists():
        raise RuntimeError(f"Finalizer refuses existing release destination: {destination}")
    final_release = temporary / "final_release"
    final_release.mkdir(parents=True)
    for entry in files:
        relative = _safe_relative(entry["relative_path"])
        target = final_release / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(materialized_root / relative, target)
        if file_sha256(target) != entry["sha256"]:
            raise RuntimeError(f"Copied final-release file hash mismatch: {relative}")

    audit_target = final_release / "release_audit"
    shutil.copytree(release_audit_root, audit_target)
    reports = final_release / "reports"
    reports.mkdir()
    shutil.copy2(selection_path, reports / "MODEL_SELECTION.json")
    shutil.copy2(site_source_success_path, reports / "SITE_SOURCE_RELEASE_SUCCESS.json")
    final_contract = {
        "status": "PASS",
        "release_eligible": True,
        "production_deployed": False,
        "release_id": release_id,
        "run_id": run_id,
        "selected_model": selected_model,
        "selection_split": "val_only",
        "test_metrics_used_for_selection": False,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "cancers": 33,
        "required_formal_cancers": sorted(REQUIRED_FORMAL_CANCERS),
        "reference_only_cancers": [],
        "pancancer_eligible_lncRNAs": int(
            materialized["pancancer_eligible_lncRNAs"]
        ),
        "pancancer_lncrna_filter": materialized["pancancer_lncrna_filter"],
        "materialization_success_sha256": file_sha256(materialized_success_path),
        "website_release_audit_sha256": file_sha256(audit_success_path),
        "selection_sha256": file_sha256(selection_path),
        "site_source_success_sha256": file_sha256(site_source_success_path),
        "materialized_file_merkle_sha256": audit["release_file_merkle_sha256"],
    }
    atomic_write_json(final_release / "FINAL_RELEASE.json", final_contract)
    inventory = []
    for path in sorted(final_release.rglob("*")):
        if path.is_file() and path.name != "FINAL_RELEASE_MANIFEST.tsv":
            inventory.append(
                {
                    "relative_path": str(path.relative_to(final_release)).replace(
                        "\\", "/"
                    ),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    write_table(pd.DataFrame(inventory), final_release / "FINAL_RELEASE_MANIFEST.tsv")
    final_contract["final_release_files"] = len(inventory)
    final_contract["final_release_file_merkle_sha256"] = canonical_json_sha256(
        inventory
    )
    atomic_write_json(temporary / "FINALIZATION_SUCCESS.json", final_contract)
    release_base.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    print(json.dumps(final_contract, ensure_ascii=False, indent=2))
    return final_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--release-audit-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--site-source-success", required=True)
    parser.add_argument("--release-base", required=True)
    args = parser.parse_args()
    finalize_release(
        run_id=args.run_id,
        release_id=args.release_id,
        materialized_root=Path(args.materialized_root),
        release_audit_root=Path(args.release_audit_root),
        selection_path=Path(args.selection),
        site_source_success_path=Path(args.site_source_success),
        release_base=Path(args.release_base),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
