from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256
from cc_hhgt.v30_integrity import canonical_json_sha256
from website.backend.v31_exact_store import is_finalized_exact_pathway_release


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "98_finalize_v31_exact_pathway_website_release.py"
)
SPEC = importlib.util.spec_from_file_location("v31_exact_website_finalizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finalizer_copies_only_verified_audited_release(tmp_path: Path) -> None:
    run_id = "RUN"
    materialized = tmp_path / "materialized"
    audit_root = tmp_path / "audit"
    materialized.mkdir()
    folds = [
        {"cancer_id": cancer}
        for cancer in [*[f"C{i:02d}" for i in range(31)], "HNSC", "LGG"]
    ]
    filter_contract = {
        "enabled": True,
        "within_cancer_min_sample_detection_rate": 0.10,
        "minimum_detected_cancers": 3,
        "evidence_cannot_bypass_filter": True,
    }
    materialized_success = {
        "status": "PASS",
        "run_id": run_id,
        "selected_model": "hgt",
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "cancers": 33,
        "reference_only_cancers": [],
        "heldout_label_columns_in_public_scores": False,
        "pancancer_eligible_lncRNAs": 2751,
        "pancancer_lncrna_filter": filter_contract,
        "fold_audits": folds,
    }
    _write_json(materialized / "SUCCESS.json", materialized_success)
    (materialized / "payload.txt").write_text("exact pathway", encoding="utf-8")
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "status": "PASS",
            "run_id": run_id,
            "selected_model": "hgt",
            "selection_split": "val_only",
            "test_metrics_used": False,
        },
    )
    site_source = tmp_path / "site_source.json"
    _write_json(
        site_source,
        {
            "status": "PASS",
            "production_requires_v31_exact_pathway": True,
            "production_deployed": False,
        },
    )
    files = []
    for path in sorted(materialized.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(materialized)).replace(
                        "\\", "/"
                    ),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    audit_root.mkdir()
    pd.DataFrame(files).to_csv(
        audit_root / "release_file_manifest.tsv", sep="\t", index=False
    )
    _write_json(
        audit_root / "SUCCESS.json",
        {
            "status": "PASS",
            "release_eligible": True,
            "run_id": run_id,
            "selected_model": "hgt",
            "reference_only_cancers": [],
            "pancancer_lncrna_filter": filter_contract,
            "materialization_success_sha256": file_sha256(
                materialized / "SUCCESS.json"
            ),
            "selection_sha256": file_sha256(selection),
            "release_file_merkle_sha256": canonical_json_sha256(files),
        },
    )
    release_base = tmp_path / "releases"
    result = MODULE.finalize_release(
        run_id=run_id,
        release_id="RELEASE",
        materialized_root=materialized,
        release_audit_root=audit_root,
        selection_path=selection,
        site_source_success_path=site_source,
        release_base=release_base,
    )
    final = release_base / "RELEASE" / "final_release"
    assert result["status"] == "PASS"
    assert result["reference_only_cancers"] == []
    assert (final / "SUCCESS.json").is_file()
    assert (release_base / "RELEASE" / "FINALIZATION_SUCCESS.json").is_file()
    assert (final / "FINAL_RELEASE_MANIFEST.tsv").is_file()
    assert not (release_base / ".RELEASE.tmp").exists()
    assert is_finalized_exact_pathway_release(final)

    marker_path = release_base / "RELEASE" / "FINALIZATION_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "FAIL"
    _write_json(marker_path, marker)
    assert not is_finalized_exact_pathway_release(final)
