from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from cc_hhgt.v32.patient_fold_authority import (
    DEFAULT_SEED,
    PatientFoldAuthorityError,
    TCGA_CANCERS,
    build_patient_fold_authority,
    validate_patient_fold_authority,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_v32_patient_fold_authority.py"
AUDITOR = ROOT / "scripts/audit_v32_patient_fold_authority_independent.py"


def _activity(path: Path) -> tuple[Path, int]:
    root = path / "formal_pathway_activity"
    rows = []
    for cancer in TCGA_CANCERS:
        for index in range(5):
            patient = f"PAT-{cancer}-{index:03d}"
            samples = [f"SAMPLE-{cancer}-{index:03d}"]
            if cancer == "BRCA" and index == 0:
                samples.append(f"SAMPLE-{cancer}-{index:03d}-SECOND")
            for sample in samples:
                rows.append(
                    {
                        "cancer_id": cancer,
                        "sample_id": sample,
                        "patient_id": patient,
                        "pathway_id": "HALLMARK_TOY",
                        "activity_score": float(index),
                    }
                )
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "part-0.parquet", index=False)
    return root, len(TCGA_CANCERS) * 5


def _build(tmp_path: Path) -> tuple[Path, Path, int]:
    activity, expected = _activity(tmp_path)
    authority = tmp_path / "authority"
    build_patient_fold_authority(
        activity_root=activity,
        output_root=authority,
        expected_patients=expected,
        launcher_path=BUILDER,
        memory_limit="512MB",
    )
    return activity, authority, expected


def test_builder_publishes_explicit_patient_first_receipt_and_independent_pass(
    tmp_path: Path,
) -> None:
    activity, authority, expected = _build(tmp_path)
    manifest = authority / "SAMPLE_PATIENT_FOLD_MAP.tsv"
    receipt = authority / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    validated = validate_patient_fold_authority(
        manifest, receipt, expected_cancers=TCGA_CANCERS
    )
    assert validated["patients"] == expected
    assert validated["samples"] == expected + 1
    assert validated["sample_fallback_used"] is False
    frame = pd.read_csv(manifest, sep="\t")
    brca_pair = frame.loc[frame.patient_id.eq("PAT-BRCA-000")]
    assert len(brca_pair) == 2
    assert brca_pair.patient_fold_id.nunique() == 1
    published = json.loads(receipt.read_text(encoding="utf-8"))
    assert published["implementation"]["old_prepare_v32_formal_executed"] is False
    assert published["gates"]["sample_barcode_patient_fallback_forbidden"] is True

    audit = tmp_path / "independent_audit"
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--activity-root",
            str(activity),
            "--authority-root",
            str(authority),
            "--output-root",
            str(audit),
            "--expected-patients",
            str(expected),
            "--memory-limit",
            "512MB",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    independent = json.loads((audit / "AUDIT.json").read_text(encoding="utf-8"))
    assert independent["status"] == "PASS_INDEPENDENT_EXPLICIT_PATIENT_FIRST_5FOLD_AUDIT"
    assert independent["observed"]["patients"] == expected
    assert independent["contract"]["production_fold_helper_imported"] is False
    assert independent["contract"]["authority_builder_imported"] is False


def test_independent_auditor_fails_closed_on_manifest_tamper(tmp_path: Path) -> None:
    activity, authority, expected = _build(tmp_path)
    tampered = tmp_path / "tampered_authority"
    shutil.copytree(authority, tampered)
    manifest = tampered / "SAMPLE_PATIENT_FOLD_MAP.tsv"
    frame = pd.read_csv(manifest, sep="\t")
    frame.loc[0, "patient_fold_id"] = (int(frame.loc[0, "patient_fold_id"]) + 1) % 5
    frame.to_csv(manifest, sep="\t", index=False, lineterminator="\n")
    output = tmp_path / "tampered_audit"
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--activity-root",
            str(activity),
            "--authority-root",
            str(tampered),
            "--output-root",
            str(output),
            "--expected-patients",
            str(expected),
            "--memory-limit",
            "512MB",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_builder_never_derives_missing_patient_id_and_refuses_output_reuse(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing_patient"
    missing.mkdir()
    pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "sample_id": ["TCGA-AA-0001-01A"],
            "pathway_id": ["HALLMARK_TOY"],
        }
    ).to_parquet(missing / "part-0.parquet", index=False)
    with pytest.raises(PatientFoldAuthorityError, match="explicit authority columns"):
        build_patient_fold_authority(
            activity_root=missing,
            output_root=tmp_path / "must_not_exist",
            expected_cancers=["BRCA"],
            expected_patients=1,
            memory_limit="512MB",
        )
    assert not (tmp_path / "must_not_exist").exists()

    activity, authority, expected = _build(tmp_path / "second")
    with pytest.raises(FileExistsError, match="refuses output reuse"):
        build_patient_fold_authority(
            activity_root=activity,
            output_root=authority,
            expected_patients=expected,
            seed=DEFAULT_SEED,
            launcher_path=BUILDER,
        )
