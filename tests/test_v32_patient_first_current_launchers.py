from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from cc_hhgt.v32.patient_first_lineage import (
    PatientFirstLineageError,
    validate_patient_first_output_lineage,
    write_patient_first_output_lineage_audit,
)
from cc_hhgt.v32.patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "artifacts/v32_patient_fold_authority_20260829_r1"
PATIENT_FOLDS = AUTHORITY / "SAMPLE_PATIENT_FOLD_MAP.tsv"
RECEIPT = AUTHORITY / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
CURRENT_LAUNCHERS = (
    "scripts/server_launch_v32_atac_patient_first_oof_r5.sh",
    "scripts/server_launch_v32_genomic_patient_first_cpu_r3.sh",
    "scripts/server_launch_v32_external_router_patient_first_r1.sh",
    "scripts/server_launch_v32_hierarchical_patient_first_gpu_r3.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_patient_first_r6.sh",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_lineage_validator_pins_both_physical_shas_and_scope(tmp_path: Path) -> None:
    fold_lineage = tmp_path / "INPUT_LINEAGE_AUDIT.json"
    _write_json(
        fold_lineage,
        {
            "status": "PASS",
            "input_artifacts": [
                {
                    "role": "folds",
                    "path": "/authority/SAMPLE_PATIENT_FOLD_MAP.tsv",
                    "sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
                }
            ],
        },
    )
    run_lineage = tmp_path / "LINEAGE.json"
    _write_json(run_lineage, {"status": "SUCCESS", "folds": 5})
    audit = validate_patient_first_output_lineage(
        patient_folds_path=PATIENT_FOLDS,
        patient_fold_authority_receipt_path=RECEIPT,
        fold_lineage_paths=[fold_lineage],
        run_lineage_paths=[run_lineage],
        component="atac",
    )
    assert audit["status"] == "PASS_FROZEN_PATIENT_FIRST_OUTPUT_LINEAGE"
    assert audit["authority"] == {
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "cancers": 33,
        "patients": 10432,
        "folds": 5,
        "legacy_patient_fold_manifest_accepted": False,
    }
    assert audit["old_fold_output_accepted"] is False


def test_lineage_validator_rejects_old_fold_output_even_with_current_gate(
    tmp_path: Path,
) -> None:
    old = tmp_path / "OLD_INPUT_LINEAGE.json"
    _write_json(
        old,
        {
            "input_artifacts": [
                {
                    "role": "folds",
                    "path": "/old/SAMPLE_PATIENT_FOLD_MAP.tsv",
                    "sha256": "d" * 64,
                }
            ]
        },
    )
    with pytest.raises(PatientFirstLineageError, match="SHA_DRIFT"):
        validate_patient_first_output_lineage(
            patient_folds_path=PATIENT_FOLDS,
            patient_fold_authority_receipt_path=RECEIPT,
            fold_lineage_paths=[old],
            component="mutation_cnv",
        )


def test_lineage_validator_rejects_legacy_filename_and_missing_map_claim(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    _write_json(
        legacy,
        {
            "path": "/old/PATIENT_FOLD_MANIFEST.tsv",
            "sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        },
    )
    with pytest.raises(PatientFirstLineageError, match="LEGACY_PATIENT_FOLD"):
        validate_patient_first_output_lineage(
            patient_folds_path=PATIENT_FOLDS,
            patient_fold_authority_receipt_path=RECEIPT,
            fold_lineage_paths=[legacy],
            component="external_router",
        )
    missing = tmp_path / "missing.json"
    _write_json(missing, {"status": "PASS"})
    with pytest.raises(PatientFirstLineageError, match="LACKS_FROZEN_MAP_SHA"):
        validate_patient_first_output_lineage(
            patient_folds_path=PATIENT_FOLDS,
            patient_fold_authority_receipt_path=RECEIPT,
            fold_lineage_paths=[missing],
            component="external_router",
        )


def test_lineage_audit_writer_refuses_reuse(tmp_path: Path) -> None:
    destination = tmp_path / "PATIENT_FIRST_LINEAGE_AUDIT.json"
    write_patient_first_output_lineage_audit({"status": "PASS"}, destination)
    with pytest.raises(PatientFirstLineageError, match="refuses output reuse"):
        write_patient_first_output_lineage_audit({"status": "PASS"}, destination)


def test_current_launchers_are_dated_pinned_and_never_resume_success() -> None:
    for relative in CURRENT_LAUNCHERS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256 in source, relative
        assert FROZEN_V32_RECEIPT_SHA256 in source, relative
        assert "SAMPLE_PATIENT_FOLD_MAP.tsv" in source, relative
        assert "PATIENT_FOLD_AUTHORITY_RECEIPT.json" in source, relative
        assert "gate_v32_patient_fold_authority.py" in source, relative
        assert "PATIENT_FOLD_MANIFEST.tsv" not in source, relative
        assert "/results/v32_routed_candidate/" not in source, relative
        assert "/inputs/v32_routed_candidate/" not in source, relative
        assert "if ! test -f" not in source, relative
        assert "exec \"$script_root/server_" not in source, relative

    atac = (ROOT / CURRENT_LAUNCHERS[0]).read_text(encoding="utf-8")
    assert "v32_atac_patient_first_oof_20260829_r5" in atac
    assert "v32_atac_fresh_oof_20260829_r3" not in atac
    assert "REFUSING_ATAC_R5_OUTPUT_REUSE" in atac

    genomic = (ROOT / CURRENT_LAUNCHERS[1]).read_text(encoding="utf-8")
    assert "v32_genomic_patient_first_20260829_r3" in genomic
    assert "run_v32_streaming_segment_cnv.py" in genomic
    assert "run_v32_genomic_training.py" in genomic
    assert "REFUSING_GENOMIC_R3_OUTPUT_REUSE" in genomic


def test_routing_launchers_preserve_test_firewall_and_select_explicit_g2() -> None:
    external = (ROOT / CURRENT_LAUNCHERS[2]).read_text(encoding="utf-8")
    assert "BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE" in external
    assert "CALLER_MUST_PIN_V32_FAIR_INPUT_ACCEPTANCE_SHA256" in external
    assert '\"graph_variant\": \"G2\"' in external

    hierarchical = (ROOT / CURRENT_LAUNCHERS[3]).read_text(encoding="utf-8")
    arm = "v32_g012_patient_first_20260829_r2/FORMAL_PREPARED_FOLDS/G2"
    assert arm in hierarchical
    blocker = hierarchical.index("BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE")
    mkdir = hierarchical.index('mkdir -p "$result_root/logs"')
    assert blocker < mkdir
    assert "exit 41" in hierarchical[:mkdir]

    staging = (ROOT / CURRENT_LAUNCHERS[4]).read_text(encoding="utf-8")
    assert "V32_G012_GRAPH_VARIANT_IS_REQUIRED" in staging
    assert 'prepared_root="$prepared_parent/$variant"' in staging
    assert "BLOCKED_PENDING_WINNER_LOCK_SEALED_TEST_INFERENCE" in staging
    assert "extract_v32_primary_fold_views.py" not in staging


def test_hierarchical_config_writes_only_new_dated_result_tree() -> None:
    config = (
        ROOT / "config/model_v3_2_hierarchical_patient_first_20260829_r3.yaml"
    ).read_text(encoding="utf-8")
    dated = "/results/v32_hierarchical_patient_first_20260829_r3/"
    assert dated in config
    assert "artifacts/routed_candidate_prepared" not in config
    assert "artifacts/v32_hierarchical_routed_candidate" not in config
    inference = (ROOT / "cc_hhgt/v32/hierarchical_candidate_inference.py").read_text(
        encoding="utf-8"
    )
    assert '"preparation_manifest"' in inference
    assert '"patient_fold_authority"' in inference


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_current_shell_launchers_pass_bash_syntax_check() -> None:
    for relative in CURRENT_LAUNCHERS:
        completed = subprocess.run(
            [shutil.which("bash") or "bash", "-n", str(ROOT / relative)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (relative, completed.stderr)
