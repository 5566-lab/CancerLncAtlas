from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from cc_hhgt.v32.g012_winner_lock import (
    G012WinnerLockError,
    lock_g012_validation_winner,
    sha256_file,
)
from cc_hhgt.v32.sealed_test_inference import _validate_winner_declaration
from cc_hhgt.v32.patient_fold_authority import (
    validate_frozen_v32_patient_fold_binding,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "artifacts/v32_patient_fold_authority_20260829_r1"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    prepared_parent = tmp_path / "prepared"
    run_parent = tmp_path / "runs"
    candidate = prepared_parent / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    prepared_parent.mkdir()
    candidate.write_bytes(b"fresh-candidate-authority")
    authority_audit = validate_frozen_v32_patient_fold_binding(
        AUTHORITY / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        AUTHORITY / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    variant_losses = {"G0": 0.40, "G1": 0.20, "G2": 0.30}
    for variant, base_loss in variant_losses.items():
        prepared = prepared_parent / variant
        prepared.mkdir()
        for name in (
            "SAMPLE_PATIENT_FOLD_MAP.tsv",
            "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
        ):
            shutil.copy2(AUTHORITY / name, prepared / name)
        _write_json(
            prepared / "PATIENT_FOLD_BINDING.json",
            {
                "format": "CC_HHGT_V3_2_FORMAL_PREPARED_PATIENT_FOLD_BINDING_V1",
                "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
                "authority": authority_audit,
                "sample_patient_fold_map": {
                    "path": str(AUTHORITY / "SAMPLE_PATIENT_FOLD_MAP.tsv"),
                    "sha256": sha256_file(AUTHORITY / "SAMPLE_PATIENT_FOLD_MAP.tsv"),
                },
                "authority_receipt": {
                    "path": str(AUTHORITY / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"),
                    "sha256": sha256_file(AUTHORITY / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"),
                },
                "sample_id_patient_fallback_used": False,
                "legacy_patient_fold_manifest_used": False,
            },
        )
        _write_json(
            prepared / "FORMAL_GRAPH_VARIANT.json",
            {
                "format": "CANCERLNCATLAS_V32_FORMAL_GRAPH_VARIANT_ROOT_V1",
                "variant": variant,
                "legacy_root_fold_payloads_allowed": False,
            },
        )
        fold_inputs = []
        for fold in range(5):
            path = prepared / f"PATIENT_FOLD_{fold}.pt"
            path.write_bytes(f"{variant}-prepared-{fold}".encode())
            fold_inputs.append({"fold": fold, "path": str(path.resolve()), "sha256": sha256_file(path)})

        run = run_parent / variant
        authorization = run / "authorization"
        training = run / "training"
        authorization.mkdir(parents=True)
        training.mkdir()
        config = run / "config.yaml"
        config.write_text("execution_control:\n  execution_mode: TRAINING\n", encoding="utf-8")
        input_manifest = authorization / "INPUT_MANIFEST.json"
        _write_json(input_manifest, {"fold_inputs": fold_inputs})
        task_manifest = authorization / "TASK_MANIFEST.tsv"
        fieldnames = [
            "task_id", "run_id", "task_type", "model", "patient_fold", "seed",
            "owner", "hardware_class", "status", "blocked_reason", "paid_task",
        ]
        run_id = f"fresh-{variant}"
        with task_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for fold in range(5):
                writer.writerow(
                    {
                        "task_id": f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|20260726",
                        "run_id": run_id,
                        "task_type": "CC_HHGT_PATIENT_FOLD",
                        "model": "CC-HHGT",
                        "patient_fold": fold,
                        "seed": 20260726,
                        "owner": "paid_gpu",
                        "hardware_class": "PAID_PREEMPTIBLE_GPU",
                        "status": "PENDING",
                        "blocked_reason": "",
                        "paid_task": "true",
                    }
                )
        hashes = {
            "code_sha256": "a" * 64,
            "config_sha256": sha256_file(config),
            "input_manifest_sha256": sha256_file(input_manifest),
            "task_manifest_sha256": sha256_file(task_manifest),
        }
        for fold in range(5):
            task_id = f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|20260726"
            task_dir = training / task_id.replace("|", "__")
            task_dir.mkdir()
            (task_dir / "best_model_state.pt").write_bytes(
                f"{variant}-fresh-checkpoint-{fold}".encode()
            )
            _write_json(
                task_dir / "SUCCESS.json",
                {
                    "status": "SUCCESS",
                    "patient_fold": fold,
                    "task_id": task_id,
                    "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
                    "evidence_integration_mode": "external_router",
                    "test_labels_read": False,
                    "test_metrics_computed": False,
                    "best_validation_loss": base_loss + fold / 1000,
                    "artifact_hashes": hashes,
                },
            )
    return {
        "prepared_parent": prepared_parent,
        "run_parent": run_parent,
        "candidate": candidate,
    }


def test_locks_lowest_mean_validation_arm_and_emits_sealed_compatible_declaration(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "winner"
    receipt = lock_g012_validation_winner(
        prepared_parent=fixture["prepared_parent"],
        run_parent=fixture["run_parent"],
        candidate_authority_path=fixture["candidate"],
        output_root=output,
        pair_fold_seed=20260826,
    )
    assert receipt["winner"] == "G1"
    declaration = json.loads(
        (output / "G012_VALIDATION_WINNER_LOCK.json").read_text(encoding="utf-8")
    )
    prepared, checkpoints = _validate_winner_declaration(
        declaration, expected_graph_variant="G1"
    )
    assert set(prepared) == set(range(5))
    assert set(checkpoints) == set(range(5))
    assert declaration["test_inputs_opened_before_winner_lock"] is False
    assert declaration["heldout_test_metrics_used_for_selection"] is False


def test_rejects_test_metric_claim_and_output_reuse(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad = next((fixture["run_parent"] / "G2" / "training").glob("*/SUCCESS.json"))
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["test_metrics_computed"] = True
    _write_json(bad, payload)
    with pytest.raises(G012WinnerLockError, match="SUCCESS lineage drift"):
        lock_g012_validation_winner(
            prepared_parent=fixture["prepared_parent"],
            run_parent=fixture["run_parent"],
            candidate_authority_path=fixture["candidate"],
            output_root=tmp_path / "bad",
            pair_fold_seed=20260826,
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(G012WinnerLockError, match="output reuse"):
        lock_g012_validation_winner(
            prepared_parent=fixture["prepared_parent"],
            run_parent=fixture["run_parent"],
            candidate_authority_path=fixture["candidate"],
            output_root=occupied,
            pair_fold_seed=20260826,
        )


def test_rejects_cross_fold_code_hash_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad = next((fixture["run_parent"] / "G2" / "training").glob("*/SUCCESS.json"))
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["artifact_hashes"]["code_sha256"] = "b" * 64
    _write_json(bad, payload)
    with pytest.raises(G012WinnerLockError, match="authorization hash closure|shared code tree"):
        lock_g012_validation_winner(
            prepared_parent=fixture["prepared_parent"],
            run_parent=fixture["run_parent"],
            candidate_authority_path=fixture["candidate"],
            output_root=tmp_path / "code-drift",
            pair_fold_seed=20260826,
        )
