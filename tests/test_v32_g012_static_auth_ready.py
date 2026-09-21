from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cc_hhgt.v32.training_guard import code_tree_sha256
from scripts.validate_v32_g012_static_auth_ready import (
    StaticAuthorizationError,
    validate_static_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    prepared_parent = tmp_path / "prepared"
    run_parent = tmp_path / "runs"
    receipt_path = tmp_path / "bootstrap" / "STATIC_AUTH_READY.json"
    code_sha = code_tree_sha256(ROOT)
    variants: dict[str, object] = {}
    fieldnames = [
        "task_id",
        "run_id",
        "task_type",
        "model",
        "patient_fold",
        "seed",
        "owner",
        "hardware_class",
        "status",
        "blocked_reason",
        "paid_task",
    ]
    for variant in ("G0", "G1", "G2"):
        run_id = f"v32-g012-{variant.lower()}-paid-gpu-20260831-r1"
        run_root = run_parent / variant
        authorization = run_root / "authorization"
        authorization.mkdir(parents=True)
        config = run_root / "config.yaml"
        config.write_text("execution_control:\n  execution_mode: TRAINING\n", encoding="utf-8")
        task_manifest = authorization / "TASK_MANIFEST.tsv"
        task_ids: list[str] = []
        with task_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for fold in range(5):
                task_id = f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|20260726"
                task_ids.append(task_id)
                writer.writerow(
                    {
                        "task_id": task_id,
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
        input_manifest = authorization / "INPUT_MANIFEST.json"
        fold_inputs = []
        receipt_fold_inputs = []
        for fold in range(5):
            prepared_path = prepared_parent / variant / f"PATIENT_FOLD_{fold}.pt"
            prepared_path.parent.mkdir(parents=True, exist_ok=True)
            prepared_path.write_bytes(f"{variant}:{fold}:prepared".encode("utf-8"))
            digest = _sha256(prepared_path)
            fold_inputs.append(
                {
                    "fold": fold,
                    "path": str(prepared_path.resolve()),
                    "sha256": digest,
                }
            )
            receipt_fold_inputs.append(
                {
                    "fold": fold,
                    "path": str(prepared_path.resolve()),
                    "sha256": digest,
                    "size_bytes": prepared_path.stat().st_size,
                }
            )
        _write_json(
            input_manifest,
            {"fold_inputs": fold_inputs},
        )
        artifact_hashes = {
            "code_sha256": code_sha,
            "config_sha256": _sha256(config),
            "input_manifest_sha256": _sha256(input_manifest),
            "task_manifest_sha256": _sha256(task_manifest),
        }
        approval = authorization / "TRAINING_APPROVAL.json"
        _write_json(
            approval,
            {
                "training_authorized": True,
                "run_id": run_id,
                "endpoint_id": "paid_gpu",
                "hardware_class": "PAID_PREEMPTIBLE_GPU",
                "approved_task_ids": task_ids,
                "paid_enabled": True,
                "max_paid_hours": 96,
                "max_cost_cny": 210,
                "artifact_hashes": artifact_hashes,
            },
        )
        success = authorization / "SUCCESS.json"
        _write_json(
            success,
            {
                "status": "AUTHORIZED_ARTIFACTS_READY",
                "artifact_hashes": artifact_hashes,
            },
        )
        variants[variant] = {
            "run_id": run_id,
            "tasks": 5,
            "folds": list(range(5)),
            "seed": 20260726,
            "fold_inputs": receipt_fold_inputs,
            "task_manifest_sha256": _sha256(task_manifest),
            "input_manifest_sha256": _sha256(input_manifest),
            "approval_sha256": _sha256(approval),
            "success_sha256": _sha256(success),
        }

    patch_ready = receipt_path.parent / "PATCH_READY.json"
    _write_json(
        patch_ready,
        {
            "status": "PATCH_READY",
            "gpu_visible": False,
            "overlay_sha256": "a" * 64,
            "code_tree_sha256": code_sha,
            "code_root": str(ROOT.resolve()),
        },
    )
    contract_paths = (
        "cc_hhgt/v32/training.py",
        "cc_hhgt/v32/training_guard.py",
        "cc_hhgt/v32/gpu_backward_probe.py",
        "scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r1.sh",
        "scripts/cloud_authorize_v32_g012_no_gpu_20260901_r1.sh",
        "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r1.sh",
        "scripts/server_launch_v32_g012_paid_gpu_20260831_r1.sh",
        "scripts/normalize_v32_task_manifest.py",
        "scripts/validate_v32_g012_static_auth_ready.py",
        "config/model_v3_2_g012_paid_gpu_20260831_r1.yaml",
    )
    _write_json(
        receipt_path,
        {
            "status": "STATIC_AUTH_READY",
            "gpu_visible": False,
            "formal_seed": 20260726,
            "code_tree_sha256": code_sha,
            "runtime_contract": {
                "python": "3.12.0",
                "torch": "test",
                "torch_geometric": "test",
                "pyyaml": "test",
                "cuda_available_during_cpu_gate": False,
            },
            "variants": variants,
            "patch_ready_sha256": _sha256(patch_ready),
            "code_contract_sha256": {
                relative: _sha256(ROOT / relative) for relative in contract_paths
            },
        },
    )
    return receipt_path, prepared_parent, run_parent


def test_static_authorization_recomputes_patch_and_full_code_hashes(tmp_path: Path) -> None:
    receipt, prepared, runs = _fixture(tmp_path)
    result = validate_static_authorization(
        receipt_path=receipt,
        code_root=ROOT,
        prepared_parent=prepared,
        run_parent=runs,
    )
    assert result["status"] == "STATIC_AUTH_READY"

    patch = receipt.parent / "PATCH_READY.json"
    payload = json.loads(patch.read_text(encoding="utf-8"))
    payload["code_tree_sha256"] = "0" * 64
    _write_json(patch, payload)
    with pytest.raises(StaticAuthorizationError, match="SHA_DRIFT=PATCH_READY"):
        validate_static_authorization(
            receipt_path=receipt,
            code_root=ROOT,
            prepared_parent=prepared,
            run_parent=runs,
        )


def test_static_authorization_rejects_missing_or_resized_fold(tmp_path: Path) -> None:
    receipt, prepared, runs = _fixture(tmp_path)
    missing = prepared / "G2" / "PATIENT_FOLD_4.pt"
    missing.unlink()
    with pytest.raises(StaticAuthorizationError, match="INPUT_MISSING_OR_SYMLINK=G2:4"):
        validate_static_authorization(
            receipt_path=receipt,
            code_root=ROOT,
            prepared_parent=prepared,
            run_parent=runs,
        )

    receipt, prepared, runs = _fixture(tmp_path / "resized")
    resized = prepared / "G1" / "PATIENT_FOLD_3.pt"
    resized.write_bytes(resized.read_bytes() + b"drift")
    with pytest.raises(StaticAuthorizationError, match="INPUT_BINDING_DRIFT=G1:3"):
        validate_static_authorization(
            receipt_path=receipt,
            code_root=ROOT,
            prepared_parent=prepared,
            run_parent=runs,
        )


def test_static_authorization_requires_formal_seed_and_deployment_contract(
    tmp_path: Path,
) -> None:
    receipt, prepared, runs = _fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["formal_seed"] = 7
    _write_json(receipt, payload)
    with pytest.raises(StaticAuthorizationError, match="FORMAL_SEED_DRIFT"):
        validate_static_authorization(
            receipt_path=receipt,
            code_root=ROOT,
            prepared_parent=prepared,
            run_parent=runs,
        )

    receipt, prepared, runs = _fixture(tmp_path / "contract")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    del payload["code_contract_sha256"][
        "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r1.sh"
    ]
    _write_json(receipt, payload)
    with pytest.raises(StaticAuthorizationError, match="CODE_CONTRACT_MISSING"):
        validate_static_authorization(
            receipt_path=receipt,
            code_root=ROOT,
            prepared_parent=prepared,
            run_parent=runs,
        )
