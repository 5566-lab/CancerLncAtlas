from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from cc_hhgt.v32.training_guard import code_tree_sha256


ROOT = Path(__file__).resolve().parents[1]
APPLY = ROOT / "scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r1.sh"
AUTHORIZE = ROOT / "scripts/cloud_authorize_v32_g012_no_gpu_20260901_r1.sh"
FINALIZE = ROOT / "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r1.sh"


def _python_heredocs(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        index += 1
        block: list[str] = []
        while index < len(lines) and lines[index] != "PY":
            block.append(lines[index])
            index += 1
        if index == len(lines):
            raise AssertionError(f"unterminated Python heredoc in {path}")
        blocks.append("\n".join(block) + "\n")
        index += 1
    return blocks


def _write_overlay(path: Path, *, special: bool = False) -> None:
    with tarfile.open(path, "w:gz") as handle:
        directory = tarfile.TarInfo("cc_hhgt")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        handle.addfile(directory)
        if special:
            fifo = tarfile.TarInfo("cc_hhgt/blocked.fifo")
            fifo.type = tarfile.FIFOTYPE
            fifo.mode = 0o644
            handle.addfile(fifo)
        else:
            payload = b"reviewed overlay\n"
            regular = tarfile.TarInfo("cc_hhgt/reviewed.py")
            regular.size = len(payload)
            regular.mode = 0o644
            handle.addfile(regular, io.BytesIO(payload))


def test_overlay_is_hashed_audited_and_extracted_from_one_descriptor(
    tmp_path: Path,
) -> None:
    processor = next(
        block
        for block in _python_heredocs(APPLY)
        if "PASS_OVERLAY_HASH_AUDIT_EXTRACT" in block
    )
    staging = tmp_path / "staging"
    for directory in ("cc_hhgt", "scripts", "config"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    overlay = tmp_path / "overlay.tar.gz"
    _write_overlay(overlay)
    digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
    result = subprocess.run(
        [sys.executable, "-c", processor, str(overlay), digest, str(staging)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (staging / "cc_hhgt/reviewed.py").read_bytes() == b"reviewed overlay\n"

    special = tmp_path / "special.tar.gz"
    _write_overlay(special, special=True)
    special_digest = hashlib.sha256(special.read_bytes()).hexdigest()
    rejected = subprocess.run(
        [sys.executable, "-c", processor, str(special), special_digest, str(staging)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "SPECIAL_OVERLAY_MEMBER_FORBIDDEN" in rejected.stderr


def test_deployment_contract_and_postprocess_bindings_are_explicit() -> None:
    authorize = AUTHORIZE.read_text(encoding="utf-8")
    finalize = FINALIZE.read_text(encoding="utf-8")
    for relative in (
        "cc_hhgt/v32/training_guard.py",
        "scripts/cloud_apply_v32_g012_code_patch_no_gpu_20260901_r1.sh",
        "scripts/cloud_authorize_v32_g012_no_gpu_20260901_r1.sh",
        "scripts/cloud_finalize_v32_g012_no_gpu_20260901_r1.sh",
    ):
        assert relative in authorize
    assert '"seed": str(formal_seed)' in authorize
    assert '"size_bytes": observed_stat.st_size' in authorize

    assert "EXISTING_WINNER_LACKS_CURRENT_BINDING" in finalize
    assert "EXISTING_ARCHIVE_LACKS_CURRENT_BINDING" in finalize
    assert 'payload.get("best_model_state_sha256")' in finalize
    assert 'payload.get("best_model_state_size_bytes")' in finalize
    assert 'marker_record.get("best_model_state_sha256")' in finalize
    assert "CPU_VALIDATED_TRAINING_OUTPUTS" in finalize
    assert "CPU_VALIDATED_WINNER" in finalize
    assert "CPU_VALIDATED_ARCHIVE" in finalize


def test_no_gpu_guards_require_nonempty_inventory_not_nvidia_smi_exit_zero() -> None:
    for path in (APPLY, AUTHORIZE, FINALIZE):
        source = path.read_text(encoding="utf-8")
        assert 'gpu_inventory="$(nvidia-smi -L 2>/dev/null || true)"' in source
        assert 'test -n "${gpu_inventory//[[:space:]]/}"' in source
        assert "nvidia-smi -L >/dev/null 2>&1" not in source


def test_cpu_finalizer_binds_marker_success_and_actual_checkpoint(tmp_path: Path) -> None:
    validator = next(
        block
        for block in _python_heredocs(FINALIZE)
        if '"CPU_VALIDATED_TRAINING_OUTPUTS"' in block
    )
    static_auth = tmp_path / "STATIC_AUTH_READY.json"
    gpu_preflight = tmp_path / "GPU_RUNTIME_PREFLIGHT.json"
    telemetry = tmp_path / "00_nvidia_smi_memory.csv"
    static_auth.write_text('{"status":"STATIC_AUTH_READY"}\n', encoding="utf-8")
    gpu_preflight.write_text('{"status":"GPU_RUNTIME_PREFLIGHT_PASS"}\n', encoding="utf-8")
    telemetry.write_text("timestamp,phase,memory\n", encoding="utf-8")
    run_parent = tmp_path / "runs"
    code_sha = code_tree_sha256(ROOT)
    marker: dict[str, object] = {
        "status": "GPU_TRAINING_COMPLETE",
        "completed_tasks": 15,
        "static_auth_ready_sha256": hashlib.sha256(static_auth.read_bytes()).hexdigest(),
        "gpu_runtime_preflight_sha256": hashlib.sha256(gpu_preflight.read_bytes()).hexdigest(),
        "nvidia_smi_memory_telemetry_sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
        "variants": {},
    }
    checkpoints: list[Path] = []
    for variant in ("G0", "G1", "G2"):
        run_id = f"v32-g012-{variant.lower()}-paid-gpu-20260831-r1"
        successes = []
        success_hashes = {}
        for fold in range(5):
            task_id = f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|20260726"
            task_root = run_parent / variant / "training" / task_id.replace("|", "__")
            task_root.mkdir(parents=True)
            checkpoint = task_root / "best_model_state.pt"
            checkpoint.write_bytes(f"checkpoint:{variant}:{fold}".encode("utf-8"))
            checkpoints.append(checkpoint)
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            success = task_root / "SUCCESS.json"
            success.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "run_id": run_id,
                        "task_id": task_id,
                        "patient_fold": fold,
                        "seed": 20260726,
                        "artifact_hashes": {"code_sha256": code_sha},
                        "optimizer_steps": 1,
                        "grad_finite": True,
                        "parameters_finite": True,
                        "parameter_delta_positive": True,
                        "best_model_state_sha256": checkpoint_sha,
                        "best_model_state_size_bytes": checkpoint.stat().st_size,
                        "prepared_artifact_sha256": "a" * 64,
                        "best_validation_loss": float(fold + 1),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            success_sha = hashlib.sha256(success.read_bytes()).hexdigest()
            success_hashes[str(fold)] = success_sha
            successes.append(
                {
                    "fold": fold,
                    "task_id": task_id,
                    "success_json_sha256": success_sha,
                    "best_model_state_sha256": checkpoint_sha,
                    "best_model_state_size_bytes": checkpoint.stat().st_size,
                }
            )
        marker["variants"][variant] = {
            "run_id": run_id,
            "tasks": 5,
            "folds": list(range(5)),
            "success_sha256": success_hashes,
            "successes": successes,
        }
    marker_path = tmp_path / "GPU_TRAINING_COMPLETE.json"
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    binding = tmp_path / "CPU_VALIDATED_TRAINING_OUTPUTS.json"
    arguments = [
        sys.executable,
        "-c",
        validator,
        str(marker_path),
        str(static_auth),
        str(gpu_preflight),
        str(telemetry),
        str(run_parent),
        str(ROOT),
        str(binding),
    ]
    passed = subprocess.run(arguments, check=False, capture_output=True, text=True, cwd=ROOT)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(binding.read_text(encoding="utf-8"))["status"] == (
        "CPU_VALIDATED_TRAINING_OUTPUTS"
    )

    checkpoints[-1].write_bytes(b"x" * checkpoints[-1].stat().st_size)
    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True, cwd=ROOT)
    assert rejected.returncode != 0
    assert "TRAINING_CHECKPOINT_BINDING_DRIFT=G2:4" in rejected.stderr
