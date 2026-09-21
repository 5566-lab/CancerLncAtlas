from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_FORMAT = "CC_HHGT_EXACT_TRAINING_STATE_V1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def training_resume_contract(
    cfg: dict[str, Any], *, model_name: str, fold_id: str, seed: int
) -> dict[str, Any]:
    """Return the effective, immutable contract for exact interruption resume.

    The formal-gate hash locks code, source configuration, inputs and assets.
    Effective sections are also recorded because guarded diagnostic invocations
    may apply explicit CLI overrides after gate validation.
    """

    return _json_ready(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "analysis_version": cfg.get("analysis_version"),
            "formal_training_gate_sha256": cfg.get("_formal_training_gate_sha256"),
            "run_id": cfg.get("_run_id"),
            "model_name": str(model_name),
            "fold_id": str(fold_id),
            "seed": int(seed),
            "training": cfg.get("training", {}),
            "state_training": cfg.get("state_training", {}),
            "runtime_graph_sampling": cfg.get("runtime_graph_sampling", {}),
            "graph_contract": cfg.get("graph_contract", {}),
            "formal_execution": cfg.get("formal_execution", {}),
            "task_contract": cfg.get("task_contract", {}),
            "candidate_universe": cfg.get("candidate_universe", {}),
            "exact_pathway_evidence": cfg.get("exact_pathway_evidence", {}),
            "pathway_hierarchy": cfg.get("pathway_hierarchy", {}),
            "residual_learning": cfg.get("residual_learning", {}),
        }
    )


def training_resume_contract_sha256(
    cfg: dict[str, Any], *, model_name: str, fold_id: str, seed: int
) -> str:
    payload = training_resume_contract(
        cfg, model_name=model_name, fold_id=fold_id, seed=seed
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state(torch) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(torch, state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"Training checkpoint RNG state is incomplete: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored because CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count changed across exact resume: "
                f"checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def training_runtime_fingerprint(torch, device: str | Any) -> dict[str, Any]:
    resolved = torch.device(device)
    payload: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": resolved.type,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "cuda_visible_device_count": int(torch.cuda.device_count()),
                "device_name": str(properties.name),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory": int(properties.total_memory),
            }
        )
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        payload["nvidia_driver_version"] = (
            driver.stdout.strip().splitlines()[0]
            if driver.returncode == 0 and driver.stdout.strip()
            else f"UNAVAILABLE:{driver.returncode}"
        )
    return _json_ready(payload)


def validate_training_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_contract_sha256: str,
    expected_model_name: str,
    expected_fold_id: str,
    expected_seed: int,
    expected_runtime_fingerprint: dict[str, Any],
) -> None:
    required = {
        "checkpoint_format",
        "resume_contract_sha256",
        "model_name",
        "fold_id",
        "seed",
        "last_epoch",
        "model_state",
        "state_decoder_state",
        "optimizer_state",
        "scaler_state",
        "rng_state",
        "history",
        "early_stopping_state",
        "runtime_fingerprint",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Training checkpoint is not resumable; missing fields: {missing}")
    disagreements = {}
    expected = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "resume_contract_sha256": str(expected_contract_sha256),
        "model_name": str(expected_model_name),
        "fold_id": str(expected_fold_id),
        "seed": int(expected_seed),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            disagreements[key] = {"expected": value, "observed": checkpoint.get(key)}
    if disagreements:
        raise RuntimeError(f"Training checkpoint resume contract mismatch: {disagreements}")
    if checkpoint["runtime_fingerprint"] != _json_ready(expected_runtime_fingerprint):
        raise RuntimeError(
            "Training checkpoint runtime fingerprint changed across exact resume: "
            f"checkpoint={checkpoint['runtime_fingerprint']}, "
            f"current={_json_ready(expected_runtime_fingerprint)}"
        )
    last_epoch = int(checkpoint["last_epoch"])
    history = checkpoint["history"]
    if last_epoch < 1 or len(history) != last_epoch:
        raise RuntimeError(
            "Training checkpoint epoch/history mismatch: "
            f"last_epoch={last_epoch}, history_rows={len(history)}"
        )
