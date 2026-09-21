from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import file_sha256


def audit_convergence_task(
    task_root: str | Path,
    *,
    expected_model: str,
    expected_fold: str,
    expected_seed: int,
    expected_cap: int,
) -> dict[str, Any]:
    root = Path(task_root).resolve()
    success_path = root / "SUCCESS.json"
    history_path = root / "training_history.tsv"
    required = [success_path, history_path, root / "best_pathway.pt", root / "best_state.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "status": "INCOMPLETE",
            "release_eligible": False,
            "task_root": str(root),
            "missing": missing,
        }
    success = json.loads(success_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path, sep="\t")
    identity = {
        "model_name": str(expected_model),
        "fold_id": str(expected_fold),
        "seed": int(expected_seed),
        "configured_epoch_cap": int(expected_cap),
    }
    disagreements = {
        key: {"expected": value, "observed": success.get(key)}
        for key, value in identity.items()
        if success.get(key) != value
    }
    if disagreements:
        raise RuntimeError(f"Convergence diagnostic identity mismatch: {disagreements}")
    if history.empty or int(history.epoch.iloc[-1]) != int(success["epochs_completed"]):
        raise RuntimeError("Convergence diagnostic history is incomplete or inconsistent")
    validation = history.loc[history.validation_full_coverage_cycle.astype(bool)].copy()
    if validation.empty:
        raise RuntimeError("Convergence diagnostic has no full-coverage validation cycle")
    score_columns = ["pathway_checkpoint_score", "state_checkpoint_score"]
    if not set(score_columns).issubset(validation):
        raise RuntimeError(f"Convergence diagnostic lacks scores: {score_columns}")
    window_cycles = min(3, max(len(validation) - 1, 0))
    gains: dict[str, float | None] = {}
    for column in score_columns:
        if window_cycles:
            gain = float(validation[column].iloc[-1] - validation[column].iloc[-1 - window_cycles])
            gains[column] = gain if np.isfinite(gain) else None
        else:
            gains[column] = None
    hard_cap = bool(success.get("hit_hard_epoch_cap"))
    joint_stop = bool(success.get("stopped_by_joint_patience"))
    patience_cycles = int(success.get("configured_runtime_checkpoint_patience_cycles", 0))
    stale = {
        "pathway": int(success.get("pathway_stale_validation_cycles", -1)),
        "state": int(success.get("state_stale_validation_cycles", -1)),
    }
    converged = bool(
        not hard_cap
        and joint_stop
        and patience_cycles > 0
        and stale["pathway"] >= patience_cycles
        and stale["state"] >= patience_cycles
    )
    min_delta = float(success.get("checkpoint_min_delta", 0.0))
    materially_improving = any(
        gain is not None and gain >= min_delta for gain in gains.values()
    )
    if converged:
        status = "PASS_CONVERGED"
    elif hard_cap and materially_improving:
        status = "FAIL_HARD_CAP_IMPROVING"
    elif hard_cap:
        status = "FAIL_HARD_CAP_UNRESOLVED"
    else:
        status = "FAIL_EARLY_STOP_CONTRACT"
    return {
        "status": status,
        "release_eligible": False,
        "diagnostic_only": True,
        "task_root": str(root),
        "identity": identity,
        "epochs_completed": int(success["epochs_completed"]),
        "validation_cycles": int(len(validation)),
        "runtime_num_chunks": int(validation.runtime_num_chunks.iloc[-1]),
        "hit_hard_epoch_cap": hard_cap,
        "stopped_by_joint_patience": joint_stop,
        "configured_patience_cycles": patience_cycles,
        "stale_validation_cycles": stale,
        "window_validation_cycles": window_cycles,
        "score_gain_over_window": gains,
        "checkpoint_min_delta": min_delta,
        "materially_improving_at_end": materially_improving,
        "best_pathway_epoch": int(success["best_pathway_epoch"]),
        "best_state_epoch": int(success["best_state_epoch"]),
        "success_sha256": file_sha256(success_path),
        "history_sha256": file_sha256(history_path),
    }

