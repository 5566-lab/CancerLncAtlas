#!/usr/bin/env python
"""Read-only independent audit for one V3.2 Evidence private-head checkpoint.

The script intentionally writes nothing.  It is suitable for piping over SSH
to the pinned server environment and emits one JSON object on stdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def history_audit(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    validation_key = next(
        (key for key in ("validation_loss", "val_loss") if rows and key in rows[0]),
        None,
    )
    best_epoch = None
    if validation_key is not None:
        finite = [
            (index, float(row[validation_key]))
            for index, row in enumerate(rows)
            if row.get(validation_key) not in (None, "")
            and np.isfinite(float(row[validation_key]))
        ]
        if finite:
            best_epoch = min(finite, key=lambda item: item[1])[0]
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "columns": list(rows[0]) if rows else [],
        "best_validation_epoch_zero_based": best_epoch,
    }


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: read_v32_evidence_checkpoint_audit.py CHECKPOINT [HISTORY]")
    checkpoint = Path(sys.argv[1]).resolve()
    history = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else checkpoint.parent / "training_history.tsv"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    private_state = payload.get("private_model_state")
    if not isinstance(private_state, Mapping) or not private_state:
        raise RuntimeError("checkpoint lacks private_model_state")
    recomputed = state_sha256(private_state)
    declared = str(payload.get("final_parameter_sha256", ""))
    keys = (
        "checkpoint_format",
        "analysis_version",
        "patient_fold",
        "seed",
        "private_parameters_fresh_init",
        "initialized_from_checkpoint",
        "historical_evidence_checkpoint_allowed",
        "historical_evidence_result_allowed",
        "pair_evidence_supervision_allowed",
        "family_spf_allowed",
        "core_frozen",
        "core_detached",
        "contains_core_parameters",
        "contains_primary_ranking_parameters",
        "initial_parameter_sha256",
        "final_parameter_sha256",
        "optimizer_steps",
        "confidence_supervision_available",
        "direction_supervision_available",
        "core_checkpoint_sha256",
        "core_parameter_sha256",
        "core_manifest_sha256",
    )
    result = {key: payload.get(key) for key in keys}
    result.update(
        {
            "path": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "recomputed_final_parameter_sha256": recomputed,
            "recomputed_matches_declared": recomputed == declared,
            "initial_differs_from_final": payload.get("initial_parameter_sha256") != declared,
            "private_state_tensor_count": len(private_state),
            "provenance_exclusion_audit": payload.get("provenance_exclusion_audit"),
            "history": history_audit(history),
        }
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
