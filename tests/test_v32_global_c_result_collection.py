import json

import pytest

from scripts.rbp_encode_v33.collect_global_c_results_no_rehash_149 import (
    required_result_paths,
    validate_success,
)


def test_result_return_requires_finished_matching_fold(tmp_path):
    fold = 2
    _, files = required_result_paths(fold)
    manifest = {}
    for name, relative in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "SUCCESS.json":
            continue
        path.write_bytes(b"verified-by-size")
        manifest[relative] = path.stat().st_size

    success = {
        "status": "SUCCESS",
        "patient_fold": fold,
        "task_id": "v32-g012-g2-c-global-binding-20260928-r1|PATIENT_FOLD_2|CC-HHGT|20260726",
        "graph_variant": "G2",
        "did_not_hit_hard_cap": True,
        "optimizer_steps": 101,
        "best_model_state_size_bytes": manifest[files["best_model_state.pt"]],
        "stop_reason": "PATIENCE",
        "completed_cycles": 8,
        "best_validation_loss": 0.2,
    }
    path = tmp_path / files["SUCCESS.json"]
    path.write_text(json.dumps(success), encoding="utf-8")
    manifest[files["SUCCESS.json"]] = path.stat().st_size
    assert validate_success(tmp_path, fold, manifest)["optimizer_steps"] == 101

    success["did_not_hit_hard_cap"] = False
    path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SUCCESS receipt is invalid"):
        validate_success(tmp_path, fold, manifest)
