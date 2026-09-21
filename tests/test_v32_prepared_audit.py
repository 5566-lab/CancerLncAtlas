from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v32.audit import build_code_only_readiness
from cc_hhgt.v32.orchestration import (
    build_dry_run_report,
    build_task_manifest,
    write_dry_run_report,
    write_task_manifest,
)
from cc_hhgt.v32.prepared import build_prepared_fold_payload


def test_prepared_payload_is_untrained_and_aligned() -> None:
    batch = {
        "candidate_batch": {"l": np.array([0, 1]), "p": np.array([1, 0]), "c": np.array([0, 0])},
        "base_logit": np.array([0.1, -0.2]),
        "conservation_context": np.zeros((2, 4)),
        "proxy_label": np.array([1, 0]),
        "weak_positive": np.array([False, False]),
    }
    hashes = {
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "task_manifest_sha256": "d" * 64,
    }
    payload = build_prepared_fold_payload(
        patient_fold=0,
        bundle=object(),
        feature_dim=2,
        legacy_model_config={"training": {}},
        train_batches=[batch],
        validation_batches=[batch],
        artifact_hashes=hashes,
        conservation_context_features=4,
    )
    assert payload["contains_optimizer_state"] is False
    assert payload["contains_trained_parameters"] is False


def test_readiness_requires_no_approval_checkpoint_or_process(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "analysis_version": "test",
                "execution_control": {
                    "execution_mode": "CODE_ONLY",
                    "training_authorized": False,
                    "paid_enabled": False,
                    "max_paid_hours": 0,
                    "max_cost_cny": 0,
                },
                "task_contract": {
                    "target_level": "exact_pathway",
                    "target_column": "pathway_id",
                    "pathway_family_role": "auxiliary_hierarchy_only",
                    "family_may_replace_target": False,
                },
                "primary_model": {
                    "pair_evidence_in_primary": False,
                    "evidence_confidence_is_separate": True,
                    "subtype_feedback": False,
                },
            }
        ),
        encoding="utf-8",
    )
    task_manifest = write_task_manifest(tmp_path / "TASK_MANIFEST.tsv", build_task_manifest())
    dry = write_dry_run_report(tmp_path / "DRY_RUN_REPORT.json", build_dry_run_report())
    readiness = build_code_only_readiness(
        repo_root=repo,
        config_path=config,
        task_manifest_path=task_manifest,
        dry_run_report_path=dry,
        test_summary={"passed": True, "tests": 1},
    )
    assert readiness["status"] == "IMPLEMENTATION_READY_NOT_TRAINED"
    assert all(readiness["checks"].values())
