from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.multimodal_fusion import FUSION_FOLD_COLUMN, artifact_sha256
from cc_hhgt.v32.routed_fair_comparison import exact_candidate_and_fold_hashes
from cc_hhgt.v32.routing_validation_winner import (
    CONTRACT_FORMAT,
    RoutingValidationWinnerError,
    lock_routing_validation_winner,
    materialize_primary_outer_after_winner_lock,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cancer in ("BRCA", "COAD"):
        for fold in range(5):
            for index in range(4):
                target = float(index % 2 == 0)
                rows.append(
                    {
                        "cancer_id": cancer,
                        "lncrna_id": f"L{fold}_{index}",
                        "pathway_id": f"P{fold}_{index}",
                        FUSION_FOLD_COLUMN: fold,
                        "selection_outer_pair_fold": (fold - 1) % 5,
                        "selection_validation_pair_fold": fold,
                        "outer_test_queried": False,
                        "fusion_target": target,
                        "primary_probability": 0.5,
                        "mutation_probability": 0.85 if target else 0.15,
                        "mutation_available": True,
                        "cnv_probability": np.nan,
                        "cnv_available": False,
                        "atac_probability": np.nan,
                        "atac_available": False,
                    }
                )
    external = pd.DataFrame(rows)
    external["discovery_adjusted_probability"] = np.where(
        external.fusion_target.eq(1), 0.85, 0.15
    )
    hierarchical = external.drop(columns="discovery_adjusted_probability").copy()
    hierarchical["hierarchical_probability"] = np.where(
        hierarchical.fusion_target.eq(1), 0.70, 0.30
    )
    return external, hierarchical


def _write_fixture(tmp_path: Path, external: pd.DataFrame, hierarchical: pd.DataFrame):
    tmp_path.mkdir(parents=True, exist_ok=True)
    external_path = tmp_path / "external.parquet"
    hierarchical_path = tmp_path / "hierarchical.parquet"
    external.to_parquet(external_path, index=False)
    hierarchical.to_parquet(hierarchical_path, index=False)
    candidate_sha, fold_sha = exact_candidate_and_fold_hashes(external)
    contract = {
        "format": CONTRACT_FORMAT,
        "candidate_universe_sha256": candidate_sha,
        "validation_pair_fold_sha256": fold_sha,
        "seed": 20260726,
        "training_budget_id": "equal-budget-test",
        "selection_split": "INNER_VALIDATION_ONLY",
        "outer_test_queries": 0,
    }
    external_contract = tmp_path / "external-contract.json"
    hierarchical_contract = tmp_path / "hierarchical-contract.json"
    external_contract.write_text(json.dumps(contract), encoding="utf-8")
    hierarchical_contract.write_text(json.dumps(contract), encoding="utf-8")
    external_success = tmp_path / "external-success.json"
    external_success.write_text(
        json.dumps(
            {
                "status": "PASS_VALIDATION_ONLY_EXTERNAL_ROUTER",
                "validation_predictions_sha256": artifact_sha256(external_path),
                "comparison_contract_sha256": artifact_sha256(external_contract),
                "outer_test_predictions_written": False,
                "outer_test_metrics_computed": False,
                "architecture_winner_selected": False,
            }
        ),
        encoding="utf-8",
    )
    hierarchical_success = tmp_path / "hierarchical-success.json"
    hierarchical_success.write_text(
        json.dumps(
            {
                "status": "PASS_VALIDATION_ONLY_HIERARCHICAL",
                "validation_predictions": {
                    "sha256": artifact_sha256(hierarchical_path)
                },
                "validation_comparison_contract": {
                    "sha256": artifact_sha256(hierarchical_contract)
                },
                "outer_test_predictions_written": False,
                "outer_test_metrics_computed": False,
                "architecture_winner_selected": False,
            }
        ),
        encoding="utf-8",
    )
    return {
        "external_predictions_path": external_path,
        "hierarchical_predictions_path": hierarchical_path,
        "external_contract_path": external_contract,
        "hierarchical_contract_path": hierarchical_contract,
        "external_success_path": external_success,
        "hierarchical_success_path": hierarchical_success,
    }


def test_validation_lock_selects_best_eligible_architecture(tmp_path: Path) -> None:
    external, hierarchical = _frames()
    inputs = _write_fixture(tmp_path, external, hierarchical)
    output = tmp_path / "winner"
    success = lock_routing_validation_winner(
        **inputs,
        output_root=output,
        run_id="v32-routing-validation-test",
        minimum_logloss_improvement=1e-4,
    )
    assert success["winner_id"] == "external_router"
    assert success["heldout_test_used_for_selection"] is False
    declaration = json.loads(
        (output / "ROUTING_VALIDATION_WINNER_LOCK.json").read_text(encoding="utf-8")
    )
    assert declaration["selection_split"] == "validation_only"
    assert declaration["outer_test_predictions_available_during_selection"] is False
    assert declaration["compared_models"] == [
        "current_primary",
        "external_router",
        "hierarchical_end_to_end",
    ]


def test_validation_lock_keeps_primary_when_both_candidates_regress(tmp_path: Path) -> None:
    external, hierarchical = _frames()
    external["discovery_adjusted_probability"] = 1.0 - external.fusion_target
    hierarchical["hierarchical_probability"] = 1.0 - hierarchical.fusion_target
    inputs = _write_fixture(tmp_path, external, hierarchical)
    winner_root = tmp_path / "winner"
    success = lock_routing_validation_winner(
        **inputs,
        output_root=winner_root,
        run_id="v32-routing-validation-primary-fallback-test",
    )
    assert success["winner_id"] == "current_primary"
    primary_path = tmp_path / "primary.parquet"
    external.to_parquet(primary_path, index=False)
    winner = winner_root / "ROUTING_VALIDATION_WINNER_LOCK.json"
    outer_root = tmp_path / "primary-outer"
    outer = materialize_primary_outer_after_winner_lock(
        primary_frame_path=primary_path,
        external_validation_predictions_path=inputs["external_predictions_path"],
        routing_winner_declaration_path=winner,
        routing_winner_declaration_sha256=artifact_sha256(winner),
        output_root=outer_root,
        run_id="v32-primary-outer-test",
    )
    assert outer["status"] == "PASS_PRIMARY_OUTER_AFTER_WINNER_LOCK"
    predictions = pd.read_parquet(
        outer_root / "primary_winner_locked_oof.PRIVATE.parquet"
    )
    assert predictions.winner_probability.eq(predictions.primary_probability).all()


def test_validation_lock_rejects_outer_query_or_unbound_contract(tmp_path: Path) -> None:
    external, hierarchical = _frames()
    external.loc[0, "outer_test_queried"] = True
    inputs = _write_fixture(tmp_path, external, hierarchical)
    with pytest.raises(RoutingValidationWinnerError, match="queried an outer test"):
        lock_routing_validation_winner(
            **inputs,
            output_root=tmp_path / "winner-outer",
            run_id="v32-routing-validation-outer-reject-test",
        )

    external, hierarchical = _frames()
    inputs = _write_fixture(tmp_path / "contract", external, hierarchical)
    contract_path = inputs["external_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["candidate_universe_sha256"] = "f" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    success_path = inputs["external_success_path"]
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["comparison_contract_sha256"] = artifact_sha256(contract_path)
    success_path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(RoutingValidationWinnerError, match="share one validation"):
        lock_routing_validation_winner(
            **inputs,
            output_root=tmp_path / "winner-contract",
            run_id="v32-routing-validation-contract-reject-test",
        )
