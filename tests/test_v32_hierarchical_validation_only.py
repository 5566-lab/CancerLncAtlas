from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from cc_hhgt.v32.hierarchical_candidate_inference import (
    HierarchicalInferenceError,
    materialize_hierarchical_outer_after_winner_lock,
    materialize_hierarchical_validation_only,
)
from cc_hhgt.v32.hierarchical_candidate_preparation import (
    prepare_hierarchical_outer_after_winner_lock,
)
from cc_hhgt.v32.hierarchical_modality_ablation import (
    materialize_hierarchical_modality_ablation_after_winner_lock,
)
from cc_hhgt.v32.multimodal_fusion import artifact_sha256
from cc_hhgt.v32.routing_validation_winner import WINNER_FORMAT


class _Encoder:
    def encode(self, graph):
        return graph


class _Model:
    def __init__(self):
        self.encoder = _Encoder()

    def load_state_dict(self, state):
        assert state == {}

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(
        self,
        graph,
        candidate,
        base_logit,
        conservation_context,
        graph_available,
        *,
        admitted,
        encoded,
    ):
        assert admitted is True
        return {"final_logit": base_logit + 0.25}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import cc_hhgt.gnn
    import cc_hhgt.v32.model

    monkeypatch.setattr(cc_hhgt.gnn, "build_model", lambda *args, **kwargs: _Encoder())
    monkeypatch.setattr(cc_hhgt.gnn, "move_graph", lambda *args, **kwargs: torch.tensor([1.0]))
    monkeypatch.setattr(
        cc_hhgt.v32.model,
        "build_v32_hierarchical_evidence_hhgt",
        lambda *args, **kwargs: _Model(),
    )
    prepared = tmp_path / "prepared"
    checkpoints = tmp_path / "checkpoints"
    prepared.mkdir()
    checkpoints.mkdir()
    node_maps = {
        "cancer": {"BRCA": 0},
        "lncRNA": {f"L{fold}": fold for fold in range(5)},
        "pathway": {f"P{fold}": fold for fold in range(5)},
    }
    for fold in range(5):
        validation_fold = (fold + 1) % 5
        candidate = {
            "l": torch.tensor([fold]),
            "p": torch.tensor([fold]),
            "c": torch.tensor([0]),
            "outer_pair_fold": torch.tensor([validation_fold], dtype=torch.int16),
            "modality_probability": torch.tensor([[0.8, float("nan"), 0.6]]),
            "modality_available": torch.tensor([[True, False, True]]),
        }
        payload = {
            "bundle": SimpleNamespace(node_maps=node_maps),
            "feature_dim": 4,
            "legacy_model_config": {},
            "conservation_context_features": 4,
            "graph": torch.tensor([1.0]),
            "validation_batches": [
                {
                    "candidate_batch": candidate,
                    "base_logit": torch.tensor([0.0]),
                    "conservation_context": torch.zeros((1, 4)),
                    "graph_available": torch.tensor([True]),
                    "proxy_label": torch.tensor([float(fold % 2)]),
                }
            ],
        }
        torch.save(payload, prepared / f"PATIENT_FOLD_{fold}.pt")
        torch.save(
            {
                "architecture_id": "HHGT_HIERARCHICAL_EVIDENCE_GATE_CANDIDATE",
                "model_state": {},
            },
            checkpoints / f"fold_{fold}.pt",
        )
    primary_validation = tmp_path / "g012-validation.parquet"
    pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": f"L{fold}",
                "pathway_id": f"P{fold}",
                "fusion_pair_fold": (fold + 1) % 5,
                "selection_outer_pair_fold": fold,
                "selection_validation_pair_fold": (fold + 1) % 5,
                "outer_test_queried": False,
                "fusion_target": float(fold % 2),
                "primary_probability": 0.55,
            }
            for fold in range(5)
        ]
    ).sort_values(["cancer_id", "lncrna_id", "pathway_id"]).to_parquet(
        primary_validation, index=False
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "primary_model": {
                    "hidden_channels": 8,
                    "dropout": 0.0,
                    "evidence_integration": {
                        "modalities": ["mutation", "cnv", "atac"]
                    },
                },
                "crossfit": {"seed": 20260726},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "PREPARATION_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "SUCCESS_TRAIN_VALIDATION_ONLY_TEST_REMAINS_SEALED",
                "patient_fold_authority": {"status": "PASS"},
                "graph_variant": "G2",
                "test_labels_absent_from_training_payload": True,
                "post_winner_lock_sealed_test_materializer_required": True,
            }
        ),
        encoding="utf-8",
    )
    return prepared, checkpoints, config, manifest, primary_validation


def test_hierarchical_validation_only_never_deserializes_test_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, checkpoints, config, manifest, primary_validation = _fixture(
        tmp_path, monkeypatch
    )
    output = tmp_path / "validation"
    result = materialize_hierarchical_validation_only(
        repo_root=tmp_path,
        config_path=config,
        prepared_root=prepared,
        checkpoint_pattern=str(checkpoints / "fold_{fold}.pt"),
        preparation_manifest_path=manifest,
        primary_validation_predictions_path=primary_validation,
        output_root=output,
        training_budget_id="equal-budget-test",
    )
    assert result["status"] == "PASS_VALIDATION_ONLY_HIERARCHICAL"
    assert result["outer_test_predictions_written"] is False
    assert result["outer_test_metrics_computed"] is False
    frame = pd.read_parquet(output / "hierarchical_validation_predictions.PRIVATE.parquet")
    assert len(frame) == 5
    assert frame.outer_test_queried.eq(False).all()
    assert set(frame.fusion_pair_fold) == set(range(5))
    assert not (output / "hierarchical_oof_predictions.parquet").exists()

    payload_path = prepared / "PATIENT_FOLD_0.pt"
    payload = torch.load(payload_path, weights_only=False)
    payload["test_batches"] = payload["validation_batches"]
    torch.save(payload, payload_path)
    with pytest.raises(HierarchicalInferenceError, match="forbidden test batches"):
        materialize_hierarchical_validation_only(
            repo_root=tmp_path,
            config_path=config,
            prepared_root=prepared,
            checkpoint_pattern=str(checkpoints / "fold_{fold}.pt"),
            preparation_manifest_path=manifest,
            primary_validation_predictions_path=primary_validation,
            output_root=tmp_path / "invalid",
            training_budget_id="equal-budget-test",
        )


def test_hierarchical_outer_payload_and_inference_require_routing_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, checkpoints, config, prep_manifest, _ = _fixture(tmp_path, monkeypatch)
    import cc_hhgt.v32.cancer_modality_router as router
    import cc_hhgt.v32.hierarchical_candidate_preparation as preparation

    monkeypatch.setattr(router, "validate_patient_oof_lineage", lambda *args, **kwargs: None)
    monkeypatch.setattr(preparation, "pair_blocked_fold", lambda lnc, pathway: int(str(lnc)[1:]))
    genomic = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 5,
            "lncrna_id": [f"L{fold}" for fold in range(5)],
            "pathway_id": [f"P{fold}" for fold in range(5)],
            "mutation_context_probability": [0.8] * 5,
            "mutation_available": [True] * 5,
            "cnv_context_probability": [float("nan")] * 5,
            "cnv_available": [False] * 5,
        }
    )
    genomic_path = tmp_path / "genomic.parquet"
    genomic.to_parquet(genomic_path, index=False)
    genomic_lineage = tmp_path / "genomic-lineage.json"
    genomic_lineage.write_text("{}", encoding="utf-8")

    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    sealed_records = []
    for fold in range(5):
        training_payload = torch.load(
            prepared / f"PATIENT_FOLD_{fold}.pt", weights_only=False
        )
        raw = training_payload["validation_batches"][0]
        sealed_payload = {
            "patient_fold": fold,
            "contains_test_labels": True,
            "contains_train_or_validation_batches": False,
            "test_batches": [raw],
        }
        path = sealed_root / f"SEALED_TEST_FOLD_{fold}.pt"
        torch.save(sealed_payload, path)
        sealed_records.append(
            {
                "patient_fold": fold,
                "test_payload": {"path": str(path), "sha256": artifact_sha256(path)},
            }
        )
    sealed_manifest = sealed_root / "SEALED_TEST_MANIFEST.json"
    sealed_manifest.write_text(
        json.dumps(
            {
                "status": "PASS_SEALED_TEST_AUTHORITY_UNOPENED",
                "payloads_opened_before_winner_lock": False,
                "test_labels_absent_from_training_payload": True,
                "folds": sealed_records,
            }
        ),
        encoding="utf-8",
    )
    hierarchical_success = tmp_path / "hierarchical-success.json"
    hierarchical_success.write_text(
        json.dumps(
            {
                "status": "PASS_VALIDATION_ONLY_HIERARCHICAL",
                "outer_test_predictions_written": False,
                "outer_test_metrics_computed": False,
                "architecture_winner_selected": False,
            }
        ),
        encoding="utf-8",
    )
    winner = tmp_path / "ROUTING_VALIDATION_WINNER_LOCK.json"
    winner.write_text(
        json.dumps(
            {
                "format": WINNER_FORMAT,
                "status": "PASS_ROUTING_VALIDATION_WINNER_LOCK",
                "winner_id": "hierarchical_end_to_end",
                "selection_split": "validation_only",
                "heldout_test_used_for_selection": False,
                "outer_test_predictions_available_during_selection": False,
                "test_metrics_used_for_selection": False,
                "winner_locked_before_outer_test_inference": True,
                "sources": {
                    "hierarchical_success": {
                        "sha256": artifact_sha256(hierarchical_success)
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    winner_sha = artifact_sha256(winner)
    outer_prepared = tmp_path / "outer-prepared"
    manifest = prepare_hierarchical_outer_after_winner_lock(
        hierarchical_prepared_root=prepared,
        hierarchical_preparation_manifest_path=prep_manifest,
        sealed_test_manifest_path=sealed_manifest,
        genomic_predictions_path=genomic_path,
        genomic_lineage_path=genomic_lineage,
        routing_winner_declaration_path=winner,
        routing_winner_declaration_sha256=winner_sha,
        hierarchical_validation_success_path=hierarchical_success,
        output_root=outer_prepared,
    )
    assert manifest["status"] == "PASS_HIERARCHICAL_OUTER_PREPARED_AFTER_WINNER_LOCK"
    for fold in range(5):
        payload = torch.load(
            outer_prepared / f"PATIENT_FOLD_{fold}.pt", weights_only=False
        )
        assert "train_batches" not in payload and "validation_batches" not in payload
        assert payload["outer_inference_authorized_after_winner_lock"] is True
        observed = payload["test_batches"][0]["candidate_batch"][
            "outer_pair_fold"
        ].tolist()
        assert observed == [fold]

    output = tmp_path / "outer-inference"
    primary_outer = tmp_path / "g012-primary-outer.parquet"
    pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": f"L{fold}",
                "pathway_id": f"P{fold}",
                "fusion_pair_fold": fold,
                "fusion_target": float(fold % 2),
                "primary_probability": 0.50,
            }
            for fold in range(5)
        ]
    ).to_parquet(primary_outer, index=False)
    result = materialize_hierarchical_outer_after_winner_lock(
        repo_root=tmp_path,
        config_path=config,
        prepared_root=outer_prepared,
        checkpoint_pattern=str(checkpoints / "fold_{fold}.pt"),
        winner_locked_test_manifest_path=(
            outer_prepared / "WINNER_LOCKED_TEST_MANIFEST.json"
        ),
        routing_winner_declaration_path=winner,
        routing_winner_declaration_sha256=winner_sha,
        primary_outer_predictions_path=primary_outer,
        output_root=output,
        training_budget_id="equal-budget-test",
    )
    assert result["status"] == "PASS_HIERARCHICAL_OUTER_AFTER_WINNER_LOCK"
    assert result["outer_inference_started_after_winner_lock"] is True
    frame = pd.read_parquet(output / "hierarchical_oof_predictions.parquet")
    assert len(frame) == 5
    assert frame.primary_probability.eq(0.50).all()
    assert (output / "hierarchical_outer_metrics.parquet").is_file()
    metrics = pd.read_parquet(output / "hierarchical_outer_metrics.parquet")
    assert metrics.outer_metric_computed_after_winner_lock.eq(True).all()
    ablation_root = tmp_path / "hierarchical-ablation"
    ablation_success = materialize_hierarchical_modality_ablation_after_winner_lock(
        repo_root=tmp_path,
        config_path=config,
        prepared_root=outer_prepared,
        checkpoint_pattern=str(checkpoints / "fold_{fold}.pt"),
        outer_predictions_path=output / "hierarchical_oof_predictions.parquet",
        outer_success_path=output / "SUCCESS.json",
        routing_winner_declaration_path=winner,
        routing_winner_declaration_sha256=winner_sha,
        output_root=ablation_root,
    )
    assert (
        ablation_success["status"]
        == "PASS_HIERARCHICAL_MODALITY_ABLATION_AFTER_WINNER_LOCK"
    )
    ablation = pd.read_parquet(
        ablation_root / "hierarchical_modality_ablation_metrics.parquet"
    )
    assert set(ablation.modality) == {"mutation", "cnv", "atac"}
    assert ablation.outer_metric_computed_after_winner_lock.eq(True).all()
    assert ablation.used_for_model_or_architecture_selection.eq(False).all()
