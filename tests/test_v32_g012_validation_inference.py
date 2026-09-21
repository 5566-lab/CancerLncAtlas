from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from cc_hhgt.v32.multimodal_fusion import artifact_sha256
from cc_hhgt.v32.prepared import PREPARED_FORMAT
from cc_hhgt.v32.sealed_test_inference import WINNER_DECLARATION_FORMAT
from cc_hhgt.v32.validation_prediction_inference import (
    ValidationPredictionInferenceError,
    materialize_g012_validation_predictions,
)


class _Encoder:
    def encode(self, graph):
        return graph


class _CoreModel:
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
        return {"final_logit": base_logit + 0.4}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import cc_hhgt.gnn
    import cc_hhgt.v32.model
    import cc_hhgt.v32.multimodal_fusion

    monkeypatch.setattr(cc_hhgt.gnn, "build_model", lambda *args, **kwargs: _Encoder())
    monkeypatch.setattr(
        cc_hhgt.v32.model,
        "build_v32_cc_hhgt_residual",
        lambda *args, **kwargs: _CoreModel(),
    )
    monkeypatch.setattr(
        cc_hhgt.v32.multimodal_fusion,
        "pair_blocked_fold",
        lambda lnc, pathway, seed=20260726: int(str(lnc)[1:]),
    )
    prepared = tmp_path / "G2"
    checkpoints = tmp_path / "checkpoints"
    prepared.mkdir()
    checkpoints.mkdir()
    node_maps = {
        "cancer": {"BRCA": 0},
        "lncRNA": {f"L{fold}": fold for fold in range(5)},
        "pathway": {f"P{fold}": fold for fold in range(5)},
    }
    prepared_records = []
    checkpoint_records = []
    for outer_fold in range(5):
        validation_fold = (outer_fold + 1) % 5
        batch = {
            "candidate_batch": {
                "l": torch.tensor([validation_fold]),
                "p": torch.tensor([validation_fold]),
                "c": torch.tensor([0]),
            },
            "base_logit": torch.tensor([0.0]),
            "conservation_context": torch.zeros((1, 4)),
            "graph_available": torch.tensor([True]),
            "proxy_label": torch.tensor([float(validation_fold % 2)]),
            "weak_positive": torch.tensor([False]),
        }
        payload = {
            "prepared_format": PREPARED_FORMAT,
            "patient_fold": outer_fold,
            "formal_graph_variant": "G2",
            "bundle": SimpleNamespace(node_maps=node_maps),
            "feature_dim": 4,
            "legacy_model_config": {},
            "conservation_context_features": 4,
            "graph": torch.tensor([1.0]),
            "train_batches": [batch],
            "validation_batches": [batch],
        }
        prepared_path = prepared / f"PATIENT_FOLD_{outer_fold}.pt"
        torch.save(payload, prepared_path)
        checkpoint_path = checkpoints / f"fold_{outer_fold}.pt"
        torch.save(
            {
                "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
                "model_state": {},
            },
            checkpoint_path,
        )
        prepared_records.append(
            {
                "patient_fold": outer_fold,
                "path": str(prepared_path),
                "sha256": artifact_sha256(prepared_path),
            }
        )
        checkpoint_records.append(
            {
                "patient_fold": outer_fold,
                "path": str(checkpoint_path),
                "sha256": artifact_sha256(checkpoint_path),
            }
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "primary_model": {
                    "hidden_channels": 8,
                    "dropout": 0.0,
                    "evidence_integration": {"mode": "external_router"},
                }
            }
        ),
        encoding="utf-8",
    )
    winner_path = tmp_path / "G012_VALIDATION_WINNER_LOCK.json"
    winner_path.write_text(
        json.dumps(
            {
                "format": WINNER_DECLARATION_FORMAT,
                "status": "PASS_VALIDATION_ONLY_WINNER_LOCK",
                "selection_scope": "VALIDATION_ONLY",
                "heldout_test_metrics_used_for_selection": False,
                "test_inputs_opened_before_winner_lock": False,
                "winner_or_checkpoint_changed_by_test": False,
                "declaration_locked": True,
                "graph_variant": "G2",
                "pair_fold_seed": 20260726,
                "config": {
                    "path": str(config_path),
                    "sha256": artifact_sha256(config_path),
                },
                "prepared_folds": prepared_records,
                "checkpoints": checkpoint_records,
            }
        ),
        encoding="utf-8",
    )
    return prepared, winner_path


def test_g012_validation_inference_never_reads_test_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, winner = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "validation"
    result = materialize_g012_validation_predictions(
        repo_root=tmp_path,
        prepared_root=prepared,
        winner_declaration_path=winner,
        winner_declaration_sha256=artifact_sha256(winner),
        output_root=output,
    )
    assert result["status"] == "PASS_G012_INNER_VALIDATION_PREDICTIONS"
    assert result["outer_test_predictions_written"] is False
    frame = pd.read_parquet(output / "G012_VALIDATION_PREDICTIONS.PRIVATE.parquet")
    assert len(frame) == 5
    assert frame.outer_test_queried.eq(False).all()
    assert set(frame.fusion_pair_fold) == set(range(5))

    payload_path = prepared / "PATIENT_FOLD_0.pt"
    payload = torch.load(payload_path, weights_only=False)
    payload["test_batches"] = payload["validation_batches"]
    torch.save(payload, payload_path)
    # The winner hash binding catches the payload mutation before any test row
    # can be deserialized by validation inference.
    with pytest.raises(ValidationPredictionInferenceError):
        materialize_g012_validation_predictions(
            repo_root=tmp_path,
            prepared_root=prepared,
            winner_declaration_path=winner,
            winner_declaration_sha256=artifact_sha256(winner),
            output_root=tmp_path / "invalid",
        )
