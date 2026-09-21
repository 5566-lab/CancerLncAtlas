from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.evidence_direction_independent_audit import (
    AUDIT_STATUS,
    EvidenceDirectionIndependentAuditError,
    audit_evidence_direction_release,
)
from cc_hhgt.v32.evidence_direction_query import (
    EvidenceDirectionProbabilityQuery,
    EvidenceDirectionQueryAssetError,
)
from cc_hhgt.v32.evidence_direction_recovery import (
    EvidenceDirectionRecoveryError,
    recover_evidence_direction_probabilities,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _entropy(probability: tuple[float, float, float]) -> float:
    values = np.asarray(probability, dtype=float)
    return -float(np.sum(values * np.log(values))) / math.log(3.0)


def _build_release(root: Path) -> tuple[Path, str]:
    root.mkdir()
    core = root / "CORE_EMBEDDING_MANIFEST.json"
    _write_json(core, {"format": "TEST_CURRENT_V32_CORE"})
    event = root / "event_lineage.parquet"
    pd.DataFrame({"event_id": ["E0"]}).to_parquet(event, index=False)

    probabilities = [
        (0.70, 0.20, 0.10),
        (0.10, 0.80, 0.10),
        (0.10, 0.20, 0.70),
        (0.60, 0.30, 0.10),
        (0.15, 0.70, 0.15),
    ]
    directions = ["negative", "neutral", "positive", "negative", "neutral"]
    available_rows: list[dict[str, object]] = []
    for fold, (probability, direction) in enumerate(zip(probabilities, directions)):
        available_rows.append(
            {
                "cancer_id": f"C{fold % 2}",
                "lncrna_id": f"ENSG{fold:05d}",
                "pathway_id": f"P{fold}",
                "published_direction": direction,
                "recovered_direction": direction,
                "direction_negative_probability": probability[0],
                "direction_neutral_probability": probability[1],
                "direction_positive_probability": probability[2],
                "direction_entropy": _entropy(probability),
                "direction_probability_available": True,
                "direction_probability_unavailable_reason": None,
                "published_direction_matches_recovered_argmax": True,
                "evidence_confidence_available": True,
                "evidence_fold": fold,
                "event_count": fold + 1,
            }
        )
    unavailable = {
        "cancer_id": "C1",
        "lncrna_id": "ENSG99999",
        "pathway_id": "P9",
        "published_direction": None,
        "recovered_direction": None,
        "direction_negative_probability": None,
        "direction_neutral_probability": None,
        "direction_positive_probability": None,
        "direction_entropy": None,
        "direction_probability_available": False,
        "direction_probability_unavailable_reason": "NO_EXACT_PATHWAY_EVENT",
        "published_direction_matches_recovered_argmax": None,
        "evidence_confidence_available": False,
        "evidence_fold": None,
        "event_count": None,
    }

    original_rows = []
    for row in [*available_rows, unavailable]:
        original_rows.append(
            {
                "cancer_id": row["cancer_id"],
                "lncrna_id": row["lncrna_id"],
                "pathway_id": row["pathway_id"],
                "direction": row["published_direction"],
                "availability": row["evidence_confidence_available"],
                "changes_primary_ranking": False,
                "main_ranking_modified": False,
            }
        )
    original = root / "evidence_private_predictions.parquet"
    pd.DataFrame(original_rows).to_parquet(original, index=False)

    checkpoints: dict[str, dict[str, object]] = {}
    manifest_folds: dict[str, dict[str, object]] = {}
    for fold in range(5):
        state = {"private.weight": torch.tensor([fold + 0.25], dtype=torch.float32)}
        final_sha = _state_sha(state)
        checkpoint = root / f"private_eventset_state_fold={fold}.pt"
        torch.save(
            {
                "checkpoint_format": "CC_HHGT_V3_2_PRIVATE_EVIDENCE_EVENTSET_V1",
                "patient_fold": fold,
                "private_parameters_fresh_init": True,
                "initialized_from_checkpoint": False,
                "historical_evidence_checkpoint_allowed": False,
                "historical_evidence_result_allowed": False,
                "contains_core_parameters": False,
                "contains_primary_ranking_parameters": False,
                "direction_supervision_available": True,
                "core_manifest_sha256": _sha(core),
                "private_model_state": state,
                "initial_parameter_sha256": "0" * 64,
                "final_parameter_sha256": final_sha,
                "optimizer_steps": fold + 1,
            },
            checkpoint,
        )
        record = {
            "path": str(checkpoint),
            "sha256": _sha(checkpoint),
            "final_parameter_sha256": final_sha,
        }
        checkpoints[str(fold)] = record
        manifest_folds[str(fold)] = {
            "checkpoint_sha256": record["sha256"],
            "n_heldout_bags": 1,
        }

    training_manifest = root / "TRAINING_MANIFEST.json"
    _write_json(
        training_manifest,
        {
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "module_id": "evidence_private_eventset",
            "status": "SUCCESS_NEWLY_TRAINED",
            "training_status": "SUCCESS_NEWLY_TRAINED",
            "training_generation": "V3.2",
            "checkpoint_format": "CC_HHGT_V3_2_PRIVATE_EVIDENCE_EVENTSET_V1",
            "historical_evidence_checkpoint_loaded": False,
            "historical_evidence_result_loaded": False,
            "old_confidence_loaded": False,
            "family_to_exact_broadcast_used": False,
            "core_frozen": True,
            "core_detached": True,
            "main_ranking_modified": False,
            "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
            "folds": manifest_folds,
        },
    )
    evidence_binding = root / "EVIDENCE_OUTPUT_BINDING.json"
    _write_json(
        evidence_binding,
        {
            "format": "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1",
            "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "five_fresh_private_heads_verified": True,
            "historical_checkpoints_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "family_to_exact_broadcast": False,
            "production_deployed": False,
            "release_ready": False,
            "artifacts": {
                "evidence_predictions": {"path": str(original), "sha256": _sha(original)},
                "event_lineage": {"path": str(event), "sha256": _sha(event)},
            },
            "authorities": {
                "training_manifest": {
                    "path": str(training_manifest),
                    "sha256": _sha(training_manifest),
                }
            },
            "checkpoints": checkpoints,
        },
    )

    parts = root / "available_parts"
    parts.mkdir()
    partition_records = []
    for fold, row in enumerate(available_rows):
        part = parts / f"evidence_fold={fold}.parquet"
        columns = [
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "direction_negative_probability",
            "direction_neutral_probability",
            "direction_positive_probability",
            "recovered_direction",
            "direction_entropy",
            "evidence_fold",
            "event_count",
        ]
        pd.DataFrame([{name: row[name] for name in columns}]).to_parquet(part, index=False)
        partition_records.append(
            {
                "fold": fold,
                "path": str(part),
                "sha256": _sha(part),
                "rows": 1,
                "checkpoint_path": checkpoints[str(fold)]["path"],
                "checkpoint_sha256": checkpoints[str(fold)]["sha256"],
                "final_parameter_sha256": checkpoints[str(fold)]["final_parameter_sha256"],
            }
        )

    public_rows = []
    for row in [*available_rows, unavailable]:
        public_rows.append(
            {
                **row,
                "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "output_format": "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITIES_V1",
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
                "used_for_fusion": False,
                "historical_checkpoint_used": False,
                "historical_prediction_used": False,
            }
        )
    artifact = root / "evidence_direction_probabilities.parquet"
    pd.DataFrame(public_rows).to_parquet(artifact, index=False)
    release = root / "EVIDENCE_DIRECTION_PROBABILITY_BINDING.json"
    _write_json(
        release,
        {
            "format": "CC_HHGT_V3_2_EVIDENCE_DIRECTION_PROBABILITY_BINDING_V1",
            "status": "SUCCESS_V32_CHECKPOINT_REINFERENCE",
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "training_performed": False,
            "source_checkpoints_newly_trained_v32": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "historical_rankings_used": False,
            "primary_ranking_unchanged": True,
            "discovery_ranking_unchanged": True,
            "used_for_fusion": False,
            "published_argmax_is_original_inference": True,
            "recovered_argmax_is_fixed_seed_reinference": True,
            "argmax_mismatch_is_not_checkpoint_drift": True,
            "stochastic_reinference_policy": (
                "FIXED_SEED_MC_DROPOUT_FROM_HASH_PINNED_V32_PRIVATE_HEADS"
            ),
            "production_deployed": False,
            "release_ready": False,
            "parameters": {
                "batch_size": 4,
                "mc_samples": 16,
                "dropout": 0.2,
                "inference_seed": 20270826,
                "max_events": 64,
            },
            "authority": {
                "evidence_binding_path": str(evidence_binding),
                "evidence_binding_sha256": _sha(evidence_binding),
                "core_manifest_path": str(core),
                "core_manifest_sha256": _sha(core),
                "event_lineage_path": str(event),
                "event_lineage_sha256": _sha(event),
                "original_predictions_path": str(original),
                "original_predictions_sha256": _sha(original),
            },
            "partitions": partition_records,
            "artifact": {"path": str(artifact), "sha256": _sha(artifact), "rows": 6},
            "counts": {
                "total_rows": 6,
                "available_rows": 5,
                "unavailable_rows": 1,
                "cancers": 2,
                "probability_sum_violations": 0,
                "null_policy_violations": 0,
                "published_recovered_argmax_mismatch_rows": 0,
                "published_recovered_argmax_match_rows": 5,
            },
        },
    )
    return release, _sha(release)


def test_independent_audit_and_hash_pinned_query(tmp_path: Path) -> None:
    release, release_sha = _build_release(tmp_path / "release")
    audit_root = tmp_path / "audit"
    success = audit_evidence_direction_release(
        binding_path=release,
        expected_binding_sha256=release_sha,
        output_root=audit_root,
        expected_total_rows=6,
        expected_available_rows=5,
        expected_cancer_count=2,
    )
    assert success["status"] == AUDIT_STATUS
    query = EvidenceDirectionProbabilityQuery(
        release,
        expected_binding_sha256=release_sha,
        audit_binding_path=success["audit_binding_path"],
        expected_audit_binding_sha256=success["audit_binding_sha256"],
    )
    available = query.query(available=True, limit=5)
    assert available["returned_rows"] == 5
    assert all(
        abs(
            row["direction_negative_probability"]
            + row["direction_neutral_probability"]
            + row["direction_positive_probability"]
            - 1.0
        )
        < 1e-6
        for row in available["rows"]
    )
    typed_null = query.query(available=False, limit=1)["rows"][0]
    assert typed_null["direction_negative_probability"] is None
    assert typed_null["direction_probability_unavailable_reason"]
    assert query.capability()["used_for_fusion"] is False


def test_query_fails_closed_after_artifact_drift(tmp_path: Path) -> None:
    release, release_sha = _build_release(tmp_path / "release")
    success = audit_evidence_direction_release(
        binding_path=release,
        expected_binding_sha256=release_sha,
        output_root=tmp_path / "audit",
        expected_total_rows=6,
        expected_available_rows=5,
        expected_cancer_count=2,
    )
    query = EvidenceDirectionProbabilityQuery(
        release,
        expected_binding_sha256=release_sha,
        audit_binding_path=success["audit_binding_path"],
        expected_audit_binding_sha256=success["audit_binding_sha256"],
    )
    artifact = Path(query.artifact_path)
    artifact.write_bytes(artifact.read_bytes() + b"DRIFT")
    with pytest.raises(EvidenceDirectionQueryAssetError, match="drifted"):
        query.query(limit=1)


def test_audit_rejects_old_checkpoint_claim(tmp_path: Path) -> None:
    release, _ = _build_release(tmp_path / "release")
    payload = json.loads(release.read_text(encoding="utf-8"))
    payload["old_checkpoint_loaded"] = True
    _write_json(release, payload)
    with pytest.raises(EvidenceDirectionIndependentAuditError, match="old_checkpoint_loaded"):
        audit_evidence_direction_release(
            binding_path=release,
            expected_binding_sha256=_sha(release),
            output_root=tmp_path / "audit",
            expected_total_rows=6,
            expected_available_rows=5,
            expected_cancer_count=2,
        )


def test_recovery_rejects_nonformal_dropout_before_io(tmp_path: Path) -> None:
    with pytest.raises(EvidenceDirectionRecoveryError, match="dropout value 0.20"):
        recover_evidence_direction_probabilities(
            evidence_binding_path=tmp_path / "missing.json",
            expected_evidence_binding_sha256="0" * 64,
            core_embedding_root=tmp_path / "core",
            output_root=tmp_path / "output",
            dropout=0.1,
        )
