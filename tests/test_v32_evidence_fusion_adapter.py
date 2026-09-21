from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.evidence_fusion_adapter import (
    BINDING_FORMAT,
    EvidenceFusionAdapterError,
    artifact_sha256,
    materialize_evidence_fusion_adapter,
)
from cc_hhgt.v32.evidence_output_binding import ANALYSIS_VERSION
from cc_hhgt.v32.evidence_semantic_wrapper import (
    SEMANTIC_WRAPPER_FORMAT,
    SEMANTIC_WRAPPER_STATUS,
)


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, duplicate_normalized_raw: bool = False) -> tuple[Path, Path]:
    candidates = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "cancer_id": ["ACC", "ACC"],
            "lncrna_id": ["LNC:ENSG1", "LNC:ENSG2"],
            "pathway_id": ["PROGENY:NFKB", "DOROTHEA:FOXM1"],
        }
    ).to_parquet(candidates, index=False)
    raw = tmp_path / "raw.parquet"
    raw_lnc = ["ENSG1", "ENSG2"]
    raw_pathway = ["PROGENY:NFkB", "DOROTHEA:FOXM1"]
    if duplicate_normalized_raw:
        raw_lnc = ["ENSG1", "LNC:ENSG1"]
        raw_pathway = ["PROGENY:NFkB", "PROGENY:NFKB"]
    pd.DataFrame(
        {
            "cancer_id": ["ACC", "ACC"],
            "lncrna_id": raw_lnc,
            "pathway_id": raw_pathway,
            "evidence_confidence_probability": [0.8, None],
            "availability": [True, False],
            "direction": ["positive", None],
            "uncertainty": [0.1, None],
            "unavailable_reason": ["", "NO_EXACT_PATHWAY_EVENT"],
            "event_count": [3, 0],
            "evidence_fold": [1, None],
            "failure_reason": ["", "NO_EXACT_PATHWAY_EVENT"],
            "analysis_version": [ANALYSIS_VERSION, ANALYSIS_VERSION],
            "training_run_id": ["V32-EVIDENCE-TRAIN-test"] * 2,
            "changes_primary_ranking": [False, False],
            "main_ranking_modified": [False, False],
        }
    ).to_parquet(raw, index=False)
    r2 = tmp_path / "R2_BINDING.json"
    _json(
        r2,
        {
            "format": "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND",
            "release_ready": False,
            "production_deployed": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "family_to_exact_broadcast": False,
            "five_fresh_private_heads_verified": True,
            "optimizer_steps_total": 5,
            "artifacts": {
                "evidence_predictions": {
                    "path": str(raw),
                    "sha256": artifact_sha256(raw),
                    "rows": 2,
                }
            },
        },
    )
    post = tmp_path / "post.json"
    _json(post, {"status": "PASS"})
    wrapper = tmp_path / "wrapper.json"
    _json(
        wrapper,
        {
            "format": SEMANTIC_WRAPPER_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": SEMANTIC_WRAPPER_STATUS,
            "release_ready": False,
            "production_deployed": False,
            "evidence_training_run_id": "V32-EVIDENCE-TRAIN-test",
            "direct_target_evidence": True,
            "direct_exact_pathway_assertion": False,
            "unavailable_encoding": "null_with_reason",
            "confidence_only": True,
            "changes_primary_ranking": False,
            "affects_discovery": False,
            "affects_primary_ranking": False,
            "raw_prediction_direct_fusion_allowed": False,
            "canonical_candidate_adapter_required": True,
            "lncrna_identifier_normalization": (
                "LNC:ENSG_TO_ENSG_FOR_TRAINING__RESTORE_LNC_PREFIX_FOR_FUSION"
            ),
            "pathway_identifier_normalization": (
                "TRIM_AND_UPPERCASE_TO_CANONICAL_PATHWAY_FOR_FUSION"
            ),
            "five_fresh_private_heads_verified": True,
            "r2_evidence_binding": {
                "path": str(r2),
                "sha256": artifact_sha256(r2),
            },
            "independent_post_audit": {
                "path": str(post),
                "sha256": artifact_sha256(post),
            },
        },
    )
    return candidates, wrapper


def test_materializes_bijective_canonical_confidence_expert(tmp_path: Path) -> None:
    candidates, wrapper = _fixture(tmp_path)
    output = tmp_path / "output"
    result = materialize_evidence_fusion_adapter(
        candidates_path=candidates,
        semantic_wrapper_path=wrapper,
        expected_semantic_wrapper_sha256=artifact_sha256(wrapper),
        output_root=output,
        strict_formal=False,
    )
    prediction = pd.read_parquet(result["prediction_path"])
    assert prediction[["cancer_id", "lncrna_id", "pathway_id"]].to_dict("records") == [
        {"cancer_id": "ACC", "lncrna_id": "LNC:ENSG1", "pathway_id": "PROGENY:NFKB"},
        {"cancer_id": "ACC", "lncrna_id": "LNC:ENSG2", "pathway_id": "DOROTHEA:FOXM1"},
    ]
    assert prediction.availability.tolist() == [True, False]
    assert prediction.evidence_confidence_probability.iloc[0] == pytest.approx(0.8)
    assert pd.isna(prediction.evidence_confidence_probability.iloc[1])
    assert prediction.lncrna_prefix_restored.all()
    assert int(prediction.pathway_spelling_normalized.sum()) == 1
    binding = json.loads(Path(result["binding_path"]).read_text(encoding="utf-8"))
    assert binding["format"] == BINDING_FORMAT
    assert binding["fusion_input_eligible"] is True
    assert binding["confidence_only"] is True
    assert binding["affects_discovery"] is False
    assert binding["raw_prediction_direct_fusion_allowed"] is False


def test_rejects_duplicate_normalized_raw_keys(tmp_path: Path) -> None:
    candidates, wrapper = _fixture(tmp_path, duplicate_normalized_raw=True)
    with pytest.raises(EvidenceFusionAdapterError, match="Raw Evidence semantic/key audit failed"):
        materialize_evidence_fusion_adapter(
            candidates_path=candidates,
            semantic_wrapper_path=wrapper,
            expected_semantic_wrapper_sha256=artifact_sha256(wrapper),
            output_root=tmp_path / "output",
            strict_formal=False,
        )
