from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.single_cell_fusion_adapter import ADAPTER_FORMAT, ANALYSIS_VERSION
from cc_hhgt.v32.single_cell_fusion_binding import (
    BINDING_FORMAT,
    FORMAL_CANDIDATE_SHA256,
    artifact_sha256,
)
from cc_hhgt.v32.single_cell_fusion_query import (
    SingleCellFusionQueryAssetError,
    SingleCellFusionQueryInputError,
    SingleCellFusionReleaseQuery,
)


def _binding(tmp_path: Path) -> tuple[Path, str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prediction = tmp_path / "single_cell_exact_fusion_expert.parquet"
    pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG000001",
                "pathway_id": "PATH:P1",
                "single_cell_replication_probability": 0.8,
                "single_cell_available": True,
                "single_cell_unavailable_reason": None,
                "analysis_version": ANALYSIS_VERSION,
                "module_id": "single_cell",
                "adapter_format": ADAPTER_FORMAT,
                "target_level": "cancer_x_lncrna_x_exact_pathway",
                "direct_target_evidence": False,
                "family_to_exact_broadcast": False,
                "changes_primary_ranking": False,
                "availability_encoding": "null_with_reason",
            },
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG000001",
                "pathway_id": "PATH:P2",
                "single_cell_replication_probability": None,
                "single_cell_available": False,
                "single_cell_unavailable_reason": "NO_SINGLE_CELL_EXACT_PAIR_MEASUREMENT",
                "analysis_version": ANALYSIS_VERSION,
                "module_id": "single_cell",
                "adapter_format": ADAPTER_FORMAT,
                "target_level": "cancer_x_lncrna_x_exact_pathway",
                "direct_target_evidence": False,
                "family_to_exact_broadcast": False,
                "changes_primary_ranking": False,
                "availability_encoding": "null_with_reason",
            },
        ]
    ).to_parquet(prediction, index=False)
    binding = tmp_path / "SINGLE_CELL_FUSION_BINDING.json"
    payload = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS_HASH_BOUND_FUSION_INPUT",
        "training_run_id": "V32-SINGLE-CELL-FRESH-TEST",
        "adapter_format": ADAPTER_FORMAT,
        "target_level": "cancer_x_lncrna_x_exact_pathway",
        "candidate_rows": 2,
        "available_rows": 1,
        "unavailable_rows": 1,
        "prediction": {
            "path": str(prediction),
            "sha256": artifact_sha256(prediction),
            "rows": 2,
        },
        "candidate_authority": {
            "path": "/formal/candidates.parquet",
            "sha256": FORMAL_CANDIDATE_SHA256,
        },
        "fusion_input_eligible": True,
        "direct_target_evidence": False,
        "affects_discovery": True,
        "affects_confidence": True,
        "changes_primary_ranking": False,
        "family_to_exact_broadcast": False,
        "unavailable_encoding": "null_with_reason",
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "release_ready": False,
        "production_deployed": False,
    }
    binding.write_text(json.dumps(payload), encoding="utf-8")
    return binding, artifact_sha256(binding), prediction


def test_hash_pinned_exact_query_preserves_typed_null_and_policy(tmp_path: Path) -> None:
    binding, digest, _ = _binding(tmp_path)
    query = SingleCellFusionReleaseQuery(
        binding, expected_binding_sha256=digest
    )
    available = query.query_associations(
        cancer_id="brca", lncrna_id="ENSG000001.9", availability=True
    )
    assert available["returned_rows"] == 1
    assert available["rows"][0]["single_cell_replication_probability"] == 0.8
    missing = query.query_associations(
        lncrna_id="LNC:ENSG000001", pathway_id="PATH:P2"
    )
    assert missing["returned_rows"] == 1
    assert missing["rows"][0]["single_cell_replication_probability"] is None
    assert missing["typed_unavailable_nulls"] is True
    assert missing["changes_primary_ranking"] is False
    assert query.capability_status()["formal_full_universe_adapter"] is True


def test_query_fails_closed_on_hash_semantic_and_input_drift(tmp_path: Path) -> None:
    binding, digest, prediction = _binding(tmp_path)
    with pytest.raises(SingleCellFusionQueryAssetError, match="requires"):
        SingleCellFusionReleaseQuery(binding, expected_binding_sha256=None)
    with pytest.raises(SingleCellFusionQueryAssetError, match="mismatch"):
        SingleCellFusionReleaseQuery(binding, expected_binding_sha256="0" * 64)
    query = SingleCellFusionReleaseQuery(binding, expected_binding_sha256=digest)
    with pytest.raises(SingleCellFusionQueryInputError, match="min_probability"):
        query.query_associations(min_probability=1.1)
    frame = pd.read_parquet(prediction)
    frame.loc[1, "single_cell_replication_probability"] = 0.5
    frame.to_parquet(prediction, index=False)
    with pytest.raises(SingleCellFusionQueryAssetError, match="SHA drift"):
        SingleCellFusionReleaseQuery(binding, expected_binding_sha256=digest)
