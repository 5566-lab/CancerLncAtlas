from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.experiment_perturbation_bridge import (
    ABSORPTION_STATUS,
    BINDING_FORMAT,
    ExperimentEvidenceBridgeError,
    build_absorption_bridge,
    compute_ablation_metrics,
    file_sha256,
)
from cc_hhgt.v32.experiment_perturbation_bridge_query import (
    ExperimentPerturbationBridgeAssetError,
    ExperimentPerturbationBridgeQuery,
)


def _experiment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "perturbation_event_id": ["BAGEV32:1"],
            "source_event_id": ["EV32:1"],
            "source_record_id": ["EVID:1"],
            "source_row_sha256": ["a" * 64],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["LNC:ENSG000001"],
            "pathway_id": ["KEGG:X"],
            "evidence_pathway_id": ["KEGG:X"],
            "partner_id": ["ENSG000002"],
            "mapping_route": ["PARTNER_EXACT_MEMBER"],
            "assay_family": ["functional_perturbation"],
            "assay_detail": ["SIRNA_KNOCKDOWN;WESTERN_BLOT"],
            "assay_detail_available": [True],
            "assay_detail_unavailable_reason": [pd.NA],
            "perturbation_methods": ["SIRNA_KNOCKDOWN"],
            "perturbation_method_available": [True],
            "readout_assays": ["WESTERN_BLOT"],
            "readout_assay_available": [True],
            "assay_detail_match_route": ["PMID_LNCRNA_GENE_EXACT"],
            "assay_detail_evidence_strength": ["HIGH"],
            "assay_detail_manual_review_required": [False],
            "direction_class": ["negative"],
            "pmid": ["123"],
            "cell_line": ["MCF7"],
            "tissue": ["breast"],
            "source_database": ["evidence_event"],
            "source_dataset": ["evidence_event"],
            "relation_type": ["regulation"],
            "family_broadcast_used": [False],
        }
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["BAGEV32:1"],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["ENSG000001"],
            "pathway_id": ["KEGG:X"],
            "partner_id": ["ENSG000002"],
            "route_type": ["PARTNER_EXACT_MEMBER"],
            "source_record_id": ["EVID:1"],
            "source_database": ["evidence_event"],
            "source_dataset": ["evidence_event"],
            "experiment_type": ["functional_perturbation"],
            "relation_type": ["regulation"],
            "pmid": ["123"],
            "cell_line": ["MCF7"],
            "tissue": ["breast"],
            "is_physical": [False],
            "physical_fact_id": [None],
            "leakage_fold": [3],
            "family_broadcast_used": [False],
        }
    )


def test_build_absorption_bridge_retains_native_identity_and_antidoublecount_flag():
    bridge = build_absorption_bridge(_experiment(), _evidence())
    assert len(bridge) == 1
    row = bridge.iloc[0]
    assert row.experiment_native_event_id == "BAGEV32:1"
    assert row.evidence_event_id == "BAGEV32:1"
    assert row.absorption_status == ABSORPTION_STATUS
    assert bool(row.separate_fusion_forbidden)
    assert not bool(row.used_for_separate_fusion)
    assert int(row.evidence_leakage_fold) == 3
    assert row.assay_detail == "SIRNA_KNOCKDOWN;WESTERN_BLOT"
    assert bool(row.assay_detail_available)
    assert row.assay_detail_evidence_strength == "HIGH"


def test_build_absorption_bridge_rejects_signature_drift():
    evidence = _evidence()
    evidence.loc[0, "pathway_id"] = "KEGG:WRONG"
    with pytest.raises(ExperimentEvidenceBridgeError, match="signatures"):
        build_absorption_bridge(_experiment(), evidence)


def test_compute_ablation_metrics_pair_blocked_and_finite():
    frame = pd.DataFrame(
        {
            "full_probability": [0.8, 0.2, 0.7, 0.1],
            "source_removed_probability": [0.6, 0.4, 0.55, 0.3],
            "fusion_target_private": [1.0, 0.0, 1.0, 0.0],
            "evidence_leakage_fold": [0, 0, 1, 1],
        }
    )
    metrics = compute_ablation_metrics(frame)
    aggregate = metrics.loc[metrics.evaluation_group.eq("ALL_PAIR_BLOCKED_OOF")].iloc[0]
    assert int(aggregate.evaluated_exact_keys) == 4
    assert aggregate.source_removal_minus_full_logloss > 0
    assert bool(aggregate.same_fold_checkpoint)
    assert not bool(aggregate.retrained)
    assert not bool(aggregate.fold_switched)


def _write_query_fixture(root: Path) -> Path:
    bridge = build_absorption_bridge(_experiment(), _evidence())
    exact = pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "lncrna_id": ["LNC:ENSG000001"],
            "pathway_id": ["KEGG:X"],
            "experiment_event_available": [True],
            "absorption_status": [ABSORPTION_STATUS],
        }
    )
    ablation = exact.copy()
    paths = {}
    for role, frame in {
        "lineage_bridge": bridge,
        "exact_query": exact,
        "source_removal_ablation": ablation,
    }.items():
        path = root / f"{role}.parquet"
        frame.to_parquet(path, index=False)
        paths[role] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "rows": len(frame),
        }
    binding = {
        "binding_format": BINDING_FORMAT,
        "status": "SUCCESS_HASH_BOUND_ABSORPTION_BRIDGE",
        "release_ready": True,
        "production_deployed": False,
        "route_determination": ABSORPTION_STATUS,
        "fully_absorbed_by_evidence_transformer": True,
        "confidence_route": "EXISTING_EVIDENCE_TRANSFORMER_ONLY",
        "separate_fusion_forbidden": True,
        "artifacts": paths,
        "invariants": {
            "experiment_native_event_identity_retained": True,
            "duplicate_fusion_created": False,
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
        },
    }
    binding_path = root / "BINDING.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    return binding_path


def test_query_returns_typed_null_for_no_experiment_event(tmp_path: Path):
    binding = _write_query_fixture(tmp_path)
    query = ExperimentPerturbationBridgeQuery(
        binding, expected_binding_sha256=file_sha256(binding)
    )
    found = query.query_native_events(cancer_id="BRCA", lncrna_id="ENSG000001")
    assert found["availability"] is True
    assert found["rows"][0]["experiment_native_event_id"] == "BAGEV32:1"
    missing = query.query_exact_keys(cancer_id="LUAD", lncrna_id="ENSG000001")
    assert missing["availability"] is False
    assert missing["unavailable_reason"] == "NO_EXPERIMENT_PERTURBATION_EVENT_FOR_FILTERS"


def test_query_rejects_unpinned_binding(tmp_path: Path):
    binding = _write_query_fixture(tmp_path)
    with pytest.raises(ExperimentPerturbationBridgeAssetError, match="SHA mismatch"):
        ExperimentPerturbationBridgeQuery(binding, expected_binding_sha256="0" * 64)
