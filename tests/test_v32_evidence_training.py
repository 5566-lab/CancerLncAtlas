from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.evidence_training import (
    ANALYSIS_VERSION,
    BagExample,
    CORE_EXPORT_FORMAT,
    CURRENT_G2_CORE_LINEAGE_FORMAT,
    EVIDENCE_SPLIT_POLICY,
    EvidenceTrainingContractError,
    EVALUATION_PROVENANCE_EXCLUSION_POLICY,
    FORMAL_CANDIDATE_SHA256,
    assert_label_blind_raw_input,
    assert_pair_isolation,
    assert_source_isolation,
    assign_leakage_safe_folds,
    build_pair_blocked_split_audit,
    build_exact_event_bags,
    build_fresh_init_contract,
    build_fresh_private_eventset_head,
    build_unavailable_prediction_frame,
    complete_prediction_frame,
    exclude_evaluation_provenance_from_training,
    fit_private_eventset_head,
    file_sha256,
    input_path_sha256,
    model_parameter_sha256,
    predict_private_eventset_head,
    read_candidate_table,
    validate_evidence_split_preflight_gate,
    validate_core_embedding_lineage,
)
from cc_hhgt.v32.full_model_contract import validate_public_module_frame
from cc_hhgt.v32.patient_fold_authority import (
    FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256,
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
)


def _members() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pathway_id": ["R-HSA-1", "R-HSA-2"],
            "gene_id": ["ENSG000002", "ENSG000003"],
            "mapping_status": ["mapped", "mapped"],
        }
    )


def _physical_rows(count: int = 2) -> pd.DataFrame:
    row = {
        "source_record_id": "REC-1",
        "lncrna_id": "ENSG000001",
        "partner_id": "ENSG000002",
        "cancer_id": "BRCA",
        "source_database": "BioGRID",
        "source_dataset": "BioGRID-5.0",
        "pmid": "12345678",
        "relation_type": "physical_binding",
        "experiment_type": "RNA pull-down",
        "direction": "positive",
        "predictive_core_eligible": True,
        "experimental": True,
    }
    return pd.DataFrame([{**row, "event_id": f"RAW-{index}"} for index in range(count)])


def test_deduplicates_events_but_retains_raw_lineage_and_physical_fact() -> None:
    evidence = _physical_rows(2)
    interaction = _physical_rows(1).drop(columns="event_id")
    interaction["interaction_id"] = "REC-1"

    result = build_exact_event_bags(evidence, interaction, _members())

    assert len(result.events) == 1
    assert result.events.iloc[0].source_occurrence_count == 3
    assert result.events.iloc[0].pathway_id == "R-HSA-1"
    assert len(result.lineage) == 3
    assert result.lineage.event_id.nunique() == 1
    assert not result.lineage.family_broadcast_used.any()
    assert len(result.physical_facts) == 1
    assert result.physical_facts.iloc[0].source_occurrence_count == 3
    assert not bool(result.physical_facts.iloc[0].is_prediction)


def test_family_only_event_is_rejected_and_never_broadcast_to_exact() -> None:
    family_only = pd.DataFrame(
        [
            {
                "event_id": "FAMILY-ONLY",
                "cancer_id": "BRCA",
                "lncrna_id": "ENSG000001",
                "pathway_family_id": "FAMILY:IMMUNE",
                "source_database": "Manual",
                "source_dataset": "curation-v1",
                "pmid": "12345678",
                "experimental": True,
            }
        ]
    )

    result = build_exact_event_bags(family_only, pd.DataFrame(), _members())

    assert result.events.empty
    assert len(result.rejected) == 1
    assert result.rejected.iloc[0].rejection_reason == "FAMILY_ONLY_NOT_BROADCAST_TO_EXACT"


def test_pair_blocked_split_is_deterministic_and_provenance_is_audited() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "event_id": f"EV-{index}",
                "cancer_id": "BRCA" if index % 2 else "LUAD",
                "lncrna_id": f"LNC-{index}",
                "pathway_id": f"PATH-{index}",
                "source_database": "SHARED-SOURCE",
                "source_dataset": "SHARED-DATASET",
                "pmid": "SHARED-PMID",
            }
        )
    # The same biological pair materialised into a second cancer must remain in
    # one fold even though cancer_id differs.
    rows.append(
        {
            **rows[0],
            "event_id": "EV-CROSS-CANCER",
            "cancer_id": "KIRC",
        }
    )
    events = pd.DataFrame(rows)

    first = assign_leakage_safe_folds(events, seed=17)
    second = assign_leakage_safe_folds(events, seed=17)

    assert first.leakage_fold.tolist() == second.leakage_fold.tolist()
    assert_pair_isolation(first)
    assert sorted(first.leakage_fold.unique()) == [0, 1, 2, 3, 4]
    assert first.groupby("source_database").leakage_fold.nunique().max() == 5
    assert first.groupby("source_dataset").leakage_fold.nunique().max() == 5
    assert first.groupby("pmid").leakage_fold.nunique().max() == 5
    assert (
        first.groupby(["lncrna_id", "pathway_id"])
        .leakage_fold.nunique()
        .max()
        == 1
    )
    with pytest.raises(EvidenceTrainingContractError, match="PMID/source/dataset"):
        assert_source_isolation(first)

    audit = build_pair_blocked_split_audit(first)
    assert audit["split_policy"] == EVIDENCE_SPLIT_POLICY
    assert audit["hard_pair_cross_fold_count"] == 0
    assert audit["hard_pair_count"] == 20
    assert audit["all_five_folds_populated"] is True
    assert audit["audit_only_provenance_overlap"]["pmid"] == {
        "tokens": 1,
        "tokens_spanning_folds": 1,
    }


def test_evaluation_pmids_and_source_events_are_removed_only_from_train() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "EVAL-0",
                "source_event_id": "SOURCE-EVENT-1",
                "source_record_id": "REC-1",
                "cancer_id": "BRCA",
                "lncrna_id": "LNC-EVAL",
                "pathway_id": "PATH-EVAL",
                "source_database": "DB",
                "source_dataset": "DS",
                "pmid": "PMID-1",
                "leakage_fold": 0,
            },
            {
                "event_id": "TRAIN-PMID-COPY",
                "source_event_id": "SOURCE-EVENT-2",
                "source_record_id": "REC-2",
                "cancer_id": "LUAD",
                "lncrna_id": "LNC-TRAIN-1",
                "pathway_id": "PATH-TRAIN-1",
                "source_database": "DB",
                "source_dataset": "DS",
                "pmid": "PMID-1",
                "leakage_fold": 2,
            },
            {
                "event_id": "TRAIN-SOURCE-COPY",
                "source_event_id": "SOURCE-EVENT-1",
                "source_record_id": "REC-3",
                "cancer_id": "KIRC",
                "lncrna_id": "LNC-TRAIN-2",
                "pathway_id": "PATH-TRAIN-2",
                "source_database": "OTHER-DB",
                "source_dataset": "OTHER-DS",
                "pmid": "PMID-2",
                "leakage_fold": 3,
            },
            {
                "event_id": "TRAIN-UNIQUE",
                "source_event_id": "SOURCE-EVENT-UNIQUE",
                "source_record_id": "REC-UNIQUE",
                "cancer_id": "KIRP",
                "lncrna_id": "LNC-TRAIN-3",
                "pathway_id": "PATH-TRAIN-3",
                "source_database": "DB",
                "source_dataset": "DS",
                "pmid": "PMID-UNIQUE",
                "leakage_fold": 4,
            },
        ]
    )

    filtered, audit = exclude_evaluation_provenance_from_training(
        events, evaluation_folds={0, 1}
    )

    assert set(filtered.event_id) == {"EVAL-0", "TRAIN-UNIQUE"}
    assert audit["train_event_rows_before"] == 3
    assert audit["train_event_rows_removed"] == 2
    assert audit["evaluation_event_rows_unchanged"] == 1
    assert audit["residual_train_evaluation_provenance_overlap_count"] == 0
    assert audit["residual_pmid_overlap_count"] == 0
    assert audit["residual_source_event_overlap_count"] == 0
    assert audit["residual_source_record_overlap_count"] == 0


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core_root(tmp_path):
    root = tmp_path / "core"
    fold_root = root / "patient_fold=0"
    fold_root.mkdir(parents=True)
    exports = {}
    for node_type in ("cancer", "lncRNA", "pathway"):
        path = fold_root / f"{node_type}.parquet"
        path.write_bytes(f"fresh-v32-{node_type}".encode())
        exports[node_type] = {
            "path": str(path),
            "sha256": _sha(path),
            "rows": 1,
            "features": 2,
        }
    manifest = {
        "export_format": CORE_EXPORT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "folds": {
            "0": {
                "patient_fold": 0,
                "trained_from_scratch": True,
                "old_checkpoint_loaded": False,
                "checkpoint_format": "CC_HHGT_V3_2_FULL_TRAINING_STATE_V1",
                "checkpoint_sha256": "a" * 64,
                "core_parameter_sha256": "b" * 64,
                "exports": exports,
            }
        },
    }
    (root / "CORE_EMBEDDING_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def test_fresh_init_contract_binds_same_fold_core_hash_and_detects_tamper(tmp_path) -> None:
    root = _core_root(tmp_path)
    lineage = validate_core_embedding_lineage(root, 0)
    contract = build_fresh_init_contract(patient_fold=0, seed=19, core_lineage=lineage)

    assert contract["private_parameters_fresh_init"] is True
    assert contract["initialized_from_checkpoint"] is False
    assert contract["historical_evidence_checkpoint_allowed"] is False
    assert contract["core_frozen"] is True
    assert contract["core_detached"] is True
    assert contract["core_checkpoint_sha256"] == "a" * 64
    assert contract["core_parameter_sha256"] == "b" * 64

    (root / "patient_fold=0" / "lncRNA.parquet").write_bytes(b"tampered")
    with pytest.raises(EvidenceTrainingContractError, match="hash mismatch"):
        validate_core_embedding_lineage(root, 0)


def test_current_g2_guard_rejects_legacy_self_declared_core(tmp_path) -> None:
    root = _core_root(tmp_path)
    manifest_path = root / "CORE_EMBEDDING_MANIFEST.json"

    with pytest.raises(EvidenceTrainingContractError, match="formal lineage is missing"):
        validate_core_embedding_lineage(
            root,
            0,
            require_current_g2_authority=True,
            expected_manifest_sha256=_sha(manifest_path),
            expected_graph_authority_receipt_sha256="c" * 64,
        )


def test_current_g2_guard_requires_external_hash_and_frozen_authorities(tmp_path) -> None:
    root = _core_root(tmp_path)
    manifest_path = root / "CORE_EMBEDDING_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph_sha = "c" * 64
    manifest["formal_lineage"] = {
        "format": CURRENT_G2_CORE_LINEAGE_FORMAT,
        "formal_graph_variant": "G2",
        "graph_authority_receipt_sha256": graph_sha,
        "patient_fold_authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "patient_authority_logical_sha256": (
            FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256
        ),
        "sealed_test_opened": False,
    }
    manifest["folds"]["0"].update(
        {
            "formal_graph_variant": "G2",
            "graph_authority_receipt_sha256": graph_sha,
            "patient_fold_authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
            "prepared_sha256": "d" * 64,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed_manifest_sha = _sha(manifest_path)

    lineage = validate_core_embedding_lineage(
        root,
        0,
        require_current_g2_authority=True,
        expected_manifest_sha256=observed_manifest_sha,
        expected_graph_authority_receipt_sha256=graph_sha,
    )
    assert lineage.formal_graph_variant == "G2"
    assert lineage.graph_authority_receipt_sha256 == graph_sha
    assert lineage.patient_fold_authority_receipt_sha256 == FROZEN_V32_RECEIPT_SHA256

    with pytest.raises(EvidenceTrainingContractError, match="externally SHA256-pinned"):
        validate_core_embedding_lineage(
            root,
            0,
            require_current_g2_authority=True,
            expected_manifest_sha256="d" * 64,
            expected_graph_authority_receipt_sha256=graph_sha,
        )


def test_private_head_is_randomly_initialized_without_core_parameters() -> None:
    pytest.importorskip("torch")
    first = build_fresh_private_eventset_head(
        event_feature_dim=32, core_feature_dim=12, hidden_dim=16, seed=101
    )
    repeated = build_fresh_private_eventset_head(
        event_feature_dim=32, core_feature_dim=12, hidden_dim=16, seed=101
    )
    different = build_fresh_private_eventset_head(
        event_feature_dim=32, core_feature_dim=12, hidden_dim=16, seed=102
    )

    assert model_parameter_sha256(first) == model_parameter_sha256(repeated)
    assert model_parameter_sha256(first) != model_parameter_sha256(different)
    assert first.private_parameters_fresh_init is True
    assert first.core_frozen is True
    assert first.core_detached is True


def test_private_attention_head_minimal_training_keeps_core_as_detached_input() -> None:
    pytest.importorskip("torch")
    examples = []
    for index in range(10):
        core = np.linspace(0, 1, 12, dtype=np.float32) + index / 100
        core.setflags(write=False)
        examples.append(
            BagExample(
                key=("BRCA", f"LNC-{index}", f"PATH-{index % 3}"),
                event_features=np.full(
                    (1 + index % 2, 32), index / 10, dtype=np.float32
                ),
                core_features=core,
                confidence_target=float(index % 2),
                direction_target=2 if index % 2 else 0,
                leakage_fold=index % 5,
                event_count=1 + index % 2,
            )
        )
    fit = fit_private_eventset_head(
        examples[:6],
        examples[6:8],
        event_feature_dim=32,
        core_feature_dim=12,
        hidden_dim=16,
        seed=7,
        epochs=1,
        patience=1,
        batch_size=3,
        device="cpu",
    )
    prediction = predict_private_eventset_head(
        fit, examples[8:], batch_size=2, mc_samples=2, device="cpu"
    )

    assert fit.optimizer_steps == 2
    assert len(prediction) == 2
    assert prediction.availability.all()
    assert prediction.evidence_confidence_probability.between(0, 1).all()


@pytest.mark.parametrize("column", ["label", "sample_weight", "score", "confidence"])
def test_forbids_pair_evidence_and_old_confidence_fields(column: str) -> None:
    frame = pd.DataFrame({column: [1], "event_id": ["x"]})
    with pytest.raises(EvidenceTrainingContractError, match="forbidden"):
        assert_label_blind_raw_input(frame, "evidence_event.parquet")

    with pytest.raises(EvidenceTrainingContractError, match="pair_evidence"):
        assert_label_blind_raw_input(pd.DataFrame({"event_id": ["x"]}), "pair_evidence.parquet")


def test_missing_exact_event_is_null_with_reason_and_rank_is_not_consumed() -> None:
    universe = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["LNC-1", "LNC-2"],
            "pathway_id": ["PATH-1", "PATH-2"],
            "main_rank": [1, 2],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "EV-1",
                "cancer_id": "BRCA",
                "lncrna_id": "LNC-1",
                "pathway_id": "PATH-1",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC-1",
                "pathway_id": "PATH-1",
                "evidence_confidence_probability": 0.8,
                "direction": "positive",
                "uncertainty": 0.2,
                "availability": True,
                "unavailable_reason": "",
                "event_count": 1,
                "evidence_fold": 0,
            }
        ]
    )

    output = complete_prediction_frame(universe, predictions, events)

    missing = output.loc[output.lncrna_id.eq("LNC-2")].iloc[0]
    assert np.isnan(missing.evidence_confidence_probability)
    assert np.isnan(missing.uncertainty)
    assert missing.availability is False or not missing.availability
    assert missing.unavailable_reason == "NO_EXACT_PATHWAY_EVENT"
    assert missing.failure_reason == "NO_EXACT_PATHWAY_EVENT"
    assert "main_rank" not in output.columns
    assert not output.changes_primary_ranking.any()
    assert not output.main_ranking_modified.any()
    validate_public_module_frame("evidence", output)


def test_candidate_reader_and_unavailable_frame_preserve_every_exact_row(
    tmp_path,
) -> None:
    raw = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA", "LUAD", "LUAD"],
            "lncrna_id": ["LNC-1", "LNC-2", "LNC-1", "LNC-2"],
            "pathway_id": ["PATH-1", "PATH-2", "PATH-1", "PATH-2"],
            "pathway_family_id": ["FAMILY-A"] * 4,
            "association_membership_probability": [0.99, 0.8, 0.7, 0.6],
            "main_rank": [1, 2, 1, 2],
            "historical_probability_v3_1": [0.5] * 4,
        }
    )
    path = tmp_path / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    raw.to_parquet(path, index=False)

    candidates = read_candidate_table(path)
    output = build_unavailable_prediction_frame(
        candidates,
        training_run_id="V32-EVIDENCE-FORMAL-TEST",
        failure_reason="STRICT_CONNECTED_COMPONENT_COUNT_1_LT_5",
    )

    assert candidates.columns.tolist() == ["cancer_id", "lncrna_id", "pathway_id"]
    assert len(output) == len(raw)
    assert output[["cancer_id", "lncrna_id", "pathway_id"]].to_records(index=False).tolist() == (
        raw[["cancer_id", "lncrna_id", "pathway_id"]].to_records(index=False).tolist()
    )
    assert output.evidence_confidence_probability.isna().all()
    assert not output.availability.any()
    assert output.failure_reason.str.strip().ne("").all()
    assert not set(raw.columns[3:]).intersection(output.columns)
    validate_public_module_frame("evidence", output)


def test_checked_in_formal_candidate_universe_is_complete_33_cancer_frame() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "formal_prepared"
        / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    )
    if not path.is_file():
        pytest.skip("Formal candidate artifact is not present in this checkout")

    candidates = read_candidate_table(path)

    assert len(candidates) == 3_300_000
    assert input_path_sha256(path) == FORMAL_CANDIDATE_SHA256
    assert candidates.cancer_id.nunique() == 33
    assert candidates.columns.tolist() == ["cancer_id", "lncrna_id", "pathway_id"]
    assert not candidates.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any()
    assert not any(
        token in column.lower()
        for column in candidates.columns
        for token in ("rank", "score", "probability", "v2_", "v3_1")
    )


def test_candidate_reader_reports_missing_parquet_engine_truthfully(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "candidate.parquet"
    path.write_bytes(b"not-read")

    def missing_engine(*args, **kwargs):
        raise ImportError("Unable to find a usable engine")

    monkeypatch.setattr(pd, "read_parquet", missing_engine)
    with pytest.raises(EvidenceTrainingContractError, match="parquet engine is unavailable"):
        read_candidate_table(path)


def test_formal_training_gate_revalidates_pair_split_and_input_hashes(tmp_path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    inputs = {}
    for name in ("evidence_event", "interaction_relation", "pathway_members"):
        path = input_root / f"{name}.parquet"
        pd.DataFrame({"value": [name]}).to_parquet(path, index=False)
        inputs[name] = path
    candidates = input_root / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    pd.DataFrame(
        {"cancer_id": ["BRCA"], "lncrna_id": ["LNC-1"], "pathway_id": ["PATH-1"]}
    ).to_parquet(candidates, index=False)
    inputs["candidates"] = candidates
    candidate_sha = input_path_sha256(candidates)
    folds = {
        str(fold): {
            "train_bags_after_provenance_exclusion": 3,
            "validation_bags": 1,
            "heldout_bags": 1,
            "train_confidence_supervised_event_rows": 2,
            "provenance_exclusion_audit": {
                "residual_pmid_overlap_count": 0,
                "residual_source_event_overlap_count": 0,
                "residual_source_record_overlap_count": 0,
                "residual_train_evaluation_provenance_overlap_count": 0,
            },
        }
        for fold in range(5)
    }
    manifest = {
        "status": "PREFLIGHT_SUCCESS",
        "training_status": "PREFLIGHT_ONLY_NO_TRAINING",
        "release_ready": False,
        "partial_not_publishable": True,
        "predictions_written": 0,
        "checkpoints_written": 0,
        "optimizer_steps": 0,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False,
        "old_confidence_loaded": False,
        "old_ranking_loaded": False,
        "split_policy": EVIDENCE_SPLIT_POLICY,
        "evaluation_provenance_exclusion_policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
        "inputs": {
            name: {"path": str(path), "sha256": input_path_sha256(path)}
            for name, path in inputs.items()
        },
        "candidate_authority": {
            "sha256": candidate_sha,
            "sha256_pinned": True,
        },
        "counts": {"physical_facts": 1},
        "split_audit": {
            "hard_pair_cross_fold_count": 0,
            "all_five_folds_populated": True,
            "active_folds": [0, 1, 2, 3, 4],
        },
        "folds": folds,
    }
    preflight_root = tmp_path / "preflight"
    preflight_root.mkdir()
    manifest_path = preflight_root / "SPLIT_PREFLIGHT.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    success_path = preflight_root / "PREFLIGHT_SUCCESS.json"
    success_path.write_text(
        json.dumps(
            {
                "status": "PREFLIGHT_SUCCESS",
                "release_ready": False,
                "partial_not_publishable": True,
                "manifest_sha256": file_sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )

    audit = validate_evidence_split_preflight_gate(
        manifest_path,
        evidence_event_path=inputs["evidence_event"],
        interaction_relation_path=inputs["interaction_relation"],
        pathway_member_path=inputs["pathway_members"],
        candidates_path=candidates,
        expected_candidate_sha256=candidate_sha,
    )
    assert audit["pair_isolation_verified"] is True
    assert audit["five_fold_provenance_exclusion_verified"] is True
    assert audit["no_prediction_or_checkpoint_artifacts_verified"] is True

    manifest["folds"]["0"]["provenance_exclusion_audit"][
        "residual_pmid_overlap_count"
    ] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    success_path.write_text(
        json.dumps(
            {
                "status": "PREFLIGHT_SUCCESS",
                "release_ready": False,
                "partial_not_publishable": True,
                "manifest_sha256": file_sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceTrainingContractError, match="residual provenance leakage"):
        validate_evidence_split_preflight_gate(
            manifest_path,
            evidence_event_path=inputs["evidence_event"],
            interaction_relation_path=inputs["interaction_relation"],
            pathway_member_path=inputs["pathway_members"],
            candidates_path=candidates,
            expected_candidate_sha256=candidate_sha,
        )
