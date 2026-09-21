from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cc_hhgt.v32.multimodal_fusion import pair_blocked_fold
from cc_hhgt.v32.primary_fold_view_audit import (
    AUDIT_FORMAT,
    PrimaryFoldViewAuditError,
    _independent_pair_fold,
    audit_primary_fold_views,
)
from cc_hhgt.v32.primary_fold_views import (
    EXTRACTION_FORMAT,
    FOLD_COLUMN,
    extract_primary_fold_views,
)
from cc_hhgt.v32.patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
)
from cc_hhgt.v32.sealed_test_inference import ACCEPTANCE_FORMAT, ACCEPTANCE_STATUS
from cc_hhgt.v32.training import PREPARED_FORMAT


CANCERS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
]


def test_independent_pair_fold_known_vectors_and_no_production_helper_dependency(
    monkeypatch,
) -> None:
    vectors = [
        ("LINC001", "hsa00010", 20260826, 3),
        (" ENSG0001 ", " KEGG:X ", 29, 4),
        ("A", "B", 0, 3),
        ("", "P", 20260826, 3),
    ]
    # Deliberately corrupt the production helper.  The independent auditor's
    # fixed known-vector results must not change.
    monkeypatch.setattr(
        "cc_hhgt.v32.multimodal_fusion.pair_blocked_fold",
        lambda *args, **kwargs: 99,
    )
    assert [
        _independent_pair_fold(lnc, pathway, seed=seed)
        for lnc, pathway, seed, _ in vectors
    ] == [expected for *_, expected in vectors]


def _fixture(tmp_path: Path):
    lncs = [f"L{index:02d}" for index in range(20)]
    pathways = [f"P{index:02d}" for index in range(5)]
    candidate = pd.DataFrame(
        [
            (cancer, lnc, pathway)
            for cancer in CANCERS
            for lnc in lncs
            for pathway in pathways
        ],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    ).sort_values(["cancer_id", "lncrna_id", "pathway_id"], kind="stable")
    candidate_path = tmp_path / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    candidate.to_parquet(candidate_path, index=False)
    budget = {
        "format": "CC_HHGT_V3_2_ROUTING_FAIR_BUDGET_V1",
        "budget_id": "test-budget",
        "scope": {
            "candidate_rows_per_cancer": 100,
            "formal_cancers": CANCERS,
            "modalities": ["mutation", "cnv", "atac"],
        },
        "seeds": {
            "optimizer_seed": 20260726,
            "pair_fold_seed": 20260826,
            "patient_fold_seed": 20260726,
        },
        "selection_policy": {
            "outer_folds": 5,
            "validation_offset": 1,
            "selection_scope": "INNER_VALIDATION_FOLD_ONLY",
            "outer_test_queries_during_selection": 0,
            "outer_test_evaluations_per_arm_fold": 1,
        },
    }
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(json.dumps(budget), encoding="utf-8")
    maps = {
        "cancer": {value: index for index, value in enumerate(CANCERS)},
        "lncRNA": {value: index for index, value in enumerate(lncs)},
        "pathway": {value: index for index, value in enumerate(pathways)},
    }
    bundle = SimpleNamespace(node_maps=maps)
    c = candidate.cancer_id.map(maps["cancer"]).to_numpy(np.int64)
    l = candidate.lncrna_id.map(maps["lncRNA"]).to_numpy(np.int64)
    p = candidate.pathway_id.map(maps["pathway"]).to_numpy(np.int64)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    payloads = {}
    # The formal payload carries one train-patient feature view and only swaps
    # the validation/test labels.  Model inputs must therefore be bitwise
    # identical across all three label views.
    splits = ("train", "validation", "test")
    for fold in range(5):
        path = prepared / f"PATIENT_FOLD_{fold}.pt"
        path.write_bytes(f"fixture-{fold}".encode())
        payload = {
            "prepared_format": "CC_HHGT_V3_2_PREPARED_FOLD_V1",
            "patient_fold": fold,
            "bundle": bundle,
            "contains_optimizer_state": False,
            "contains_trained_parameters": False,
            "artifact_hashes": {"code_sha256": "a" * 64},
            "input_scope": {
                "candidate_rows_per_split": len(candidate),
                "budget_per_cancer": 100,
                "discovery_features_split": "train_patients_only",
                "replication_labels_splits": ["validation", "test"],
                "held_out_effects_as_features": False,
            },
        }
        for split in splits:
            # Two batches exercise strict ordering across batch boundaries.
            batches = []
            for start, stop in ((0, 1700), (1700, len(candidate))):
                batches.append(
                    {
                        "candidate_batch": {
                            "c": c[start:stop],
                            "l": l[start:stop],
                            "p": p[start:stop],
                        },
                        "base_logit": np.full(stop - start, fold - 0.7, np.float32),
                        "conservation_context": np.column_stack(
                            [
                                np.full(stop - start, fold, np.float32),
                                np.arange(start, stop, dtype=np.float32) / len(candidate),
                            ]
                        ),
                        "graph_available": np.ones(stop - start, dtype=bool),
                        "direction_label": (
                            np.arange(start, stop, dtype=np.int64) % 2
                        ).astype(np.float32),
                        "direction_available": np.ones(stop - start, dtype=bool),
                        "proxy_label": (
                            (np.arange(start, stop) + fold + (split == "test")) % 2
                        ).astype(np.float32),
                    }
                )
            payload[f"{split}_batches"] = batches
        payloads[path.resolve()] = payload
    return candidate, candidate_path, budget_path, prepared, payloads


def test_extracts_five_source_matched_split_views(tmp_path: Path) -> None:
    candidate, candidate_path, budget_path, prepared, payloads = _fixture(tmp_path)
    output = tmp_path / "primary_fold_views"
    manifest = extract_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        output_root=output,
        payload_loader=lambda path: payloads[path.resolve()],
    )
    assert manifest["format"] == EXTRACTION_FORMAT
    assert manifest["status"] == "EXTRACTED_AWAITING_INDEPENDENT_AUDIT_NO_TRAINING"
    assert manifest["view_count"] == 15
    assert manifest["one_single_aggregated_primary_table_is_fair_training_authority"] is False
    assert not (output / "SUCCESS.json").exists()

    pair_folds = np.array(
        [
            pair_blocked_fold(lnc, pathway, seed=20260826)
            for lnc, pathway in candidate[["lncrna_id", "pathway_id"]].itertuples(
                index=False, name=None
            )
        ]
    )
    for outer in range(5):
        views = {
            split: pd.read_parquet(
                output
                / f"outer_pair_fold={outer}"
                / f"split={split}"
                / "part-0.parquet"
            )
            for split in ("train", "validation", "test")
        }
        assert sum(map(len, views.values())) == len(candidate)
        assert set(views["train"][FOLD_COLUMN]) == set(range(5)) - {
            outer,
            (outer + 1) % 5,
        }
        assert set(views["validation"][FOLD_COLUMN]) == {(outer + 1) % 5}
        assert set(views["test"][FOLD_COLUMN]) == {outer}
        assert set(views["train"].source_split) == {"train"}
        assert set(views["validation"].source_split) == {"validation"}
        assert set(views["test"].source_split) == {"test"}
        assert all(frame.source_patient_fold.eq(outer).all() for frame in views.values())
        observed = pd.concat(views.values(), ignore_index=True).sort_values(
            ["cancer_id", "lncrna_id", "pathway_id"], kind="stable"
        )
        assert observed[["cancer_id", "lncrna_id", "pathway_id"]].reset_index(
            drop=True
        ).equals(candidate.reset_index(drop=True))
        assert np.array_equal(
            observed[FOLD_COLUMN].to_numpy(), pair_folds
        )


def test_refuses_prepared_scope_leakage_flag(tmp_path: Path) -> None:
    _, candidate_path, budget_path, prepared, payloads = _fixture(tmp_path)
    payloads[(prepared / "PATIENT_FOLD_0.pt").resolve()]["input_scope"][
        "held_out_effects_as_features"
    ] = True
    try:
        extract_primary_fold_views(
            prepared_root=prepared,
            candidate_authority_path=candidate_path,
            budget_contract_path=budget_path,
            output_root=tmp_path / "bad",
            payload_loader=lambda path: payloads[path.resolve()],
        )
    except Exception as exc:
        assert "leakage/scope contract drift" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("leaky prepared payload was accepted")


def test_independent_audit_recomputes_source_values(tmp_path: Path) -> None:
    _, candidate_path, budget_path, prepared, payloads = _fixture(tmp_path)
    output = tmp_path / "primary_fold_views"
    extract_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        output_root=output,
        payload_loader=lambda path: payloads[path.resolve()],
    )
    audit_root = tmp_path / "audit"
    marker = audit_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        extraction_manifest_path=output / "EXTRACTION_MANIFEST.json",
        output_root=audit_root,
        payload_loader=lambda path: payloads[path.resolve()],
    )
    assert marker["format"] == AUDIT_FORMAT
    assert marker["primary_views_passed"] is True
    assert marker["modalities_passed"] is False
    assert marker["arm_training_authorized"] is False
    report = json.loads((audit_root / "AUDIT.json").read_text())
    assert report["outer_test_labels_present_only_in_test_view"] is True
    assert report["independent_pair_fold_implementation"] is True
    assert report[
        "train_discovery_features_bitwise_identical_across_label_views"
    ] is True
    assert report["article_fair_comparison_authorized"] is False


def test_independent_audit_refuses_held_out_feature_drift(tmp_path: Path) -> None:
    _, candidate_path, budget_path, prepared, payloads = _fixture(tmp_path)
    output = tmp_path / "primary_fold_views"
    extract_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        output_root=output,
        payload_loader=lambda path: payloads[path.resolve()],
    )
    payloads[(prepared / "PATIENT_FOLD_0.pt").resolve()]["test_batches"][0][
        "base_logit"
    ][0] += np.float32(0.25)
    try:
        audit_primary_fold_views(
            prepared_root=prepared,
            candidate_authority_path=candidate_path,
            budget_contract_path=budget_path,
            extraction_manifest_path=output / "EXTRACTION_MANIFEST.json",
            output_root=tmp_path / "bad_feature_audit",
            payload_loader=lambda source: payloads[source.resolve()],
        )
    except PrimaryFoldViewAuditError as exc:
        assert "held-out feature drift" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("independent audit accepted held-out feature drift")


def test_independent_audit_refuses_value_drift(tmp_path: Path) -> None:
    _, candidate_path, budget_path, prepared, payloads = _fixture(tmp_path)
    output = tmp_path / "primary_fold_views"
    extract_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        output_root=output,
        payload_loader=lambda path: payloads[path.resolve()],
    )
    path = output / "outer_pair_fold=0" / "split=test" / "part-0.parquet"
    drifted = pd.read_parquet(path)
    drifted.loc[0, "primary_probability"] = np.float32(0.123456)
    drifted.to_parquet(path, index=False)
    try:
        audit_primary_fold_views(
            prepared_root=prepared,
            candidate_authority_path=candidate_path,
            budget_contract_path=budget_path,
            extraction_manifest_path=output / "EXTRACTION_MANIFEST.json",
            output_root=tmp_path / "bad_audit",
            payload_loader=lambda source: payloads[source.resolve()],
        )
    except PrimaryFoldViewAuditError as exc:
        assert "source/output mismatch" in str(exc) or "manifest/output mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("independent audit accepted primary value drift")


def test_v2_training_payload_uses_winner_locked_sealed_test_and_independent_audit(
    tmp_path: Path,
) -> None:
    candidate, candidate_path, budget_path, prepared, payloads = _fixture(tmp_path)
    pair_folds = np.array(
        [
            pair_blocked_fold(lnc, pathway, seed=20260826)
            for lnc, pathway in candidate[["lncrna_id", "pathway_id"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=np.int8,
    )
    probability = np.empty(len(candidate), dtype=np.float32)
    target = np.empty(len(candidate), dtype=np.uint8)
    for fold in range(5):
        payload = payloads[(prepared / f"PATIENT_FOLD_{fold}.pt").resolve()]
        payload["prepared_format"] = PREPARED_FORMAT
        payload["formal_graph_variant"] = "G2"
        payload["input_scope"]["replication_labels_splits"] = ["validation"]
        test_batches = payload.pop("test_batches")
        mask = pair_folds == fold
        offset = 0
        for batch in test_batches:
            size = len(batch["base_logit"])
            local = mask[offset : offset + size]
            probability[offset : offset + size][local] = (
                1.0 / (1.0 + np.exp(-batch["base_logit"][local]))
            ).astype(np.float32)
            target[offset : offset + size][local] = batch["proxy_label"][local].astype(
                np.uint8
            )
            offset += size
        assert offset == len(candidate)
        assert "test_batches" not in payload

    sealed_root = tmp_path / "sealed_winner"
    sealed_root.mkdir()
    primary_path = sealed_root / "PRIMARY_FUSION_FRAME.PRIVATE.parquet"
    primary = candidate.copy()
    primary["primary_probability"] = probability
    primary["fusion_target"] = target
    primary[FOLD_COLUMN] = pair_folds
    primary["source_patient_fold"] = pair_folds
    primary["source_split"] = "test"
    primary.to_parquet(primary_path, index=False)
    import hashlib

    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    acceptance_path = sealed_root / "WINNER_LOCK_SEALED_TEST_INFERENCE_ACCEPTANCE.json"
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "status": ACCEPTANCE_STATUS,
        "graph_variant": "G2",
        "winner_id": "G2",
        "winner_declaration_sha256": "d" * 64,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "candidate_authority_sha256": sha(candidate_path),
        "pair_fold_seed": 20260826,
        "test_labels_absent_from_training_payload": True,
        "sealed_payloads_opened_only_after_winner_lock": True,
        "test_metrics_used_for_winner_selection": False,
        "winner_or_checkpoint_changed_by_test": False,
        "old_fold_outputs_used": False,
        "artifacts_read_only": True,
        "primary_fusion_frame": {
            "path": str(primary_path),
            "sha256": sha(primary_path),
        },
    }
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

    output = tmp_path / "primary_fold_views_v2"
    manifest = extract_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        output_root=output,
        sealed_test_acceptance_path=acceptance_path,
        expected_graph_variant="G2",
        payload_loader=lambda path: payloads[path.resolve()],
    )
    assert manifest["test_labels_absent_from_training_payload"] is True
    assert manifest["test_view_source"] == "POST_WINNER_LOCK_SEALED_TEST_INFERENCE"
    assert all("test_batches" not in payload for payload in payloads.values())

    audit_root = tmp_path / "primary_fold_views_v2_audit"
    marker = audit_primary_fold_views(
        prepared_root=prepared,
        candidate_authority_path=candidate_path,
        budget_contract_path=budget_path,
        extraction_manifest_path=output / "EXTRACTION_MANIFEST.json",
        output_root=audit_root,
        payload_loader=lambda path: payloads[path.resolve()],
    )
    assert marker["primary_views_passed"] is True
    report = json.loads((audit_root / "AUDIT.json").read_text(encoding="utf-8"))
    assert report["test_source_is_post_winner_lock_sealed_inference"] is True
    assert report["training_payload_test_batches_read"] is False
    assert all(
        row["test_batches_absent_from_training_payload"]
        for row in report["source_prepared_folds"]
    )
