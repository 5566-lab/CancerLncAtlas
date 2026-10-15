from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import torch


pytest.importorskip("torch_geometric")

from cc_hhgt.v32.single_cell_hhgt import SingleCellHHGT
from scripts import train_single_cell_hhgt_149 as trainer
from scripts import validate_single_cell_hhgt_training_149 as validator


STATIC_FIELDS = {
    "gene_pathway_membership",
    "pathway_family_membership",
    "external_relation_types",
    "external_edge_batches",
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": trainer.sha256(path),
    }


def _write_static_asset(root: Path, cancer_id: str = "C1") -> Path:
    authority_root = root / "graph_assets" / "cancers" / cancer_id / "STATIC"
    authority_success = authority_root / "SUCCESS.json"
    matrix_identity = authority_root.parent / "MATRIX_IDENTITY.json"
    _write_json(
        authority_success,
        {"format": trainer.GRAPH_ASSET_FORMAT, "state": "CANCER_STATIC_READY"},
    )
    _write_json(
        matrix_identity,
        {
            "cancer_id": cancer_id,
            "node_counts": {"lncRNA": 2, "gene": 3, "pathway": 2, "pathway_family": 2},
        },
    )
    static_path = root / "episodes" / "cancers" / cancer_id / "EPISODE_STATIC.pt"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": trainer.STATIC_FORMAT,
            "cancer_id": cancer_id,
            "graph_asset_static_success": _record(authority_success),
            "matrix_identity": _record(matrix_identity),
            "node_counts": {
                "lncRNA": 2,
                "gene": 3,
                "pathway": 2,
                "pathway_family": 2,
            },
            "gene_pathway_membership": {
                "row_index": torch.tensor([0, 1, 2]),
                "column_index": torch.tensor([0, 0, 1]),
                "weight": torch.ones(3),
                "shape": [3, 2],
            },
            "pathway_family_membership": {
                "row_index": torch.tensor([0, 0, 1]),
                "column_index": torch.tensor([0, 1, 1]),
                "weight": torch.tensor([1.0, 0.5, 1.0]),
                "shape": [2, 2],
            },
            "external_relation_types": [
                ["gene", "physical_target", "pathway"],
                ["pathway", "physical_target_reverse", "gene"],
            ],
            "external_edge_batches": [
                {
                    "source_type": "gene",
                    "relation": "physical_target",
                    "target_type": "pathway",
                    "edge_index": torch.tensor([[0, 1], [0, 1]]),
                    "edge_weight": torch.tensor([1.0, 1.0]),
                    "available": True,
                    "source_hash": "synthetic-physical-binding-sha256",
                    "provenance": "synthetic:physical_binding",
                    "source_donors": [],
                },
                {
                    "source_type": "pathway",
                    "relation": "physical_target_reverse",
                    "target_type": "gene",
                    "edge_index": torch.tensor([[0, 1], [0, 1]]),
                    "edge_weight": torch.tensor([1.0, 1.0]),
                    "available": True,
                    "source_hash": "synthetic-physical-binding-sha256",
                    "provenance": "synthetic:physical_binding",
                    "source_donors": [],
                },
            ],
        },
        static_path,
    )
    return static_path


def _episode_payload(
    fold: int, role: str, static_path: Path, *,
    cancer_id: str = "C1", support_limited: bool = False,
) -> dict:
    donor_count = 3 if support_limited else 5
    outer_train = [f"{cancer_id}_F{fold}_D{index}" for index in range(donor_count)]
    outer_query = [f"{cancer_id}_F{fold}_Q0"]
    validation = [outer_train[1]]
    validation_fold = (fold + 1) % trainer.N_OUTER_FOLDS
    if role == "fit" and support_limited:
        labels = [outer_train[0]]
        target = labels
        validation = []
        source = outer_train[1:]
        excluded = outer_query + target
        inner_fold = (fold + 1) % trainer.N_OUTER_FOLDS
        geometry_isolation = "outer_and_inner_target_fold"
    elif role == "fit":
        labels = [outer_train[0]]
        target = labels
        source = outer_train[2:]
        excluded = outer_query + target + validation
        inner_fold = (fold + 2) % trainer.N_OUTER_FOLDS
        geometry_isolation = "outer_and_inner_target_and_validation_fold"
    elif role == "validation":
        labels = validation
        target = validation
        source = [outer_train[0], *outer_train[2:]]
        excluded = outer_query + validation
        inner_fold = validation_fold
        geometry_isolation = "outer_and_inner_target_fold"
    elif role == "query":
        labels = outer_query
        target = []
        validation = []
        source = outer_train
        excluded = outer_query
        inner_fold = None
        geometry_isolation = "outer_fold_only"
    else:
        raise AssertionError(role)
    generator = torch.Generator().manual_seed(100 + fold)
    features = {
        "lncRNA": torch.randn(2, 3, generator=generator),
        "gene": torch.randn(3, 3, generator=generator),
        "pathway": torch.randn(2, 3, generator=generator),
        "pathway_family": torch.randn(2, 3, generator=generator),
        "cell_context": torch.randn(1, 4, generator=generator),
    }
    donor = labels[0]
    common_pair_features = torch.tensor(
        [[0.2, -0.1, 0.8], [0.7, 0.4, -0.2], [-0.5, 0.3, 0.1], [0.6, -0.2, 0.5]]
    )
    without_lasso = common_pair_features.clone()
    without_lasso[:, 0] = 0
    return {
        "format": trainer.EPISODE_FORMAT,
        "episode_id": f"{cancer_id}|fold={fold}|T|{role}",
        "cancer_id": cancer_id,
        "outer_fold_id": fold,
        "inner_fold_id": inner_fold,
        "role": role,
        "cell_key": "T",
        "label_donors": labels,
        "source_donors": source,
        "excluded_donors": excluded,
        "outer_train_donors": outer_train,
        "outer_query_donors": outer_query,
        "inner_target_donors": target,
        "validation_donors": validation,
        "validation_provider": bool(role == "validation"),
        "strict_validation_bound": bool(role == "fit" and validation),
        "support_limited_fit": bool(role == "fit" and not validation),
        "minimum_source_donors": 2 if role == "fit" and not validation else 3,
        "geometry_isolation": geometry_isolation,
        "geometry_source_donors": source,
        "geometry_excluded_donors": excluded,
        "feature_source_donors": source,
        "lasso_source_donors": source,
        "pair_feature_source_donors": source,
        "static_graph_asset": _record(static_path),
        "node_global_indices": {
            "lncRNA": torch.arange(2),
            "gene": torch.arange(3),
            "pathway": torch.arange(2),
            "pathway_family": torch.arange(2),
        },
        "features": features,
        "sparse_lasso_edges": {
            "response_index": torch.tensor([0, 1]),
            "gene_index": torch.tensor([0, 1]),
            "coefficient": torch.tensor([0.7, -0.6]),
            "stability": torch.tensor([0.9, 0.8]),
        },
        "lasso_stability_threshold": 0.5,
        "lnc_index": torch.tensor([0, 1, 0, 1]),
        "pathway_index": torch.tensor([0, 1, 1, 0]),
        "pair_features_by_arm": {
            "with_lasso": common_pair_features,
            "without_lasso": without_lasso,
        },
        "pair_feature_names": ["lasso_effect", "n_metacells", "dispersion"],
        "lasso_derived_pair_feature_mask": [True, False, False],
        "target_rho": torch.tensor([0.65, -0.55, 0.35, -0.25]),
        "row_metadata": {
            "dataset_id": [f"{cancer_id}_D1"] * 4,
            "cancer_id": [cancer_id] * 4,
            "cell_key": ["T"] * 4,
            "query_donor_id": [donor] * 4,
            "lncrna_id": ["L0", "L1", "L0", "L1"],
            "pathway_id": ["P0", "P1", "P1", "P0"],
        },
    }


def _write_episode_fixture(root: Path) -> tuple[Path, Path]:
    graph_root = root / "graph_assets"
    graph_manifest = graph_root / "GRAPH_ASSET_MANIFEST.json"
    _write_json(
        graph_manifest,
        {
            "format": trainer.GRAPH_ASSET_FORMAT,
            "state": "GRAPH_ASSETS_COMPLETE",
            "target_host": "149",
            "folds": [],
        },
    )
    static_path = _write_static_asset(root)
    limited_static_path = _write_static_asset(root, "C_LOW")
    episode_root = root / "episodes"
    folds = []
    for fold in range(5):
        records = []
        for role in trainer.ROLES:
            path = episode_root / f"outer_fold={fold}" / f"{role}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_episode_payload(fold, role, static_path), path)
            records.append(_record(path))
        limited_path = episode_root / f"outer_fold={fold}" / "C_LOW_support_limited_fit.pt"
        torch.save(
            _episode_payload(
                fold, "fit", limited_static_path,
                cancer_id="C_LOW", support_limited=True,
            ),
            limited_path,
        )
        records.append(_record(limited_path))
        folds.append({"outer_fold_id": fold, "episodes": records})
    _write_json(
        episode_root / "SC_HHGT_EPISODE_MANIFEST.json",
        {
            "format": trainer.EPISODE_MANIFEST_FORMAT,
            "state": "EPISODES_COMPLETE",
            "target_host": "149",
            "graph_asset_manifest": _record(graph_manifest),
            "expected_zero_query_cancer_folds": [
                {"cancer_id": "C_LOW", "outer_fold_id": fold}
                for fold in range(5)
            ],
            "folds": folds,
        },
    )
    return episode_root, graph_root


def _tiny_settings(episode_root: Path, graph_root: Path, output: Path):
    return trainer.TrainingSettings(
        episode_root=episode_root,
        graph_assets_root=graph_root,
        output_root=output,
        seed=7301,
        epochs=1,
        patience=1,
        prediction_batch_size=2,
        max_rows_per_episode_step=2,
        learning_rate=0.01,
        weight_decay=0.0,
        hidden_channels=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )


def test_host_gate_and_outer_only_fit_geometry_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="REQUIRES_HOST_149"):
        trainer.require_host_149("not-149")
    episode_root, graph_root = _write_episode_fixture(tmp_path)
    settings = _tiny_settings(episode_root, graph_root, tmp_path / "unused")
    descriptors, _ = trainer.load_episode_manifest(settings)
    fit = next(value for value in descriptors[0] if value.role == "fit")
    payload = torch.load(fit.path, map_location="cpu", weights_only=True)
    payload["geometry_isolation"] = "outer_fold_only"
    torch.save(payload, fit.path)
    changed = replace(fit, record=_record(fit.path))
    with pytest.raises(RuntimeError, match="TARGET_AND_VALIDATION_REFIT_GEOMETRY"):
        trainer.load_episode(changed, "with_lasso")


def test_large_training_inputs_use_receipt_metadata_and_single_load() -> None:
    manifest_source = inspect.getsource(trainer.load_episode_manifest)
    load_source = inspect.getsource(trainer.load_episode)
    assert "path, record = verify_file_record_metadata(" in manifest_source
    assert "sha256(descriptor.path)" not in load_source
    assert "sha256(descriptor.static_path)" not in load_source
    assert load_source.count("_load_weights_only(descriptor.path)") == 1


def test_episode_rejects_static_duplication_and_lasso_pair_leak(tmp_path):
    episode_root, graph_root = _write_episode_fixture(tmp_path)
    settings = _tiny_settings(episode_root, graph_root, tmp_path / "unused")
    descriptors, _ = trainer.load_episode_manifest(settings)
    fit = next(value for value in descriptors[0] if value.role == "fit")

    original_payload = torch.load(fit.path, map_location="cpu", weights_only=True)
    contract = validator._episode_input_contract(
        original_payload, _record(fit.path), {}
    )
    original_audit = trainer.load_episode(fit, "with_lasso").audit("with_lasso")
    validator._validate_audit_matches_input(original_audit, contract, "with_lasso", 0)
    forged_audit = dict(original_audit)
    forged_audit["source_donors"] = [*forged_audit["source_donors"], "FORGED_DONOR"]
    with pytest.raises(validator.IndependentValidationError, match="audit/input drift"):
        validator._validate_audit_matches_input(forged_audit, contract, "with_lasso", 0)

    duplicate = torch.load(fit.path, map_location="cpu", weights_only=True)
    duplicate["gene_pathway_membership"] = {"forbidden": True}
    torch.save(duplicate, fit.path)
    with pytest.raises(RuntimeError, match="DUPLICATES_SHARED_STATIC"):
        trainer._scan_descriptor(fit.path, _record(fit.path), 0, episode_root, {})

    clean = _episode_payload(0, "fit", fit.static_path)
    clean["pair_features_by_arm"]["without_lasso"][:, 0] = 0.25
    torch.save(clean, fit.path)
    with pytest.raises(RuntimeError, match="matched arm pair feature contract"):
        trainer._scan_descriptor(fit.path, _record(fit.path), 0, episode_root, {})


def test_independent_validator_replays_early_stopping():
    history = [
        {"validation_loss": 1.0, "selected_as_best": True, "stale_epochs": 0},
        {"validation_loss": 0.995, "selected_as_best": False, "stale_epochs": 1},
        {"validation_loss": 0.98, "selected_as_best": True, "stale_epochs": 0},
        {"validation_loss": 0.981, "selected_as_best": False, "stale_epochs": 1},
        {"validation_loss": 0.982, "selected_as_best": False, "stale_epochs": 2},
    ]
    assert validator._replay_model_selection(
        history, {"epochs": 5, "patience": 2, "minimum_validation_improvement": 0.01}
    ) == (2, 0.98)
    history[1]["selected_as_best"] = True
    with pytest.raises(validator.IndependentValidationError, match="cannot be replayed"):
        validator._replay_model_selection(
            history,
            {"epochs": 5, "patience": 2, "minimum_validation_improvement": 0.01},
        )


def test_five_fold_training_and_independent_validation_close(tmp_path, monkeypatch):
    episode_root, graph_root = _write_episode_fixture(tmp_path)
    output = tmp_path / "training"
    encode_calls = 0
    original_encode = SingleCellHHGT.encode_graph

    def counted_encode(self, graph, *, message_passing=True):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(self, graph, message_passing=message_passing)

    monkeypatch.setattr(SingleCellHHGT, "encode_graph", counted_encode)
    result = trainer.run_training(
        _tiny_settings(episode_root, graph_root, output), hostname="149"
    )
    assert result["state"] == "TRAINING_COMPLETE"
    assert result["trained_model_count"] == 10
    assert encode_calls == 40  # two fit graphs plus validation/query, each encoded once

    manifest = json.loads((output / "CHECKPOINT_MANIFEST.json").read_text())
    assert manifest["model_count"] == 10
    for fold in range(5):
        records = {
            row["arm"]: row
            for row in manifest["models"]
            if row["outer_fold_id"] == fold
        }
        assert records["with_lasso"]["initial_state_sha256"] == records["without_lasso"]["initial_state_sha256"]
        assert records["with_lasso"]["final_hgt_state_sha256"] != records["with_lasso"]["initial_hgt_state_sha256"]
        for arm in trainer.ARMS:
            report_path = output / f"arm={arm}" / f"outer_fold={fold}" / "TRAINING_REPORT.json"
            report = json.loads(report_path.read_text())
            assert report["model_selection_source"] == "inner_validation_only"
            assert report["maximum_live_episode_graphs"] == 1
            assert report["history"][0]["sampling_coverage"] == 0.5
            assert {row["role"] for row in report["episode_audits"]} == {
                "fit", "validation", "query"
            }
            if arm == "without_lasso":
                assert all(
                    row["lasso_pair_feature_nonzero_values"] == 0
                    for row in report["episode_audits"]
                )
            limited = [
                row for row in report["episode_audits"]
                if row["cancer_id"] == "C_LOW"
            ]
            assert len(limited) == 1
            assert limited[0]["role"] == "fit"
            assert limited[0]["support_limited_fit"] is True
            assert limited[0]["validation_provider"] is False
            assert limited[0]["strict_validation_bound"] is False
            assert limited[0]["validation_donors"] == []
            assert limited[0]["geometry_isolation"] == "outer_and_inner_target_fold"
            strict_fit = [
                row for row in report["episode_audits"]
                if row["cancer_id"] == "C1" and row["role"] == "fit"
            ][0]
            assert strict_fit["validation_provider"] is False
            assert strict_fit["strict_validation_bound"] is True
            query_audit = [
                row for row in report["episode_audits"] if row["role"] == "query"
            ][0]
            assert query_audit["validation_donors"] == []
    assert len(
        {
            row["initial_state_sha256"]
            for row in manifest["models"]
            if row["arm"] == "with_lasso"
        }
    ) == 5

    sample_episode = torch.load(
        episode_root / "outer_fold=0" / "fit.pt", map_location="cpu", weights_only=True
    )
    assert STATIC_FIELDS.isdisjoint(sample_episode)
    assert isinstance(sample_episode["static_graph_asset"], dict)


def test_fit_episode_coverage_precedes_model_selection(tmp_path):
    episode_root, graph_root = _write_episode_fixture(tmp_path)
    output = tmp_path / "coverage_training"
    settings = replace(
        _tiny_settings(episode_root, graph_root, output),
        epochs=2,
        max_fit_episodes_per_epoch=1,
    )
    trainer.run_training(settings, hostname="149")
    report = json.loads(
        (
            output / "arm=with_lasso" / "outer_fold=0" / "TRAINING_REPORT.json"
        ).read_text()
    )
    history = report["history"]
    assert history[0]["model_selection_eligible"] is False
    assert history[0]["validation_loss"] is None
    assert history[0]["validation_rows"] == 0
    assert history[0]["cumulative_fit_episode_coverage"] == 0.5
    assert history[1]["model_selection_eligible"] is True
    assert history[1]["selected_as_best"] is True
    assert history[1]["cumulative_fit_episode_coverage"] == 1.0
    result = validator.validate_training(output, hostname="149")
    assert result["status"] == "PASS"


def test_parallel_fold_workers_close(tmp_path):
    episode_root, graph_root = _write_episode_fixture(tmp_path)
    output = tmp_path / "parallel_training"
    settings = replace(
        _tiny_settings(episode_root, graph_root, output),
        training_workers=2,
        threads_per_worker=1,
    )
    result = trainer.run_training(settings, hostname="149")
    assert result["trained_model_count"] == 10
    assert result["oof_prediction_rows"] == 40
    assert validator.validate_training(output, hostname="149")["status"] == "PASS"

    audit = validator.run_validation(output, hostname="149")
    assert audit["status"] == "PASS"
    assert audit["validated_model_count"] == 10
    assert audit["strict_inner_validation_model_selection"] is True
    assert audit["shared_static_graph_assets"] is True
    assert audit["one_graph_encode_per_episode_step"] is True
    assert audit["arm_specific_lasso_pair_features"] is True
    assert audit["support_limited_fit_without_validation_selection"] is True

    prediction = output / "arm=with_lasso" / "outer_fold=0" / "QUERY_PREDICTIONS.parquet"
    prediction.write_bytes(prediction.read_bytes() + b"tamper")
    with pytest.raises(validator.IndependentValidationError, match="hash drift|checksum drift"):
        validator.validate_training(output, hostname="149")


def test_validator_does_not_import_training_implementation():
    source = inspect.getsource(validator)
    assert "import train_single_cell_hhgt_149" not in source
    assert "from scripts.train_single_cell_hhgt_149" not in source
