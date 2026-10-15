#!/usr/bin/env python3
"""Audit and atomically promote the completed 28-cancer SC-HHGT run on host 149."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KIRC", "LAML", "LGG", "LUAD", "LUSC", "MESO",
    "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THYM", "UCEC", "UVM",
)
ARMS = ("with_lasso", "without_lasso")
EXPECTED_SOURCE_TOTALS = {
    "eligible_donors": 383,
    "donors_with_oof_rows": 343,
    "metacells": 1_415_989,
    "query_rows": 10_593_692,
    "query_unique_pairs": 1_672_090,
    "new_query_rows": 2_240_711,
    "active_outer_folds": 139,
}
AUDIT_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_CLOSURE_AUDIT_V1"
POINTER_FORMAT = "CANCERLNCATLAS_V32_CURRENT_SINGLE_CELL_HEAD_V2"
ACTIVATION_FORMAT = "CANCERLNCATLAS_V32_SC_HHGT_ACTIVATION_V1"
EVENT_ID = "V32-SINGLE-CELL-SC-HHGT-ACTIVATED-20261005-R1"


class ClosureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"MISSING_{label}:{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClosureError(f"INVALID_{label}:{path}:{error}") from error
    require(isinstance(value, dict), f"NON_OBJECT_{label}:{path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"MISSING_FILE:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_record(
    record: Any,
    expected: Path,
    label: str,
    *,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"INVALID_RECORD_{label}")
    path = Path(str(record.get("path", ""))).resolve()
    expected = expected.resolve()
    require(path == expected, f"PATH_DRIFT_{label}:{path}:{expected}")
    if allowed_root is not None:
        root = allowed_root.resolve()
        require(path == root or root in path.parents, f"PATH_ESCAPE_{label}:{path}")
    observed = file_record(path)
    require(
        observed["sha256"] == record.get("sha256")
        and observed["bytes"] == int(record.get("bytes", -1)),
        f"HASH_OR_SIZE_DRIFT_{label}:{path}",
    )
    return observed


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def require_host_149(hostname: str | None = None) -> None:
    observed = hostname or socket.gethostname()
    require(observed == "149", f"SC_HHGT_CLOSURE_REQUIRES_HOST_149:{observed}")


def audit_source(root: Path, expected_validation_sha256: str) -> dict[str, Any]:
    validation_path = root / "MATERIALIZATION_VALIDATION.json"
    require(sha256(validation_path) == expected_validation_sha256, "SOURCE_VALIDATION_SHA_DRIFT")
    value = read_json(validation_path, "SOURCE_VALIDATION")
    require(
        value.get("format") == "CANCERLNCATLAS_V32_SC_METACELL_MATERIALIZATION_VALIDATION_V1"
        and value.get("status") == "PASS"
        and value.get("closure_status") == "PASS_FULL_28_CANCER_MATERIALIZATION_CLOSURE"
        and value.get("target_host") == "149",
        "SOURCE_VALIDATION_CONTRACT_FAILED",
    )
    require(tuple(value.get("validated_cancers", ())) == CANCERS, "SOURCE_CANCER_UNIVERSE_DRIFT")
    rows = value.get("cancers")
    require(isinstance(rows, list) and len(rows) == len(CANCERS), "SOURCE_CANCER_ROWS_FAILED")
    by_cancer = {str(item.get("cancer_id")): item for item in rows}
    require(tuple(sorted(by_cancer)) == CANCERS, "SOURCE_CANCER_ROWS_DRIFT")
    totals = {
        "eligible_donors": sum(int(item["eligible_donors"]) for item in rows),
        "donors_with_oof_rows": sum(int(item["donors_with_oof_rows"]) for item in rows),
        "metacells": sum(int(fold["metacells"]) for item in rows for fold in item["folds"]),
        "query_rows": sum(int(item["query_rows"]) for item in rows),
        "query_unique_pairs": sum(int(item["query_unique_pairs"]) for item in rows),
        "new_query_rows": sum(int(item["new_query_rows"]) for item in rows),
        "active_outer_folds": sum(len(item["folds"]) for item in rows),
    }
    require(totals == EXPECTED_SOURCE_TOTALS, f"SOURCE_TOTALS_DRIFT:{totals}")
    launch = value.get("launch_preflight")
    require(isinstance(launch, Mapping), "SOURCE_LAUNCH_PREFLIGHT_ABSENT")
    bindings = launch.get("bindings")
    require(isinstance(bindings, Mapping), "SOURCE_LAUNCH_BINDINGS_ABSENT")
    preprocessing_record = bindings.get("preprocessing_binding")
    require(isinstance(preprocessing_record, Mapping), "PREPROCESSING_BINDING_ABSENT")
    preprocessing_path = Path(str(preprocessing_record.get("path", "")))
    verify_record(
        preprocessing_record, preprocessing_path, "PREPROCESSING_BINDING"
    )
    preprocessing = read_json(preprocessing_path, "PREPROCESSING_BINDING")
    preprocessing_cancers = preprocessing.get("cancers")
    allowed_annotation_origins = {
        "VERIFIED_HISTORICAL_HARMONY_SINGLER_EXACT_CELLS",
        "VERIFIED_EXISTING_SEURAT_HARMONY_AND_REFERENCE_LABELS",
        "RECOMPUTED_CURRENT_COHORT",
    }
    require(
        preprocessing.get("format") == "SC_PREPROCESSING_TRAINING_BINDING_V1"
        and preprocessing.get("status") == "PASS"
        and preprocessing.get("target_host") == "149"
        and preprocessing.get("expression_policy")
        == "UNCORRECTED_COUNTS_LOGNORMALIZED_FOR_UCELL_HARMONY_FOR_EMBEDDINGS_AND_ANNOTATION"
        and preprocessing.get("source_celltype_and_malignancy_preserved") is True
        and isinstance(preprocessing_cancers, list)
        and tuple(sorted(str(item.get("cancer_id")) for item in preprocessing_cancers))
        == CANCERS
        and all(
            item.get("annotation_origin") in allowed_annotation_origins
            for item in preprocessing_cancers
        ),
        "PREPROCESSING_BINDING_CONTRACT_FAILED",
    )
    folds: dict[str, list[int]] = {}
    for cancer in CANCERS:
        item = by_cancer[cancer]
        success = root / "cancers" / cancer / "SUCCESS.json"
        require(sha256(success) == item.get("success_sha256"), f"SOURCE_SUCCESS_DRIFT:{cancer}")
        require(Path(str(item.get("success_path", ""))).resolve() == success.resolve(), f"SOURCE_PATH_DRIFT:{cancer}")
        fold_ids = sorted(int(fold["outer_fold_id"]) for fold in item["folds"])
        require(len(fold_ids) == len(set(fold_ids)) and set(fold_ids).issubset(range(5)), f"SOURCE_FOLDS_FAILED:{cancer}")
        folds[cancer] = fold_ids
    return {
        "validation": file_record(validation_path),
        "cancers": list(CANCERS),
        "totals": totals,
        "folds": folds,
        "preprocessing": {
            "binding": file_record(preprocessing_path),
            "within_cancer_harmony_embedding": True,
            "cell_annotation": True,
            "expression_policy": preprocessing["expression_policy"],
            "source_celltype_and_malignancy_preserved": True,
            "annotation_origin_counts": {
                origin: sum(
                    item.get("annotation_origin") == origin
                    for item in preprocessing_cancers
                )
                for origin in sorted(allowed_annotation_origins)
            },
        },
    }


def _audit_complete_batch(root: Path, label: str) -> dict[str, Any]:
    path = root / "batch_supervisor" / "STATUS.json"
    value = read_json(path, f"{label}_BATCH_STATUS")
    require(
        value.get("state") == "COMPLETE" and int(value.get("failures", -1)) == 0
        and value.get("target_host") == "149",
        f"{label}_BATCH_NOT_COMPLETE",
    )
    states = value.get("states")
    require(isinstance(states, Mapping) and tuple(sorted(states)) == CANCERS, f"{label}_BATCH_CANCERS_DRIFT")
    require(
        all(str(item.get("state")) in {"COMPLETE", "ALREADY_COMPLETE"} for item in states.values()),
        f"{label}_BATCH_HAS_INCOMPLETE_CANCER",
    )
    return file_record(path)


def audit_graph(root: Path, source_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    batch = _audit_complete_batch(root, "GRAPH")
    preflight_path = root / "LAUNCH_PREFLIGHT.json"
    preflight = read_json(preflight_path, "GRAPH_PREFLIGHT")
    require(
        preflight.get("format") == "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSET_PREFLIGHT_V1"
        and preflight.get("target_host") == "149"
        and preflight.get("cpu_only") is True
        and preflight.get("paid_compute") is False
        and Path(str(preflight.get("source_metacell_result_root", ""))).resolve() == source_root.resolve()
        and preflight.get("source_validation_sha256") == source["validation"]["sha256"],
        "GRAPH_PREFLIGHT_CONTRACT_FAILED",
    )
    for key in ("launcher", "materializer", "graph_asset_module", "frozen_metacell_materializer"):
        record = preflight.get(key)
        require(isinstance(record, Mapping), f"GRAPH_PREFLIGHT_MISSING_{key}")
        verify_record(record, Path(str(record.get("path", ""))), f"GRAPH_PREFLIGHT_{key}")
    cancer_receipts = []
    for cancer in CANCERS:
        path = root / "cancers" / cancer / "SUCCESS.json"
        receipt = read_json(path, f"GRAPH_CANCER_{cancer}")
        require(
            receipt.get("format") == "CANCERLNCATLAS_V32_SC_HHGT_CANCER_SUCCESS_V1"
            and receipt.get("state") == "CANCER_GRAPH_ASSETS_COMPLETE"
            and receipt.get("target_host") == "149"
            and receipt.get("cancer_id") == cancer,
            f"GRAPH_CANCER_CONTRACT_FAILED:{cancer}",
        )
        expected_folds = list(source["folds"][cancer])
        require(receipt.get("outer_fold_ids") == expected_folds, f"GRAPH_FOLD_SET_DRIFT:{cancer}")
        static_path = root / "cancers" / cancer / "STATIC" / "SUCCESS.json"
        verify_record(receipt.get("cancer_static_success"), static_path, f"GRAPH_STATIC_{cancer}", allowed_root=root)
        static = read_json(static_path, f"GRAPH_STATIC_{cancer}")
        require(
            static.get("state") == "CANCER_STATIC_GRAPH_ASSETS_READY"
            and static.get("cancer_id") == cancer
            and int(static.get("pathways_without_family_edges", -1)) == 0
            and int(static.get("pathway_family_edges", -1)) == int(static.get("pathways", -2))
            and int(static.get("primary_external_edges", 0)) > 0,
            f"GRAPH_STATIC_CONTRACT_FAILED:{cancer}",
        )
        for name, record in static.get("files", {}).items():
            verify_record(record, root / "cancers" / cancer / "STATIC" / name, f"GRAPH_STATIC_FILE_{cancer}_{name}", allowed_root=root)
        records = receipt.get("outer_fold_successes")
        require(isinstance(records, list) and len(records) == len(expected_folds), f"GRAPH_FOLD_RECORDS_FAILED:{cancer}")
        for fold, record in zip(expected_folds, records):
            fold_path = root / "cancers" / cancer / f"outer_fold={fold}" / "SUCCESS.json"
            verify_record(record, fold_path, f"GRAPH_FOLD_{cancer}_{fold}", allowed_root=root)
            fold_receipt = read_json(fold_path, f"GRAPH_FOLD_{cancer}_{fold}")
            require(
                fold_receipt.get("state") == "FOLD_GRAPH_ASSETS_READY"
                and fold_receipt.get("cancer_id") == cancer
                and int(fold_receipt.get("outer_fold_id", -1)) == fold
                and fold_receipt.get("geometry_isolation") == "outer_fold_only"
                and fold_receipt.get("lasso_coefficients_computed") is False
                and set(fold_receipt.get("source_donors", ())).isdisjoint(fold_receipt.get("query_donors", ())),
                f"GRAPH_FOLD_CONTRACT_FAILED:{cancer}:{fold}",
            )
        cancer_receipts.append(file_record(path))
    manifest_path = root / "GRAPH_ASSET_MANIFEST.json"
    manifest = read_json(manifest_path, "GRAPH_MANIFEST")
    require(
        manifest.get("format") == "CANCERLNCATLAS_V32_SC_HHGT_GRAPH_ASSETS_V1"
        and manifest.get("state") == "GRAPH_ASSETS_COMPLETE"
        and tuple(manifest.get("cancers", ())) == CANCERS
        and int(manifest.get("cancer_count", -1)) == 28
        and int(manifest.get("fold_count", -1)) == EXPECTED_SOURCE_TOTALS["active_outer_folds"],
        "GRAPH_MANIFEST_CONTRACT_FAILED",
    )
    return {"batch_status": batch, "preflight": file_record(preflight_path), "manifest": file_record(manifest_path), "cancer_receipts": cancer_receipts}


def audit_nested(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    batch = _audit_complete_batch(root, "NESTED")
    receipts = []
    for cancer in CANCERS:
        path = root / "cancers" / cancer / "SUCCESS.json"
        value = read_json(path, f"NESTED_CANCER_{cancer}")
        records = value.get("outer_fold_receipts")
        require(
            value.get("state") == "CANCER_NESTED_ASSETS_COMPLETE"
            and value.get("target_host") == "149"
            and value.get("cancer_id") == cancer
            and isinstance(records, list)
            and len(records) == len(source["folds"][cancer]),
            f"NESTED_CANCER_CONTRACT_FAILED:{cancer}",
        )
        for fold, record in zip(source["folds"][cancer], records):
            fold_path = root / "cancers" / cancer / f"outer_fold={fold}" / "SUCCESS.json"
            verify_record(record, fold_path, f"NESTED_FOLD_{cancer}_{fold}", allowed_root=root)
            fold_receipt = read_json(fold_path, f"NESTED_FOLD_{cancer}_{fold}")
            require(
                fold_receipt.get("state") == "OUTER_FOLD_NESTED_ASSETS_COMPLETE"
                and int(fold_receipt.get("outer_fold_id", -1)) == fold
                and isinstance(fold_receipt.get("episode_receipts"), list)
                and fold_receipt["episode_receipts"],
                f"NESTED_FOLD_CONTRACT_FAILED:{cancer}:{fold}",
            )
        receipts.append(file_record(path))
    return {"batch_status": batch, "cancer_receipts": receipts}


def audit_episodes(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    batch = _audit_complete_batch(root, "EPISODE")
    receipts = []
    for cancer in CANCERS:
        path = root / "cancers" / cancer / "SUCCESS.json"
        value = read_json(path, f"EPISODE_CANCER_{cancer}")
        require(
            value.get("state") == "CANCER_EPISODES_COMPLETE"
            and value.get("target_host") == "149"
            and value.get("cancer_id") == cancer
            and int(value.get("episode_count", 0)) > 0
            and int(value.get("source_query_row_count", -1)) == int(
                next(item["query_rows"] for item in source["source_cancer_rows"] if item["cancer_id"] == cancer)
            ),
            f"EPISODE_CANCER_CONTRACT_FAILED:{cancer}",
        )
        receipts.append(file_record(path))
    manifest_path = root / "SC_HHGT_EPISODE_MANIFEST.json"
    manifest = read_json(manifest_path, "EPISODE_MANIFEST")
    require(
        manifest.get("format") == "CANCERLNCATLAS_V32_SC_HHGT_EPISODE_MANIFEST_V1"
        and manifest.get("state") == "EPISODES_COMPLETE"
        and manifest.get("target_host") == "149"
        and tuple(manifest.get("cancers", ())) == CANCERS
        and int(manifest.get("cancer_count", -1)) == 28
        and int(manifest.get("outer_folds", -1)) == 5
        and int(manifest.get("role_row_counts", {}).get("query", -1)) == EXPECTED_SOURCE_TOTALS["query_rows"]
        and set(manifest.get("role_counts", {})) == {"fit", "validation", "query"},
        "EPISODE_MANIFEST_CONTRACT_FAILED",
    )
    folds = manifest.get("folds")
    require(isinstance(folds, list) and [int(item.get("outer_fold_id", -1)) for item in folds] == list(range(5)), "EPISODE_FOLD_GRID_FAILED")
    return {"batch_status": batch, "manifest": file_record(manifest_path), "cancer_receipts": receipts, "episode_count": int(manifest["episode_count"]), "role_counts": manifest["role_counts"], "role_row_counts": manifest["role_row_counts"]}


def audit_training(root: Path, episode: Mapping[str, Any]) -> dict[str, Any]:
    success_path = root / "SUCCESS.json"
    success = read_json(success_path, "TRAINING_SUCCESS")
    require(
        success.get("format") == "CANCERLNCATLAS_V32_SC_HHGT_SUCCESS_V1"
        and success.get("state") == "TRAINING_COMPLETE"
        and success.get("target_host") == "149"
        and success.get("arms") == list(ARMS)
        and int(success.get("outer_folds", -1)) == 5
        and int(success.get("trained_model_count", -1)) == 10
        and int(success.get("oof_prediction_rows", -1)) == 2 * EXPECTED_SOURCE_TOTALS["query_rows"],
        "TRAINING_SUCCESS_CONTRACT_FAILED",
    )
    config_path = root / "RUN_CONFIG.json"
    config = read_json(config_path, "TRAINING_CONFIG")
    architecture = config.get("architecture", {})
    loss = config.get("loss", {})
    policy = config.get("episode_policy", {})
    require(
        config.get("target_host") == "149" and config.get("device") == "cpu"
        and config.get("cuda_used") is False
        and architecture.get("family") == "SC_HHGT_WEIGHTED_HGTCONV"
        and architecture.get("cell_context_node") is True
        and architecture.get("signed_edge_weight_path") is True
        and architecture.get("frozen_bulk_core") is False
        and architecture.get("mlp_only") is False
        and loss.get("target") == "signed_within_donor_metacell_spearman_rho"
        and policy.get("outer_query_excluded_from_all_fitting") is True,
        "TRAINING_CONFIG_CONTRACT_FAILED",
    )
    checkpoint_path = root / "CHECKPOINT_MANIFEST.json"
    checkpoint = read_json(checkpoint_path, "CHECKPOINT_MANIFEST")
    models = checkpoint.get("models")
    require(
        checkpoint.get("status") == "SUCCESS" and int(checkpoint.get("model_count", -1)) == 10
        and isinstance(models, list) and len(models) == 10,
        "CHECKPOINT_MANIFEST_CONTRACT_FAILED",
    )
    observed = {(str(item.get("arm")), int(item.get("outer_fold_id", -1))) for item in models}
    require(observed == {(arm, fold) for arm in ARMS for fold in range(5)}, "CHECKPOINT_ARM_FOLD_GRID_FAILED")
    require(
        all(
            item.get("initial_state_sha256") != item.get("final_state_sha256")
            and item.get("initial_hgt_state_sha256") != item.get("final_hgt_state_sha256")
            for item in models
        ),
        "CHECKPOINT_PARAMETERS_UNCHANGED",
    )
    validation_path = root / "INDEPENDENT_VALIDATION.json"
    validation = read_json(validation_path, "INDEPENDENT_VALIDATION")
    required_true = (
        "all_hgt_gradients_positive", "all_hgt_parameters_changed",
        "fresh_fold_initializations_independent", "ablation_initializations_matched",
        "outer_query_donor_isolation", "inner_training_donor_isolation",
        "nested_refit_training_geometry", "strict_inner_validation_model_selection",
        "arm_specific_lasso_pair_features", "external_evidence_used",
        "signed_rho_smooth_l1",
    )
    require(
        validation.get("format") == "CANCERLNCATLAS_V32_SC_HHGT_INDEPENDENT_VALIDATION_V1"
        and validation.get("status") == "PASS"
        and validation.get("target_host") == "149"
        and Path(str(validation.get("training_root", ""))).resolve() == root.resolve()
        and validation.get("training_success_sha256") == sha256(success_path)
        and int(validation.get("validated_model_count", -1)) == 10
        and int(validation.get("prediction_rows", -1)) == 2 * EXPECTED_SOURCE_TOTALS["query_rows"]
        and validation.get("frozen_bulk_core_mlp_used") is False
        and all(validation.get(key) is True for key in required_true),
        "INDEPENDENT_VALIDATION_CONTRACT_FAILED",
    )
    return {
        "success": file_record(success_path), "run_config": file_record(config_path),
        "checkpoint_manifest": file_record(checkpoint_path),
        "independent_validation": file_record(validation_path),
        "independent_validator_source": file_record(
            Path(__file__).resolve().parent / "validate_single_cell_hhgt_training_149.py"
        ),
        "trained_model_count": 10,
        "oof_prediction_rows": int(success["oof_prediction_rows"]),
        "primary_arm": "with_lasso", "ablation_arm": "without_lasso",
        "episode_manifest": episode["manifest"],
    }


def run_audit(args: argparse.Namespace, *, hostname: str | None = None) -> dict[str, Any]:
    require_host_149(hostname)
    source = audit_source(args.source_root, args.expected_source_validation_sha256)
    source["source_cancer_rows"] = read_json(
        args.source_root / "MATERIALIZATION_VALIDATION.json", "SOURCE_VALIDATION"
    )["cancers"]
    graph = audit_graph(args.graph_root, args.source_root, source)
    nested = audit_nested(args.nested_root, source)
    episode = audit_episodes(args.episode_root, source)
    training = audit_training(args.training_root, episode)
    source.pop("source_cancer_rows")
    result = {
        "format": AUDIT_FORMAT, "status": "PASS_READY_FOR_FORMAL_PROMOTION",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "target_host": "149",
        "cpu_only": True, "cuda_used": False, "paid_compute": False,
        "architecture_family": "SC_HHGT_WEIGHTED_HGTCONV",
        "estimand": "signed_within_donor_metacell_spearman_rho",
        "old_mlp_promoted": False, "formal_cancers": list(CANCERS),
        "source": source, "graph": graph, "nested": nested,
        "episodes": episode, "training": training,
        "closure_auditor": file_record(Path(__file__).resolve()),
    }
    args.audit_root.mkdir(parents=True, exist_ok=True)
    audit_path = args.audit_root / "SC_HHGT_CLOSURE_AUDIT.json"
    if audit_path.exists():
        require(read_json(audit_path, "EXISTING_CLOSURE_AUDIT") == result, "CLOSURE_AUDIT_ALREADY_EXISTS_WITH_DRIFT")
    else:
        atomic_json(audit_path, result)
    return result


def build_pointer(args: argparse.Namespace, audit: Mapping[str, Any], audit_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": POINTER_FORMAT,
        "status": "PASS_ACTIVATED_SC_HHGT",
        "target_host": "149",
        "activated_at_utc": audit["audited_at_utc"],
        "activation_event_id": EVENT_ID,
        "architecture_family": "SC_HHGT_WEIGHTED_HGTCONV",
        "architecture": {
            "trainable_hgtconv": True, "cell_context_node": True,
            "weighted_lasso_edges": True, "external_evidence_edges": True,
            "signed_edge_weight_path": True, "frozen_bulk_core": False,
            "mlp_only": False,
        },
        "estimand": "signed_within_donor_metacell_spearman_rho",
        "observation_unit": "non_overlapping_metacells_within_donor_and_cell_context",
        "split_policy": "donor_blocked_outer_oof_with_nested_inner_validation",
        "qualified_cancers": list(CANCERS),
        "training_cancers": list(CANCERS),
        "prediction_cancers": list(CANCERS),
        "training_started": True, "training_complete": True,
        "trained_model_count": 10, "outer_folds": 5,
        "primary_arm": "with_lasso", "ablation_arm": "without_lasso",
        "cpu_only": True, "cuda_used": False, "paid_compute": False,
        "old_mlp_promoted": False, "backup_created": False,
        "code_root": str(Path(__file__).resolve().parents[1]),
        "launcher": str(Path(__file__).resolve().parent / "train_single_cell_hhgt_149.py"),
        "inputs": str(args.episode_root.resolve()),
        "input_manifest": audit["episodes"]["manifest"],
        "input_manifest_sha256": audit["episodes"]["manifest"]["sha256"],
        "result_root": str(args.training_root.resolve()),
        "training_success": audit["training"]["success"],
        "independent_validation": audit["training"]["independent_validation"],
        "oof_predictions": file_record(args.training_root / "OOF_PREDICTIONS.parquet"),
        "closure_audit": dict(audit_record),
        "previous_pointer_sha256": args.expected_old_pointer_sha256,
        "source_summary": dict(audit["source"]["totals"]),
        "preprocessing": dict(audit["source"]["preprocessing"]),
    }


def build_log_event(pointer: Mapping[str, Any], audit_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "timestamp_utc": pointer["activated_at_utc"],
        "event_id": EVENT_ID,
        "version": "V3.2",
        "event": "SINGLE_CELL_SC_HHGT_28_CANCER_FIVE_FOLD_TRAINED_AND_ACTIVATED",
        "status": "PASS_ACTIVATED",
        "authority_host": "149",
        "architecture_family": "SC_HHGT_WEIGHTED_HGTCONV",
        "formal_cancers": 28, "outer_folds": 5, "trained_models": 10,
        "primary_arm": "with_lasso", "ablation_arm": "without_lasso",
        "estimand": pointer["estimand"],
        "observation_unit": pointer["observation_unit"],
        "donor_oof": True, "cell_context_node": True,
        "weighted_lasso_edges": True, "external_evidence_edges": True,
        "hgt_backpropagation_audited": True,
        "old_mlp_promoted": False,
        "cpu_only": True, "cuda_used": False, "paid_compute": False,
        "source_summary": pointer["source_summary"],
        "preprocessing": pointer["preprocessing"],
        "training_success": pointer["training_success"],
        "independent_validation": pointer["independent_validation"],
        "closure_audit": dict(audit_record),
        "previous_pointer_sha256": pointer["previous_pointer_sha256"],
        "backup_created": False,
    }


def publish(
    pointer_path: Path,
    log_path: Path,
    expected_old_pointer_sha256: str,
    pointer: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[str, str]:
    old_pointer = pointer_path.read_bytes()
    require(hashlib.sha256(old_pointer).hexdigest() == expected_old_pointer_sha256, "FORMAL_POINTER_CHANGED_SINCE_PREFLIGHT")
    old_log = log_path.read_bytes()
    require(old_log.endswith(b"\n"), "VERSION_LOG_MISSING_FINAL_NEWLINE")
    entries = [json.loads(line) for line in old_log.decode("utf-8").splitlines() if line.strip()]
    require(all(item.get("event_id") != EVENT_ID for item in entries), "ACTIVATION_EVENT_ALREADY_EXISTS")
    pointer_bytes = (json.dumps(pointer, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    event_bytes = (json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    atomic_bytes(pointer_path, pointer_bytes)
    try:
        atomic_bytes(log_path, old_log + event_bytes)
    except BaseException:
        atomic_bytes(pointer_path, old_pointer)
        raise
    pointer_sha = hashlib.sha256(pointer_bytes).hexdigest()
    log_sha = sha256(log_path)
    require(sha256(pointer_path) == pointer_sha, "POINTER_POSTCOMMIT_HASH_FAILED")
    return pointer_sha, log_sha


def promote(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    audit_path = args.audit_root / "SC_HHGT_CLOSURE_AUDIT.json"
    audit_record = file_record(audit_path)
    pointer = build_pointer(args, audit, audit_record)
    event = build_log_event(pointer, audit_record)
    lock = args.audit_root / ".formal_promotion.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ClosureError(f"PROMOTION_LOCK_EXISTS:{lock}") from error
    try:
        pointer_sha, log_sha = publish(
            args.pointer, args.version_log, args.expected_old_pointer_sha256,
            pointer, event,
        )
        activation = {
            "format": ACTIVATION_FORMAT, "status": "PASS_ACTIVATED",
            "activated_at_utc": pointer["activated_at_utc"], "target_host": "149",
            "event_id": EVENT_ID, "pointer": file_record(args.pointer),
            "version_log_path": str(args.version_log.resolve()),
            "version_log_sha256": log_sha, "pointer_sha256": pointer_sha,
            "closure_audit": audit_record, "old_pointer_sha256": args.expected_old_pointer_sha256,
            "backup_created": False, "old_mlp_promoted": False,
        }
        atomic_json(args.audit_root / "ACTIVATION_RECEIPT.json", activation)
        return activation
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--graph-root", type=Path, required=True)
    result.add_argument("--nested-root", type=Path, required=True)
    result.add_argument("--episode-root", type=Path, required=True)
    result.add_argument("--training-root", type=Path, required=True)
    result.add_argument("--audit-root", type=Path, required=True)
    result.add_argument("--pointer", type=Path, required=True)
    result.add_argument("--version-log", type=Path, required=True)
    result.add_argument("--expected-source-validation-sha256", required=True)
    result.add_argument("--expected-old-pointer-sha256", required=True)
    result.add_argument("--promote", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    audit = run_audit(args)
    if args.promote:
        output = promote(args, audit)
    else:
        output = audit
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
