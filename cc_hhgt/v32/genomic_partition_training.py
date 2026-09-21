"""Resource-bounded global Mutation/CNV heads over cancer partitions.

The trainer preserves the original cross-cancer sampling algorithm exactly,
but never holds the 3.3-million-row candidate string table or ten full fold
prediction arrays in process memory.  A resource-stage SUCCESS is only an
input to this module and is never treated as genomic OOF success.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .genomic_resource_staging import (
    MODALITIES,
    TARGET_KEYS,
    GenomicResourceError,
    _available_memory_bytes,
    _load_stage,
    _process_rss_bytes,
    artifact_sha256,
    seal_genomic_memmap_predictions,
    write_genomic_memmap_partition,
)
from .genomic_training import (
    ANALYSIS_VERSION,
    MODULE_ID,
    N_FOLDS,
    PRIVATE_CHECKPOINT_FORMAT,
    _DOMAIN_FEATURES,
    GenomicTrainingConfig,
    GenomicTrainingError,
    _assert_source_name,
    _atomic_json,
    _audit_sources,
    _balanced_indices,
    _candidate_core,
    _file_sha256,
    _fit_head,
    _group_calls,
    _predict_head,
    _read_table,
    _split_patients,
    _validate_core_manifest,
    build_exact_pathway_calls,
    candidate_statistics,
    exact_pathway_callable_entities,
    load_fold_core_embeddings,
    normalise_candidates,
    normalise_cnv_calls,
    normalise_cnv_entity_coverage,
    normalise_exact_membership,
    normalise_gene_mutation_calls,
    normalise_lncrna_mutation_calls,
    normalise_patient_folds,
)


TRAINING_FORMAT = "CC_HHGT_V3_2_PARTITIONED_GLOBAL_GENOMIC_TRAINING_V1"
CHECKPOINT_MANIFEST_FORMAT = (
    "CC_HHGT_V3_2_PARTITIONED_GLOBAL_GENOMIC_CHECKPOINT_MANIFEST_V1"
)
SPLIT_POLICY_FORMAT = "CC_HHGT_V3_2_PARTITIONED_GLOBAL_GENOMIC_SPLIT_POLICY_V1"


class PartitionedGenomicTrainingError(RuntimeError):
    """Raised when partition training or its resource/leakage gates fail."""


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _partition_frame(record: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or artifact_sha256(path) != record.get("sha256"):
        raise PartitionedGenomicTrainingError(
            f"Candidate partition hash drift: {record.get('cancer_id')}"
        )
    frame = normalise_candidates(pd.read_parquet(path, columns=list(TARGET_KEYS)))
    cancer = str(record.get("cancer_id", "")).upper()
    if not frame.cancer_id.eq(cancer).all() or len(frame) != int(record.get("rows", -1)):
        raise PartitionedGenomicTrainingError(
            f"Candidate partition scope drift: {cancer}"
        )
    return frame


def collect_examples_partitioned(
    candidate_partitions: Sequence[Mapping[str, Any]],
    lncrna_by_cancer: Mapping[str, pd.DataFrame],
    pathway_by_cancer: Mapping[str, pd.DataFrame],
    patients_by_cancer: Mapping[str, Sequence[str]],
    core: Any,
    *,
    modality: str,
    min_pair_callable: int,
    maximum: int,
    seed: int,
    lncrna_callable_by_cancer: Mapping[str, set[str]] | None = None,
    pathway_callable_by_cancer: Mapping[str, set[str]] | None = None,
    resource_observer: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Byte-equivalent partition form of genomic_training._collect_examples.

    The order of operations intentionally mirrors the original implementation:
    global cancer count, ``ceil(maximum/n_cancers)`` quota (minimum two), one
    ``rng(seed+offset).choice`` per cancer, cancer-order concatenation, then the
    unchanged ``_balanced_indices(seed+100000)`` operation.
    """

    records = list(candidate_partitions)
    cancers = tuple(str(record.get("cancer_id", "")).upper() for record in records)
    if not records or len(cancers) != len(set(cancers)):
        raise PartitionedGenomicTrainingError("Candidate partition cancer order is invalid")
    quota = max(2, int(math.ceil(maximum / max(len(cancers), 1))))
    core_parts: list[np.ndarray] = []
    domain_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for offset, record in enumerate(records):
        cancer = cancers[offset]
        local = _partition_frame(record)
        stats = candidate_statistics(
            local,
            lncrna_by_cancer.get(cancer, pd.DataFrame()),
            pathway_by_cancer.get(cancer, pd.DataFrame()),
            patients_by_cancer.get(cancer, ()),
            modality=modality,
            min_pair_callable=min_pair_callable,
            lncrna_callable_entities=(
                lncrna_callable_by_cancer.get(cancer, set())
                if lncrna_callable_by_cancer is not None
                else None
            ),
            pathway_callable_entities=(
                pathway_callable_by_cancer.get(cancer, set())
                if pathway_callable_by_cancer is not None
                else None
            ),
        )
        eligible = np.flatnonzero(stats.available)
        if len(eligible):
            rng = np.random.default_rng(seed + offset)
            chosen = rng.choice(eligible, size=min(len(eligible), quota), replace=False)
            core_values, core_available = _candidate_core(local.iloc[chosen], core)
            chosen_available = np.flatnonzero(core_available)
            if len(chosen_available):
                core_parts.append(core_values[chosen_available])
                domain_parts.append(stats.domain[chosen][chosen_available])
                label_parts.append(stats.labels[chosen][chosen_available])
        if resource_observer is not None:
            resource_observer(f"collect:{modality}:{cancer}")
        del local, stats
    if not label_parts:
        width = core.lncrna.values.shape[1] * 2
        return (
            np.empty((0, width), np.float32),
            np.empty((0, len(_DOMAIN_FEATURES)), np.float32),
            np.empty(0, np.float32),
        )
    core_values = np.concatenate(core_parts)
    domain_values = np.concatenate(domain_parts)
    labels = np.concatenate(label_parts)
    choice = _balanced_indices(labels, min(maximum, len(labels)), seed + 100_000)
    if not len(choice):
        width = core.lncrna.values.shape[1] * 2
        return (
            np.empty((0, width), np.float32),
            np.empty((0, len(_DOMAIN_FEATURES)), np.float32),
            np.empty(0, np.float32),
        )
    return core_values[choice], domain_values[choice], labels[choice]


def _patient_split_trace(
    split_maps: Mapping[str, Mapping[str, Sequence[str]]], fold: int, modality: str
) -> dict[str, Any]:
    sets = {
        split: {
            f"{cancer}\t{patient}"
            for cancer, patients in mapping.items()
            for patient in patients
        }
        for split, mapping in split_maps.items()
    }
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        raise PartitionedGenomicTrainingError("Patient split sets overlap")
    return {
        "patient_fold": int(fold),
        "modality": modality,
        "patient_counts": {key: len(value) for key, value in sets.items()},
        "patient_set_sha256": {
            key: _canonical_json_sha256(sorted(value)) for key, value in sets.items()
        },
        "scaler_fit_split": "train",
        "optimizer_fit_split": "train",
        "early_stopping_split": "validation",
        "checkpoint_selection_split": "validation",
        "test_split_use": "PREDICTION_ONLY_AFTER_CHECKPOINT_SELECTION",
        "test_labels_or_features_enter_fit": False,
    }


def _assert_empty_resource_stage(manifest: Mapping[str, Any], stage_root: Path) -> None:
    if (stage_root / "PREDICTIONS_READY.json").exists():
        raise PartitionedGenomicTrainingError("Resource stage predictions are already sealed")
    receipt_root = stage_root / "write_receipts"
    if receipt_root.exists() and any(receipt_root.rglob("cancer_id=*.json")):
        raise PartitionedGenomicTrainingError("Resource stage already contains write receipts")
    expected_shape = (len(MODALITIES), N_FOLDS, int(manifest["candidate_rows"]))
    expected_dtypes = {
        "fold_probability": np.dtype("float32"),
        "fold_available": np.dtype("bool"),
        "fold_processed": np.dtype("bool"),
    }
    arrays = {
        key: np.load(Path(str(record["path"])), mmap_mode="r")
        for key, record in manifest["memmaps"].items()
    }
    try:
        for key, value in arrays.items():
            if tuple(value.shape) != expected_shape or value.dtype != expected_dtypes[key]:
                raise PartitionedGenomicTrainingError(
                    f"Empty resource memmap shape/dtype drift: {key}"
                )
        chunk = int(manifest["candidate_rows_per_cancer"])
        for start in range(0, expected_shape[2], chunk):
            stop = min(expected_shape[2], start + chunk)
            if np.asarray(arrays["fold_processed"][:, :, start:stop]).any():
                raise PartitionedGenomicTrainingError(
                    "Resource stage contains processed prediction rows"
                )
            if np.asarray(arrays["fold_available"][:, :, start:stop]).any():
                raise PartitionedGenomicTrainingError(
                    "Empty resource stage contains available prediction rows"
                )
            if not np.isnan(
                np.asarray(arrays["fold_probability"][:, :, start:stop])
            ).all():
                raise PartitionedGenomicTrainingError(
                    "Empty resource stage probabilities are not all NaN"
                )
    finally:
        arrays.clear()


def run_partitioned_genomic_training(
    *,
    resource_staging_success_path: str | Path,
    patient_folds_path: str | Path,
    pathway_gene_membership_path: str | Path,
    sample_gene_mutation_path: str | Path,
    sample_lncrna_mutation_path: str | Path,
    mc3_path: str | Path,
    core_embedding_manifest_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    cnv_streaming_success_path: str | Path | None = None,
    cnv_gene_calls_path: str | Path | None = None,
    cnv_lncrna_calls_path: str | Path | None = None,
    cnv_entity_coverage_path: str | Path | None = None,
    gistic_source_path: str | Path | None = None,
    lncrna_node_type: str = "lncRNA",
    pathway_node_type: str = "pathway",
    config: GenomicTrainingConfig | None = None,
    enforce_host_budget: bool = True,
    allow_development_long_cnv: bool = False,
) -> dict[str, Any]:
    """Train ten global private heads and write cancer partitions to memmap."""

    settings = config or GenomicTrainingConfig()
    settings.validate()
    if not training_run_id or not str(training_run_id).startswith("v32-"):
        raise PartitionedGenomicTrainingError("training_run_id must start with v32-")
    stage_success_path = Path(resource_staging_success_path).resolve()
    try:
        _, resource_manifest = _load_stage(
            stage_success_path, verify_external_hashes=True
        )
    except GenomicResourceError as exc:
        raise PartitionedGenomicTrainingError(str(exc)) from exc
    stage_root = stage_success_path.parent
    _assert_empty_resource_stage(resource_manifest, stage_root)
    budget_path = Path(str(resource_manifest["resource_budget"]["path"])).resolve()
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    downstream_budget = budget["downstream_budget"]

    paths = {
        "candidates": Path(str(resource_manifest["candidate_authority"]["path"])).resolve(),
        "patient_folds": Path(patient_folds_path).resolve(),
        "membership": Path(pathway_gene_membership_path).resolve(),
        "sample_gene_mutation": Path(sample_gene_mutation_path).resolve(),
        "sample_lncrna_mutation": Path(sample_lncrna_mutation_path).resolve(),
        "mc3": Path(mc3_path).resolve(),
        "core_manifest": Path(core_embedding_manifest_path).resolve(),
    }
    optional_paths = {
        "cnv_streaming_success": (
            Path(cnv_streaming_success_path).resolve()
            if cnv_streaming_success_path is not None
            else None
        ),
        "cnv_gene_calls": (
            Path(cnv_gene_calls_path).resolve() if cnv_gene_calls_path is not None else None
        ),
        "cnv_lncrna_calls": (
            Path(cnv_lncrna_calls_path).resolve()
            if cnv_lncrna_calls_path is not None
            else None
        ),
        "cnv_entity_coverage": (
            Path(cnv_entity_coverage_path).resolve()
            if cnv_entity_coverage_path is not None
            else None
        ),
        "gistic_source": (
            Path(gistic_source_path).resolve() if gistic_source_path is not None else None
        ),
    }
    for path in (*paths.values(), *(p for p in optional_paths.values() if p is not None)):
        if not path.exists():
            raise PartitionedGenomicTrainingError(f"Missing partition trainer input: {path}")
        _assert_source_name(path, core_parent=path == paths["core_manifest"])
    long_mode = optional_paths["cnv_gene_calls"] is not None or optional_paths["cnv_lncrna_calls"] is not None
    if (optional_paths["cnv_gene_calls"] is None) != (optional_paths["cnv_lncrna_calls"] is None):
        raise PartitionedGenomicTrainingError("Long CNV mode requires both call tables")
    if optional_paths["cnv_streaming_success"] is not None and long_mode:
        raise PartitionedGenomicTrainingError("Choose streaming or development-long CNV, not both")
    if optional_paths["cnv_streaming_success"] is None and not long_mode:
        raise PartitionedGenomicTrainingError("Partition training requires a CNV input mode")
    if long_mode and not allow_development_long_cnv:
        raise PartitionedGenomicTrainingError(
            "Formal partition training requires the aggregate streaming CNV store"
        )
    if optional_paths["cnv_entity_coverage"] is not None and not long_mode:
        raise PartitionedGenomicTrainingError(
            "CNV entity coverage is only valid with development-long CNV calls"
        )
    stage_cnv = resource_manifest["cnv_streaming_aggregate_success"]
    if optional_paths["cnv_streaming_success"] is not None:
        if (
            optional_paths["cnv_streaming_success"]
            != Path(str(stage_cnv["path"])).resolve()
            or artifact_sha256(optional_paths["cnv_streaming_success"])
            != stage_cnv["sha256"]
        ):
            raise PartitionedGenomicTrainingError(
                "Trainer CNV streaming gate differs from the resource stage"
            )

    output = Path(output_root).resolve()
    output_staging = output.with_name(f".{output.name}.staging")
    if output.exists() or output_staging.exists():
        raise PartitionedGenomicTrainingError(
            f"Partition training refuses output reuse: {output}"
        )
    available_memory = _available_memory_bytes()
    free_disk = int(shutil.disk_usage(output.parent).free)
    host_checks = {
        "available_memory_known": available_memory is not None,
        "available_memory_meets_budget": (
            available_memory is not None
            and available_memory
            >= int(downstream_budget["minimum_host_available_memory_bytes"])
        ),
        "free_disk_meets_budget": free_disk
        >= int(downstream_budget["minimum_host_free_disk_bytes"]),
    }
    if enforce_host_budget and not all(host_checks.values()):
        raise PartitionedGenomicTrainingError(
            f"Partition trainer host budget failed before output: {host_checks}"
        )
    rss_limit = int(downstream_budget["max_declared_process_rss_bytes"])
    resource_samples: list[dict[str, Any]] = []

    def observe(stage: str) -> None:
        rss = _process_rss_bytes()
        resource_samples.append({"stage": stage, "process_rss_bytes": rss})
        if rss is not None and rss > rss_limit:
            raise PartitionedGenomicTrainingError(
                f"Partition trainer RSS {rss} exceeds declared limit {rss_limit}: {stage}"
            )

    observe("preload")
    source_audit = _audit_sources(
        candidates_path=paths["candidates"],
        patient_folds_path=paths["patient_folds"],
        membership_path=paths["membership"],
        sample_gene_mutation_path=paths["sample_gene_mutation"],
        sample_lncrna_mutation_path=paths["sample_lncrna_mutation"],
        mc3_path=paths["mc3"],
        cnv_gene_calls_path=optional_paths["cnv_gene_calls"],
        cnv_lncrna_calls_path=optional_paths["cnv_lncrna_calls"],
        cnv_entity_coverage_path=optional_paths["cnv_entity_coverage"],
        cnv_segment_root=None,
        cnv_download_complete_path=None,
        entity_intervals_path=None,
        gistic_source_path=optional_paths["gistic_source"],
        cnv_streaming_success_path=optional_paths["cnv_streaming_success"],
    )
    core_manifest, core_manifest_sha, core_parameter_composite = _validate_core_manifest(
        paths["core_manifest"]
    )
    folds = normalise_patient_folds(_read_table(paths["patient_folds"]))
    membership = normalise_exact_membership(_read_table(paths["membership"]))
    formal_cancers = tuple(str(value) for value in resource_manifest["formal_cancers"])
    missing_cancers = sorted(set(formal_cancers) - set(folds.cancer_id))
    if missing_cancers:
        raise PartitionedGenomicTrainingError(
            f"Candidate cancers missing patient folds: {missing_cancers}"
        )
    gene_mutation = normalise_gene_mutation_calls(
        _read_table(paths["sample_gene_mutation"])
    )
    lnc_mutation = normalise_lncrna_mutation_calls(
        _read_table(paths["sample_lncrna_mutation"])
    )
    fold_keys = folds[["cancer_id", "patient_id"]]
    gene_mutation = gene_mutation.merge(
        fold_keys,
        on=["cancer_id", "patient_id"],
        how="inner",
        validate="many_to_one",
    )
    lnc_mutation = lnc_mutation.merge(
        fold_keys,
        on=["cancer_id", "patient_id"],
        how="inner",
        validate="many_to_one",
    )
    pathway_mutation = build_exact_pathway_calls(gene_mutation, membership)
    observe("mutation_sources_ready")

    cnv_static_coverage: dict[str, dict[str, set[str]]] | None = None
    cnv_pathway_coverage: dict[str, set[str]] | None = None
    streaming_payload: dict[str, Any] | None = None
    if optional_paths["cnv_streaming_success"] is not None:
        from .segment_cnv_streaming import (
            open_streaming_partitions,
            validate_streaming_store,
        )

        streaming_payload = validate_streaming_store(
            optional_paths["cnv_streaming_success"],
            expected_bindings={
                "candidates": paths["candidates"],
                "patient_folds": paths["patient_folds"],
                "pathway_membership": paths["membership"],
                "sample_gene_mutation": paths["sample_gene_mutation"],
                "sample_lncrna_mutation": paths["sample_lncrna_mutation"],
                "mc3": paths["mc3"],
                "core_embedding_manifest": paths["core_manifest"],
            },
        )
        compact_cnv = open_streaming_partitions(optional_paths["cnv_streaming_success"])
        cnv_lnc_by_cancer: Mapping[str, Any] = compact_cnv
        cnv_path_by_cancer: Mapping[str, Any] = compact_cnv
    else:
        if optional_paths["cnv_entity_coverage"] is not None:
            cnv_static_coverage = normalise_cnv_entity_coverage(
                _read_table(optional_paths["cnv_entity_coverage"])
            )
            cnv_pathway_coverage = exact_pathway_callable_entities(
                membership, cnv_static_coverage["gene"]
            )
        gene_cnv = normalise_cnv_calls(
            _read_table(optional_paths["cnv_gene_calls"]),
            entity_kind="gene",
            event_threshold=settings.cnv_event_threshold,
        )
        lnc_cnv = normalise_cnv_calls(
            _read_table(optional_paths["cnv_lncrna_calls"]),
            entity_kind="lncrna",
            event_threshold=settings.cnv_event_threshold,
        )
        gene_cnv = gene_cnv.merge(
            fold_keys,
            on=["cancer_id", "patient_id"],
            how="inner",
            validate="many_to_one",
        )
        lnc_cnv = lnc_cnv.merge(
            fold_keys,
            on=["cancer_id", "patient_id"],
            how="inner",
            validate="many_to_one",
        )
        pathway_cnv = build_exact_pathway_calls(gene_cnv, membership)
        cnv_lnc_by_cancer = _group_calls(lnc_cnv)
        cnv_path_by_cancer = _group_calls(pathway_cnv)
    sources: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], Any, Any]] = {
        "mutation": (
            _group_calls(lnc_mutation),
            _group_calls(pathway_mutation),
            None,
            None,
        ),
        "cnv": (
            cnv_lnc_by_cancer,
            cnv_path_by_cancer,
            cnv_static_coverage["lncrna"] if cnv_static_coverage is not None else None,
            cnv_pathway_coverage,
        ),
    }
    observe("all_sources_ready")

    output_staging.mkdir(parents=True)
    checkpoint_root = output_staging / "checkpoints"
    checkpoint_root.mkdir()
    checkpoint_rows: list[dict[str, Any]] = []
    split_traces: list[dict[str, Any]] = []
    core_hashes_before: dict[str, str] = {}
    partitions = resource_manifest["candidate_partitions"]
    for fold in range(N_FOLDS):
        core = load_fold_core_embeddings(
            paths["core_manifest"],
            core_manifest,
            fold,
            lncrna_node_type=lncrna_node_type,
            pathway_node_type=pathway_node_type,
        )
        core_hashes_before.update(core.input_hashes)
        split_maps = {
            split: _split_patients(folds, fold, split)
            for split in ("train", "validation", "test")
        }
        for modality, (
            lnc_by_cancer,
            path_by_cancer,
            lnc_callable_by_cancer,
            path_callable_by_cancer,
        ) in sources.items():
            train_core, train_domain, train_labels = collect_examples_partitioned(
                partitions,
                lnc_by_cancer,
                path_by_cancer,
                split_maps["train"],
                core,
                modality=modality,
                min_pair_callable=settings.min_pair_callable,
                maximum=settings.max_train_rows,
                seed=settings.seed + fold,
                lncrna_callable_by_cancer=lnc_callable_by_cancer,
                pathway_callable_by_cancer=path_callable_by_cancer,
                resource_observer=observe,
            )
            validation_core, validation_domain, validation_labels = collect_examples_partitioned(
                partitions,
                lnc_by_cancer,
                path_by_cancer,
                split_maps["validation"],
                core,
                modality=modality,
                min_pair_callable=settings.min_pair_callable,
                maximum=settings.max_validation_rows,
                seed=settings.seed + 50_000 + fold,
                lncrna_callable_by_cancer=lnc_callable_by_cancer,
                pathway_callable_by_cancer=path_callable_by_cancer,
                resource_observer=observe,
            )
            if len(train_labels) == 0 or len(np.unique(train_labels)) != 2:
                raise PartitionedGenomicTrainingError(
                    f"{modality} fold {fold} lacks two-class train rows"
                )
            if len(validation_labels) == 0 or len(np.unique(validation_labels)) != 2:
                raise PartitionedGenomicTrainingError(
                    f"{modality} fold {fold} lacks two-class validation rows; "
                    "train-loss checkpoint fallback is forbidden"
                )
            split_trace = _patient_split_trace(split_maps, fold, modality)
            head, metadata, mean, scale, history = _fit_head(
                train_core,
                train_domain,
                train_labels,
                validation_core,
                validation_domain,
                validation_labels,
                modality=modality,
                fold=fold,
                config=settings,
            )
            observe(f"fit:{modality}:fold={fold}")
            import torch

            checkpoint_path = checkpoint_root / f"{modality}_patient_fold_{fold}.pt"
            checkpoint_partial = checkpoint_root / f".{checkpoint_path.name}.partial"
            torch.save(
                {
                    "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
                    "analysis_version": ANALYSIS_VERSION,
                    "module_id": MODULE_ID,
                    "modality": modality,
                    "patient_fold": fold,
                    "initialization": metadata,
                    "model_state": head.state_dict(),
                    "domain_features": list(_DOMAIN_FEATURES),
                    "domain_mean": mean,
                    "domain_scale": scale,
                    "history": history,
                    "core_checkpoint_sha256": core.checkpoint_sha256,
                    "core_parameter_sha256": core.core_parameter_sha256,
                    "old_checkpoint_loaded": False,
                    "old_predictions_used": False,
                    "split_policy": split_trace,
                },
                checkpoint_partial,
            )
            os.replace(checkpoint_partial, checkpoint_path)
            checkpoint_sha = _file_sha256(checkpoint_path)
            checkpoint_success_path = checkpoint_root / (
                f"{modality}_patient_fold_{fold}.SUCCESS.json"
            )
            _atomic_json(
                checkpoint_success_path,
                {
                    "format": TRAINING_FORMAT,
                    "status": "CHECKPOINT_SELECTED_ON_VALIDATION_ONLY",
                    "modality": modality,
                    "patient_fold": fold,
                    "checkpoint_path": str(
                        output / "checkpoints" / checkpoint_path.name
                    ),
                    "checkpoint_sha256": checkpoint_sha,
                    "train_rows": int(len(train_labels)),
                    "validation_rows": int(len(validation_labels)),
                    "test_rows_consumed_by_fit": 0,
                },
            )
            checkpoint_rows.append(
                {
                    "modality": modality,
                    "patient_fold": fold,
                    "path": str(output / "checkpoints" / checkpoint_path.name),
                    "sha256": checkpoint_sha,
                    "train_rows": int(len(train_labels)),
                    "validation_rows": int(len(validation_labels)),
                    "best_validation_loss": float(metadata["best_validation_loss"]),
                    "checkpoint_selection_split": "validation",
                    "scaler_fit_split": "train",
                    "test_rows_consumed_by_fit": 0,
                    "core_checkpoint_sha256": core.checkpoint_sha256,
                    "core_parameter_sha256": core.core_parameter_sha256,
                    "source_checkpoint_sha256": None,
                }
            )
            split_traces.append(split_trace)

            # Test-fold features are first evaluated after the validation-only
            # checkpoint is fixed.  Labels are never passed to _fit_head.
            for record in partitions:
                cancer = str(record["cancer_id"])
                local = _partition_frame(record)
                stats = candidate_statistics(
                    local,
                    lnc_by_cancer.get(cancer, pd.DataFrame()),
                    path_by_cancer.get(cancer, pd.DataFrame()),
                    split_maps["test"].get(cancer, ()),
                    modality=modality,
                    min_pair_callable=settings.min_pair_callable,
                    lncrna_callable_entities=(
                        lnc_callable_by_cancer.get(cancer, set())
                        if lnc_callable_by_cancer is not None
                        else None
                    ),
                    pathway_callable_entities=(
                        path_callable_by_cancer.get(cancer, set())
                        if path_callable_by_cancer is not None
                        else None
                    ),
                )
                values = np.full(len(local), np.nan, dtype=np.float32)
                core_values, core_available = _candidate_core(local, core)
                predict_mask = stats.available & core_available
                if predict_mask.any():
                    values[predict_mask] = _predict_head(
                        head,
                        core_values[predict_mask],
                        stats.domain[predict_mask],
                        mean,
                        scale,
                        settings.prediction_batch_size,
                    )
                try:
                    write_genomic_memmap_partition(
                        staging_success_path=stage_success_path,
                        modality=modality,
                        patient_fold=fold,
                        cancer_id=cancer,
                        candidate_keys=local,
                        probability=values,
                        available=predict_mask.astype(bool),
                    )
                except GenomicResourceError as exc:
                    raise PartitionedGenomicTrainingError(str(exc)) from exc
                observe(f"predict:{modality}:fold={fold}:{cancer}")
                del local, stats, values, core_values, core_available, predict_mask
            del train_core, train_domain, train_labels
            del validation_core, validation_domain, validation_labels, head

    if len(checkpoint_rows) != len(MODALITIES) * N_FOLDS:
        raise PartitionedGenomicTrainingError("Exactly ten genomic checkpoints are required")
    for path_text, before in core_hashes_before.items():
        if _file_sha256(path_text) != before:
            raise PartitionedGenomicTrainingError(
                f"Frozen core embedding changed during partition training: {path_text}"
            )
    if _file_sha256(paths["core_manifest"]) != core_manifest_sha:
        raise PartitionedGenomicTrainingError(
            "Core embedding manifest changed during partition training"
        )
    try:
        predictions_ready = seal_genomic_memmap_predictions(
            staging_success_path=stage_success_path
        )
    except GenomicResourceError as exc:
        raise PartitionedGenomicTrainingError(str(exc)) from exc
    ready_path = stage_root / "PREDICTIONS_READY.json"
    observe("predictions_sealed")

    split_policy = {
        "format": SPLIT_POLICY_FORMAT,
        "outer_test_policy": "PREDICTION_ONLY_AFTER_VALIDATION_CHECKPOINT_SELECTION",
        "scaling_fit": "TRAIN_PATIENTS_ONLY",
        "optimizer_fit": "TRAIN_PATIENTS_ONLY",
        "early_stopping": "VALIDATION_PATIENTS_ONLY",
        "checkpoint_choice": "VALIDATION_PATIENTS_ONLY",
        "test_keys_or_labels_enter_scaling_early_stop_or_checkpoint": False,
        "traces": split_traces,
    }
    checkpoint_manifest = {
        "format": CHECKPOINT_MANIFEST_FORMAT,
        "status": "TEN_GLOBAL_HEAD_CHECKPOINTS_COMPLETE",
        "records": checkpoint_rows,
        "checkpoint_count": len(checkpoint_rows),
        "global_head_scope": "ONE_CROSS_CANCER_HEAD_PER_MODALITY_AND_FOLD",
        "collector_semantics": {
            "per_cancer_quota": "max(2,ceil(maximum/n_cancers))",
            "per_cancer_choice": "rng(seed+offset).choice_without_replacement",
            "concatenation": "FORMAL_CANCER_ORDER",
            "final_balance": "original__balanced_indices_seed_plus_100000",
        },
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
    }
    lineage = {
        "format": TRAINING_FORMAT,
        "status": "TRAINING_COMPLETE_PREDICTIONS_SEALED_NOT_RELEASED",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": str(training_run_id),
        "resource_stage_success_path": str(stage_success_path),
        "resource_stage_success_sha256": artifact_sha256(stage_success_path),
        "resource_stage_is_genomic_oof_success": False,
        "candidate_authority": resource_manifest["candidate_authority"],
        "ordered_exact_candidate_key_sha256": resource_manifest[
            "ordered_exact_candidate_key_sha256"
        ],
        "cnv_mode": (
            "STREAMING_COMPACT_GDC_SEGMENT"
            if streaming_payload is not None
            else "DEVELOPMENT_LONG_CALL_TABLES"
        ),
        "cnv_streaming_aggregate_success": stage_cnv,
        "patient_folds": N_FOLDS,
        "modalities": list(MODALITIES),
        "checkpoint_count": len(checkpoint_rows),
        "global_cross_cancer_heads": True,
        "configuration": asdict(settings),
        "core_manifest_sha256": core_manifest_sha,
        "core_parameter_composite_sha256": core_parameter_composite,
        "core_parameters_before_sha256": core_hashes_before,
        "core_parameters_after_sha256": {
            path: _file_sha256(path) for path in core_hashes_before
        },
        "input_lineage_audit_status": source_audit["status"],
        "split_policy": split_policy,
        "predictions_ready_path": str(ready_path),
        "predictions_ready_sha256": artifact_sha256(ready_path),
        "predictions_ready_format": predictions_ready["format"],
        "formal_genomic_oof_success_emitted": False,
        "release_ready": False,
    }
    observed_values = [
        int(item["process_rss_bytes"])
        for item in resource_samples
        if item["process_rss_bytes"] is not None
    ]
    resource_audit = {
        "format": TRAINING_FORMAT,
        "status": "RESOURCE_BUDGET_PASS",
        "declared_process_rss_limit_bytes": rss_limit,
        "sampled_process_rss_max_bytes": max(observed_values) if observed_values else None,
        "host_available_memory_bytes_at_start": available_memory,
        "host_free_disk_bytes_at_start": free_disk,
        "host_checks": host_checks,
        "host_checks_enforced": bool(enforce_host_budget),
        "candidate_residency": "ONE_CANCER_PARTITION_AT_A_TIME",
        "fold_prediction_storage": "DISK_MEMMAP",
        "samples": resource_samples,
    }
    _atomic_json(output_staging / "INPUT_LINEAGE_AUDIT.json", source_audit)
    _atomic_json(output_staging / "SPLIT_POLICY.json", split_policy)
    _atomic_json(output_staging / "CHECKPOINT_MANIFEST.json", checkpoint_manifest)
    _atomic_json(output_staging / "RESOURCE_AUDIT.json", resource_audit)
    _atomic_json(output_staging / "LINEAGE.json", lineage)
    training_success = {
        "format": TRAINING_FORMAT,
        "status": "TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED",
        "training_run_id": str(training_run_id),
        "lineage_path": str(output / "LINEAGE.json"),
        "lineage_sha256": artifact_sha256(output_staging / "LINEAGE.json"),
        "checkpoint_manifest_path": str(output / "CHECKPOINT_MANIFEST.json"),
        "checkpoint_manifest_sha256": artifact_sha256(
            output_staging / "CHECKPOINT_MANIFEST.json"
        ),
        "predictions_ready_path": str(ready_path),
        "predictions_ready_sha256": artifact_sha256(ready_path),
        "checkpoint_count": len(checkpoint_rows),
        "resource_stage_is_genomic_oof_success": False,
        "formal_genomic_oof_success_emitted": False,
        "release_ready": False,
        "success_written_last": True,
    }
    _atomic_json(output_staging / "TRAINING_SUCCESS.json", training_success)
    os.replace(output_staging, output)
    return training_success


__all__ = [
    "CHECKPOINT_MANIFEST_FORMAT",
    "PartitionedGenomicTrainingError",
    "SPLIT_POLICY_FORMAT",
    "TRAINING_FORMAT",
    "collect_examples_partitioned",
    "run_partitioned_genomic_training",
]
