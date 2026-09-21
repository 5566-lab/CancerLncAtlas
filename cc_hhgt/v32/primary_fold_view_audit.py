"""Independent, fail-closed audit of V3.2 primary fold views."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .multimodal_fusion import TARGET_KEYS, artifact_sha256
from .primary_fold_views import EXTRACTION_FORMAT, FOLD_COLUMN, N_FOLDS, VIEW_COLUMNS
from .training import PREPARED_FORMAT


AUDIT_FORMAT = "CC_HHGT_V3_2_PRIMARY_FOLD_VIEWS_INDEPENDENT_AUDIT_V1"


class PrimaryFoldViewAuditError(RuntimeError):
    pass


def _independent_pair_fold(lncrna_id: object, pathway_id: object, *, seed: int) -> int:
    """Reimplement the frozen fold formula without importing production code."""

    token = f"{str(lncrna_id).strip()}|{str(pathway_id).strip()}|{int(seed)}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big") % N_FOLDS


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimaryFoldViewAuditError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PrimaryFoldViewAuditError(f"{label} is not a mapping")
    return value


def _key(cancer: object, lnc: object, pathway: object) -> tuple[str, str, str]:
    value = (str(cancer).upper(), str(lnc), str(pathway))
    if not all(value) or value[0] != str(cancer):
        raise PrimaryFoldViewAuditError("Non-canonical exact candidate key")
    return value


def _key_bytes(value: tuple[str, str, str]) -> bytes:
    return "\t".join(value).encode("utf-8")


def _update_strict_keys(
    rows: Iterable[tuple[object, object, object]],
    *,
    digest: "hashlib._Hash",
    previous: tuple[str, str, str] | None,
    counts: Counter[str],
    label: str,
) -> tuple[tuple[str, str, str] | None, int]:
    added = 0
    for row in rows:
        value = _key(*row)
        if previous is not None and value <= previous:
            raise PrimaryFoldViewAuditError(f"{label} exact-key order/uniqueness drift")
        previous = value
        digest.update(_key_bytes(value) + b"\n")
        counts[value[0]] += 1
        added += 1
    return previous, added


def _candidate_contract(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    if missing := sorted(set(TARGET_KEYS) - set(source.schema_arrow.names)):
        raise PrimaryFoldViewAuditError(f"Candidate authority lacks: {missing}")
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    previous = None
    rows = 0
    for batch in source.iter_batches(columns=list(TARGET_KEYS), batch_size=100_000):
        previous, added = _update_strict_keys(
            batch.to_pandas().itertuples(index=False, name=None),
            digest=digest,
            previous=previous,
            counts=counts,
            label="candidate authority",
        )
        rows += added
    return {
        "rows": rows,
        "ordered_exact_candidate_key_sha256": digest.hexdigest(),
        "cancer_counts": dict(counts),
        "sha256": artifact_sha256(path),
    }


def _default_loader(path: Path) -> Mapping[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _array(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover
        raise PrimaryFoldViewAuditError(f"Cannot convert {label} to an array") from exc


def _sigmoid(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        import torch

        tensor = value.detach().cpu()
        if tensor.ndim != 1:
            raise PrimaryFoldViewAuditError("Source base_logit is not a vector")
        return torch.sigmoid(tensor).numpy().astype(np.float32, copy=False)
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise PrimaryFoldViewAuditError("Source base_logit is not a vector")
    raw = raw.astype(np.float64, copy=False)
    result = np.empty_like(raw)
    positive = raw >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result.astype(np.float32)


def _inverse(payload: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    bundle = payload.get("bundle")
    maps = getattr(bundle, "node_maps", None)
    if not isinstance(maps, Mapping):
        raise PrimaryFoldViewAuditError("Source payload lacks graph node maps")
    result = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        if kind not in maps or not isinstance(maps[kind], Mapping):
            raise PrimaryFoldViewAuditError(f"Source payload lacks {kind} map")
        result[kind] = {int(index): str(identifier) for identifier, index in maps[kind].items()}
        if len(result[kind]) != len(maps[kind]):
            raise PrimaryFoldViewAuditError(f"Source {kind} map is not one-to-one")
    return result


def _source_batch(
    raw: Mapping[str, Any], inverse: Mapping[str, Mapping[int, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = raw.get("candidate_batch")
    if not isinstance(candidate, Mapping):
        raise PrimaryFoldViewAuditError("Source batch lacks candidate_batch")
    index = {}
    for short in ("c", "l", "p"):
        values = _array(candidate.get(short), f"candidate_batch.{short}")
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise PrimaryFoldViewAuditError(f"Source candidate {short} is not an integer vector")
        index[short] = values.astype(np.int64, copy=False)
    size = len(index["c"])
    if len(index["l"]) != size or len(index["p"]) != size:
        raise PrimaryFoldViewAuditError("Source candidate index lengths differ")
    try:
        cancer = np.asarray(
            [inverse["cancer"][int(item)].upper() for item in index["c"]], dtype=object
        )
        lnc = np.asarray([inverse["lncRNA"][int(item)] for item in index["l"]], dtype=object)
        pathway = np.asarray(
            [inverse["pathway"][int(item)] for item in index["p"]], dtype=object
        )
    except KeyError as exc:
        raise PrimaryFoldViewAuditError("Source batch references an absent node") from exc
    probability = _sigmoid(raw.get("base_logit"))
    target = _array(raw.get("proxy_label"), "proxy_label")
    if target.ndim != 1 or len(target) != size or len(probability) != size:
        raise PrimaryFoldViewAuditError("Source primary arrays differ in length")
    target = target.astype(np.float64, copy=False)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise PrimaryFoldViewAuditError("Source primary probability is invalid")
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise PrimaryFoldViewAuditError("Source proxy label is not binary")
    return cancer, lnc, pathway, probability, target.astype(np.uint8)


def _expected_mask(folds: np.ndarray, split: str, outer: int) -> np.ndarray:
    validation = (outer + 1) % N_FOLDS
    if split == "train":
        return ~np.isin(folds, [outer, validation])
    if split == "validation":
        return folds == validation
    if split == "test":
        return folds == outer
    raise PrimaryFoldViewAuditError(f"Unknown split: {split}")


def _value_digest_update(
    digest: "hashlib._Hash",
    cancer: np.ndarray,
    lnc: np.ndarray,
    pathway: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
) -> None:
    for values in zip(cancer, lnc, pathway, probability, target):
        exact = _key(values[0], values[1], values[2])
        digest.update(_key_bytes(exact))
        digest.update(b"\t" + struct.pack("<fB", float(values[3]), int(values[4])) + b"\n")


def _source_view_contract(
    payload: Mapping[str, Any],
    *,
    split: str,
    outer: int,
    pair_seed: int,
) -> dict[str, Any]:
    inverse = _inverse(payload)
    batches = payload.get(f"{split}_batches")
    if not isinstance(batches, Sequence) or not batches:
        raise PrimaryFoldViewAuditError(f"Source fold {outer}/{split} lacks batches")
    source_digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    value_digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    pair_counts: Counter[int] = Counter()
    source_previous = None
    selected_previous = None
    source_rows = 0
    selected_rows = 0
    for raw in batches:
        cancer, lnc, pathway, probability, target = _source_batch(raw, inverse)
        source_previous, added = _update_strict_keys(
            zip(cancer, lnc, pathway),
            digest=source_digest,
            previous=source_previous,
            counts=source_counts,
            label=f"source fold {outer}/{split}",
        )
        source_rows += added
        folds = np.fromiter(
            (
                _independent_pair_fold(left, right, seed=pair_seed)
                for left, right in zip(lnc, pathway)
            ),
            dtype=np.int8,
            count=len(lnc),
        )
        keep = _expected_mask(folds, split, outer)
        selected_previous, added = _update_strict_keys(
            zip(cancer[keep], lnc[keep], pathway[keep]),
            digest=selected_digest,
            previous=selected_previous,
            counts=selected_counts,
            label=f"selected source fold {outer}/{split}",
        )
        selected_rows += added
        pair_counts.update(map(int, folds[keep].tolist()))
        _value_digest_update(
            value_digest,
            cancer[keep],
            lnc[keep],
            pathway[keep],
            probability[keep],
            target[keep],
        )
    return {
        "source_rows": source_rows,
        "source_ordered_key_sha256": source_digest.hexdigest(),
        "source_cancer_counts": dict(source_counts),
        "rows": selected_rows,
        "ordered_exact_key_sha256": selected_digest.hexdigest(),
        "key_primary_target_sha256": value_digest.hexdigest(),
        "cancer_counts": dict(selected_counts),
        "pair_folds": sorted(pair_counts),
    }


def _validate_split_feature_parity(payload: Mapping[str, Any], *, outer: int) -> dict[str, Any]:
    """Prove held-out labels are not accompanied by held-out feature values.

    The formal preparation contract duplicates train-patient discovery
    features across the three label views.  Validation/test may change only
    label-control tensors, never candidate identity, base logit, conservation
    context, graph availability, or train-discovery direction features.
    """

    split_batches = {
        split: payload.get(f"{split}_batches")
        for split in ("train", "validation", "test")
    }
    if any(not isinstance(value, Sequence) or not value for value in split_batches.values()):
        raise PrimaryFoldViewAuditError(f"Fold {outer} lacks a split batch list")
    lengths = {split: len(value) for split, value in split_batches.items()}
    if len(set(lengths.values())) != 1:
        raise PrimaryFoldViewAuditError(f"Fold {outer} split batch counts differ")
    feature_paths = (
        ("candidate_batch", "c"),
        ("candidate_batch", "l"),
        ("candidate_batch", "p"),
        (None, "base_logit"),
        (None, "conservation_context"),
        (None, "graph_available"),
        (None, "direction_label"),
        (None, "direction_available"),
    )
    compared_rows = 0
    for batch_index, batches in enumerate(
        zip(
            split_batches["train"],
            split_batches["validation"],
            split_batches["test"],
        )
    ):
        train, validation, test = batches
        for container, name in feature_paths:
            values = []
            for split, raw in zip(("train", "validation", "test"), batches):
                value = raw.get(container, {}).get(name) if container else raw.get(name)
                if value is None:
                    raise PrimaryFoldViewAuditError(
                        f"Fold {outer}/{split} batch {batch_index} lacks feature {name}"
                    )
                values.append(_array(value, f"{split}.{name}"))
            if not (
                values[0].shape == values[1].shape == values[2].shape
                and values[0].dtype == values[1].dtype == values[2].dtype
                and np.array_equal(values[0], values[1], equal_nan=True)
                and np.array_equal(values[0], values[2], equal_nan=True)
            ):
                raise PrimaryFoldViewAuditError(
                    f"Fold {outer} held-out feature drift in {name} batch {batch_index}"
                )
        compared_rows += int(len(_array(train["base_logit"], "train.base_logit")))
    return {
        "batch_count": lengths["train"],
        "rows": compared_rows,
        "bitwise_identical_feature_paths": [
            f"{container}.{name}" if container else name
            for container, name in feature_paths
        ],
        "validation_or_test_label_tensors_compared_as_features": False,
    }


def _validate_train_validation_feature_parity(
    payload: Mapping[str, Any], *, outer: int
) -> dict[str, Any]:
    """Compare only unsealed training/validation feature views.

    The test view is independently sourced from the post-winner-lock
    acceptance and therefore must never be present in this payload.
    """

    split_batches = {
        split: payload.get(f"{split}_batches") for split in ("train", "validation")
    }
    if any(not isinstance(value, Sequence) or not value for value in split_batches.values()):
        raise PrimaryFoldViewAuditError(f"Fold {outer} lacks train/validation batches")
    if len(split_batches["train"]) != len(split_batches["validation"]):
        raise PrimaryFoldViewAuditError(f"Fold {outer} train/validation batch counts differ")
    feature_paths = (
        ("candidate_batch", "c"),
        ("candidate_batch", "l"),
        ("candidate_batch", "p"),
        (None, "base_logit"),
        (None, "conservation_context"),
        (None, "graph_available"),
        (None, "direction_label"),
        (None, "direction_available"),
    )
    rows = 0
    for batch_index, batches in enumerate(
        zip(split_batches["train"], split_batches["validation"], strict=True)
    ):
        for container, name in feature_paths:
            values = []
            for split, raw in zip(("train", "validation"), batches, strict=True):
                value = raw.get(container, {}).get(name) if container else raw.get(name)
                if value is None:
                    raise PrimaryFoldViewAuditError(
                        f"Fold {outer}/{split} batch {batch_index} lacks feature {name}"
                    )
                values.append(_array(value, f"{split}.{name}"))
            if not (
                values[0].shape == values[1].shape
                and values[0].dtype == values[1].dtype
                and np.array_equal(values[0], values[1], equal_nan=True)
            ):
                raise PrimaryFoldViewAuditError(
                    f"Fold {outer} train/validation feature drift in {name} batch {batch_index}"
                )
        rows += int(len(_array(batches[0]["base_logit"], "train.base_logit")))
    return {
        "batch_count": len(split_batches["train"]),
        "rows": rows,
        "bitwise_identical_feature_paths": [
            f"{container}.{name}" if container else name for container, name in feature_paths
        ],
        "test_payload_or_labels_inspected": False,
    }


def _sealed_source_view_contract(
    path: Path, *, outer: int, pair_seed: int
) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    if missing := sorted(set(VIEW_COLUMNS) - set(frame.columns)):
        raise PrimaryFoldViewAuditError(f"Sealed primary frame lacks: {missing}")
    observed_fold = pd.to_numeric(frame[FOLD_COLUMN], errors="raise").astype(int)
    selected = frame.loc[observed_fold.eq(outer), list(VIEW_COLUMNS)].copy()
    if selected.empty:
        raise PrimaryFoldViewAuditError(f"Sealed primary frame lacks fold {outer}")
    key_digest = hashlib.sha256()
    value_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    previous = None
    previous, rows = _update_strict_keys(
        selected[list(TARGET_KEYS)].itertuples(index=False, name=None),
        digest=key_digest,
        previous=previous,
        counts=counts,
        label=f"sealed source fold {outer}/test",
    )
    cancer = selected.cancer_id.astype(str).str.upper().to_numpy(object)
    lnc = selected.lncrna_id.astype(str).to_numpy(object)
    pathway = selected.pathway_id.astype(str).to_numpy(object)
    probability = selected.primary_probability.to_numpy(dtype=np.float32)
    target = selected.fusion_target.to_numpy(dtype=np.uint8)
    expected = np.fromiter(
        (
            _independent_pair_fold(left, right, seed=pair_seed)
            for left, right in zip(lnc, pathway)
        ),
        dtype=np.int8,
        count=len(selected),
    )
    if (
        not np.all(expected == outer)
        or not np.all(selected.source_patient_fold.to_numpy() == outer)
        or not selected.source_split.astype(str).eq("test").all()
    ):
        raise PrimaryFoldViewAuditError("Sealed source patient/pair-fold lineage drift")
    _value_digest_update(value_digest, cancer, lnc, pathway, probability, target)
    return {
        "rows": rows,
        "ordered_exact_key_sha256": key_digest.hexdigest(),
        "key_primary_target_sha256": value_digest.hexdigest(),
        "cancer_counts": dict(counts),
        "pair_folds": [outer],
    }


def _output_view_contract(
    path: Path, *, split: str, outer: int, pair_seed: int
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if missing := sorted(set(VIEW_COLUMNS) - set(parquet.schema_arrow.names)):
        raise PrimaryFoldViewAuditError(f"Output view lacks: {missing}")
    key_digest = hashlib.sha256()
    value_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    pair_counts: Counter[int] = Counter()
    previous = None
    rows = 0
    for batch in parquet.iter_batches(columns=list(VIEW_COLUMNS), batch_size=100_000):
        frame = batch.to_pandas()
        cancer = frame.cancer_id.astype(str).str.upper().to_numpy(object)
        if not frame.cancer_id.astype(str).eq(cancer).all():
            raise PrimaryFoldViewAuditError("Output cancer keys are not canonical uppercase")
        lnc = frame.lncrna_id.astype(str).to_numpy(object)
        pathway = frame.pathway_id.astype(str).to_numpy(object)
        previous, added = _update_strict_keys(
            zip(cancer, lnc, pathway),
            digest=key_digest,
            previous=previous,
            counts=counts,
            label=f"output fold {outer}/{split}",
        )
        rows += added
        source_fold = frame.source_patient_fold.to_numpy()
        source_split = frame.source_split.astype(str).to_numpy()
        if not np.all(source_fold == outer) or not np.all(source_split == split):
            raise PrimaryFoldViewAuditError("Output source fold/split metadata drift")
        observed_fold = frame[FOLD_COLUMN].to_numpy()
        expected_fold = np.fromiter(
            (
                _independent_pair_fold(left, right, seed=pair_seed)
                for left, right in zip(lnc, pathway)
            ),
            dtype=np.int8,
            count=len(frame),
        )
        if not np.array_equal(observed_fold.astype(np.int64), expected_fold.astype(np.int64)):
            raise PrimaryFoldViewAuditError("Output pair-fold assignment drift")
        if not _expected_mask(expected_fold, split, outer).all():
            raise PrimaryFoldViewAuditError("Output split contains a forbidden pair fold")
        pair_counts.update(map(int, expected_fold.tolist()))
        probability = frame.primary_probability.to_numpy(dtype=np.float32)
        target = frame.fusion_target.to_numpy(dtype=np.float64)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise PrimaryFoldViewAuditError("Output primary probability is invalid")
        if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
            raise PrimaryFoldViewAuditError("Output target is not binary")
        _value_digest_update(
            value_digest,
            cancer,
            lnc,
            pathway,
            probability,
            target.astype(np.uint8),
        )
    return {
        "rows": rows,
        "ordered_exact_key_sha256": key_digest.hexdigest(),
        "key_primary_target_sha256": value_digest.hexdigest(),
        "cancer_counts": dict(counts),
        "pair_folds": sorted(pair_counts),
        "sha256": artifact_sha256(path),
    }


def audit_primary_fold_views(
    *,
    prepared_root: str | Path,
    candidate_authority_path: str | Path,
    budget_contract_path: str | Path,
    extraction_manifest_path: str | Path,
    output_root: str | Path,
    payload_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared = Path(prepared_root).resolve()
    candidate_path = Path(candidate_authority_path).resolve()
    budget_path = Path(budget_contract_path).resolve()
    manifest_path = Path(extraction_manifest_path).resolve()
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise PrimaryFoldViewAuditError(f"Audit refuses output reuse: {output}")
    for label, path in (
        ("candidate authority", candidate_path),
        ("budget", budget_path),
        ("extraction manifest", manifest_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise PrimaryFoldViewAuditError(f"Missing {label}: {path}")
    budget = _json(budget_path, "fair budget")
    if budget.get("format") != "CC_HHGT_V3_2_ROUTING_FAIR_BUDGET_V1":
        raise PrimaryFoldViewAuditError("Fair budget format drift")
    manifest = _json(manifest_path, "extraction manifest")
    if (
        manifest.get("format") != EXTRACTION_FORMAT
        or manifest.get("status") != "EXTRACTED_AWAITING_INDEPENDENT_AUDIT_NO_TRAINING"
        or manifest.get("training_run") is not False
        or manifest.get("winner_selection_run") is not False
        or manifest.get("formal_release_success_emitted") is not False
        or manifest.get("one_single_aggregated_primary_table_is_fair_training_authority")
        is not False
    ):
        raise PrimaryFoldViewAuditError("Extraction manifest state/format drift")
    if manifest.get("budget_contract", {}).get("sha256") != artifact_sha256(budget_path):
        raise PrimaryFoldViewAuditError("Extraction is not bound to this fair budget")
    sealed_declaration = manifest.get("sealed_test_acceptance")
    sealed_mode = isinstance(sealed_declaration, Mapping)
    sealed_primary_path: Path | None = None
    if sealed_mode:
        acceptance_path = Path(str(sealed_declaration.get("path", ""))).resolve()
        if (
            not acceptance_path.is_file()
            or artifact_sha256(acceptance_path) != sealed_declaration.get("sha256")
        ):
            raise PrimaryFoldViewAuditError("Sealed-test acceptance path/SHA drift")
        acceptance = _json(acceptance_path, "sealed-test acceptance")
        if (
            acceptance.get("format")
            != "CC_HHGT_V3_2_WINNER_LOCK_SEALED_TEST_ACCEPTANCE_V1"
            or acceptance.get("status") != "PASS_WINNER_LOCK_SEALED_TEST_INFERENCE"
            or acceptance.get("test_labels_absent_from_training_payload") is not True
            or acceptance.get("sealed_payloads_opened_only_after_winner_lock") is not True
            or acceptance.get("test_metrics_used_for_winner_selection") is not False
            or acceptance.get("winner_or_checkpoint_changed_by_test") is not False
            or acceptance.get("old_fold_outputs_used") is not False
        ):
            raise PrimaryFoldViewAuditError("Sealed-test acceptance governance drift")
        primary_record = acceptance.get("primary_fusion_frame")
        if not isinstance(primary_record, Mapping):
            raise PrimaryFoldViewAuditError("Sealed-test acceptance lacks primary frame")
        sealed_primary_path = Path(str(primary_record.get("path", ""))).resolve()
        if (
            not sealed_primary_path.is_file()
            or artifact_sha256(sealed_primary_path) != primary_record.get("sha256")
        ):
            raise PrimaryFoldViewAuditError("Sealed primary frame path/SHA drift")
    candidate = _candidate_contract(candidate_path)
    declared_candidate = manifest.get("candidate_authority", {})
    for key in ("sha256", "rows", "ordered_exact_candidate_key_sha256", "cancer_counts"):
        if declared_candidate.get(key) != candidate[key]:
            raise PrimaryFoldViewAuditError(f"Candidate authority declaration drift: {key}")
    expected_rows = int(candidate["rows"])
    pair_seed = int(budget["seeds"]["pair_fold_seed"])
    loader = payload_loader or _default_loader
    source_by_fold = {
        int(item["patient_fold"]): item for item in manifest.get("source_prepared_folds", [])
    }
    view_by_key = {
        (int(item["outer_pair_fold"]), str(item["split"])): item
        for item in manifest.get("views", [])
    }
    if set(source_by_fold) != set(range(N_FOLDS)) or set(view_by_key) != {
        (fold, split)
        for fold in range(N_FOLDS)
        for split in ("train", "validation", "test")
    }:
        raise PrimaryFoldViewAuditError("Extraction manifest fold/view inventory drift")
    staging.mkdir(parents=True)
    audit_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for outer in range(N_FOLDS):
        source_path = prepared / f"PATIENT_FOLD_{outer}.pt"
        declared_source = source_by_fold[outer]
        if Path(str(declared_source.get("path", ""))).resolve() != source_path:
            raise PrimaryFoldViewAuditError(f"Prepared fold {outer} path drift")
        source_sha = artifact_sha256(source_path)
        if declared_source.get("sha256") != source_sha:
            raise PrimaryFoldViewAuditError(f"Prepared fold {outer} hash drift")
        payload = loader(source_path)
        if not isinstance(payload, Mapping):
            raise PrimaryFoldViewAuditError(f"Prepared fold {outer} is not a mapping")
        scope = payload.get("input_scope", {})
        expected_format = PREPARED_FORMAT if sealed_mode else "CC_HHGT_V3_2_PREPARED_FOLD_V1"
        expected_replication = {"validation"} if sealed_mode else {"validation", "test"}
        if (
            payload.get("prepared_format") != expected_format
            or int(payload.get("patient_fold", -1)) != outer
            or payload.get("contains_optimizer_state") is not False
            or payload.get("contains_trained_parameters") is not False
            or scope.get("discovery_features_split") != "train_patients_only"
            or scope.get("held_out_effects_as_features") is not False
            or set(scope.get("replication_labels_splits", [])) != expected_replication
            or (sealed_mode and "test_batches" in payload)
        ):
            raise PrimaryFoldViewAuditError(f"Prepared fold {outer} leakage metadata drift")
        feature_parity = (
            _validate_train_validation_feature_parity(payload, outer=outer)
            if sealed_mode
            else _validate_split_feature_parity(payload, outer=outer)
        )
        selected_union = 0
        audit_splits = ("train", "validation") + (("test",) if not sealed_mode else ())
        for split in audit_splits:
            source_contract = _source_view_contract(
                payload, split=split, outer=outer, pair_seed=pair_seed
            )
            if (
                source_contract["source_rows"] != expected_rows
                or source_contract["source_ordered_key_sha256"]
                != candidate["ordered_exact_candidate_key_sha256"]
                or source_contract["source_cancer_counts"] != candidate["cancer_counts"]
            ):
                raise PrimaryFoldViewAuditError(
                    f"Prepared fold {outer}/{split} full candidate universe drift"
                )
            declared_view = view_by_key[(outer, split)]
            view_path = Path(str(declared_view.get("path", ""))).resolve()
            if not view_path.is_file():
                raise PrimaryFoldViewAuditError(f"Missing extracted view: {view_path}")
            output_contract = _output_view_contract(
                view_path, split=split, outer=outer, pair_seed=pair_seed
            )
            for key in (
                "rows",
                "ordered_exact_key_sha256",
                "key_primary_target_sha256",
                "cancer_counts",
                "pair_folds",
            ):
                if output_contract[key] != source_contract[key]:
                    raise PrimaryFoldViewAuditError(
                        f"Fold {outer}/{split} source/output mismatch: {key}"
                    )
            for key in (
                "rows",
                "ordered_exact_key_sha256",
                "key_primary_target_sha256",
                "cancer_counts",
                "sha256",
            ):
                if declared_view.get(key) != output_contract[key]:
                    raise PrimaryFoldViewAuditError(
                        f"Fold {outer}/{split} manifest/output mismatch: {key}"
                    )
            selected_union += int(output_contract["rows"])
            audit_records.append(
                {
                    "outer_pair_fold": outer,
                    "split": split,
                    "path": str(view_path),
                    **output_contract,
                    "source_values_recomputed_independently": True,
                }
            )
        if sealed_mode:
            assert sealed_primary_path is not None
            split = "test"
            source_contract = _sealed_source_view_contract(
                sealed_primary_path, outer=outer, pair_seed=pair_seed
            )
            declared_view = view_by_key[(outer, split)]
            view_path = Path(str(declared_view.get("path", ""))).resolve()
            output_contract = _output_view_contract(
                view_path, split=split, outer=outer, pair_seed=pair_seed
            )
            for key in (
                "rows",
                "ordered_exact_key_sha256",
                "key_primary_target_sha256",
                "cancer_counts",
                "pair_folds",
            ):
                if output_contract[key] != source_contract[key]:
                    raise PrimaryFoldViewAuditError(
                        f"Fold {outer}/test sealed-source/output mismatch: {key}"
                    )
            for key in (
                "rows",
                "ordered_exact_key_sha256",
                "key_primary_target_sha256",
                "cancer_counts",
                "sha256",
            ):
                if declared_view.get(key) != output_contract[key]:
                    raise PrimaryFoldViewAuditError(
                        f"Fold {outer}/test manifest/output mismatch: {key}"
                    )
            selected_union += int(output_contract["rows"])
            audit_records.append(
                {
                    "outer_pair_fold": outer,
                    "split": "test",
                    "path": str(view_path),
                    **output_contract,
                    "source_values_recomputed_from_hash_bound_sealed_frame": True,
                    "training_payload_test_batches_read": False,
                }
            )
        if selected_union != expected_rows:
            raise PrimaryFoldViewAuditError(f"Outer fold {outer} split union is incomplete")
        source_records.append(
            {
                "patient_fold": outer,
                "path": str(source_path),
                "sha256": source_sha,
                "all_three_source_splits_equal_candidate_authority": not sealed_mode,
                "train_validation_source_splits_equal_candidate_authority": True,
                "sealed_test_source_bound_to_acceptance": sealed_mode,
                "training_payload_splits": (
                    ["train", "validation"] if sealed_mode else ["train", "validation", "test"]
                ),
                "test_batches_absent_from_training_payload": sealed_mode,
                "train_discovery_features_bitwise_identical_across_label_views": True,
                "feature_parity": feature_parity,
            }
        )
        del payload
        gc.collect()
    report = {
        "format": AUDIT_FORMAT,
        "status": "PASS_PRIMARY_VIEWS_ONLY_NOT_ARM_TRAINING_AUTHORITY",
        "analysis_version": "CancerLncAtlas_V3.2_ROUTING_FAIR_FOLD_VIEWS",
        "extraction_manifest": {
            "path": str(manifest_path),
            "sha256": artifact_sha256(manifest_path),
        },
        "candidate_authority": {
            "path": str(candidate_path),
            **candidate,
        },
        "budget_contract": {
            "path": str(budget_path),
            "sha256": artifact_sha256(budget_path),
            "budget_id": budget["budget_id"],
        },
        "source_prepared_folds": source_records,
        "views": audit_records,
        "view_count": len(audit_records),
        "same_outer_source_fold_for_train_validation_test": True,
        "source_train_validation_test_exact_keys_recomputed": True,
        "sealed_test_acceptance": sealed_declaration,
        "test_source_is_post_winner_lock_sealed_inference": sealed_mode,
        "training_payload_test_batches_read": False if sealed_mode else True,
        "train_discovery_features_bitwise_identical_across_label_views": True,
        "pair_split_recomputed_from_pinned_seed": True,
        "independent_pair_fold_implementation": True,
        "source_primary_probability_recomputed_with_same_float32_sigmoid": True,
        "source_target_recomputed_from_proxy_label": True,
        "outer_test_pair_keys_disjoint_from_fit_and_early_stop_keys": True,
        "outer_test_labels_present_only_in_test_view": True,
        "old_prediction_or_checkpoint_feature_detected": False,
        "training_run": False,
        "comparison_metrics_computed": False,
        "modalities_assessed_by_this_audit": False,
        "arm_consumers_assessed_by_this_audit": False,
        "article_fair_comparison_authorized": False,
    }
    _atomic_json(staging / "AUDIT.json", report)
    marker = {
        "format": AUDIT_FORMAT,
        "status": report["status"],
        "audit_path": str(output / "AUDIT.json"),
        "audit_sha256": artifact_sha256(staging / "AUDIT.json"),
        "primary_views_passed": True,
        "modalities_passed": False,
        "arm_training_authorized": False,
        "success_written_last": True,
    }
    _atomic_json(staging / "SUCCESS.json", marker)
    os.replace(staging, output)
    return marker


__all__ = [
    "AUDIT_FORMAT",
    "PrimaryFoldViewAuditError",
    "_independent_pair_fold",
    "audit_primary_fold_views",
]
