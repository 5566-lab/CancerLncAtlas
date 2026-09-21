"""Extract fold-specific primary views from the fresh V3.2 prepared payloads.

The routed architecture comparison cannot be trained from one pair-aggregated
table.  For outer pair fold ``f`` both arms must consume values from the same
explicit G0/G1/G2 ``PATIENT_FOLD_f.pt`` payload plus the separately locked
sealed-test materialization:

* training pairs use that payload's ``train_batches``;
* the inner validation pair fold uses ``validation_batches``; and
* the outer test pair fold comes only from
  ``WINNER_LOCK_SEALED_TEST_INFERENCE_ACCEPTANCE.json``.

This module performs only immutable input extraction.  It neither trains a
model nor emits a release ``SUCCESS.json``.  A separate independent audit must
approve the extracted views before either architecture may consume them.
"""
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

from .multimodal_fusion import TARGET_KEYS, artifact_sha256, pair_blocked_fold
from .training import PREPARED_FORMAT


N_FOLDS = 5
FOLD_COLUMN = "fusion_pair_fold"
EXTRACTION_FORMAT = "CC_HHGT_V3_2_PRIMARY_FOLD_VIEWS_V1"
VIEW_COLUMNS = (
    *TARGET_KEYS,
    FOLD_COLUMN,
    "primary_probability",
    "fusion_target",
    "source_patient_fold",
    "source_split",
)


class PrimaryFoldViewError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_budget(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimaryFoldViewError(f"Unreadable fair budget: {path}") from exc
    if value.get("format") != "CC_HHGT_V3_2_ROUTING_FAIR_BUDGET_V1":
        raise PrimaryFoldViewError("Fair budget format drift")
    policy = value.get("selection_policy", {})
    if (
        int(policy.get("outer_folds", -1)) != N_FOLDS
        or int(policy.get("validation_offset", -1)) != 1
        or policy.get("selection_scope") != "INNER_VALIDATION_FOLD_ONLY"
        or int(policy.get("outer_test_queries_during_selection", -1)) != 0
        or int(policy.get("outer_test_evaluations_per_arm_fold", -1)) != 1
    ):
        raise PrimaryFoldViewError("Fair budget split policy drift")
    seeds = value.get("seeds", {})
    for key in ("optimizer_seed", "pair_fold_seed", "patient_fold_seed"):
        if key not in seeds:
            raise PrimaryFoldViewError(f"Fair budget lacks seed: {key}")
    scope = value.get("scope", {})
    cancers = [str(item).upper() for item in scope.get("formal_cancers", [])]
    if len(cancers) != 33 or len(set(cancers)) != 33:
        raise PrimaryFoldViewError("Fair budget is not the exact 33-cancer scope")
    if int(scope.get("candidate_rows_per_cancer", -1)) <= 0:
        raise PrimaryFoldViewError("Fair budget candidate row count is invalid")
    return value


def _read_sealed_test_acceptance(
    path: Path,
    *,
    candidate_authority: Mapping[str, Any],
    pair_seed: int,
    expected_graph_variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from .patient_fold_authority import (
        FROZEN_V32_RECEIPT_SHA256,
        FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    )
    from .sealed_test_inference import (
        ACCEPTANCE_FORMAT,
        ACCEPTANCE_STATUS,
        PRIMARY_COLUMNS,
    )

    try:
        acceptance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimaryFoldViewError(f"Unreadable sealed-test acceptance: {path}") from exc
    expected = {
        "format": ACCEPTANCE_FORMAT,
        "status": ACCEPTANCE_STATUS,
        "graph_variant": expected_graph_variant,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "candidate_authority_sha256": candidate_authority["sha256"],
        "pair_fold_seed": int(pair_seed),
        "test_labels_absent_from_training_payload": True,
        "sealed_payloads_opened_only_after_winner_lock": True,
        "test_metrics_used_for_winner_selection": False,
        "winner_or_checkpoint_changed_by_test": False,
        "old_fold_outputs_used": False,
        "artifacts_read_only": True,
    }
    drift = {
        key: {"observed": acceptance.get(key), "required": value}
        for key, value in expected.items()
        if acceptance.get(key) != value
    }
    if drift:
        raise PrimaryFoldViewError(f"Sealed-test acceptance contract drift: {drift}")
    declaration = acceptance.get("primary_fusion_frame")
    if not isinstance(declaration, Mapping):
        raise PrimaryFoldViewError("Sealed-test acceptance lacks primary fusion frame")
    primary_path = Path(str(declaration.get("path", "")))
    if not primary_path.is_absolute():
        primary_path = path.parent / primary_path
    if primary_path.is_symlink() or not primary_path.is_file():
        raise PrimaryFoldViewError(f"Missing sealed primary fusion frame: {primary_path}")
    primary_path = primary_path.resolve()
    if artifact_sha256(primary_path) != declaration.get("sha256"):
        raise PrimaryFoldViewError("Sealed primary fusion frame SHA drift")
    frame = pd.read_parquet(primary_path)
    if missing := sorted(set(PRIMARY_COLUMNS) - set(frame.columns)):
        raise PrimaryFoldViewError(f"Sealed primary fusion frame lacks: {missing}")
    frame = frame[list(PRIMARY_COLUMNS)].copy()
    frame["cancer_id"] = frame.cancer_id.astype(str).str.upper()
    for column in ("lncrna_id", "pathway_id"):
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)
    key_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    previous = None
    previous, rows = _strict_key_stream(
        frame[list(TARGET_KEYS)].itertuples(index=False, name=None),
        label="sealed primary fusion frame",
        key_digest=key_digest,
        cancer_counts=counts,
        previous=previous,
    )
    if (
        rows != candidate_authority["rows"]
        or key_digest.hexdigest()
        != candidate_authority["ordered_exact_candidate_key_sha256"]
        or dict(counts) != candidate_authority["cancer_counts"]
    ):
        raise PrimaryFoldViewError("Sealed primary fusion frame candidate universe drift")
    observed_folds = pd.to_numeric(frame[FOLD_COLUMN], errors="raise").astype(int)
    source_folds = pd.to_numeric(frame.source_patient_fold, errors="raise").astype(int)
    expected_folds = np.fromiter(
        (
            pair_blocked_fold(lnc, pathway, seed=pair_seed)
            for lnc, pathway in frame[["lncrna_id", "pathway_id"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=np.int8,
        count=len(frame),
    )
    if (
        not np.array_equal(observed_folds.to_numpy(), expected_folds.astype(int))
        or not np.array_equal(source_folds.to_numpy(), expected_folds.astype(int))
        or not frame.source_split.astype(str).eq("test").all()
    ):
        raise PrimaryFoldViewError("Sealed primary fusion frame fold/source lineage drift")
    probability = pd.to_numeric(frame.primary_probability, errors="raise").to_numpy(float)
    target = pd.to_numeric(frame.fusion_target, errors="raise").to_numpy(float)
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
        or not np.isfinite(target).all()
        or not np.isin(target, [0.0, 1.0]).all()
    ):
        raise PrimaryFoldViewError("Sealed primary probability/target domain drift")
    return frame, {
        "path": str(path),
        "sha256": artifact_sha256(path),
        "primary_fusion_frame": {
            "path": str(primary_path),
            "sha256": artifact_sha256(primary_path),
            "rows": int(len(frame)),
        },
        "winner_declaration_sha256": acceptance["winner_declaration_sha256"],
        "graph_variant": acceptance["graph_variant"],
    }


def _key_bytes(cancer: object, lnc: object, pathway: object) -> bytes:
    return f"{cancer}\t{lnc}\t{pathway}".encode("utf-8")


def _strict_key_stream(
    rows: Iterable[tuple[object, object, object]],
    *,
    label: str,
    key_digest: "hashlib._Hash",
    cancer_counts: Counter[str],
    previous: tuple[str, str, str] | None,
) -> tuple[tuple[str, str, str] | None, int]:
    count = 0
    for raw_cancer, raw_lnc, raw_pathway in rows:
        key = (str(raw_cancer).upper(), str(raw_lnc), str(raw_pathway))
        if not all(key) or key[0] != str(raw_cancer):
            raise PrimaryFoldViewError(f"{label} contains a non-canonical exact key")
        if previous is not None and key <= previous:
            raise PrimaryFoldViewError(f"{label} is not strict exact-key order")
        previous = key
        key_digest.update(_key_bytes(*key) + b"\n")
        cancer_counts[key[0]] += 1
        count += 1
    return previous, count


def _candidate_authority_contract(
    path: Path, *, cancers: Sequence[str], rows_per_cancer: int
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    missing = sorted(set(TARGET_KEYS) - set(parquet.schema_arrow.names))
    if missing:
        raise PrimaryFoldViewError(f"Candidate authority lacks exact keys: {missing}")
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    previous: tuple[str, str, str] | None = None
    rows = 0
    for batch in parquet.iter_batches(columns=list(TARGET_KEYS), batch_size=100_000):
        frame = batch.to_pandas()
        previous, added = _strict_key_stream(
            frame.itertuples(index=False, name=None),
            label="candidate authority",
            key_digest=digest,
            cancer_counts=counts,
            previous=previous,
        )
        rows += added
    expected_counts = {str(cancer): int(rows_per_cancer) for cancer in cancers}
    if rows != rows_per_cancer * len(cancers) or dict(counts) != expected_counts:
        raise PrimaryFoldViewError(
            f"Candidate authority scope drift: rows={rows}, counts={dict(counts)}"
        )
    return {
        "path": str(path),
        "sha256": artifact_sha256(path),
        "rows": rows,
        "ordered_exact_candidate_key_sha256": digest.hexdigest(),
        "cancer_counts": expected_counts,
    }


def _default_payload_loader(path: Path) -> Mapping[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _numpy(value: Any, *, label: str) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive around tensor libraries
        raise PrimaryFoldViewError(f"Cannot convert {label} to a CPU array") from exc


def _sigmoid_float32(value: Any) -> np.ndarray:
    # Formal payloads are torch float32 tensors.  Using torch.sigmoid here is
    # important: the hierarchical inference path uses that exact operation.
    if hasattr(value, "detach"):
        import torch

        tensor = value.detach().cpu()
        if tensor.ndim != 1:
            raise PrimaryFoldViewError("base_logit is not one-dimensional")
        return torch.sigmoid(tensor).numpy().astype(np.float32, copy=False)
    array = np.asarray(value)
    if array.ndim != 1:
        raise PrimaryFoldViewError("base_logit is not one-dimensional")
    array = array.astype(np.float64, copy=False)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output.astype(np.float32)


def _inverse_node_maps(bundle: Any) -> dict[str, dict[int, str]]:
    maps = getattr(bundle, "node_maps", None)
    if not isinstance(maps, Mapping):
        raise PrimaryFoldViewError("Prepared graph bundle lacks node_maps")
    result: dict[str, dict[int, str]] = {}
    for kind in ("cancer", "lncRNA", "pathway"):
        if kind not in maps or not isinstance(maps[kind], Mapping):
            raise PrimaryFoldViewError(f"Prepared graph lacks {kind} node map")
        inverse = {int(index): str(identifier) for identifier, index in maps[kind].items()}
        if len(inverse) != len(maps[kind]):
            raise PrimaryFoldViewError(f"Prepared graph {kind} node map is not one-to-one")
        result[kind] = inverse
    return result


def _batch_arrays(
    raw: Mapping[str, Any], inverse: Mapping[str, Mapping[int, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = raw.get("candidate_batch")
    if not isinstance(candidate, Mapping):
        raise PrimaryFoldViewError("Prepared batch lacks candidate_batch")
    indices: dict[str, np.ndarray] = {}
    for key in ("l", "p", "c"):
        if key not in candidate:
            raise PrimaryFoldViewError(f"Prepared candidate batch lacks {key}")
        array = _numpy(candidate[key], label=f"candidate_batch.{key}")
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise PrimaryFoldViewError(f"candidate_batch.{key} is not an integer vector")
        indices[key] = array.astype(np.int64, copy=False)
    size = len(indices["l"])
    if any(len(indices[key]) != size for key in ("p", "c")):
        raise PrimaryFoldViewError("Prepared candidate indices differ in length")
    try:
        cancers = np.asarray(
            [inverse["cancer"][int(item)].upper() for item in indices["c"]], dtype=object
        )
        lncs = np.asarray(
            [inverse["lncRNA"][int(item)] for item in indices["l"]], dtype=object
        )
        pathways = np.asarray(
            [inverse["pathway"][int(item)] for item in indices["p"]], dtype=object
        )
    except KeyError as exc:
        raise PrimaryFoldViewError("Prepared batch references an absent graph node") from exc
    probability = _sigmoid_float32(raw.get("base_logit"))
    target = _numpy(raw.get("proxy_label"), label="proxy_label")
    if target.ndim != 1 or len(target) != size or len(probability) != size:
        raise PrimaryFoldViewError("Prepared primary arrays differ from candidate length")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise PrimaryFoldViewError("Prepared primary probability is invalid")
    target_float = target.astype(np.float64, copy=False)
    if not np.isfinite(target_float).all() or not np.isin(target_float, [0.0, 1.0]).all():
        raise PrimaryFoldViewError("Prepared proxy label is not binary")
    return cancers, lncs, pathways, probability, target_float.astype(np.uint8)


def _keep_mask(pair_folds: np.ndarray, *, source_split: str, outer_fold: int) -> np.ndarray:
    validation_fold = (outer_fold + 1) % N_FOLDS
    if source_split == "train":
        return ~np.isin(pair_folds, [outer_fold, validation_fold])
    if source_split == "validation":
        return pair_folds == validation_fold
    if source_split == "test":
        return pair_folds == outer_fold
    raise PrimaryFoldViewError(f"Unknown prepared split: {source_split}")


def _update_value_digest(
    digest: "hashlib._Hash",
    cancers: np.ndarray,
    lncs: np.ndarray,
    pathways: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
) -> None:
    for cancer, lnc, pathway, prob, label in zip(
        cancers, lncs, pathways, probability, target
    ):
        digest.update(_key_bytes(cancer, lnc, pathway))
        digest.update(b"\t" + struct.pack("<fB", float(prob), int(label)) + b"\n")


def _validate_payload_metadata(
    payload: Mapping[str, Any],
    *,
    fold: int,
    expected_rows: int,
    rows_per_cancer: int,
    allow_legacy_embedded_test: bool = False,
) -> None:
    expected_format = (
        "CC_HHGT_V3_2_PREPARED_FOLD_V1"
        if allow_legacy_embedded_test
        else PREPARED_FORMAT
    )
    if (
        payload.get("prepared_format") != expected_format
        or int(payload.get("patient_fold", -1)) != fold
    ):
        raise PrimaryFoldViewError(f"Prepared fold {fold} identity/format drift")
    if payload.get("contains_optimizer_state") is not False:
        raise PrimaryFoldViewError(f"Prepared fold {fold} contains optimizer state")
    if payload.get("contains_trained_parameters") is not False:
        raise PrimaryFoldViewError(f"Prepared fold {fold} contains trained parameters")
    scope = payload.get("input_scope")
    if not isinstance(scope, Mapping):
        raise PrimaryFoldViewError(f"Prepared fold {fold} lacks input_scope")
    if (
        int(scope.get("candidate_rows_per_split", -1)) != expected_rows
        or int(scope.get("budget_per_cancer", -1)) != rows_per_cancer
        or scope.get("discovery_features_split") != "train_patients_only"
        or scope.get("held_out_effects_as_features") is not False
        or set(scope.get("replication_labels_splits", []))
        != ({"validation", "test"} if allow_legacy_embedded_test else {"validation"})
    ):
        raise PrimaryFoldViewError(f"Prepared fold {fold} leakage/scope contract drift")
    if not allow_legacy_embedded_test and any(
        str(key).lower().startswith("test") for key in payload
    ):
        raise PrimaryFoldViewError(
            f"Prepared fold {fold} crosses the sealed-test firewall"
        )
    hashes = payload.get("artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise PrimaryFoldViewError(f"Prepared fold {fold} lacks artifact hashes")


def extract_primary_fold_views(
    *,
    prepared_root: str | Path,
    candidate_authority_path: str | Path,
    budget_contract_path: str | Path,
    output_root: str | Path,
    sealed_test_acceptance_path: str | Path | None = None,
    expected_graph_variant: str = "G2",
    payload_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract five outer-fold train/validation/test primary authorities.

    ``payload_loader`` exists solely to permit small non-Torch contract tests.
    Formal callers must omit it so that the exact torch payload is loaded.
    """

    prepared = Path(prepared_root).resolve()
    candidate_path = Path(candidate_authority_path).resolve()
    budget_path = Path(budget_contract_path).resolve()
    output = Path(output_root).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise PrimaryFoldViewError(f"Primary fold extraction refuses output reuse: {output}")
    for label, path in (
        ("candidate authority", candidate_path),
        ("fair budget", budget_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise PrimaryFoldViewError(f"Missing {label}: {path}")
    budget = _read_budget(budget_path)
    cancers = [str(item).upper() for item in budget["scope"]["formal_cancers"]]
    rows_per_cancer = int(budget["scope"]["candidate_rows_per_cancer"])
    expected_rows = rows_per_cancer * len(cancers)
    pair_seed = int(budget["seeds"]["pair_fold_seed"])
    authority = _candidate_authority_contract(
        candidate_path, cancers=cancers, rows_per_cancer=rows_per_cancer
    )
    sealed_frame: pd.DataFrame | None = None
    sealed_record: dict[str, Any] | None = None
    if sealed_test_acceptance_path is not None:
        acceptance_path = Path(sealed_test_acceptance_path).resolve()
        sealed_frame, sealed_record = _read_sealed_test_acceptance(
            acceptance_path,
            candidate_authority=authority,
            pair_seed=pair_seed,
            expected_graph_variant=expected_graph_variant,
        )
    elif payload_loader is None:
        raise PrimaryFoldViewError(
            "Formal extraction requires winner-lock sealed-test acceptance; "
            "training payload test_batches are forbidden"
        )
    loader = payload_loader or _default_payload_loader
    prepared_patient_binding: dict[str, Any] | None = None
    if payload_loader is None:
        from .patient_fold_authority import (
            FROZEN_V32_RECEIPT_SHA256,
            FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            validate_frozen_v32_prepared_fold_binding,
        )

        prepared_patient_binding = validate_frozen_v32_prepared_fold_binding(prepared)
        try:
            variant_marker = json.loads(
                (prepared / "FORMAL_GRAPH_VARIANT.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PrimaryFoldViewError("Prepared root lacks graph-variant marker") from exc
        if (
            variant_marker.get("format")
            != "CANCERLNCATLAS_V32_FORMAL_GRAPH_VARIANT_ROOT_V1"
            or variant_marker.get("variant") != expected_graph_variant
            or variant_marker.get("legacy_root_fold_payloads_allowed") is not False
        ):
            raise PrimaryFoldViewError("Prepared root graph variant drift")
    staging.mkdir(parents=True)
    if prepared_patient_binding is not None:
        for filename in (
            "SAMPLE_PATIENT_FOLD_MAP.tsv",
            "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
            "PATIENT_FOLD_BINDING.json",
        ):
            (staging / filename).write_bytes((prepared / filename).read_bytes())
    records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for outer_fold in range(N_FOLDS):
            source_path = prepared / f"PATIENT_FOLD_{outer_fold}.pt"
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                raise PrimaryFoldViewError(f"Missing prepared fold: {source_path}")
            source_sha_before = artifact_sha256(source_path)
            payload = loader(source_path)
            if not isinstance(payload, Mapping):
                raise PrimaryFoldViewError(f"Prepared fold {outer_fold} is not a mapping")
            legacy_embedded_test = (
                payload_loader is not None
                and sealed_frame is None
                and "test_batches" in payload
            )
            _validate_payload_metadata(
                payload,
                fold=outer_fold,
                expected_rows=expected_rows,
                rows_per_cancer=rows_per_cancer,
                allow_legacy_embedded_test=legacy_embedded_test,
            )
            if not legacy_embedded_test and payload.get("formal_graph_variant") != expected_graph_variant:
                raise PrimaryFoldViewError(
                    f"Prepared fold {outer_fold} graph variant drift"
                )
            if prepared_patient_binding is not None:
                payload_binding = payload.get("patient_fold_authority")
                if (
                    not isinstance(payload_binding, Mapping)
                    or payload_binding.get("status")
                    != "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY"
                    or payload_binding.get("sample_id_patient_fallback_used") is not False
                    or payload_binding.get("legacy_patient_fold_manifest_used") is not False
                    or payload_binding.get("sample_patient_fold_map", {}).get("sha256")
                    != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
                    or payload_binding.get("authority_receipt", {}).get("sha256")
                    != FROZEN_V32_RECEIPT_SHA256
                ):
                    raise PrimaryFoldViewError(
                        f"Prepared fold {outer_fold} lacks the frozen patient authority binding"
                    )
            inverse = _inverse_node_maps(payload["bundle"])
            source_records.append(
                {
                    "patient_fold": outer_fold,
                    "path": str(source_path),
                    "bytes": int(source_path.stat().st_size),
                    "sha256": source_sha_before,
                    "artifact_hashes": dict(payload["artifact_hashes"]),
                }
            )
            fold_selected_rows = 0
            source_splits = ("train", "validation") + (
                ("test",) if legacy_embedded_test else ()
            )
            for source_split in source_splits:
                batches = payload.get(f"{source_split}_batches")
                if not isinstance(batches, Sequence) or not batches:
                    raise PrimaryFoldViewError(
                        f"Prepared fold {outer_fold}/{source_split} lacks batches"
                    )
                destination = (
                    staging
                    / f"outer_pair_fold={outer_fold}"
                    / f"split={source_split}"
                    / "part-0.parquet"
                )
                destination.parent.mkdir(parents=True)
                writer = None
                source_key_digest = hashlib.sha256()
                selected_key_digest = hashlib.sha256()
                value_digest = hashlib.sha256()
                source_counts: Counter[str] = Counter()
                selected_counts: Counter[str] = Counter()
                pair_counts: Counter[int] = Counter()
                source_previous: tuple[str, str, str] | None = None
                selected_previous: tuple[str, str, str] | None = None
                source_rows = 0
                selected_rows = 0
                try:
                    for raw in batches:
                        if not isinstance(raw, Mapping):
                            raise PrimaryFoldViewError("Prepared batch is not a mapping")
                        cancer, lnc, pathway, probability, target = _batch_arrays(raw, inverse)
                        source_frame = pd.DataFrame(
                            {
                                "cancer_id": cancer,
                                "lncrna_id": lnc,
                                "pathway_id": pathway,
                            }
                        )
                        source_previous, added = _strict_key_stream(
                            source_frame.itertuples(index=False, name=None),
                            label=f"fold {outer_fold}/{source_split} source",
                            key_digest=source_key_digest,
                            cancer_counts=source_counts,
                            previous=source_previous,
                        )
                        source_rows += added
                        pair_folds = np.fromiter(
                            (
                                pair_blocked_fold(left, right, seed=pair_seed)
                                for left, right in zip(lnc, pathway)
                            ),
                            dtype=np.int8,
                            count=len(lnc),
                        )
                        keep = _keep_mask(
                            pair_folds,
                            source_split=source_split,
                            outer_fold=outer_fold,
                        )
                        if not keep.any():
                            continue
                        kept_cancer = cancer[keep]
                        kept_lnc = lnc[keep]
                        kept_pathway = pathway[keep]
                        kept_probability = probability[keep]
                        kept_target = target[keep]
                        kept_folds = pair_folds[keep]
                        selected_frame = pd.DataFrame(
                            {
                                "cancer_id": kept_cancer,
                                "lncrna_id": kept_lnc,
                                "pathway_id": kept_pathway,
                            }
                        )
                        selected_previous, added = _strict_key_stream(
                            selected_frame.itertuples(index=False, name=None),
                            label=f"fold {outer_fold}/{source_split} selected",
                            key_digest=selected_key_digest,
                            cancer_counts=selected_counts,
                            previous=selected_previous,
                        )
                        selected_rows += added
                        pair_counts.update(map(int, kept_folds.tolist()))
                        _update_value_digest(
                            value_digest,
                            kept_cancer,
                            kept_lnc,
                            kept_pathway,
                            kept_probability,
                            kept_target,
                        )
                        view = selected_frame
                        view[FOLD_COLUMN] = kept_folds.astype(np.int8, copy=False)
                        view["primary_probability"] = kept_probability.astype(
                            np.float32, copy=False
                        )
                        view["fusion_target"] = kept_target.astype(np.uint8, copy=False)
                        view["source_patient_fold"] = np.int8(outer_fold)
                        view["source_split"] = source_split
                        table = pa.Table.from_pandas(view[list(VIEW_COLUMNS)], preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                        writer.write_table(table, row_group_size=len(view))
                finally:
                    if writer is not None:
                        writer.close()
                if writer is None:
                    raise PrimaryFoldViewError(
                        f"Fold {outer_fold}/{source_split} selected no candidates"
                    )
                if (
                    source_rows != expected_rows
                    or source_key_digest.hexdigest()
                    != authority["ordered_exact_candidate_key_sha256"]
                    or dict(source_counts) != authority["cancer_counts"]
                ):
                    raise PrimaryFoldViewError(
                        f"Prepared fold {outer_fold}/{source_split} candidate universe drift"
                    )
                expected_pair_folds = (
                    set(range(N_FOLDS)) - {outer_fold, (outer_fold + 1) % N_FOLDS}
                    if source_split == "train"
                    else {
                        (outer_fold + 1) % N_FOLDS
                        if source_split == "validation"
                        else outer_fold
                    }
                )
                if set(pair_counts) != expected_pair_folds:
                    raise PrimaryFoldViewError(
                        f"Fold {outer_fold}/{source_split} pair-fold membership drift"
                    )
                fold_selected_rows += selected_rows
                final_path = output / destination.relative_to(staging)
                records.append(
                    {
                        "outer_pair_fold": outer_fold,
                        "source_patient_fold": outer_fold,
                        "split": source_split,
                        "pair_folds": sorted(expected_pair_folds),
                        "path": str(final_path),
                        "bytes": int(destination.stat().st_size),
                        "sha256": artifact_sha256(destination),
                        "rows": selected_rows,
                        "cancer_counts": dict(sorted(selected_counts.items())),
                        "ordered_exact_key_sha256": selected_key_digest.hexdigest(),
                        "key_primary_target_sha256": value_digest.hexdigest(),
                        "value_source": (
                            "LEGACY_EMBEDDED_TEST_FIXTURE"
                            if source_split == "test"
                            else "TRAINING_PREPARED_PAYLOAD"
                        ),
                    }
                )
            if not legacy_embedded_test:
                if sealed_frame is None:
                    raise PrimaryFoldViewError(
                        "Prepared training payload has no test batches and no sealed acceptance"
                    )
                source_split = "test"
                selected = sealed_frame.loc[
                    pd.to_numeric(sealed_frame[FOLD_COLUMN], errors="raise")
                    .astype(int)
                    .eq(outer_fold)
                ].copy()
                if selected.empty:
                    raise PrimaryFoldViewError(
                        f"Sealed primary frame has no test rows for fold {outer_fold}"
                    )
                selected["source_patient_fold"] = np.int8(outer_fold)
                selected["source_split"] = source_split
                destination = (
                    staging
                    / f"outer_pair_fold={outer_fold}"
                    / "split=test"
                    / "part-0.parquet"
                )
                destination.parent.mkdir(parents=True)
                selected = selected[list(VIEW_COLUMNS)].sort_values(
                    list(TARGET_KEYS), kind="stable"
                ).reset_index(drop=True)
                table = pa.Table.from_pandas(selected, preserve_index=False)
                pq.write_table(table, destination, compression="zstd", row_group_size=100_000)
                selected_key_digest = hashlib.sha256()
                value_digest = hashlib.sha256()
                selected_counts: Counter[str] = Counter()
                previous = None
                previous, selected_rows = _strict_key_stream(
                    selected[list(TARGET_KEYS)].itertuples(index=False, name=None),
                    label=f"fold {outer_fold}/test sealed selected",
                    key_digest=selected_key_digest,
                    cancer_counts=selected_counts,
                    previous=previous,
                )
                _update_value_digest(
                    value_digest,
                    selected.cancer_id.to_numpy(object),
                    selected.lncrna_id.to_numpy(object),
                    selected.pathway_id.to_numpy(object),
                    selected.primary_probability.to_numpy(np.float32),
                    selected.fusion_target.to_numpy(np.uint8),
                )
                fold_selected_rows += selected_rows
                final_path = output / destination.relative_to(staging)
                records.append(
                    {
                        "outer_pair_fold": outer_fold,
                        "source_patient_fold": outer_fold,
                        "split": "test",
                        "pair_folds": [outer_fold],
                        "path": str(final_path),
                        "bytes": int(destination.stat().st_size),
                        "sha256": artifact_sha256(destination),
                        "rows": selected_rows,
                        "cancer_counts": dict(sorted(selected_counts.items())),
                        "ordered_exact_key_sha256": selected_key_digest.hexdigest(),
                        "key_primary_target_sha256": value_digest.hexdigest(),
                        "value_source": "POST_WINNER_LOCK_SEALED_TEST_INFERENCE",
                    }
                )
            if fold_selected_rows != expected_rows:
                raise PrimaryFoldViewError(
                    f"Outer fold {outer_fold} split union is incomplete: {fold_selected_rows}"
                )
            if artifact_sha256(source_path) != source_sha_before:
                raise PrimaryFoldViewError(f"Prepared fold mutated during extraction: {source_path}")
            del payload, inverse
            gc.collect()
        manifest = {
            "format": EXTRACTION_FORMAT,
            "status": "EXTRACTED_AWAITING_INDEPENDENT_AUDIT_NO_TRAINING",
            "analysis_version": "CancerLncAtlas_V3.2_ROUTING_FAIR_FOLD_VIEWS",
            "candidate_authority": authority,
            "budget_contract": {
                "path": str(budget_path),
                "sha256": artifact_sha256(budget_path),
                "budget_id": budget["budget_id"],
            },
            "pair_fold_seed": pair_seed,
            "patient_fold_seed": int(budget["seeds"]["patient_fold_seed"]),
            "patient_fold_authority": prepared_patient_binding,
            "graph_variant": expected_graph_variant,
            "sealed_test_acceptance": sealed_record,
            "outer_folds": N_FOLDS,
            "validation_offset": 1,
            "source_prepared_folds": source_records,
            "views": records,
            "view_count": len(records),
            "same_source_patient_fold_for_both_arms_required": True,
            "one_single_aggregated_primary_table_is_fair_training_authority": False,
            "outer_test_labels_may_enter_training_or_early_stopping": False,
            "training_payload_test_batches_present": bool(
                payload_loader is not None and sealed_record is None
            ),
            "test_view_source": (
                "POST_WINNER_LOCK_SEALED_TEST_INFERENCE"
                if sealed_record is not None
                else "LEGACY_TEST_FIXTURE_ONLY"
            ),
            "test_labels_absent_from_training_payload": sealed_record is not None,
            "old_predictions_used": False,
            "training_run": False,
            "winner_selection_run": False,
            "formal_release_success_emitted": False,
        }
        _atomic_json(staging / "EXTRACTION_MANIFEST.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        # Keep a partial staging directory for forensic inspection.  It has no
        # SUCCESS marker and can never be mistaken for an authority.
        raise


__all__ = [
    "EXTRACTION_FORMAT",
    "FOLD_COLUMN",
    "N_FOLDS",
    "PrimaryFoldViewError",
    "VIEW_COLUMNS",
    "extract_primary_fold_views",
]
