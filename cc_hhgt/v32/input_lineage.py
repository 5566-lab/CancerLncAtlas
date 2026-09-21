"""Read-only lineage gate for inputs consumed by newly trained V3.2 models.

The gate deliberately distinguishes reusable source data from historical model
results.  A historical result does not become an input merely because it was
renamed ``standardized_input``: both declared roles and inspectable table
columns are checked.  The module has no writer and never repairs an artifact.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INPUT_LINEAGE_POLICY_VERSION = "CancerLncAtlas_V3.2_input_lineage_v1"

_ROLE_ALIASES = {
    "raw": "raw_data",
    "raw_input": "raw_data",
    "raw_data": "raw_data",
    "annotation": "static_annotation",
    "static": "static_annotation",
    "static_annotation": "static_annotation",
    "standardized": "standardized_input",
    "standardised_input": "standardized_input",
    "standardized_input": "standardized_input",
    "restandardized_input": "standardized_input",
    "re_standardized_input": "standardized_input",
    "split_manifest": "split_manifest",
    "training_label": "training_label",
    "v32_core_checkpoint": "v32_core_checkpoint",
}
ALLOWED_SOURCE_ROLES = frozenset(_ROLE_ALIASES.values())

_USE_ROLE_ALIASES = {
    "primary_input": "core_input",
    "core_input": "core_input",
    "auxiliary_input": "aux_input",
    "aux_input": "aux_input",
    "aux_parent": "aux_parent",
    "parent_checkpoint": "aux_parent",
    "training_target": "training_target",
    "split_control": "split_control",
}
ALLOWED_USE_ROLES = frozenset(_USE_ROLE_ALIASES.values())

# These are result roles, not reusable data roles.  Matching is token based so
# variants such as ``historical_oof_prediction`` fail closed as well.
_FORBIDDEN_ROLE_TOKENS = (
    "checkpoint",
    "embedding",
    "oof",
    "prediction",
    "probability",
    "ranking",
    "web_table",
    "release_table",
    "public_output",
    "model_output",
)

# Old-generation standardized inputs receive the intentionally broad blacklist
# requested by the V3.2 retraining contract.  Static annotations are not
# subject to this list (for example, a pathway membership weight is valid).
_OLD_STANDARDIZED_EXACT_BLACKLIST = frozenset(
    {
        "label",
        "label_class",
        "proxy_label",
        "association_proxy_label",
        "strong_association_label",
        "strong_evidence_label",
        "held_out_proxy_label",
        "sample_weight",
        "score",
        "prediction",
        "probability",
        "ranking",
        "rank",
        "oof",
        "logit",
        "calibrated_probability",
        "association_membership_probability",
        "regulatory_evidence_confidence",
        "fold_rank_percentile",
        "fold_selection_frequency",
    }
)
_OLD_STANDARDIZED_SUFFIX_BLACKLIST = (
    "_label",
    "_sample_weight",
    "_score",
    "_prediction",
    "_probability",
    "_ranking",
    "_rank",
    "_logit",
)

# Result semantics that are unsafe even if somebody relabels an old result as
# generation V3.2.  This narrower list does not reject legitimate activity or
# assay scores in freshly standardized source data.
_RESULT_EXACT_COLUMNS = frozenset(
    {
        "association_membership_probability",
        "evidence_confidence_probability",
        "regulatory_evidence_confidence",
        "fold_rank_percentile",
        "fold_selection_frequency",
        "predicted_label",
        "prediction",
        "probability",
        "ranking",
        "rank",
        "rank_score",
        "ranking_score",
        "percentile",
        "oof_prediction",
        "oof_probability",
        "pred",
        "y_pred",
        "y_hat",
        "prob",
        "p_hat",
        "predicted_score",
        "node_embedding",
        "pair_embedding",
    }
)
_RESULT_COLUMN_TOKENS = (
    "historical_probability",
    "legacy_probability",
    "oof_prediction",
    "oof_probability",
    "fold_prediction",
    "model_prediction",
    "predicted_probability",
    "rank_percentile",
    "selection_frequency",
    "_embedding_",
    "embedding_dim",
    "_v2_6",
    "_v2_7",
    "_v2_8",
    "_v2_9",
    "_v3_0",
    "_v3_1",
)

_PAIR_EVIDENCE_FORBIDDEN = frozenset({"label", "sample_weight", "score"})
_CHECKPOINT_SUFFIXES = frozenset(
    {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".joblib", ".pkl", ".pickle"}
)
_TABLE_SUFFIXES = frozenset({".parquet", ".csv", ".tsv", ".txt", ".maf", ".json", ".jsonl", ".ndjson"})


class InputLineageError(RuntimeError):
    """Raised when one or more proposed V3.2 inputs fail the lineage gate."""


@dataclass(frozen=True)
class InputLineageArtifact:
    """Declarative metadata supplied for one immutable input artifact."""

    path: str | Path
    generation: str
    source_role: str
    outcome_derived: bool
    fold_fitted: bool
    use_role: str = "core_input"
    artifact_id: str | None = None
    expected_sha256: str | None = None
    source_target_level: str | None = None
    target_level: str | None = None
    transformation: str | None = None
    mapping_policy: str | None = None
    family_to_exact_broadcast: bool = False


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Return a deterministic content hash for a file or directory tree."""

    source = Path(path)
    if source.is_symlink():
        raise InputLineageError(f"Symlink inputs are not admitted: {source}")
    if source.is_file():
        return _file_sha256(source)
    if not source.is_dir():
        raise InputLineageError(f"Input artifact is missing: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise InputLineageError(f"Input artifact directory is empty: {source}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise InputLineageError(f"Symlink inputs are not admitted: {item}")
        relative = item.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_v32_generation(value: Any) -> bool:
    return bool(re.search(r"(?<!\d)v?3[._-]?2(?!\d)", str(value).strip().lower()))


def _table_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] in {".gz", ".bz2", ".xz", ".zip"}:
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def _columns_from_json(path: Path, *, line_delimited: bool) -> list[str]:
    if line_delimited:
        with path.open("r", encoding="utf-8-sig") as handle:
            first = next((line for line in handle if line.strip()), "")
        value = json.loads(first) if first else None
    else:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, list):
            value = value[0] if value else None
    if isinstance(value, Mapping):
        return sorted(map(str, value.keys()))
    return []


def _inspect_file_columns(path: Path) -> tuple[list[str], str]:
    suffix = _table_suffix(path)
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names), "parquet_schema"
    if suffix in {".csv", ".tsv", ".txt", ".maf"}:
        import pandas as pd

        separator = "," if suffix == ".csv" else "\t"
        frame = pd.read_csv(path, sep=separator, nrows=0, comment="#", compression="infer")
        return list(map(str, frame.columns)), "delimited_header"
    if suffix == ".json":
        return _columns_from_json(path, line_delimited=False), "json_first_record"
    if suffix in {".jsonl", ".ndjson"}:
        return _columns_from_json(path, line_delimited=True), "jsonl_first_record"
    return [], "not_tabular"


def _inspect_columns(path: Path) -> tuple[list[str], str, str | None]:
    try:
        if path.is_file():
            columns, mode = _inspect_file_columns(path)
            return sorted(set(columns)), mode, None
        table_files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and _table_suffix(item) in _TABLE_SUFFIXES
        )
        if not table_files:
            return [], "directory_without_tabular_files", None
        columns: set[str] = set()
        modes: set[str] = set()
        for item in table_files:
            local_columns, mode = _inspect_file_columns(item)
            columns.update(local_columns)
            modes.add(mode)
        return sorted(columns), "directory_union:" + ",".join(sorted(modes)), None
    except Exception as exc:  # a lineage audit reports failure; it never repairs
        return [], "schema_error", f"{type(exc).__name__}: {exc}"


def _forbidden_role_kind(role: str) -> str | None:
    token = _normalise_token(role)
    for forbidden in _FORBIDDEN_ROLE_TOKENS:
        if forbidden in token:
            return forbidden
    return None


def _path_result_kind(path: Path) -> str | None:
    name = _normalise_token(path.name)
    if _table_suffix(path) in _CHECKPOINT_SUFFIXES:
        return "checkpoint"
    patterns = {
        "embedding": ("embedding", "embeddings"),
        "oof": ("oof", "out_of_fold", "fold_prediction"),
        "prediction": ("prediction", "predictions", "predicted_probability"),
        "probability": ("probability", "probabilities"),
        "ranking": ("ranking", "ranked_pairs", "ranked_candidates"),
        "web_table": ("web_table", "website_table", "web_export"),
        "release_table": ("release_table", "public_release", "release_predictions"),
    }
    for kind, candidates in patterns.items():
        if any(candidate in name for candidate in candidates):
            return kind
    return None


def _old_standardized_blacklisted(columns: Sequence[str]) -> list[str]:
    bad: list[str] = []
    for original in columns:
        column = _normalise_token(original)
        if column in _OLD_STANDARDIZED_EXACT_BLACKLIST or column.endswith(
            _OLD_STANDARDIZED_SUFFIX_BLACKLIST
        ):
            bad.append(str(original))
    return sorted(set(bad))


def _result_columns(columns: Sequence[str]) -> list[str]:
    bad: list[str] = []
    vector_columns: list[str] = []
    for original in columns:
        column = _normalise_token(original)
        if column in _RESULT_EXACT_COLUMNS or any(token in column for token in _RESULT_COLUMN_TOKENS):
            bad.append(str(original))
        elif column.startswith("embedding_") or column.endswith("_embedding"):
            bad.append(str(original))
        elif re.fullmatch(r"(?:emb|embedding|latent|vector)_?\d+", column):
            vector_columns.append(str(original))
    # A single column named ``vector_1`` can occur in source metadata.  Two or
    # more numbered latent/vector dimensions constitute embedding content.
    if len(vector_columns) >= 2:
        bad.extend(vector_columns)
    return sorted(set(bad))


def _pair_evidence_forbidden(columns: Sequence[str]) -> list[str]:
    bad: list[str] = []
    for original in columns:
        column = _normalise_token(original)
        if column in _PAIR_EVIDENCE_FORBIDDEN or column.endswith(
            ("_label", "_sample_weight", "_score")
        ):
            bad.append(str(original))
    return sorted(set(bad))


def _looks_like_pair_evidence(path: Path, source_role: str, artifact_id: str, columns: Sequence[str]) -> bool:
    joined = "_".join(
        [_normalise_token(path.name), _normalise_token(source_role), _normalise_token(artifact_id)]
    )
    if "pair_evidence" in joined:
        return True
    normalised = {_normalise_token(column) for column in columns}
    return "pair_evidence_id" in normalised or "pair_evidence_score" in normalised


def _family_broadcast_reasons(spec: Mapping[str, Any], columns: Sequence[str]) -> list[str]:
    reasons: list[str] = []
    broadcast_flag = spec.get("family_to_exact_broadcast", False)
    if broadcast_flag is True:
        reasons.append("FAMILY_TO_EXACT_BROADCAST_FORBIDDEN: explicit flag is true")
    elif type(broadcast_flag) is not bool:
        reasons.append("INVALID_FAMILY_BROADCAST_FLAG: an explicit boolean is required")
    descriptor = " ".join(
        str(spec.get(key, ""))
        for key in ("transformation", "mapping_policy", "derivation", "aggregation_policy")
    ).lower()
    descriptor_token = _normalise_token(descriptor)
    if (
        ("broadcast" in descriptor_token and "family" in descriptor_token)
        or "family_to_exact" in descriptor_token
        or ("many_to_many" in descriptor_token and "exact_pathway" in descriptor_token)
    ):
        reasons.append("FAMILY_TO_EXACT_BROADCAST_FORBIDDEN: mapping policy broadcasts family values")
    source_level = _normalise_token(spec.get("source_target_level", ""))
    target_level = _normalise_token(spec.get("target_level", ""))
    if source_level in {"family", "pathway_family", "pathway_family_id"} and target_level in {
        "exact",
        "exact_pathway",
        "pathway_id",
    }:
        reasons.append("FAMILY_TO_EXACT_BROADCAST_FORBIDDEN: family target is mapped to exact pathway")

    normalised = {_normalise_token(column) for column in columns}
    broadcast_columns = sorted(
        column
        for column in normalised
        if column in {"family_to_exact_broadcast", "broadcast_from_family", "source_pathway_family_id"}
    )
    if broadcast_columns:
        reasons.append(
            "FAMILY_TO_EXACT_BROADCAST_FORBIDDEN: broadcast marker columns="
            + repr(broadcast_columns)
        )
    if {"pathway_family_id", "pathway_id"}.issubset(normalised):
        unsafe_values = _old_standardized_blacklisted(columns) + _result_columns(columns)
        if unsafe_values:
            reasons.append(
                "FAMILY_TO_EXACT_BROADCAST_FORBIDDEN: family and exact IDs carry result values="
                + repr(sorted(set(unsafe_values)))
            )
    return reasons


def _as_mapping(spec: InputLineageArtifact | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(spec, InputLineageArtifact):
        return asdict(spec)
    if not isinstance(spec, Mapping):
        raise TypeError("Each lineage artifact must be a mapping or InputLineageArtifact")
    return dict(spec)


def audit_input_artifact(spec: InputLineageArtifact | Mapping[str, Any]) -> dict[str, Any]:
    """Inspect one artifact without mutating it and return a PASS/FAIL record."""

    raw = _as_mapping(spec)
    reasons: list[str] = []
    path_value = raw.get("path")
    path = Path(str(path_value)).expanduser() if path_value not in (None, "") else Path("")
    generation = str(raw.get("generation", "")).strip()
    declared_role = str(raw.get("source_role", "")).strip()
    source_role = _ROLE_ALIASES.get(_normalise_token(declared_role), _normalise_token(declared_role))
    declared_use_role = raw.get(
        "use_role", raw.get("usage_role", raw.get("consumer_role", "core_input"))
    )
    use_role = _USE_ROLE_ALIASES.get(
        _normalise_token(declared_use_role), _normalise_token(declared_use_role)
    )
    artifact_id = str(
        raw.get("artifact_id")
        or (str(path.resolve()) if path_value else "<missing>")
    )
    outcome_derived = raw.get("outcome_derived")
    fold_fitted = raw.get("fold_fitted")

    if path_value in (None, ""):
        reasons.append("MISSING_PATH: path is required")
    if not generation:
        reasons.append("MISSING_GENERATION: generation is required")
    if not declared_role:
        reasons.append("MISSING_SOURCE_ROLE: source_role is required")
    if type(outcome_derived) is not bool:
        reasons.append("INVALID_OUTCOME_DERIVED: an explicit boolean is required")
    if type(fold_fitted) is not bool:
        reasons.append("INVALID_FOLD_FITTED: an explicit boolean is required")
    if use_role not in ALLOWED_USE_ROLES:
        reasons.append(f"UNREGISTERED_USE_ROLE: {declared_use_role!r}")

    exists = bool(path_value) and path.exists()
    if not exists:
        reasons.append(f"MISSING_ARTIFACT: {path}")

    digest: str | None = None
    columns: list[str] = []
    inspection_mode = "not_inspected"
    schema_error: str | None = None
    if exists:
        try:
            digest = artifact_sha256(path)
        except Exception as exc:
            reasons.append(f"HASH_FAILED: {type(exc).__name__}: {exc}")
        columns, inspection_mode, schema_error = _inspect_columns(path)
        if schema_error:
            reasons.append(f"SCHEMA_INSPECTION_FAILED: {schema_error}")

    expected = raw.get("expected_sha256", raw.get("sha256"))
    if expected not in (None, ""):
        expected_text = str(expected).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_text):
            reasons.append("INVALID_EXPECTED_SHA256: expected_sha256 must be 64 lowercase hex characters")
        elif digest is not None and digest.lower() != expected_text:
            reasons.append(
                f"SHA256_MISMATCH: expected={expected_text} observed={digest.lower()}"
            )

    current_generation = _is_v32_generation(generation)
    forbidden_role = _forbidden_role_kind(declared_role)
    checkpoint_exception = source_role == "v32_core_checkpoint"
    if forbidden_role and not checkpoint_exception:
        reasons.append(f"FORBIDDEN_RESULT_ROLE: {forbidden_role}")
    if source_role not in ALLOWED_SOURCE_ROLES:
        reasons.append(f"UNREGISTERED_SOURCE_ROLE: {declared_role!r}")

    path_result = _path_result_kind(path) if path_value else None
    if path_result and not (checkpoint_exception and path_result == "checkpoint"):
        reasons.append(f"FORBIDDEN_RESULT_PATH: artifact looks like {path_result}")

    if checkpoint_exception:
        if not current_generation:
            reasons.append("HISTORICAL_CHECKPOINT_FORBIDDEN: aux parents must be newly trained V3.2")
        if use_role != "aux_parent":
            reasons.append("V32_CORE_CHECKPOINT_NOT_AUX_PARENT: core checkpoint is aux_parent only")
    elif use_role == "aux_parent":
        reasons.append("INVALID_AUX_PARENT: only a V3.2 core checkpoint may be an aux parent")

    if type(outcome_derived) is bool and outcome_derived and use_role in {
        "core_input",
        "aux_input",
        "split_control",
    }:
        reasons.append("OUTCOME_DERIVED_FEATURE_FORBIDDEN: outcome-derived artifacts are not features")

    if source_role == "training_label":
        if not current_generation:
            reasons.append("HISTORICAL_TRAINING_LABEL_FORBIDDEN: regenerate labels inside V3.2")
        if use_role != "training_target":
            reasons.append("TRAINING_LABEL_ROLE_MISMATCH: training labels are training_target only")
    elif use_role == "training_target" and source_role not in {"raw_data"}:
        reasons.append("TRAINING_TARGET_ROLE_MISMATCH: target must be raw_data or training_label")

    if source_role == "split_manifest" and type(outcome_derived) is bool and outcome_derived:
        reasons.append("OUTCOME_DERIVED_SPLIT_FORBIDDEN: split manifests must be outcome-independent")

    if source_role == "standardized_input" and not current_generation:
        if outcome_derived is not False:
            reasons.append("OLD_STANDARDIZED_OUTCOME_DERIVED: must be false")
        if fold_fitted is not False:
            reasons.append("OLD_STANDARDIZED_FOLD_FITTED: must be false")
        if schema_error or not columns:
            reasons.append("OLD_STANDARDIZED_COLUMNS_UNVERIFIED: inspectable non-empty schema required")
        bad_old_columns = _old_standardized_blacklisted(columns)
        if bad_old_columns:
            reasons.append(
                "OLD_STANDARDIZED_FORBIDDEN_COLUMNS: " + repr(bad_old_columns)
            )

    result_columns = _result_columns(columns)
    if result_columns and source_role != "training_label":
        reasons.append("FORBIDDEN_RESULT_COLUMNS: " + repr(result_columns))

    if _looks_like_pair_evidence(path, declared_role, artifact_id, columns):
        bad_pair_columns = _pair_evidence_forbidden(columns)
        if bad_pair_columns:
            reasons.append(
                "PAIR_EVIDENCE_FORBIDDEN_COLUMNS: " + repr(bad_pair_columns)
            )

    reasons.extend(_family_broadcast_reasons(raw, columns))
    reasons = list(dict.fromkeys(reasons))
    return {
        "artifact_id": artifact_id,
        "path": str(path.resolve()) if path_value else "",
        "exists": exists,
        "sha256": digest,
        "generation": generation,
        "source_role": source_role,
        "outcome_derived": outcome_derived,
        "fold_fitted": fold_fitted,
        "use_role": use_role,
        "columns": columns,
        "column_inspection": inspection_mode,
        "schema_error": schema_error,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }


def audit_input_lineage(
    artifacts: Sequence[InputLineageArtifact | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic, read-only lineage audit for an input collection."""

    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise TypeError("artifacts must be a sequence of lineage mappings")
    records = [audit_input_artifact(spec) for spec in artifacts]
    duplicate_ids = sorted(
        artifact_id
        for artifact_id in {record["artifact_id"] for record in records}
        if sum(record["artifact_id"] == artifact_id for record in records) > 1
    )
    if duplicate_ids:
        for record in records:
            if record["artifact_id"] in duplicate_ids:
                record["status"] = "FAIL"
                record["reasons"].append(
                    f"DUPLICATE_ARTIFACT_ID: {record['artifact_id']!r}"
                )
    payload: dict[str, Any] = {
        "policy_version": INPUT_LINEAGE_POLICY_VERSION,
        "audit_mode": "READ_ONLY",
        "input_count": len(records),
        "status": "PASS" if records and all(record["status"] == "PASS" for record in records) else "FAIL",
        "artifacts": records,
    }
    payload["lineage_sha256"] = _canonical_json_sha256(payload)
    return payload


def validate_input_lineage(
    artifacts: Sequence[InputLineageArtifact | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a passing audit or raise a compact fail-closed exception."""

    audit = audit_input_lineage(artifacts)
    if audit["status"] != "PASS":
        failures = {
            record["artifact_id"]: record["reasons"]
            for record in audit["artifacts"]
            if record["status"] != "PASS"
        }
        raise InputLineageError(f"V3.2 input lineage rejected: {failures}")
    return audit


__all__ = [
    "ALLOWED_SOURCE_ROLES",
    "ALLOWED_USE_ROLES",
    "INPUT_LINEAGE_POLICY_VERSION",
    "InputLineageArtifact",
    "InputLineageError",
    "artifact_sha256",
    "audit_input_artifact",
    "audit_input_lineage",
    "validate_input_lineage",
]
