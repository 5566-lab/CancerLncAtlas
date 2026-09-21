"""Fail-closed query layer for registered V3.2 staging module artifacts.

Only artifacts mounted by a validated V3.2 release registry are queryable.
The module never searches historical release directories and never accepts a
filename as provenance.  State, Clinical, and Mutation/CNV are deliberately
kept as auxiliary result layers; none of the queries changes the primary
exact-pathway ranking.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .release_registry import (
    V32_PREFIX,
    ReleaseRegistryError,
    ValidatedReleaseRegistry,
    artifact_sha256,
    load_release_registry,
)


MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 1_000_000
CLINICAL_ENDPOINTS = frozenset({"DFI", "DFS", "DSS", "OS", "PFI", "PFS"})
CLINICAL_ENTITY_TYPES = frozenset({"exact_pathway", "lncRNA", "state"})
GENOMIC_MODALITIES = frozenset({"mutation", "cnv"})

STATE_ROLE = "state_typed_predictions"
CLINICAL_PATIENT_ROLE = "clinical_patient_risk"
CLINICAL_ENTITY_ROLE = "clinical_entity_association_public"
CLINICAL_ENTITY_LINEAGE_ROLE = "clinical_entity_association_lineage"
CLINICAL_ENTITY_VALIDATION_ROLE = "clinical_entity_association_validation"
GENOMIC_ROLE = "mutation_cnv_typed_predictions"

_ROLE_CONTRACTS: Mapping[str, tuple[str, str]] = {
    STATE_ROLE: ("state", "v32_public_prediction"),
    CLINICAL_PATIENT_ROLE: ("clinical", "v32_public_prediction"),
    CLINICAL_ENTITY_ROLE: ("clinical", "v32_public_statistical_result"),
    CLINICAL_ENTITY_LINEAGE_ROLE: ("clinical", "v32_public_metadata"),
    CLINICAL_ENTITY_VALIDATION_ROLE: ("clinical", "v32_public_metadata"),
    GENOMIC_ROLE: ("mutation_cnv", "v32_public_prediction"),
}

_REQUIRED_COLUMNS: Mapping[str, frozenset[str]] = {
    STATE_ROLE: frozenset(
        {
            "cancer_id",
            "lncrna_id",
            "state_id",
            "state_membership_probability",
            "state_effect",
            "association_direction",
            "availability",
            "availability_reason",
            "folds_available",
            "folds_expected",
            "model_version",
            "analysis_version",
            "training_run_id",
            "changes_primary_ranking",
        }
    ),
    CLINICAL_PATIENT_ROLE: frozenset(
        {
            "cancer_id",
            "subject_type",
            "subject_id",
            "clinical_endpoint",
            "clinical_relevance_probability",
            "availability",
            "failure_reason",
            "analysis_version",
            "training_run_id",
            "changes_primary_ranking",
        }
    ),
    CLINICAL_ENTITY_ROLE: frozenset(
        {
            "cancer_id",
            "subject_type",
            "subject_id",
            "clinical_endpoint",
            "clinical_relevance_probability",
            "availability",
            "failure_reason",
            "direction",
            "analysis_version",
            "old_checkpoint_loaded",
            "old_predictions_used_as_features",
            "changes_primary_ranking",
        }
    ),
    GENOMIC_ROLE: frozenset(
        {
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "mutation_context_probability",
            "mutation_available",
            "mutation_unavailable_reason",
            "mutation_patient_folds_with_prediction",
            "cnv_context_probability",
            "cnv_available",
            "cnv_unavailable_reason",
            "cnv_patient_folds_with_prediction",
            "analysis_version",
            "training_run_id",
            "module_id",
            "target_level",
            "prediction_format",
            "changes_primary_ranking",
        }
    ),
}


class StagingQueryError(RuntimeError):
    """Base error for the V3.2 auxiliary staging query layer."""


class StagingQueryAssetError(StagingQueryError):
    """Raised when a mounted artifact or its lineage fails closed."""


class StagingQueryInputError(StagingQueryError):
    """Raised for an unsupported or unsafe query filter."""


class StagingModuleUnavailableError(StagingQueryAssetError):
    """Raised when a query is attempted against an unmounted module."""


@dataclass(frozen=True)
class _MountedArtifact:
    role: str
    module_id: str
    path: Path
    sha256: str
    row_count: int


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _clean_filter(value: str | None, name: str, *, upper: bool = False) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        raise StagingQueryInputError(f"{name} may not be empty")
    if len(cleaned) > 512 or any(ord(character) < 32 for character in cleaned):
        raise StagingQueryInputError(f"{name} is not a valid staging query value")
    return cleaned.upper() if upper else cleaned


def _pagination(limit: int, offset: int) -> tuple[int, int]:
    if not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise StagingQueryInputError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
    if not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise StagingQueryInputError(f"offset must be between 0 and {MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


class V32StagingQuery:
    """Lazily query hash-bound V3.2 State, Clinical, and Genomic artifacts."""

    def __init__(self, registry: ValidatedReleaseRegistry) -> None:
        self.registry = registry
        self._descriptors = {
            str(item["role"]): item for item in registry.manifest["website_artifacts"]
        }
        self._mounted: dict[str, _MountedArtifact] = {}
        self._disabled: dict[str, str] = {}
        self._lineage_paths: dict[str, tuple[Path, str]] = {}
        self._prepare_module("state", (STATE_ROLE,))
        self._prepare_module(
            "clinical",
            (
                CLINICAL_PATIENT_ROLE,
                CLINICAL_ENTITY_ROLE,
                CLINICAL_ENTITY_LINEAGE_ROLE,
                CLINICAL_ENTITY_VALIDATION_ROLE,
            ),
        )
        self._prepare_module("mutation_cnv", (GENOMIC_ROLE,))
        if "state" not in self._disabled:
            self._validate_state()
        if "clinical" not in self._disabled:
            self._validate_clinical()
        if "mutation_cnv" not in self._disabled:
            self._validate_genomic()

    @classmethod
    def from_registry(cls, path: str | Path) -> "V32StagingQuery":
        return cls(load_release_registry(path, require_staging=True))

    def _resolve_lineage(self, module_id: str) -> tuple[Path, str]:
        module = self.registry.manifest["modules"][module_id]
        raw_path = Path(str(module["lineage_path"]))
        path = raw_path if raw_path.is_absolute() else self.registry.registry_path.parent / raw_path
        path = path.resolve()
        expected = str(module["lineage_sha256"]).lower()
        observed = artifact_sha256(path)
        if observed != expected:
            raise StagingQueryAssetError(
                f"{module_id} lineage hash drift: expected={expected}, observed={observed}"
            )
        return path, expected

    def _prepare_module(self, module_id: str, roles: Sequence[str]) -> None:
        module = self.registry.manifest["modules"][module_id]
        if module["status"] != "SUCCESS_NEWLY_TRAINED":
            self._disabled[module_id] = str(module.get("reason_code", "MODULE_NOT_NEWLY_TRAINED"))
            return
        missing = [role for role in roles if role not in self.registry.artifacts]
        if missing:
            self._disabled[module_id] = "REGISTERED_QUERY_ARTIFACTS_MISSING:" + ",".join(missing)
            return
        self._lineage_paths[module_id] = self._resolve_lineage(module_id)
        for role in roles:
            descriptor = self._descriptors.get(role)
            expected_module, expected_kind = _ROLE_CONTRACTS[role]
            if (
                not isinstance(descriptor, Mapping)
                or descriptor.get("source_module") != expected_module
                or descriptor.get("artifact_kind") != expected_kind
                or descriptor.get("generation") != self.registry.analysis_version
            ):
                raise StagingQueryAssetError(
                    f"Registered artifact {role} violates its V3.2 staging role contract"
                )
            path = self.registry.artifact(role)
            if role in _REQUIRED_COLUMNS:
                if path.suffix.casefold() not in {".parquet", ".pq"} or not path.is_file():
                    raise StagingQueryAssetError(f"Registered query artifact is not Parquet: {role}")
                parquet = pq.ParquetFile(path)
                columns = set(parquet.schema_arrow.names)
                missing_columns = sorted(_REQUIRED_COLUMNS[role] - columns)
                if missing_columns:
                    raise StagingQueryAssetError(
                        f"Registered artifact {role} lacks typed columns: {missing_columns}"
                    )
                row_count = int(parquet.metadata.num_rows)
                if row_count < 1:
                    raise StagingQueryAssetError(f"Registered artifact {role} is empty")
            else:
                row_count = 1
            self._mounted[role] = _MountedArtifact(
                role=role,
                module_id=module_id,
                path=path,
                sha256=self.registry.artifact_hashes[role],
                row_count=row_count,
            )

    def _read_lineage(self, module_id: str) -> Mapping[str, Any]:
        path, _ = self._lineage_paths[module_id]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StagingQueryAssetError(f"Cannot read {module_id} V3.2 lineage") from exc
        if not isinstance(value, Mapping):
            raise StagingQueryAssetError(f"{module_id} V3.2 lineage is not an object")
        return value

    def _assert_hashes_current(self, module_id: str, roles: Sequence[str]) -> None:
        try:
            registry_observed = artifact_sha256(self.registry.registry_path)
        except ReleaseRegistryError as exc:
            raise StagingQueryAssetError("The mounted V3.2 release registry is no longer readable") from exc
        if registry_observed != self.registry.registry_sha256:
            raise StagingQueryAssetError(
                "V3.2 release registry hash drift: "
                f"expected={self.registry.registry_sha256}, observed={registry_observed}"
            )
        path, expected = self._lineage_paths[module_id]
        try:
            observed = artifact_sha256(path)
        except ReleaseRegistryError as exc:
            raise StagingQueryAssetError(f"{module_id} lineage is no longer readable") from exc
        if observed != expected:
            raise StagingQueryAssetError(
                f"{module_id} lineage hash drift: expected={expected}, observed={observed}"
            )
        for role in roles:
            mount = self._mounted[role]
            try:
                observed = artifact_sha256(mount.path)
            except ReleaseRegistryError as exc:
                raise StagingQueryAssetError(f"{role} artifact is no longer readable") from exc
            if observed != mount.sha256:
                raise StagingQueryAssetError(
                    f"{role} artifact hash drift: expected={mount.sha256}, observed={observed}"
                )

    def _duckdb_row(self, sql: str, parameters: Sequence[Any]) -> Mapping[str, Any]:
        connection = duckdb.connect(database=":memory:")
        try:
            frame = connection.execute(sql, list(parameters)).fetchdf()
        finally:
            connection.close()
        if len(frame) != 1:
            raise StagingQueryAssetError("Internal staging validation did not return one row")
        return frame.iloc[0].to_dict()

    def _duckdb_frame(self, sql: str, parameters: Sequence[Any]) -> pd.DataFrame:
        connection = duckdb.connect(database=":memory:")
        try:
            return connection.execute(sql, list(parameters)).fetchdf()
        finally:
            connection.close()

    @staticmethod
    def _assert_zero(summary: Mapping[str, Any], role: str) -> None:
        failures = {
            str(key): int(value)
            for key, value in summary.items()
            if key != "n_rows" and value is not None and int(value) != 0
        }
        if int(summary.get("n_rows", 0)) < 1 or failures:
            raise StagingQueryAssetError(
                f"Registered artifact {role} failed V3.2 typed provenance validation: {failures}"
            )

    def _validate_state(self) -> None:
        mount = self._mounted[STATE_ROLE]
        module = self.registry.manifest["modules"]["state"]
        lineage = self._read_lineage("state")
        if lineage.get("prediction_sha256") != mount.sha256:
            raise StagingQueryAssetError("State artifact is not bound to the registered V3.2 lineage")
        if int(lineage.get("prediction_rows", -1)) != mount.row_count:
            raise StagingQueryAssetError("State artifact row count differs from V3.2 lineage")
        summary = self._duckdb_row(
            """
            SELECT count(*) AS n_rows,
                   count_if(analysis_version IS DISTINCT FROM ?) AS bad_version,
                   count_if(training_run_id IS DISTINCT FROM ?) AS bad_run,
                   count_if(model_version IS DISTINCT FROM 'V3.2') AS bad_model,
                   count_if(changes_primary_ranking IS DISTINCT FROM false) AS changes_primary,
                   count_if(state_membership_probability IS NOT NULL AND
                            (state_membership_probability < 0 OR state_membership_probability > 1))
                       AS bad_probability,
                   count_if(availability AND state_membership_probability IS NULL) AS missing_available,
                   count_if(NOT availability AND state_membership_probability IS NOT NULL)
                       AS populated_unavailable
            FROM read_parquet(?)
            """,
            (module["analysis_version"], module["training_run_id"], str(mount.path)),
        )
        self._assert_zero(summary, STATE_ROLE)

    def _validate_clinical(self) -> None:
        patient = self._mounted[CLINICAL_PATIENT_ROLE]
        entity = self._mounted[CLINICAL_ENTITY_ROLE]
        module = self.registry.manifest["modules"]["clinical"]
        module_lineage = self._read_lineage("clinical")
        if int(module_lineage.get("public_prediction_rows", -1)) != patient.row_count:
            raise StagingQueryAssetError("Clinical patient artifact row count differs from V3.2 lineage")
        entity_lineage_path = self._mounted[CLINICAL_ENTITY_LINEAGE_ROLE].path
        entity_lineage = json.loads(entity_lineage_path.read_text(encoding="utf-8"))
        if not isinstance(entity_lineage, Mapping):
            raise StagingQueryAssetError("Clinical entity lineage is not an object")
        if entity_lineage.get("summary_sha256") != entity.sha256:
            raise StagingQueryAssetError("Clinical entity artifact is not bound to its V3.2 lineage")
        if int(entity_lineage.get("summary_rows", entity.row_count)) != entity.row_count:
            raise StagingQueryAssetError("Clinical entity artifact row count differs from V3.2 lineage")
        patient_summary = self._duckdb_row(
            """
            SELECT count(*) AS n_rows,
                   count_if(analysis_version IS DISTINCT FROM ?) AS bad_version,
                   count_if(training_run_id IS DISTINCT FROM ?) AS bad_run,
                   count_if(subject_type IS DISTINCT FROM 'patient') AS bad_subject_type,
                   count_if(subject_id IS NULL OR trim(subject_id) = '') AS missing_internal_subject,
                   count_if(upper(clinical_endpoint) NOT IN ('DFI','DFS','DSS','OS','PFI','PFS'))
                       AS bad_endpoint,
                   count_if(changes_primary_ranking IS DISTINCT FROM false) AS changes_primary,
                   count_if(clinical_relevance_probability IS NOT NULL AND
                            (clinical_relevance_probability < 0 OR
                             clinical_relevance_probability > 1)) AS bad_probability,
                   count_if(availability AND clinical_relevance_probability IS NULL)
                       AS missing_available
            FROM read_parquet(?)
            """,
            (module["analysis_version"], module["training_run_id"], str(patient.path)),
        )
        self._assert_zero(patient_summary, CLINICAL_PATIENT_ROLE)
        entity_summary = self._duckdb_row(
            """
            SELECT count(*) AS n_rows,
                   count_if(analysis_version IS DISTINCT FROM ?) AS bad_version,
                   count_if(subject_type NOT IN ('exact_pathway','lncRNA','state'))
                       AS bad_subject_type,
                   count_if(regexp_matches(subject_id,
                            '^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}(?:$|-)'))
                       AS patient_identifier_in_entity_output,
                   count_if(upper(clinical_endpoint) NOT IN ('DFI','DFS','DSS','OS','PFI','PFS'))
                       AS bad_endpoint,
                   count_if(old_checkpoint_loaded IS DISTINCT FROM false) AS old_checkpoint,
                   count_if(old_predictions_used_as_features IS DISTINCT FROM false)
                       AS old_prediction,
                   count_if(changes_primary_ranking IS DISTINCT FROM false) AS changes_primary,
                   count_if(clinical_relevance_probability IS NOT NULL AND
                            (clinical_relevance_probability < 0 OR
                             clinical_relevance_probability > 1)) AS bad_probability,
                   count_if(availability AND clinical_relevance_probability IS NULL)
                       AS missing_available,
                   count_if(NOT availability AND clinical_relevance_probability IS NOT NULL)
                       AS populated_unavailable
            FROM read_parquet(?)
            """,
            (entity_lineage["analysis_version"], str(entity.path)),
        )
        self._assert_zero(entity_summary, CLINICAL_ENTITY_ROLE)

    def _validate_genomic(self) -> None:
        mount = self._mounted[GENOMIC_ROLE]
        module = self.registry.manifest["modules"]["mutation_cnv"]
        lineage = self._read_lineage("mutation_cnv")
        if lineage.get("prediction_sha256") != mount.sha256:
            raise StagingQueryAssetError(
                "Mutation/CNV artifact is not bound to the registered V3.2 lineage"
            )
        if int(lineage.get("prediction_rows", -1)) != mount.row_count:
            raise StagingQueryAssetError("Mutation/CNV artifact row count differs from V3.2 lineage")
        summary = self._duckdb_row(
            """
            SELECT count(*) AS n_rows,
                   count_if(analysis_version IS DISTINCT FROM ?) AS bad_version,
                   count_if(training_run_id IS DISTINCT FROM ?) AS bad_run,
                   count_if(module_id IS DISTINCT FROM 'mutation_cnv') AS bad_module,
                   count_if(prediction_format IS DISTINCT FROM
                            'CC_HHGT_V3_2_GENOMIC_TYPED_PREDICTIONS_V1') AS bad_format,
                   count_if(changes_primary_ranking IS DISTINCT FROM false) AS changes_primary,
                   count_if(mutation_context_probability IS NOT NULL AND
                            (mutation_context_probability < 0 OR
                             mutation_context_probability > 1)) AS bad_mutation_probability,
                   count_if(cnv_context_probability IS NOT NULL AND
                            (cnv_context_probability < 0 OR cnv_context_probability > 1))
                       AS bad_cnv_probability,
                   count_if(mutation_available AND mutation_context_probability IS NULL)
                       AS missing_mutation,
                   count_if(cnv_available AND cnv_context_probability IS NULL) AS missing_cnv,
                   count_if(NOT mutation_available AND mutation_context_probability IS NOT NULL)
                       AS populated_unavailable_mutation,
                   count_if(NOT cnv_available AND cnv_context_probability IS NOT NULL)
                       AS populated_unavailable_cnv
            FROM read_parquet(?)
            """,
            (module["analysis_version"], module["training_run_id"], str(mount.path)),
        )
        self._assert_zero(summary, GENOMIC_ROLE)

    def capability_status(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for module_id in ("state", "clinical", "mutation_cnv"):
            disabled_reason = self._disabled.get(module_id)
            result[module_id] = {
                "enabled": disabled_reason is None,
                "status": (
                    "ENABLED_HASH_BOUND_NEW_V32"
                    if disabled_reason is None
                    else "DISABLED_FAIL_CLOSED"
                ),
                "reason": disabled_reason,
            }
        return result

    def _require_module(self, module_id: str) -> None:
        if module_id in self._disabled:
            raise StagingModuleUnavailableError(
                f"{module_id} staging query is disabled: {self._disabled[module_id]}"
            )

    def _provenance(self, module_id: str, roles: Sequence[str]) -> dict[str, Any]:
        module = self.registry.manifest["modules"][module_id]
        return {
            "release_id": self.registry.release_id,
            "release_registry_sha256": self.registry.registry_sha256,
            "analysis_version": module["analysis_version"],
            "training_run_id": module["training_run_id"],
            "module_status": module["status"],
            "new_training_attestation": module["new_training_attestation"],
            "old_checkpoint_loaded": module["old_checkpoint_loaded"],
            "old_predictions_used_as_features": module["old_predictions_used_as_features"],
            "old_rankings_used_as_outputs": module["old_rankings_used_as_outputs"],
            "module_lineage_sha256": module["lineage_sha256"],
            "artifact_sha256": {role: self._mounted[role].sha256 for role in roles},
        }

    def query_state(
        self,
        *,
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        state_id: str | None = None,
        availability: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._require_module("state")
        limit, offset = _pagination(limit, offset)
        cancer_id = _clean_filter(cancer_id, "cancer_id", upper=True)
        lncrna_id = _clean_filter(lncrna_id, "lncrna_id", upper=True)
        state_id = _clean_filter(state_id, "state_id")
        self._assert_hashes_current("state", (STATE_ROLE,))
        mount = self._mounted[STATE_ROLE]
        clauses: list[str] = []
        values: list[Any] = [str(mount.path)]
        for column, value in (
            ("cancer_id", cancer_id),
            ("lncrna_id", lncrna_id),
            ("state_id", state_id),
        ):
            if value is not None:
                clauses.append(f"lower({column}) = lower(?)")
                values.append(value)
        if availability is not None:
            clauses.append("availability = ?")
            values.append(bool(availability))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        count = self._duckdb_row(
            f"SELECT count(*) AS n_rows FROM read_parquet(?){where}", values
        )
        frame = self._duckdb_frame(
            f"""
            SELECT cancer_id, lncrna_id, state_id, state_membership_probability,
                   state_effect, association_direction, availability,
                   nullif(availability_reason, '') AS availability_reason,
                   folds_available, folds_expected, analysis_version, training_run_id
            FROM read_parquet(?)
            {where}
            ORDER BY availability DESC,
                     state_membership_probability DESC NULLS LAST,
                     cancer_id, lncrna_id, state_id
            LIMIT ? OFFSET ?
            """,
            [*values, limit, offset],
        )
        return {
            "module": "state",
            "filters": {
                "cancer_id": cancer_id,
                "lncrna_id": lncrna_id,
                "state_id": state_id,
                "availability": availability,
            },
            "total_rows": int(count["n_rows"]),
            "returned_rows": len(frame),
            "limit": limit,
            "offset": offset,
            "results": _records(frame),
            "provenance": self._provenance("state", (STATE_ROLE,)),
        }

    def query_clinical(
        self,
        *,
        clinical_endpoint: str,
        cancer_id: str | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        availability: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._require_module("clinical")
        limit, offset = _pagination(limit, offset)
        endpoint = _clean_filter(clinical_endpoint, "clinical_endpoint", upper=True)
        if endpoint not in CLINICAL_ENDPOINTS:
            raise StagingQueryInputError(
                "clinical_endpoint must be one of " + ", ".join(sorted(CLINICAL_ENDPOINTS))
            )
        cancer_id = _clean_filter(cancer_id, "cancer_id", upper=True)
        subject_id = _clean_filter(subject_id, "subject_id")
        if subject_type is not None:
            normal_type = str(subject_type).strip().casefold().replace("-", "_")
            type_map = {
                "lncrna": "lncRNA",
                "exact_pathway": "exact_pathway",
                "state": "state",
            }
            subject_type = type_map.get(normal_type)
            if subject_type not in CLINICAL_ENTITY_TYPES:
                raise StagingQueryInputError(
                    "subject_type must be exact_pathway, lncRNA, or state"
                )
        self._assert_hashes_current(
            "clinical",
            (
                CLINICAL_PATIENT_ROLE,
                CLINICAL_ENTITY_ROLE,
                CLINICAL_ENTITY_LINEAGE_ROLE,
                CLINICAL_ENTITY_VALIDATION_ROLE,
            ),
        )
        entity_mount = self._mounted[CLINICAL_ENTITY_ROLE]
        patient_mount = self._mounted[CLINICAL_PATIENT_ROLE]
        entity_clauses = ["upper(clinical_endpoint) = ?"]
        entity_values: list[Any] = [str(entity_mount.path), endpoint]
        patient_clauses = ["upper(clinical_endpoint) = ?"]
        patient_values: list[Any] = [str(patient_mount.path), endpoint]
        if cancer_id is not None:
            entity_clauses.append("upper(cancer_id) = ?")
            entity_values.append(cancer_id)
            patient_clauses.append("upper(cancer_id) = ?")
            patient_values.append(cancer_id)
        if subject_type is not None:
            entity_clauses.append("subject_type = ?")
            entity_values.append(subject_type)
        if subject_id is not None:
            entity_clauses.append("lower(subject_id) = lower(?)")
            entity_values.append(subject_id)
        if availability is not None:
            entity_clauses.append("availability = ?")
            entity_values.append(bool(availability))
            patient_clauses.append("availability = ?")
            patient_values.append(bool(availability))
        entity_where = " WHERE " + " AND ".join(entity_clauses)
        patient_where = " WHERE " + " AND ".join(patient_clauses)
        entity_count = self._duckdb_row(
            f"SELECT count(*) AS n_rows FROM read_parquet(?){entity_where}", entity_values
        )
        patient_count = self._duckdb_row(
            f"SELECT count(*) AS n_rows FROM read_parquet(?){patient_where}", patient_values
        )
        entity_frame = self._duckdb_frame(
            f"""
            SELECT cancer_id, subject_type, subject_id, clinical_endpoint,
                   clinical_relevance_probability, availability,
                   nullif(failure_reason, '') AS failure_reason, direction,
                   analysis_version
            FROM read_parquet(?)
            {entity_where}
            ORDER BY availability DESC,
                     clinical_relevance_probability DESC NULLS LAST,
                     cancer_id, subject_type, subject_id
            LIMIT ? OFFSET ?
            """,
            [*entity_values, limit, offset],
        )
        # subject_id is intentionally not selected.  The response-local row
        # number cannot be joined back to TCGA patient identifiers.
        patient_frame = self._duckdb_frame(
            f"""
            SELECT cancer_id, clinical_endpoint, clinical_relevance_probability,
                   availability, nullif(failure_reason, '') AS failure_reason,
                   analysis_version, training_run_id
            FROM read_parquet(?)
            {patient_where}
            ORDER BY availability DESC,
                     clinical_relevance_probability DESC NULLS LAST,
                     cancer_id, subject_id
            LIMIT ? OFFSET ?
            """,
            [*patient_values, limit, offset],
        )
        patient_results = _records(patient_frame)
        for index, row in enumerate(patient_results, start=offset + 1):
            row["patient_result_id"] = f"response-patient-{index:06d}"
        return {
            "module": "clinical",
            "filters": {
                "clinical_endpoint": endpoint,
                "cancer_id": cancer_id,
                "entity_subject_type": subject_type,
                "entity_subject_id": subject_id,
                "availability": availability,
            },
            "entity_results": {
                "total_rows": int(entity_count["n_rows"]),
                "returned_rows": len(entity_frame),
                "results": _records(entity_frame),
            },
            "patient_results": {
                "total_rows": int(patient_count["n_rows"]),
                "returned_rows": len(patient_results),
                "identifier_policy": (
                    "response_local_sequence_only; raw patient identifiers, patient folds, "
                    "and model artifact identifiers are never returned"
                ),
                "results": patient_results,
            },
            "limit": limit,
            "offset": offset,
            "provenance": self._provenance(
                "clinical", (CLINICAL_PATIENT_ROLE, CLINICAL_ENTITY_ROLE)
            ),
        }

    def query_genomic(
        self,
        *,
        modality: str,
        cancer_id: str | None = None,
        lncrna_id: str | None = None,
        pathway_id: str | None = None,
        availability: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._require_module("mutation_cnv")
        limit, offset = _pagination(limit, offset)
        modality = str(modality).strip().casefold()
        if modality not in GENOMIC_MODALITIES:
            raise StagingQueryInputError("modality must be mutation or cnv")
        cancer_id = _clean_filter(cancer_id, "cancer_id", upper=True)
        lncrna_id = _clean_filter(lncrna_id, "lncrna_id", upper=True)
        pathway_id = _clean_filter(pathway_id, "pathway_id")
        self._assert_hashes_current("mutation_cnv", (GENOMIC_ROLE,))
        mount = self._mounted[GENOMIC_ROLE]
        probability_column = f"{modality}_context_probability"
        availability_column = f"{modality}_available"
        reason_column = f"{modality}_unavailable_reason"
        folds_column = f"{modality}_patient_folds_with_prediction"
        clauses: list[str] = []
        values: list[Any] = [str(mount.path)]
        for column, value in (
            ("cancer_id", cancer_id),
            ("lncrna_id", lncrna_id),
            ("pathway_id", pathway_id),
        ):
            if value is not None:
                clauses.append(f"lower({column}) = lower(?)")
                values.append(value)
        if availability is not None:
            clauses.append(f"{availability_column} = ?")
            values.append(bool(availability))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        count = self._duckdb_row(
            f"SELECT count(*) AS n_rows FROM read_parquet(?){where}", values
        )
        frame = self._duckdb_frame(
            f"""
            SELECT cancer_id, lncrna_id, pathway_id,
                   {probability_column} AS context_probability,
                   {availability_column} AS availability,
                   nullif({reason_column}, '') AS unavailable_reason,
                   {folds_column} AS patient_folds_with_prediction,
                   target_level, analysis_version, training_run_id
            FROM read_parquet(?)
            {where}
            ORDER BY {availability_column} DESC,
                     {probability_column} DESC NULLS LAST,
                     cancer_id, lncrna_id, pathway_id
            LIMIT ? OFFSET ?
            """,
            [*values, limit, offset],
        )
        return {
            "module": "mutation_cnv",
            "modality": modality,
            "filters": {
                "cancer_id": cancer_id,
                "lncrna_id": lncrna_id,
                "pathway_id": pathway_id,
                "availability": availability,
            },
            "total_rows": int(count["n_rows"]),
            "returned_rows": len(frame),
            "limit": limit,
            "offset": offset,
            "results": _records(frame),
            "provenance": self._provenance("mutation_cnv", (GENOMIC_ROLE,)),
        }


__all__ = [
    "CLINICAL_ENDPOINTS",
    "GENOMIC_MODALITIES",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "StagingModuleUnavailableError",
    "StagingQueryAssetError",
    "StagingQueryError",
    "StagingQueryInputError",
    "V32StagingQuery",
]
