"""Hash-pinned staging query for the corrected directional CNV-only head."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_BINDING_V1"
MAX_QUERY_LIMIT = 1000
MAX_QUERY_OFFSET = 1_000_000
CNV_DOWNLOAD_IDS = frozenset({"cnv_context", "cnv_associations", "cnv_coverage"})


class DirectionalCNVQueryError(RuntimeError):
    pass


class DirectionalCNVQueryAssetError(DirectionalCNVQueryError):
    pass


class DirectionalCNVQueryInputError(DirectionalCNVQueryError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DirectionalCNVQueryAssetError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionalCNVQueryAssetError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectionalCNVQueryAssetError(f"{label} must be a JSON object")
    return value


def _clean(value: Any | None, label: str, *, upper: bool = False) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or len(cleaned) > 200 or any(ord(char) < 32 for char in cleaned):
        raise DirectionalCNVQueryInputError(f"{label} is invalid")
    return cleaned.upper() if upper else cleaned


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                key: (
                    None
                    if value is None
                    or value is pd.NA
                    or (isinstance(value, float) and not math.isfinite(value))
                    else value.item() if hasattr(value, "item") else value
                )
                for key, value in record.items()
            }
        )
    return rows


class DirectionalCNVReleaseQuery:
    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_sha256: str,
        audit_binding_path: str | Path,
        expected_audit_sha256: str,
    ) -> None:
        binding_source = Path(binding_path)
        if binding_source.is_symlink() or not binding_source.is_file():
            raise DirectionalCNVQueryAssetError(
                "directional CNV binding is missing or unsafe"
            )
        self.binding_path = binding_source.resolve()
        self.binding_sha256 = expected_sha256.strip().lower()
        if len(self.binding_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.binding_sha256
        ):
            raise DirectionalCNVQueryAssetError("binding SHA-256 is malformed")
        if _sha256(self.binding_path) != self.binding_sha256:
            raise DirectionalCNVQueryAssetError("directional CNV binding SHA-256 drift")
        self.binding = _load_json(self.binding_path, "directional CNV binding")
        if (
            self.binding.get("format") != BINDING_FORMAT
            or self.binding.get("status") != "PASS"
            or self.binding.get("analysis_version") != ANALYSIS_VERSION
            or self.binding.get("staging_queryable") is not True
            or self.binding.get("release_ready") is not False
            or self.binding.get("release_blocker") != "CNV_ROUTER_INCREMENT_NOT_TESTED"
            or self.binding.get("changes_primary_ranking") is not False
        ):
            raise DirectionalCNVQueryAssetError("directional CNV binding contract drift")
        artifacts = self.binding.get("artifacts")
        if not isinstance(artifacts, dict):
            raise DirectionalCNVQueryAssetError("directional CNV binding lacks artifacts")
        self._pins: dict[str, tuple[Path, str, int | None]] = {}
        for name in ("predictions", "coverage", "lineage", "transformation_audit"):
            record = artifacts.get(name)
            if not isinstance(record, dict):
                raise DirectionalCNVQueryAssetError(f"directional CNV {name} pin is absent")
            source = Path(str(record.get("path", "")))
            if source.is_symlink() or not source.is_file():
                raise DirectionalCNVQueryAssetError(
                    f"directional CNV {name} artifact is missing or unsafe"
                )
            path = source.resolve()
            expected = str(record.get("sha256", "")).lower()
            rows = int(record["rows"]) if "rows" in record else None
            self._pins[name] = (path, expected, rows)
        self.prediction_path = self._pins["predictions"][0]
        self.coverage_path = self._pins["coverage"][0]
        self._assert_pins()
        self._validate_provenance()
        self._validate_independent_audit(
            audit_binding_path,
            expected_audit_sha256,
        )
        self._validate_artifacts()

    def _assert_pins(self) -> None:
        if _sha256(self.binding_path) != self.binding_sha256:
            raise DirectionalCNVQueryAssetError("mounted directional CNV binding drifted")
        for label, (path, expected, _) in self._pins.items():
            if (
                len(expected) != 64
                or path.is_symlink()
                or not path.is_file()
                or _sha256(path) != expected
            ):
                raise DirectionalCNVQueryAssetError(
                    f"mounted directional CNV {label} artifact drifted"
                )

    def _validate_provenance(self) -> None:
        lineage = _load_json(self._pins["lineage"][0], "directional CNV lineage")
        audit = _load_json(
            self._pins["transformation_audit"][0],
            "directional CNV transformation audit",
        )
        checks = audit.get("checks")
        required_true_checks = {
            "source_prediction_hashes_recomputed",
            "source_independent_audit_bound",
            "signed_directional_source_only",
            "available_probability_finite",
            "typed_unavailable_probability_null",
        }
        if (
            lineage.get("format")
            != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_LINEAGE_V1"
            or lineage.get("status") != "PASS"
            or lineage.get("mutation_features_used") is not False
            or lineage.get("primary_ranking_changed") is not False
            or audit.get("format")
            != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_TRANSFORMATION_AUDIT_V1"
            or audit.get("status") != "PASS"
            or not isinstance(checks, dict)
            or any(checks.get(name) is not True for name in required_true_checks)
            or checks.get("mutation_features_used") is not False
            or checks.get("superseded_combined_cnv_read") is not False
            or checks.get("primary_ranking_changed") is not False
            or audit.get("prediction_sha256") != self._pins["predictions"][1]
            or audit.get("coverage_sha256") != self._pins["coverage"][1]
        ):
            raise DirectionalCNVQueryAssetError(
                "directional CNV provenance or transformation-audit contract drift"
            )

    def _validate_independent_audit(
        self,
        audit_binding_path: str | Path,
        expected_audit_sha256: str,
    ) -> None:
        source = Path(audit_binding_path)
        if source.is_symlink() or not source.is_file():
            raise DirectionalCNVQueryAssetError(
                "directional CNV independent-audit binding is missing or unsafe"
            )
        expected = str(expected_audit_sha256).strip().lower()
        if (
            len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
            or _sha256(source) != expected
        ):
            raise DirectionalCNVQueryAssetError(
                "directional CNV independent-audit binding SHA-256 drift"
            )
        binding = _load_json(source, "directional CNV independent-audit binding")
        release_record = binding.get("release_binding")
        report_record = binding.get("report")
        if (
            binding.get("format")
            != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_INDEPENDENT_AUDIT_BINDING_V1"
            or binding.get("status") != "PASS"
            or binding.get("accepted_for_staging_api_integration") is not True
            or binding.get("production_deployed") is not False
            or binding.get("release_ready") is not False
            or not isinstance(release_record, dict)
            or Path(str(release_record.get("path", ""))).resolve()
            != self.binding_path
            or release_record.get("sha256") != self.binding_sha256
            or not isinstance(report_record, dict)
        ):
            raise DirectionalCNVQueryAssetError(
                "directional CNV independent-audit binding contract drift"
            )
        report_source = Path(str(report_record.get("path", "")))
        if (
            report_source.is_symlink()
            or not report_source.is_file()
            or _sha256(report_source) != str(report_record.get("sha256", "")).lower()
        ):
            raise DirectionalCNVQueryAssetError(
                "directional CNV independent-audit report hash drift"
            )
        report = _load_json(report_source, "directional CNV independent-audit report")
        checks = report.get("checks")
        required_true = {
            "artifact_hashes_recomputed",
            "source_hashes_recomputed",
            "coverage_rederived",
            "signed_directional_source_only",
            "mutation_columns_absent",
            "typed_null_contract",
            "probability_contract",
            "five_fold_count_contract",
        }
        if (
            report.get("format")
            != "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_INDEPENDENT_AUDIT_V1"
            or report.get("status") != "PASS"
            or report.get("independent_of_materializer_implementation") is not True
            or report.get("materializer_imported") is not False
            or report.get("accepted_for_staging_api_integration") is not True
            or report.get("release_binding_sha256") != self.binding_sha256
            or report.get("prediction_sha256") != self._pins["predictions"][1]
            or report.get("coverage_sha256") != self._pins["coverage"][1]
            or not isinstance(checks, dict)
            or any(checks.get(name) is not True for name in required_true)
            or checks.get("superseded_combined_cnv_exposed") is not False
            or checks.get("changes_primary_ranking") is not False
        ):
            raise DirectionalCNVQueryAssetError(
                "directional CNV independent-audit report contract drift"
            )
        self.audit_binding_path = source.resolve()
        self.audit_binding_sha256 = expected
        self.independent_audit = report

    def _validate_artifacts(self) -> None:
        required = {
            "cancer_id", "lncrna_id", "pathway_id", "cnv_context_probability",
            "cnv_context_probability_sd", "cnv_patient_folds_with_prediction",
            "cnv_patient_fold_count", "cnv_available", "cnv_unavailable_reason",
            "local_cnv_available_fold_count", "pathway_cnv_available_fold_count",
            "cnv_pair_callable_patients_across_oof", "training_run_id",
            "analysis_version", "module_id", "target_level", "prediction_format",
            "changes_primary_ranking",
        }
        connection = duckdb.connect(":memory:")
        try:
            columns = {
                row[0]
                for row in connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(self.prediction_path)],
                ).fetchall()
            }
            if missing := sorted(required - columns):
                raise DirectionalCNVQueryAssetError(
                    f"directional CNV prediction columns are missing: {missing}"
                )
            check = connection.execute(
                """
                SELECT count(*), count(DISTINCT cancer_id),
                       count_if(cnv_available AND
                                (cnv_context_probability IS NULL OR
                                 NOT isfinite(cnv_context_probability) OR
                                 cnv_context_probability < 0 OR
                                 cnv_context_probability > 1)),
                       count_if(NOT cnv_available AND cnv_context_probability IS NOT NULL),
                       count_if(cnv_available AND
                                nullif(trim(cnv_unavailable_reason), '') IS NOT NULL),
                       count_if(NOT cnv_available AND
                                nullif(trim(cnv_unavailable_reason), '') IS NULL),
                       count_if(cnv_patient_fold_count != 5),
                       count_if(cnv_patient_folds_with_prediction < 0 OR
                                cnv_patient_folds_with_prediction > 5 OR
                                (cnv_available AND
                                 cnv_patient_folds_with_prediction = 0) OR
                                (NOT cnv_available AND
                                 cnv_patient_folds_with_prediction != 0)),
                       count_if(analysis_version != ? OR module_id != 'cnv' OR
                                changes_primary_ranking)
                FROM read_parquet(?)
                """,
                [ANALYSIS_VERSION, str(self.prediction_path)],
            ).fetchone()
            expected_rows = self._pins["predictions"][2]
            if (
                expected_rows is None
                or int(check[0]) != expected_rows
                or int(check[1]) != 33
                or any(int(value) != 0 for value in check[2:])
            ):
                raise DirectionalCNVQueryAssetError(
                    f"directional CNV prediction content validation failed: {check}"
                )
            coverage = connection.execute(
                """
                SELECT count(*), count(DISTINCT cancer_id),
                       count_if(candidate_rows <= 0 OR available_rows < 0 OR
                                typed_unavailable_rows < 0 OR
                                available_rows + typed_unavailable_rows != candidate_rows)
                FROM read_parquet(?)
                """,
                [str(self.coverage_path)],
            ).fetchone()
            if (
                self._pins["coverage"][2] is None
                or int(coverage[0]) != self._pins["coverage"][2]
                or int(coverage[1]) != 33
                or int(coverage[2]) != 0
            ):
                raise DirectionalCNVQueryAssetError(
                    f"directional CNV coverage content validation failed: {coverage}"
                )
        finally:
            connection.close()

    def provenance(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "binding_path": str(self.binding_path),
            "binding_sha256": self.binding_sha256,
            "prediction_sha256": self._pins["predictions"][1],
            "independent_audit_sha256": self.audit_binding_sha256,
            "source_generation": "CURRENT_V3.2_DIRECTIONAL_CNV_ONLY",
            # The directional CNV head is an independently trained five-fold
            # overlay.  Keep the same explicit freshness contract used by the
            # primary and other V3.2 heads so a web response cannot be
            # mistaken for a legacy/loaded checkpoint result.
            "new_training_attestation": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
            "superseded_combined_cnv_used": False,
            "mutation_features_used": False,
            "changes_primary_ranking": False,
        }

    def query(
        self,
        *,
        cancer_id: Any | None = None,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        availability: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._assert_pins()
        if not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
            raise DirectionalCNVQueryInputError("limit is outside the allowed range")
        if not isinstance(offset, int) or not 0 <= offset <= MAX_QUERY_OFFSET:
            raise DirectionalCNVQueryInputError("offset is outside the allowed range")
        if availability is not None and not isinstance(availability, bool):
            raise DirectionalCNVQueryInputError("availability must be true, false or null")
        cancer = _clean(cancer_id, "cancer_id", upper=True)
        lnc = _clean(lncrna_id, "lncrna_id")
        pathway = _clean(pathway_id, "pathway_id")
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("cancer_id", cancer), ("lncrna_id", lnc), ("pathway_id", pathway)
        ):
            if value is not None:
                clauses.append(f"lower({column}) = lower(?)")
                parameters.append(value)
        if availability is not None:
            clauses.append("cnv_available = ?")
            parameters.append(availability)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        connection = duckdb.connect(":memory:")
        try:
            total = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet(?){where}",
                    [str(self.prediction_path), *parameters],
                ).fetchone()[0]
            )
            frame = connection.execute(
                f"""
                SELECT cancer_id, lncrna_id, pathway_id,
                       cnv_context_probability AS context_probability,
                       cnv_context_probability_sd AS context_probability_sd,
                       cnv_available AS availability,
                       nullif(cnv_unavailable_reason, '') AS unavailable_reason,
                       cnv_patient_folds_with_prediction AS patient_folds_with_prediction,
                       local_cnv_available_fold_count,
                       pathway_cnv_available_fold_count,
                       cnv_pair_callable_patients_across_oof,
                       target_level, analysis_version, training_run_id
                FROM read_parquet(?) {where}
                ORDER BY cnv_available DESC,
                         cnv_context_probability DESC NULLS LAST,
                         cancer_id, lncrna_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                [str(self.prediction_path), *parameters, limit, offset],
            ).fetchdf()
        finally:
            connection.close()
        return {
            "module": "cnv",
            "modality": "cnv",
            "filters": {
                "cancer_id": cancer,
                "lncrna_id": lnc,
                "pathway_id": pathway,
                "availability": availability,
            },
            "total_rows": total,
            "returned_rows": len(frame),
            "limit": limit,
            "offset": offset,
            "results": _records(frame),
            "provenance": self.provenance(),
        }

    def coverage(self, *, cancer_id: Any | None = None) -> dict[str, Any]:
        self._assert_pins()
        cancer = _clean(cancer_id, "cancer_id", upper=True)
        where = " WHERE cancer_id = ?" if cancer is not None else ""
        parameters = [str(self.coverage_path), *([cancer] if cancer is not None else [])]
        connection = duckdb.connect(":memory:")
        try:
            frame = connection.execute(
                f"SELECT * FROM read_parquet(?){where} ORDER BY cancer_id", parameters
            ).fetchdf()
        finally:
            connection.close()
        return {
            "module": "cnv",
            "status": "QUERYABLE_AUDITED_STAGING",
            "cancer_id": cancer,
            "count": len(frame),
            "rows": _records(frame),
            "typed_nulls": True,
            "provenance": self.provenance(),
        }

    def download_entry(self, download_id: str) -> dict[str, Any]:
        """Return a public entry that supersedes the three historical CNV files."""

        self._assert_pins()
        if download_id not in CNV_DOWNLOAD_IDS:
            raise DirectionalCNVQueryInputError(
                f"unknown directional CNV download: {download_id}"
            )
        role = "coverage" if download_id == "cnv_coverage" else "predictions"
        path, digest, rows = self._pins[role]
        return {
            "download_id": download_id,
            "capability_ids": ["cnv"],
            "status": "READY_FILE",
            "download_implemented": True,
            "data_present": True,
            "contract_complete": True,
            "model_version": "V3.2",
            "availability_encoding": "null_with_reason",
            "unavailable_fill_value": None,
            "family_to_exact_broadcast": False,
            "supersedes_historical_combined_cnv": True,
            "source_generation": "CURRENT_V3.2_DIRECTIONAL_CNV_ONLY",
            "file": {
                "relative_name": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rows": rows,
            },
        }

    def resolve_download(self, download_id: str) -> dict[str, Any]:
        """Resolve and re-hash a corrected CNV download at request time."""

        entry = self.download_entry(download_id)
        role = "coverage" if download_id == "cnv_coverage" else "predictions"
        path, digest, _ = self._pins[role]
        return {
            "kind": "file",
            "download_id": download_id,
            "path": str(path),
            "relative_name": entry["file"]["relative_name"],
            "sha256": digest,
            "bytes": entry["file"]["bytes"],
        }


__all__ = [
    "DirectionalCNVQueryAssetError",
    "DirectionalCNVQueryError",
    "DirectionalCNVQueryInputError",
    "DirectionalCNVReleaseQuery",
    "CNV_DOWNLOAD_IDS",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
]
