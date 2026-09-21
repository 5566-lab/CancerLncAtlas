"""Read-only queries for the independently audited V3.2 formal-23 single-cell data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import duckdb
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
# The accepted formal-23 overlay is the R11 hash-bound release assembled on
# server 149.  Individual source partitions retain their immutable R7/R11
# provenance in the parquet manifests; this label identifies the release
# contract exposed by this query surface and prevents the old R7-only label
# from being mistaken for the current generation.
FORMAL23_RELEASE_GENERATION = "V3.2_R11_FORMAL23_HASH_BOUND_PARTITIONS"
MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 10_000_000
SINGLE_CELL_FORMAL23_DOWNLOAD_IDS = frozenset(
    {"single_cell_associations", "single_cell_activity", "single_cell_ucell"}
)
SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS = frozenset(
    {
        *SINGLE_CELL_FORMAL23_DOWNLOAD_IDS,
        "single_cell_pseudotime",
        "single_cell_figures",
    }
)
FORMAL_CANCERS = frozenset(
    {
        "ACC", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
        "HNSC", "KIRC", "LAML", "LGG", "LUSC", "MESO", "OV", "PCPG",
        "READ", "SARC", "SKCM", "TGCT", "THYM", "UCEC", "UVM",
    }
)
ALL_CANCERS = frozenset(
    {
        "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
        "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
        "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
        "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
        "UVM",
    }
)
REQUIRED_FILES = frozenset(
    {
        "RESOURCE_ESTIMATE.json",
        "association_context_availability.parquet",
        "association_evidence.parquet",
        "association_lncrna_testability.parquet",
        "lncrna_donor_celltype_summary.parquet",
        "pathway_availability.parquet",
        "pathway_donor_celltype_summary.parquet",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:\-]+$")


class SingleCellFormal23Error(RuntimeError):
    pass


class SingleCellFormal23AssetError(SingleCellFormal23Error):
    pass


class SingleCellFormal23InputError(SingleCellFormal23Error):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SingleCellFormal23AssetError(f"{label} is missing or unsafe")
    return source.resolve()


def _json(path: str | Path, label: str) -> dict[str, Any]:
    source = _safe_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellFormal23AssetError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SingleCellFormal23AssetError(f"{label} must be a JSON object")
    return value


def _clean_identifier(value: Any | None, label: str) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or len(token) > 256 or not _IDENTIFIER.fullmatch(token):
        raise SingleCellFormal23InputError(f"{label} is invalid")
    return token


def _clean_cell_type(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    if not token or len(token) > 256 or any(ord(char) < 32 for char in token):
        raise SingleCellFormal23InputError("cell_type is invalid")
    return token


def _clean_compartment(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token not in {"malignant", "immune", "stromal", "other_unresolved"}:
        raise SingleCellFormal23InputError("compartment is invalid")
    return token


def _availability(value: Any) -> str:
    token = str(value or "ALL").strip().upper()
    if token not in {"ALL", "AVAILABLE", "UNAVAILABLE"}:
        raise SingleCellFormal23InputError(
            "availability must be ALL, AVAILABLE, or UNAVAILABLE"
        )
    return token


def _bounds(limit: Any, offset: Any) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise SingleCellFormal23InputError(f"limit must be 1..{MAX_QUERY_LIMIT}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise SingleCellFormal23InputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            cleaned[str(key)] = (
                None
                if missing or (isinstance(value, float) and not math.isfinite(value))
                else value
            )
        result.append(cleaned)
    return result


class SingleCellFormal23Query:
    """Hash-pinned donor-level expression, activity and association queries."""

    def __init__(self, success_path: str | Path, *, expected_sha256: str) -> None:
        self.success_path = _safe_file(success_path, "single-cell formal-23 SUCCESS")
        expected = str(expected_sha256).strip().lower()
        if not _SHA256.fullmatch(expected) or _sha256(self.success_path) != expected:
            raise SingleCellFormal23AssetError("single-cell formal-23 SUCCESS SHA drift")
        self.success_sha256 = expected
        success = _json(self.success_path, "single-cell formal-23 SUCCESS")
        typed = success.get("typed_unavailable")
        if (
            success.get("format")
            != "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_SUCCESS_V1"
            or success.get("status") != "SUCCESS"
            or success.get("formal_eligible_cancer_count") != 23
            or set(success.get("formal_eligible_cancers", [])) != FORMAL_CANCERS
            or success.get("typed_unavailable_cancer_count") != 10
            or not isinstance(typed, dict)
            or set(typed) != ALL_CANCERS - FORMAL_CANCERS
            or success.get("typed_unavailable_rows_are_null") is not True
            or success.get("typed_unavailable_changes_primary_score") is not False
            or success.get("full_33_single_cell_coverage_claimed") is not False
            or success.get("scientific_module_ready_for_binding") is not True
        ):
            raise SingleCellFormal23AssetError(
                "single-cell formal-23 SUCCESS contract drift"
            )
        self.typed_unavailable = {str(k): str(v) for k, v in typed.items()}
        self.binding_path = _safe_file(
            success.get("formal_binding_path", ""), "single-cell formal-23 binding"
        )
        self.audit_path = _safe_file(
            success.get("independent_audit_path", ""),
            "single-cell formal-23 independent audit",
        )
        if (
            _sha256(self.binding_path) != success.get("formal_binding_sha256")
            or _sha256(self.audit_path) != success.get("independent_audit_sha256")
        ):
            raise SingleCellFormal23AssetError("single-cell formal-23 outer hash drift")
        self.binding_sha256 = str(success["formal_binding_sha256"])
        self.audit_sha256 = str(success["independent_audit_sha256"])
        binding = _json(self.binding_path, "single-cell formal-23 binding")
        audit = _json(self.audit_path, "single-cell formal-23 independent audit")
        if (
            binding.get("format")
            != "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_BINDING_V1"
            or binding.get("status") != "BOUND_23_OF_23_PENDING_INDEPENDENT_AUDIT"
            or binding.get("formal_bound_cancer_count") != 23
            or set(binding.get("formal_cancer_universe", [])) != FORMAL_CANCERS
            or binding.get("historical_assets_relabelled_fresh") is not False
            or audit.get("format")
            != "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_INDEPENDENT_AUDIT_V1"
            or audit.get("status") != "PASS_23_OF_23_INDEPENDENTLY_VERIFIED"
            or audit.get("formal_cancer_count") != 23
            or set(audit.get("formal_cancer_universe", [])) != FORMAL_CANCERS
            or audit.get("cohort_success_sha256") != self.binding_sha256
        ):
            raise SingleCellFormal23AssetError(
                "single-cell formal-23 binding or audit contract drift"
            )
        per_cancer = audit.get("per_cancer")
        if not isinstance(per_cancer, list) or len(per_cancer) != 23:
            raise SingleCellFormal23AssetError("formal-23 audit lacks 23 cancer records")
        self.paths: dict[str, dict[str, tuple[Path, str]]] = {}
        self.cancer_metadata: dict[str, dict[str, Any]] = {}
        for record in per_cancer:
            if not isinstance(record, dict):
                raise SingleCellFormal23AssetError("formal-23 cancer record is invalid")
            cancer = str(record.get("cancer_id", ""))
            if cancer not in FORMAL_CANCERS or cancer in self.paths:
                raise SingleCellFormal23AssetError("formal-23 cancer identity drift")
            output_root = Path(str(record.get("output_root", ""))).resolve()
            manifest = record.get("file_manifest")
            entries = manifest.get("entries") if isinstance(manifest, dict) else None
            manifest_path = _safe_file(
                manifest.get("path", "") if isinstance(manifest, dict) else "",
                f"{cancer} file manifest",
            )
            if (
                not isinstance(entries, list)
                or {str(item.get("relative_path", "")) for item in entries} != REQUIRED_FILES
                or _sha256(manifest_path) != manifest.get("sha256")
                or manifest_path.parent != output_root
            ):
                raise SingleCellFormal23AssetError(f"{cancer} file manifest drift")
            cancer_paths: dict[str, tuple[Path, str]] = {}
            for item in entries:
                relative = str(item["relative_path"])
                path = _safe_file(output_root / relative, f"{cancer} {relative}")
                digest = str(item.get("sha256", "")).lower()
                if not _SHA256.fullmatch(digest) or path.parent != output_root:
                    raise SingleCellFormal23AssetError(f"{cancer} file declaration drift")
                cancer_paths[relative] = (path, digest)
            self.paths[cancer] = cancer_paths
            self.cancer_metadata[cancer] = {
                "dataset_id": str(record.get("dataset_id", "")),
                "cells": int(record.get("cells", 0)),
                "donors": int(record.get("donors", 0)),
                "mapped_source_universe_lncrnas": int(
                    record.get("mapped_source_universe_lncrnas", 0)
                ),
                "association_evidence_rows": int(
                    record.get("association_evidence", {}).get("rows", 0)
                ),
                "generated_in_current_r11_run": bool(
                    record.get("generated_in_current_r11_run")
                ),
            }
        if set(self.paths) != FORMAL_CANCERS:
            raise SingleCellFormal23AssetError("formal-23 cancer closure drift")
        tree_payload = [
            {
                "cancer_id": cancer,
                "association_evidence_sha256": self.paths[cancer][
                    "association_evidence.parquet"
                ][1],
                "pathway_donor_celltype_summary_sha256": self.paths[cancer][
                    "pathway_donor_celltype_summary.parquet"
                ][1],
            }
            for cancer in sorted(FORMAL_CANCERS)
        ]
        self.download_tree_sha256 = hashlib.sha256(
            json.dumps(
                tree_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self.success = success

    def _assert_outer(self) -> None:
        if (
            _sha256(self.success_path) != self.success_sha256
            or _sha256(self.binding_path) != self.binding_sha256
            or _sha256(self.audit_path) != self.audit_sha256
        ):
            raise SingleCellFormal23AssetError("single-cell formal-23 authority drift")

    def _artifact(self, cancer: str, filename: str) -> Path:
        path, digest = self.paths[cancer][filename]
        if _sha256(path) != digest:
            raise SingleCellFormal23AssetError(
                f"single-cell formal-23 artifact hash drift: {cancer}/{filename}"
            )
        return path

    @staticmethod
    def _connect() -> duckdb.DuckDBPyConnection:
        return duckdb.connect(":memory:")

    def capability_status(self) -> dict[str, Any]:
        self._assert_outer()
        return {
            "module": "single_cell_formal23",
            "status": "QUERYABLE_AUDITED_FORMAL23",
            "formal_eligible_cancers": sorted(FORMAL_CANCERS),
            "formal_eligible_cancer_count": 23,
            "typed_unavailable": self.typed_unavailable,
            "typed_unavailable_cancer_count": 10,
            "full_33_single_cell_coverage_claimed": False,
            "single_cell_currently_changes_secondary_score": False,
            "changes_exact_primary_score": False,
            "association_semantics": "DONOR_PSEUDOBULK_SPEARMAN_WITH_BH_FDR",
            "association_is_exploratory_not_causal": True,
            "learned_celltype_probability_available": False,
            "total_cells": int(self.success["total_cells"]),
            "total_association_evidence_rows": int(
                self.success["total_association_evidence_rows"]
            ),
            "binding_sha256": self.binding_sha256,
            "independent_audit_sha256": self.audit_sha256,
            "release_ready": False,
            "production_deployed": False,
        }

    def _cancer(self, value: Any) -> str:
        cancer = str(value or "").strip().upper()
        if cancer not in ALL_CANCERS:
            raise SingleCellFormal23InputError("cancer_id is not in the 33-cancer authority")
        return cancer

    def _typed_unavailable_response(
        self, cancer: str, query_kind: str, filters: dict[str, Any]
    ) -> dict[str, Any]:
        return self._response(
            query_kind,
            [
                {
                    "cancer_id": cancer,
                    "availability": False,
                    "unavailable_reason": self.typed_unavailable[cancer],
                }
            ],
            filters,
        )

    def _response(
        self, query_kind: str, rows: list[dict[str, Any]], filters: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "module": "single_cell_formal23",
            "query_kind": query_kind,
            "learned_association": False,
            "association_is_exploratory_not_causal": True,
            "biological_unit": "DONOR",
            "cell_as_independent_replicate": False,
            "single_cell_currently_changes_secondary_score": False,
            "changes_exact_primary_score": False,
            "filters": filters,
            "returned_rows": len(rows),
            "rows": rows,
            "provenance": {
                "analysis_version": ANALYSIS_VERSION,
                "release_generation": FORMAL23_RELEASE_GENERATION,
                "source_generation": "V3.2_R7_FRESH_FROM_RAW_H5",
                "success_sha256": self.success_sha256,
                "binding_sha256": self.binding_sha256,
                "independent_audit_sha256": self.audit_sha256,
                "historical_results_relabelled_fresh": False,
            },
        }

    def query_associations(
        self,
        *,
        cancer_id: Any,
        lncrna_id: Any | None = None,
        pathway_id: Any | None = None,
        compartment: Any | None = None,
        cell_type: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
    ) -> dict[str, Any]:
        self._assert_outer()
        cancer = self._cancer(cancer_id)
        lnc = _clean_identifier(lncrna_id, "lncrna_id")
        pathway = _clean_identifier(pathway_id, "pathway_id")
        compartment_value = _clean_compartment(compartment)
        cell = _clean_cell_type(cell_type)
        availability_value = _availability(availability)
        limit_value, offset_value = _bounds(limit, offset)
        filters = {
            "cancer_id": cancer,
            "lncrna_id": lnc,
            "pathway_id": pathway,
            "compartment": compartment_value,
            "cell_type": cell,
            "cell_type_filter_applied": False,
            "cell_type_resolution": "COMPARTMENT_ONLY",
            "availability": availability_value,
            "limit": limit_value,
            "offset": offset_value,
        }
        if cancer in self.typed_unavailable:
            return self._typed_unavailable_response(
                cancer, "donor_blocked_lncrna_exact_pathway_association", filters
            )
        if availability_value == "UNAVAILABLE":
            return self._response(
                "donor_blocked_lncrna_exact_pathway_association", [], filters
            )
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        for column, value in (
            ("lncrna_id", lnc),
            ("pathway_id", pathway),
            ("compartment", compartment_value),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        path = self._artifact(cancer, "association_evidence.parquet")
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT dataset_id, cancer_id, compartment, lncrna_id, lncrna_symbol,
                       pathway_id, n_donors, spearman_rho, nominal_p,
                       bh_q_global_tests, fdr_0_10_pass,
                       true AS association_available,
                       NULL::VARCHAR AS unavailable_reason,
                       multiple_testing_adjustment,
                       bh_q_is_conservative_upper_bound, biological_unit,
                       cell_as_independent_replicate, evidence_scope,
                       source_generation
                FROM read_parquet(?)
                WHERE {' AND '.join(clauses)}
                ORDER BY fdr_0_10_pass DESC, bh_q_global_tests,
                         abs(spearman_rho) DESC, lncrna_id, pathway_id
                LIMIT ? OFFSET ?
                """,
                [str(path), *parameters, limit_value, offset_value],
            ).fetchdf()
        finally:
            connection.close()
        return self._response(
            "donor_blocked_lncrna_exact_pathway_association",
            _records(frame),
            filters,
        )

    def query_lncrna_celltype(
        self,
        *,
        cancer_id: Any,
        lncrna_id: Any | None = None,
        cell_type: Any | None = None,
        compartment: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
    ) -> dict[str, Any]:
        self._assert_outer()
        cancer = self._cancer(cancer_id)
        lnc = _clean_identifier(lncrna_id, "lncrna_id")
        cell = _clean_cell_type(cell_type)
        compartment_value = _clean_compartment(compartment)
        availability_value = _availability(availability)
        limit_value, offset_value = _bounds(limit, offset)
        filters = {
            "cancer_id": cancer,
            "lncrna_id": lnc,
            "cell_type": cell,
            "compartment": compartment_value,
            "availability": availability_value,
            "limit": limit_value,
            "offset": offset_value,
        }
        if cancer in self.typed_unavailable:
            return self._typed_unavailable_response(
                cancer, "donor_aggregated_lncrna_celltype_expression", filters
            )
        if availability_value == "UNAVAILABLE":
            return self._response(
                "donor_aggregated_lncrna_celltype_expression", [], filters
            )
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        for column, value in (
            ("lncrna_id", lnc),
            ("cell_type_major", cell),
            ("compartment", compartment_value),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        path = self._artifact(cancer, "lncrna_donor_celltype_summary.parquet")
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT dataset_id, cancer_id, lncrna_id, lncrna_symbol,
                       cell_type_major AS cell_type, compartment,
                       count(DISTINCT patient_id) AS donor_count,
                       sum(cell_count) AS cell_count,
                       sum(detected_cell_count) AS detected_cell_count,
                       sum(detected_cell_count)::DOUBLE / nullif(sum(cell_count), 0)
                         AS detection_rate,
                       sum(mean_expression * cell_count) / nullif(sum(cell_count), 0)
                         AS mean_expression,
                       first(expression_summary_scale) AS expression_summary_scale,
                       true AS availability, NULL::VARCHAR AS unavailable_reason,
                       first(source_generation) AS source_generation
                FROM read_parquet(?)
                WHERE {' AND '.join(clauses)}
                GROUP BY dataset_id, cancer_id, lncrna_id, lncrna_symbol,
                         cell_type_major, compartment
                ORDER BY detection_rate DESC, lncrna_id, cell_type
                LIMIT ? OFFSET ?
                """,
                [str(path), *parameters, limit_value, offset_value],
            ).fetchdf()
        finally:
            connection.close()
        return self._response(
            "donor_aggregated_lncrna_celltype_expression",
            _records(frame),
            filters,
        )

    def query_pathway_activity(
        self,
        *,
        cancer_id: Any,
        pathway_id: Any | None = None,
        cell_type: Any | None = None,
        compartment: Any | None = None,
        availability: Any = "ALL",
        limit: Any = 100,
        offset: Any = 0,
        **_: Any,
    ) -> dict[str, Any]:
        self._assert_outer()
        cancer = self._cancer(cancer_id)
        pathway = _clean_identifier(pathway_id, "pathway_id")
        cell = _clean_cell_type(cell_type)
        compartment_value = _clean_compartment(compartment)
        availability_value = _availability(availability)
        limit_value, offset_value = _bounds(limit, offset)
        filters = {
            "cancer_id": cancer,
            "pathway_id": pathway,
            "cell_type": cell,
            "compartment": compartment_value,
            "availability": availability_value,
            "limit": limit_value,
            "offset": offset_value,
        }
        if cancer in self.typed_unavailable:
            return self._typed_unavailable_response(
                cancer, "donor_aggregated_celltype_pathway_activity", filters
            )
        clauses = ["cancer_id = ?"]
        parameters: list[Any] = [cancer]
        for column, value in (
            ("pathway_id", pathway),
            ("cell_type_major", cell),
            ("compartment", compartment_value),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if availability_value != "ALL":
            clauses.append("ucell_available = ?")
            parameters.append(availability_value == "AVAILABLE")
        path = self._artifact(cancer, "pathway_donor_celltype_summary.parquet")
        connection = self._connect()
        try:
            frame = connection.execute(
                f"""
                SELECT dataset_id, cancer_id, pathway_id,
                       cell_type_major AS cell_type, compartment,
                       count(DISTINCT patient_id) AS donor_count,
                       count(DISTINCT patient_id) FILTER (WHERE ucell_available)
                         AS available_donor_count,
                       sum(cell_count) AS cell_count,
                       avg(ucell_score_mean) FILTER (WHERE ucell_available)
                         AS ucell_score_mean,
                       bool_or(ucell_available) AS availability,
                       CASE WHEN bool_or(ucell_available) THEN NULL
                            ELSE string_agg(DISTINCT nullif(unavailable_reason, ''), ';')
                       END AS unavailable_reason,
                       first(biological_aggregation_unit) AS biological_aggregation_unit,
                       first(source_generation) AS source_generation
                FROM read_parquet(?)
                WHERE {' AND '.join(clauses)}
                GROUP BY dataset_id, cancer_id, pathway_id,
                         cell_type_major, compartment
                ORDER BY availability DESC, ucell_score_mean DESC NULLS LAST,
                         pathway_id, cell_type
                LIMIT ? OFFSET ?
                """,
                [str(path), *parameters, limit_value, offset_value],
            ).fetchdf()
        finally:
            connection.close()
        return self._response(
            "donor_aggregated_celltype_pathway_activity",
            _records(frame),
            filters,
        )

    def coverage_summary(
        self,
        *,
        cancer_id: Any | None = None,
        formal_input_eligible: bool | None = None,
        limit: Any = 33,
        offset: Any = 0,
        **_: Any,
    ) -> dict[str, Any]:
        self._assert_outer()
        selected = self._cancer(cancer_id) if cancer_id is not None else None
        if formal_input_eligible is not None and not isinstance(
            formal_input_eligible, bool
        ):
            raise SingleCellFormal23InputError(
                "formal_input_eligible must be true, false or null"
            )
        limit_value, offset_value = _bounds(limit, offset)
        rows = []
        for cancer in sorted(ALL_CANCERS):
            if selected is not None and cancer != selected:
                continue
            is_formal = cancer in self.cancer_metadata
            if (
                formal_input_eligible is not None
                and is_formal is not formal_input_eligible
            ):
                continue
            if is_formal:
                rows.append(
                    {
                        "cancer_id": cancer,
                        "formal_input_eligible": True,
                        "availability": True,
                        "unavailable_reason": None,
                        **self.cancer_metadata[cancer],
                    }
                )
            else:
                rows.append(
                    {
                        "cancer_id": cancer,
                        "formal_input_eligible": False,
                        "availability": False,
                        "unavailable_reason": self.typed_unavailable[cancer],
                        "dataset_id": None,
                        "cells": None,
                        "donors": None,
                        "mapped_source_universe_lncrnas": None,
                        "association_evidence_rows": None,
                        "generated_in_current_r11_run": False,
                    }
                )
        paged = rows[offset_value : offset_value + limit_value]
        response = self._response(
            "formal23_coverage_summary",
            paged,
            {
                "cancer_id": selected,
                "formal_input_eligible": formal_input_eligible,
                "limit": limit_value,
                "offset": offset_value,
            },
        )
        response["total_rows"] = len(rows)
        return response

    @staticmethod
    def _download_filename(download_id: str) -> str:
        if download_id == "single_cell_associations":
            return "association_evidence.parquet"
        if download_id in {"single_cell_activity", "single_cell_ucell"}:
            return "pathway_donor_celltype_summary.parquet"
        raise SingleCellFormal23InputError(
            f"single-cell download is not a formal-23 data product: {download_id}"
        )

    def download_entry(self, download_id: str) -> dict[str, Any]:
        """Describe a current formal-23 partitioned download, or its typed gap."""

        self._assert_outer()
        if download_id not in SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS:
            raise SingleCellFormal23InputError(
                f"unknown single-cell download: {download_id}"
            )
        if download_id not in SINGLE_CELL_FORMAL23_DOWNLOAD_IDS:
            reason = (
                "FORMAL23_PSEUDOTIME_NOT_GENERATED"
                if download_id == "single_cell_pseudotime"
                else "NO_HASH_BOUND_FRESH_V32_SINGLE_CELL_FIGURE_FILES"
            )
            return {
                "download_id": download_id,
                "capability_ids": ["single_cell"],
                "status": "GAP_TYPED_UNAVAILABLE",
                "download_implemented": False,
                "data_present": False,
                "contract_complete": True,
                "model_version": "V3.2",
                "unavailable_reason": reason,
                "supersedes_historical_single_cell": True,
            }
        filename = self._download_filename(download_id)
        parts = []
        for cancer in sorted(FORMAL_CANCERS):
            path, digest = self.paths[cancer][filename]
            parts.append(
                {
                    "cancer_id": cancer,
                    "relative_name": f"{cancer}/{filename}",
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
        return {
            "download_id": download_id,
            "capability_ids": ["single_cell"],
            "status": "READY_PARTS",
            "download_implemented": True,
            "data_present": True,
            "contract_complete": True,
            "model_version": "V3.2",
            "release_generation": FORMAL23_RELEASE_GENERATION,
            "source_generation": "V3.2_R7_FRESH_FROM_RAW_H5",
            "biological_unit": "DONOR",
            "cell_as_independent_replicate": False,
            "formal_eligible_cancer_count": 23,
            "typed_unavailable_cancer_count": 10,
            "full_33_single_cell_coverage_claimed": False,
            "supersedes_historical_single_cell": True,
            "sha256_tree": self.download_tree_sha256,
            "parts": parts,
        }

    def resolve_download_part(
        self, download_id: str, part_index: Any
    ) -> dict[str, Any]:
        """Resolve and re-hash one formal-23 cancer partition at request time."""

        entry = self.download_entry(download_id)
        if entry["status"] != "READY_PARTS":
            raise SingleCellFormal23InputError(
                f"single-cell download {download_id} is unavailable"
            )
        if isinstance(part_index, bool):
            raise SingleCellFormal23InputError("download part index is invalid")
        try:
            index = int(part_index)
        except (TypeError, ValueError) as exc:
            raise SingleCellFormal23InputError("download part index is invalid") from exc
        parts = entry["parts"]
        if index < 0 or index >= len(parts):
            raise SingleCellFormal23InputError("unknown single-cell download part index")
        part = parts[index]
        cancer = str(part["cancer_id"])
        filename = self._download_filename(download_id)
        path = self._artifact(cancer, filename)
        return {
            "kind": "part",
            "download_id": download_id,
            "part_index": index,
            "path": str(path),
            "relative_name": part["relative_name"],
            "sha256": part["sha256"],
            "sha256_tree": entry["sha256_tree"],
            "bytes": part["bytes"],
            "cancer_id": cancer,
        }


__all__ = [
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_OFFSET",
    "SINGLE_CELL_FORMAL23_DOWNLOAD_IDS",
    "SINGLE_CELL_SUPERSEDED_DOWNLOAD_IDS",
    "SingleCellFormal23AssetError",
    "SingleCellFormal23InputError",
    "SingleCellFormal23Query",
]
