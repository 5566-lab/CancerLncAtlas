"""Hash-pinned publication/query layer for fresh V3.2 single-cell diagnostics.

The diagnostic trajectory uses an inferred CytoTRACE2/UCell consensus root.
It is therefore served as secondary diagnostic evidence with zero primary and
secondary model weight, never as an explicit experimental trajectory root.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
SOURCE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_ARTIFACT_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_PUBLICATION_BINDING_V1"
BINDING_STATUS = "PASS_HASH_BOUND_17_CANCER_DIAGNOSTIC_PUBLICATION"
PAGE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_QUERY_PAGE_V1"
DOWNLOAD_MANIFEST_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_DOWNLOAD_MANIFEST_V1"
)
ROOT_PROVENANCE = "INFERRED_CYTOTRACE2_UCELL_CONSENSUS"
FORMAL_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML",
    "LGG", "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM",
    "UCEC",
)
EXPECTED_PATHWAYS = 2_135
EXPECTED_DOROTHEA_EDGES = 13_153
EXPECTED_EXCLUDED_DOROTHEA_EDGES = 70
EXPECTED_FIGURES = (
    "cytotrace2_score_umap.pdf",
    "dynamic_gene_heatmap.pdf",
    "immune_subtype_umap.pdf",
    "monocle3_branches.pdf",
    "monocle3_pseudotime.pdf",
    "pseudotime_0_1_umap.pdf",
    "stemness_cluster_consensus.pdf",
    "stemness_ucell_umap.pdf",
    "trajectory_cluster_umap.pdf",
)
MAX_QUERY_LIMIT = 500
MAX_QUERY_OFFSET = 2_000_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SingleCellDiagnosticPublicationError(RuntimeError):
    """Base error for the diagnostic publication contract."""


class SingleCellDiagnosticAssetError(SingleCellDiagnosticPublicationError):
    """Raised when a bound artifact is missing, unsafe, or has drifted."""


class SingleCellDiagnosticInputError(SingleCellDiagnosticPublicationError):
    """Raised when a query parameter violates the public contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SingleCellDiagnosticAssetError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellDiagnosticAssetError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SingleCellDiagnosticAssetError(f"{label} must be a JSON object")
    return value


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise SingleCellDiagnosticAssetError(f"{label} is missing or unsafe: {source}")
    return source


def _declared_sha(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if _SHA256.fullmatch(digest) is None:
        raise SingleCellDiagnosticAssetError(f"Invalid SHA256 for {label}")
    return digest


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_cancer(value: str) -> str:
    if not isinstance(value, str):
        raise SingleCellDiagnosticInputError("cancer_id must be a string")
    cancer = value.strip().upper()
    if cancer not in FORMAL_CANCERS:
        raise SingleCellDiagnosticInputError(
            "cancer_id must be one of: " + ", ".join(FORMAL_CANCERS)
        )
    return cancer


def _normalise_text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SingleCellDiagnosticInputError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > 512:
        raise SingleCellDiagnosticInputError(
            f"{label} must be non-empty and at most 512 characters"
        )
    return text


def _validate_page(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise SingleCellDiagnosticInputError(
            f"limit must be an integer in 1..{MAX_QUERY_LIMIT}"
        )
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= MAX_QUERY_OFFSET:
        raise SingleCellDiagnosticInputError(
            f"offset must be an integer in 0..{MAX_QUERY_OFFSET}"
        )
    return limit, offset


def _normalise_relative_path(value: str) -> str:
    if not isinstance(value, str):
        raise SingleCellDiagnosticInputError("relative_path must be a string")
    candidate = PurePosixPath(value.strip().replace("\\", "/"))
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or ":" in candidate.parts[0]
    ):
        raise SingleCellDiagnosticInputError("relative_path is unsafe")
    return candidate.as_posix()


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
        for row in frame.to_dict(orient="records")
    ]


def _tsv_rows(path: Path) -> int:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="strict") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    source = _safe_file(path, path.name)
    record: dict[str, Any] = {
        "path": str(source),
        "sha256": artifact_sha256(source),
        "bytes": source.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _numeric_pseudotime_rows(path: Path) -> int:
    total = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=["pseudotime", "pseudotime_0_1"],
        chunksize=250_000,
        low_memory=False,
    ):
        total += int(
            pd.to_numeric(chunk["pseudotime"], errors="coerce").notna().sum()
        )
    return total


def _load_dorothea_qa(path: Path, cancer: str) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    _require(len(frame) == 1, f"{cancer} DoRothEA QA must have one row")
    row = frame.iloc[0].to_dict()
    _require(str(row.get("status")) == "PASS", f"{cancer} DoRothEA QA did not PASS")
    _require(
        int(row.get("exact_unweighted_edges", -1)) == EXPECTED_DOROTHEA_EDGES
        and int(row.get("signed_intersection_edges", -1)) == EXPECTED_DOROTHEA_EDGES
        and int(row.get("missing_exact_signed_edges", -1)) == 0
        and int(row.get("excluded_signed_reference_edges", -1))
        == EXPECTED_EXCLUDED_DOROTHEA_EDGES,
        f"{cancer} DoRothEA exact signed intersection drift",
    )
    return {str(key): _json_value(value) for key, value in row.items()}


def materialize_publication_binding(
    *,
    diagnostic_manifest_path: str | Path,
    expected_diagnostic_manifest_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Bind all 17 fresh diagnostic outputs without copying their payloads."""

    source_path = _safe_file(diagnostic_manifest_path, "diagnostic manifest")
    expected_sha = _declared_sha(
        expected_diagnostic_manifest_sha256, "diagnostic manifest"
    )
    _require(
        artifact_sha256(source_path) == expected_sha,
        "Diagnostic manifest SHA drift",
    )
    source = _load_json(source_path, "diagnostic manifest")
    _require(source.get("format") == SOURCE_FORMAT, "Diagnostic manifest format drift")
    _require(source.get("analysis_version") == ANALYSIS_VERSION, "Analysis version drift")
    _require(
        source.get("status") == "DIAGNOSTIC_COMPLETE_NOT_MODEL_FUSED",
        "Diagnostic computation is not complete",
    )
    _require(source.get("cancers") == list(FORMAL_CANCERS), "17-cancer order drift")
    _require(source.get("exact_pathway_count") == EXPECTED_PATHWAYS, "Pathway count drift")
    for field, expected in (
        ("historical_predictions_used", False),
        ("historical_rankings_used", False),
        ("historical_checkpoints_used", False),
        ("historical_sc_trajectory_outputs_used", False),
        ("diagnostic_only", True),
        ("model_fusion_permitted", False),
        ("primary_score_weight", 0),
        ("secondary_score_weight", 0),
        ("root_is_explicit", False),
        ("production_deployed", False),
    ):
        _require(source.get(field) == expected, f"Invalid source policy: {field}")
    _require(source.get("root_provenance") == ROOT_PROVENANCE, "Root provenance drift")

    source_artifacts = source.get("artifacts")
    _require(isinstance(source_artifacts, dict), "Diagnostic artifact map is missing")
    cancer_records: list[dict[str, Any]] = []
    total_numeric = 0
    total_pathway_rows = 0
    total_figures = 0
    for cancer in FORMAL_CANCERS:
        declared = source_artifacts.get(cancer)
        _require(isinstance(declared, dict), f"Diagnostic manifest lacks {cancer}")
        for role in ("activity", "pathway_stats", "lncrna_pathway"):
            record = declared.get(role)
            _require(isinstance(record, dict), f"{cancer} lacks {role}")
            path = _safe_file(record.get("path", ""), f"{cancer} {role}")
            _require(
                artifact_sha256(path) == _declared_sha(record.get("sha256"), f"{cancer} {role}"),
                f"{cancer} {role} SHA drift",
            )
        stage = Path(str(declared["activity"]["path"])).resolve().parent
        role_paths = {
            "activity": stage / "v32_exact2135_activity.tsv.gz",
            "pathway_stats": stage / "v32_exact2135_pathway_stats.tsv.gz",
            "lncrna_pathway": stage / "v32_exact2135_lncrna_pathway.tsv.gz",
            "pseudotime_cells": stage / "sc_malignant_pseudotime.tsv.gz",
        }
        artifacts: dict[str, Any] = {}
        for role, path in role_paths.items():
            rows = (
                int(declared[role]["rows"])
                if role in declared
                else _tsv_rows(_safe_file(path, f"{cancer} {role}"))
            )
            artifacts[role] = _file_record(path, rows=rows)
        numeric_rows = _numeric_pseudotime_rows(role_paths["pseudotime_cells"])
        _require(numeric_rows > 0, f"{cancer} has no numeric pseudotime values")
        artifacts["pseudotime_cells"]["numeric_pseudotime_rows"] = numeric_rows

        qa_path = stage / "dorothea_signed_intersection_qa.tsv"
        qa = _load_dorothea_qa(qa_path, cancer)
        qa_record = _file_record(qa_path, rows=1)
        qa_record["summary"] = qa

        figures: list[dict[str, Any]] = []
        for name in EXPECTED_FIGURES:
            path = stage / "figures" / name
            record = _file_record(path)
            record.update(
                {
                    "figure_id": Path(name).stem,
                    "file_name": name,
                    "media_type": "application/pdf",
                    "relative_path": f"figures/{name}",
                }
            )
            figures.append(record)

        downloads = {
            "activity.tsv.gz": artifacts["activity"],
            "pathway_stats.tsv.gz": artifacts["pathway_stats"],
            "lncrna_pathway.tsv.gz": artifacts["lncrna_pathway"],
            "pseudotime_cells.tsv.gz": artifacts["pseudotime_cells"],
            **{row["relative_path"]: row for row in figures},
        }
        cancer_records.append(
            {
                "cancer_id": cancer,
                "artifacts": artifacts,
                "dorothea_qa": qa_record,
                "figures": figures,
                "downloads": downloads,
                "numeric_pseudotime_rows": numeric_rows,
                "figure_count": len(figures),
            }
        )
        total_numeric += numeric_rows
        total_pathway_rows += int(artifacts["pathway_stats"]["rows"])
        total_figures += len(figures)

    payload: dict[str, Any] = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": BINDING_STATUS,
        "diagnostic_manifest": {"path": str(source_path), "sha256": expected_sha},
        "cancers": list(FORMAL_CANCERS),
        "cancer_count": len(FORMAL_CANCERS),
        "exact_pathway_count": EXPECTED_PATHWAYS,
        "records": cancer_records,
        "numeric_pseudotime_rows": total_numeric,
        "exact_pathway_association_rows": total_pathway_rows,
        "figure_count": total_figures,
        "contract_artifacts": {
            "v32_sc_pseudotime": {
                "status": "SERVER_HASH_PINNED_DIAGNOSTIC_READY",
                "rows": total_numeric,
                "cancers": len(FORMAL_CANCERS),
                "root_provenance": ROOT_PROVENANCE,
                "root_is_explicit": False,
            },
            "v32_sc_figure_manifest": {
                "status": "SERVER_HASH_PINNED_FIGURES_READY",
                "rows": total_figures,
                "cancers": len(FORMAL_CANCERS),
            },
        },
        "root_provenance": ROOT_PROVENANCE,
        "root_is_explicit": False,
        "diagnostic_only": True,
        "model_fusion_permitted": False,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "historical_outputs_used": False,
        "changes_primary_ranking": False,
        "release_ready_for_staging": True,
        "production_deployed": False,
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    destination = Path(output_root).resolve()
    _require(not destination.exists(), f"Refusing to overwrite output root: {destination}")
    destination.mkdir(parents=True)
    binding_path = destination / "SINGLE_CELL_DIAGNOSTIC_PUBLICATION_BINDING.json"
    _atomic_json(binding_path, payload)
    success = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_PUBLICATION_SUCCESS_V1",
        "status": "PASS",
        "binding_path": str(binding_path),
        "binding_sha256": artifact_sha256(binding_path),
        "cancer_count": len(FORMAL_CANCERS),
        "numeric_pseudotime_rows": total_numeric,
        "figure_count": total_figures,
        "production_deployed": False,
    }
    _atomic_json(destination / "SUCCESS.json", success)
    return success


class SingleCellDiagnosticPublicationQuery:
    """Read-only, request-time-rehashed diagnostic query surface."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str,
    ) -> None:
        self.binding_path = _safe_file(binding_path, "diagnostic publication binding")
        expected = _declared_sha(expected_binding_sha256, "publication binding")
        _require(
            artifact_sha256(self.binding_path) == expected,
            "Diagnostic publication binding SHA drift",
        )
        binding = _load_json(self.binding_path, "diagnostic publication binding")
        _require(binding.get("format") == BINDING_FORMAT, "Publication format drift")
        _require(binding.get("analysis_version") == ANALYSIS_VERSION, "Analysis version drift")
        _require(binding.get("status") == BINDING_STATUS, "Publication is not ready")
        _require(binding.get("cancers") == list(FORMAL_CANCERS), "Cancer order drift")
        _require(binding.get("root_provenance") == ROOT_PROVENANCE, "Root provenance drift")
        for field, expected_value in (
            ("root_is_explicit", False),
            ("diagnostic_only", True),
            ("model_fusion_permitted", False),
            ("primary_score_weight", 0),
            ("secondary_score_weight", 0),
            ("historical_outputs_used", False),
            ("changes_primary_ranking", False),
            ("release_ready_for_staging", True),
            ("production_deployed", False),
        ):
            _require(binding.get(field) == expected_value, f"Invalid publication policy: {field}")
        raw_records = binding.get("records")
        _require(isinstance(raw_records, list) and len(raw_records) == 17, "Record count drift")
        self.records = {str(row.get("cancer_id")): row for row in raw_records}
        _require(set(self.records) == set(FORMAL_CANCERS), "Publication cancer set drift")
        self.binding = binding

    @staticmethod
    def _verify_record(record: Mapping[str, Any], label: str) -> Path:
        path = _safe_file(record.get("path", ""), label)
        expected = _declared_sha(record.get("sha256"), label)
        _require(artifact_sha256(path) == expected, f"Request-time SHA drift: {label}")
        _require(path.stat().st_size == int(record.get("bytes", -1)), f"Size drift: {label}")
        return path

    @staticmethod
    def _read_page(
        path: Path,
        *,
        limit: int,
        offset: int,
        pathway_id: str | None,
    ) -> tuple[list[dict[str, Any]], int]:
        rows: list[pd.DataFrame] = []
        matched = 0
        remaining_offset = offset
        for chunk in pd.read_csv(
            path, sep="\t", compression="gzip", chunksize=100_000, low_memory=False
        ):
            if pathway_id is not None:
                if "pathway_id" not in chunk.columns:
                    raise SingleCellDiagnosticInputError(
                        "pathway_id is valid only for PATHWAY-level trajectory"
                    )
                chunk = chunk.loc[chunk["pathway_id"].astype(str) == pathway_id]
            matched += len(chunk)
            if remaining_offset >= len(chunk):
                remaining_offset -= len(chunk)
                continue
            selected = chunk.iloc[remaining_offset : remaining_offset + (limit - sum(map(len, rows)))]
            remaining_offset = 0
            if not selected.empty:
                rows.append(selected)
            if sum(map(len, rows)) >= limit:
                break
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return _records(frame), matched

    def capability_status(self) -> dict[str, Any]:
        return {
            "module": "single_cell_diagnostic",
            "analysis_version": ANALYSIS_VERSION,
            "status": self.binding["status"],
            "cancers": list(FORMAL_CANCERS),
            "cancer_count": 17,
            "numeric_pseudotime_rows": self.binding["numeric_pseudotime_rows"],
            "figure_count": self.binding["figure_count"],
            "root_provenance": ROOT_PROVENANCE,
            "root_is_explicit": False,
            "diagnostic_only": True,
            "model_fusion_permitted": False,
            "primary_score_weight": 0,
            "secondary_score_weight": 0,
            "historical_outputs_used": False,
            "production_deployed": False,
        }

    def query_trajectory(
        self,
        *,
        cancer_id: str,
        level: str = "PATHWAY",
        pathway_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        normal_level = str(level).strip().upper()
        if normal_level not in {"PATHWAY", "CELL"}:
            raise SingleCellDiagnosticInputError("level must be PATHWAY or CELL")
        pathway = _normalise_text(pathway_id, "pathway_id")
        if normal_level == "CELL" and pathway is not None:
            raise SingleCellDiagnosticInputError(
                "pathway_id is valid only for PATHWAY-level trajectory"
            )
        limit, offset = _validate_page(limit, offset)
        role = "pathway_stats" if normal_level == "PATHWAY" else "pseudotime_cells"
        record = self.records[cancer]["artifacts"][role]
        path = self._verify_record(record, f"{cancer} {role}")
        rows, matched = self._read_page(
            path, limit=limit, offset=offset, pathway_id=pathway
        )
        return {
            "format": PAGE_FORMAT,
            "module": "single_cell_diagnostic",
            "query_kind": "numeric_pseudotime_trajectory",
            "cancer_id": cancer,
            "level": normal_level,
            "pathway_id": pathway,
            "availability": True,
            "rows": rows,
            "returned_rows": len(rows),
            "matched_rows_scanned": matched,
            "limit": limit,
            "offset": offset,
            "pseudotime_numeric_values": int(
                self.records[cancer]["numeric_pseudotime_rows"]
            ),
            "root_provenance": ROOT_PROVENANCE,
            "root_is_explicit": False,
            "diagnostic_only": True,
            "primary_score_weight": 0,
            "secondary_score_weight": 0,
            "changes_primary_ranking": False,
            "production_deployed": False,
        }

    def figure_manifest(self, *, cancer_id: str) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        rows = []
        for record in self.records[cancer]["figures"]:
            self._verify_record(record, f"{cancer} figure {record.get('figure_id')}")
            rows.append(
                {
                    "figure_id": record["figure_id"],
                    "file_name": record["file_name"],
                    "media_type": record["media_type"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                    "download_relative_path": record["relative_path"],
                }
            )
        return {
            "module": "single_cell_diagnostic",
            "query_kind": "formal_figure_availability",
            "cancer_id": cancer,
            "availability": True,
            "rows": rows,
            "returned_rows": len(rows),
            "root_provenance": ROOT_PROVENANCE,
            "root_is_explicit": False,
            "diagnostic_only": True,
            "production_deployed": False,
        }

    def resolve_figure(self, *, cancer_id: str, figure_id: str) -> dict[str, Any] | None:
        cancer = _normalise_cancer(cancer_id)
        figure = _normalise_text(figure_id, "figure_id")
        assert figure is not None
        for record in self.records[cancer]["figures"]:
            if record["figure_id"] == figure:
                path = self._verify_record(record, f"{cancer} figure {figure}")
                return {
                    "path": path,
                    "media_type": record["media_type"],
                    "file_name": record["file_name"],
                    "sha256": record["sha256"],
                }
        return None

    def download_manifest(self, *, cancer_id: str) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        rows = []
        for relative, record in sorted(self.records[cancer]["downloads"].items()):
            path = self._verify_record(record, f"{cancer} download {relative}")
            rows.append(
                {
                    "relative_path": relative,
                    "sha256": record["sha256"],
                    "bytes": path.stat().st_size,
                    "rows": record.get("rows"),
                }
            )
        return {
            "format": DOWNLOAD_MANIFEST_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "cancer_id": cancer,
            "files": rows,
            "file_count": len(rows),
            "request_time_payload_rehash": True,
            "root_provenance": ROOT_PROVENANCE,
            "root_is_explicit": False,
            "diagnostic_only": True,
            "production_deployed": False,
        }

    def resolve_download(self, *, cancer_id: str, relative_path: str) -> dict[str, Any]:
        cancer = _normalise_cancer(cancer_id)
        relative = _normalise_relative_path(relative_path)
        record = self.records[cancer]["downloads"].get(relative)
        if not isinstance(record, Mapping):
            raise SingleCellDiagnosticInputError(
                "relative_path is not a declared diagnostic release artifact"
            )
        path = self._verify_record(record, f"{cancer} download {relative}")
        media_type = "application/pdf" if relative.endswith(".pdf") else "application/gzip"
        return {
            "path": path,
            "media_type": media_type,
            "file_name": Path(relative).name,
            "sha256": record["sha256"],
        }


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "BINDING_STATUS",
    "FORMAL_CANCERS",
    "MAX_QUERY_LIMIT",
    "SingleCellDiagnosticPublicationError",
    "SingleCellDiagnosticAssetError",
    "SingleCellDiagnosticInputError",
    "SingleCellDiagnosticPublicationQuery",
    "materialize_publication_binding",
]
