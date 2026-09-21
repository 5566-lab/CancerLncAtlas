"""Fresh V3.2 experimental-perturbation evidence materialisation.

This module is deliberately a *fact layer*, not another predictor.  It reads
only the current raw evidence-event table, the formal V3.2 exact candidate
universe and the static exact-pathway member table used by the current
Evidence Transformer lineage.  Functional-perturbation records are mapped
literally through ``partner_id -> exact pathway member``.  A pathway family is
never expanded and no historical score, rank, prediction or checkpoint is an
admissible input.

The resulting facts may affect the confidence endpoint only after the current
fresh Evidence Transformer has a final binding and a verified ablation.  This
materialiser therefore never creates a probability and always emits
``release_ready=false`` while that binding is pending.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from cc_hhgt.v32.evidence_training import (
    ANALYSIS_VERSION,
    EvidenceTrainingContractError,
    _cancer,
    _clean,
    _direction_target,
    _identity,
    build_exact_event_bags,
    file_sha256,
    materialize_candidate_events,
)


MODULE_ID = "experiment_perturbation"
MATERIALISATION_FORMAT = "CC_HHGT_V3_2_EXPERIMENT_PERTURBATION_FACTS_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_EXPERIMENT_PERTURBATION_BINDING_V1"

FORMAL_EVIDENCE_EVENT_SHA256 = (
    "b330a54c1865e4bfb1c9f16374f79ec7aa46b170c364d0e6c43541e6a5375626"
)
FORMAL_CANDIDATE_UNIVERSE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_PATHWAY_MEMBERS_SHA256 = (
    "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
)
FORMAL_ASSAY_DETAIL_SHA256 = (
    "a31c366e780cb7ae3f4e9f974e10d1db5b6fd57ce72144b63058a6f2f28f3c7f"
)
FORMAL_INPUT_ROWS = {
    "evidence_event": 479_238,
    "candidate_universe": 3_300_000,
    "pathway_members": 375_002,
    "selected_functional_perturbation": 2_155,
}
FORMAL_ASSAY_DETAIL_ROWS = 2_155

ASSAY_DETAIL_COLUMNS = (
    "evidence_event_id",
    "assay_detail",
    "assay_detail_available",
    "assay_detail_unavailable_reason",
    "perturbation_methods",
    "perturbation_method_available",
    "readout_assays",
    "readout_assay_available",
    "match_route",
    "evidence_strength",
    "manual_review_required",
    "source_database",
    "source_record_ids",
    "source_context",
    "changes_primary_ranking",
    "changes_discovery_ranking",
)

ARTIFACT_FILENAMES = {
    "v32_experiment_events": "v32_experiment_events.parquet",
    "v32_perturbation_exact_pathway": "v32_perturbation_exact_pathway.parquet",
    "v32_experiment_provenance": "v32_experiment_provenance.parquet",
    "v32_experiment_rejected": "v32_experiment_rejected.parquet",
}
MANIFEST_FILENAME = "EXPERIMENT_PERTURBATION_MANIFEST.json"
BINDING_FILENAME = "EXPERIMENT_PERTURBATION_BINDING.json"

PENDING_CONFIDENCE_REASON = (
    "CURRENT_V32_EVIDENCE_TRANSFORMER_FINAL_BINDING_AND_ABLATION_PENDING"
)

_DERIVED_COLUMN_TOKENS = {
    "label",
    "score",
    "rank",
    "prediction",
    "probability",
    "logit",
    "checkpoint",
    "oldconfidence",
    "neuralprobability",
    "pairevidencescore",
    "pairevidencelabel",
    "familyspf",
}


class ExperimentPerturbationContractError(EvidenceTrainingContractError):
    """Raised when the independent perturbation fact-layer contract is broken."""


@dataclass(frozen=True)
class ExperimentPerturbationMaterialisation:
    output_dir: Path
    artifact_paths: Mapping[str, Path]
    manifest_path: Path
    binding_path: Path
    manifest: Mapping[str, Any]
    binding: Mapping[str, Any]


def _normalised_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _nullable_text(value: object) -> object:
    text = _clean(value)
    return text if text else pd.NA


def _direction_class(target: int) -> object:
    return {0: "negative", 1: "neutral", 2: "positive"}.get(int(target), pd.NA)


def _stable_id(prefix: str, *values: object) -> str:
    payload = "\x1f".join("" if pd.isna(value) else str(value) for value in values)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parquet_write(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _assert_regular_local_input(path: Path, *, role: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ExperimentPerturbationContractError(f"{role} is not a local regular file: {resolved}")
    lowered = str(resolved).replace("\\", "/").lower()
    forbidden_path_patterns = (
        r"pair[_-]?evidence",
        r"(?:old|legacy|historical)[^/]*(?:prediction|ranking|score|checkpoint)",
        r"(?:prediction|ranking|checkpoint)[^/]*(?:old|legacy|historical)",
    )
    if any(re.search(pattern, lowered) for pattern in forbidden_path_patterns):
        raise ExperimentPerturbationContractError(
            f"{role} path looks like a forbidden historical/derived result: {resolved}"
        )
    return resolved


def _assert_no_derived_columns(frame: pd.DataFrame, *, role: str) -> None:
    forbidden: list[str] = []
    for column in frame.columns:
        token = _normalised_token(column)
        if token in _DERIVED_COLUMN_TOKENS or any(
            marker in token
            for marker in (
                "oldconfidence",
                "neuralprobability",
                "predictionscore",
                "pairevidence",
                "familyspf",
            )
        ):
            forbidden.append(str(column))
    if forbidden:
        raise ExperimentPerturbationContractError(
            f"{role} contains forbidden learned/historical columns: {sorted(forbidden)}"
        )


def _is_functional_perturbation(value: object) -> bool:
    text = re.sub(r"[^a-z0-9]+", "_", _clean(value).lower()).strip("_")
    tokens = set(text.split("_"))
    return text == "functional_perturbation" or "perturbation" in tokens


def _input_metadata(path: Path, frame: pd.DataFrame, sha256: str, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256,
        "bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "read_only": True,
    }


def _artifact_metadata(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }


def _normalise_output_types(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    for column in value.columns:
        if column.endswith("_available") or column in {
            "mapping_available",
            "family_broadcast_used",
            "evidence_fact_available",
            "release_ready",
            "changes_primary_ranking",
            "changes_discovery_ranking",
            "independent_probability_generated",
            "assay_detail_manual_review_required",
        }:
            value[column] = value[column].astype(bool)
    for column in (
        "direction_target",
        "source_occurrence_count",
        "exact_pathway_count",
    ):
        if column in value.columns:
            value[column] = pd.to_numeric(value[column], errors="coerce").astype("Int32")
    object_columns = value.select_dtypes(include=["object"]).columns
    for column in object_columns:
        value[column] = value[column].astype("string")
    return value


def _validate_assay_detail(
    frame: pd.DataFrame,
    *,
    selected_event_ids: pd.Series,
) -> pd.DataFrame:
    """Validate the additive assay sidecar without treating inferred text as fact."""

    missing = sorted(set(ASSAY_DETAIL_COLUMNS) - set(frame.columns))
    if missing:
        raise ExperimentPerturbationContractError(
            f"assay_detail lacks required columns: {missing}"
        )
    value = frame.loc[:, ASSAY_DETAIL_COLUMNS].copy()
    value["evidence_event_id"] = value["evidence_event_id"].map(_clean)
    if value.evidence_event_id.eq("").any() or value.evidence_event_id.duplicated().any():
        raise ExperimentPerturbationContractError(
            "assay_detail evidence_event_id must be non-empty and unique"
        )
    selected_ids = {_clean(item) for item in selected_event_ids}
    detail_ids = set(value.evidence_event_id)
    if selected_ids != detail_ids:
        raise ExperimentPerturbationContractError(
            "assay_detail must cover exactly the selected perturbation events; "
            f"missing={len(selected_ids - detail_ids)}, extra={len(detail_ids - selected_ids)}"
        )
    for column in (
        "assay_detail_available",
        "perturbation_method_available",
        "readout_assay_available",
        "manual_review_required",
        "changes_primary_ranking",
        "changes_discovery_ranking",
    ):
        if value[column].isna().any():
            raise ExperimentPerturbationContractError(
                f"assay_detail {column} must be explicitly true/false"
            )
        value[column] = value[column].astype(bool)
    if value.changes_primary_ranking.any() or value.changes_discovery_ranking.any():
        raise ExperimentPerturbationContractError(
            "assay_detail is an evidence sidecar and cannot change either ranking"
        )
    available_missing = value.assay_detail_available & value.assay_detail.map(
        lambda item: not bool(_clean(item))
    )
    if available_missing.any():
        raise ExperimentPerturbationContractError(
            "assay_detail_available=true requires a non-empty assay_detail"
        )
    unavailable_without_reason = (~value.assay_detail_available) & value[
        "assay_detail_unavailable_reason"
    ].map(lambda item: not bool(_clean(item)))
    if unavailable_without_reason.any():
        raise ExperimentPerturbationContractError(
            "assay_detail_available=false requires a typed unavailable reason"
        )
    return value


def _assay_detail_payload(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    if detail is None:
        return {
            "assay_detail": pd.NA,
            "assay_detail_available": False,
            "assay_detail_unavailable_reason": (
                "AUTHORITATIVE_INPUT_CONTAINS_FAMILY_LEVEL_ASSAY_ONLY"
            ),
            "perturbation_methods": pd.NA,
            "perturbation_method_available": False,
            "readout_assays": pd.NA,
            "readout_assay_available": False,
            "assay_detail_match_route": "SIDECAR_NOT_SUPPLIED",
            "assay_detail_evidence_strength": pd.NA,
            "assay_detail_manual_review_required": True,
            "assay_detail_source_database": pd.NA,
            "assay_detail_source_record_ids": pd.NA,
            "assay_detail_source_context": pd.NA,
        }
    return {
        "assay_detail": _nullable_text(detail.get("assay_detail", "")),
        "assay_detail_available": bool(detail.get("assay_detail_available", False)),
        "assay_detail_unavailable_reason": _nullable_text(
            detail.get("assay_detail_unavailable_reason", "")
        ),
        "perturbation_methods": _nullable_text(detail.get("perturbation_methods", "")),
        "perturbation_method_available": bool(
            detail.get("perturbation_method_available", False)
        ),
        "readout_assays": _nullable_text(detail.get("readout_assays", "")),
        "readout_assay_available": bool(detail.get("readout_assay_available", False)),
        "assay_detail_match_route": _nullable_text(detail.get("match_route", "")),
        "assay_detail_evidence_strength": _nullable_text(
            detail.get("evidence_strength", "")
        ),
        "assay_detail_manual_review_required": bool(
            detail.get("manual_review_required", True)
        ),
        "assay_detail_source_database": _nullable_text(
            detail.get("source_database", "")
        ),
        "assay_detail_source_record_ids": _nullable_text(
            detail.get("source_record_ids", "")
        ),
        "assay_detail_source_context": _nullable_text(detail.get("source_context", "")),
    }


def _formal_lnc_map(candidates: pd.DataFrame) -> dict[str, str]:
    mapping = pd.DataFrame(
        {
            "evidence_lncrna_id": candidates["lncrna_id"].map(_identity),
            "lncrna_id": candidates["lncrna_id"].map(_clean),
        }
    ).drop_duplicates()
    collisions = mapping.groupby("evidence_lncrna_id").lncrna_id.nunique()
    if (collisions > 1).any():
        bad = collisions.loc[collisions > 1].index[:5].tolist()
        raise ExperimentPerturbationContractError(
            f"Formal candidate lncRNA identifiers are not one-to-one after normalisation: {bad}"
        )
    return dict(mapping.itertuples(index=False, name=None))


def _formal_pathway_map(candidates: pd.DataFrame) -> dict[str, str]:
    mapping = pd.DataFrame(
        {
            "evidence_pathway_id": candidates["pathway_id"].map(_identity),
            "pathway_id": candidates["pathway_id"].map(_clean),
        }
    ).drop_duplicates()
    collisions = mapping.groupby("evidence_pathway_id").pathway_id.nunique()
    if (collisions > 1).any():
        bad = collisions.loc[collisions > 1].index[:5].tolist()
        raise ExperimentPerturbationContractError(
            f"Formal candidate pathway identifiers are not one-to-one after normalisation: {bad}"
        )
    return dict(mapping.itertuples(index=False, name=None))


def _build_source_event_table(
    selected: pd.DataFrame,
    events: pd.DataFrame,
    lineage: pd.DataFrame,
    rejected: pd.DataFrame,
    formal_lnc: Mapping[str, str],
    formal_pathway: Mapping[str, str],
    evidence_sha256: str,
    assay_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    event_paths = events[["event_id", "pathway_id"]].copy()
    mapped = lineage.merge(event_paths, on="event_id", how="left", validate="many_to_one")
    mapped["_row_key"] = mapped.source_row_index.map(str)
    rejected_value = rejected.copy()
    rejected_value["_row_key"] = rejected_value.source_row_index.map(str)

    mapped_groups = {key: group for key, group in mapped.groupby("_row_key", sort=False)}
    rejected_groups = {
        key: group for key, group in rejected_value.groupby("_row_key", sort=False)
    }
    detail_by_event = (
        {
            str(row["evidence_event_id"]): row
            for row in assay_detail.to_dict("records")
        }
        if assay_detail is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for source_index, row in selected.iterrows():
        key = str(source_index)
        mapped_group = mapped_groups.get(key)
        rejected_group = rejected_groups.get(key)
        has_mapping = mapped_group is not None and not mapped_group.empty
        has_rejection = rejected_group is not None and not rejected_group.empty
        if has_mapping == has_rejection:
            raise ExperimentPerturbationContractError(
                f"Source row {source_index} must have exactly one mapped/rejected disposition"
            )
        disposition = mapped_group if has_mapping else rejected_group
        assert disposition is not None
        first = disposition.iloc[0]
        evidence_lnc = _identity(row.get("lncrna_id", ""))
        partner = _identity(row.get("partner_id", ""))
        direction_target = _direction_target(row.get("direction", ""))
        evidence_pathways = (
            sorted(mapped_group.pathway_id.dropna().astype(str).unique().tolist())
            if has_mapping
            else []
        )
        pathways = [formal_pathway[pathway] for pathway in evidence_pathways]
        routes = (
            sorted(mapped_group.mapping_route.dropna().astype(str).unique().tolist())
            if has_mapping
            else []
        )
        rejection_reason = (
            pd.NA if has_mapping else _nullable_text(first.get("rejection_reason", ""))
        )
        raw_event_id = _clean(first.get("raw_event_id", ""))
        source_record_id = _clean(row.get("evidence_event_id", ""), raw_event_id)
        detail_payload = _assay_detail_payload(detail_by_event.get(source_record_id))
        rows.append(
            {
                "experiment_event_id": raw_event_id,
                "source_record_id": source_record_id,
                "source_event_hash": _nullable_text(row.get("independent_event_hash", "")),
                "source_row_index": str(source_index),
                "source_row_sha256": _clean(first.get("source_row_sha256", "")),
                "source_input_sha256": evidence_sha256,
                "cancer_id": _cancer(row.get("cancer_id", "")),
                "cancer_scope": "PAN_CANCER" if _cancer(row.get("cancer_id", "")) == "PAN_CANCER" else "CANCER_SPECIFIC",
                "raw_lncrna_id": _nullable_text(row.get("lncrna_id", "")),
                "lncrna_id": formal_lnc.get(evidence_lnc, _nullable_text(row.get("lncrna_id", ""))),
                "evidence_lncrna_id": evidence_lnc if evidence_lnc else pd.NA,
                "lncrna_id_available": bool(evidence_lnc),
                "raw_partner_id": _nullable_text(row.get("partner_id", "")),
                "partner_id": partner if partner else pd.NA,
                "partner_id_available": bool(partner),
                "assay_family": _clean(row.get("experiment_family", "")),
                **detail_payload,
                "relation_type": _nullable_text(row.get("relation_type", "")),
                "direction_raw": _nullable_text(row.get("direction", "")),
                "direction_target": direction_target if direction_target >= 0 else pd.NA,
                "direction_class": _direction_class(direction_target),
                "direction_available": direction_target >= 0,
                "pmid": _nullable_text(row.get("pmid", "")),
                "pmid_available": bool(_clean(row.get("pmid", ""))),
                "cell_line": _nullable_text(row.get("cell_line", "")),
                "cell_line_available": bool(_clean(row.get("cell_line", ""))),
                "tissue": _nullable_text(row.get("tissue", "")),
                "tissue_available": bool(_clean(row.get("tissue", ""))),
                "manual_review_status": _nullable_text(row.get("manual_review_status", "")),
                "mapping_available": has_mapping,
                "mapping_status": "MAPPED_TO_EXACT_PATHWAY" if has_mapping else "REJECTED_TYPED",
                "mapping_route": ";".join(routes) if routes else pd.NA,
                "exact_pathway_count": len(pathways),
                "exact_pathway_ids": ";".join(pathways) if pathways else pd.NA,
                "rejection_reason": rejection_reason,
                "family_broadcast_used": False,
                "evidence_class": "EXPERIMENTAL_FUNCTIONAL_PERTURBATION",
                "independent_probability_generated": False,
                "confidence_impact_status": PENDING_CONFIDENCE_REASON,
                "release_ready": False,
            }
        )
    value = _normalise_output_types(pd.DataFrame(rows))
    if value.experiment_event_id.duplicated().any():
        raise ExperimentPerturbationContractError(
            "Authoritative evidence_event_id must be unique in the perturbation subset"
        )
    return value.sort_values(["mapping_available", "experiment_event_id"], ascending=[False, True]).reset_index(drop=True)


def _build_provenance_table(
    events: pd.DataFrame,
    lineage: pd.DataFrame,
    rejected: pd.DataFrame,
    formal_lnc: Mapping[str, str],
    formal_pathway: Mapping[str, str],
    evidence_sha256: str,
) -> pd.DataFrame:
    event_columns = [
        "event_id",
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "partner_id",
        "pmid",
        "experiment_type",
        "relation_type",
        "direction_raw",
        "direction_target",
        "tissue",
        "cell_line",
        "source_database",
        "source_dataset",
        "source_record_id",
    ]
    mapped = lineage.merge(
        events[event_columns], on="event_id", how="left", validate="many_to_one"
    )
    mapped_rows: list[dict[str, Any]] = []
    for row in mapped.to_dict("records"):
        direction_target = int(row["direction_target"])
        evidence_lnc = str(row["lncrna_id"])
        mapped_rows.append(
            {
                "provenance_id": row["lineage_id"],
                "event_id": row["event_id"],
                "source_record_id": row["source_record_id"],
                "raw_event_id": row["raw_event_id"],
                "source_kind": row["source_kind"],
                "source_input": row["source_input"],
                "source_input_sha256": evidence_sha256,
                "source_row_index": str(row["source_row_index"]),
                "source_row_sha256": row["source_row_sha256"],
                "cancer_id": row["cancer_id"],
                "lncrna_id": formal_lnc.get(evidence_lnc, evidence_lnc),
                "evidence_lncrna_id": evidence_lnc,
                "partner_id": row["partner_id"],
                "pathway_id": formal_pathway[str(row["pathway_id"])],
                "evidence_pathway_id": row["pathway_id"],
                "assay_family": row["experiment_type"],
                "relation_type": row["relation_type"],
                "direction_raw": _nullable_text(row["direction_raw"]),
                "direction_target": direction_target if direction_target >= 0 else pd.NA,
                "direction_class": _direction_class(direction_target),
                "pmid": _nullable_text(row["pmid"]),
                "cell_line": _nullable_text(row["cell_line"]),
                "tissue": _nullable_text(row["tissue"]),
                "source_database": row["source_database"],
                "source_dataset": row["source_dataset"],
                "mapping_available": True,
                "mapping_route": row["mapping_route"],
                "static_member_id": _nullable_text(row["static_member_id"]),
                "family_id_observed_but_unused": _nullable_text(row["family_id_observed_but_unused"]),
                "family_broadcast_used": bool(row["family_broadcast_used"]),
                "rejection_reason": pd.NA,
            }
        )

    rejected_rows: list[dict[str, Any]] = []
    for row in rejected.to_dict("records"):
        evidence_lnc = _clean(row.get("lncrna_id", ""))
        direction_target = int(row.get("direction_target", -1))
        rejection_reason = _clean(row.get("rejection_reason", ""))
        provenance_id = _stable_id(
            "REJ32",
            row.get("raw_event_id", ""),
            row.get("source_row_sha256", ""),
            rejection_reason,
        )
        rejected_rows.append(
            {
                "provenance_id": provenance_id,
                "event_id": pd.NA,
                "source_record_id": row.get("source_record_id", row.get("raw_event_id", "")),
                "raw_event_id": row.get("raw_event_id", ""),
                "source_kind": row.get("source_kind", "evidence_event"),
                "source_input": row.get("source_input", ""),
                "source_input_sha256": evidence_sha256,
                "source_row_index": str(row.get("source_row_index", "")),
                "source_row_sha256": row.get("source_row_sha256", ""),
                "cancer_id": row.get("cancer_id", "PAN_CANCER"),
                "lncrna_id": formal_lnc.get(evidence_lnc, evidence_lnc or pd.NA),
                "evidence_lncrna_id": evidence_lnc or pd.NA,
                "partner_id": _nullable_text(row.get("partner_id", "")),
                "pathway_id": pd.NA,
                "evidence_pathway_id": pd.NA,
                "assay_family": row.get("experiment_type", "functional_perturbation"),
                "relation_type": _nullable_text(row.get("relation_type", "")),
                "direction_raw": _nullable_text(row.get("direction_raw", "")),
                "direction_target": direction_target if direction_target >= 0 else pd.NA,
                "direction_class": _direction_class(direction_target),
                "pmid": _nullable_text(row.get("pmid", "")),
                "cell_line": _nullable_text(row.get("cell_line", "")),
                "tissue": _nullable_text(row.get("tissue", "")),
                "source_database": row.get("source_database", "evidence_event"),
                "source_dataset": row.get("source_dataset", "evidence_event"),
                "mapping_available": False,
                "mapping_route": pd.NA,
                "static_member_id": pd.NA,
                "family_id_observed_but_unused": _nullable_text(row.get("pathway_family_id", "")),
                "family_broadcast_used": False,
                "rejection_reason": rejection_reason,
            }
        )
    value = _normalise_output_types(pd.DataFrame(mapped_rows + rejected_rows))
    if value.provenance_id.duplicated().any():
        raise ExperimentPerturbationContractError("Provenance identifiers are not unique")
    return value.sort_values(["mapping_available", "provenance_id"], ascending=[False, True]).reset_index(drop=True)


def _build_exact_pathway_table(
    materialised: pd.DataFrame,
    lineage: pd.DataFrame,
    formal_lnc: Mapping[str, str],
    formal_pathway: Mapping[str, str],
    evidence_sha256: str,
    assay_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    first_lineage = (
        lineage.sort_values(["event_id", "lineage_id"])
        .drop_duplicates("event_id")
        [["event_id", "raw_event_id", "source_row_index", "source_row_sha256"]]
        .rename(columns={"event_id": "source_event_id"})
    )
    value = materialised.merge(
        first_lineage, on="source_event_id", how="left", validate="many_to_one"
    )
    detail_by_event = (
        assay_detail.set_index("evidence_event_id").to_dict("index")
        if assay_detail is not None
        else {}
    )
    detail_payloads = [
        _assay_detail_payload(detail_by_event.get(_clean(source_record_id)))
        for source_record_id in value.source_record_id
    ]
    detail_frame = pd.DataFrame(detail_payloads, index=value.index)
    evidence_lnc = value.lncrna_id.astype(str)
    direction_numeric = pd.to_numeric(value.direction_target, errors="coerce")
    result = pd.DataFrame(
        {
            "perturbation_event_id": value.event_id,
            "source_event_id": value.source_event_id,
            "raw_event_id": value.raw_event_id,
            "source_record_id": value.source_record_id,
            "source_row_index": value.source_row_index.map(str),
            "source_row_sha256": value.source_row_sha256,
            "source_input_sha256": evidence_sha256,
            "cancer_id": value.cancer_id,
            "source_cancer_scope": "PAN_CANCER",
            "cancer_materialisation_route": "PAN_CANCER_TO_FORMAL_CANDIDATE_LITERAL_PAIR",
            "lncrna_id": evidence_lnc.map(formal_lnc),
            "evidence_lncrna_id": evidence_lnc,
            "pathway_id": value.pathway_id.astype(str).map(formal_pathway),
            "evidence_pathway_id": value.pathway_id,
            "partner_id": value.partner_id,
            "mapping_route": value.route_type,
            "static_member_id": value.static_member_id,
            "assay_family": value.experiment_type,
            **{column: detail_frame[column] for column in detail_frame.columns},
            "relation_type": value.relation_type,
            "direction_raw": value.direction_raw.map(_nullable_text),
            "direction_target": direction_numeric.where(direction_numeric.ge(0), pd.NA),
            "direction_class": direction_numeric.map(
                lambda item: _direction_class(int(item)) if pd.notna(item) else pd.NA
            ),
            "direction_available": direction_numeric.ge(0),
            "pmid": value.pmid.map(_nullable_text),
            "pmid_available": value.pmid.map(lambda item: bool(_clean(item))),
            "cell_line": value.cell_line.map(_nullable_text),
            "cell_line_available": value.cell_line.map(lambda item: bool(_clean(item)) and _clean(item).lower() != "unknown"),
            "tissue": value.tissue.map(_nullable_text),
            "tissue_available": value.tissue.map(lambda item: bool(_clean(item)) and _clean(item).lower() != "unknown"),
            "source_database": value.source_database,
            "source_dataset": value.source_dataset,
            "source_occurrence_count": value.source_occurrence_count,
            "evidence_fact_available": True,
            "family_broadcast_used": False,
            "independent_probability_generated": False,
            "confidence_impact_status": PENDING_CONFIDENCE_REASON,
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
            "release_ready": False,
        }
    )
    if result.lncrna_id.isna().any():
        raise ExperimentPerturbationContractError(
            "A materialised perturbation event has no formal candidate lncRNA identifier"
        )
    if result.pathway_id.isna().any():
        raise ExperimentPerturbationContractError(
            "A materialised perturbation event has no formal candidate pathway identifier"
        )
    result = _normalise_output_types(result)
    if result.perturbation_event_id.duplicated().any():
        raise ExperimentPerturbationContractError(
            "Materialised perturbation_event_id must be unique"
        )
    return result.sort_values(
        ["cancer_id", "lncrna_id", "pathway_id", "perturbation_event_id"]
    ).reset_index(drop=True)


def _validate_no_predictive_payload(frame: pd.DataFrame, *, role: str) -> None:
    declarative_guard_columns = {
        "independent_probability_generated",
        "changes_primary_ranking",
        "changes_discovery_ranking",
    }
    forbidden = [
        str(column)
        for column in frame.columns
        if any(
            token in _normalised_token(column)
            for token in ("probability", "score", "logit", "ranking", "predictedconfidence")
        )
        and column not in declarative_guard_columns
    ]
    if forbidden:
        raise ExperimentPerturbationContractError(
            f"{role} unexpectedly contains predictive payload columns: {forbidden}"
        )


def materialise_experiment_perturbation(
    *,
    evidence_event_path: Path,
    candidate_universe_path: Path,
    pathway_members_path: Path,
    output_dir: Path,
    expected_hashes: Mapping[str, str] | None = None,
    assay_detail_path: Path | None = None,
    formal: bool = True,
    runner_path: Path | None = None,
) -> ExperimentPerturbationMaterialisation:
    """Materialise current raw perturbation facts into exact V3.2 contexts.

    The function performs no fitting and emits no probability.  ``formal``
    pins both the three authoritative SHA-256 values and their expected row
    counts.  Tests may provide fixture hashes with ``formal=False``; hash
    binding remains mandatory in either mode.
    """

    evidence_path = _assert_regular_local_input(evidence_event_path, role="evidence_event")
    candidate_path = _assert_regular_local_input(candidate_universe_path, role="candidate_universe")
    members_path = _assert_regular_local_input(pathway_members_path, role="pathway_members")
    detail_path = (
        _assert_regular_local_input(assay_detail_path, role="assay_detail")
        if assay_detail_path is not None
        else None
    )
    resolved_output = Path(output_dir).resolve()
    if resolved_output.exists():
        raise ExperimentPerturbationContractError(
            f"Output directory already exists; refusing to mix lineages: {resolved_output}"
        )

    default_expected = {
        "evidence_event": FORMAL_EVIDENCE_EVENT_SHA256,
        "candidate_universe": FORMAL_CANDIDATE_UNIVERSE_SHA256,
        "pathway_members": FORMAL_PATHWAY_MEMBERS_SHA256,
    }
    if detail_path is not None:
        default_expected["assay_detail"] = FORMAL_ASSAY_DETAIL_SHA256
    expected = dict(expected_hashes or default_expected)
    required_roles = {"evidence_event", "candidate_universe", "pathway_members"}
    if detail_path is not None:
        required_roles.add("assay_detail")
    if set(expected) != required_roles:
        raise ExperimentPerturbationContractError(
            f"expected_hashes must bind exactly {sorted(required_roles)}"
        )
    actual_hashes = {
        "evidence_event": file_sha256(evidence_path),
        "candidate_universe": file_sha256(candidate_path),
        "pathway_members": file_sha256(members_path),
    }
    if detail_path is not None:
        actual_hashes["assay_detail"] = file_sha256(detail_path)
    mismatches = {
        role: {"expected": expected[role], "actual": actual_hashes[role]}
        for role in expected
        if expected[role] != actual_hashes[role]
    }
    if mismatches:
        raise ExperimentPerturbationContractError(
            f"Authoritative input SHA-256 mismatch: {mismatches}"
        )

    evidence = pd.read_parquet(evidence_path)
    candidates = pd.read_parquet(
        candidate_path, columns=["cancer_id", "lncrna_id", "pathway_id"]
    )
    members = pd.read_parquet(members_path)
    for role, frame in (
        ("evidence_event", evidence),
        ("candidate_universe", candidates),
        ("pathway_members", members),
    ):
        _assert_no_derived_columns(frame, role=role)

    if "experiment_family" not in evidence.columns:
        raise ExperimentPerturbationContractError(
            "evidence_event lacks the required experiment_family assay field"
        )
    selection = evidence.experiment_family.map(_is_functional_perturbation)
    selected = evidence.loc[selection].copy()
    if selected.empty:
        raise ExperimentPerturbationContractError(
            "No functional/perturbation assay records were found"
        )
    assay_detail = None
    if detail_path is not None:
        assay_detail = _validate_assay_detail(
            pd.read_parquet(detail_path),
            selected_event_ids=selected["evidence_event_id"],
        )
    if formal:
        actual_rows = {
            "evidence_event": len(evidence),
            "candidate_universe": len(candidates),
            "pathway_members": len(members),
            "selected_functional_perturbation": len(selected),
        }
        if actual_rows != FORMAL_INPUT_ROWS:
            raise ExperimentPerturbationContractError(
                f"Formal row-count contract mismatch: expected={FORMAL_INPUT_ROWS}, actual={actual_rows}"
            )
        if assay_detail is not None and len(assay_detail) != FORMAL_ASSAY_DETAIL_ROWS:
            raise ExperimentPerturbationContractError(
                "Formal assay-detail row-count contract mismatch: "
                f"expected={FORMAL_ASSAY_DETAIL_ROWS}, actual={len(assay_detail)}"
            )

    # This independent layer accepts only literal partner -> exact-member
    # mappings.  Direct pathway assertions are intentionally blanked even if an
    # unexpected input column is present.  A family column is retained solely
    # so the shared builder can emit the explicit no-broadcast rejection.
    direct_exact_assertions_ignored = 0
    for column in ("pathway_id", "exact_pathway_id"):
        if column in selected.columns:
            direct_exact_assertions_ignored += int(selected[column].map(lambda item: bool(_clean(item))).sum())
            selected[column] = ""

    empty_interactions = selected.iloc[0:0].copy()
    build = build_exact_event_bags(
        selected,
        empty_interactions,
        members,
        candidate_universe=candidates,
        evidence_input_name=str(evidence_path),
        interaction_input_name="EMPTY_BY_CONTRACT_NO_INTERACTION_RELATION",
    )
    if not build.events.empty and set(build.events.route_type.astype(str)) != {"PARTNER_EXACT_MEMBER"}:
        raise ExperimentPerturbationContractError(
            "Perturbation facts must use PARTNER_EXACT_MEMBER literal mapping only"
        )
    if not build.lineage.empty and build.lineage.family_broadcast_used.astype(bool).any():
        raise ExperimentPerturbationContractError("Pathway-family broadcast was detected")
    if not build.events.empty and build.events.is_model_prediction.astype(bool).any():
        raise ExperimentPerturbationContractError("A model prediction entered the fact layer")

    formal_lnc = _formal_lnc_map(candidates)
    formal_pathway = _formal_pathway_map(candidates)
    source_events = _build_source_event_table(
        selected,
        build.events,
        build.lineage,
        build.rejected,
        formal_lnc,
        formal_pathway,
        actual_hashes["evidence_event"],
        assay_detail,
    )
    provenance = _build_provenance_table(
        build.events,
        build.lineage,
        build.rejected,
        formal_lnc,
        formal_pathway,
        actual_hashes["evidence_event"],
    )
    materialised = materialize_candidate_events(build.events, candidates)
    exact_pathway = _build_exact_pathway_table(
        materialised,
        build.lineage,
        formal_lnc,
        formal_pathway,
        actual_hashes["evidence_event"],
        assay_detail,
    )
    rejected_output = source_events.loc[~source_events.mapping_available].copy().reset_index(drop=True)

    for role, frame in (
        ("v32_experiment_events", source_events),
        ("v32_perturbation_exact_pathway", exact_pathway),
        ("v32_experiment_provenance", provenance),
        ("v32_experiment_rejected", rejected_output),
    ):
        _validate_no_predictive_payload(frame, role=role)
    if source_events.family_broadcast_used.any() or exact_pathway.family_broadcast_used.any() or provenance.family_broadcast_used.any():
        raise ExperimentPerturbationContractError("Family broadcast invariant failed")
    if not source_events.cancer_id.eq("PAN_CANCER").all():
        raise ExperimentPerturbationContractError(
            "The formal raw evidence subset must state its PAN_CANCER scope explicitly"
        )

    candidate_keys = candidates.drop_duplicates(["cancer_id", "lncrna_id", "pathway_id"])
    exact_key_audit = exact_pathway[["cancer_id", "lncrna_id", "pathway_id"]].drop_duplicates()
    unmatched = exact_key_audit.merge(
        candidate_keys,
        on=["cancer_id", "lncrna_id", "pathway_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unmatched_count = int(unmatched._merge.ne("both").sum())
    if unmatched_count:
        raise ExperimentPerturbationContractError(
            f"{unmatched_count} exact perturbation keys are outside the formal candidate universe"
        )

    resolved_output.mkdir(parents=True, exist_ok=False)
    artifact_frames = {
        "v32_experiment_events": source_events,
        "v32_perturbation_exact_pathway": exact_pathway,
        "v32_experiment_provenance": provenance,
        "v32_experiment_rejected": rejected_output,
    }
    artifact_paths: dict[str, Path] = {}
    artifact_metadata: dict[str, dict[str, Any]] = {}
    for artifact_id, frame in artifact_frames.items():
        path = resolved_output / ARTIFACT_FILENAMES[artifact_id]
        _parquet_write(path, frame)
        artifact_paths[artifact_id] = path
        artifact_metadata[artifact_id] = _artifact_metadata(path, frame)

    module_path = Path(__file__).resolve()
    code_files = {
        "experiment_perturbation_module": {
            "path": str(module_path),
            "sha256": file_sha256(module_path),
        }
    }
    if runner_path is not None:
        resolved_runner = _assert_regular_local_input(runner_path, role="runner")
        code_files["runner"] = {
            "path": str(resolved_runner),
            "sha256": file_sha256(resolved_runner),
        }

    mapped_source_rows = int(source_events.mapping_available.sum())
    rejected_source_rows = int((~source_events.mapping_available).sum())
    rejection_counts = {
        str(key): int(value)
        for key, value in rejected_output.rejection_reason.value_counts(dropna=False).items()
    }
    input_artifacts = {
        "evidence_event": _input_metadata(
            evidence_path, evidence, actual_hashes["evidence_event"], "raw_authoritative_event_facts"
        ),
        "candidate_universe": _input_metadata(
            candidate_path, candidates, actual_hashes["candidate_universe"], "current_v32_formal_exact_candidate_universe"
        ),
        "pathway_members": _input_metadata(
            members_path, members, actual_hashes["pathway_members"], "current_evidence_lineage_exact_membership"
        ),
    }
    if detail_path is not None and assay_detail is not None:
        input_artifacts["assay_detail"] = _input_metadata(
            detail_path,
            assay_detail,
            actual_hashes["assay_detail"],
            "additive_hash_bound_assay_detail_sidecar",
        )
    manifest: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "materialisation_format": MATERIALISATION_FORMAT,
        "status": "MATERIALISATION_SUCCESS",
        "formal": bool(formal),
        "local_filesystem_only": True,
        "remote_reads_used": False,
        "remote_writes_used": False,
        "deployment_performed": False,
        "release_ready": False,
        "release_blockers": [PENDING_CONFIDENCE_REASON],
        "confidence_impact_policy": {
            "endpoint": "confidence_only",
            "status": PENDING_CONFIDENCE_REASON,
            "determined_by": "CURRENT_FRESH_V32_EVIDENCE_TRANSFORMER_BINDING_AND_ABLATION",
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
            "independent_probability_generated": False,
        },
        "input_artifacts": input_artifacts,
        "input_sha256_contract": {key: expected[key] for key in sorted(expected)},
        "selection_contract": {
            "assay_field": "experiment_family",
            "accepted_class": "functional/perturbation_only",
            "input_rows": int(len(evidence)),
            "selected_rows": int(len(selected)),
            "excluded_rows": int(len(evidence) - len(selected)),
            "assay_detail_sidecar_supplied": assay_detail is not None,
            "assay_detail_sidecar_rows": int(len(assay_detail)) if assay_detail is not None else 0,
        },
        "assay_detail_contract": {
            "sidecar_supplied": assay_detail is not None,
            "authoritative_evidence_overwritten": False,
            "one_row_per_selected_event": assay_detail is not None,
            "assay_detail_available_rows": (
                int(assay_detail.assay_detail_available.sum()) if assay_detail is not None else 0
            ),
            "manual_review_required_rows": (
                int(assay_detail.manual_review_required.sum()) if assay_detail is not None else 0
            ),
            "match_route_counts": (
                {
                    str(key): int(value)
                    for key, value in assay_detail.match_route.value_counts(dropna=False).items()
                }
                if assay_detail is not None
                else {}
            ),
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
        },
        "mapping_contract": {
            "route": "partner_id_to_literal_exact_pathway_member",
            "allowed_route_types": ["PARTNER_EXACT_MEMBER"],
            "pan_cancer_explicit": True,
            "pan_cancer_materialised_only_through_formal_candidate_pairs": True,
            "direct_exact_assertions_accepted": False,
            "direct_exact_assertions_ignored": direct_exact_assertions_ignored,
            "pathway_family_broadcast_used": False,
            "pathway_family_member_table_read": False,
            "identifier_mapping": "literal_ensembl_normalisation_only_no_learned_crosswalk",
        },
        "artifacts": artifact_metadata,
        "row_audit": {
            "selected_source_rows": int(len(selected)),
            "mapped_source_rows": mapped_source_rows,
            "rejected_source_rows": rejected_source_rows,
            "source_row_accounting_pass": mapped_source_rows + rejected_source_rows == len(selected),
            "canonical_pan_cancer_exact_event_rows": int(len(build.events)),
            "raw_lineage_rows": int(len(build.lineage)),
            "raw_rejection_rows": int(len(build.rejected)),
            "materialised_exact_event_rows": int(len(exact_pathway)),
            "materialised_unique_exact_keys": int(
                exact_pathway[["cancer_id", "lncrna_id", "pathway_id"]].drop_duplicates().shape[0]
            ),
            "materialised_cancers": int(exact_pathway.cancer_id.nunique()),
            "materialised_lncrnas": int(exact_pathway.lncrna_id.nunique()),
            "materialised_pathways": int(exact_pathway.pathway_id.nunique()),
            "candidate_key_mismatches": unmatched_count,
            "rejection_reason_counts": rejection_counts,
        },
        "lineage_policy": {
            "old_predictions_used": False,
            "old_rankings_used": False,
            "old_checkpoints_used": False,
            "old_derived_evidence_used": False,
            "training_performed": False,
            "probability_or_score_generated": False,
            "shared_builder": "cc_hhgt.v32.evidence_training.build_exact_event_bags",
            "raw_source_row_lineage_preserved": True,
        },
        "execution_code": code_files,
    }
    manifest_path = resolved_output / MANIFEST_FILENAME
    _json_write(manifest_path, manifest)
    manifest_sha256 = file_sha256(manifest_path)
    binding: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "binding_format": BINDING_FORMAT,
        "status": "MATERIALISATION_SUCCESS_RELEASE_PENDING",
        "release_ready": False,
        "release_blockers": [PENDING_CONFIDENCE_REASON],
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "artifacts": artifact_metadata,
        "input_sha256": actual_hashes,
        "execution_code": code_files,
        "invariants": {
            "functional_perturbation_only": True,
            "partner_literal_exact_member_mapping_only": True,
            "family_broadcast_used": False,
            "pan_cancer_explicit": True,
            "old_results_used": False,
            "independent_probability_generated": False,
            "changes_primary_ranking": False,
        },
    }
    binding_path = resolved_output / BINDING_FILENAME
    _json_write(binding_path, binding)
    return ExperimentPerturbationMaterialisation(
        output_dir=resolved_output,
        artifact_paths=artifact_paths,
        manifest_path=manifest_path,
        binding_path=binding_path,
        manifest=manifest,
        binding=binding,
    )


def validate_experiment_perturbation_binding(output_dir: Path) -> Mapping[str, Any]:
    """Independently re-hash a completed local materialisation binding."""

    root = Path(output_dir).resolve()
    binding_path = root / BINDING_FILENAME
    if not binding_path.is_file():
        raise ExperimentPerturbationContractError(f"Missing binding: {binding_path}")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("binding_format") != BINDING_FORMAT:
        raise ExperimentPerturbationContractError("Unexpected perturbation binding format")
    if binding.get("release_ready") is not False:
        raise ExperimentPerturbationContractError(
            "A fact-only materialisation cannot be release-ready before Evidence final binding"
        )
    manifest_info = binding.get("manifest", {})
    manifest_path = Path(manifest_info.get("path", "")).resolve()
    if not manifest_path.is_file() or file_sha256(manifest_path) != manifest_info.get("sha256"):
        raise ExperimentPerturbationContractError("Manifest hash binding failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_ready") is not False:
        raise ExperimentPerturbationContractError("Manifest release_ready must remain false")
    for role, info in binding.get("input_sha256", {}).items():
        input_path = Path(manifest["input_artifacts"][role]["path"])
        if not input_path.is_file() or file_sha256(input_path) != info:
            raise ExperimentPerturbationContractError(f"Input hash binding failed for {role}")
    for artifact_id, info in binding.get("artifacts", {}).items():
        if artifact_id not in ARTIFACT_FILENAMES:
            raise ExperimentPerturbationContractError(f"Unknown bound artifact: {artifact_id}")
        path = Path(info["path"])
        if not path.is_file() or file_sha256(path) != info["sha256"]:
            raise ExperimentPerturbationContractError(f"Artifact hash binding failed for {artifact_id}")
        frame = pd.read_parquet(path)
        if len(frame) != int(info["rows"]):
            raise ExperimentPerturbationContractError(f"Artifact row binding failed for {artifact_id}")
        _validate_no_predictive_payload(frame, role=artifact_id)
    for code_id, info in binding.get("execution_code", {}).items():
        path = Path(info["path"])
        if not path.is_file() or file_sha256(path) != info["sha256"]:
            raise ExperimentPerturbationContractError(f"Execution-code hash failed for {code_id}")
    if binding.get("invariants", {}).get("family_broadcast_used") is not False:
        raise ExperimentPerturbationContractError("Family-broadcast invariant failed")
    return binding
