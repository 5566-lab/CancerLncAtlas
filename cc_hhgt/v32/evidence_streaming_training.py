"""Bounded-memory staging for fresh V3.2 Evidence-head training.

The historical :func:`evidence_training.build_exact_event_bags` is the semantic
reference, but it requires all raw interactions and all derived event rows in
Python memory.  This module keeps that exact-event contract while moving the
large joins, de-duplication, candidate attachment and per-fold provenance
filtering to DuckDB/Parquet.  Python only holds one Arrow batch, one event bag,
or the five fold-size counters at a time.

This is deliberately a *staging* module.  It cannot initialise an optimiser,
load a checkpoint, train a head, publish a prediction, or touch a web service.
Its passing receipt says ``formal_training_started=false``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .evidence_training import (
    BagExample,
    CoreFeatureBundle,
    EVALUATION_PROVENANCE_EXCLUSION_POLICY,
    EVIDENCE_SPLIT_POLICY,
    EVENT_FEATURE_FIELDS,
    EXACT_KEYS,
    FORMAL_CANDIDATE_ROWS,
    FORMAL_CANDIDATE_SHA256,
    N_FOLDS,
    UNKNOWN,
    EvidenceTrainingContractError,
    _clean,
    _column,
    _cancer,
    _event_feature_matrix,
    _identity,
    _normalize_source_rows,
    _normalized_name,
    _stable_sha256,
    assert_label_blind_raw_input,
    file_sha256,
    normalize_candidates,
    normalize_exact_pathway_members,
)


STREAMING_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_STAGE_CONTEXT_R6"
CONVERSION_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_RECOVERED_PARQUET_CONVERSION_V1"
CONVERSION_PASS = "PASS_LOSSLESS_PARQUET_CONVERSION"
PATIENT_RECEIPT_FORMAT = "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_RECEIPT_V1"
STAGING_STATUS = "READY_FOR_FRESH_V32_EVIDENCE_TRAINING_NOT_STARTED"
NO_EVENT_REASON = "NO_EXACT_EVIDENCE_EVENT"
FORMAL_RECOVERED_INTERACTION_ROWS = 6_160_707

EMPTY_EVENT_COLUMNS = (
    "evidence_event_id", "lncrna_id", "partner_id", "pathway_id",
    "pathway_family_id", "cancer_id", "source_database", "source_dataset",
    "source_record_id", "pmid", "experiment_family", "relation_type",
    "direction", "is_experimental", "is_predicted", "source_row_sha256",
)

FIREWALL_COLUMN_TOKENS = {
    "label", "primarylabel", "primaryscore", "pairscore", "pairevidencelabel",
    "pairevidencescore", "neuralprobability", "crosscancerprobability",
    "survivaltime", "survivalevent", "ostime", "osevent", "pfstime",
    "pfsevent", "sealedtestlabel", "testlabel", "heldoutlabel",
}


class StreamingEvidenceStageError(EvidenceTrainingContractError):
    """Raised when bounded Evidence staging cannot prove its input contract."""


@dataclass(frozen=True)
class StreamingEventBag:
    """One exact cancer/lncRNA/pathway event bag read from sorted Parquet."""

    key: tuple[str, str, str]
    events: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class StreamingEventBagBatch:
    """A bounded group of complete bags; no bag crosses returned batches."""

    bags: tuple[StreamingEventBag, ...]
    event_rows: int


@dataclass(frozen=True)
class StreamingTrainerBatch:
    """A bounded set of legacy-equivalent BagExample objects for one fold."""

    examples: tuple[BagExample, ...]
    missing_core: tuple[Mapping[str, Any], ...]


def _require_runtime() -> tuple[Any, Any, Any]:
    try:
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by deployment gate
        raise StreamingEvidenceStageError(
            "duckdb and pyarrow are required for bounded Evidence staging"
        ) from exc
    return duckdb, pa, pq


def _quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha_contract(
    paths: Mapping[str, Path], expected_hashes: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    if set(paths) != set(expected_hashes):
        raise StreamingEvidenceStageError(
            "Expected-hash roles must exactly match input path roles"
        )
    result: dict[str, dict[str, Any]] = {}
    for role, raw_path in paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file() or path.is_symlink():
            raise StreamingEvidenceStageError(f"Missing or unsafe {role}: {path}")
        observed = file_sha256(path)
        expected = str(expected_hashes[role]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or observed != expected:
            raise StreamingEvidenceStageError(
                f"{role} SHA256 mismatch: {observed} != {expected}"
            )
        result[role] = {
            "path": str(path), "sha256": observed, "bytes": path.stat().st_size
        }
    return result


def _validate_conversion_receipt(
    path: Path,
    *,
    relation_path: Path,
    event_path: Path,
    relation_sha256: str,
    event_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != CONVERSION_FORMAT or payload.get("status") != CONVERSION_PASS:
        raise StreamingEvidenceStageError("Recovered-Parquet conversion receipt is not passing")
    if not payload.get("contract", {}).get("independent_validation_required_and_bound"):
        raise StreamingEvidenceStageError("Conversion receipt does not bind independent validation")
    for role, actual_path, actual_sha in (
        ("interaction_relation", relation_path, relation_sha256),
        ("evidence_event", event_path, event_sha256),
    ):
        declared = payload.get("outputs", {}).get(role, {})
        if Path(str(declared.get("path", ""))).resolve() != actual_path.resolve():
            raise StreamingEvidenceStageError(f"Conversion receipt binds another {role} path")
        if declared.get("sha256") != actual_sha:
            raise StreamingEvidenceStageError(f"Conversion receipt binds another {role} hash")
    return payload


def _validate_empty_event_authority(path: Path) -> None:
    _, _, pq = _require_runtime()
    parquet = pq.ParquetFile(path)
    if tuple(parquet.schema_arrow.names) != EMPTY_EVENT_COLUMNS:
        raise StreamingEvidenceStageError("Empty Evidence authority schema drifted")
    if parquet.metadata.num_rows != 0:
        raise StreamingEvidenceStageError("Evidence event authority must be header-only/zero-row")


def _validate_patient_authority(
    authority_path: Path,
    receipt_path: Path,
    *,
    authority_sha256: str,
    strict_formal: bool,
) -> dict[str, Any]:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("format") != PATIENT_RECEIPT_FORMAT or int(receipt.get("n_folds", 0)) != N_FOLDS:
        raise StreamingEvidenceStageError("Patient-first authority receipt is not admissible")
    matched = False
    for artifact in receipt.get("artifacts", {}).values():
        if (
            artifact.get("filename") == authority_path.name
            and artifact.get("sha256") == authority_sha256
        ):
            matched = True
    if not matched:
        raise StreamingEvidenceStageError("Patient receipt does not bind the supplied authority")
    gates = receipt.get("gates", {})
    required_gates = (
        "all_five_folds_per_cancer", "patient_cross_cancer_count_zero",
        "patient_cross_fold_count_zero", "output_reuse_forbidden",
    )
    if not all(gates.get(name) is True for name in required_gates):
        raise StreamingEvidenceStageError("Patient authority receipt has a failed firewall gate")
    frame = pd.read_csv(authority_path, sep="\t", dtype="string")
    needed = {"cancer_id", "patient_id", "patient_fold_id"}
    if not needed.issubset(frame.columns):
        raise StreamingEvidenceStageError("Patient authority lacks patient-first fold columns")
    fold = pd.to_numeric(frame.patient_fold_id, errors="raise").astype(int)
    if set(fold) != set(range(N_FOLDS)):
        raise StreamingEvidenceStageError("Patient authority does not populate all five folds")
    if frame.assign(_fold=fold).groupby("patient_id")._fold.nunique().max() != 1:
        raise StreamingEvidenceStageError("A patient crosses patient folds")
    if frame.groupby("patient_id").cancer_id.nunique().max() != 1:
        raise StreamingEvidenceStageError("A patient crosses cancers")
    if strict_formal and (
        receipt.get("observed", {}).get("cancers") != 33
        or receipt.get("observed", {}).get("patients") != 10_432
    ):
        raise StreamingEvidenceStageError("Formal patient authority cardinality drifted")
    return receipt


def _assert_schema_firewall(frame: pd.DataFrame, role: str) -> None:
    bad = sorted(
        str(column)
        for column in frame.columns
        if _normalized_name(column) in FIREWALL_COLUMN_TOKENS
    )
    if bad:
        raise StreamingEvidenceStageError(
            f"{role} crossed the evaluation/test firewall: {bad}"
        )


def _iter_parquet_batches(path: Path, *, batch_size: int) -> Iterator[pd.DataFrame]:
    _, _, pq = _require_runtime()
    if batch_size <= 0:
        raise StreamingEvidenceStageError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def _mapping_state(
    identifier: object,
    route: object,
    count: object,
    status: object,
    *,
    entity: str,
) -> str:
    route_text = _clean(route).lower()
    status_text = _clean(status).lower()
    try:
        candidates = int(float(_clean(count, "0")))
    except ValueError:
        candidates = 0
    if "ambiguous" in route_text or candidates > 1:
        return "AMBIGUOUS"
    if not _clean(identifier) or re.search(
        r"unmapped|unresolved|invalid|rejected", route_text
    ):
        return "UNMAPPED"
    # The rematerialized table also carries a combined status.  Consult only
    # the entity-specific clause so an ambiguous partner cannot relabel a
    # uniquely mapped lncRNA (and vice versa).
    if entity == "lncrna" and re.search(r"(?:^|_)ambiguous_lncrna", status_text):
        return "AMBIGUOUS"
    if entity == "partner" and "partner_ambiguous" in status_text:
        return "AMBIGUOUS"
    if entity == "lncrna" and re.search(r"(?:^|_)unresolved_lncrna", status_text):
        return "UNMAPPED"
    if entity == "partner" and "partner_unresolved" in status_text:
        return "UNMAPPED"
    return "MAPPED_UNIQUE"


def _normalized_raw_schema(pa: Any) -> Any:
    strings = (
        "source_kind", "source_input", "source_row_sha256", "source_sha256",
        "raw_event_id", "source_record_id", "cancer_id", "lncrna_id",
        "partner_id", "pathway_id", "pathway_family_id", "source_database",
        "source_dataset", "pmid", "doi", "experiment_id", "biosample",
        "intervention", "readout", "mediator_id", "experiment_type",
        "experiment_raw", "experiment_family", "assay_subtype",
        "graph_assay_class", "relation_type", "direction_raw", "tissue",
        "cell_line", "species", "evidence_tier", "manual_review_status",
        "lncrna_mapping_state", "partner_mapping_state", "input_mapping_status",
    )
    fields = [pa.field(name, pa.string(), nullable=False) for name in strings]
    fields.extend(
        [
            pa.field("source_row_index", pa.int64(), nullable=False),
            pa.field("direction_target", pa.int8(), nullable=False),
            pa.field("confidence_target", pa.float64(), nullable=True),
            pa.field("is_experimental", pa.bool_(), nullable=False),
            pa.field("is_computational", pa.bool_(), nullable=False),
            pa.field("is_model_prediction", pa.bool_(), nullable=False),
            pa.field("is_physical", pa.bool_(), nullable=False),
            pa.field("family_only", pa.bool_(), nullable=False),
        ]
    )
    return pa.schema(fields)


def _stage_normalized_raw(
    *,
    sources: Sequence[tuple[str, Path]],
    output_path: Path,
    identifier_map: Mapping[str, str],
    batch_size: int,
) -> dict[str, int]:
    _, pa, pq = _require_runtime()
    schema = _normalized_raw_schema(pa)
    writer = pq.ParquetWriter(output_path, schema, compression="zstd", use_dictionary=True)
    counts: dict[str, int] = {}
    try:
        for source_kind, path in sources:
            offset = 0
            for frame in _iter_parquet_batches(path, batch_size=batch_size):
                frame.index = np.arange(offset, offset + len(frame), dtype=np.int64)
                assert_label_blind_raw_input(frame, str(path))
                columns = {_normalized_name(c): str(c) for c in frame.columns}
                status_col = columns.get("mappingstatus")
                lnc_route_col = columns.get("lncrnamappingroute")
                lnc_count_col = columns.get("lncrnamappingcandidatecount")
                partner_route_col = columns.get("partnermappingroute")
                partner_count_col = columns.get("partnermappingcandidatecount")
                records: list[dict[str, Any]] = []
                for raw, (_, source_row) in zip(
                    _normalize_source_rows(
                        frame, source_kind=source_kind, input_name=str(path),
                        id_map=identifier_map,
                    ),
                    frame.iterrows(),
                ):
                    status = _clean(source_row[status_col]) if status_col else ""
                    raw.pop("pmid_tokens", None)
                    raw["source_row_index"] = int(raw["source_row_index"])
                    raw["lncrna_mapping_state"] = _mapping_state(
                        raw["lncrna_id"],
                        source_row[lnc_route_col] if lnc_route_col else "",
                        source_row[lnc_count_col] if lnc_count_col else "",
                        status, entity="lncrna",
                    )
                    raw["partner_mapping_state"] = _mapping_state(
                        raw["partner_id"],
                        source_row[partner_route_col] if partner_route_col else "",
                        source_row[partner_count_col] if partner_count_col else "",
                        status, entity="partner",
                    )
                    raw["input_mapping_status"] = status
                    for field in schema:
                        if pa.types.is_string(field.type):
                            raw[field.name] = _clean(raw.get(field.name))
                    records.append(raw)
                if records:
                    writer.write_table(pa.Table.from_pylist(records, schema=schema))
                offset += len(frame)
            counts[source_kind] = counts.get(source_kind, 0) + offset
    finally:
        writer.close()
    return counts


def _stage_members(
    path: Path,
    output_path: Path,
    *,
    identifier_map: Mapping[str, str],
    batch_size: int,
) -> int:
    _, pa, pq = _require_runtime()
    schema = pa.schema(
        [
            pa.field("pathway_id", pa.string(), nullable=False),
            pa.field("member_id", pa.string(), nullable=False),
            pa.field("member_type", pa.string(), nullable=False),
            pa.field("mapping_status", pa.string(), nullable=False),
            pa.field("static_member_id", pa.string(), nullable=False),
        ]
    )
    writer = pq.ParquetWriter(output_path, schema, compression="zstd", use_dictionary=True)
    rows = 0
    try:
        for frame in _iter_parquet_batches(path, batch_size=batch_size):
            _assert_schema_firewall(frame, "exact pathway membership")
            normalized = normalize_exact_pathway_members(frame, identifier_map)
            if not normalized.empty:
                normalized = normalized[list(schema.names)]
                writer.write_table(
                    pa.Table.from_pandas(
                        normalized, schema=schema, preserve_index=False, safe=True
                    )
                )
                rows += len(normalized)
    finally:
        writer.close()
    return rows


def _stage_candidates(path: Path, output_path: Path, *, batch_size: int) -> int:
    _, pa, pq = _require_runtime()
    schema = pa.schema(
        [
            pa.field("cancer_id", pa.string(), nullable=False),
            pa.field("lncrna_id", pa.string(), nullable=False),
            pa.field("pathway_id", pa.string(), nullable=False),
            pa.field("candidate_order", pa.int64(), nullable=False),
        ]
    )
    writer = pq.ParquetWriter(output_path, schema, compression="zstd", use_dictionary=True)
    offset = 0
    try:
        for frame in _iter_parquet_batches(path, batch_size=batch_size):
            _assert_schema_firewall(frame, "formal candidate universe")
            normalized = normalize_candidates(frame)
            normalized["candidate_order"] = np.arange(
                offset, offset + len(normalized), dtype=np.int64
            )
            writer.write_table(
                pa.Table.from_pandas(
                    normalized, schema=schema, preserve_index=False, safe=True
                )
            )
            offset += len(normalized)
    finally:
        writer.close()
    return offset


def _stable_sql(*expressions: str) -> str:
    cleaned = [f"coalesce(cast({value} AS VARCHAR), '')" for value in expressions]
    return "sha256(" + (" || chr(31) || ".join(cleaned) if cleaned else "''") + ")"


def _copy_query(connection: Any, query: str, path: Path, *, row_group_size: int) -> None:
    connection.execute(
        f"COPY ({query}) TO {_quote(path)} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_size)})"
    )


def _materialize_exact_tables(
    connection: Any,
    *,
    normalized_raw_path: Path,
    member_stage_path: Path,
    candidate_stage_path: Path,
    output_root: Path,
    row_group_size: int,
) -> dict[str, int]:
    """Perform all potentially large joins/de-duplication inside DuckDB."""

    connection.execute(
        f"CREATE VIEW raw_events AS SELECT * FROM read_parquet({_quote(normalized_raw_path)})"
    )
    connection.execute(
        "CREATE VIEW members AS SELECT pathway_id, member_id, "
        "min(member_type) AS member_type, min(static_member_id) AS static_member_id "
        f"FROM read_parquet({_quote(member_stage_path)}) "
        "GROUP BY pathway_id, member_id"
    )
    connection.execute(
        f"CREATE VIEW candidates AS SELECT * FROM read_parquet({_quote(candidate_stage_path)})"
    )
    candidate_metrics = connection.execute(
        "SELECT count(*), count(DISTINCT cancer_id), "
        "count(DISTINCT (cancer_id, lncrna_id, pathway_id)), "
        "sum(CASE WHEN cancer_id='' OR lncrna_id='' OR pathway_id='' THEN 1 ELSE 0 END) "
        "FROM candidates"
    ).fetchone()
    if int(candidate_metrics[0]) != int(candidate_metrics[2]) or int(candidate_metrics[3]) != 0:
        raise StreamingEvidenceStageError("Candidate universe is empty-keyed or non-unique")
    member_metrics = connection.execute(
        "SELECT count(*), count(DISTINCT (pathway_id, member_id)) FROM members"
    ).fetchone()
    if int(member_metrics[0]) != int(member_metrics[1]):
        raise StreamingEvidenceStageError("Static exact membership is not unique")

    # A direct exact assertion wins over a partner-derived route to the same
    # pathway, exactly matching routes.setdefault() in build_exact_event_bags.
    connection.execute(
        "CREATE VIEW unfiltered_routes AS "
        "SELECT source_kind, source_row_index, pathway_id, route_type, "
        "static_member_id, member_type FROM ("
        " SELECT source_kind, source_row_index, pathway_id, "
        " 'DIRECT_EXACT_ASSERTION' AS route_type, '' AS static_member_id, "
        f" '{UNKNOWN}' AS member_type FROM raw_events WHERE pathway_id <> '' "
        " UNION ALL "
        " SELECT r.source_kind, r.source_row_index, m.pathway_id, "
        " 'PARTNER_EXACT_MEMBER' AS route_type, m.static_member_id, m.member_type "
        " FROM raw_events r JOIN members m ON r.partner_id=m.member_id "
        ") q QUALIFY row_number() OVER ("
        "PARTITION BY source_kind, source_row_index, pathway_id ORDER BY route_type"
        ")=1"
    )
    connection.execute(
        "CREATE VIEW eligible_routes AS SELECT r.*, u.pathway_id AS route_pathway_id, "
        "u.route_type, u.static_member_id, u.member_type FROM raw_events r "
        "JOIN unfiltered_routes u USING(source_kind, source_row_index) "
        "WHERE r.lncrna_id <> '' "
        "AND r.cancer_id NOT IN ('CONTEXT_UNRESOLVED','CONTEXT_UNSPECIFIED') "
        "AND NOT (regexp_matches(r.relation_type, '(?i)ceRNA|sponge') "
        "AND r.pathway_id='') "
        "AND EXISTS (SELECT 1 FROM candidates c WHERE "
        "c.lncrna_id=r.lncrna_id AND c.pathway_id=u.pathway_id AND "
        "(r.cancer_id='PAN_CANCER' OR c.cancer_id=r.cancer_id))"
    )

    physical_signature = _stable_sql(
        "source_database", "source_dataset", "source_record_id", "pmid",
        "lncrna_id", "partner_id", "relation_type", "experiment_type",
        "cancer_id", "tissue", "cell_line", "species", "doi",
        "experiment_id", "biosample", "intervention", "readout", "mediator_id",
        "experiment_raw", "graph_assay_class", "assay_subtype",
    )
    physical_all = (
        "SELECT 'PHYS:' || substr(" + physical_signature + ",1,24) AS physical_fact_id, "
        "cancer_id, lncrna_id, partner_id, relation_type, experiment_type, "
        "experiment_raw, experiment_family, assay_subtype, graph_assay_class, "
        "source_database, source_dataset, source_record_id, pmid, "
        "doi, experiment_id, biosample, intervention, readout, mediator_id, "
        "tissue, cell_line, species, source_row_sha256, "
        "false AS is_prediction FROM raw_events "
        "WHERE is_physical AND lncrna_id<>'' AND partner_id<>''"
    )
    physical_path = output_root / "physical_interaction_facts.parquet"
    _copy_query(
        connection,
        "SELECT * EXCLUDE(_rn) FROM (SELECT p.*, count(*) OVER "
        "(PARTITION BY physical_fact_id) AS source_occurrence_count, "
        "row_number() OVER (PARTITION BY physical_fact_id ORDER BY physical_fact_id) _rn "
        f"FROM ({physical_all}) p) WHERE _rn=1 ORDER BY physical_fact_id",
        physical_path,
        row_group_size=row_group_size,
    )

    event_signature = _stable_sql(
        "cancer_id", "lncrna_id", "route_pathway_id", "partner_id",
        "source_database", "source_dataset", "source_record_id", "pmid",
        "relation_type", "experiment_type", "direction_raw", "doi",
        "experiment_id", "biosample", "intervention", "readout", "mediator_id",
        "tissue", "cell_line", "species", "experiment_raw",
        "graph_assay_class", "assay_subtype",
    )
    physical_for_event = _stable_sql(
        "source_database", "source_dataset", "source_record_id", "pmid",
        "lncrna_id", "partner_id", "relation_type", "experiment_type",
        "cancer_id", "tissue", "cell_line", "species", "doi",
        "experiment_id", "biosample", "intervention", "readout", "mediator_id",
        "experiment_raw", "graph_assay_class", "assay_subtype",
    )
    connection.execute(
        "CREATE VIEW all_exact_events AS SELECT "
        "'EV32:' || substr(" + event_signature + ",1,24) AS event_id, "
        "cancer_id, lncrna_id, route_pathway_id AS pathway_id, partner_id, "
        "coalesce(nullif(member_type,''),'unknown') AS member_type, route_type, "
        "static_member_id, CASE WHEN is_physical AND partner_id<>'' THEN "
        "'PHYS:' || substr(" + physical_for_event + ",1,24) ELSE '' END AS physical_fact_id, "
        "source_database, source_dataset, source_record_id, pmid, doi, "
        "experiment_id, biosample, intervention, readout, mediator_id, "
        "experiment_type, experiment_raw, experiment_family, assay_subtype, "
        "graph_assay_class, relation_type, direction_raw, "
        "CASE WHEN route_type='DIRECT_EXACT_ASSERTION' "
        "THEN direction_target ELSE -1 END AS direction_target, "
        "CASE WHEN route_type='DIRECT_EXACT_ASSERTION' "
        "THEN confidence_target ELSE NULL END AS confidence_target, "
        "manual_review_status, tissue, cell_line, species, "
        "is_experimental, is_computational, is_physical, "
        "false AS is_model_prediction, source_kind, source_input, source_row_index, "
        "source_row_sha256, source_sha256, raw_event_id, pathway_family_id, "
        "lncrna_mapping_state, partner_mapping_state, input_mapping_status "
        "FROM eligible_routes"
    )
    source_events_path = output_root / "source_exact_events.parquet"
    event_public_columns = (
        "event_id, cancer_id, lncrna_id, pathway_id, partner_id, member_type, "
        "route_type, static_member_id, physical_fact_id, source_database, "
        "source_dataset, source_record_id, pmid, doi, experiment_id, biosample, "
        "intervention, readout, mediator_id, experiment_type, experiment_raw, "
        "experiment_family, assay_subtype, graph_assay_class, relation_type, "
        "direction_raw, direction_target, confidence_target, manual_review_status, "
        "tissue, cell_line, species, is_experimental, is_computational, is_physical, "
        "is_model_prediction, source_occurrence_count, review_conflict"
    )
    # Conflicting explicit quality reviews remain unknown regardless of input
    # row order or database duplication.
    connection.execute(
        "CREATE VIEW review_checked_events AS SELECT * "
        "EXCLUDE(confidence_target, _qmin, _qmax), "
        "CASE WHEN _qmin <> _qmax THEN NULL ELSE confidence_target END "
        "AS confidence_target, coalesce(_qmin <> _qmax, false) "
        "AS review_conflict FROM (SELECT e.*, "
        "min(confidence_target) OVER (PARTITION BY event_id) AS _qmin, "
        "max(confidence_target) OVER (PARTITION BY event_id) AS _qmax "
        "FROM all_exact_events e)"
    )
    _copy_query(
        connection,
        f"SELECT {event_public_columns} FROM (SELECT e.*, count(*) OVER "
        "(PARTITION BY event_id) source_occurrence_count, row_number() OVER "
        "(PARTITION BY event_id ORDER BY route_type, source_row_sha256) _rn "
        "FROM review_checked_events e) "
        "WHERE _rn=1 ORDER BY cancer_id, lncrna_id, pathway_id, event_id",
        source_events_path,
        row_group_size=row_group_size,
    )
    connection.execute(
        f"CREATE VIEW source_events AS SELECT * FROM read_parquet({_quote(source_events_path)})"
    )

    lineage_id = _stable_sql(
        "event_id", "source_kind", "source_row_index", "source_row_sha256"
    )
    lineage_path = output_root / "event_lineage.parquet"
    _copy_query(
        connection,
        "SELECT * EXCLUDE(_rn) FROM (SELECT "
        "'LIN32:' || substr(" + lineage_id + ",1,24) AS lineage_id, event_id, "
        "raw_event_id, source_kind, source_input, source_row_index, source_row_sha256, "
        "source_sha256, route_type AS mapping_route, static_member_id, "
        "pathway_family_id AS family_id_observed_but_unused, "
        "false AS family_broadcast_used, lncrna_mapping_state, partner_mapping_state, "
        "input_mapping_status, row_number() OVER (PARTITION BY "
        "'LIN32:' || substr(" + lineage_id + ",1,24) ORDER BY event_id) _rn "
        "FROM all_exact_events) WHERE _rn=1 ORDER BY lineage_id",
        lineage_path,
        row_group_size=row_group_size,
    )

    rejected_path = output_root / "rejected_raw_events.parquet"
    _copy_query(
        connection,
        "SELECT r.*, CASE "
        "WHEN r.lncrna_id='' THEN 'LNC_ID_UNMAPPED' "
        "WHEN r.cancer_id IN ('CONTEXT_UNRESOLVED','CONTEXT_UNSPECIFIED') "
        "THEN 'CANCER_CONTEXT_UNRESOLVED_NOT_BROADCAST' "
        "WHEN regexp_matches(r.relation_type, '(?i)ceRNA|sponge') "
        "AND r.pathway_id='' "
        "THEN 'MEDIATED_RELATION_REQUIRES_MEDIATOR_NOT_DIRECT_EDGE' "
        "WHEN r.family_only AND NOT EXISTS (SELECT 1 FROM unfiltered_routes u "
        " WHERE u.source_kind=r.source_kind AND u.source_row_index=r.source_row_index) "
        "THEN 'FAMILY_ONLY_NOT_BROADCAST_TO_EXACT' "
        "WHEN EXISTS (SELECT 1 FROM unfiltered_routes u WHERE "
        " u.source_kind=r.source_kind AND u.source_row_index=r.source_row_index) "
        "THEN 'OUTSIDE_EXACT_CANDIDATE_UNIVERSE' "
        "ELSE 'NO_EXACT_PATHWAY_OR_STATIC_MEMBER_MAPPING' END AS rejection_reason "
        "FROM raw_events r WHERE NOT EXISTS (SELECT 1 FROM eligible_routes e WHERE "
        "e.source_kind=r.source_kind AND e.source_row_index=r.source_row_index) "
        "ORDER BY source_kind, source_row_index",
        rejected_path,
        row_group_size=row_group_size,
    )

    bagev_signature = _stable_sql("c.cancer_id", "e.event_id")
    candidate_events_path = output_root / "candidate_exact_events.parquet"
    _copy_query(
        connection,
        "SELECT * EXCLUDE(_rn) FROM (SELECT "
        "'BAGEV32:' || substr(" + bagev_signature + ",1,24) AS event_id, "
        "c.cancer_id, e.* EXCLUDE(event_id,cancer_id), e.event_id AS source_event_id, "
        "row_number() OVER (PARTITION BY c.cancer_id,e.lncrna_id,e.pathway_id,"
        "'BAGEV32:' || substr(" + bagev_signature + ",1,24) ORDER BY e.event_id) _rn "
        "FROM source_events e JOIN candidates c ON c.lncrna_id=e.lncrna_id "
        "AND c.pathway_id=e.pathway_id AND "
        "(e.cancer_id='PAN_CANCER' OR c.cancer_id=e.cancer_id)) "
        "WHERE _rn=1 ORDER BY cancer_id,lncrna_id,pathway_id,event_id",
        candidate_events_path,
        row_group_size=row_group_size,
    )

    availability_path = output_root / "candidate_event_availability.parquet"
    _copy_query(
        connection,
        "SELECT c.cancer_id,c.lncrna_id,c.pathway_id, count(e.event_id) AS event_count, "
        "count(e.event_id)>0 AS event_bag_available, false AS prediction_available, "
        "CASE WHEN count(e.event_id)=0 "
        f"THEN '{NO_EVENT_REASON}' ELSE 'FRESH_EVIDENCE_TRAINING_NOT_STARTED' END "
        "AS failure_reason, "
        "CAST(NULL AS DOUBLE) AS confidence_probability, "
        "CAST(NULL AS VARCHAR) AS direction_label, "
        "false AS formal_training_started FROM candidates c LEFT JOIN "
        f"read_parquet({_quote(candidate_events_path)}) e USING(cancer_id,lncrna_id,pathway_id) "
        "GROUP BY c.cancer_id,c.lncrna_id,c.pathway_id,c.candidate_order "
        "ORDER BY c.candidate_order",
        availability_path,
        row_group_size=row_group_size,
    )
    counts = {
        "candidate_rows": int(candidate_metrics[0]),
        "candidate_cancers": int(candidate_metrics[1]),
        "static_exact_members": int(member_metrics[0]),
    }
    for key, path in (
        ("source_exact_events", source_events_path),
        ("candidate_exact_events", candidate_events_path),
        ("event_lineage", lineage_path),
        ("physical_facts", physical_path),
        ("rejected_raw_events", rejected_path),
        ("candidate_availability", availability_path),
    ):
        counts[key] = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({_quote(path)})"
            ).fetchone()[0]
        )
    return counts


def _materialize_pair_folds(
    connection: Any,
    *,
    candidate_events_path: Path,
    output_root: Path,
    seed: int,
    row_group_size: int,
    require_all_folds: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    """Assign the exact legacy pair-balanced folds without loading event rows."""

    _, pa, pq = _require_runtime()
    seed_text = str(int(seed))
    tie_sha = _stable_sql(_quote(seed_text), "pair_token")
    query = (
        "SELECT lncrna_id,pathway_id,event_rows,pair_token," + tie_sha
        + " AS tie_sha256 FROM (SELECT lncrna_id,pathway_id,count(*) AS event_rows,"
        "'pair:' || lncrna_id || '|' || pathway_id AS pair_token FROM "
        f"read_parquet({_quote(candidate_events_path)}) GROUP BY lncrna_id,pathway_id) "
        "ORDER BY event_rows DESC,tie_sha256 ASC"
    )
    assignment_path = output_root / "pair_fold_assignment.parquet"
    schema = pa.schema(
        [
            pa.field("lncrna_id", pa.string(), nullable=False),
            pa.field("pathway_id", pa.string(), nullable=False),
            pa.field("event_rows", pa.int64(), nullable=False),
            pa.field("pair_token", pa.string(), nullable=False),
            pa.field("split_component_id", pa.string(), nullable=False),
            pa.field("leakage_fold", pa.int8(), nullable=False),
        ]
    )
    writer = pq.ParquetWriter(assignment_path, schema, compression="zstd")
    fold_sizes = [0] * N_FOLDS
    pair_counts = [0] * N_FOLDS
    total_pairs = 0
    cursor = connection.execute(query)
    try:
        while rows := cursor.fetchmany(50_000):
            output_rows: list[dict[str, Any]] = []
            for lnc, pathway, event_rows, pair_token, tie in rows:
                smallest = min(fold_sizes)
                choices = [fold for fold, size in enumerate(fold_sizes) if size == smallest]
                fold = choices[int(str(tie)[:12], 16) % len(choices)]
                count = int(event_rows)
                fold_sizes[fold] += count
                pair_counts[fold] += 1
                total_pairs += 1
                output_rows.append(
                    {
                        "lncrna_id": str(lnc),
                        "pathway_id": str(pathway),
                        "event_rows": count,
                        "pair_token": str(pair_token),
                        "split_component_id": "SPLIT32:"
                        + _stable_sha256(seed, pair_token)[:24],
                        "leakage_fold": fold,
                    }
                )
            writer.write_table(pa.Table.from_pylist(output_rows, schema=schema))
    finally:
        writer.close()
    active = [fold for fold, count in enumerate(pair_counts) if count]
    if require_all_folds and active != list(range(N_FOLDS)):
        raise StreamingEvidenceStageError(
            f"Pair-blocked staging did not populate all five folds: {active}"
        )
    folded_path = output_root / "candidate_exact_events_pair_folded.parquet"
    component_hash = _stable_sql(_quote(seed_text), "'pair:' || e.lncrna_id || '|' || e.pathway_id")
    _copy_query(
        connection,
        "SELECT e.*, 'SPLIT32:' || substr(" + component_hash
        + ",1,24) AS split_component_id, a.leakage_fold FROM "
        f"read_parquet({_quote(candidate_events_path)}) e JOIN "
        f"read_parquet({_quote(assignment_path)}) a USING(lncrna_id,pathway_id) "
        "ORDER BY cancer_id,lncrna_id,pathway_id,event_id",
        folded_path,
        row_group_size=row_group_size,
    )
    cross_fold = int(
        connection.execute(
            "SELECT count(*) FROM (SELECT lncrna_id,pathway_id,count(DISTINCT leakage_fold) n "
            f"FROM read_parquet({_quote(folded_path)}) GROUP BY lncrna_id,pathway_id HAVING n<>1)"
        ).fetchone()[0]
    )
    if cross_fold:
        raise StreamingEvidenceStageError("lncRNA--exact-pathway pairs cross folds")
    return assignment_path, folded_path, {
        "split_policy": EVIDENCE_SPLIT_POLICY,
        "seed": int(seed),
        "hard_split_unit": "cancer_agnostic_lncrna_exact_pathway_pair",
        "hard_pair_count": total_pairs,
        "hard_pair_cross_fold_count": cross_fold,
        "all_five_folds_populated": active == list(range(N_FOLDS)),
        "active_folds": active,
        "event_rows_by_fold": {str(i): fold_sizes[i] for i in range(N_FOLDS)},
        "biological_pairs_by_fold": {str(i): pair_counts[i] for i in range(N_FOLDS)},
    }


def _materialize_fold_staging(
    connection: Any,
    *,
    folded_events_path: Path,
    output_root: Path,
    row_group_size: int,
) -> dict[str, Any]:
    """Write five-fold artifacts and prove independent train/validation/evaluation splits."""

    connection.execute(
        f"CREATE VIEW folded_events AS SELECT * FROM read_parquet({_quote(folded_events_path)})"
    )
    connection.execute(
        "CREATE VIEW event_pmid_tokens AS SELECT DISTINCT event_id,leakage_fold,"
        "lower(trim(token)) AS pmid_token FROM folded_events, "
        "unnest(regexp_split_to_array(coalesce(pmid,''), '[;,|\\s]+')) AS x(token) "
        "WHERE trim(token)<>'' AND lower(trim(token))<>'unknown'"
    )
    connection.execute(
        "CREATE VIEW event_dataset_tokens AS SELECT DISTINCT event_id,leakage_fold,"
        "lower(token) AS dataset_token FROM folded_events, "
        "unnest(regexp_extract_all(upper(coalesce(source_dataset,'')), "
        "'(GSE[0-9]+|SRP[0-9]+|PRJ[A-Z]{2}[0-9]+|E-MTAB-[0-9]+|EGAS[0-9]+)')) AS x(token)"
    )

    # Materialize every protected-fold exclusion index before auditing the
    # outer-fold train/validation selections, which need two such indexes.
    for fold in range(N_FOLDS):
        fold_root = output_root / f"pair_fold={fold}"
        fold_root.mkdir()
        evaluation_path = fold_root / "evaluation_events.parquet"
        exclusions_path = fold_root / "train_provenance_exclusions.parquet"
        _copy_query(
            connection,
            f"SELECT * FROM folded_events WHERE leakage_fold={fold} "
            "ORDER BY cancer_id,lncrna_id,pathway_id,event_id",
            evaluation_path,
            row_group_size=row_group_size,
        )
        exclusion_query = (
            "SELECT * FROM (SELECT t.event_id, "
            "EXISTS (SELECT 1 FROM event_pmid_tokens tp JOIN event_pmid_tokens ep "
            "ON tp.pmid_token=ep.pmid_token WHERE tp.event_id=t.event_id "
            f"AND ep.leakage_fold={fold}) AS shared_protected_pmid, "
            "EXISTS (SELECT 1 FROM event_dataset_tokens td JOIN event_dataset_tokens ed "
            "ON td.dataset_token=ed.dataset_token WHERE td.event_id=t.event_id "
            f"AND ed.leakage_fold={fold}) AS shared_protected_dataset, "
            "EXISTS (SELECT 1 FROM folded_events e WHERE e.leakage_fold="
            f"{fold} AND e.source_event_id<>'' AND e.source_event_id=t.source_event_id) "
            "AS shared_protected_source_event, "
            "EXISTS (SELECT 1 FROM folded_events e WHERE e.leakage_fold="
            f"{fold} AND e.source_record_id<>'' AND lower(e.source_database)="
            "lower(t.source_database) AND lower(e.source_dataset)=lower(t.source_dataset) "
            "AND e.source_record_id=t.source_record_id) AS shared_protected_source_record "
            f"FROM folded_events t WHERE t.leakage_fold<>{fold}) q WHERE "
            "shared_protected_pmid OR shared_protected_dataset OR "
            "shared_protected_source_event OR shared_protected_source_record "
            "ORDER BY event_id"
        )
        _copy_query(
            connection, exclusion_query, exclusions_path, row_group_size=row_group_size
        )

    def residual_counts(selection_sql: str, protected_folds: tuple[int, ...]) -> tuple[int, int, int, int]:
        protected = ",".join(str(value) for value in protected_folds)
        row = connection.execute(
            "WITH kept AS (" + selection_sql + "), "
            "kept_pmids AS (SELECT DISTINCT lower(trim(token)) token FROM kept, "
            "unnest(regexp_split_to_array(coalesce(pmid,''), '[;,|\\s]+')) q(token) "
            "WHERE trim(token)<>'' AND lower(trim(token))<>'unknown'), "
            "protected_pmids AS (SELECT DISTINCT pmid_token token FROM event_pmid_tokens "
            f"WHERE leakage_fold IN ({protected})), "
            "kept_datasets AS (SELECT DISTINCT lower(token) token FROM kept, "
            "unnest(regexp_extract_all(upper(coalesce(source_dataset,'')), "
            "'(GSE[0-9]+|SRP[0-9]+|PRJ[A-Z]{2}[0-9]+|E-MTAB-[0-9]+|EGAS[0-9]+)')) q(token)), "
            "protected_datasets AS (SELECT DISTINCT dataset_token token FROM event_dataset_tokens "
            f"WHERE leakage_fold IN ({protected})) "
            "SELECT "
            "(SELECT count(*) FROM kept_pmids JOIN protected_pmids USING(token)), "
            "(SELECT count(*) FROM kept_datasets JOIN protected_datasets USING(token)), "
            "(SELECT count(*) FROM kept k JOIN folded_events e ON "
            f"e.leakage_fold IN ({protected}) AND e.source_event_id<>'' "
            "AND e.source_event_id=k.source_event_id), "
            "(SELECT count(*) FROM kept k JOIN folded_events e ON "
            f"e.leakage_fold IN ({protected}) AND e.source_record_id<>'' "
            "AND lower(e.source_database)=lower(k.source_database) "
            "AND lower(e.source_dataset)=lower(k.source_dataset) "
            "AND e.source_record_id=k.source_record_id)"
        ).fetchone()
        return tuple(int(value) for value in row)

    per_fold: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        validation_fold = (fold + 1) % N_FOLDS
        evaluation_exclusion = output_root / f"pair_fold={fold}" / "train_provenance_exclusions.parquet"
        validation_exclusion = output_root / f"pair_fold={validation_fold}" / "train_provenance_exclusions.parquet"
        validation_sql = (
            "SELECT e.* FROM folded_events e LEFT JOIN "
            f"read_parquet({_quote(evaluation_exclusion)}) x USING(event_id) "
            f"WHERE e.leakage_fold={validation_fold} AND x.event_id IS NULL"
        )
        train_sql = (
            "SELECT e.* FROM folded_events e LEFT JOIN "
            f"read_parquet({_quote(evaluation_exclusion)}) x USING(event_id) LEFT JOIN "
            f"read_parquet({_quote(validation_exclusion)}) v USING(event_id) "
            f"WHERE e.leakage_fold NOT IN ({fold},{validation_fold}) "
            "AND x.event_id IS NULL AND v.event_id IS NULL"
        )
        evaluation_rows = int(connection.execute(
            f"SELECT count(*) FROM folded_events WHERE leakage_fold={fold}"
        ).fetchone()[0])
        validation_before = int(connection.execute(
            f"SELECT count(*) FROM folded_events WHERE leakage_fold={validation_fold}"
        ).fetchone()[0])
        validation_after = int(connection.execute(
            f"SELECT count(*) FROM ({validation_sql})"
        ).fetchone()[0])
        train_before = int(connection.execute(
            f"SELECT count(*) FROM folded_events WHERE leakage_fold NOT IN ({fold},{validation_fold})"
        ).fetchone()[0])
        train_after = int(connection.execute(
            f"SELECT count(*) FROM ({train_sql})"
        ).fetchone()[0])
        validation_residual = residual_counts(validation_sql, (fold,))
        train_residual = residual_counts(train_sql, (fold, validation_fold))
        if any(validation_residual) or any(train_residual):
            raise StreamingEvidenceStageError(
                f"Fold {fold} retained protected provenance: "
                f"validation={validation_residual}, train={train_residual}"
            )
        selection = {
            "format": STREAMING_FORMAT,
            "fold": fold,
            "evaluation_fold": fold,
            "validation_fold": validation_fold,
            "training_folds": [
                value for value in range(N_FOLDS)
                if value not in {fold, validation_fold}
            ],
            "training_storage_policy": (
                "BASE_EVENTS_MINUS_EVALUATION_AND_VALIDATION_FOLDS_"
                "MINUS_BOTH_PROVENANCE_EXCLUSION_INDEXES"
            ),
            "base_events_relative_path": folded_events_path.name,
            "evaluation_events_relative_path": (
                f"pair_fold={fold}/evaluation_events.parquet"
            ),
            "evaluation_exclusions_relative_path": str(
                evaluation_exclusion.relative_to(output_root)
            ),
            "validation_exclusions_relative_path": str(
                validation_exclusion.relative_to(output_root)
            ),
            "evaluation_provenance_exclusion_policy": (
                EVALUATION_PROVENANCE_EXCLUSION_POLICY
            ),
            "evaluation_used_for_model_selection": False,
            "formal_training_started": False,
        }
        _json_write(
            output_root / f"pair_fold={fold}" / "TRAIN_SELECTION.json",
            selection,
        )
        per_fold[str(fold)] = {
            "evaluation_fold": fold,
            "validation_fold": validation_fold,
            "training_folds": selection["training_folds"],
            "evaluation_event_rows": evaluation_rows,
            "validation_event_rows_before_provenance_exclusion": validation_before,
            "validation_event_rows_removed": validation_before - validation_after,
            "validation_event_rows_after": validation_after,
            "train_event_rows_before_provenance_exclusion": train_before,
            "train_event_rows_removed": train_before - train_after,
            "train_event_rows_after": train_after,
            "residual_validation_evaluation_provenance_overlap_count": sum(validation_residual),
            "residual_train_protected_provenance_overlap_count": sum(train_residual),
            "residual_train_evaluation_provenance_overlap_count": sum(train_residual),
        }
    return {
        "policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
        "split_contract": "THREE_WAY_TRAIN_VALIDATION_EVALUATION",
        "validation_rotation": "(evaluation_fold+1)%5",
        "evaluation_used_for_model_selection": False,
        "folds": per_fold,
        "training_rows_are_not_duplicated_on_disk": True,
        "bounded_reader_required": True,
    }

def iter_fold_event_bag_batches(
    stage_root: Path,
    *,
    fold: int,
    split: str,
    arrow_batch_rows: int = 50_000,
    max_bags_per_batch: int = 256,
    max_events_per_batch: int = 50_000,
    max_single_bag_events: int | None = None,
    per_bag_event_limit: int | None = None,
) -> Iterator[StreamingEventBagBatch]:
    """Read complete event bags without materialising a fold in Python memory."""

    duckdb, _, _ = _require_runtime()
    root = Path(stage_root).resolve()
    if split not in {"train", "validation", "evaluation"}:
        raise StreamingEvidenceStageError(
            "split must be train, validation or evaluation"
        )
    if fold not in range(N_FOLDS):
        raise StreamingEvidenceStageError("fold must be in [0,4]")
    if min(arrow_batch_rows, max_bags_per_batch, max_events_per_batch) <= 0:
        raise StreamingEvidenceStageError("reader bounds must be positive")
    single_bag_limit = (
        max_events_per_batch
        if max_single_bag_events is None
        else int(max_single_bag_events)
    )
    if single_bag_limit <= 0 or single_bag_limit > max_events_per_batch:
        raise StreamingEvidenceStageError(
            "max_single_bag_events must be positive and no larger than max_events_per_batch"
        )
    if per_bag_event_limit is not None and int(per_bag_event_limit) <= 0:
        raise StreamingEvidenceStageError("per_bag_event_limit must be positive")
    manifest_path = root / "STAGING_MANIFEST.json"
    if not manifest_path.is_file():
        raise StreamingEvidenceStageError("Streaming staging manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != STREAMING_FORMAT or manifest.get("status") != STAGING_STATUS:
        raise StreamingEvidenceStageError("Streaming stage is not passing")
    base = root / "candidate_exact_events_pair_folded.parquet"
    evaluation_exclusion = (
        root / f"pair_fold={fold}" / "train_provenance_exclusions.parquet"
    )
    validation_fold = (fold + 1) % N_FOLDS
    validation_exclusion = (
        root / f"pair_fold={validation_fold}" / "train_provenance_exclusions.parquet"
    )
    for artifact in (base, evaluation_exclusion, validation_exclusion):
        if not artifact.is_file():
            raise StreamingEvidenceStageError(
                f"Fold staging artifact missing: {artifact}"
            )
    if split == "evaluation":
        selection_query = (
            f"SELECT * FROM read_parquet({_quote(base)}) WHERE leakage_fold={fold}"
        )
    elif split == "validation":
        selection_query = (
            f"SELECT e.* FROM read_parquet({_quote(base)}) e LEFT JOIN "
            f"read_parquet({_quote(evaluation_exclusion)}) x USING(event_id) "
            f"WHERE e.leakage_fold={validation_fold} AND x.event_id IS NULL"
        )
    else:
        selection_query = (
            f"SELECT e.* FROM read_parquet({_quote(base)}) e LEFT JOIN "
            f"read_parquet({_quote(evaluation_exclusion)}) x USING(event_id) "
            f"LEFT JOIN read_parquet({_quote(validation_exclusion)}) v USING(event_id) "
            f"WHERE e.leakage_fold NOT IN ({fold},{validation_fold}) "
            "AND x.event_id IS NULL AND v.event_id IS NULL"
        )
    if per_bag_event_limit is not None:
        selection_query = (
            "SELECT * FROM (SELECT s.*, row_number() OVER (PARTITION BY "
            "cancer_id,lncrna_id,pathway_id ORDER BY event_id) AS _bag_row "
            f"FROM ({selection_query}) s) WHERE _bag_row<={int(per_bag_event_limit)}"
        )
    query = (
        f"SELECT * EXCLUDE(_bag_row) FROM ({selection_query}) "
        "ORDER BY cancer_id,lncrna_id,pathway_id,event_id"
        if per_bag_event_limit is not None
        else selection_query + " ORDER BY cancer_id,lncrna_id,pathway_id,event_id"
    )
    connection = duckdb.connect(":memory:")
    connection.execute("SET memory_limit=\'2GB\'")
    reader = connection.execute(query).to_arrow_reader(batch_size=arrow_batch_rows)
    pending_key: tuple[str, str, str] | None = None
    pending_rows: list[Mapping[str, Any]] = []
    output_bags: list[StreamingEventBag] = []
    output_events = 0

    def flush_pending() -> StreamingEventBag | None:
        nonlocal pending_key, pending_rows
        if pending_key is None:
            return None
        if len(pending_rows) > single_bag_limit:
            raise StreamingEvidenceStageError(
                f"One event bag exceeds max_single_bag_events={single_bag_limit}"
            )
        bag = StreamingEventBag(pending_key, tuple(pending_rows))
        pending_key, pending_rows = None, []
        return bag

    try:
        for record_batch in reader:
            for row in record_batch.to_pylist():
                key = tuple(str(row[column]) for column in EXACT_KEYS)
                if pending_key is not None and key != pending_key:
                    bag = flush_pending()
                    assert bag is not None
                    if output_bags and (
                        len(output_bags) >= max_bags_per_batch
                        or output_events + len(bag.events) > max_events_per_batch
                    ):
                        yield StreamingEventBagBatch(tuple(output_bags), output_events)
                        output_bags, output_events = [], 0
                    output_bags.append(bag)
                    output_events += len(bag.events)
                if pending_key is None:
                    pending_key = key
                pending_rows.append(row)
                if len(pending_rows) > single_bag_limit:
                    raise StreamingEvidenceStageError(
                        "Single evidence bag exceeds the raw-memory limit; "
                        "explicit source aggregation is required"
                    )
        bag = flush_pending()
        if bag is not None:
            if output_bags and (
                len(output_bags) >= max_bags_per_batch
                or output_events + len(bag.events) > max_events_per_batch
            ):
                yield StreamingEventBagBatch(tuple(output_bags), output_events)
                output_bags, output_events = [], 0
            output_bags.append(bag)
            output_events += len(bag.events)
        if output_bags:
            yield StreamingEventBagBatch(tuple(output_bags), output_events)
    finally:
        connection.close()


def iter_fold_trainer_batches(
    stage_root: Path,
    *,
    fold: int,
    split: str,
    core: CoreFeatureBundle,
    event_feature_dim: int = 128,
    max_events_per_bag: int = 64,
    max_bags_per_batch: int = 256,
) -> Iterator[StreamingTrainerBatch]:
    """Stream bounded examples after deriving supervision from each complete bag."""

    from .evidence_training import build_bag_examples

    if max_events_per_bag <= 0:
        raise StreamingEvidenceStageError("max_events_per_bag must be positive")
    raw_bag_limit = max(50_000, max_events_per_bag)
    for packet in iter_fold_event_bag_batches(
        stage_root,
        fold=fold,
        split=split,
        max_bags_per_batch=max_bags_per_batch,
        max_events_per_batch=raw_bag_limit,
        max_single_bag_events=raw_bag_limit,
        per_bag_event_limit=None,
    ):
        examples: list[BagExample] = []
        missing_core: list[Mapping[str, Any]] = []
        for bag in packet.bags:
            values, absent = build_bag_examples(
                pd.DataFrame(bag.events),
                core,
                event_feature_dim=event_feature_dim,
                max_events=max_events_per_bag,
                independent_supervision=True,
            )
            examples.extend(values)
            missing_core.extend(absent.to_dict("records"))
        yield StreamingTrainerBatch(tuple(examples), tuple(missing_core))

def validate_streaming_stage(stage_root: Path) -> dict[str, Any]:
    """Fail closed on any post-staging mutation or accidental training claim."""

    root = Path(stage_root).resolve()
    manifest_path = root / "STAGING_MANIFEST.json"
    success_path = root / "SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise StreamingEvidenceStageError("Streaming stage lacks manifest/SUCCESS")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if manifest.get("format") != STREAMING_FORMAT or manifest.get("status") != STAGING_STATUS:
        raise StreamingEvidenceStageError("Streaming stage manifest is not passing")
    if success.get("format") != STREAMING_FORMAT or success.get("status") != STAGING_STATUS:
        raise StreamingEvidenceStageError("Streaming stage SUCCESS is not passing")
    if manifest.get("formal_training_started") is not False:
        raise StreamingEvidenceStageError("Staging receipt makes an impermissible training claim")
    if manifest.get("production_deployed") is not False:
        raise StreamingEvidenceStageError("Staging receipt makes an impermissible deployment claim")
    if success.get("staging_manifest_sha256") != file_sha256(manifest_path):
        raise StreamingEvidenceStageError("SUCCESS does not bind the staging manifest")
    for relative, record in manifest.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise StreamingEvidenceStageError(f"Staging artifact missing/unsafe: {relative}")
        if file_sha256(path) != record.get("sha256"):
            raise StreamingEvidenceStageError(f"Staging artifact hash drift: {relative}")
    return manifest


def materialize_streaming_evidence_stage(
    *,
    interaction_relation_path: Path,
    empty_evidence_event_path: Path,
    exact_pathway_members_path: Path,
    formal_candidates_path: Path,
    conversion_manifest_path: Path,
    patient_fold_authority_path: Path,
    patient_fold_receipt_path: Path,
    output_root: Path,
    expected_hashes: Mapping[str, str],
    identifier_map: Mapping[str, str] | None = None,
    strict_formal: bool = True,
    batch_size: int = 50_000,
    row_group_size: int = 100_000,
    seed: int = 20260726,
    memory_limit: str = "8GB",
    threads: int = 4,
    direct_evidence_event_path: Path | None = None,
    r62_core_authority_path: Path | None = None,
) -> dict[str, Any]:
    """Create a hash-pinned, non-overwriting, bounded Evidence training stage.

    ``expected_hashes`` is mandatory for every authority, including both
    receipts.  A caller therefore cannot swap a validated recovered Parquet for
    a path-compatible file.  Setting ``strict_formal=False`` exists only for
    small equivalence fixtures; it is recorded in the receipt and cannot be
    mistaken for the 3.3M formal stage.
    """

    duckdb, _, _ = _require_runtime()
    paths = {
        "interaction_relation": Path(interaction_relation_path),
        "empty_evidence_event": Path(empty_evidence_event_path),
        "exact_pathway_members": Path(exact_pathway_members_path),
        "formal_candidates": Path(formal_candidates_path),
        "conversion_manifest": Path(conversion_manifest_path),
        "patient_fold_authority": Path(patient_fold_authority_path),
        "patient_fold_receipt": Path(patient_fold_receipt_path),
    }
    if direct_evidence_event_path is not None:
        paths["direct_evidence_event"] = Path(direct_evidence_event_path)
    inputs = _sha_contract(paths, expected_hashes)
    expected_candidate_sha = FORMAL_CANDIDATE_SHA256
    expected_candidate_rows = FORMAL_CANDIDATE_ROWS
    expected_candidate_cancers = 33
    r62_authority_sha = None
    if r62_core_authority_path is not None:
        if not strict_formal:
            raise StreamingEvidenceStageError(
                "R6.2 production authority cannot be combined with fixture mode"
            )
        from .core_export_contract_r4 import validate_export
        try:
            authority, r62_authority_sha, _ = validate_export(r62_core_authority_path)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise StreamingEvidenceStageError(f"R6.2 core authority failed validation: {exc}") from exc
        candidate = authority.get("candidates")
        if not isinstance(candidate, Mapping):
            raise StreamingEvidenceStageError("R6.2 core authority lacks the candidate binding")
        expected_candidate_sha = str(candidate.get("sha256", ""))
        expected_candidate_rows = int(candidate.get("rows", 0))
        expected_candidate_cancers = int(candidate.get("cancers", 0))
        if (re.fullmatch(r"[0-9a-f]{64}", expected_candidate_sha) is None
                or expected_candidate_rows < 1 or expected_candidate_cancers != 33):
            raise StreamingEvidenceStageError("R6.2 candidate authority is incomplete")
    if strict_formal and inputs["formal_candidates"]["sha256"] != expected_candidate_sha:
        raise StreamingEvidenceStageError("Formal candidate authority SHA256 drifted")
    conversion = _validate_conversion_receipt(
        paths["conversion_manifest"],
        relation_path=paths["interaction_relation"],
        event_path=paths["empty_evidence_event"],
        relation_sha256=inputs["interaction_relation"]["sha256"],
        event_sha256=inputs["empty_evidence_event"]["sha256"],
    )
    _validate_empty_event_authority(paths["empty_evidence_event"])
    patient_receipt = _validate_patient_authority(
        paths["patient_fold_authority"], paths["patient_fold_receipt"],
        authority_sha256=inputs["patient_fold_authority"]["sha256"],
        strict_formal=strict_formal,
    )
    target = Path(output_root).resolve()
    temporary = target.with_name(f".{target.name}.tmp")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Evidence staging refuses output reuse: {target}")
    temporary.mkdir(parents=True)
    work = temporary / "_work"
    work.mkdir()
    spill = work / "duckdb_spill"
    spill.mkdir()
    normalized_raw = work / "normalized_raw.parquet"
    normalized_members = work / "normalized_members.parquet"
    normalized_candidates = work / "normalized_candidates.parquet"
    database = work / "stage.duckdb"
    connection = None
    try:
        event_sources = (
            ("evidence_event", paths["empty_evidence_event"]),
            ("interaction_relation", paths["interaction_relation"]),
        )
        if "direct_evidence_event" in paths:
            event_sources += (("evidence_event", paths["direct_evidence_event"]),)
        raw_counts = _stage_normalized_raw(
            sources=event_sources,
            output_path=normalized_raw,
            identifier_map=dict(identifier_map or {}),
            batch_size=batch_size,
        )
        staged_members = _stage_members(
            paths["exact_pathway_members"], normalized_members,
            identifier_map=dict(identifier_map or {}), batch_size=batch_size,
        )
        staged_candidates = _stage_candidates(
            paths["formal_candidates"], normalized_candidates, batch_size=batch_size
        )
        if "direct_evidence_event" not in paths and raw_counts.get("evidence_event") != 0:
            raise StreamingEvidenceStageError("Empty Evidence authority produced event rows")
        declared_interactions = int(conversion.get("counts", {}).get("rows", -1))
        if raw_counts.get("interaction_relation") != declared_interactions:
            raise StreamingEvidenceStageError(
                "Recovered interaction count disagrees with conversion receipt"
            )
        if strict_formal and staged_candidates != expected_candidate_rows:
            raise StreamingEvidenceStageError("Formal candidate row count drifted")
        if (strict_formal and r62_core_authority_path is None
                and raw_counts.get("interaction_relation") != FORMAL_RECOVERED_INTERACTION_ROWS):
            raise StreamingEvidenceStageError("Formal recovered interaction row count drifted")
        connection = duckdb.connect(str(database))
        connection.execute(f"SET memory_limit={_quote(memory_limit)}")
        connection.execute(f"SET threads={max(1, int(threads))}")
        connection.execute(f"SET temp_directory={_quote(spill)}")
        counts = _materialize_exact_tables(
            connection,
            normalized_raw_path=normalized_raw,
            member_stage_path=normalized_members,
            candidate_stage_path=normalized_candidates,
            output_root=temporary,
            row_group_size=row_group_size,
        )
        if strict_formal and (
            counts["candidate_rows"] != expected_candidate_rows
            or counts["candidate_cancers"] != expected_candidate_cancers
        ):
            raise StreamingEvidenceStageError("Formal candidate universe cardinality drifted")
        _, folded_events, split_audit = _materialize_pair_folds(
            connection,
            candidate_events_path=temporary / "candidate_exact_events.parquet",
            output_root=temporary,
            seed=seed,
            row_group_size=row_group_size,
            require_all_folds=True,
        )
        provenance_audit = _materialize_fold_staging(
            connection,
            folded_events_path=folded_events,
            output_root=temporary,
            row_group_size=row_group_size,
        )
        connection.close()
        connection = None
        shutil.rmtree(work)

        artifact_records: dict[str, dict[str, Any]] = {}
        for path in sorted(p for p in temporary.rglob("*") if p.is_file()):
            relative = path.relative_to(temporary).as_posix()
            artifact_records[relative] = {
                "sha256": file_sha256(path), "bytes": path.stat().st_size
            }
        generated = datetime.now(timezone.utc).isoformat()
        manifest: dict[str, Any] = {
            "format": STREAMING_FORMAT,
            "status": STAGING_STATUS,
            "generated_at_utc": generated,
            "output_root": str(target),
            "strict_formal_3_3m_contract": bool(strict_formal and r62_core_authority_path is None),
            "strict_formal_r62_contract": bool(r62_core_authority_path is not None),
            "r62_core_authority_sha256": r62_authority_sha,
            "formal_training_started": False,
            "optimizer_initialized": False,
            "checkpoint_loaded": False,
            "production_deployed": False,
            "production_port_8260_touched": False,
            "server_accessed_by_this_materialization": False,
            "real_6160707_server_preflight_executed": False,
            "inputs": inputs,
            "input_receipts": {
                "conversion_status": conversion["status"],
                "independent_rematerialization_validation_bound": True,
                "patient_authority_format": patient_receipt["format"],
                "patient_cross_fold_count_zero": True,
            },
            "counts": {
                **raw_counts,
                "staged_exact_member_rows_before_global_dedup": staged_members,
                "staged_candidate_rows": staged_candidates,
                **counts,
            },
            "exact_event_contract": {
                "semantic_reference": "build_exact_event_bags+materialize_candidate_events",
                "supervision_semantics": "INDEPENDENT_REVIEWED_DIRECT_STUDIES_ONLY",
                "targets_aggregated_before_event_token_bound": True,
                "direct_exact_assertion": True,
                "partner_to_static_exact_member": True,
                "family_broadcast_used": False,
                "candidate_local_filter": True,
                "candidate_pan_cancer_filter_and_attachment": True,
                "physical_facts_are_predictions": False,
                "mapping_states": ["MAPPED_UNIQUE", "AMBIGUOUS", "UNMAPPED"],
                "event_and_lineage_ids_deterministic_sha256": True,
            },
            "split_audit": split_audit,
            "evaluation_provenance_exclusion_audit": provenance_audit,
            "test_firewall": {
                "patient_fold_authority_hash_pinned": True,
                "patient_level_rows_joined_to_event_supervision": False,
                "outcome_or_sealed_test_columns_allowed": False,
                "evaluation_rows_unchanged": True,
            },
            "bounded_memory_contract": {
                "raw_arrow_batch_rows": int(batch_size),
                "parquet_row_group_size": int(row_group_size),
                "duckdb_memory_limit": str(memory_limit),
                "duckdb_threads": max(1, int(threads)),
                "entire_interaction_table_loaded_in_pandas": False,
                "python_event_rows_global_list_built": False,
                "fold_train_rows_duplicated_on_disk": False,
                "trainer_interface": "iter_fold_event_bag_batches",
            },
            "typed_unavailable_contract": {
                "artifact": "candidate_event_availability.parquet",
                "prediction_available_before_training": False,
                "no_event_failure_reason": NO_EVENT_REASON,
            },
            "artifacts": artifact_records,
        }
        manifest_path = temporary / "STAGING_MANIFEST.json"
        _json_write(manifest_path, manifest)
        success = {
            "format": STREAMING_FORMAT,
            "status": STAGING_STATUS,
            "generated_at_utc": generated,
            "staging_manifest_sha256": file_sha256(manifest_path),
            "formal_training_started": False,
            "production_deployed": False,
            "production_port_8260_touched": False,
        }
        _json_write(temporary / "SUCCESS.json", success)
        os.replace(temporary, target)
        return validate_streaming_stage(target)
    except Exception:
        if connection is not None:
            connection.close()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "CONVERSION_FORMAT",
    "CONVERSION_PASS",
    "EMPTY_EVENT_COLUMNS",
    "FORMAL_RECOVERED_INTERACTION_ROWS",
    "NO_EVENT_REASON",
    "PATIENT_RECEIPT_FORMAT",
    "STAGING_STATUS",
    "STREAMING_FORMAT",
    "StreamingEvidenceStageError",
    "StreamingEventBag",
    "StreamingEventBagBatch",
    "StreamingTrainerBatch",
    "iter_fold_event_bag_batches",
    "iter_fold_trainer_batches",
    "materialize_streaming_evidence_stage",
    "validate_streaming_stage",
]
