"""Fresh V3.2 known-positive rank-recovery external validation.

Historical material is admitted only as raw evidence or as an identifier/task
definition.  Historical metrics, details, ranks, and predictions are neither
read nor accepted.  All ranks are rebuilt from the current V3.2 five-fold
exact-pathway ensemble after recomputing PMID overlap against the exact raw
evidence sources used by V3.2 training.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CC_HHGT_V3_2_FRESH_EXTERNAL_VALIDATION_BINDING_V1"
RELEASE_STATUS = "SUCCESS_FRESH_V32_EXTERNAL_VALIDATION_HASH_BOUND"
RANK_CUTOFFS = (10, 25, 50, 100)
TOP_PATHWAYS_PER_LNCRNA = 5
FORMAL_TASK_ROWS = 645_602
FORMAL_PREDICTION_ROWS = 3_300_000
FORMAL_CANDIDATE_PAIRS = 76_734
FORMAL_CANCERS = 33
FORMAL_LNCRNAS = 8_541

FORMAL_TASK_DEFINITION_SHA256 = (
    "6b47cc31f06b0d9969d49ee3654934fe0eaf93963b09c3b0706971a099667cb4"
)
FORMAL_PREDICTION_SHA256 = (
    "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1"
)
FORMAL_PREDICTION_LINEAGE_SHA256 = (
    "892f13c7a0109d644f93806fefc5b254117cef0329285ad53c8d98107459ca08"
)
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_TRAINING_EVIDENCE_SHA256 = (
    "b330a54c1865e4bfb1c9f16374f79ec7aa46b170c364d0e6c43541e6a5375626"
)
FORMAL_TRAINING_INTERACTION_SHA256 = (
    "bcc79ec17f3e1a57fb64c7087b19a474bfe056c496140253a8080c324b19e8ed"
)

RAW_SOURCE_AUTHORITIES: dict[str, dict[str, Any]] = {
    "lnc2cancer": {
        "sha256": "f8c7ed996b75a0dff02e9806aedd5a740b0b29e1e9d54c9b2d3f64ede3c9ea10",
        "rows": 24_170,
        "required_columns": {"name", "cancer type", "methods", "pubmed id"},
    },
    "lncrnadisease": {
        "sha256": "432d0d0b0043d94a15afee46df059be6b156c704b1254faa618eaa6de006d7d9",
        "rows": 207_537,
        "required_columns": {
            "ncRNA Symbol", "Disease Name", "Validated Method//Prediction Method",
            "PubMed ID",
        },
    },
    "rnadisease_experimental": {
        "sha256": "c4e45d4a662915187918ca2e1f5f8eb47061ced599f255c28221a07002837aca",
        "rows": 65_781,
        "required_columns": {"RDID", "RNA Symbol", "Disease Name", "PMID"},
    },
    "rnadisease_predicted": {
        "sha256": "f7445567756a0f578d09fb32a97e12b5a67ac2afc4ddaaaca62b955e4556767e",
        "rows": 348_062,
        "required_columns": {"RDID", "RNA_symbol", "disease_name", "method_name"},
    },
    "gse85011": {
        "sha256": "c3206b5974a8e10f1dcfdacbb02e6955d7b9228ed9b2c09925123e0df2a12810",
        "rows": 112,
        "required_columns": {
            "dataset_accession", "gsm", "cell_line", "target_raw", "pmid",
        },
    },
}

TASK_DEFINITION_COLUMNS = (
    "source_database",
    "source_row_id",
    "lncrna_raw",
    "disease_raw",
    "pmid",
    "evidence_tier",
    "is_experimental",
    "is_predicted",
    "dataset_accession",
    "cell_line",
    "growth_modifier_hit",
    "lncrna_id",
    "lncrna_mapping_status",
    "cancer_id",
    "cancer_mapping_status",
)
HISTORICAL_DERIVED_COLUMNS_IGNORED = (
    "training_pmid_overlap",
    "validation_role",
    "primary_validation_eligible",
    "consistency_validation_eligible",
    "external_evidence_id",
    "independent_event_hash",
    "analysis_version",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PMID = re.compile(r"\b(\d{6,9})\b")


class ExternalValidationReleaseError(RuntimeError):
    """Raised when external-validation materialization fails closed."""


def artifact_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ExternalValidationReleaseError(f"Artifact is missing or unsafe: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ExternalValidationReleaseError(f"{label} is missing or unsafe: {source}")
    return source


def _json_load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalValidationReleaseError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ExternalValidationReleaseError(f"{label} must be a JSON object")
    return value


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def normalize_pmid(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    match = _PMID.search(str(value).strip())
    return match.group(1) if match else ""


def _validate_hash(
    path: Path,
    expected: str,
    label: str,
    strict_formal_authority: bool,
) -> str:
    observed = artifact_sha256(path)
    if strict_formal_authority and observed != expected:
        raise ExternalValidationReleaseError(
            f"Formal {label} SHA256 mismatch: {observed} != {expected}"
        )
    return observed


def _validate_raw_sources(
    raw_sources: dict[str, Path],
    *,
    strict_formal_authority: bool,
) -> dict[str, dict[str, Any]]:
    if set(raw_sources) != set(RAW_SOURCE_AUTHORITIES):
        raise ExternalValidationReleaseError(
            f"Raw source roles must be exactly {sorted(RAW_SOURCE_AUTHORITIES)}"
        )
    declarations: dict[str, dict[str, Any]] = {}
    for role, specification in RAW_SOURCE_AUTHORITIES.items():
        path = _safe_file(raw_sources[role], f"raw source {role}")
        digest = _validate_hash(
            path,
            specification["sha256"],
            f"raw source {role}",
            strict_formal_authority,
        )
        try:
            header = pd.read_csv(path, sep="\t", nrows=0)
        except Exception as exc:
            raise ExternalValidationReleaseError(
                f"Raw source is unreadable: {role}"
            ) from exc
        required = set(specification["required_columns"])
        if missing := sorted(required - set(header.columns)):
            raise ExternalValidationReleaseError(
                f"Raw source {role} lacks columns: {missing}"
            )
        observed_rows: int | None = None
        if strict_formal_authority:
            first_column = str(header.columns[0])
            observed_rows = len(
                pd.read_csv(path, sep="\t", usecols=[first_column], dtype="string")
            )
            if observed_rows != int(specification["rows"]):
                raise ExternalValidationReleaseError(
                    f"Raw source {role} row count drift: {observed_rows}"
                )
        declarations[role] = {
            "path": str(path),
            "sha256": digest,
            "rows": observed_rows if observed_rows is not None else None,
            "role": "HISTORICAL_RAW_KNOWN_POSITIVE_SOURCE_ONLY",
        }
    return declarations


def _load_task_definition(
    path: Path,
    *,
    strict_formal_authority: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        con = duckdb.connect(":memory:")
        relation = f"read_parquet({_sql_path(path)})"
        columns = {
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        }
        row_count = int(con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0])
    finally:
        con.close()
    if missing := sorted(set(TASK_DEFINITION_COLUMNS) - columns):
        raise ExternalValidationReleaseError(
            f"Known-positive task definition lacks columns: {missing}"
        )
    forbidden_result_tokens = (
        "metric", "hits_at", "recall_at", "fold_enrichment", "rank_score",
        "percentile", "prediction_probability",
    )
    result_columns = sorted(
        column
        for column in columns
        if any(token in column.lower() for token in forbidden_result_tokens)
    )
    if result_columns:
        raise ExternalValidationReleaseError(
            f"Task definition contains forbidden historical results: {result_columns}"
        )
    if strict_formal_authority and row_count != FORMAL_TASK_ROWS:
        raise ExternalValidationReleaseError(
            f"Formal task-definition row count drift: {row_count}"
        )
    # Deliberately request only the definition whitelist.  Old eligibility,
    # overlap, result IDs, and version fields are never loaded into memory.
    frame = pd.read_parquet(path, columns=list(TASK_DEFINITION_COLUMNS))
    if len(frame) != row_count:
        raise ExternalValidationReleaseError("Task-definition read was incomplete")
    frame["source_database"] = frame.source_database.astype("string")
    frame["source_row_id"] = frame.source_row_id.astype("string")
    frame["lncrna_id"] = frame.lncrna_id.astype("string")
    frame["cancer_id"] = frame.cancer_id.astype("string")
    frame["pmid_normalized"] = frame.pmid.map(normalize_pmid).astype("string")
    for column in ("is_experimental", "is_predicted", "growth_modifier_hit"):
        frame[column] = frame[column].fillna(False).astype(bool)
    audit = {
        "path": str(path),
        "sha256": artifact_sha256(path),
        "rows": row_count,
        "role": "HISTORICAL_IDENTIFIER_AND_KNOWN_POSITIVE_TASK_DEFINITION_ONLY",
        "columns_present": sorted(columns),
        "columns_consumed": list(TASK_DEFINITION_COLUMNS),
        "historical_derived_columns_explicitly_ignored": list(
            HISTORICAL_DERIVED_COLUMNS_IGNORED
        ),
        "historical_metrics_read": False,
        "historical_details_read": False,
        "historical_ranks_read": False,
        "historical_predictions_read": False,
    }
    return frame, audit


def _training_pmids(path: Path) -> set[str]:
    con = duckdb.connect(":memory:")
    relation = f"read_parquet({_sql_path(path)})"
    try:
        columns = {
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        }
        if "pmid" not in columns:
            raise ExternalValidationReleaseError(
                f"Training raw source lacks pmid: {path}"
            )
        values = con.execute(
            f"""
            SELECT DISTINCT regexp_extract(cast(pmid AS VARCHAR), '([0-9]{{6,9}})', 1)
            FROM {relation}
            WHERE regexp_matches(cast(pmid AS VARCHAR), '[0-9]{{6,9}}')
            """
        ).fetchall()
    finally:
        con.close()
    return {str(row[0]) for row in values if row[0]}


def _validate_prediction_authority(
    prediction_path: Path,
    lineage_path: Path,
    candidate_path: Path,
    *,
    strict_formal_authority: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lineage = _json_load(lineage_path, "Current V3.2 prediction lineage")
    required_lineage = {
        "analysis_version": ANALYSIS_VERSION,
        "five_fold_ensemble": True,
        "folds": 5,
        "trained_from_scratch": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "prediction_rows": FORMAL_PREDICTION_ROWS if strict_formal_authority else lineage.get("prediction_rows"),
    }
    for key, expected in required_lineage.items():
        if lineage.get(key) != expected:
            raise ExternalValidationReleaseError(
                f"Current V3.2 prediction lineage has invalid {key}: {lineage.get(key)!r}"
            )
    if lineage.get("prediction_sha256") != artifact_sha256(prediction_path):
        raise ExternalValidationReleaseError("Current prediction lineage SHA is stale")

    con = duckdb.connect(":memory:")
    prediction = f"read_parquet({_sql_path(prediction_path)})"
    candidate = f"read_parquet({_sql_path(candidate_path)})"
    try:
        prediction_columns = {
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM {prediction}").fetchall()
        }
        required = {
            "cancer_id", "lncrna_id", "pathway_id",
            "association_membership_probability", "n_folds_available",
            "analysis_version", "training_run_id",
        }
        if missing := sorted(required - prediction_columns):
            raise ExternalValidationReleaseError(
                f"Current V3.2 predictions lack columns: {missing}"
            )
        prediction_audit = con.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS keys,
                   count(DISTINCT (cancer_id, lncrna_id)) AS pairs,
                   count(DISTINCT cancer_id) AS cancers,
                   count(DISTINCT lncrna_id) AS lncrnas,
                   count_if(association_membership_probability NOT BETWEEN 0 AND 1
                            OR association_membership_probability IS NULL
                            OR n_folds_available <> 5
                            OR analysis_version <> ?) AS invalid
            FROM {prediction}
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        candidate_audit = con.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS keys,
                   count(DISTINCT (cancer_id, lncrna_id)) AS pairs
            FROM {candidate}
            """
        ).fetchone()
        key_drift = int(
            con.execute(
                f"""
                SELECT count(*) FROM (
                    (SELECT cancer_id, lncrna_id, pathway_id FROM {prediction}
                     EXCEPT
                     SELECT cancer_id, lncrna_id, pathway_id FROM {candidate})
                    UNION ALL
                    (SELECT cancer_id, lncrna_id, pathway_id FROM {candidate}
                     EXCEPT
                     SELECT cancer_id, lncrna_id, pathway_id FROM {prediction})
                )
                """
            ).fetchone()[0]
        )
        ranks = con.execute(
            f"""
            WITH ordered AS (
                SELECT cancer_id::VARCHAR AS cancer_id,
                       lncrna_id::VARCHAR AS lncrna_id,
                       pathway_id::VARCHAR AS pathway_id,
                       association_membership_probability::DOUBLE AS probability,
                       row_number() OVER (
                           PARTITION BY cancer_id, lncrna_id
                           ORDER BY association_membership_probability DESC,
                                    pathway_id
                       ) AS pathway_rank
                FROM {prediction}
            ), aggregated AS (
                SELECT cancer_id, lncrna_id,
                       max(probability)::DOUBLE AS max_pathway_probability,
                       avg(probability) FILTER (
                           WHERE pathway_rank <= {TOP_PATHWAYS_PER_LNCRNA}
                       )::DOUBLE AS mean_top5_pathway_probability,
                       count(*) FILTER (
                           WHERE pathway_rank <= {TOP_PATHWAYS_PER_LNCRNA}
                       )::BIGINT AS n_top_pathways,
                       max(CASE WHEN pathway_rank = 1 THEN pathway_id END)
                           AS top_pathway_id
                FROM ordered
                GROUP BY cancer_id, lncrna_id
            ), scored AS (
                SELECT *,
                       (0.7 * max_pathway_probability
                        + 0.3 * mean_top5_pathway_probability)::DOUBLE
                           AS lncrna_rank_score
                FROM aggregated
            )
            SELECT *,
                   row_number() OVER (
                       PARTITION BY cancer_id
                       ORDER BY lncrna_rank_score DESC, lncrna_id
                   )::BIGINT AS rank,
                   count(*) OVER (PARTITION BY cancer_id)::BIGINT AS n_ranked_lncrnas
            FROM scored
            ORDER BY cancer_id, rank
            """
        ).fetchdf()
    finally:
        con.close()

    prediction_audit = tuple(map(int, prediction_audit))
    candidate_audit = tuple(map(int, candidate_audit))
    if (
        prediction_audit[0] != prediction_audit[1]
        or prediction_audit[5]
        or candidate_audit[0] != candidate_audit[1]
        or prediction_audit[:1] != candidate_audit[:1]
        or prediction_audit[2] != candidate_audit[2]
        or key_drift
    ):
        raise ExternalValidationReleaseError(
            "Current prediction/candidate semantic alignment failed"
        )
    if strict_formal_authority and (
        prediction_audit[0] != FORMAL_PREDICTION_ROWS
        or prediction_audit[2] != FORMAL_CANDIDATE_PAIRS
        or prediction_audit[3] != FORMAL_CANCERS
        or prediction_audit[4] != FORMAL_LNCRNAS
    ):
        raise ExternalValidationReleaseError(
            f"Formal prediction counts are invalid: {prediction_audit}"
        )
    ranks["percentile"] = 1.0 - (
        (ranks["rank"] - 1.0) / ranks["n_ranked_lncrnas"].clip(lower=1)
    )
    ranks["ranking_policy"] = (
        "0.7_MAX_MEMBERSHIP_PROBABILITY_PLUS_0.3_MEAN_TOP5_MEMBERSHIP_PROBABILITY"
    )
    return ranks, {
        "prediction_rows": prediction_audit[0],
        "candidate_pairs": prediction_audit[2],
        "cancers": prediction_audit[3],
        "lncrnas": prediction_audit[4],
        "lineage_training_run_id": lineage.get("training_run_id"),
        "five_fold_ensemble": True,
        "fresh_rank_recomputed": True,
    }


def _pmid_overlap_audit(
    task: pd.DataFrame,
    evidence_pmids: set[str],
    interaction_pmids: set[str],
) -> pd.DataFrame:
    frame = task.copy()
    frame["pmid_available"] = frame.pmid_normalized.astype(str).ne("")
    frame["training_evidence_event_present"] = frame.pmid_normalized.astype(str).isin(
        evidence_pmids
    ) & frame.pmid_available
    frame["training_interaction_relation_present"] = frame.pmid_normalized.astype(
        str
    ).isin(interaction_pmids) & frame.pmid_available
    frame["training_pmid_overlap"] = (
        frame.training_evidence_event_present
        | frame.training_interaction_relation_present
    )
    frame["mapped_context"] = frame.lncrna_id.notna() & frame.cancer_id.notna()
    frame["primary_before_pmid"] = (
        frame.mapped_context & frame.is_experimental & ~frame.is_predicted
    )
    frame["primary_after_pmid"] = (
        frame.primary_before_pmid
        & frame.pmid_available
        & ~frame.training_pmid_overlap
    )
    audit = (
        frame.groupby(
            ["source_database", "pmid_normalized"],
            dropna=False,
            observed=True,
            as_index=False,
        )
        .agg(
            external_records=("source_row_id", "size"),
            mapped_context_records=("mapped_context", "sum"),
            experimental_records=("is_experimental", "sum"),
            predicted_records=("is_predicted", "sum"),
            primary_records_before_pmid=("primary_before_pmid", "sum"),
            primary_records_eligible_after_pmid=("primary_after_pmid", "sum"),
            training_evidence_event_present=(
                "training_evidence_event_present", "max"
            ),
            training_interaction_relation_present=(
                "training_interaction_relation_present", "max"
            ),
            training_pmid_overlap=("training_pmid_overlap", "max"),
        )
    )
    audit["pmid_available"] = audit.pmid_normalized.astype(str).ne("")
    audit["excluded_from_primary"] = (
        ~audit.pmid_available | audit.training_pmid_overlap
    )
    audit["exclusion_reason"] = np.select(
        [~audit.pmid_available, audit.training_pmid_overlap],
        ["PMID_UNAVAILABLE_FOR_OVERLAP_AUDIT", "PMID_PRESENT_IN_V32_TRAINING_SOURCE"],
        default=None,
    )
    audit["pmid_normalized"] = audit.pmid_normalized.replace("", pd.NA)
    return audit.sort_values(
        ["source_database", "pmid_available", "pmid_normalized"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _eligibility_masks(task: pd.DataFrame, training_pmids: set[str]) -> dict[str, pd.Series]:
    mapped = task.lncrna_id.notna() & task.cancer_id.notna()
    pmid_available = task.pmid_normalized.astype(str).ne("")
    overlap = task.pmid_normalized.astype(str).isin(training_pmids) & pmid_available
    primary_before = mapped & task.is_experimental & ~task.is_predicted
    primary = primary_before & pmid_available & ~overlap
    secondary_before = mapped & task.is_predicted
    secondary = secondary_before & ~overlap
    return {
        "mapped": mapped,
        "pmid_available": pmid_available,
        "overlap": overlap,
        "primary_before": primary_before,
        "primary": primary,
        "secondary_before": secondary_before,
        "secondary": secondary,
    }


def _aggregate_positive_details(
    task: pd.DataFrame,
    masks: dict[str, pd.Series],
    ranks: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for role, mask_name, basis in (
        (
            "primary_known_positive",
            "primary",
            "EXPERIMENTAL_OR_CLINICAL_WITH_NONOVERLAPPING_PMID",
        ),
        (
            "secondary_consistency",
            "secondary",
            "COMPUTATIONAL_PREDICTION_TASK_DEFINITION_ONLY",
        ),
    ):
        selected = task.loc[masks[mask_name]].copy()
        if selected.empty:
            continue
        selected["pmid_value"] = selected.pmid_normalized.astype(str)
        selected["dataset_value"] = selected.dataset_accession.fillna("").astype(str)
        selected["cell_line_value"] = selected.cell_line.fillna("").astype(str)
        grouped = (
            selected.groupby(
                ["source_database", "cancer_id", "lncrna_id"],
                observed=True,
                as_index=False,
            )
            .agg(
                evidence_record_count=("source_row_id", "size"),
                independent_pmid_count=(
                    "pmid_value", lambda value: len({item for item in value if item})
                ),
                independent_pmids=(
                    "pmid_value",
                    lambda value: ";".join(sorted({item for item in value if item})),
                ),
                dataset_accessions=(
                    "dataset_value",
                    lambda value: ";".join(sorted({item for item in value if item})),
                ),
                cell_lines=(
                    "cell_line_value",
                    lambda value: ";".join(sorted({item for item in value if item})),
                ),
            )
        )
        grouped["validation_role"] = role
        grouped["known_positive_basis"] = basis
        grouped["pmid_overlap_exclusion_applied"] = True
        grouped["training_pmid_overlap_in_used_records"] = False
        parts.append(grouped)
    if not parts:
        raise ExternalValidationReleaseError("No external-validation positives remain")
    details = pd.concat(parts, ignore_index=True)

    primary_context = task.loc[masks["primary_before"]].copy()
    if not primary_context.empty:
        primary_context["overlap_excluded"] = masks["overlap"].loc[
            primary_context.index
        ]
        primary_context["missing_pmid_excluded"] = ~masks["pmid_available"].loc[
            primary_context.index
        ]
        excluded = (
            primary_context.groupby(
                ["source_database", "cancer_id", "lncrna_id"],
                observed=True,
                as_index=False,
            )
            .agg(
                overlapping_records_excluded=("overlap_excluded", "sum"),
                missing_pmid_records_excluded=("missing_pmid_excluded", "sum"),
            )
        )
        excluded["validation_role"] = "primary_known_positive"
        details = details.merge(
            excluded,
            on=[
                "validation_role", "source_database", "cancer_id", "lncrna_id"
            ],
            how="left",
            validate="one_to_one",
        )
    else:
        details["overlapping_records_excluded"] = 0
        details["missing_pmid_records_excluded"] = 0
    for column in ("overlapping_records_excluded", "missing_pmid_records_excluded"):
        details[column] = details[column].fillna(0).astype("int64")
    details = details.merge(
        ranks,
        on=["cancer_id", "lncrna_id"],
        how="left",
        validate="many_to_one",
    )
    details["availability"] = details["rank"].notna()
    details["failure_reason"] = np.where(
        details.availability,
        None,
        "NOT_IN_CURRENT_V32_CANDIDATE_UNIVERSE",
    )
    for cutoff in RANK_CUTOFFS:
        details[f"recovered_at_{cutoff}"] = (
            details.availability & details["rank"].le(cutoff)
        )
    return details.sort_values(
        ["validation_role", "source_database", "cancer_id", "lncrna_id"],
        kind="stable",
    ).reset_index(drop=True)


def _cohort_audit(task: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    primary = task.loc[masks["primary_before"]].copy()
    primary["pmid_missing"] = ~masks["pmid_available"].loc[primary.index]
    primary["pmid_overlap"] = masks["overlap"].loc[primary.index]
    primary["eligible"] = masks["primary"].loc[primary.index]
    if not primary.empty:
        value = (
            primary.groupby(
                ["source_database", "cancer_id"], observed=True, as_index=False
            )
            .agg(
                task_records_before_exclusion=("source_row_id", "size"),
                missing_pmid_records_excluded=("pmid_missing", "sum"),
                overlapping_pmid_records_excluded=("pmid_overlap", "sum"),
                eligible_records_after_exclusion=("eligible", "sum"),
            )
        )
        value["validation_role"] = "primary_known_positive"
        parts.append(value)
    secondary = task.loc[masks["secondary_before"]].copy()
    secondary["pmid_missing"] = ~masks["pmid_available"].loc[secondary.index]
    secondary["pmid_overlap"] = masks["overlap"].loc[secondary.index]
    secondary["eligible"] = masks["secondary"].loc[secondary.index]
    if not secondary.empty:
        value = (
            secondary.groupby(
                ["source_database", "cancer_id"], observed=True, as_index=False
            )
            .agg(
                task_records_before_exclusion=("source_row_id", "size"),
                missing_pmid_records_excluded=("pmid_missing", "sum"),
                overlapping_pmid_records_excluded=("pmid_overlap", "sum"),
                eligible_records_after_exclusion=("eligible", "sum"),
            )
        )
        value["validation_role"] = "secondary_consistency"
        parts.append(value)
    if not parts:
        raise ExternalValidationReleaseError("No mapped external-validation cohorts exist")
    return pd.concat(parts, ignore_index=True)


def _build_metrics(
    task: pd.DataFrame,
    masks: dict[str, pd.Series],
    details: pd.DataFrame,
    ranks: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    cohort = _cohort_audit(task, masks)
    rank_sizes = ranks.groupby("cancer_id", observed=True).n_ranked_lncrnas.max()
    detail_groups = {
        key: value
        for key, value in details.groupby(
            ["validation_role", "source_database", "cancer_id"], observed=True
        )
    }
    rows: list[dict[str, Any]] = []
    for record in cohort.itertuples(index=False):
        key = (record.validation_role, record.source_database, record.cancer_id)
        group = detail_groups.get(key, details.iloc[0:0])
        matched = group.loc[group.availability]
        n_positive = int(group.lncrna_id.nunique())
        n_matched = int(matched.lncrna_id.nunique())
        n_ranked = int(rank_sizes.get(record.cancer_id, 0))
        if int(record.eligible_records_after_exclusion) == 0:
            available = False
            failure = "NO_INDEPENDENT_POSITIVE_AFTER_PMID_EXCLUSION"
        elif n_positive == 0:
            available = False
            failure = "NO_UNIQUE_POSITIVE_AFTER_DEDUPLICATION"
        elif n_matched == 0 or n_ranked == 0:
            available = False
            failure = "NO_POSITIVE_IN_CURRENT_V32_CANDIDATE_UNIVERSE"
        else:
            available = True
            failure = None
        median_rank = float(matched["rank"].median()) if len(matched) else math.nan
        median_percentile = (
            float(matched["percentile"].median()) if len(matched) else math.nan
        )
        for cutoff in RANK_CUTOFFS:
            effective_k = min(cutoff, n_ranked)
            hits = int(matched.loc[matched["rank"].le(effective_k), "lncrna_id"].nunique())
            expected_hits = (
                effective_k * n_matched / n_ranked if n_ranked and n_matched else math.nan
            )
            rows.append(
                {
                    "validation_role": record.validation_role,
                    "source_database": record.source_database,
                    "cancer_id": record.cancer_id,
                    "availability": available,
                    "failure_reason": failure,
                    "task_records_before_exclusion": int(
                        record.task_records_before_exclusion
                    ),
                    "missing_pmid_records_excluded": int(
                        record.missing_pmid_records_excluded
                    ),
                    "overlapping_pmid_records_excluded": int(
                        record.overlapping_pmid_records_excluded
                    ),
                    "eligible_records_after_exclusion": int(
                        record.eligible_records_after_exclusion
                    ),
                    "n_external_positive_lncrnas": n_positive,
                    "n_external_positive_in_candidate_universe": n_matched,
                    "n_ranked_lncrnas": n_ranked,
                    "median_positive_rank": median_rank,
                    "median_positive_percentile": median_percentile,
                    "k": cutoff,
                    "effective_k": effective_k,
                    "hits_at_k": hits if available else pd.NA,
                    "annotation_hit_rate_at_k": (
                        hits / effective_k if available and effective_k else math.nan
                    ),
                    "recall_at_k": (
                        hits / n_matched if available and n_matched else math.nan
                    ),
                    "expected_hits_at_k": expected_hits if available else math.nan,
                    "fold_enrichment_at_k": (
                        hits / expected_hits
                        if available and expected_hits and expected_hits > 0
                        else math.nan
                    ),
                    "analysis_version": ANALYSIS_VERSION,
                    "computation_run_id": run_id,
                }
            )
    metrics = pd.DataFrame(rows)
    for column in (
        "task_records_before_exclusion", "missing_pmid_records_excluded",
        "overlapping_pmid_records_excluded", "eligible_records_after_exclusion",
        "n_external_positive_lncrnas", "n_external_positive_in_candidate_universe",
        "n_ranked_lncrnas", "k", "effective_k", "hits_at_k",
    ):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce").astype("Int64")
    return metrics.sort_values(
        ["validation_role", "source_database", "cancer_id", "k"], kind="stable"
    ).reset_index(drop=True)


def _source_summary(
    task: pd.DataFrame,
    masks: dict[str, pd.Series],
    details: pd.DataFrame,
) -> pd.DataFrame:
    work = task.copy()
    work["mapped_context"] = masks["mapped"]
    work["pmid_available"] = masks["pmid_available"]
    work["training_pmid_overlap"] = masks["overlap"]
    work["primary_eligible"] = masks["primary"]
    work["secondary_eligible"] = masks["secondary"]
    summary = (
        work.groupby("source_database", observed=True, as_index=False)
        .agg(
            task_records=("source_row_id", "size"),
            mapped_context_records=("mapped_context", "sum"),
            pmid_available_records=("pmid_available", "sum"),
            training_pmid_overlap_records=("training_pmid_overlap", "sum"),
            primary_eligible_records=("primary_eligible", "sum"),
            secondary_eligible_records=("secondary_eligible", "sum"),
        )
    )
    detail = (
        details.groupby("source_database", observed=True, as_index=False)
        .agg(
            unique_positive_contexts=("lncrna_id", "size"),
            rank_available_contexts=("availability", "sum"),
        )
    )
    return summary.merge(detail, on="source_database", how="left").fillna(
        {"unique_positive_contexts": 0, "rank_available_contexts": 0}
    )


def materialize_external_validation_release(
    *,
    task_definition_path: str | Path,
    raw_source_paths: dict[str, str | Path],
    training_evidence_path: str | Path,
    training_interaction_path: str | Path,
    current_prediction_path: str | Path,
    current_prediction_lineage_path: str | Path,
    current_candidate_path: str | Path,
    output_root: str | Path,
    runner_path: str | Path,
    strict_formal_authority: bool = True,
) -> dict[str, Any]:
    """Build a new immutable external-validation release from approved inputs."""

    task_path = _safe_file(task_definition_path, "Known-positive task definition")
    training_evidence = _safe_file(training_evidence_path, "V3.2 training evidence")
    training_interaction = _safe_file(
        training_interaction_path, "V3.2 training interaction relation"
    )
    prediction = _safe_file(current_prediction_path, "Current V3.2 predictions")
    prediction_lineage = _safe_file(
        current_prediction_lineage_path, "Current V3.2 prediction lineage"
    )
    candidates = _safe_file(current_candidate_path, "Current V3.2 candidates")
    code_path = _safe_file(Path(__file__), "External-validation materializer code")
    runner = _safe_file(runner_path, "External-validation runner")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise ExternalValidationReleaseError(
            f"Refusing to overwrite external-validation release: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw_paths = {
        role: _safe_file(path, f"raw source {role}")
        for role, path in raw_source_paths.items()
    }
    raw_declarations = _validate_raw_sources(
        raw_paths, strict_formal_authority=strict_formal_authority
    )
    task_sha = _validate_hash(
        task_path,
        FORMAL_TASK_DEFINITION_SHA256,
        "task definition",
        strict_formal_authority,
    )
    training_evidence_sha = _validate_hash(
        training_evidence,
        FORMAL_TRAINING_EVIDENCE_SHA256,
        "training evidence",
        strict_formal_authority,
    )
    training_interaction_sha = _validate_hash(
        training_interaction,
        FORMAL_TRAINING_INTERACTION_SHA256,
        "training interaction relation",
        strict_formal_authority,
    )
    prediction_sha = _validate_hash(
        prediction,
        FORMAL_PREDICTION_SHA256,
        "current V3.2 prediction",
        strict_formal_authority,
    )
    prediction_lineage_sha = _validate_hash(
        prediction_lineage,
        FORMAL_PREDICTION_LINEAGE_SHA256,
        "current prediction lineage",
        strict_formal_authority,
    )
    candidate_sha = _validate_hash(
        candidates,
        FORMAL_CANDIDATE_SHA256,
        "current candidates",
        strict_formal_authority,
    )

    task, schema_audit = _load_task_definition(
        task_path, strict_formal_authority=strict_formal_authority
    )
    evidence_pmids = _training_pmids(training_evidence)
    interaction_pmids = _training_pmids(training_interaction)
    training_pmids = evidence_pmids | interaction_pmids
    if not training_pmids:
        raise ExternalValidationReleaseError("No V3.2 training PMIDs could be audited")
    ranks, rank_audit = _validate_prediction_authority(
        prediction,
        prediction_lineage,
        candidates,
        strict_formal_authority=strict_formal_authority,
    )
    algorithm_policy = {
        "task_definition_use": "HISTORICAL_RAW_OR_IDENTIFIER_MAPPING_ONLY",
        "primary_positive": (
            "mapped experimental/non-predicted record with normalized PMID absent "
            "from both V3.2 training raw sources"
        ),
        "secondary_consistency": "mapped predicted task record; never primary evidence",
        "rank_source": "current V3.2 five-fold exact-pathway ensemble only",
        "rank_score": (
            "0.7 * max exact-pathway membership probability + 0.3 * mean top-5 "
            "exact-pathway membership probability"
        ),
        "rank_tie_break": "lncrna_id ascending",
        "cutoffs": list(RANK_CUTOFFS),
        "missing_pmid_primary_policy": "EXCLUDE_FAIL_CLOSED",
    }
    run_id = "V32-EXTERNAL-VALIDATION-" + _canonical_json_sha256(
        {
            "task": task_sha,
            "training_evidence": training_evidence_sha,
            "training_interaction": training_interaction_sha,
            "prediction": prediction_sha,
            "prediction_lineage": prediction_lineage_sha,
            "candidate": candidate_sha,
            "raw": {key: value["sha256"] for key, value in raw_declarations.items()},
            "algorithm": algorithm_policy,
            "code": artifact_sha256(code_path),
        }
    )[:16].upper()

    overlap = _pmid_overlap_audit(task, evidence_pmids, interaction_pmids)
    masks = _eligibility_masks(task, training_pmids)
    details = _aggregate_positive_details(task, masks, ranks)
    metrics = _build_metrics(task, masks, details, ranks, run_id)
    source_summary = _source_summary(task, masks, details)
    for frame in (ranks, details, overlap, source_summary):
        frame["analysis_version"] = ANALYSIS_VERSION
        frame["computation_run_id"] = run_id

    primary_details = details.validation_role.eq("primary_known_positive")
    if (
        details.duplicated(
            ["validation_role", "source_database", "cancer_id", "lncrna_id"]
        ).any()
        or details.loc[primary_details, "independent_pmid_count"].le(0).any()
        or details.loc[
            primary_details, "training_pmid_overlap_in_used_records"
        ].any()
        or ranks.duplicated(["cancer_id", "lncrna_id"]).any()
        or not ranks.lncrna_rank_score.between(0, 1).all()
        or not ranks.percentile.between(0, 1).all()
    ):
        raise ExternalValidationReleaseError("Fresh rank/detail semantic checks failed")
    if strict_formal_authority and (
        len(ranks) != FORMAL_CANDIDATE_PAIRS
        or int(primary_details.sum()) != 7_193
        or int(details.loc[primary_details, "availability"].sum()) != 4_327
    ):
        raise ExternalValidationReleaseError(
            "Formal fresh primary-positive coverage drifted"
        )

    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        paths = {
            "metrics": stage / "external_validation_metrics.parquet",
            "details": stage / "external_validation_cohort_details.parquet",
            "overlap_audit": stage / "external_validation_overlap_audit.parquet",
            "fresh_ranks": stage / "fresh_lncrna_rank.parquet",
            "source_summary": stage / "external_validation_source_summary.parquet",
        }
        metrics.to_parquet(paths["metrics"], index=False, compression="zstd")
        details.to_parquet(paths["details"], index=False, compression="zstd")
        overlap.to_parquet(paths["overlap_audit"], index=False, compression="zstd")
        ranks.to_parquet(paths["fresh_ranks"], index=False, compression="zstd")
        source_summary.to_parquet(
            paths["source_summary"], index=False, compression="zstd"
        )
        schema_path = stage / "TASK_DEFINITION_AUDIT.json"
        schema_audit.update(
            {
                "status": "PASS_DEFINITION_ONLY",
                "raw_authorities": raw_declarations,
                "old_external_result_artifacts_opened": False,
            }
        )
        _json_write(schema_path, schema_audit)
        pmid_audit_path = stage / "TRAINING_PMID_AUDIT.json"
        pmid_audit = {
            "status": "PASS_RECOMPUTED_FROM_V32_TRAINING_RAW_SOURCES",
            "evidence_event_distinct_pmids": len(evidence_pmids),
            "interaction_relation_distinct_pmids": len(interaction_pmids),
            "training_union_distinct_pmids": len(training_pmids),
            "training_union_pmid_sha256": _canonical_json_sha256(
                sorted(training_pmids)
            ),
            "external_task_records_with_training_pmid_overlap": int(
                masks["overlap"].sum()
            ),
            "primary_records_after_overlap_exclusion": int(masks["primary"].sum()),
            "missing_pmid_primary_records_excluded": int(
                (masks["primary_before"] & ~masks["pmid_available"]).sum()
            ),
            "pmid_values_not_written_as_training_list": True,
        }
        _json_write(pmid_audit_path, pmid_audit)

        final = {key: destination / value.name for key, value in paths.items()}
        final_schema = destination / schema_path.name
        final_pmid_audit = destination / pmid_audit_path.name
        final_lineage = destination / "MODULE_LINEAGE.json"
        final_binding = destination / "EXTERNAL_VALIDATION_BINDING.json"
        final_source = destination / "SOURCE_INPUTS.json"
        source_inputs = {
            "analysis_version": ANALYSIS_VERSION,
            "input_policy": "HISTORICAL_RAW_OR_TASK_DEFINITION_PLUS_CURRENT_V32_AUTHORITIES_ONLY",
            "historical_metrics_used": False,
            "historical_details_used": False,
            "historical_ranks_used": False,
            "historical_predictions_used": False,
            "historical_task_definition": {
                "path": str(task_path),
                "sha256": task_sha,
                "rows": len(task),
                "role": "DEFINITION_ONLY",
            },
            "historical_raw_sources": raw_declarations,
            "v32_training_pmid_sources": {
                "evidence_event": {
                    "path": str(training_evidence),
                    "sha256": training_evidence_sha,
                    "role": "CURRENT_V32_TRAINING_RAW_SOURCE_FOR_PMID_EXCLUSION",
                },
                "interaction_relation": {
                    "path": str(training_interaction),
                    "sha256": training_interaction_sha,
                    "role": "CURRENT_V32_TRAINING_RAW_SOURCE_FOR_PMID_EXCLUSION",
                },
            },
            "current_v32_rank_source": {
                "prediction": {
                    "path": str(prediction),
                    "sha256": prediction_sha,
                    "rows": rank_audit["prediction_rows"],
                },
                "prediction_lineage": {
                    "path": str(prediction_lineage),
                    "sha256": prediction_lineage_sha,
                },
                "candidates": {
                    "path": str(candidates),
                    "sha256": candidate_sha,
                },
            },
        }
        source_path = stage / final_source.name
        _json_write(source_path, source_inputs)
        completed_at = datetime.now(timezone.utc).isoformat()
        lineage = {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "external_validation",
            "status": "SUCCESS_FRESH_V32_KNOWN_POSITIVE_RANK_RECOVERY",
            "computation_run_id": run_id,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "fresh_rank_recovery_calculation": True,
            "pmid_overlap_recomputed": True,
            "training_not_applicable": True,
            "historical_data_use_role": "RAW_OR_TASK_DEFINITION_ONLY",
            "historical_metrics_used": False,
            "historical_details_used": False,
            "historical_ranks_used": False,
            "historical_predictions_used": False,
            "current_v32_predictions_used_as_only_rank_source": True,
            "all_output_rows_generated_current_run": True,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
            "algorithm_policy": algorithm_policy,
            "rank_audit": rank_audit,
            "code": {"path": str(code_path), "sha256": artifact_sha256(code_path)},
            "runner": {"path": str(runner), "sha256": artifact_sha256(runner)},
        }
        lineage_path = stage / final_lineage.name
        _json_write(lineage_path, lineage)

        counts = {
            "task_records": len(task),
            "training_union_distinct_pmids": len(training_pmids),
            "external_overlap_records": int(masks["overlap"].sum()),
            "primary_eligible_records": int(masks["primary"].sum()),
            "primary_unique_positives": int(primary_details.sum()),
            "primary_rank_available": int(
                details.loc[primary_details, "availability"].sum()
            ),
            "secondary_unique_positives": int((~primary_details).sum()),
            "secondary_rank_available": int(
                details.loc[~primary_details, "availability"].sum()
            ),
            "detail_rows": len(details),
            "metric_rows": len(metrics),
            "available_metric_rows": int(metrics.availability.sum()),
            "overlap_audit_rows": len(overlap),
            "fresh_rank_rows": len(ranks),
            "source_summary_rows": len(source_summary),
            "rank_cutoffs": len(RANK_CUTOFFS),
        }
        artifacts: dict[str, dict[str, Any]] = {}
        for role, path in paths.items():
            artifacts[role] = {
                "path": str(final[role]),
                "sha256": artifact_sha256(path),
                "rows": int(
                    {
                        "metrics": len(metrics),
                        "details": len(details),
                        "overlap_audit": len(overlap),
                        "fresh_ranks": len(ranks),
                        "source_summary": len(source_summary),
                    }[role]
                ),
            }
        artifacts["task_definition_audit"] = {
            "path": str(final_schema), "sha256": artifact_sha256(schema_path)
        }
        artifacts["training_pmid_audit"] = {
            "path": str(final_pmid_audit), "sha256": artifact_sha256(pmid_audit_path)
        }
        artifacts["source_inputs"] = {
            "path": str(final_source), "sha256": artifact_sha256(source_path)
        }
        artifacts["module_lineage"] = {
            "path": str(final_lineage), "sha256": artifact_sha256(lineage_path)
        }
        binding = {
            "format": BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": RELEASE_STATUS,
            "module_id": "external_validation",
            "result_role": "SECONDARY_INDEPENDENT_KNOWN_POSITIVE_RANK_RECOVERY",
            "computation_run_id": run_id,
            "formal_authority": bool(strict_formal_authority),
            "fresh_rank_recovery_calculation": True,
            "pmid_overlap_recomputed": True,
            "training_not_applicable": True,
            "historical_data_use_role": "RAW_OR_TASK_DEFINITION_ONLY",
            "historical_metrics_used": False,
            "historical_details_used": False,
            "historical_ranks_used": False,
            "historical_predictions_used": False,
            "current_v32_predictions_used_as_only_rank_source": True,
            "all_output_rows_generated_current_run": True,
            "changes_primary_ranking": False,
            "release_ready": False,
            "production_deployed": False,
            "rank_cutoffs": list(RANK_CUTOFFS),
            "algorithm_policy": algorithm_policy,
            "counts": counts,
            "source_inputs": source_inputs,
            "artifacts": artifacts,
        }
        binding_path = stage / final_binding.name
        _json_write(binding_path, binding)
        binding_sha = artifact_sha256(binding_path)
        _json_write(
            stage / "SUCCESS.json",
            {
                "status": RELEASE_STATUS,
                "analysis_version": ANALYSIS_VERSION,
                "computation_run_id": run_id,
                "binding": final_binding.name,
                "binding_sha256": binding_sha,
                "fresh_rank_recovery_calculation": True,
                "pmid_overlap_recomputed": True,
                "changes_primary_ranking": False,
                "release_ready": False,
                "production_deployed": False,
            },
        )
        stage.replace(destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return {
        "output_root": str(destination),
        "binding": str(destination / "EXTERNAL_VALIDATION_BINDING.json"),
        "binding_sha256": binding_sha,
        "computation_run_id": run_id,
        "counts": counts,
        "release_ready": False,
        "production_deployed": False,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "BINDING_FORMAT",
    "ExternalValidationReleaseError",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_PREDICTION_LINEAGE_SHA256",
    "FORMAL_PREDICTION_SHA256",
    "FORMAL_TASK_DEFINITION_SHA256",
    "RANK_CUTOFFS",
    "RAW_SOURCE_AUTHORITIES",
    "RELEASE_STATUS",
    "artifact_sha256",
    "materialize_external_validation_release",
    "normalize_pmid",
]
