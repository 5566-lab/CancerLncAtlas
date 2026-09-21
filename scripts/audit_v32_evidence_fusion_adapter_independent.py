#!/usr/bin/env python3
"""Read-only, independent audit of the V3.2 Evidence canonical fusion adapter.

This program deliberately does not import or call ``evidence_fusion_adapter``.
It reconstructs the expected 3.3M-row output directly from the exact candidate
authority and the raw Evidence predictions, then performs a null-safe comparison
of every output column and every canonical key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import duckdb


ROOT = Path(__file__).resolve().parents[1]
AUDIT_FORMAT = "CC_HHGT_V3_2_EVIDENCE_FUSION_INDEPENDENT_POST_AUDIT_V1"
BINDING_FORMAT = (
    "CC_HHGT_V3_2_EVIDENCE_FUSION_INDEPENDENT_POST_AUDIT_BINDING_V1"
)
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
EXPECTED_ROWS = 3_300_000
EXPECTED_AVAILABLE = 825_753
EXPECTED_UNAVAILABLE = 2_474_247
EXPECTED_PREFIX_RESTORED = 3_300_000
EXPECTED_PATHWAY_NORMALIZED = 12_141
EXPECTED_ADAPTER_BINDING_SHA256 = (
    "6aa9eb9f391cfaa1cdba42c04cb6d3258e4d8574706f89a09a0d018d1226fa3d"
)
EXPECTED_PRIMARY_SHA256 = (
    "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash file names and contents without relying on timestamps or metadata."""
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for child in sorted((p for p in path.rglob("*") if p.is_file()), key=str):
        relative = child.relative_to(path).as_posix()
        sha = file_sha256(child)
        size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
        records.append({"path": str(child.resolve()), "relative_path": relative,
                        "bytes": size, "sha256": sha})
    return digest.hexdigest(), records


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def bound_path(record: Any, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    path = Path(str(record.get("path", "")))
    sha = str(record.get("sha256", ""))
    if not path.is_file() or len(sha) != 64:
        raise ValueError(f"Invalid bound record for {label}: {record!r}")
    return path, sha


def _embedded_file_records(value: Any, prefix: str = "$") -> Iterable[dict[str, str]]:
    """Yield path/hash pairs embedded in a binding, including *_path fields."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield {
                "json_path": prefix,
                "path": str(value["path"]),
                "sha256": str(value["sha256"]),
            }
        if "local_path" in value and "sha256" in value:
            yield {
                "json_path": prefix + ".local_path",
                "path": str(value["local_path"]),
                "sha256": str(value["sha256"]),
            }
        for key, path_value in value.items():
            if not key.endswith("_path") or key in {"declared_path", "local_path"}:
                continue
            sha_key = key[:-5] + "_sha256"
            if sha_key in value:
                yield {
                    "json_path": f"{prefix}.{key}",
                    "path": str(path_value),
                    "sha256": str(value[sha_key]),
                }
        for key, child in value.items():
            yield from _embedded_file_records(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _embedded_file_records(child, f"{prefix}[{index}]")


def verify_embedded_file_records(document: dict[str, Any]) -> dict[str, Any]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for record in _embedded_file_records(document):
        unique[(record["path"], record["sha256"])] = record
    verified: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for record in unique.values():
        path = Path(record["path"])
        if not path.is_file():
            missing.append(record)
            continue
        observed = file_sha256(path)
        result = {**record, "observed_sha256": observed}
        if observed == record["sha256"]:
            verified.append(result)
        else:
            mismatched.append(result)
    return {
        "declared_records": len(unique),
        "verified_records": len(verified),
        "missing_records": missing,
        "mismatched_records": mismatched,
        "records": sorted(verified, key=lambda x: (x["path"], x["sha256"])),
        "status": "PASS" if not missing and not mismatched else "FAIL",
    }


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    return con.execute(query).fetchone()[0]


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def equal(self, name: str, observed: Any, expected: Any) -> None:
        self.rows.append({
            "name": name,
            "expected": expected,
            "observed": observed,
            "status": "PASS" if observed == expected else "FAIL",
        })

    @property
    def failures(self) -> list[str]:
        return [str(row["name"]) for row in self.rows if row["status"] != "PASS"]


def create_views(
    con: duckdb.DuckDBPyConnection,
    *,
    candidate: Path,
    raw: Path,
    prediction: Path,
) -> None:
    candidate_sql = sql_path(candidate)
    raw_sql = sql_path(raw)
    prediction_sql = sql_path(prediction)
    con.execute(f"""
        CREATE TEMP VIEW candidate_source AS
        SELECT
            cancer_id,
            lncrna_id,
            pathway_id,
            pathway_family_id,
            upper(trim(cancer_id)) AS norm_cancer,
            CASE
                WHEN starts_with(upper(trim(lncrna_id)), 'LNC:')
                    THEN substr(upper(trim(lncrna_id)), 5)
                ELSE upper(trim(lncrna_id))
            END AS norm_lncrna,
            upper(trim(pathway_id)) AS norm_pathway
        FROM read_parquet('{candidate_sql}')
    """)
    con.execute(f"""
        CREATE TEMP VIEW raw_source AS
        SELECT
            *,
            upper(trim(cancer_id)) AS norm_cancer,
            CASE
                WHEN starts_with(upper(trim(lncrna_id)), 'LNC:')
                    THEN substr(upper(trim(lncrna_id)), 5)
                ELSE upper(trim(lncrna_id))
            END AS norm_lncrna,
            upper(trim(pathway_id)) AS norm_pathway
        FROM read_parquet('{raw_sql}')
    """)
    con.execute(f"""
        CREATE TEMP VIEW adapter_output AS
        SELECT * FROM read_parquet('{prediction_sql}')
    """)
    # Independent reconstruction: candidate authority supplies emitted identifiers;
    # raw Evidence supplies every payload value.  No adapter implementation is used.
    con.execute("""
        CREATE TEMP VIEW independent_expected AS
        SELECT
            c.cancer_id,
            c.lncrna_id,
            c.pathway_id,
            r.evidence_confidence_probability,
            r.availability,
            r.direction,
            r.uncertainty,
            r.unavailable_reason,
            r.event_count,
            r.evidence_fold,
            r.failure_reason,
            r.analysis_version,
            r.training_run_id,
            r.changes_primary_ranking,
            r.main_ranking_modified,
            (r.lncrna_id IS DISTINCT FROM c.lncrna_id) AS lncrna_prefix_restored,
            (r.pathway_id IS DISTINCT FROM c.pathway_id) AS pathway_spelling_normalized,
            true::BOOLEAN AS canonical_candidate_key_verified,
            true::BOOLEAN AS direct_target_evidence,
            true::BOOLEAN AS confidence_only,
            false::BOOLEAN AS affects_discovery,
            false::BOOLEAN AS family_to_exact_broadcast
        FROM candidate_source c
        INNER JOIN raw_source r
          ON c.norm_cancer = r.norm_cancer
         AND c.norm_lncrna = r.norm_lncrna
         AND c.norm_pathway = r.norm_pathway
    """)


def audit_rows(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    key_expr = "struct_pack(cancer_id := cancer_id, lncrna_id := lncrna_id, pathway_id := pathway_id)"
    norm_key_expr = (
        "struct_pack(cancer_id := norm_cancer, lncrna_id := norm_lncrna, "
        "pathway_id := norm_pathway)"
    )
    source_counts: dict[str, Any] = {}
    for name, view, expr in (
        ("candidate", "candidate_source", key_expr),
        ("raw", "raw_source", key_expr),
        ("adapter", "adapter_output", key_expr),
    ):
        row = con.execute(f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT {expr}) AS distinct_exact_keys,
                count(*) FILTER (
                    WHERE cancer_id IS NULL OR lncrna_id IS NULL OR pathway_id IS NULL
                ) AS null_key_rows
            FROM {view}
        """).fetchone()
        source_counts[name] = {
            "rows": row[0], "distinct_exact_keys": row[1], "null_key_rows": row[2]
        }
    for name, view in (("candidate", "candidate_source"), ("raw", "raw_source")):
        source_counts[name]["distinct_normalized_keys"] = scalar(
            con, f"SELECT count(DISTINCT {norm_key_expr}) FROM {view}"
        )

    closure = con.execute("""
        SELECT
            (SELECT count(*) FROM independent_expected) AS rebuilt_rows,
            (SELECT count(*) FROM independent_expected e
             LEFT JOIN adapter_output a USING (cancer_id, lncrna_id, pathway_id)
             WHERE a.cancer_id IS NULL) AS rebuilt_minus_adapter,
            (SELECT count(*) FROM adapter_output a
             LEFT JOIN independent_expected e USING (cancer_id, lncrna_id, pathway_id)
             WHERE e.cancer_id IS NULL) AS adapter_minus_rebuilt,
            (SELECT count(*) FROM candidate_source c
             LEFT JOIN raw_source r
               ON c.norm_cancer = r.norm_cancer
              AND c.norm_lncrna = r.norm_lncrna
              AND c.norm_pathway = r.norm_pathway
             WHERE r.norm_cancer IS NULL) AS candidate_minus_raw_normalized,
            (SELECT count(*) FROM raw_source r
             LEFT JOIN candidate_source c
               ON c.norm_cancer = r.norm_cancer
              AND c.norm_lncrna = r.norm_lncrna
              AND c.norm_pathway = r.norm_pathway
             WHERE c.norm_cancer IS NULL) AS raw_minus_candidate_normalized
    """).fetchone()

    payload_columns = [
        "evidence_confidence_probability", "availability", "direction", "uncertainty",
        "unavailable_reason", "event_count", "evidence_fold", "failure_reason",
        "analysis_version", "training_run_id", "changes_primary_ranking",
        "main_ranking_modified", "lncrna_prefix_restored",
        "pathway_spelling_normalized", "canonical_candidate_key_verified",
        "direct_target_evidence", "confidence_only", "affects_discovery",
        "family_to_exact_broadcast",
    ]
    mismatch_expressions = [
        f"count(*) FILTER (WHERE e.{column} IS DISTINCT FROM a.{column}) AS {column}"
        for column in payload_columns
    ]
    any_mismatch = " OR ".join(
        f"e.{column} IS DISTINCT FROM a.{column}" for column in payload_columns
    )
    comparison = con.execute(f"""
        SELECT
            count(*) AS joined_rows,
            count(*) FILTER (WHERE {any_mismatch}) AS rowwise_any_mismatch,
            {', '.join(mismatch_expressions)}
        FROM independent_expected e
        INNER JOIN adapter_output a USING (cancer_id, lncrna_id, pathway_id)
    """).fetchone()
    column_mismatch_rows = {
        column: comparison[index + 2] for index, column in enumerate(payload_columns)
    }

    semantics = con.execute("""
        SELECT
            count(*) FILTER (WHERE availability IS TRUE) AS available_rows,
            count(*) FILTER (WHERE availability IS FALSE) AS unavailable_rows,
            count(*) FILTER (WHERE availability IS NULL) AS null_availability_rows,
            count(*) FILTER (
                WHERE availability IS FALSE
                  AND evidence_confidence_probability IS NULL
                  AND uncertainty IS NULL
                  AND unavailable_reason = 'NO_EXACT_PATHWAY_EVENT'
                  AND failure_reason = 'NO_EXACT_PATHWAY_EVENT'
            ) AS correctly_typed_unavailable_rows,
            count(*) FILTER (
                WHERE availability IS FALSE
                  AND (evidence_confidence_probability IS NOT NULL OR uncertainty IS NOT NULL)
            ) AS filled_unavailable_rows,
            count(*) FILTER (
                WHERE availability IS FALSE
                  AND (unavailable_reason IS NULL OR trim(unavailable_reason) = '')
            ) AS missing_unavailable_reason_rows,
            count(*) FILTER (
                WHERE availability IS TRUE
                  AND (evidence_confidence_probability IS NULL
                       OR evidence_confidence_probability < 0
                       OR evidence_confidence_probability > 1)
            ) AS bad_available_probability_rows,
            count(*) FILTER (
                WHERE availability IS TRUE
                  AND (uncertainty IS NULL OR uncertainty < 0 OR uncertainty > 1)
            ) AS bad_available_uncertainty_rows,
            count(*) FILTER (WHERE starts_with(lncrna_id, 'LNC:')) AS lnc_prefixed_rows,
            count(*) FILTER (WHERE lncrna_prefix_restored IS TRUE) AS prefix_restored_rows,
            count(*) FILTER (WHERE pathway_spelling_normalized IS TRUE)
                AS pathway_normalized_rows,
            count(*) FILTER (WHERE canonical_candidate_key_verified IS NOT TRUE)
                AS unverified_candidate_rows,
            count(*) FILTER (WHERE direct_target_evidence IS NOT TRUE)
                AS non_direct_target_rows,
            count(*) FILTER (WHERE confidence_only IS NOT TRUE)
                AS non_confidence_only_rows,
            count(*) FILTER (WHERE affects_discovery IS TRUE) AS affects_discovery_rows,
            count(*) FILTER (WHERE family_to_exact_broadcast IS TRUE)
                AS family_broadcast_rows,
            count(*) FILTER (WHERE changes_primary_ranking IS TRUE)
                AS changes_primary_ranking_rows,
            count(*) FILTER (WHERE main_ranking_modified IS TRUE)
                AS main_ranking_modified_rows,
            count(*) FILTER (WHERE analysis_version IS DISTINCT FROM
                'CancerLncAtlas_V3.2_FULL_MULTITASK') AS analysis_version_mismatch_rows
        FROM adapter_output
    """).fetchone()
    semantic_names = [
        "available_rows", "unavailable_rows", "null_availability_rows",
        "correctly_typed_unavailable_rows", "filled_unavailable_rows",
        "missing_unavailable_reason_rows", "bad_available_probability_rows",
        "bad_available_uncertainty_rows", "lnc_prefixed_rows", "prefix_restored_rows",
        "pathway_normalized_rows", "unverified_candidate_rows", "non_direct_target_rows",
        "non_confidence_only_rows", "affects_discovery_rows", "family_broadcast_rows",
        "changes_primary_ranking_rows", "main_ranking_modified_rows",
        "analysis_version_mismatch_rows",
    ]
    semantic_counts = dict(zip(semantic_names, semantics))

    normalization = con.execute("""
        SELECT
            count(*) FILTER (WHERE r.lncrna_id IS DISTINCT FROM c.lncrna_id)
                AS independently_rebuilt_prefix_rows,
            count(*) FILTER (WHERE r.pathway_id IS DISTINCT FROM c.pathway_id)
                AS independently_rebuilt_pathway_rows,
            count(*) FILTER (WHERE r.cancer_id IS DISTINCT FROM c.cancer_id)
                AS independently_rebuilt_cancer_rows,
            count(*) FILTER (WHERE starts_with(r.lncrna_id, 'LNC:'))
                AS raw_prefixed_rows,
            count(*) FILTER (WHERE NOT starts_with(c.lncrna_id, 'LNC:'))
                AS candidate_unprefixed_rows
        FROM candidate_source c
        INNER JOIN raw_source r
          ON c.norm_cancer = r.norm_cancer
         AND c.norm_lncrna = r.norm_lncrna
         AND c.norm_pathway = r.norm_pathway
    """).fetchone()
    normalization_names = [
        "independently_rebuilt_prefix_rows", "independently_rebuilt_pathway_rows",
        "independently_rebuilt_cancer_rows", "raw_prefixed_rows",
        "candidate_unprefixed_rows",
    ]
    return {
        "sources": source_counts,
        "normalized_closure": {
            "rebuilt_rows": closure[0],
            "rebuilt_minus_adapter": closure[1],
            "adapter_minus_rebuilt": closure[2],
            "candidate_minus_raw_normalized": closure[3],
            "raw_minus_candidate_normalized": closure[4],
        },
        "normalization": dict(zip(normalization_names, normalization)),
        "rowwise_comparison": {
            "joined_rows": comparison[0],
            "rowwise_reconstruction_difference": comparison[1],
            "column_mismatch_rows": column_mismatch_rows,
        },
        "typed_null_and_routing": semantic_counts,
    }


def add_row_checks(checks: Checks, audit: dict[str, Any]) -> None:
    for source in ("candidate", "raw", "adapter"):
        counts = audit["sources"][source]
        checks.equal(f"{source}_rows", counts["rows"], EXPECTED_ROWS)
        checks.equal(
            f"{source}_distinct_exact_keys", counts["distinct_exact_keys"], EXPECTED_ROWS
        )
        checks.equal(f"{source}_null_key_rows", counts["null_key_rows"], 0)
    for source in ("candidate", "raw"):
        checks.equal(
            f"{source}_distinct_normalized_keys",
            audit["sources"][source]["distinct_normalized_keys"], EXPECTED_ROWS,
        )
    closure = audit["normalized_closure"]
    checks.equal("independent_rebuild_rows", closure["rebuilt_rows"], EXPECTED_ROWS)
    for field in (
        "rebuilt_minus_adapter", "adapter_minus_rebuilt",
        "candidate_minus_raw_normalized", "raw_minus_candidate_normalized",
    ):
        checks.equal(field, closure[field], 0)
    comparison = audit["rowwise_comparison"]
    checks.equal("rowwise_joined_rows", comparison["joined_rows"], EXPECTED_ROWS)
    checks.equal(
        "rowwise_reconstruction_difference",
        comparison["rowwise_reconstruction_difference"], 0,
    )
    for column, count in comparison["column_mismatch_rows"].items():
        checks.equal(f"column_mismatch:{column}", count, 0)
    normalization = audit["normalization"]
    checks.equal(
        "independently_rebuilt_prefix_rows",
        normalization["independently_rebuilt_prefix_rows"], EXPECTED_PREFIX_RESTORED,
    )
    checks.equal(
        "independently_rebuilt_pathway_rows",
        normalization["independently_rebuilt_pathway_rows"], EXPECTED_PATHWAY_NORMALIZED,
    )
    checks.equal(
        "independently_rebuilt_cancer_rows",
        normalization["independently_rebuilt_cancer_rows"], 0,
    )
    checks.equal("raw_prefixed_rows", normalization["raw_prefixed_rows"], 0)
    checks.equal(
        "candidate_unprefixed_rows", normalization["candidate_unprefixed_rows"], 0
    )
    semantic = audit["typed_null_and_routing"]
    expected = {
        "available_rows": EXPECTED_AVAILABLE,
        "unavailable_rows": EXPECTED_UNAVAILABLE,
        "null_availability_rows": 0,
        "correctly_typed_unavailable_rows": EXPECTED_UNAVAILABLE,
        "filled_unavailable_rows": 0,
        "missing_unavailable_reason_rows": 0,
        "bad_available_probability_rows": 0,
        "bad_available_uncertainty_rows": 0,
        "lnc_prefixed_rows": EXPECTED_ROWS,
        "prefix_restored_rows": EXPECTED_PREFIX_RESTORED,
        "pathway_normalized_rows": EXPECTED_PATHWAY_NORMALIZED,
        "unverified_candidate_rows": 0,
        "non_direct_target_rows": 0,
        "non_confidence_only_rows": 0,
        "affects_discovery_rows": 0,
        "family_broadcast_rows": 0,
        "changes_primary_ranking_rows": 0,
        "main_ranking_modified_rows": 0,
        "analysis_version_mismatch_rows": 0,
    }
    for field, value in expected.items():
        checks.equal(field, semantic[field], value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    adapter_binding_path = args.adapter_binding.resolve()
    if file_sha256(adapter_binding_path) != EXPECTED_ADAPTER_BINDING_SHA256:
        raise RuntimeError("Adapter binding hash changed before independent audit")
    adapter_binding = read_json(adapter_binding_path)
    candidate_path, candidate_bound_sha = bound_path(
        adapter_binding.get("candidate_authority"), "candidate authority"
    )
    raw_path, raw_bound_sha = bound_path(
        adapter_binding.get("raw_evidence_prediction"), "raw Evidence prediction"
    )
    prediction_path, prediction_bound_sha = bound_path(
        adapter_binding.get("prediction"), "adapter prediction"
    )
    wrapper_path, wrapper_bound_sha = bound_path(
        adapter_binding.get("semantic_wrapper"), "semantic wrapper"
    )
    r2_binding_path, r2_bound_sha = bound_path(
        adapter_binding.get("r2_evidence_binding"), "R2 Evidence binding"
    )
    earlier_audit_path, earlier_audit_bound_sha = bound_path(
        adapter_binding.get("independent_post_audit"), "earlier Evidence post-audit"
    )
    adapter_root = adapter_binding_path.parent
    adapter_directory_before, adapter_files_before = directory_sha256(adapter_root)

    wrapper = read_json(wrapper_path)
    r2_binding = read_json(r2_binding_path)
    earlier_audit = read_json(earlier_audit_path)
    primary_lineage_path = args.primary_lineage.resolve()
    primary_lineage = read_json(primary_lineage_path)
    primary_prediction_path = Path(str(primary_lineage["prediction_path"]))

    input_hashes = {
        "adapter_binding": file_sha256(adapter_binding_path),
        "candidate_authority": file_sha256(candidate_path),
        "raw_evidence_prediction": file_sha256(raw_path),
        "prediction": file_sha256(prediction_path),
        "semantic_wrapper": file_sha256(wrapper_path),
        "r2_evidence_binding": file_sha256(r2_binding_path),
        "earlier_independent_post_audit": file_sha256(earlier_audit_path),
        "primary_lineage": file_sha256(primary_lineage_path),
        "primary_prediction": file_sha256(primary_prediction_path),
    }
    checks = Checks()
    checks.equal(
        "adapter_binding_sha256", input_hashes["adapter_binding"],
        EXPECTED_ADAPTER_BINDING_SHA256,
    )
    checks.equal("candidate_authority_sha256", input_hashes["candidate_authority"], candidate_bound_sha)
    checks.equal("raw_evidence_prediction_sha256", input_hashes["raw_evidence_prediction"], raw_bound_sha)
    checks.equal("adapter_prediction_sha256", input_hashes["prediction"], prediction_bound_sha)
    checks.equal("semantic_wrapper_sha256", input_hashes["semantic_wrapper"], wrapper_bound_sha)
    checks.equal("r2_evidence_binding_sha256", input_hashes["r2_evidence_binding"], r2_bound_sha)
    checks.equal(
        "earlier_independent_post_audit_sha256",
        input_hashes["earlier_independent_post_audit"], earlier_audit_bound_sha,
    )
    checks.equal("analysis_version", adapter_binding.get("analysis_version"), ANALYSIS_VERSION)
    checks.equal("adapter_binding_status", adapter_binding.get("status"), "PASS_HASH_BOUND_CANONICAL_FUSION_INPUT")
    checks.equal("adapter_binding_fusion_input_eligible", adapter_binding.get("fusion_input_eligible"), True)
    checks.equal("adapter_binding_confidence_only", adapter_binding.get("confidence_only"), True)
    checks.equal("adapter_binding_affects_discovery", adapter_binding.get("affects_discovery"), False)
    checks.equal("adapter_binding_family_broadcast", adapter_binding.get("family_to_exact_broadcast"), False)
    checks.equal("adapter_binding_changes_primary", adapter_binding.get("changes_primary_ranking"), False)
    checks.equal("adapter_binding_raw_direct_fusion", adapter_binding.get("raw_prediction_direct_fusion_allowed"), False)
    checks.equal("wrapper_status", wrapper.get("status"), "PASS_HASH_BOUND_EVIDENCE_SEMANTICS")
    checks.equal("wrapper_confidence_only", wrapper.get("confidence_only"), True)
    checks.equal("wrapper_affects_discovery", wrapper.get("affects_discovery"), False)
    checks.equal("wrapper_changes_primary", wrapper.get("changes_primary_ranking"), False)
    checks.equal("wrapper_raw_direct_fusion", wrapper.get("raw_prediction_direct_fusion_allowed"), False)
    checks.equal("r2_binding_status", r2_binding.get("status"), "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND")
    checks.equal("r2_binding_family_broadcast", r2_binding.get("family_to_exact_broadcast"), False)
    checks.equal("r2_binding_historical_checkpoints", r2_binding.get("historical_checkpoints_used"), False)
    checks.equal("r2_binding_historical_predictions", r2_binding.get("historical_predictions_used"), False)
    checks.equal("earlier_audit_status", earlier_audit.get("status"), "PASS")
    checks.equal("primary_prediction_declared_sha256", primary_lineage.get("prediction_sha256"), EXPECTED_PRIMARY_SHA256)
    checks.equal("primary_prediction_observed_sha256", input_hashes["primary_prediction"], EXPECTED_PRIMARY_SHA256)
    checks.equal("primary_prediction_rows", primary_lineage.get("prediction_rows"), EXPECTED_ROWS)

    chain_audits = {
        "adapter_binding_embedded_files": verify_embedded_file_records(adapter_binding),
        "semantic_wrapper_embedded_files": verify_embedded_file_records(wrapper),
        "r2_evidence_binding_embedded_files": verify_embedded_file_records(r2_binding),
        "earlier_independent_post_audit_embedded_files": verify_embedded_file_records(earlier_audit),
        "primary_lineage_embedded_files": verify_embedded_file_records(primary_lineage),
    }
    for name, value in chain_audits.items():
        checks.equal(f"sha_chain:{name}", value["status"], "PASS")

    con = duckdb.connect()
    con.execute(f"SET threads={max(1, int(args.threads))}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    try:
        create_views(
            con, candidate=candidate_path, raw=raw_path, prediction=prediction_path
        )
        row_audit = audit_rows(con)
        add_row_checks(checks, row_audit)
        event_lineage_path = Path(str(r2_binding["artifacts"]["event_lineage"]["path"]))
        event_sql = sql_path(event_lineage_path)
        lineage = con.execute(f"""
            SELECT
                count(*) AS rows,
                count(*) FILTER (WHERE family_broadcast_used IS TRUE)
                    AS family_broadcast_rows,
                count(*) FILTER (WHERE is_model_prediction IS TRUE)
                    AS model_prediction_rows
            FROM read_parquet('{event_sql}')
        """).fetchone()
        lineage_audit = {
            "rows": lineage[0],
            "family_broadcast_rows": lineage[1],
            "model_prediction_rows": lineage[2],
        }
        checks.equal("event_lineage_rows", lineage_audit["rows"], 12_887_868)
        checks.equal("event_lineage_family_broadcast_rows", lineage_audit["family_broadcast_rows"], 0)
        checks.equal("event_lineage_model_prediction_rows", lineage_audit["model_prediction_rows"], 0)
    finally:
        con.close()

    adapter_directory_after, adapter_files_after = directory_sha256(adapter_root)
    checks.equal("adapter_directory_read_only", adapter_directory_after, adapter_directory_before)
    checks.equal("adapter_files_read_only", adapter_files_after, adapter_files_before)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    audit_code_path = Path(__file__).resolve()
    report_path = output_root / "EVIDENCE_FUSION_POST_AUDIT.json"
    binding_path = output_root / "EVIDENCE_FUSION_POST_AUDIT_BINDING.json"
    success_path = output_root / "SUCCESS.json"

    report = {
        "format": AUDIT_FORMAT,
        "status": "PASS" if not checks.failures else "FAIL",
        "analysis_version": ANALYSIS_VERSION,
        "audit_independence": {
            "adapter_implementation_imported": False,
            "adapter_implementation_called": False,
            "reconstruction_source": "EXACT_CANDIDATE_AUTHORITY_PLUS_RAW_EVIDENCE",
            "comparison": "FULL_3_3M_NULL_SAFE_KEY_AND_COLUMN_COMPARISON",
            "audited_materialization_modified": False,
        },
        "dependencies": {
            "adapter_binding": {"path": str(adapter_binding_path), "sha256": input_hashes["adapter_binding"]},
            "prediction": {"path": str(prediction_path.resolve()), "sha256": input_hashes["prediction"]},
            "candidate_authority": {"path": str(candidate_path.resolve()), "sha256": input_hashes["candidate_authority"]},
            "raw_evidence_prediction": {"path": str(raw_path.resolve()), "sha256": input_hashes["raw_evidence_prediction"]},
            "semantic_wrapper": {"path": str(wrapper_path.resolve()), "sha256": input_hashes["semantic_wrapper"]},
            "r2_evidence_binding": {"path": str(r2_binding_path.resolve()), "sha256": input_hashes["r2_evidence_binding"]},
            "earlier_independent_post_audit": {"path": str(earlier_audit_path.resolve()), "sha256": input_hashes["earlier_independent_post_audit"]},
            "primary_lineage": {"path": str(primary_lineage_path), "sha256": input_hashes["primary_lineage"]},
            "primary_prediction": {"path": str(primary_prediction_path.resolve()), "sha256": input_hashes["primary_prediction"], "rows": EXPECTED_ROWS},
        },
        "sha_chain": chain_audits,
        "materialization_read_only_proof": {
            "directory": str(adapter_root.resolve()),
            "directory_sha256_before": adapter_directory_before,
            "directory_sha256_after": adapter_directory_after,
            "files_before": adapter_files_before,
            "files_after": adapter_files_after,
        },
        "independent_reconstruction": row_audit,
        "event_lineage": lineage_audit,
        "primary_integrity": {
            "prediction_sha256_declared": primary_lineage.get("prediction_sha256"),
            "prediction_sha256_observed": input_hashes["primary_prediction"],
            "prediction_rows": primary_lineage.get("prediction_rows"),
            "adapter_changes_primary_rows": row_audit["typed_null_and_routing"]["changes_primary_ranking_rows"],
            "adapter_main_ranking_modified_rows": row_audit["typed_null_and_routing"]["main_ranking_modified_rows"],
            "adapter_affects_discovery_rows": row_audit["typed_null_and_routing"]["affects_discovery_rows"],
            "primary_ranking_unchanged": (
                input_hashes["primary_prediction"] == EXPECTED_PRIMARY_SHA256
                and row_audit["typed_null_and_routing"]["changes_primary_ranking_rows"] == 0
                and row_audit["typed_null_and_routing"]["main_ranking_modified_rows"] == 0
                and row_audit["typed_null_and_routing"]["affects_discovery_rows"] == 0
            ),
        },
        "checks": checks.rows,
        "failed_checks": checks.failures,
        "release_decision": {
            "fusion_input_eligible": not checks.failures,
            "confidence_only": True,
            "primary_ranking_unchanged": not checks.failures,
            "raw_prediction_direct_fusion_allowed": False,
            "family_to_exact_broadcast": False,
            "production_deployed": False,
            "release_ready": False,
        },
    }
    write_json(report_path, report)
    report_sha = file_sha256(report_path)
    binding = {
        "format": BINDING_FORMAT,
        "status": report["status"],
        "analysis_version": ANALYSIS_VERSION,
        "report": {"path": str(report_path), "sha256": report_sha},
        "adapter_binding": {"path": str(adapter_binding_path), "sha256": input_hashes["adapter_binding"]},
        "prediction": {"path": str(prediction_path.resolve()), "sha256": input_hashes["prediction"], "rows": EXPECTED_ROWS},
        "candidate_authority": {"path": str(candidate_path.resolve()), "sha256": input_hashes["candidate_authority"], "rows": EXPECTED_ROWS},
        "raw_evidence_prediction": {"path": str(raw_path.resolve()), "sha256": input_hashes["raw_evidence_prediction"], "rows": EXPECTED_ROWS},
        "semantic_wrapper": {"path": str(wrapper_path.resolve()), "sha256": input_hashes["semantic_wrapper"]},
        "r2_evidence_binding": {"path": str(r2_binding_path.resolve()), "sha256": input_hashes["r2_evidence_binding"]},
        "primary_exact_pathway": {"path": str(primary_prediction_path.resolve()), "sha256": input_hashes["primary_prediction"], "rows": EXPECTED_ROWS},
        "audit_code": {"path": str(audit_code_path), "sha256": file_sha256(audit_code_path)},
        "candidate_rows": EXPECTED_ROWS,
        "available_rows": row_audit["typed_null_and_routing"]["available_rows"],
        "unavailable_rows": row_audit["typed_null_and_routing"]["unavailable_rows"],
        "lncrna_prefix_restored_rows": row_audit["normalization"]["independently_rebuilt_prefix_rows"],
        "pathway_spelling_normalized_rows": row_audit["normalization"]["independently_rebuilt_pathway_rows"],
        "rowwise_reconstruction_difference": row_audit["rowwise_comparison"]["rowwise_reconstruction_difference"],
        "failed_checks": checks.failures,
        "fusion_input_eligible": not checks.failures,
        "primary_ranking_unchanged": report["primary_integrity"]["primary_ranking_unchanged"] and not checks.failures,
        "confidence_only": True,
        "affects_discovery": False,
        "family_to_exact_broadcast": False,
        "raw_prediction_direct_fusion_allowed": False,
        "production_deployed": False,
        "release_ready": False,
    }
    write_json(binding_path, binding)
    binding_sha = file_sha256(binding_path)
    success = {
        "status": report["status"],
        "binding": binding_path.name,
        "binding_sha256": binding_sha,
        "report": report_path.name,
        "report_sha256": report_sha,
        "fusion_input_eligible": binding["fusion_input_eligible"],
        "primary_ranking_unchanged": binding["primary_ranking_unchanged"],
        "rowwise_reconstruction_difference": binding["rowwise_reconstruction_difference"],
        "production_deployed": False,
        "release_ready": False,
    }
    write_json(success_path, success)
    result = {
        "status": report["status"],
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "binding_path": str(binding_path),
        "binding_sha256": binding_sha,
        "success_path": str(success_path),
        "checks": len(checks.rows),
        "failed_checks": checks.failures,
    }
    if checks.failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-binding", type=Path,
        default=ROOT / "artifacts" / "v32_evidence_fusion_adapter_20260826_r1"
        / "EVIDENCE_FUSION_BINDING.json",
    )
    parser.add_argument(
        "--primary-lineage", type=Path,
        default=ROOT / "artifacts" / "v32_full_multitask" / "exact_pathway_release_r2"
        / "MODULE_LINEAGE.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "artifacts"
        / "v32_evidence_fusion_adapter_20260826_r1_independent_post_audit",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="12GB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
