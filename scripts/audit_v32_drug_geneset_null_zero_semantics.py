#!/usr/bin/env python3
"""Read-only audit for null/unavailable-to-zero semantic drift.

The audit intentionally performs no writes.  It inspects the public Drug
available-prediction table and the current V3.2 exact Gene Set/enrichment
tables, then emits one JSON object to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def _relation(path: Path) -> tuple[str, list[str]]:
    resolved = path.resolve(strict=True)
    files = (
        sorted(resolved.rglob("*.parquet"), key=lambda item: item.as_posix().casefold())
        if resolved.is_dir()
        else [resolved]
    )
    if not files:
        raise RuntimeError(f"No Parquet files found: {resolved}")
    placeholders = ",".join("?" for _ in files)
    return (
        f"read_parquet([{placeholders}], hive_partitioning=false)",
        [str(item) for item in files],
    )


def _one(connection: duckdb.DuckDBPyConnection, sql: str, params: list[str]) -> dict[str, Any]:
    cursor = connection.execute(sql, params)
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Audit query unexpectedly returned no row")
    return dict(zip((column[0] for column in cursor.description), row, strict=True))


def _drug(connection: duckdb.DuckDBPyConnection, root: Path) -> dict[str, Any]:
    relation, params = _relation(root)
    probability = "drug_response_association_probability"
    result = _one(
        connection,
        f"""
        SELECT
          count(*) AS rows,
          count(DISTINCT (cancer_id, lncrna_id, drug_id)) AS unique_keys,
          count(*) FILTER (WHERE availability IS NOT TRUE) AS unavailable_rows,
          count(*) FILTER (WHERE {probability} IS NULL) AS null_probability_rows,
          count(*) FILTER (WHERE {probability} = 0) AS exact_zero_probability_rows,
          count(*) FILTER (
            WHERE availability IS NOT TRUE AND {probability} = 0
          ) AS unavailable_filled_zero_rows,
          count(*) FILTER (
            WHERE availability IS TRUE AND {probability} IS NULL
          ) AS available_null_probability_rows,
          count(*) FILTER (
            WHERE availability IS NOT TRUE AND {probability} IS NOT NULL
          ) AS unavailable_nonnull_probability_rows,
          count(*) FILTER (
            WHERE {probability} IS NOT NULL
              AND (NOT isfinite(CAST({probability} AS DOUBLE))
                   OR CAST({probability} AS DOUBLE) NOT BETWEEN 0 AND 1)
          ) AS invalid_nonnull_probability_rows
        FROM {relation}
        """,
        params,
    )
    result["unavailable_or_null_filled_as_zero_detected"] = bool(
        result["unavailable_filled_zero_rows"]
    )
    result["semantic_gate_passed"] = all(
        int(result[key]) == 0
        for key in (
            "unavailable_rows",
            "null_probability_rows",
            "unavailable_filled_zero_rows",
            "available_null_probability_rows",
            "unavailable_nonnull_probability_rows",
            "invalid_nonnull_probability_rows",
        )
    )
    return result


def _gene_set_exact(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> dict[str, Any]:
    relation, params = _relation(path)
    probability = "association_membership_probability"
    result = _one(
        connection,
        f"""
        SELECT
          count(*) AS rows,
          count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_keys,
          count(*) FILTER (WHERE {probability} IS NULL) AS null_probability_rows,
          count(*) FILTER (WHERE {probability} = 0) AS exact_zero_probability_rows,
          count(*) FILTER (
            WHERE {probability} IS NOT NULL
              AND (NOT isfinite(CAST({probability} AS DOUBLE))
                   OR CAST({probability} AS DOUBLE) NOT BETWEEN 0 AND 1)
          ) AS invalid_nonnull_probability_rows
        FROM {relation}
        """,
        params,
    )
    result["null_probabilities_preserved_not_zero_filled"] = bool(
        result["null_probability_rows"]
    ) and not bool(result["exact_zero_probability_rows"])
    result["semantic_gate_passed"] = not bool(
        result["invalid_nonnull_probability_rows"]
    )
    return result


def _gene_set_enrichment(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> dict[str, Any]:
    relation, params = _relation(path)
    result = _one(
        connection,
        f"""
        SELECT
          count(*) AS rows,
          count(*) FILTER (
            WHERE genomic_available_member_count = 0
          ) AS genomic_available_count_zero_rows,
          count(*) FILTER (
            WHERE genomic_available_member_count = 0
              AND mean_genomic_native_probability IS NULL
          ) AS genomic_zero_count_mean_null_rows,
          count(*) FILTER (
            WHERE genomic_available_member_count = 0
              AND mean_genomic_native_probability = 0
          ) AS genomic_unavailable_mean_filled_zero_rows,
          count(*) FILTER (
            WHERE single_cell_available_member_count = 0
          ) AS single_cell_available_count_zero_rows,
          count(*) FILTER (
            WHERE single_cell_available_member_count = 0
              AND mean_single_cell_native_probability IS NULL
          ) AS single_cell_zero_count_mean_null_rows,
          count(*) FILTER (
            WHERE single_cell_available_member_count = 0
              AND mean_single_cell_native_probability = 0
          ) AS single_cell_unavailable_mean_filled_zero_rows,
          count(*) FILTER (
            WHERE evidence_available_member_count = 0
          ) AS evidence_available_count_zero_rows,
          count(*) FILTER (
            WHERE evidence_available_member_count = 0
              AND mean_evidence_native_probability IS NULL
          ) AS evidence_zero_count_mean_null_rows,
          count(*) FILTER (
            WHERE evidence_available_member_count = 0
              AND mean_evidence_native_probability = 0
          ) AS evidence_unavailable_mean_filled_zero_rows,
          count(*) FILTER (
            WHERE physical_supported_member_count = 0
          ) AS physical_supported_count_zero_rows,
          count(*) FILTER (
            WHERE physical_supported_member_count = 0
              AND best_physical_ora_fdr IS NULL
          ) AS physical_zero_count_fdr_null_rows,
          count(*) FILTER (
            WHERE independent_support_channel_fraction = 0
          ) AS support_fraction_exact_zero_rows,
          count(*) FILTER (
            WHERE independent_support_channel_fraction IS NULL
          ) AS support_fraction_null_rows,
          count(*) FILTER (
            WHERE independent_support_available_member_channel_count = 0
          ) AS available_denominator_zero_rows,
          count(*) FILTER (
            WHERE independent_support_available_member_channel_count = 0
              AND independent_support_channel_fraction IS NULL
              AND independent_support_availability IS FALSE
              AND independent_support_unavailable_reason =
                'NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE'
          ) AS available_denominator_zero_typed_null_rows,
          count(*) FILTER (
            WHERE independent_support_available_member_channel_count = 0
              AND independent_support_channel_fraction = 0
          ) AS available_denominator_zero_filled_zero_rows,
          count(*) FILTER (
            WHERE independent_support_available_member_channel_count > 0
              AND independent_support_channel_hits = 0
              AND independent_support_channel_fraction = 0
              AND independent_support_availability IS TRUE
              AND independent_support_unavailable_reason IS NULL
          ) AS available_denominator_positive_true_zero_support_rows,
          count(*) FILTER (
            WHERE single_cell_available_member_count = 0
              AND evidence_available_member_count = 0
              AND physical_supported_member_count = 0
              AND mean_single_cell_native_probability IS NULL
              AND mean_evidence_native_probability IS NULL
              AND best_physical_ora_fdr IS NULL
              AND independent_support_channel_fraction = 0
          ) AS all_included_channels_absent_but_fraction_zero_rows
        FROM {relation}
        """,
        params,
    )
    result["probability_nulls_filled_as_zero_detected"] = any(
        int(result[key]) > 0
        for key in (
            "genomic_unavailable_mean_filled_zero_rows",
            "single_cell_unavailable_mean_filled_zero_rows",
            "evidence_unavailable_mean_filled_zero_rows",
        )
    )
    result["available_denominator_zero_filled_as_zero_detected"] = bool(
        result["available_denominator_zero_filled_zero_rows"]
    )
    result["zero_fraction_is_typed_available_no_support"] = (
        int(result["support_fraction_exact_zero_rows"])
        == int(result["available_denominator_positive_true_zero_support_rows"])
    )
    result["missingness_semantic_gate_passed"] = (
        int(result["available_denominator_zero_rows"])
        == int(result["available_denominator_zero_typed_null_rows"])
        == int(result["support_fraction_null_rows"])
        and int(result["available_denominator_zero_filled_zero_rows"]) == 0
    )
    return result


def _candidate_available_denominator(
    connection: duckdb.DuckDBPyConnection,
    member_root: Path,
    fusion_path: Path,
    physical_path: Path,
) -> dict[str, Any]:
    members, member_params = _relation(member_root)
    fusion, fusion_params = _relation(fusion_path)
    physical, physical_params = _relation(physical_path)
    params = [*member_params, *fusion_params]
    for _ in range(4):
        params.extend(member_params)
        params.extend(physical_params)
    return _one(
        connection,
        f"""
        WITH member_fusion AS (
          SELECT m.geneset_id, m.cancer_id, m.lncrna_id, m.pathway_id,
                 f.single_cell_native_available,
                 f.evidence_transformer_native_available
          FROM {members} m
          JOIN {fusion} f USING (cancer_id, lncrna_id, pathway_id)
        ), native_summary AS (
          SELECT geneset_id, count(*) member_count,
                 count(*) FILTER (WHERE single_cell_native_available)
                   single_cell_available_member_count,
                 count(*) FILTER (WHERE evidence_transformer_native_available)
                   evidence_available_member_count
          FROM member_fusion GROUP BY geneset_id
        ), physical_available_candidates AS (
          SELECT DISTINCT m.geneset_id, m.lncrna_id
          FROM {members} m JOIN {physical} p
            ON p.lncrna_id=m.lncrna_id AND p.availability IS TRUE
           AND p.cancer_scope='GLOBAL_NOT_CANCER_SPECIFIC'
          UNION
          SELECT DISTINCT m.geneset_id, m.lncrna_id
          FROM {members} m JOIN {physical} p
            ON p.lncrna_id=m.lncrna_id AND p.availability IS TRUE
           AND p.cancer_scope=m.cancer_id
        ), physical_available AS (
          SELECT geneset_id, count(*) physical_available_member_count
          FROM physical_available_candidates GROUP BY geneset_id
        ), physical_supported_candidates AS (
          SELECT DISTINCT m.geneset_id, m.lncrna_id
          FROM {members} m JOIN {physical} p
            ON p.lncrna_id=m.lncrna_id AND p.pathway_id=m.pathway_id
           AND p.availability IS TRUE
           AND p.cancer_scope='GLOBAL_NOT_CANCER_SPECIFIC'
          UNION
          SELECT DISTINCT m.geneset_id, m.lncrna_id
          FROM {members} m JOIN {physical} p
            ON p.lncrna_id=m.lncrna_id AND p.pathway_id=m.pathway_id
           AND p.availability IS TRUE AND p.cancer_scope=m.cancer_id
        ), physical_supported AS (
          SELECT geneset_id, count(*) physical_supported_member_count
          FROM physical_supported_candidates GROUP BY geneset_id
        ), summary AS (
          SELECT n.*,
                 coalesce(a.physical_available_member_count,0)
                   physical_available_member_count,
                 coalesce(s.physical_supported_member_count,0)
                   physical_supported_member_count,
                 n.single_cell_available_member_count+
                   n.evidence_available_member_count+
                   coalesce(a.physical_available_member_count,0)
                   available_denominator,
                 n.single_cell_available_member_count+
                   n.evidence_available_member_count+
                   coalesce(s.physical_supported_member_count,0) numerator
          FROM native_summary n
          LEFT JOIN physical_available a USING (geneset_id)
          LEFT JOIN physical_supported s USING (geneset_id)
        )
        SELECT count(*) AS rows,
               count(*) FILTER (WHERE numerator=0) AS old_exact_zero_rows,
               count(*) FILTER (WHERE available_denominator=0)
                 AS should_be_null_with_reason_rows,
               count(*) FILTER (WHERE available_denominator>0 AND numerator=0)
                 AS legitimate_available_denominator_zero_support_rows,
               count(*) FILTER (
                 WHERE physical_available_member_count>0 AND
                       physical_supported_member_count=0
               ) AS physical_available_without_pathway_support_rows,
               min(available_denominator) AS min_available_denominator,
               max(available_denominator) AS max_available_denominator
        FROM summary
        """,
        params,
    )


def _coverage(connection: duckdb.DuckDBPyConnection, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    cursor = connection.execute(
        """
        SELECT cancer_id, coverage_status, publishable_genesets
        FROM read_csv_auto(?, delim='\\t', header=true)
        WHERE coverage_status <> 'AVAILABLE'
        ORDER BY cancer_id
        """,
        [str(resolved)],
    )
    return {
        "typed_unavailable_rows": [
            {
                "cancer_id": str(cancer),
                "coverage_status": str(status),
                "publishable_genesets": int(count),
            }
            for cancer, status, count in cursor.fetchall()
        ],
        "zero_is_explicit_count_not_probability": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-root", type=Path)
    parser.add_argument("--gene-set-exact", type=Path, required=True)
    parser.add_argument("--gene-set-enrichment", type=Path, required=True)
    parser.add_argument("--gene-set-coverage", type=Path, required=True)
    parser.add_argument("--gene-set-members", type=Path)
    parser.add_argument("--multimodal-fusion", type=Path)
    parser.add_argument("--physical-enrichment", type=Path)
    args = parser.parse_args()

    denominator_inputs = (
        args.gene_set_members,
        args.multimodal_fusion,
        args.physical_enrichment,
    )
    if any(value is not None for value in denominator_inputs) and not all(
        value is not None for value in denominator_inputs
    ):
        parser.error(
            "--gene-set-members, --multimodal-fusion, and --physical-enrichment "
            "must be supplied together"
        )

    connection = duckdb.connect()
    connection.execute("SET threads=2")
    try:
        payload: dict[str, Any] = {
            "format": "CANCERLNCATLAS_V32_DRUG_GENESET_NULL_ZERO_READ_ONLY_AUDIT_V2",
            "read_only": True,
            "writes_performed": False,
            "drug_available_predictions": (
                _drug(connection, args.drug_root) if args.drug_root else None
            ),
            "gene_set_exact_associations": _gene_set_exact(
                connection, args.gene_set_exact
            ),
            "gene_set_enrichment": _gene_set_enrichment(
                connection, args.gene_set_enrichment
            ),
            "gene_set_coverage": _coverage(connection, args.gene_set_coverage),
            "gene_set_candidate_available_denominator": (
                _candidate_available_denominator(
                    connection,
                    args.gene_set_members,
                    args.multimodal_fusion,
                    args.physical_enrichment,
                )
                if all(value is not None for value in denominator_inputs)
                else None
            ),
        }
    finally:
        connection.close()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
