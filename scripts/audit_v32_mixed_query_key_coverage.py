#!/usr/bin/env python
"""Out-of-core audit of the V3.2 mixed-query Cartesian key universe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import duckdb


FORMAT = "CANCERLNCATLAS_V32_MIXED_QUERY_KEY_COVERAGE_V1"
SCOPE = "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN"
OBSERVED_SCOPE = "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_OBSERVED_ROWS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(args: argparse.Namespace) -> dict[str, object]:
    source = args.exact_association.resolve()
    output = args.output.resolve()
    scratch = args.scratch.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError("Exact association must be one regular Parquet file")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite key-coverage receipt: {output}")
    scratch.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {int(args.threads)}")
    connection.execute("SET temp_directory = ?", [str(scratch)])
    connection.execute("SET memory_limit = ?", [args.memory_limit])
    try:
        description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]
        ).fetchall()
        names = {str(row[0]) for row in description}
        required = {
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "association_membership_probability",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Exact association lacks audit columns: {missing}")
        availability = (
            "coalesce(try_cast(availability AS BOOLEAN), FALSE)"
            if "availability" in names
            else "TRUE"
        )
        eligible = (
            "coalesce(try_cast(eligible_for_mixed_query AS BOOLEAN), FALSE)"
            if "eligible_for_mixed_query" in names
            else "TRUE"
        )
        reason = (
            "trim(coalesce(cast(availability_reason AS VARCHAR), ''))"
            if "availability_reason" in names
            else "''"
        )
        query = f"""
            WITH normalized AS (
                SELECT
                    upper(trim(cast(cancer_id AS VARCHAR))) AS cancer_id,
                    'LNC:' || CASE
                        WHEN starts_with(
                            regexp_replace(upper(trim(cast(lncrna_id AS VARCHAR))),
                                           '^(GENE:|LNC:)', ''),
                            'ENSG'
                        ) THEN regexp_replace(
                            regexp_replace(upper(trim(cast(lncrna_id AS VARCHAR))),
                                           '^(GENE:|LNC:)', ''),
                            '\\..*$', ''
                        )
                        ELSE regexp_replace(upper(trim(cast(lncrna_id AS VARCHAR))),
                                            '^(GENE:|LNC:)', '')
                    END AS lncrna_id,
                    trim(cast(pathway_id AS VARCHAR)) AS pathway_id,
                    try_cast(association_membership_probability AS DOUBLE) AS probability,
                    {availability} AS availability,
                    {eligible} AS eligible,
                    {reason} AS availability_reason
                FROM read_parquet(?)
            )
            SELECT
                count(*)::BIGINT AS observed_rows,
                count(DISTINCT struct_pack(cancer_id := cancer_id,
                                           lncrna_id := lncrna_id))::BIGINT AS pair_count,
                count(DISTINCT pathway_id)::BIGINT AS pathway_count,
                count(DISTINCT struct_pack(cancer_id := cancer_id,
                                           lncrna_id := lncrna_id,
                                           pathway_id := pathway_id))::BIGINT AS distinct_key_count,
                count_if(NOT eligible)::BIGINT AS ineligible_count,
                count_if(cancer_id = '' OR lncrna_id = 'LNC:' OR pathway_id = '')::BIGINT
                    AS empty_key_count,
                count_if(availability AND
                         (probability IS NULL OR probability < 0 OR probability > 1))::BIGINT
                    AS invalid_available_probability_count,
                count_if(NOT availability AND probability IS NOT NULL)::BIGINT
                    AS nonnull_not_evaluated_probability_count,
                count_if(NOT availability AND availability_reason = '')::BIGINT
                    AS missing_not_evaluated_reason_count,
                count_if(availability)::BIGINT AS available_key_count
            FROM normalized
        """
        row = connection.execute(query, [str(source)]).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("DuckDB returned no key-coverage audit row")
    (
        observed,
        pair_count,
        pathway_count,
        distinct_keys,
        ineligible_count,
        empty_key_count,
        invalid_available,
        nonnull_not_evaluated,
        missing_reason,
        available_count,
    ) = map(int, row)
    if args.coverage_scope == OBSERVED_SCOPE:
        expected = distinct_keys
    else:
        expected = pair_count * pathway_count
    duplicate_count = observed - distinct_keys
    missing_count = max(expected - distinct_keys, 0)
    complete = (
        observed > 0
        and expected == observed == distinct_keys
        and duplicate_count == 0
        and missing_count == 0
        and ineligible_count == 0
        and empty_key_count == 0
        and invalid_available == 0
        and nonnull_not_evaluated == 0
        and missing_reason == 0
    )
    source_sha256 = _sha256(source)
    receipt: dict[str, object] = {
        "format": FORMAT,
        "status": "PASS" if complete else "FAIL",
        "scope": args.coverage_scope,
        "source_exact_association_path": str(source),
        "source_exact_association_sha256": source_sha256,
        "eligible_pair_authority_sha256": source_sha256,
        "eligible_pair_count": pair_count,
        "pathway_count": pathway_count,
        "expected_key_count": expected,
        "observed_key_count": observed,
        "distinct_key_count": distinct_keys,
        "missing_key_count": missing_count,
        "duplicate_key_count": duplicate_count,
        "complete_key_coverage": complete,
        "unscored_key_encoding": "NULL_WITH_TYPED_NOT_EVALUATED_REASON",
        "available_key_count": available_count,
        "not_evaluated_key_count": observed - available_count,
        "ineligible_key_count": ineligible_count,
        "empty_key_count": empty_key_count,
        "invalid_available_probability_count": invalid_available,
        "nonnull_not_evaluated_probability_count": nonnull_not_evaluated,
        "missing_not_evaluated_reason_count": missing_reason,
        "independent_of_asset_builder": True,
        "asset_builder_imported": False,
        "audit_engine": f"duckdb_{duckdb.__version__}_out_of_core_exact_distinct",
        "scratch_directory": str(scratch),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    if not complete:
        raise ValueError(f"Key coverage failed; forensic receipt written to {output}")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-association", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--coverage-scope",
        choices=(SCOPE, OBSERVED_SCOPE),
        default=SCOPE,
        help="Use Cartesian completeness or the observed candidate-row universe.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
