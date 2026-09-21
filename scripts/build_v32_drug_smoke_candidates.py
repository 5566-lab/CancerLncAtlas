#!/usr/bin/env python3
"""Build a tiny, outcome-free V3.2 drug-staging smoke universe.

The selected pathway must be present in all 33 cancer candidates, targetable
according to native DrugCentral/HGNC annotations, and represented by at least
one lncRNA in the native Cell Model Passports expression matrix.  No historical
drug association, prediction, ranking, or checkpoint is read.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from cc_hhgt.v32.drug_staging import _stage_drug_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-candidates", required=True, type=Path)
    parser.add_argument("--pathway-membership", required=True, type=Path)
    parser.add_argument("--cmp-expression", required=True, type=Path)
    parser.add_argument("--drugcentral-target", required=True, type=Path)
    parser.add_argument("--hgnc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke universe: {args.output}")

    raw_ids = pd.read_csv(
        args.cmp_expression,
        skiprows=[1, 2, 3],
        usecols=["Unnamed: 1"],
        dtype=str,
    )["Unnamed: 1"].dropna()
    raw_ids = raw_ids.astype(str).str.replace(r"\..*$", "", regex=True).str.upper()
    expression_ids = pd.DataFrame({"lncrna_id": "LNC:" + raw_ids}).drop_duplicates()

    with tempfile.TemporaryDirectory(prefix="v32_drug_smoke_") as temporary:
        target_path = Path(temporary) / "drug_gene_target.static.parquet"
        _, target_audit = _stage_drug_targets(
            args.drugcentral_target, args.hgnc, target_path
        )
        con = duckdb.connect()
        con.register("expression_ids", expression_ids)
        con.execute(
            f"""
            CREATE TEMP VIEW pathway_drug AS
            SELECT DISTINCT m.pathway_id, t.drug_id
            FROM read_parquet('{sql_path(args.pathway_membership)}') m
            JOIN read_parquet('{sql_path(target_path)}') t
              ON t.gene_id = CASE
                WHEN starts_with(m.gene_id, 'GENE:')
                  THEN regexp_replace(m.gene_id, '\\.[0-9]+$', '')
                ELSE 'GENE:' || regexp_replace(m.gene_id, '\\.[0-9]+$', '')
              END
            """
        )
        selected = con.execute(
            f"""
            SELECT c.pathway_id,
                   count(DISTINCT p.drug_id) AS drug_count,
                   count(DISTINCT c.cancer_id) AS cancer_count
            FROM read_parquet('{sql_path(args.exact_candidates)}') c
            JOIN pathway_drug p USING(pathway_id)
            JOIN expression_ids e USING(lncrna_id)
            GROUP BY c.pathway_id
            HAVING count(DISTINCT c.cancer_id) = 33
            ORDER BY drug_count ASC, c.pathway_id ASC
            LIMIT 1
            """
        ).fetchone()
        if selected is None:
            raise RuntimeError(
                "No native-targetable, CMP-expressed pathway spans all 33 cancers"
            )
        pathway_id, drug_count, cancer_count = selected
        safe_pathway = str(pathway_id).replace("'", "''")
        con.execute(
            f"""
            COPY (
              SELECT cancer_id, lncrna_id, pathway_id
              FROM (
                SELECT c.cancer_id, c.lncrna_id, c.pathway_id,
                       row_number() OVER (
                         PARTITION BY c.cancer_id ORDER BY c.lncrna_id
                       ) AS rn
                FROM read_parquet('{sql_path(args.exact_candidates)}') c
                JOIN expression_ids e USING(lncrna_id)
                WHERE c.pathway_id = '{safe_pathway}'
              )
              WHERE rn = 1
              ORDER BY cancer_id
            ) TO '{sql_path(args.output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        counts = con.execute(
            f"""
            SELECT count(*), count(DISTINCT cancer_id), count(DISTINCT lncrna_id)
            FROM read_parquet('{sql_path(args.output)}')
            """
        ).fetchone()
        con.close()

    print(
        {
            "pathway_id": pathway_id,
            "native_target_drugs": int(drug_count),
            "candidate_cancers": int(cancer_count),
            "output_rows": int(counts[0]),
            "output_cancers": int(counts[1]),
            "output_lncrnas": int(counts[2]),
            "drugcentral_audit": target_audit,
            "output": str(args.output.resolve()),
            "old_association_tables_read": False,
        }
    )


if __name__ == "__main__":
    main()
