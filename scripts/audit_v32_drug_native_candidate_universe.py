#!/usr/bin/env python3
"""Preflight the native-target-derived V3.2 drug candidate universe.

This audit writes only a compact DrugCentral target mapping and a JSON report.
It counts the exact distinct ``cancer_id, lncRNA_id, drug_id`` universe in
DuckDB without materialising the candidate triples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--exact-candidates", required=True, type=Path)
    parser.add_argument("--pathway-membership", required=True, type=Path)
    parser.add_argument("--drugcentral-target", required=True, type=Path)
    parser.add_argument("--hgnc", required=True, type=Path)
    parser.add_argument("--gdsc1", required=True, type=Path)
    parser.add_argument("--gdsc2", required=True, type=Path)
    parser.add_argument("--prism-compounds", required=True, type=Path)
    parser.add_argument("--target-output", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.drug_staging import (
        ANALYSIS_VERSION,
        TCGA_CANCERS,
        _file_sha256,
        _stage_drug_targets,
        canonical_drug_id,
    )

    for destination in (args.target_output, args.output_json):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite preflight artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    _, target_audit = _stage_drug_targets(
        args.drugcentral_target.resolve(), args.hgnc.resolve(), args.target_output.resolve()
    )

    import duckdb
    import pandas as pd

    native_names: list[pd.Series] = []
    for path in (args.gdsc1, args.gdsc2):
        native_names.append(
            pd.read_csv(path, usecols=["DRUG_NAME"], dtype=str, low_memory=False)["DRUG_NAME"]
        )
    prism = pd.read_csv(args.prism_compounds, dtype=str, low_memory=False)
    prism_name_column = next(
        (column for column in ("Drug.Name", "name", "drug_name") if column in prism),
        None,
    )
    if prism_name_column is None:
        raise RuntimeError("PRISM compound list has no drug-name column")
    native_names.append(prism[prism_name_column])
    assayed_drugs = pd.DataFrame(
        {
            "drug_id": sorted(
                {
                    canonical_drug_id(name)
                    for series in native_names
                    for name in series.dropna().astype(str)
                    if canonical_drug_id(name)
                }
            )
        }
    )

    con = duckdb.connect()
    con.register("native_assayed_drugs", assayed_drugs)
    con.execute(
        f"""
        CREATE TEMP VIEW pathway_drug AS
        SELECT DISTINCT m.pathway_id, t.drug_id
        FROM read_parquet('{sql_path(args.pathway_membership)}') m
        JOIN read_parquet('{sql_path(args.target_output)}') t
          ON t.gene_id = CASE
            WHEN starts_with(m.gene_id, 'GENE:')
              THEN regexp_replace(m.gene_id, '\\.[0-9]+$', '')
            ELSE 'GENE:' || regexp_replace(m.gene_id, '\\.[0-9]+$', '')
          END
        """
    )
    con.execute(
        """
        CREATE TEMP VIEW pathway_drug_native_assayed AS
        SELECT DISTINCT p.pathway_id, p.drug_id
        FROM pathway_drug p
        JOIN native_assayed_drugs a USING(drug_id)
        """
    )

    def count_coverage(view_name: str) -> list[dict[str, object]]:
        rows = con.execute(
            f"""
            SELECT c.cancer_id,
                   count(DISTINCT (c.lncrna_id, p.drug_id)) AS candidate_rows,
                   count(DISTINCT c.lncrna_id) AS candidate_lncrnas,
                   count(DISTINCT p.drug_id) AS candidate_drugs
            FROM read_parquet('{sql_path(args.exact_candidates)}') c
            JOIN {view_name} p USING(pathway_id)
            GROUP BY c.cancer_id
            ORDER BY c.cancer_id
            """
        ).fetchall()
        return [
            {
                "cancer_id": str(cancer),
                "candidate_rows": int(count),
                "candidate_lncrnas": int(lncrnas),
                "candidate_drugs": int(drugs),
            }
            for cancer, count, lncrnas, drugs in rows
        ]

    coverage = count_coverage("pathway_drug")
    native_assayed_coverage = count_coverage("pathway_drug_native_assayed")
    pathway_drug_rows = int(
        con.execute("SELECT count(*) FROM pathway_drug").fetchone()[0]
    )
    assayed_pathway_drug_rows = int(
        con.execute("SELECT count(*) FROM pathway_drug_native_assayed").fetchone()[0]
    )
    con.close()
    observed = {row["cancer_id"] for row in coverage}
    missing = sorted(set(TCGA_CANCERS) - observed)
    native_assayed_observed = {row["cancer_id"] for row in native_assayed_coverage}
    native_assayed_missing = sorted(set(TCGA_CANCERS) - native_assayed_observed)
    payload: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "audit_type": "NATIVE_TARGET_DERIVED_CANDIDATE_COUNT_WITHOUT_MATERIALISATION",
        "status": "PASS" if not missing else "FAIL",
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
        "pathway_drug_rows": pathway_drug_rows,
        "candidate_rows": int(sum(int(row["candidate_rows"]) for row in coverage)),
        "candidate_cancers": len(observed),
        "missing_cancers": missing,
        "coverage": coverage,
        "native_assayed_drugs": int(len(assayed_drugs)),
        "native_assayed_pathway_drug_rows": assayed_pathway_drug_rows,
        "native_assayed_candidate_rows": int(
            sum(int(row["candidate_rows"]) for row in native_assayed_coverage)
        ),
        "native_assayed_candidate_cancers": len(native_assayed_observed),
        "native_assayed_missing_cancers": native_assayed_missing,
        "native_assayed_coverage": native_assayed_coverage,
        "native_assay_intersection_uses_response_values": False,
        "drugcentral": target_audit,
        "inputs": {
            "exact_candidates": {
                "path": str(args.exact_candidates.resolve()),
                "sha256": _file_sha256(args.exact_candidates.resolve()),
            },
            "pathway_membership": {
                "path": str(args.pathway_membership.resolve()),
                "sha256": _file_sha256(args.pathway_membership.resolve()),
            },
            "drugcentral_target": {
                "path": str(args.drugcentral_target.resolve()),
                "sha256": _file_sha256(args.drugcentral_target.resolve()),
            },
            "hgnc": {
                "path": str(args.hgnc.resolve()),
                "sha256": _file_sha256(args.hgnc.resolve()),
            },
            "gdsc1": {
                "path": str(args.gdsc1.resolve()),
                "sha256": _file_sha256(args.gdsc1.resolve()),
            },
            "gdsc2": {
                "path": str(args.gdsc2.resolve()),
                "sha256": _file_sha256(args.gdsc2.resolve()),
            },
            "prism_compounds": {
                "path": str(args.prism_compounds.resolve()),
                "sha256": _file_sha256(args.prism_compounds.resolve()),
            },
        },
        "target_artifact": {
            "path": str(args.target_output.resolve()),
            "sha256": _file_sha256(args.target_output.resolve()),
            "bytes": args.target_output.stat().st_size,
        },
    }
    atomic_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
