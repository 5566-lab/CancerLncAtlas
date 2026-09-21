from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .common import input_path, read_table, write_table


def _parquet_glob(path: Path) -> str:
    value = path / "**" / "*.parquet" if path.is_dir() else path
    return value.as_posix()


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and np.isfinite(value):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _column_names(connection: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    description = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).fetchdf()
    return set(description.column_name.astype(str))


def _portable_candidate_path(cfg: dict[str, Any], destination: Path) -> str:
    """Return a path that remains valid after the atomic asset-directory rename.

    Formal assets are first built below ``.assets.tmp`` and then atomically
    renamed to ``assets``.  Persisting the absolute staging path would leave a
    dead path in the release manifest, so paths are recorded relative to the
    relocatable results root instead.
    """

    return destination.relative_to(Path(cfg["_results"])).as_posix()


def build_exact_candidate_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    """Materialize measured exact-pathway candidates without a Cartesian product.

    The 33-cancer bulk association table already records the measured
    lncRNA-pathway universe.  Reusing those identities avoids fabricating
    unmeasured pairs and keeps the exact-pathway release tractable.  The
    outcome-free pan-cancer lncRNA filter is applied before any evidence join.
    """

    from .candidates import (
        build_bulk_detection,
        build_pancancer_lnc_eligibility,
        build_sc_detection,
    )

    settings = cfg["candidate_universe"]
    bulk_detection = build_bulk_detection(cfg)
    sc_detection = build_sc_detection(cfg)
    cancers = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    reference = set(map(str, cfg["analysis_cancers"].get("reference_only", [])))
    excluded = set(map(str, cfg["analysis_cancers"].get("exclude_from_training", [])))
    formal_cancers = [
        cancer
        for cancer in cancers.cancer_id.astype(str)
        if cancer not in reference | excluded
    ]
    eligibility = build_pancancer_lnc_eligibility(
        bulk_detection, formal_cancers, settings
    )
    eligibility_path = cfg["_results"] / "tables" / "pancancer_lncrna_eligibility.parquet"
    write_table(eligibility, eligibility_path)
    if settings.get("pancancer_lnc_filter", {}).get("enabled", False):
        eligible_count = int(eligibility.eligible.astype(bool).sum())
        if eligible_count == 0:
            raise RuntimeError("Pan-cancer lncRNA filter removed the exact-pathway universe")

    detection = bulk_detection.merge(
        sc_detection, on=["cancer_id", "lncrna_id"], how="outer"
    )
    detection["bulk_detection_rate"] = pd.to_numeric(
        detection.bulk_detection_rate, errors="coerce"
    ).fillna(0.0)
    detection["sc_detection_rate"] = pd.to_numeric(
        detection.sc_detection_rate, errors="coerce"
    ).fillna(0.0)
    detection = detection.merge(
        eligibility[
            [
                "lncrna_id",
                "detected_cancers",
                "formal_cancer_count",
                "filter_semantics",
                "eligible",
            ]
        ],
        on="lncrna_id",
        how="inner",
        validate="many_to_one",
    )
    detection = detection.loc[detection.eligible.astype(bool)].copy()
    detection_path = cfg["_results"] / "tables" / "exact_candidate_detection.parquet"
    write_table(detection, detection_path)

    evidence_path = cfg["_results"] / "tables" / "pair_evidence.parquet"
    family_path = cfg["_results"] / "tables" / "pathway_family_member.parquet"
    context_path = cfg["_results"] / "tables" / "exact_pathway_context.parquet"
    if not evidence_path.exists() or not family_path.exists():
        raise RuntimeError("Exact-pathway evidence and hierarchy must precede candidates")
    association_path = input_path(cfg, "bulk_lnc_pathway")
    out_root = cfg["_results"] / "tables" / "candidate_universe"
    out_root.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    evidence_columns = _column_names(connection, evidence_path)
    strong_association_expression = (
        "ev.strong_association_label"
        if "strong_association_label" in evidence_columns
        else "ev.strong_evidence_label"
    )
    label_semantics_expression = (
        "ev.label_semantics" if "label_semantics" in evidence_columns else "NULL"
    )
    context_columns = _column_names(connection, context_path) if context_path.exists() else set()
    feature_expressions = []
    for column in cfg["training"]["feature_columns"]:
        if column in {"bulk_detection_rate", "sc_detection_rate"}:
            # These are already emitted from the eligible identity table.
            continue
        if column == "state_support" and column in context_columns:
            expression = "coalesce(CAST(ctx.state_support AS DOUBLE), 0.0)"
        elif column in evidence_columns:
            # Parquet evidence columns deliberately retain their semantic
            # dtypes (for example ``bulk_available`` is BOOLEAN).  Decoder
            # features, however, are one numeric matrix.  Cast before
            # COALESCE so DuckDB never has to combine BOOLEAN and DECIMAL.
            expression = f"coalesce(CAST(ev.{column} AS DOUBLE), 0.0)"
        else:
            expression = "0.0"
        feature_expressions.append(f"{expression} AS {column}")

    lineage = dict(cfg.get("_formal_lineage", {}))
    lineage_columns = [
        f"{_sql_literal(value)} AS {column}" for column, value in sorted(lineage.items())
    ]
    sample_hash = cfg.get("_sample_universe_sha256")
    context_join = (
        f"LEFT JOIN read_parquet('{context_path.as_posix()}') ctx USING(cancer_id, pathway_id)"
        if context_path.exists()
        else ""
    )
    manifest_rows = []
    for cancer in cancers.cancer_id.astype(str):
        destination = out_root / f"cancer_id={cancer}" / "part-0.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            cached = connection.execute(
                f"SELECT count(*) FROM read_parquet('{destination.as_posix()}')"
            ).fetchone()[0]
            manifest_rows.append(
                {
                    "cancer_id": cancer,
                    "n_candidates": int(cached),
                    "status": "CACHED",
                    "path": _portable_candidate_path(cfg, destination),
                    "target_level": "exact_pathway",
                }
            )
            continue
        tier = "reference_only" if cancer in reference else "primary"
        extras = [
            *feature_expressions,
            *lineage_columns,
            f"{_sql_literal(sample_hash)} AS sample_universe_sha256",
        ]
        select_extras = ",\n          ".join(extras)
        query = f"""
        WITH measured AS (
          SELECT DISTINCT cancer_id, lncrna_id, pathway_id
          FROM read_parquet('{_parquet_glob(association_path)}', hive_partitioning=1)
          WHERE cancer_id = {_sql_literal(cancer)}
        ),
        eligible AS (
          SELECT
            m.cancer_id,
            m.lncrna_id,
            m.pathway_id,
            d.bulk_detection_rate,
            d.sc_detection_rate,
            d.detected_cancers,
            d.formal_cancer_count,
            d.filter_semantics,
            f.pathway_family_id
          FROM measured m
          INNER JOIN read_parquet('{detection_path.as_posix()}') d
            USING(cancer_id, lncrna_id)
          INNER JOIN read_parquet('{family_path.as_posix()}') f USING(pathway_id)
          LEFT JOIN read_parquet('{evidence_path.as_posix()}') observed
            USING(cancer_id, lncrna_id, pathway_id)
          WHERE d.bulk_detection_rate >= {float(settings['bulk_min_detection_rate'])}
             OR d.sc_detection_rate >= {float(settings['sc_min_detection_rate'])}
             OR observed.pair_id IS NOT NULL
        )
        SELECT
          'CAND:' || substr(
            sha256(e.cancer_id || '|' || e.lncrna_id || '|' || e.pathway_id), 1, 20
          ) AS candidate_id,
          e.cancer_id,
          e.lncrna_id,
          e.pathway_id,
          e.pathway_family_id,
          coalesce(ev.label_class, 'unlabeled') AS label_class,
          CAST(
            coalesce(CAST(ev.association_proxy_label AS BOOLEAN), FALSE)
            AS TINYINT
          ) AS association_proxy_label,
          CAST(
            coalesce(CAST({strong_association_expression} AS BOOLEAN), FALSE)
            AS TINYINT
          ) AS strong_association_label,
          CAST(
            coalesce(CAST({strong_association_expression} AS BOOLEAN), FALSE)
            AS TINYINT
          ) AS strong_evidence_label,
          CAST(
            coalesce(CAST({strong_association_expression} AS BOOLEAN), FALSE)
            AS TINYINT
          ) AS label,
          coalesce(
            {label_semantics_expression},
            'EXACT_PATHWAY_PATIENT_ASSOCIATION_V2:cancer_x_lncrna_x_pathway_id'
          ) AS label_semantics,
          coalesce(ev.sample_weight, {float(cfg['training']['unlabeled_weight'])})
            AS sample_weight,
          coalesce(ev.direction, 'unknown') AS direction,
          coalesce(ev.observed_evidence_score, 0.0) AS observed_evidence_score,
          e.bulk_detection_rate,
          e.sc_detection_rate,
          e.detected_cancers AS pancancer_detected_cancers,
          e.formal_cancer_count AS pancancer_formal_cancer_count,
          e.filter_semantics AS pancancer_filter_semantics,
          {_sql_literal(tier)} AS analysis_tier,
          'FROZEN_CANONICAL_SAMPLE_UNIVERSE' AS evaluation_policy,
          {select_extras}
        FROM eligible e
        LEFT JOIN read_parquet('{evidence_path.as_posix()}') ev
          USING(cancer_id, lncrna_id, pathway_id)
        {context_join}
        """
        connection.execute(
            f"COPY ({query}) TO '{destination.as_posix()}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        audit = connection.execute(
            f"""
            SELECT
              count(*) AS n_candidates,
              count(DISTINCT lncrna_id) AS n_lncrnas,
              count(DISTINCT pathway_id) AS n_pathways,
              sum(association_proxy_label) AS n_positive,
              sum(strong_evidence_label) AS n_strong,
              count_if(pathway_family_id IS NULL) AS missing_family
            FROM read_parquet('{destination.as_posix()}')
            """
        ).fetchone()
        if int(audit[0]) == 0 or int(audit[5]) != 0:
            raise RuntimeError(
                f"Invalid exact-pathway candidate partition {cancer}: {audit}"
            )
        manifest_rows.append(
            {
                "cancer_id": cancer,
                "n_candidates": int(audit[0]),
                "n_lncRNAs": int(audit[1]),
                "n_pathways": int(audit[2]),
                "n_positive": int(audit[3]),
                "n_strong": int(audit[4]),
                "n_pancancer_eligible_lncRNAs": int(eligibility.eligible.sum()),
                "status": "PASS",
                "path": _portable_candidate_path(cfg, destination),
                "target_level": "exact_pathway",
                "target_column": "pathway_id",
                "family_role": "auxiliary_hierarchy_context",
            }
        )
    connection.close()
    manifest = pd.DataFrame(manifest_rows)
    write_table(manifest, cfg["_results"] / "tables" / "candidate_universe_manifest.tsv")
    return manifest
