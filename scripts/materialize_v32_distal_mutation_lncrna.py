#!/usr/bin/env python3
"""Materialize MC3-WES mutations overlapping DataS7-linked distal ATAC peaks.

The output is a patient x lncRNA feature table bound to the frozen five-fold
authority.  GRCh37 MC3 coordinates are never compared directly with the
GRCh38 ATAC peaks; a separately audited lifted-coordinate authority is
required.  Because MC3 is WES, zero means no *reported* event in a linked peak,
not locus-level wild-type callability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd


EXPECTED_MC3_SHA256 = "cb0caa0e3a80a24cc4ccaaacc95c690a4032cc87c8ac78519a5b33f3aa3f6b81"
EXPECTED_CHAIN_SHA256 = "5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve(strict=True)).replace("'", "''").replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc3", type=Path, required=True)
    parser.add_argument("--variant-events", type=Path, required=True)
    parser.add_argument("--lifted-coordinates", type=Path, required=True)
    parser.add_argument("--sample-crosswalk", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--peak-set", type=Path, required=True)
    parser.add_argument("--datas7-links", type=Path, required=True)
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        name: value.resolve(strict=True)
        for name, value in {
            "mc3": args.mc3,
            "variant_events": args.variant_events,
            "lifted_coordinates": args.lifted_coordinates,
            "sample_crosswalk": args.sample_crosswalk,
            "chain": args.chain,
            "peak_set": args.peak_set,
            "datas7_links": args.datas7_links,
            "patient_folds": args.patient_folds,
        }.items()
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Distal mutation output reuse is forbidden: {output}")
    if sha256(paths["mc3"]) != EXPECTED_MC3_SHA256:
        raise RuntimeError("MC3 source SHA256 differs from the pinned authority")
    if sha256(paths["chain"]) != EXPECTED_CHAIN_SHA256:
        raise RuntimeError("hg19ToHg38 chain SHA256 differs from the pinned authority")

    links = pd.read_csv(paths["datas7_links"], sep="\t", compression="gzip")
    if not {"lncrna_id", "peak_name"}.issubset(links):
        raise RuntimeError("DataS7 topology lacks lncRNA/peak identifiers")
    links = links[["lncrna_id", "peak_name"]].drop_duplicates()
    links["gene_id"] = (
        links["lncrna_id"].astype(str).str.removeprefix("LNC:").str.split(".").str[0]
    )
    peaks = pd.read_csv(paths["peak_set"], sep="\t")
    if not {"seqnames", "start", "end", "name"}.issubset(peaks):
        raise RuntimeError("ATAC peak set lacks coordinate columns")
    peaks = peaks.loc[peaks["name"].astype(str).isin(set(links["peak_name"].astype(str)))].copy()
    if peaks["name"].duplicated().any() or set(links["peak_name"]) - set(peaks["name"]):
        raise RuntimeError("DataS7 topology does not map uniquely to the ATAC peak set")
    peaks["chromosome_grch38"] = peaks["seqnames"].astype(str).str.removeprefix("chr")
    peaks["peak_start0"] = pd.to_numeric(peaks["start"], errors="raise").astype("int64") - 1
    peaks["peak_end0"] = pd.to_numeric(peaks["end"], errors="raise").astype("int64")
    peak_links = links.merge(
        peaks[["name", "chromosome_grch38", "peak_start0", "peak_end0"]],
        left_on="peak_name",
        right_on="name",
        how="inner",
        validate="many_to_one",
    )[["gene_id", "peak_name", "chromosome_grch38", "peak_start0", "peak_end0"]]
    if peak_links.empty or peak_links.duplicated().any():
        raise RuntimeError("Linked distal peak coordinate table is empty or duplicated")

    folds = pd.read_csv(paths["patient_folds"], sep="\t")
    if not {"cancer_id", "patient_id", "patient_fold_id"}.issubset(folds):
        raise RuntimeError("Frozen patient-fold authority lacks required columns")
    folds["cancer_id"] = folds["cancer_id"].astype(str).str.upper()
    folds["patient_id"] = folds["patient_id"].astype(str).str[:12]
    folds["patient_fold_id"] = pd.to_numeric(folds["patient_fold_id"], errors="raise").astype("int8")
    folds = folds[["cancer_id", "patient_id", "patient_fold_id"]].drop_duplicates()
    if folds.duplicated(["cancer_id", "patient_id"]).any() or set(folds.patient_fold_id) != set(range(5)):
        raise RuntimeError("Frozen patient-fold authority is not unique and exact folds 0..4")

    output.mkdir(parents=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='48GB'")
    con.register("peak_links_input", peak_links)
    con.register("formal_folds_input", folds)
    con.execute("CREATE TEMP TABLE peak_links AS SELECT * FROM peak_links_input")
    con.execute("CREATE TEMP TABLE formal_folds AS SELECT * FROM formal_folds_input")
    event_path = sql_path(paths["variant_events"])
    coordinate_path = sql_path(paths["lifted_coordinates"])
    crosswalk_path = sql_path(paths["sample_crosswalk"])

    event_stats = con.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT variant_id) AS variants,
               count(DISTINCT patient_id) AS patients,
               count(DISTINCT cancer_id) AS cancers,
               min(source_reference_build) AS min_build,
               max(source_reference_build) AS max_build
        FROM read_parquet('{event_path}')
        """
    ).fetchone()
    coordinate_stats = con.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT variant_id) AS variants,
               count(DISTINCT variant_id) FILTER (WHERE liftover_available) AS mapped_variants,
               (
                 SELECT count(*) FROM (
                   SELECT variant_id
                   FROM read_parquet('{coordinate_path}')
                   GROUP BY variant_id
                   HAVING count(DISTINCT row(
                     chromosome_grch38, start0_grch38, end0_grch38,
                     strand_liftover, liftover_available, failure_reason
                   )) > 1
                 ) conflicts
               ) AS conflicting_variant_ids
        FROM read_parquet('{coordinate_path}')
        """
    ).fetchone()
    if event_stats[4] != "GRCh37" or event_stats[5] != "GRCh37":
        raise RuntimeError("Variant-event source build is not uniformly GRCh37")
    if coordinate_stats[2] < 1 or coordinate_stats[3] != 0:
        raise RuntimeError("Lifted-coordinate authority is empty or has conflicting coordinates")

    # The historical materializer retained repeated identical rows when the
    # same genomic variant occurred in multiple samples.  They are harmless
    # but must be collapsed before joining to patient events; conflicting
    # coordinates for one variant_id were rejected above.
    con.execute(
        f"""
        CREATE TEMP TABLE lifted_coordinate_authority AS
        SELECT DISTINCT variant_id, chromosome_grch38, start0_grch38,
                        end0_grch38, strand_liftover, liftover_available,
                        failure_reason
        FROM read_parquet('{coordinate_path}')
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE assayed_patients AS
        SELECT f.cancer_id, f.patient_id, f.patient_fold_id,
               coalesce(bool_or(c.mutation_assay_available), false) AS mutation_assay_available,
               coalesce(bool_or(c.locus_coverage_available), false) AS locus_coverage_available
        FROM formal_folds f
        LEFT JOIN read_parquet('{crosswalk_path}') c
          ON upper(c.cancer_id) = f.cancer_id
         AND substr(c.patient_id, 1, 12) = f.patient_id
        GROUP BY ALL
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE mapped_events AS
        SELECT DISTINCT upper(e.cancer_id) AS cancer_id,
               substr(e.patient_id, 1, 12) AS patient_id,
               e.variant_id,
               cast(c.chromosome_grch38 AS varchar) AS chromosome_grch38,
               cast(c.start0_grch38 AS bigint) AS start0_grch38,
               cast(c.end0_grch38 AS bigint) AS end0_grch38
        FROM read_parquet('{event_path}') e
        JOIN lifted_coordinate_authority c USING (variant_id)
        JOIN formal_folds f
          ON upper(e.cancer_id) = f.cancer_id
         AND substr(e.patient_id, 1, 12) = f.patient_id
        WHERE c.liftover_available
          AND c.start0_grch38 IS NOT NULL
          AND c.end0_grch38 IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE site_recurrence AS
        SELECT cancer_id, variant_id,
               count(DISTINCT patient_id) AS site_patient_count
        FROM mapped_events
        GROUP BY ALL
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE overlap_events AS
        SELECT DISTINCT m.cancer_id, m.patient_id, p.gene_id, p.peak_name,
               m.variant_id, r.site_patient_count
        FROM mapped_events m
        JOIN peak_links p
          ON m.chromosome_grch38 = p.chromosome_grch38
         AND m.start0_grch38 < p.peak_end0
         AND m.end0_grch38 > p.peak_start0
        JOIN site_recurrence r USING (cancer_id, variant_id)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE positive_burden AS
        WITH variant_gene AS (
          SELECT cancer_id, patient_id, gene_id, variant_id,
                 max(site_patient_count) AS site_patient_count
          FROM overlap_events
          GROUP BY ALL
        ), peak_gene AS (
          SELECT cancer_id, patient_id, gene_id,
                 count(DISTINCT peak_name) AS distal_peak_count
          FROM overlap_events
          GROUP BY ALL
        )
        SELECT v.cancer_id, v.patient_id, v.gene_id,
               count(*) AS distal_variant_count,
               p.distal_peak_count,
               ln(1.0 + count(*)) AS distal_mutation_burden,
               sum(ln(1.0 + site_patient_count)) AS distal_mutation_recurrence_weighted,
               max(site_patient_count) AS max_site_patient_recurrence
        FROM variant_gene v
        JOIN peak_gene p USING (cancer_id, patient_id, gene_id)
        GROUP BY v.cancer_id, v.patient_id, v.gene_id, p.distal_peak_count
        """
    )

    positive_tmp = output / ".POSITIVE_DISTAL_MUTATION_EVENTS.tmp.parquet"
    positive_path = output / "POSITIVE_DISTAL_MUTATION_EVENTS.parquet"
    dense_tmp = output / ".patient_lncrna_distal_mutation.tmp.parquet"
    dense_path = output / "patient_lncrna_distal_mutation.parquet"
    assay_tmp = output / ".PATIENT_MUTATION_ASSAY.tmp.parquet"
    assay_path = output / "PATIENT_MUTATION_ASSAY.parquet"
    catalog_tmp = output / ".DISTAL_LNCRNA_CATALOG.tmp.parquet"
    catalog_path = output / "DISTAL_LNCRNA_CATALOG.parquet"
    for path in (positive_tmp, dense_tmp, assay_tmp, catalog_tmp):
        if path.exists():
            path.unlink()
    con.execute(f"COPY positive_burden TO '{sql_path_for_output(positive_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY assayed_patients TO '{sql_path_for_output(assay_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(
        f"COPY (SELECT DISTINCT gene_id FROM peak_links ORDER BY gene_id) TO '{sql_path_for_output(catalog_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"""
        COPY (
          SELECT a.cancer_id, a.patient_id, a.patient_fold_id,
                 'LNC:' || g.gene_id AS lncrna_id,
                 CASE WHEN a.mutation_assay_available
                      THEN coalesce(p.distal_variant_count, 0) END AS distal_variant_count,
                 CASE WHEN a.mutation_assay_available
                      THEN coalesce(p.distal_peak_count, 0) END AS distal_peak_count,
                 CASE WHEN a.mutation_assay_available
                      THEN coalesce(p.distal_mutation_burden, 0.0) END AS distal_mutation_burden,
                 CASE WHEN a.mutation_assay_available
                      THEN coalesce(p.distal_mutation_recurrence_weighted, 0.0) END AS distal_mutation_recurrence_weighted,
                 CASE WHEN a.mutation_assay_available
                      THEN coalesce(p.max_site_patient_recurrence, 0) END AS max_site_patient_recurrence,
                 a.mutation_assay_available AS distal_mutation_burden__available,
                 a.locus_coverage_available AS distal_mutation_locus_callable
          FROM assayed_patients a
          CROSS JOIN (SELECT DISTINCT gene_id FROM peak_links) g
          LEFT JOIN positive_burden p
            ON p.cancer_id = a.cancer_id
           AND p.patient_id = a.patient_id
           AND p.gene_id = g.gene_id
          ORDER BY a.cancer_id, a.patient_id, g.gene_id
        ) TO '{sql_path_for_output(dense_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    for temporary, final in (
        (positive_tmp, positive_path),
        (dense_tmp, dense_path),
        (assay_tmp, assay_path),
        (catalog_tmp, catalog_path),
    ):
        os.replace(temporary, final)

    output_stats = con.execute(
        f"""
        SELECT
          (SELECT count(*) FROM positive_burden) AS positive_rows,
          (SELECT count(DISTINCT patient_id) FROM positive_burden) AS positive_patients,
          (SELECT count(DISTINCT gene_id) FROM positive_burden) AS positive_lncrnas,
          (SELECT count(DISTINCT variant_id) FROM overlap_events) AS overlapping_variants,
          (SELECT count(DISTINCT peak_name) FROM overlap_events) AS mutated_linked_peaks,
          (SELECT count(*) FROM assayed_patients) AS formal_patients,
          (SELECT count(*) FILTER (WHERE mutation_assay_available) FROM assayed_patients) AS assayed_patients,
          (SELECT count(*) FILTER (WHERE locus_coverage_available) FROM assayed_patients) AS locus_callable_patients
        """
    ).fetchone()
    con.close()
    audit = {
        "format": "CC_HHGT_V3_2_DISTAL_REGULATORY_MUTATION_FEATURES_V1",
        "status": "PASS_DISTAL_MUTATION_FEATURES",
        "coordinate_contract": {
            "mc3_source_build": "GRCh37",
            "atac_peak_build": "GRCh38",
            "liftover_required": True,
            "direct_grch37_grch38_overlap_forbidden": True,
            "chain_sha256": EXPECTED_CHAIN_SHA256,
        },
        "source_policy": "MC3_WES_REPORTED_EVENTS_OVERLAPPING_DATAS7_LINKED_GRCH38_ATAC_PEAKS",
        "zero_semantics": "NO_REPORTED_MC3_WES_EVENT_IN_LINKED_ATAC_PEAKS_NOT_CONFIRMED_LOCUS_WILDTYPE",
        "locus_callability_claimed": False,
        "nearby_site_regularisation": "EVENTS_COLLAPSED_TO_LINKED_PEAK_THEN_LNCRNA;NO_PER_SITE_MODEL_COEFFICIENT",
        "recurrence_regularisation": "SUM_LOG1P_WITHIN_CANCER_EXACT_SITE_PATIENT_COUNT",
        "missing_assay_assumed_zero": False,
        "event_rows": int(event_stats[0]),
        "unique_event_variants": int(event_stats[1]),
        "event_patients": int(event_stats[2]),
        "event_cancers": int(event_stats[3]),
        "coordinate_rows": int(coordinate_stats[0]),
        "coordinate_unique_variants": int(coordinate_stats[1]),
        "coordinate_mapped_variants": int(coordinate_stats[2]),
        "coordinate_conflicting_variant_ids": int(coordinate_stats[3]),
        "identical_coordinate_duplicates_collapsed": int(coordinate_stats[0] - coordinate_stats[1]),
        "linked_peak_rows": int(len(peak_links)),
        "linked_lncrnas": int(peak_links.gene_id.nunique()),
        "positive_rows": int(output_stats[0]),
        "positive_patients": int(output_stats[1]),
        "positive_lncrnas": int(output_stats[2]),
        "overlapping_variants": int(output_stats[3]),
        "mutated_linked_peaks": int(output_stats[4]),
        "formal_patients": int(output_stats[5]),
        "assayed_patients": int(output_stats[6]),
        "locus_callable_patients": int(output_stats[7]),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (positive_path, dense_path, assay_path, catalog_path)
        },
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


def sql_path_for_output(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
