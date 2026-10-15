"""Recover completed HM450 sample batches without re-reading raw Beta files.

Legacy batches become explicit, newly validated analysis inputs. This does not
claim that their original raw-file checksums were revalidated, and cannot
authorize deletion of the original downloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import pandas as pd

from .distal_topology import methylation_probe_lineage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def read_patient_folds(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"cancer_id": str, "patient_id": str})
    columns = ["cancer_id", "patient_id", "patient_fold_id"]
    if not set(columns).issubset(frame) or frame[columns].isna().any().any():
        raise ValueError("Patient-fold authority lacks complete identifiers or folds")
    frame = frame[columns].copy()
    frame["cancer_id"] = frame.cancer_id.str.upper()
    frame["patient_id"] = frame.patient_id.str[:12]
    fold = pd.to_numeric(frame.patient_fold_id, errors="raise")
    if not fold.isin(range(5)).all():
        raise ValueError("Patient folds must be integers 0 through 4")
    frame["patient_fold_id"] = fold.astype(int)
    frame = frame.drop_duplicates()
    if frame.duplicated(["cancer_id", "patient_id"]).any():
        raise ValueError("Patient has conflicting fold assignments")
    return frame


def raw_deletion_eligible(audit: dict[str, Any]) -> bool:
    """Analysis completion and permission to discard raw inputs are distinct."""
    expected = audit.get("download_files_expected", 0)
    return (audit.get("raw_download_delete_ready") is True
            and isinstance(expected, int) and expected > 0
            and audit.get("download_files_md5_verified") == expected
            and audit.get("download_bytes_md5_verified", 0) > 0)


def recover_methylation_batches(
    batch_root: Path, selected_path: Path, probe_map_path: Path, fold_path: Path,
    output: Path, *, download_root: Path, threads: int = 8,
) -> dict[str, Any]:
    import duckdb

    if socket.gethostname() != "149":
        raise RuntimeError("Methylation recovery must execute on target host 149")
    if threads < 1:
        raise ValueError("threads must be positive")
    batch_root = batch_root.resolve(strict=True)
    paths = sorted(batch_root.glob("BATCH_*.parquet"))
    if not paths:
        raise ValueError("No completed methylation batches")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Recovery requires a new output directory")
    selected = pd.read_csv(selected_path, sep="\t", dtype=str)
    keys = ["file_id", "cancer_id", "patient_id", "sample_id"]
    if not set(keys).issubset(selected) or selected.empty or selected[keys].isna().any().any():
        raise ValueError("Selected manifest lacks complete sample identities")
    if selected.file_id.duplicated().any():
        raise ValueError("Selected manifest contains duplicate file IDs")
    selected["cancer_id"] = selected.cancer_id.str.upper()
    selected["patient_id"] = selected.patient_id.str[:12]
    probes = pd.read_parquet(probe_map_path)
    probe_keys = ["probe_id", "lncrna_id", "region_type"]
    if not set(probe_keys).issubset(probes) or probes[probe_keys].isna().any().any():
        raise ValueError("Probe map lacks complete identifiers")
    probes = probes[probe_keys].drop_duplicates()
    if probes.empty or not set(probes.region_type).issubset({"promoter", "distal"}):
        raise ValueError("Probe map is empty or contains an unknown region")
    folds = read_patient_folds(fold_path)
    lineage = methylation_probe_lineage(probe_map_path)
    sources = {"selected_files": selected_path, "probe_map": probe_map_path, "patient_folds": fold_path}
    inputs = {k: {"path": str(p.resolve()), "sha256": _sha256(p)} for k, p in sources.items()}
    authority_path = batch_root.parent / "BATCH_INPUT_AUTHORITY.json"
    if authority_path.is_file():
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        if any(authority.get(k + "_sha256") != value["sha256"] for k, value in inputs.items()):
            raise ValueError("Recovery inputs differ from the original batch aggregation authority")
    else:
        # Row identities/counts cannot prove which probes produced an average.
        # In particular, an old Data S7 aggregate cannot acquire a new topology
        # by supplying a different probe map during recovery.
        lineage = {**lineage, "predictive_evaluation_eligible": False,
                   "recovery_limitation": "ORIGINAL_BATCH_AGGREGATION_AUTHORITY_ABSENT"}
    output.mkdir(parents=True)
    _json(output / "LAUNCH_PREFLIGHT.json", {
        "target_host": "149", "observed_host": socket.gethostname(), "training_started": False,
        "stage": "RECOVER_METHYLATION_SAMPLE_BATCHES", "inputs": inputs,
        "batch_root": str(batch_root), "raw_files_read": 0,
    })
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET temp_directory=?", [str(output / "duckdb_tmp")])
    try:
        con.read_parquet([str(p) for p in paths]).create_view("batches")
        con.register("selected", selected[keys])
        con.register("probes", probes)
        con.register("folds", folds)
        columns = {r[0] for r in con.execute("DESCRIBE batches").fetchall()}
        required = keys + ["lncrna_id", "promoter_methylation_beta", "distal_methylation_beta",
                           "promoter_probe_count", "distal_probe_count"]
        if not set(required).issubset(columns):
            raise ValueError("Completed methylation batches have an incomplete schema")
        identity_errors = con.execute("""
            SELECT count(*) FROM batches b LEFT JOIN selected s USING(file_id)
            LEFT JOIN (SELECT DISTINCT lncrna_id FROM probes) p USING(lncrna_id)
            WHERE s.file_id IS NULL OR p.lncrna_id IS NULL
               OR b.cancer_id IS DISTINCT FROM s.cancer_id
               OR b.patient_id IS DISTINCT FROM s.patient_id
               OR b.sample_id IS DISTINCT FROM s.sample_id
        """).fetchone()[0]
        duplicate_rows = con.execute("""
            SELECT count(*) FROM (SELECT file_id,lncrna_id FROM batches
            GROUP BY file_id,lncrna_id HAVING count(*) > 1)
        """).fetchone()[0]
        if identity_errors or duplicate_rows:
            raise ValueError("Methylation batches contain invalid identities or duplicate file/lncRNA rows")
        for region in ["promoter", "distal"]:
            beta, count = f"{region}_methylation_beta", f"{region}_probe_count"
            invalid = con.execute(f"""
                SELECT count(*) FROM batches b LEFT JOIN (
                  SELECT lncrna_id,count(*) AS mapped_probes FROM probes
                  WHERE region_type='{region}' GROUP BY lncrna_id
                ) p USING(lncrna_id)
                WHERE ({beta} IS NOT NULL AND
                  (NOT isfinite({beta}) OR {beta} NOT BETWEEN 0 AND 1 OR
                   {count} IS NULL OR {count} <= 0 OR p.mapped_probes IS NULL))
                  OR ({count} IS NOT NULL AND
                  (NOT isfinite({count}) OR {count} < 0 OR {count} <> floor({count})
                   OR {count} > coalesce(p.mapped_probes,0)))
                  OR ({beta} IS NULL AND coalesce({count},0) > 0)
            """).fetchone()[0]
            if invalid:
                raise ValueError(f"Invalid {region} Beta/probe count in completed batches")
        records = con.execute("""
            SELECT file_id,count(*) AS output_lncrnas,
              count(promoter_methylation_beta) AS promoter_available,
              count(distal_methylation_beta) AS distal_available
            FROM batches GROUP BY file_id ORDER BY file_id
        """).fetchdf()
        missing = set(selected.file_id) - set(records.file_id)
        # New fresh runs persist processing records before final aggregation,
        # including samples with zero mapped probes. Old runs may lack these.
        records_source = batch_root.parent / "SAMPLE_PROCESSING_RECORDS.parquet"
        if missing and records_source.is_file():
            original_records = pd.read_parquet(records_source)
            empty = original_records.loc[original_records.file_id.isin(missing)].copy()
            if (empty.file_id.duplicated().any() or set(empty.file_id) != missing
                    or not (empty[["output_lncrnas", "promoter_available", "distal_available"]] == 0).all().all()):
                raise ValueError("Missing files are not supported by zero-output sample records")
            records = pd.concat([records, empty], ignore_index=True)
        elif missing:
            raise ValueError(f"Incomplete batch recovery: {len(missing)} selected files have no processing evidence")
        con.execute("""CREATE TEMP TABLE sample_level AS
            SELECT cancer_id,patient_id,sample_id,lncrna_id,
              avg(promoter_methylation_beta) AS promoter_methylation_beta,
              avg(distal_methylation_beta) AS distal_methylation_beta,
              avg(promoter_probe_count) FILTER (WHERE promoter_probe_count>0) AS promoter_probe_count,
              avg(distal_probe_count) FILTER (WHERE distal_probe_count>0) AS distal_probe_count
            FROM batches GROUP BY ALL""")
        unmatched = con.execute("""SELECT count(*) FROM (
            SELECT DISTINCT cancer_id,patient_id FROM sample_level
            EXCEPT SELECT cancer_id,patient_id FROM folds)""").fetchone()[0]
        temporary = output / ".patient_lncrna_methylation.tmp.parquet"
        con.execute(f"""COPY (
            WITH patient_level AS (
              SELECT cancer_id,patient_id,lncrna_id,
                avg(promoter_methylation_beta) AS promoter_methylation_beta,
                avg(distal_methylation_beta) AS distal_methylation_beta,
                avg(promoter_probe_count) AS promoter_probe_count,
                avg(distal_probe_count) AS distal_probe_count,
                count(promoter_methylation_beta) AS promoter_sample_count,
                count(distal_methylation_beta) AS distal_sample_count
              FROM sample_level GROUP BY ALL)
            SELECT p.cancer_id,p.patient_id,f.patient_fold_id,p.lncrna_id,
              p.promoter_methylation_beta,
              coalesce(isfinite(p.promoter_methylation_beta),false) AS promoter_methylation_beta__available,
              p.distal_methylation_beta,
              coalesce(isfinite(p.distal_methylation_beta),false) AS distal_methylation_beta__available,
              p.promoter_probe_count,p.distal_probe_count,p.promoter_sample_count,p.distal_sample_count
            FROM patient_level p JOIN folds f USING(cancer_id,patient_id)
            ORDER BY p.cancer_id,p.patient_id,p.lncrna_id
        ) TO '{_sql_path(temporary)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        stats = con.execute("""SELECT count(*) AS row_count,count(DISTINCT cancer_id) AS cancers,
            count(DISTINCT (cancer_id,patient_id)) AS patients,count(DISTINCT lncrna_id) AS lncrnas,
            count(*) FILTER(WHERE promoter_methylation_beta__available) AS promoter_rows,
            count(*) FILTER(WHERE distal_methylation_beta__available) AS distal_rows
            FROM read_parquet(?)""", [str(temporary)]).fetchone()
        if stats[0] == 0:
            raise ValueError("Recovered methylation has no rows aligned to patient folds")
        final = output / "patient_lncrna_methylation.parquet"
        os.replace(temporary, final)
    finally:
        con.close()
    records_path = output / "SAMPLE_PROCESSING_RECORDS.parquet"
    records.sort_values("file_id").to_parquet(records_path, index=False, compression="zstd")
    artifact = lambda p: {"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)}
    audit = {
        "format": "CC_HHGT_V3_2_PATIENT_LNCRNA_METHYLATION_FEATURES_V1",
        "implementation": "COMPLETED_BATCH_RECOVERY_V2", "target_host": "149",
        "status": "PASS_PATIENT_LNCRNA_METHYLATION_FEATURES", "training_started": False,
        "topology_lineage": lineage,
        **dict(zip(["rows", "cancers", "patients", "lncrnas", "promoter_available_rows", "distal_available_rows"], map(int, stats))),
        "recovered_files": len(selected), "raw_files_read_this_run": 0,
        "download_files_expected": len(selected), "download_files_md5_verified": 0,
        "download_bytes_md5_verified": 0, "raw_download_delete_ready": False,
        "raw_download_delete_target": str(download_root.resolve()),
        "raw_download_delete_condition": "RAW_CHECKSUM_VERIFICATION_AND_DISTAL_HEAD_CLOSURE_REQUIRED",
        "recovery_provenance": "EXISTING_BATCHES_NEWLY_VALIDATED_AND_HASHED_NOT_RAW_REVERIFIED",
        "sample_aggregation_policy": "MEAN_WITHIN_EXACT_SAMPLE_ID_THEN_EQUAL_SAMPLE_WEIGHT_MEAN_WITHIN_PATIENT",
        "probe_aggregation_policy": "MEAN_FINITE_BETA_WITHIN_LNCRNA_REGION",
        "probe_count_semantics": "MEAN_INFORMATIVE_PROBES_PER_FILE_THEN_SAMPLE",
        "missing_beta_assumed_zero": False, "unaligned_patients_excluded": unmatched,
        "inputs": inputs, "batch_inputs": [artifact(p) for p in paths],
        "outputs": {"patient_features": artifact(final), "sample_records": artifact(records_path),
                    "sample_batches": len(paths)},
    }
    if records_source.is_file():
        audit["recovery_sample_processing_records"] = artifact(records_source)
    if authority_path.is_file():
        audit["original_batch_input_authority"] = artifact(authority_path)
    _json(output / "AUDIT.json", audit)
    _json(output / "SUCCESS.json", {"status": "SUCCESS", "target_host": "149",
          "audit_sha256": _sha256(output / "AUDIT.json"), "training_started": False})
    return audit
