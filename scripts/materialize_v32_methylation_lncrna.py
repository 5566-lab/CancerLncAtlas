#!/usr/bin/env python3
"""Compress downloaded GDC HM450 beta files into patient-lncRNA features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_PROBE_MAP: pd.DataFrame | None = None
_PROBE_IDS: set[str] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialise_worker(probe_map_path: str) -> None:
    global _PROBE_MAP, _PROBE_IDS
    _PROBE_MAP = pd.read_parquet(probe_map_path)
    required = ["probe_id", "lncrna_id", "region_type"]
    if not set(required).issubset(_PROBE_MAP) or _PROBE_MAP[required].isna().any().any():
        raise ValueError("Incomplete probe map")
    if _PROBE_MAP.empty or not set(_PROBE_MAP.region_type).issubset({"promoter", "distal"}):
        raise ValueError("Empty probe map or unknown region")
    _PROBE_MAP["probe_id"] = _PROBE_MAP["probe_id"].astype(str)
    _PROBE_MAP = _PROBE_MAP.drop_duplicates(["probe_id", "lncrna_id", "region_type"])
    _PROBE_IDS = set(_PROBE_MAP["probe_id"])


def _process_file(task: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _PROBE_MAP is None or _PROBE_IDS is None:
        raise RuntimeError("Methylation worker was not initialised")
    source = Path(task["path"])
    beta = pd.read_csv(
        source,
        sep="\t",
        header=None,
        names=["probe_id", "beta"],
        usecols=[0, 1],
        dtype={"probe_id": str},
        na_values=["NA", "NaN", "nan", ""],
        engine="c",
    )
    if beta["probe_id"].isna().any() or beta["probe_id"].duplicated().any():
        raise ValueError(f"Missing or duplicate probe ID: {source}")
    beta["beta"] = pd.to_numeric(beta["beta"], errors="raise")
    finite = np.isfinite(beta["beta"].to_numpy(float))
    if np.isinf(beta["beta"].to_numpy(float)).any() or ((beta.loc[finite, "beta"] < 0) | (beta.loc[finite, "beta"] > 1)).any():
        raise RuntimeError(f"Beta value outside [0,1]: {source}")
    beta = beta.loc[finite & beta["probe_id"].isin(_PROBE_IDS)]
    merged = beta.merge(_PROBE_MAP, on="probe_id", how="inner", validate="many_to_many")
    grouped = (
        merged.groupby(["lncrna_id", "region_type"], observed=True, sort=True)["beta"]
        .agg(["mean", "count"])
        .reset_index()
    )
    means = grouped.pivot(index="lncrna_id", columns="region_type", values="mean")
    counts = grouped.pivot(index="lncrna_id", columns="region_type", values="count")
    result = pd.DataFrame(index=sorted(set(means.index) | set(counts.index)))
    for region in ("promoter", "distal"):
        result[f"{region}_methylation_beta"] = means.get(region, pd.Series(dtype=float))
        result[f"{region}_probe_count"] = counts.get(region, pd.Series(dtype=float))
    result = result.reset_index(names="lncrna_id")
    result.insert(0, "sample_id", task["sample_id"])
    result.insert(0, "patient_id", task["patient_id"])
    result.insert(0, "cancer_id", task["cancer_id"])
    result.insert(0, "file_id", task["file_id"])
    record = {
        "file_id": task["file_id"],
        "input_rows": int(len(beta)),
        "mapped_rows": int(len(merged)),
        "output_lncrnas": int(len(result)),
        "promoter_available": int(result["promoter_methylation_beta"].notna().sum()),
        "distal_available": int(result["distal_methylation_beta"].notna().sum()),
    }
    return result, record


def _verify_file(task: dict[str, str]) -> dict[str, Any]:
    path = Path(task["path"])
    if not path.is_file():
        return {"file_id": task["file_id"], "status": "MISSING", "path": str(path)}
    observed_size = path.stat().st_size
    expected_size = int(task["file_size"])
    if observed_size != expected_size:
        return {
            "file_id": task["file_id"],
            "status": "SIZE_MISMATCH",
            "expected_size": expected_size,
            "observed_size": observed_size,
        }
    observed_md5 = md5(path)
    if observed_md5.lower() != task["md5sum"].lower():
        return {
            "file_id": task["file_id"],
            "status": "MD5_MISMATCH",
            "expected_md5": task["md5sum"].lower(),
            "observed_md5": observed_md5.lower(),
        }
    return {"file_id": task["file_id"], "status": "PASS", "bytes": observed_size}


def sql_output_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-files", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--probe-map", type=Path, required=True)
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-batches", type=Path,
                        help="Recover complete sample_feature_batches into a new output directory")
    parser.add_argument("--target-host", choices=["149"], default="149")
    args = parser.parse_args(argv)
    if socket.gethostname() != args.target_host:
        raise RuntimeError("Methylation materialization must execute on target host 149")
    if args.resume_batches is not None:
        from cc_hhgt.v32.methylation_recovery import recover_methylation_batches
        recover_methylation_batches(
            args.resume_batches, args.selected_files, args.probe_map, args.patient_folds,
            args.output, download_root=args.download_root, threads=args.workers,
        )
        return 0

    selected_path = args.selected_files.resolve(strict=True)
    download_root = args.download_root.resolve(strict=True)
    probe_map_path = args.probe_map.resolve(strict=True)
    fold_path = args.patient_folds.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Methylation feature output reuse refused: {output}")
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")

    selected = pd.read_csv(selected_path, sep="\t", dtype=str)
    required = {"file_id", "file_name", "md5sum", "file_size", "patient_id", "cancer_id", "sample_id"}
    if missing := sorted(required - set(selected)):
        raise RuntimeError(f"Selected methylation manifest lacks: {missing}")
    if selected.empty or selected[list(required)].isna().any().any():
        raise ValueError("Selected methylation manifest is empty or contains missing identities")
    if selected["file_id"].duplicated().any():
        raise RuntimeError("Selected methylation manifest has duplicate files")
    selected["cancer_id"] = selected.cancer_id.str.upper()
    selected["patient_id"] = selected.patient_id.str[:12]
    from cc_hhgt.v32.distal_topology import methylation_probe_lineage
    from cc_hhgt.v32.methylation_recovery import read_patient_folds
    lineage = methylation_probe_lineage(probe_map_path)
    folds = read_patient_folds(fold_path)
    selected = selected.sort_values(["cancer_id", "patient_id", "sample_id", "file_id"], kind="stable").reset_index(drop=True)
    tasks: list[dict[str, str]] = []
    for row in selected.to_dict(orient="records"):
        tasks.append(
            {
                **{key: str(value) for key, value in row.items()},
                "path": str(download_root / str(row["file_id"]) / str(row["file_name"])),
            }
        )

    output.mkdir(parents=True)
    (output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps({
        "target_host": "149", "observed_host": socket.gethostname(),
        "stage": "METHYLATION_FEATURE_MATERIALIZATION", "training_started": False,
        "topology_lineage": lineage,
    }, indent=2) + "\n", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=min(8, args.workers)) as pool:
        verification = list(pool.map(_verify_file, tasks))
    failures = [record for record in verification if record["status"] != "PASS"]
    if failures:
        retry = selected.loc[selected["file_id"].isin({row["file_id"] for row in failures})]
        retry = retry.assign(state=retry.get("state", "released"))
        retry_path = download_root.parent / "RETRY_MISSING_OR_INVALID_METHYLATION.gdc_manifest.tsv"
        retry[["file_id", "file_name", "md5sum", "file_size", "state"]].rename(
            columns={"file_id": "id", "md5sum": "md5", "file_size": "size"}
        ).to_csv(retry_path, sep="\t", index=False)
        raise RuntimeError(
            f"Methylation download preflight failed for {len(failures)} files; retry manifest: {retry_path}"
        )

    batch_root = output / "sample_feature_batches"
    batch_root.mkdir()
    (output / "BATCH_INPUT_AUTHORITY.json").write_text(json.dumps({
        "probe_map_sha256": sha256(probe_map_path), "selected_files_sha256": sha256(selected_path),
        "patient_folds_sha256": sha256(fold_path), "target_host": "149",
    }, indent=2) + "\n", encoding="utf-8")
    records: list[dict[str, Any]] = []
    batch_paths: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise_worker,
        initargs=(str(probe_map_path),),
    ) as pool:
        for batch_index, start in enumerate(range(0, len(tasks), args.batch_size)):
            chunk = tasks[start : start + args.batch_size]
            results = list(pool.map(_process_file, chunk, chunksize=1))
            frames = [item[0] for item in results]
            records.extend(item[1] for item in results)
            batch = pd.concat(frames, ignore_index=True)
            batch_path = batch_root / f"BATCH_{batch_index:05d}.parquet"
            temporary = batch_root / f".BATCH_{batch_index:05d}.tmp.parquet"
            batch.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, batch_path)
            batch_paths.append(batch_path)
            print(
                json.dumps(
                    {
                        "status": "METHYLATION_SAMPLE_BATCH_COMPLETE",
                        "batch": batch_index,
                        "files_completed": min(start + len(chunk), len(tasks)),
                        "files_total": len(tasks),
                        "rows": int(len(batch)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    # Persist zero-output samples too, before any final SQL that may fail.
    records_path = output / "SAMPLE_PROCESSING_RECORDS.parquet"
    pd.DataFrame(records).sort_values("file_id", kind="stable").to_parquet(
        records_path, index=False, compression="zstd"
    )
    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='64GB'")
    con.register("folds", folds)
    batch_glob = sql_output_path(batch_root / "BATCH_*.parquet")
    final_tmp = output / ".patient_lncrna_methylation.tmp.parquet"
    final_path = output / "patient_lncrna_methylation.parquet"
    con.execute(
        f"""
        COPY (
          WITH sample_level AS (
            SELECT cancer_id, patient_id, sample_id, lncrna_id,
                   avg(promoter_methylation_beta) FILTER (WHERE isfinite(promoter_methylation_beta)) AS promoter_methylation_beta,
                   avg(distal_methylation_beta) FILTER (WHERE isfinite(distal_methylation_beta)) AS distal_methylation_beta,
                   avg(promoter_probe_count) FILTER (WHERE promoter_probe_count > 0) AS promoter_probe_count,
                   avg(distal_probe_count) FILTER (WHERE distal_probe_count > 0) AS distal_probe_count
            FROM read_parquet('{batch_glob}')
            GROUP BY ALL
          ), patient_level AS (
            SELECT cancer_id, patient_id, lncrna_id,
                   avg(promoter_methylation_beta) FILTER (WHERE isfinite(promoter_methylation_beta)) AS promoter_methylation_beta,
                   avg(distal_methylation_beta) FILTER (WHERE isfinite(distal_methylation_beta)) AS distal_methylation_beta,
                   avg(promoter_probe_count) FILTER (WHERE promoter_probe_count > 0) AS promoter_probe_count,
                   avg(distal_probe_count) FILTER (WHERE distal_probe_count > 0) AS distal_probe_count,
                   count(*) FILTER (WHERE isfinite(promoter_methylation_beta)) AS promoter_sample_count,
                   count(*) FILTER (WHERE isfinite(distal_methylation_beta)) AS distal_sample_count
            FROM sample_level
            GROUP BY ALL
          )
          SELECT p.cancer_id, p.patient_id, f.patient_fold_id, p.lncrna_id,
                 p.promoter_methylation_beta,
                 coalesce(isfinite(p.promoter_methylation_beta), false) AS promoter_methylation_beta__available,
                 p.distal_methylation_beta,
                 coalesce(isfinite(p.distal_methylation_beta), false) AS distal_methylation_beta__available,
                 p.promoter_probe_count, p.distal_probe_count,
                 p.promoter_sample_count, p.distal_sample_count
          FROM patient_level p
          JOIN folds f USING (cancer_id, patient_id)
          ORDER BY p.cancer_id, p.patient_id, p.lncrna_id
        ) TO '{sql_output_path(final_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    final_stats = con.execute(
        f"""
        SELECT count(*) AS row_count, count(DISTINCT cancer_id) AS cancers,
               count(DISTINCT (cancer_id, patient_id)) AS patients,
               count(DISTINCT lncrna_id) AS lncrnas,
               count(*) FILTER (WHERE promoter_methylation_beta__available) AS promoter_rows,
               count(*) FILTER (WHERE distal_methylation_beta__available) AS distal_rows
        FROM read_parquet('{sql_output_path(final_tmp)}')
        """
    ).fetchone()
    con.close()
    if final_stats[0] == 0:
        raise ValueError("Methylation has no rows aligned to the patient-fold authority")
    os.replace(final_tmp, final_path)

    audit = {
        "format": "CC_HHGT_V3_2_PATIENT_LNCRNA_METHYLATION_FEATURES_V1",
        "status": "PASS_PATIENT_LNCRNA_METHYLATION_FEATURES",
        "target_host": "149",
        "training_started": False,
        "topology_lineage": lineage,
        "probe_count_semantics": "MEAN_INFORMATIVE_PROBES_PER_FILE_THEN_SAMPLE",
        "download_files_expected": int(len(tasks)),
        "download_files_md5_verified": int(len(verification)),
        "download_bytes_md5_verified": int(sum(int(item["bytes"]) for item in verification)),
        "sample_aggregation_policy": "MEAN_WITHIN_EXACT_SAMPLE_ID_THEN_EQUAL_SAMPLE_WEIGHT_MEAN_WITHIN_PATIENT",
        "probe_aggregation_policy": "MEAN_FINITE_BETA_WITHIN_LNCRNA_REGION",
        "missing_beta_assumed_zero": False,
        "rows": int(final_stats[0]),
        "cancers": int(final_stats[1]),
        "patients": int(final_stats[2]),
        "lncrnas": int(final_stats[3]),
        "promoter_available_rows": int(final_stats[4]),
        "distal_available_rows": int(final_stats[5]),
        "inputs": {
            "selected_files": {"path": str(selected_path), "sha256": sha256(selected_path)},
            "probe_map": {"path": str(probe_map_path), "sha256": sha256(probe_map_path)},
            "patient_folds": {"path": str(fold_path), "sha256": sha256(fold_path)},
        },
        "outputs": {
            "patient_features": {"path": str(final_path), "sha256": sha256(final_path), "bytes": final_path.stat().st_size},
            "sample_records": {"path": str(records_path), "sha256": sha256(records_path), "bytes": records_path.stat().st_size},
            "sample_batches": len(batch_paths),
        },
        "raw_download_delete_ready": True,
        "raw_download_delete_target": str(download_root),
        "raw_download_delete_condition": "ONLY_AFTER_DISTAL_HEAD_SUCCESS_AND_INDEPENDENT_CLOSURE_AUDIT_PASS",
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
