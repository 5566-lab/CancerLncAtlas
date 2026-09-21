#!/usr/bin/env python3
"""Compress downloaded GDC HM450 beta files into patient-lncRNA features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    _PROBE_MAP["probe_id"] = _PROBE_MAP["probe_id"].astype(str)
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
    beta["beta"] = pd.to_numeric(beta["beta"], errors="coerce")
    finite = np.isfinite(beta["beta"].to_numpy(float))
    if ((beta.loc[finite, "beta"] < 0) | (beta.loc[finite, "beta"] > 1)).any():
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


def main() -> int:
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-files", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--probe-map", type=Path, required=True)
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

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
    if selected["file_id"].duplicated().any() or selected["file_name"].duplicated().any():
        raise RuntimeError("Selected methylation manifest has duplicate files")
    selected = selected.sort_values(["cancer_id", "patient_id", "sample_id", "file_id"], kind="stable").reset_index(drop=True)
    tasks: list[dict[str, str]] = []
    for row in selected.to_dict(orient="records"):
        tasks.append(
            {
                **{key: str(value) for key, value in row.items()},
                "path": str(download_root / str(row["file_id"]) / str(row["file_name"])),
            }
        )

    with ThreadPoolExecutor(max_workers=min(8, args.workers)) as pool:
        verification = list(pool.map(_verify_file, tasks))
    failures = [record for record in verification if record["status"] != "PASS"]
    if failures:
        retry = selected.loc[selected["file_id"].isin({row["file_id"] for row in failures})]
        retry_path = download_root.parent / "RETRY_MISSING_OR_INVALID_METHYLATION.gdc_manifest.tsv"
        retry[["file_id", "file_name", "md5sum", "file_size", "state"]].rename(
            columns={"file_id": "id", "md5sum": "md5", "file_size": "size"}
        ).to_csv(retry_path, sep="\t", index=False)
        raise RuntimeError(
            f"Methylation download preflight failed for {len(failures)} files; retry manifest: {retry_path}"
        )

    output.mkdir(parents=True)
    batch_root = output / "sample_feature_batches"
    batch_root.mkdir()
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

    folds = pd.read_csv(fold_path, sep="\t")
    folds["cancer_id"] = folds["cancer_id"].astype(str).str.upper()
    folds["patient_id"] = folds["patient_id"].astype(str).str[:12]
    folds = folds[["cancer_id", "patient_id", "patient_fold_id"]].drop_duplicates()
    if folds.duplicated(["cancer_id", "patient_id"]).any():
        raise RuntimeError("Patient-fold authority is duplicated")
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
                   sum(coalesce(promoter_probe_count, 0)) AS promoter_probe_count,
                   sum(coalesce(distal_probe_count, 0)) AS distal_probe_count
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
        SELECT count(*) rows, count(DISTINCT cancer_id) cancers,
               count(DISTINCT patient_id) patients,
               count(DISTINCT lncrna_id) lncrnas,
               count(*) FILTER (WHERE promoter_methylation_beta__available) promoter_rows,
               count(*) FILTER (WHERE distal_methylation_beta__available) distal_rows
        FROM read_parquet('{sql_output_path(final_tmp)}')
        """
    ).fetchone()
    con.close()
    os.replace(final_tmp, final_path)

    records_path = output / "SAMPLE_PROCESSING_RECORDS.parquet"
    pd.DataFrame(records).sort_values("file_id", kind="stable").to_parquet(
        records_path, index=False, compression="zstd"
    )
    audit = {
        "format": "CC_HHGT_V3_2_PATIENT_LNCRNA_METHYLATION_FEATURES_V1",
        "status": "PASS_PATIENT_LNCRNA_METHYLATION_FEATURES",
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
