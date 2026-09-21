"""Bounded, restartable GDC segment CNV materialization.

The on-disk contract stores compact patient-by-candidate-entity matrices.  Raw
patient-by-all-entity long tables are never constructed.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .gdc_segment_cnv import (
    TCGA_CANCERS,
    _rename_directory_noreplace,
    selected_segment_target,
    sha256_file,
    validate_segment_download_gate,
)


FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_V1"
PARTITION_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_PARTITION_V1"
FOLD_FORMAT = "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_FOLD_V1"


class StreamingCNVError(RuntimeError):
    pass


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _patient(value: Any) -> str:
    text = str(value).strip().replace(".", "-").upper()
    return text[:12] if text.startswith("TCGA-") else text


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, sep="\t")


def _observed_process_memory_bytes() -> tuple[int, int]:
    """Return current RSS and retained high-water RSS without an optional dependency."""

    status = Path("/proc/self/status")
    if not status.is_file():
        return 0, 0
    values: dict[str, int] = {}
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            label, raw = line.split(":", 1)
            values[label] = int(raw.strip().split()[0]) * 1024
    return values.get("VmRSS", 0), values.get("VmHWM", 0)


def _enforce_observed_memory(max_memory_bytes: int, *, stage: str) -> tuple[int, int]:
    rss, hwm = _observed_process_memory_bytes()
    if max(rss, hwm) > int(max_memory_bytes):
        raise StreamingCNVError(
            "OBSERVED_MEMORY_GATE: "
            f"stage={stage} rss_bytes={rss} hwm_bytes={hwm} > {int(max_memory_bytes)}"
        )
    return rss, hwm


def implementation_hashes(repo_root: str | Path | None = None) -> dict[str, str]:
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    paths = {
        "segment_cnv_streaming.py": Path(__file__).resolve(),
        "genomic_training.py": root / "cc_hhgt/v32/genomic_training.py",
        "run_v32_streaming_segment_cnv.py": root / "scripts/run_v32_streaming_segment_cnv.py",
        "run_v32_genomic_training.py": root / "scripts/run_v32_genomic_training.py",
        "server_launch_v32_cnv_router_cpu.sh": root / "scripts/server_launch_v32_cnv_router_cpu.sh",
    }
    if missing := [name for name, path in paths.items() if not path.is_file()]:
        raise StreamingCNVError(f"Streaming implementation files missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _normalise_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required_candidates = {"cancer_id", "lncrna_id", "pathway_id"}
    if missing := required_candidates - set(candidates):
        raise StreamingCNVError(f"Candidates lack columns: {sorted(missing)}")
    c = candidates[list(required_candidates)].drop_duplicates().copy()
    c["cancer_id"] = c.cancer_id.astype(str).str.upper()
    return c


def _normalise_shared_inputs(
    folds: pd.DataFrame, intervals: pd.DataFrame, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_col = "patient_id"
    fold_col = "patient_fold_id" if "patient_fold_id" in folds else "patient_fold"
    if patient_col not in folds or fold_col not in folds or "cancer_id" not in folds:
        raise StreamingCNVError("Fold table lacks cancer/patient/fold columns")
    kind_col = "entity_type" if "entity_type" in intervals else "feature_type"
    entity_col = "entity_id" if "entity_id" in intervals else "gene_id"
    chrom_col = "chromosome" if "chromosome" in intervals else "chrom"
    if not {kind_col, entity_col, chrom_col, "start", "end"}.issubset(intervals):
        raise StreamingCNVError("Interval table lacks entity/chromosome/start/end columns")
    if not {"pathway_id", "gene_id"}.issubset(membership):
        raise StreamingCNVError("Exact membership lacks pathway_id/gene_id")
    f = folds[["cancer_id", patient_col, fold_col]].copy()
    f.columns = ["cancer_id", "patient_id", "patient_fold_id"]
    f["cancer_id"] = f.cancer_id.astype(str).str.upper()
    if f[["cancer_id", "patient_id"]].isna().any().any():
        raise StreamingCNVError("Fold table contains null cancer/patient IDs")
    f["patient_id"] = f.patient_id.astype(str).str.strip()
    if f.patient_id.eq("").any():
        raise StreamingCNVError("Fold table contains empty patient_id")
    f["patient_fold_id"] = pd.to_numeric(f.patient_fold_id, errors="raise").astype(int)
    conflicts = f.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique()
    if conflicts.gt(1).any():
        raise StreamingCNVError("One explicit patient crosses CNV folds")
    f = f.drop_duplicates(["cancer_id", "patient_id"])
    observed = f.groupby("cancer_id", observed=True).patient_fold_id.agg(
        lambda value: frozenset(map(int, value))
    )
    if observed.ne(frozenset(range(5))).any():
        raise StreamingCNVError("Fold authority is not exact five-fold in every cancer")
    iv = intervals[[kind_col, entity_col, chrom_col, "start", "end"]].copy()
    iv.columns = ["entity_type", "entity_id", "chromosome", "start", "end"]
    iv["entity_type"] = iv.entity_type.astype(str).str.lower().replace(
        {"protein_coding": "gene", "lncrna_gene": "lncrna"}
    )
    iv["chromosome"] = iv.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
    iv["start"] = pd.to_numeric(iv.start, errors="raise").astype(np.int64)
    iv["end"] = pd.to_numeric(iv.end, errors="raise").astype(np.int64)
    iv = iv.loc[iv.entity_type.isin(["gene", "lncrna"])].drop_duplicates()
    m = membership[["pathway_id", "gene_id"]].dropna().astype(str).drop_duplicates()
    return f, iv, m


def _normalise_inputs(
    candidates: pd.DataFrame, folds: pd.DataFrame, intervals: pd.DataFrame, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f, iv, m = _normalise_shared_inputs(folds, intervals, membership)
    return _normalise_candidates(candidates), f, iv, m


def _read_cancer_candidates(path: Path, cancer_id: str) -> pd.DataFrame:
    """Read one cancer only; the 3.3M-row object table must never be resident globally."""

    cancer = str(cancer_id).upper()
    columns = ["cancer_id", "lncrna_id", "pathway_id"]
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=columns, filters=[("cancer_id", "==", cancer)])
    else:
        pieces = []
        for chunk in pd.read_csv(path, sep="\t", usecols=columns, chunksize=100_000):
            local = chunk.loc[chunk.cancer_id.astype(str).str.upper().eq(cancer)]
            if not local.empty:
                pieces.append(local)
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    result = _normalise_candidates(frame)
    if not result.empty and set(result.cancer_id) != {cancer}:
        raise StreamingCNVError(f"Cancer-partitioned candidate read leaked another cancer: {cancer}")
    return result


def _manifest_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if str(row.get("selected_for_patient", "")).lower() not in {"true", "1"}:
                continue
            key = (str(row["cancer_id"]).upper(), _patient(row["case_submitter_id"]))
            if key in result:
                raise StreamingCNVError(f"Duplicate selected segment patient: {key}")
            result[key] = dict(row)
    return result


def _mapped_entities(
    segment: pd.DataFrame,
    intervals: pd.DataFrame,
    entity_ids: Sequence[str],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count_entities = len(entity_ids)
    callable_value = np.zeros(count_entities, dtype=np.uint8)
    event = np.zeros(count_entities, dtype=np.uint8)
    burden = np.full(count_entities, np.nan, dtype=np.float32)
    if not count_entities or intervals.empty or segment.empty:
        return event, callable_value, burden
    index = {value: position for position, value in enumerate(entity_ids)}
    total = np.zeros(count_entities, dtype=np.float64)
    counts = np.zeros(count_entities, dtype=np.int32)
    for chrom, local_iv in intervals.groupby("chromosome", observed=True, sort=False):
        local_seg = segment.loc[segment.chromosome.eq(str(chrom))].sort_values("start")
        if local_seg.empty:
            continue
        starts = local_seg.start.to_numpy(np.int64)
        ends = local_seg.end.to_numpy(np.int64)
        values = local_seg.value.to_numpy(float)
        midpoint = ((local_iv.start.to_numpy(np.int64) + local_iv.end.to_numpy(np.int64)) // 2)
        position = np.searchsorted(starts, midpoint, side="right") - 1
        valid = (position >= 0) & (midpoint <= ends[np.maximum(position, 0)])
        valid &= np.isfinite(values[np.maximum(position, 0)])
        if not valid.any():
            continue
        entity_position = local_iv.entity_id.astype(str).map(index).to_numpy(float)
        valid &= np.isfinite(entity_position)
        if not valid.any():
            continue
        ei = entity_position[valid].astype(int)
        observed = np.abs(values[position[valid]])
        np.add.at(total, ei, observed)
        np.add.at(counts, ei, 1)
        np.maximum.at(event, ei, observed >= float(threshold))
    callable_value[counts > 0] = 1
    burden[counts > 0] = (total[counts > 0] / counts[counts > 0]).astype(np.float32)
    return event, callable_value, burden


def _write_npy(path: Path, shape: tuple[int, int], dtype: Any) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def materialize_cancer_partition(
    *, cancer_id: str, staging_root: Path, output_root: Path,
    candidates: pd.DataFrame, folds: pd.DataFrame, intervals: pd.DataFrame,
    membership: pd.DataFrame, selected_rows: Mapping[tuple[str, str], Mapping[str, str]],
    context_sha256: str, event_threshold: float = 0.30,
    max_memory_bytes: int = 536_870_912, max_segment_rows: int = 2_000_000,
) -> dict[str, Any]:
    cancer = str(cancer_id).upper()
    final = output_root / f"cancer={cancer}"
    if final.exists():
        return validate_cancer_partition(final, context_sha256=context_sha256)
    local_candidates = candidates.loc[candidates.cancer_id.eq(cancer)]
    local_folds = folds.loc[folds.cancer_id.eq(cancer)].sort_values("patient_id")
    if local_candidates.empty or local_folds.empty:
        raise StreamingCNVError(f"{cancer} lacks candidates or folds")
    lnc_ids = tuple(sorted(local_candidates.lncrna_id.astype(str).unique()))
    pathway_ids = tuple(sorted(local_candidates.pathway_id.astype(str).unique()))
    local_membership = membership.loc[membership.pathway_id.isin(pathway_ids)].copy()
    genes = tuple(sorted(local_membership.gene_id.astype(str).unique()))
    needed = set(lnc_ids) | set(genes)
    local_intervals = intervals.loc[intervals.entity_id.astype(str).isin(needed)].copy()
    patient_ids = tuple(local_folds.patient_id.astype(str))
    matrix_bytes = len(patient_ids) * (len(lnc_ids) + len(pathway_ids)) * 6
    interval_work_bytes = len(local_intervals) * 64 + (len(genes) + len(lnc_ids)) * 32
    if interval_work_bytes > int(max_memory_bytes):
        raise StreamingCNVError(
            f"MEMORY_GATE: estimated transient bytes {interval_work_bytes} > {max_memory_bytes}"
        )
    rss, hwm = _enforce_observed_memory(max_memory_bytes, stage=f"{cancer}:before_partition")
    observed_peak_process_bytes = max(rss, hwm)
    token = context_sha256[:12]
    revision = 0
    while True:
        suffix = "" if revision == 0 else f".r{revision}"
        building = output_root / f".cancer={cancer}.building.{token}{suffix}"
        try:
            building.mkdir(parents=True)
            break
        except FileExistsError:
            revision += 1
    incomplete = building / "INCOMPLETE.json"
    state = {"status": "BUILDING", "cancer_id": cancer, "context_sha256": context_sha256}
    incomplete.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    try:
        (building / "patient_ids.json").write_text(json.dumps(patient_ids) + "\n", encoding="utf-8")
        (building / "lncrna_ids.json").write_text(json.dumps(lnc_ids) + "\n", encoding="utf-8")
        (building / "pathway_ids.json").write_text(json.dumps(pathway_ids) + "\n", encoding="utf-8")
        shape_l = (len(patient_ids), len(lnc_ids))
        shape_p = (len(patient_ids), len(pathway_ids))
        arrays = {
            "lncrna_event.npy": _write_npy(building / "lncrna_event.npy", shape_l, np.uint8),
            "lncrna_callable.npy": _write_npy(building / "lncrna_callable.npy", shape_l, np.uint8),
            "lncrna_burden.npy": _write_npy(building / "lncrna_burden.npy", shape_l, np.float32),
            "pathway_event.npy": _write_npy(building / "pathway_event.npy", shape_p, np.uint8),
            "pathway_callable.npy": _write_npy(building / "pathway_callable.npy", shape_p, np.uint8),
            "pathway_burden.npy": _write_npy(building / "pathway_burden.npy", shape_p, np.float32),
        }
        gene_index = {gene: index for index, gene in enumerate(genes)}
        pathway_members = [
            np.asarray([gene_index[g] for g in local_membership.loc[
                local_membership.pathway_id.eq(pathway), "gene_id"
            ].astype(str)], dtype=int)
            for pathway in pathway_ids
        ]
        max_observed_rows = 0
        processed = 0
        typed_unavailable = 0
        for patient_row, patient_id in enumerate(patient_ids):
            rss, hwm = _enforce_observed_memory(
                max_memory_bytes, stage=f"{cancer}:patient={patient_id}:before_read"
            )
            observed_peak_process_bytes = max(observed_peak_process_bytes, rss, hwm)
            arrays["lncrna_event.npy"][patient_row] = 0
            arrays["lncrna_callable.npy"][patient_row] = 0
            arrays["lncrna_burden.npy"][patient_row] = np.nan
            arrays["pathway_event.npy"][patient_row] = 0
            arrays["pathway_callable.npy"][patient_row] = 0
            arrays["pathway_burden.npy"][patient_row] = np.nan
            manifest_row = selected_rows.get((cancer, patient_id))
            if manifest_row is None:
                typed_unavailable += 1
                continue
            path = selected_segment_target(manifest_row, staging_root)
            if not path.is_file():
                raise StreamingCNVError(f"Staged segment disappeared: {path}")
            segment = pd.read_csv(path, sep="\t")
            max_observed_rows = max(max_observed_rows, len(segment))
            if len(segment) > int(max_segment_rows):
                raise StreamingCNVError(
                    f"ROW_GATE: {cancer}/{patient_id} has {len(segment)} rows > {max_segment_rows}"
                )
            observed_transient_bytes = interval_work_bytes + len(segment) * 64
            if observed_transient_bytes > int(max_memory_bytes):
                raise StreamingCNVError(
                    f"MEMORY_GATE: observed transient bytes {observed_transient_bytes} > {max_memory_bytes}"
                )
            columns = {str(column).lower(): column for column in segment.columns}
            try:
                segment = segment[[columns["chromosome"], columns["start"], columns["end"], columns["segment_mean"]]].copy()
            except KeyError as exc:
                raise StreamingCNVError(f"Malformed segment columns: {path}") from exc
            segment.columns = ["chromosome", "start", "end", "value"]
            segment["chromosome"] = segment.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
            segment["start"] = pd.to_numeric(segment.start, errors="raise").astype(np.int64)
            segment["end"] = pd.to_numeric(segment.end, errors="raise").astype(np.int64)
            segment["value"] = pd.to_numeric(segment.value, errors="coerce")
            gene_iv = local_intervals.loc[local_intervals.entity_type.eq("gene")]
            lnc_iv = local_intervals.loc[local_intervals.entity_type.eq("lncrna")]
            ge, gc, gb = _mapped_entities(segment, gene_iv, genes, event_threshold)
            le, lc, lb = _mapped_entities(segment, lnc_iv, lnc_ids, event_threshold)
            rss, hwm = _enforce_observed_memory(
                max_memory_bytes, stage=f"{cancer}:patient={patient_id}:after_mapping"
            )
            observed_peak_process_bytes = max(observed_peak_process_bytes, rss, hwm)
            arrays["lncrna_event.npy"][patient_row] = le
            arrays["lncrna_callable.npy"][patient_row] = lc
            arrays["lncrna_burden.npy"][patient_row] = lb
            for pathway_position, members in enumerate(pathway_members):
                if not len(members):
                    continue
                event = bool(ge[members].any())
                callable_pathway = event or bool(gc[members].all())
                if callable_pathway:
                    arrays["pathway_event.npy"][patient_row, pathway_position] = int(event)
                    arrays["pathway_callable.npy"][patient_row, pathway_position] = 1
                    arrays["pathway_burden.npy"][patient_row, pathway_position] = float(
                        np.nansum(gb[members])
                    )
            processed += 1
        for value in arrays.values():
            value.flush()
        del value
        del arrays
        fold_records = []
        for fold in range(5):
            fold_dir = building / f"fold={fold}"
            fold_dir.mkdir()
            indices = np.flatnonzero(local_folds.patient_fold_id.to_numpy(int) == fold).astype(np.int32)
            np.save(fold_dir / "test_patient_rows.npy", indices, allow_pickle=False)
            fold_payload = {
                "format": FOLD_FORMAT, "status": "SUCCESS", "cancer_id": cancer,
                "patient_fold": fold, "context_sha256": context_sha256,
                "test_patient_rows_sha256": sha256_file(fold_dir / "test_patient_rows.npy"),
                "test_patients": int(len(indices)),
            }
            (fold_dir / "SUCCESS.json").write_text(
                json.dumps(fold_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            fold_records.append(fold_payload)
        files = {}
        for path in sorted(building.rglob("*")):
            if path.is_file() and path.name not in {"INCOMPLETE.json", "SUCCESS.json"}:
                files[path.relative_to(building).as_posix()] = sha256_file(path)
        payload = {
            "format": PARTITION_FORMAT, "status": "SUCCESS", "cancer_id": cancer,
            "context_sha256": context_sha256, "patients": len(patient_ids),
            "processed_segment_patients": processed, "typed_unavailable_patients": typed_unavailable,
            "lncrnas": len(lnc_ids), "pathways": len(pathway_ids), "membership_genes": len(genes),
            "matrix_disk_bytes_estimate": matrix_bytes,
            "memory_gate_bytes": int(max_memory_bytes),
            "estimated_peak_transient_bytes": int(interval_work_bytes + max_observed_rows * 64),
            "observed_peak_process_bytes": int(observed_peak_process_bytes),
            "observed_process_memory_gate_enforced": True,
            "max_segment_rows_observed": max_observed_rows,
            "max_segment_rows_gate": int(max_segment_rows),
            "max_materialized_long_rows": 0,
            "folds": fold_records, "files": files,
        }
        (building / "SUCCESS.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        incomplete.unlink()
        _rename_directory_noreplace(building, final)
        return payload
    except Exception as exc:
        if building.exists():
            state.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
            incomplete.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def validate_cancer_partition(path: str | Path, *, context_sha256: str) -> dict[str, Any]:
    root = Path(path).resolve()
    marker = root / "SUCCESS.json"
    if not marker.is_file():
        raise StreamingCNVError(f"Partition lacks SUCCESS: {root}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("format") != PARTITION_FORMAT or payload.get("status") != "SUCCESS":
        raise StreamingCNVError(f"Invalid partition SUCCESS: {root}")
    if payload.get("context_sha256") != context_sha256:
        raise StreamingCNVError(f"Partition context drift: {root}")
    for relative, expected in payload.get("files", {}).items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise StreamingCNVError(f"Partition file drift: {target}")
    if {int(item["patient_fold"]) for item in payload.get("folds", [])} != set(range(5)):
        raise StreamingCNVError(f"Partition fold coverage incomplete: {root}")
    return payload


def materialize_streaming_store(
    *, download_complete_path: str | Path, staging_root: str | Path,
    candidates_path: str | Path, patient_folds_path: str | Path,
    entity_intervals_path: str | Path, membership_path: str | Path,
    mutation_gene_path: str | Path, mutation_lncrna_path: str | Path,
    mc3_path: str | Path, core_manifest_path: str | Path,
    output_root: str | Path, cancers: Sequence[str] = TCGA_CANCERS,
    event_threshold: float = 0.30, max_memory_bytes: int = 536_870_912,
    max_segment_rows: int = 2_000_000, repo_root: str | Path | None = None,
) -> dict[str, Any]:
    bindings = {
        "candidates": Path(candidates_path).resolve(), "patient_folds": Path(patient_folds_path).resolve(),
        "entity_intervals": Path(entity_intervals_path).resolve(), "pathway_membership": Path(membership_path).resolve(),
        "sample_gene_mutation": Path(mutation_gene_path).resolve(),
        "sample_lncrna_mutation": Path(mutation_lncrna_path).resolve(), "mc3": Path(mc3_path).resolve(),
        "core_embedding_manifest": Path(core_manifest_path).resolve(),
    }
    gate = validate_segment_download_gate(
        download_complete_path, staging_root=staging_root, bindings=bindings
    )
    implementation = implementation_hashes(repo_root)
    context = {
        "format": FORMAT, "download_gate": str(Path(download_complete_path).resolve()),
        "download_gate_sha256": sha256_file(Path(download_complete_path)),
        "staging_root": str(Path(staging_root).resolve()),
        "bindings": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in bindings.items()},
        "implementation_sha256": implementation, "event_threshold": float(event_threshold),
        "max_memory_bytes": int(max_memory_bytes), "max_segment_rows": int(max_segment_rows),
        "candidate_loading": "CANCER_FILTERED_NO_GLOBAL_OBJECT_TABLE",
        "observed_process_memory_gate_enforced": True,
    }
    context_sha = _json_sha(context)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "RUN_CONTEXT.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != {**context, "context_sha256": context_sha}:
            raise StreamingCNVError("Streaming output root context drift")
    else:
        context_path.write_text(
            json.dumps({**context, "context_sha256": context_sha}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    folds, intervals, membership = _normalise_shared_inputs(
        _read(bindings["patient_folds"]), _read(bindings["entity_intervals"]),
        _read(bindings["pathway_membership"]),
    )
    _enforce_observed_memory(max_memory_bytes, stage="shared_inputs_loaded")
    selected = _manifest_rows(Path(gate["manifest_tsv"]))
    requested = tuple(dict.fromkeys(str(value).upper() for value in cancers))
    if unknown := sorted(set(requested) - set(TCGA_CANCERS)):
        raise StreamingCNVError(f"Unknown cancers: {unknown}")
    for cancer in requested:
        local_candidates = _read_cancer_candidates(bindings["candidates"], cancer)
        if local_candidates.empty:
            raise StreamingCNVError(f"{cancer} candidate partition is empty")
        _enforce_observed_memory(max_memory_bytes, stage=f"{cancer}:candidates_loaded")
        materialize_cancer_partition(
            cancer_id=cancer, staging_root=Path(staging_root).resolve(), output_root=output,
            candidates=local_candidates, folds=folds, intervals=intervals, membership=membership,
            selected_rows=selected, context_sha256=context_sha, event_threshold=event_threshold,
            max_memory_bytes=max_memory_bytes, max_segment_rows=max_segment_rows,
        )
        del local_candidates
        gc.collect()
        _enforce_observed_memory(max_memory_bytes, stage=f"{cancer}:after_release")
    partitions = {}
    missing = []
    for cancer in TCGA_CANCERS:
        path = output / f"cancer={cancer}"
        if not path.is_dir():
            missing.append(cancer)
            continue
        payload = validate_cancer_partition(path, context_sha256=context_sha)
        partitions[cancer] = {
            "path": str(path), "success_sha256": sha256_file(path / "SUCCESS.json"),
            "files_composite_sha256": _json_sha(payload["files"]),
        }
    if missing:
        return {"status": "INCOMPLETE", "context_sha256": context_sha, "missing_cancers": missing}
    success_payload = {
        "format": FORMAT, "status": "SUCCESS", "context_sha256": context_sha,
        "run_context_sha256": sha256_file(context_path), "cancers": partitions,
        "patient_folds": 5, "required_cancers": list(TCGA_CANCERS),
        "typed_unavailable_is_never_zero": True,
    }
    success = output / "SUCCESS.json"
    if success.exists():
        if json.loads(success.read_text(encoding="utf-8")) != success_payload:
            raise StreamingCNVError("Streaming aggregate SUCCESS drift")
    else:
        temporary = output / ".SUCCESS.json.partial"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(success_payload, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, success)
    return success_payload


def validate_streaming_store(
    success_path: str | Path, *, expected_bindings: Mapping[str, str | Path] | None = None
) -> dict[str, Any]:
    success = Path(success_path).resolve()
    payload = json.loads(success.read_text(encoding="utf-8"))
    root = success.parent
    context_path = root / "RUN_CONTEXT.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT or payload.get("status") != "SUCCESS":
        raise StreamingCNVError("Streaming aggregate is not SUCCESS")
    if payload.get("run_context_sha256") != sha256_file(context_path):
        raise StreamingCNVError("Streaming run context hash drift")
    if context.get("implementation_sha256") != implementation_hashes():
        raise StreamingCNVError("Streaming implementation hash drift")
    if expected_bindings:
        for name, raw_path in expected_bindings.items():
            path = Path(raw_path).resolve()
            item = context.get("bindings", {}).get(name, {})
            if item.get("path") != str(path) or item.get("sha256") != sha256_file(path):
                raise StreamingCNVError(f"Streaming binding drift: {name}")
    if set(payload.get("cancers", {})) != set(TCGA_CANCERS):
        raise StreamingCNVError("Streaming aggregate does not cover full33")
    for cancer, item in payload["cancers"].items():
        partition = Path(item["path"]).resolve()
        current = validate_cancer_partition(partition, context_sha256=payload["context_sha256"])
        if sha256_file(partition / "SUCCESS.json") != item["success_sha256"]:
            raise StreamingCNVError(f"Streaming partition SUCCESS drift: {cancer}")
        if _json_sha(current["files"]) != item["files_composite_sha256"]:
            raise StreamingCNVError(f"Streaming partition inventory drift: {cancer}")
    return payload


@dataclass
class CompactCNVPartition:
    root: Path

    def __post_init__(self) -> None:
        self.patient_ids = tuple(json.loads((self.root / "patient_ids.json").read_text(encoding="utf-8")))
        self.lncrna_ids = tuple(json.loads((self.root / "lncrna_ids.json").read_text(encoding="utf-8")))
        self.pathway_ids = tuple(json.loads((self.root / "pathway_ids.json").read_text(encoding="utf-8")))
        self.patient_index = {value: index for index, value in enumerate(self.patient_ids)}
        self.lncrna_index = {value: index for index, value in enumerate(self.lncrna_ids)}
        self.pathway_index = {value: index for index, value in enumerate(self.pathway_ids)}
        self.le = np.load(self.root / "lncrna_event.npy", mmap_mode="r")
        self.lc = np.load(self.root / "lncrna_callable.npy", mmap_mode="r")
        self.lb = np.load(self.root / "lncrna_burden.npy", mmap_mode="r")
        self.pe = np.load(self.root / "pathway_event.npy", mmap_mode="r")
        self.pc = np.load(self.root / "pathway_callable.npy", mmap_mode="r")
        self.pb = np.load(self.root / "pathway_burden.npy", mmap_mode="r")

    def candidate_statistics_arrays(
        self, candidates: pd.DataFrame, patients: Sequence[str], min_pair_callable: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = len(candidates)
        domain = np.zeros((size, 6), np.float32)
        labels = np.full(size, np.nan, np.float32)
        available = np.zeros(size, bool)
        reasons = np.full(size, "CNV_NO_EXPLICIT_PAIR_CALLABILITY", object)
        rows = np.asarray([self.patient_index[p] for p in patients if p in self.patient_index], int)
        if not len(rows):
            reasons[:] = "CNV_NOT_AVAILABLE_FOR_CANCER"
            return domain, labels, available, reasons
        li_all = np.asarray([self.lncrna_index.get(str(v), -1) for v in candidates.lncrna_id], int)
        pi_all = np.asarray([self.pathway_index.get(str(v), -1) for v in candidates.pathway_id], int)
        for start in range(0, size, 512):
            stop = min(size, start + 512)
            li, pi = li_all[start:stop], pi_all[start:stop]
            valid = (li >= 0) & (pi >= 0)
            local = np.flatnonzero(valid)
            if not len(local):
                reasons[start:stop][li < 0] = "CNV_LNCRNA_UNAVAILABLE"
                reasons[start:stop][pi < 0] = "CNV_PATHWAY_UNAVAILABLE"
                continue
            lx = self.le[np.ix_(rows, li[valid])].astype(float)
            py = self.pe[np.ix_(rows, pi[valid])].astype(float)
            lb = self.lb[np.ix_(rows, li[valid])].astype(float)
            pb = self.pb[np.ix_(rows, pi[valid])].astype(float)
            pair = self.lc[np.ix_(rows, li[valid])].astype(bool) & self.pc[np.ix_(rows, pi[valid])].astype(bool)
            count = pair.sum(0).astype(float)
            safe = np.maximum(count, 1)
            x, y = np.where(pair, lx, 0), np.where(pair, py, 0)
            xs, ys, xys = x.sum(0), y.sum(0), (x * y).sum(0)
            n11, n10, n01 = xys, xs - xys, ys - xys
            n00 = count - n11 - n10 - n01
            denominator = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
            phi = np.divide(n11*n00-n10*n01, denominator, out=np.full_like(count, np.nan), where=denominator > 0)
            indices = start + local
            domain[indices] = np.column_stack([
                np.log1p(count), count / len(rows), xs/safe, ys/safe,
                np.where(pair, np.abs(lb), 0).sum(0)/safe,
                np.where(pair, np.abs(pb), 0).sum(0)/safe,
            ]).astype(np.float32)
            eligible = (count >= min_pair_callable) & (xs > 0) & (xs < count) & (ys > 0) & (ys < count) & np.isfinite(phi)
            labels[indices[eligible]] = (phi[eligible] > 0).astype(np.float32)
            available[indices[eligible]] = True
            reasons[indices[count < min_pair_callable]] = "CNV_INSUFFICIENT_PAIR_CALLABILITY"
            reasons[indices[(count >= min_pair_callable) & ~eligible]] = "CNV_NO_EXPLICIT_WT_EVENT_VARIATION"
            reasons[indices[eligible]] = ""
        return domain, labels, available, reasons


def open_streaming_partitions(success_path: str | Path) -> dict[str, CompactCNVPartition]:
    payload = validate_streaming_store(success_path)
    return {
        cancer: CompactCNVPartition(Path(item["path"]).resolve())
        for cancer, item in payload["cancers"].items()
    }


__all__ = [
    "CompactCNVPartition", "StreamingCNVError", "implementation_hashes",
    "materialize_cancer_partition", "materialize_streaming_store",
    "open_streaming_partitions", "validate_cancer_partition", "validate_streaming_store",
]
