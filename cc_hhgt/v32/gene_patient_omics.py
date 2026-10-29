"""Measured protein-coding gene assays, prepared on 149 without parameter fitting.

Matrices are physically patient x gene. CNV is the signed segment value at
the GENCODE gene midpoint. Methylation averages distinct finite promoter CpGs
per file, files per sample, then samples per patient. No pathway/lncRNA proxy.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import socket

import numpy as np
import pandas as pd

from .current_core_export import sha256, verified_path
from .formal_expression_sources import record, read, write

INPUT = "GENE_PATIENT_ASSAYS_INPUT_V1"
OUTPUT = "GENE_PATIENT_ASSAYS_VERIFIED_V1"
CHANNELS = {"cnv_signed": "log2_copy_ratio", "methylation_promoter_beta": "beta"}
PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}


def host():
    if socket.gethostname() != "149":
        raise RuntimeError("GENE_ASSAYS_REQUIRE_HOST_149")


def stable(identifier):
    # Keep the PAR_Y suffix: these genes must not collapse onto chrX.
    return re.sub(r"\.\d+(?=_PAR_Y$|$)", "", identifier)


def reference(gtf, requested, hm450):
    """Derive gene bodies and strand-aware transcript promoters from one GTF."""
    from scripts.derive_gencode_promoters import parse_attrs
    genes, promoters = [], []
    with gzip.open(gtf, "rt") as handle:
        header = []
        for line in handle:
            if line.startswith("#"):
                header.append(line.strip()); continue
            f = line.rstrip().split("\t")
            if len(f) != 9 or f[0] not in PRIMARY or f[2] not in {"gene", "transcript"}:
                continue
            a = parse_attrs(f[8]); gene = stable(a.get("gene_id", ""))
            if gene not in requested or a.get("gene_type", a.get("gene_biotype")) != "protein_coding":
                continue
            start, end = int(f[3]), int(f[4])
            if start < 1 or end < start or f[6] not in {"+", "-"}:
                raise ValueError("GENE_REFERENCE_COORDINATES")
            if f[2] == "gene":
                genes.append((gene, a["gene_id"], f[0], start, end, f[6]))
            else:
                tss = start - 1 if f[6] == "+" else end - 1
                left, right = ((tss-1000, tss+101) if f[6] == "+" else (tss-100, tss+1001))
                promoters.append((gene, f[0], max(0, left), right, a["transcript_id"]))
    if not any("GRCh38" in line for line in header):
        raise ValueError("GENE_REFERENCE_BUILD_NOT_GRCH38")
    bodies = pd.DataFrame(genes, columns=["gene_id", "versioned_gene_id", "chromosome", "start", "end", "strand"])
    if bodies.empty or bodies.gene_id.duplicated().any():
        raise ValueError("GENE_REFERENCE_EMPTY_OR_DUPLICATE")
    bodies = bodies.sort_values("gene_id").reset_index(drop=True)
    prom = pd.DataFrame(promoters, columns=["gene_id", "chromosome", "start0", "end0", "transcript_id"])
    prom = prom.loc[prom.gene_id.isin(bodies.gene_id)].drop_duplicates()
    probes = pd.read_csv(hm450, sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg", "CpG_end"])
    probes = probes.loc[probes.probeID.str.startswith("cg", na=False) & probes.CpG_chrm.isin(PRIMARY)].copy()
    for c in ["CpG_beg", "CpG_end"]:
        probes[c] = pd.to_numeric(probes[c], errors="coerce")
    probes = probes.loc[probes.CpG_beg.ge(0) & probes.CpG_end.gt(probes.CpG_beg)].copy()
    if probes.probeID.duplicated().any() or not np.equal(probes[["CpG_beg", "CpG_end"]], np.floor(probes[["CpG_beg", "CpG_end"]])).all().all():
        raise ValueError("GENE_PROBE_COORDINATES_OR_DUPLICATES")
    import duckdb
    with duckdb.connect() as con:
        con.register("p", probes); con.register("r", prom)
        mapping = con.execute("""SELECT DISTINCT p.probeID AS probe_id, r.gene_id
            FROM p JOIN r ON p.CpG_chrm=r.chromosome
            AND p.CpG_beg<r.end0 AND p.CpG_end>r.start0
            ORDER BY probe_id,gene_id""").fetchdf()
    if mapping.empty:
        raise ValueError("GENE_PROMOTER_PROBE_MAP_EMPTY")
    return bodies, prom, mapping, sorted(requested - set(bodies.gene_id))


def verified_bytes(row):
    path = Path(row["path"])
    data = path.read_bytes()
    if len(data) != int(row["bytes"]) or hashlib.md5(data, usedforsecurity=False).hexdigest() != row["md5"]:
        raise ValueError("GENE_RAW_SIZE_OR_MD5_MISMATCH: " + str(path))
    return data, {**row, "sha256": hashlib.sha256(data).hexdigest()}


_BODIES = _MAP = _GENES = None


def init_worker(bodies, mapping):
    global _BODIES, _MAP, _GENES
    _BODIES, _MAP = bodies, mapping
    _GENES = tuple(bodies.gene_id)


def assay_file(row):
    data, receipt = verified_bytes(row)
    if row["channel"] == "cnv_signed":
        from .directional_cnv import map_segments_directional
        raw = pd.read_csv(io.BytesIO(data), sep="\t")
        raw.columns = raw.columns.str.lower()
        frame = raw[["chromosome", "start", "end", "segment_mean"]].rename(columns={"segment_mean": "value"})
        for name in ["start", "end"]:
            values = pd.to_numeric(frame[name], errors="raise")
            if not np.isfinite(values).all() or (values != np.floor(values)).any() or values.lt(1).any():
                raise ValueError("GENE_CNV_COORDINATE_INVALID")
        if frame.chromosome.isna().any() or frame.end.lt(frame.start).any():
            raise ValueError("GENE_CNV_INTERVAL_INVALID")
        frame["chromosome"] = frame.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
        shared = {}
        for chrom, part in frame.sort_values(["chromosome", "start"]).groupby("chromosome"):
            starts, ends = part.start.to_numpy(), part.end.to_numpy()
            if (starts[1:] < ends[:-1]).any():
                raise ValueError("GENE_CNV_SEGMENTS_OVERLAP: " + row["file_id"])
            shared[chrom] = set(starts[1:][starts[1:] == ends[:-1]])
        values = pd.to_numeric(frame.value, errors="raise")
        if not np.isfinite(values).all():
            raise ValueError("GENE_CNV_VALUE_INVALID")
        frame["value"] = values
        intervals = _BODIES.rename(columns={"gene_id": "entity_id"})
        result = map_segments_directional(frame, intervals, _GENES).signed_mean
        # GDC SNP6 files sometimes share an endpoint between adjacent segments.
        # A midpoint at that coordinate is ambiguous: never choose a direction.
        ambiguous = np.array([(r.start+r.end)//2 in shared.get(r.chromosome.removeprefix("chr"), set())
                              for r in _BODIES.itertuples()], dtype=bool)
        result[ambiguous] = np.nan
        receipt.update(shared_boundary_count=sum(map(len, shared.values())), ambiguous_gene_count=int(ambiguous.sum()))
    else:
        beta = pd.read_csv(io.BytesIO(data), sep="\t", header=None, names=["probe_id", "beta"],
                           usecols=[0, 1], na_values=["NA", "NaN", "nan", ""])
        if beta.probe_id.isna().any() or beta.probe_id.duplicated().any():
            raise ValueError("GENE_METHYLATION_DUPLICATE_PROBE")
        beta["beta"] = pd.to_numeric(beta.beta, errors="raise")
        if np.isinf(beta.beta).any() or not beta.beta.dropna().between(0, 1).all():
            raise ValueError("GENE_METHYLATION_BETA_INVALID")
        matched = _MAP.merge(beta.dropna(subset=["beta"]), on="probe_id", validate="many_to_one")
        means = matched.groupby("gene_id").beta.mean().reindex(_GENES)
        result = means.to_numpy(dtype=np.float32)
    receipt["observed_genes"] = int(np.isfinite(result).sum())
    return result, receipt


def input_context(config_path):
    cfg = read(config_path)
    if cfg.get("format") != INPUT or cfg.get("target_host") != "149" or cfg.get("genome_build") != "GRCh38":
        raise ValueError("GENE_INPUT_FORMAT_HOST_BUILD")
    keys = ["gtf", "hm450", "genes", "patient_folds", "raw_files"]
    files = {k: verified_path(cfg[k], Path(config_path).parent) for k in keys}
    for k, item in cfg.get("source_authorities", {}).items():
        files["authority:" + k] = verified_path(item, Path(config_path).parent)
    requested = read(files["genes"])
    if not requested or len(requested) != len(set(requested)) or any(not re.fullmatch(r"ENSG\d+(?:_PAR_Y)?", g) for g in requested):
        raise ValueError("GENE_IDS_NOT_STABLE_ENSEMBL")
    patients = pd.read_csv(files["patient_folds"], sep="\t", dtype={"patient_id": str, "cancer_id": str})
    patients = patients[["cancer_id", "patient_id", "patient_fold_id"]].drop_duplicates()
    if patients.isna().any().any() or patients.patient_id.duplicated().any() or not patients.patient_fold_id.isin(range(5)).all():
        raise ValueError("GENE_PATIENT_FOLDS_INVALID")
    patients = patients.sort_values(["cancer_id", "patient_id"]).reset_index(drop=True)
    raw = pd.read_parquet(files["raw_files"])
    required = ["channel", "cancer_id", "patient_id", "sample_id", "file_id", "path", "bytes", "md5", "sample_type"]
    if not set(required) <= set(raw) or raw[required].isna().any().any() or raw.empty:
        raise ValueError("GENE_RAW_FILES_INCOMPLETE")
    if not set(raw.channel) <= set(CHANNELS) or raw.duplicated(["channel", "file_id"]).any() or raw.path.duplicated().any():
        raise ValueError("GENE_RAW_FILES_DUPLICATED_OR_CHANNEL")
    if not raw.sample_type.isin(["Primary Tumor", "Primary Blood Derived Cancer - Peripheral Blood", "Metastatic", "Recurrent Tumor"]).all():
        raise ValueError("GENE_RAW_NOT_TUMOR")
    if not raw.md5.str.fullmatch(r"[0-9a-f]{32}").all() or not pd.to_numeric(raw.bytes, errors="raise").gt(0).all():
        raise ValueError("GENE_RAW_CHECKSUM_OR_SIZE_INVALID")
    identities = patients.set_index("patient_id").cancer_id
    if not raw.cancer_id.eq(raw.patient_id.map(identities)).all():
        raise ValueError("GENE_RAW_PATIENT_CANCER_MISMATCH")
    if raw.groupby("sample_id").patient_id.nunique().gt(1).any():
        raise ValueError("GENE_RAW_SAMPLE_IDENTITY_CONFLICT")
    # Current GDC CNV authority selects exactly one source per patient.
    if raw.loc[raw.channel.eq("cnv_signed")].patient_id.duplicated().any():
        raise ValueError("GENE_CNV_MULTIPLE_SELECTED_FILES")
    return cfg, files, set(requested), patients, raw


def materialize(config_path, output, *, workers=4):
    host(); config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("GENE_WORKERS_OUT_OF_RANGE")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
          "source_config": record(config_path), "operation": OUTPUT})
    cfg, files, requested, patients, raw = input_context(config_path)
    bodies, promoters, mapping, excluded = reference(files["gtf"], requested, files["hm450"])
    bodies.to_parquet(output / "genes.parquet", index=False)
    promoters.to_parquet(output / "transcript_promoters.parquet", index=False)
    mapping.to_parquet(output / "promoter_probe_map.parquet", index=False)
    patients.to_parquet(output / "patients.parquet", index=False)
    write(output / "unresolved_or_non_coding_genes.json", excluded)
    partitions, sources = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(bodies, mapping)) as pool:
        for cancer, axis in patients.groupby("cancer_id", sort=True):
            part = output / cancer; part.mkdir()
            identifiers = axis.patient_id.tolist(); positions = {p: i for i, p in enumerate(identifiers)}
            write(part / "patient_ids.json", identifiers)
            for channel in CHANNELS:
                values = np.full((len(axis), len(bodies)), np.nan, dtype=np.float32)
                rows = raw.loc[raw.cancer_id.eq(cancer) & raw.channel.eq(channel)].sort_values(["patient_id", "sample_id", "file_id"])
                results = iter(pool.map(assay_file, rows.to_dict("records"), chunksize=1))
                # Per-patient tasks bound memory and preserve equal sample weights.
                for patient, group in rows.groupby("patient_id", sort=True):
                    sample_values = []
                    for _, sample in group.groupby("sample_id", sort=True):
                        arrays = []
                        for _ in range(len(sample)):
                            array, receipt = next(results)
                            arrays.append(array); sources.append(receipt)
                        sample_values.append(finite_mean(np.stack(arrays)))
                    values[positions[patient]] = finite_mean(np.stack(sample_values))
                np.save(part / (channel + ".npy"), values, allow_pickle=False)
                np.save(part / (channel + "__available.npy"), np.isfinite(values), allow_pickle=False)
                print(json.dumps({"cancer_id": cancer, "channel": channel, "files_verified": len(rows),
                                  "observed_patients": int(np.isfinite(values).any(axis=1).sum())}), flush=True)
            partitions.append({"cancer_id": cancer, "patients": record(part / "patient_ids.json"),
                "channels": {c: {"values": record(part / (c + ".npy")), "available": record(part / (c + "__available.npy"))} for c in CHANNELS}})
    pd.DataFrame(sources).to_parquet(output / "verified_raw_files.parquet", index=False)
    # Metadata/annotation must remain exactly the declared inputs during the run.
    for k, path in files.items():
        original = cfg["source_authorities"][k.split(":", 1)[1]] if k.startswith("authority:") else cfg[k]
        if sha256(path) != original["sha256"]:
            raise ValueError("GENE_INPUT_CHANGED_DURING_PREPARATION")
    payload = {"format": OUTPUT, "status": "PASS", "target_host": "149", "training_started": False,
        "source_config": record(config_path), "source_artifacts": {k: record(p) for k, p in files.items()},
        "node_type": "gene", "gene_type": "protein_coding", "genome_build": "GRCh38", "units": CHANNELS,
        "matrix_orientation": "patient_by_gene", "cnv_definition": "GENCODE_GENE_MIDPOINT_SIGNED_SEGMENT_MEAN_1_BASED_CLOSED_SHARED_BOUNDARY_MASKED",
        "methylation_definition": "DISTINCT_CPG_TRANSCRIPT_TSS_MINUS1000_PLUS100_INCLUSIVE_BED0_OVERLAP",
        "aggregation": "FINITE_PROBES_PER_FILE_THEN_FILES_PER_SAMPLE_THEN_SAMPLES_PER_PATIENT",
        "genes": record(output / "genes.parquet"), "patients": record(output / "patients.parquet"),
        "promoter_probe_map": record(output / "promoter_probe_map.parquet"),
        "transcript_promoters": record(output / "transcript_promoters.parquet"),
        "verified_raw_files": record(output / "verified_raw_files.parquet"),
        "excluded_requested_genes": record(output / "unresolved_or_non_coding_genes.json"),
        "raw_file_count": len(sources), "partitions": partitions}
    payload["implementation"] = {str(p): record(p) for p in [Path(__file__).resolve(),
        Path(__file__).with_name("directional_cnv.py").resolve(),
        Path(__file__).resolve().parents[2] / "scripts/derive_gencode_promoters.py"]}
    write(output / "COMPLETE.json", payload)
    return output / "COMPLETE.json"


def finite_mean(values):
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0, dtype=np.float64), count,
                     out=np.full(values.shape[1], np.nan), where=count > 0).astype(np.float32)


def aggregate(entry, source, authority, selected, fold):
    """Stream matrix partitions to node features without a 200M-row long table."""
    channel = entry["channel"]
    path = source(entry["measurements"], channel + ":gene_product")
    data = read(path)
    if (data.get("format") != OUTPUT or data.get("status") != "PASS" or data.get("target_host") != "149"
            or data.get("node_type") != "gene" or data.get("gene_type") != "protein_coding"
            or data.get("genome_build") != "GRCh38" or data.get("matrix_orientation") != "patient_by_gene"
            or data.get("units", {}).get(channel) != entry["unit"]):
        raise ValueError("GENE_PRODUCT_CONTRACT_INVALID")
    def pin(item, role):
        file = verified_path(item, path.parent)
        return source(record(file), channel + ":gene:" + role)
    for role, item in data["source_artifacts"].items(): pin(item, "source:" + role)
    for role, item in data["implementation"].items(): pin(item, "implementation:" + role)
    for role in ["source_config", "promoter_probe_map", "transcript_promoters", "verified_raw_files", "excluded_requested_genes"]:
        pin(data[role], role)
    genes = pd.read_parquet(pin(data["genes"], "genes"))
    patients = pd.read_parquet(pin(data["patients"], "patients"))
    if (genes.empty or genes.gene_id.duplicated().any() or patients.patient_id.duplicated().any()
            or not genes.gene_id.str.fullmatch(r"ENSG\d+(?:_PAR_Y)?").all()
            or not patients.patient_fold_id.isin(range(5)).all()):
        raise ValueError("GENE_PRODUCT_AXES_INVALID")
    known = authority[["patient_id", "cancer_id", "partition"]].drop_duplicates()
    joined = patients.merge(known, on=["patient_id", "cancer_id"], how="left", validate="one_to_one")
    expected_split = np.where(joined.patient_fold_id.eq(fold), "test",
                              np.where(joined.patient_fold_id.eq((fold+1) % 5), "validation", "train"))
    if joined.partition.isna().any() or not joined.partition.eq(expected_split).all():
        raise ValueError("GENE_PRODUCT_PATIENT_FOLD_MISMATCH")
    eligible = set(selected.patient_id)
    sums = np.zeros(len(genes), dtype=np.float64); counts = np.zeros(len(genes), dtype=np.int64)
    lineage, seen = [], set(); input_rows = missing_rows = observed_patients = 0
    for part in data["partitions"]:
        ids = read(pin(part["patients"], part["cancer_id"] + ":patients"))
        expected = patients.loc[patients.cancer_id.eq(part["cancer_id"]), "patient_id"].tolist()
        if ids != expected or not ids or seen.intersection(ids):
            raise ValueError("GENE_PRODUCT_PARTITION_AXES_INVALID")
        seen.update(ids)
        values = np.load(pin(part["channels"][channel]["values"], part["cancer_id"] + ":values"), mmap_mode="r", allow_pickle=False)
        mask = np.load(pin(part["channels"][channel]["available"], part["cancer_id"] + ":available"), mmap_mode="r", allow_pickle=False)
        if values.shape != (len(ids), len(genes)) or mask.shape != values.shape or np.isinf(values).any() or not np.isin(mask, [0, 1]).all() or not np.array_equal(mask.astype(bool), np.isfinite(values)):
            raise ValueError("GENE_PRODUCT_MATRIX_OR_AVAILABILITY_INVALID")
        if channel == "methylation_promoter_beta" and ((values[mask.astype(bool)] < 0).any() or (values[mask.astype(bool)] > 1).any()):
            raise ValueError("GENE_PRODUCT_BETA_INVALID")
        input_rows += values.size; missing_rows += int((~mask.astype(bool)).sum())
        for i, patient in enumerate(ids):
            if patient not in eligible: continue
            available = np.asarray(mask[i], dtype=bool); number = int(available.sum())
            sums[available] += values[i, available]; counts += available
            observed_patients += int(number > 0)
            lineage.append({"patient_id": patient, "cancer_id": part["cancer_id"], "channel": channel,
                            "source_row": i, "observed_gene_count": number})
    if seen != set(patients.patient_id) or eligible - seen:
        raise ValueError("GENE_PRODUCT_PATIENT_INVENTORY_INCOMPLETE")
    keep = counts > 0
    evidence = pd.DataFrame({"node_type": "gene", "canonical_id": "GENE:" + genes.loc[keep, "gene_id"],
                            "channel": channel, "raw_value": sums[keep] / counts[keep]}).reset_index(drop=True)
    coverage = {"input_rows": input_rows, "missing_rows": missing_rows, "contributing_rows": int(counts.sum()),
                "contributing_patients": observed_patients, "observed_nodes": len(evidence),
                "unit": entry["unit"], "input_granularity": "patient",
                "lineage_layout": "PATIENT_MATRIX_ROW_WITH_HASH_BOUND_GENE_AXIS"}
    return evidence, pd.DataFrame(lineage), coverage
