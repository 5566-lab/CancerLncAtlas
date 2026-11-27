"""Rebuild the CPTAC RNA inputs from public GDC STAR / GENCODE v36 files."""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json, read_json, sha
from .cptac_materialize import FORMAT, feature_record, gene_id
from .state_protein_map import ProteinResolver, pinned, reference_genes

GENE_MODEL = "GENCODE v36"
TPM = "tpm_unstranded"
SUMMARY = {"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"}
MATRIX = "rna_TumorNormal.parquet"
SYMBOLS = "RNA_TUMOR_SYMBOL_TPM.tsv.gz"
CODING = "PROTEIN_CODING_SYMBOLS.tsv"
GDC_FIELDS = {"data_type": "Gene Expression Quantification", "access": "open"}
SAMPLE_FIELDS = ["source_cohort", "native_case_id", "patient_id", "tissue_type"]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def read_star(path, hit):
    data = path.read_bytes()
    if len(data) != int(hit["file_size"]) or hashlib.md5(data).hexdigest() != hit["md5sum"]:
        raise ValueError(f"GDC size or MD5 mismatch: {path}")
    stream = io.BytesIO(data)
    if stream.readline().strip() != b"# gene-model: GENCODE v36":
        raise ValueError(f"Unexpected GDC gene model: {path}")
    table = pd.read_csv(stream, sep="\t", usecols=["gene_id", "gene_name", "gene_type", TPM],
                        dtype=str, keep_default_na=False)
    summary = table.gene_id.str.startswith("N_")
    if len(table.loc[summary]) != len(SUMMARY) or set(table.loc[summary, "gene_id"]) != SUMMARY:
        raise ValueError(f"Unexpected STAR count summary: {path}")
    table = table.loc[~summary].set_index("gene_id")
    if not table.index.is_unique or not table.index.str.fullmatch(r"ENSG\d+(?:\.\d+)?(?:_PAR_Y)?").all():
        raise ValueError(f"Invalid Ensembl gene axis: {path}")
    tpm = pd.to_numeric(table.pop(TPM), errors="raise").to_numpy(dtype=np.float64)
    if not len(table) or not np.isfinite(tpm).all() or np.any(tpm < 0):
        raise ValueError(f"Missing or invalid gene TPM: {path}")
    return table, tpm


def check_sample(row, hit):
    if (any(hit.get(k) != v for k, v in GDC_FIELDS.items())
            or hit.get("analysis", {}).get("workflow_type") != "STAR - Counts"
            or hit["md5sum"] != row.gdc_md5):
        raise ValueError(f"Unexpected GDC source for {row.sample_id}")
    if (len(hit["cases"]) != 1 or hit["cases"][0]["submitter_id"] != row.native_case_id
            or not re.fullmatch(r"[0-9a-f-]{36}", row.gdc_file_id)):
        raise ValueError(f"GDC file/case identity mismatch: {row.sample_id}")
    case = hit["cases"][0]
    if case["project"]["project_id"] not in {"CPTAC-2", "CPTAC-3"}:
        raise ValueError(f"Unexpected GDC project: {row.sample_id}")
    entities = hit["associated_entities"]
    if (len(entities) != 1 or entities[0]["entity_type"] != "aliquot"
            or entities[0]["entity_id"] != row.gdc_aliquot_id
            or entities[0]["entity_submitter_id"] != row.gdc_aliquot_submitter_id):
        raise ValueError(f"GDC aliquot mismatch: {row.sample_id}")
    tissue = {"Tumor": "Primary Tumor", "Normal": "Solid Tissue Normal"}[row.tissue_type]
    # One sequenced aliquot may derive from multiple documented parent samples.
    parents = [s for s in case["samples"] if any(a["aliquot_id"] == row.gdc_aliquot_id
               for portion in s.get("portions", []) for analyte in portion.get("analytes", [])
               for a in analyte.get("aliquots", []))]
    if (not parents or any(s["sample_type"] != tissue for s in parents)
            or {s["sample_id"] for s in parents} != set(row.gdc_sample_id.split("|"))
            or {s["submitter_id"] for s in parents} != set(row.gdc_sample_submitter_id.split("|"))):
        raise ValueError(f"GDC specimen type/ID mismatch: {row.sample_id}")


def check_match_axis(baseline, matches):
    old = baseline.loc[baseline.layer == "rna"].set_index("sample_id")
    match = matches.set_index("sample_id")
    if not old.index.is_unique or not match.index.is_unique or set(old.index) != set(match.index):
        raise ValueError("GDC match table must account for every original RNA column once")
    match = match.loc[old.index]
    if not old[SAMPLE_FIELDS].equals(match[SAMPLE_FIELDS]) or not match.layer.eq("rna").all():
        raise ValueError("GDC matches disagree with the CPTAC patient/specimen authority")
    if not match.match_status.isin(["MATCHED", "MISSING", "AMBIGUOUS"]).all():
        raise ValueError("Unknown specimen match status")
    selected = match.loc[match.match_status == "MATCHED"]
    if not selected.gdc_file_id.is_unique or not selected.gdc_aliquot_id.is_unique:
        raise ValueError("One GDC file/aliquot assigned to multiple samples")
    return match.reset_index()


def write_cohort(directory, matches, metadata, cache, axis):
    samples = matches.loc[matches.match_status.eq("MATCHED")]
    directory.mkdir(parents=True, exist_ok=True)
    values = np.empty((len(axis), len(samples)), dtype=np.float64, order="F")
    for i, row in enumerate(samples.itertuples(index=False)):
        hit = metadata[row.gdc_file_id]
        check_sample(row, hit)
        table, tpm = read_star(cache / (row.gdc_file_id + ".tsv"), hit)
        if not table.index.equals(axis.index):
            positions = table.index.get_indexer(axis.index)
            if len(table) != len(axis) or (positions < 0).any():
                raise ValueError(f"GDC gene universe differs for {row.sample_id}")
            table, tpm = table.iloc[positions], tpm[positions]
        if not table.equals(axis):
            raise ValueError(f"Gene labels differ under the same gene model for {row.sample_id}")
        values[:, i] = tpm
    stable = axis.index.map(gene_id).rename("feature_id")
    if not stable.is_unique:
        raise ValueError("Ensembl version removal causes a gene collision")
    cols = samples.sample_id.tolist()
    frame = pd.DataFrame(values, index=stable, columns=cols)
    np.log2(1 + frame).reset_index().to_parquet(directory / MATRIX, index=False)
    tumor_cols = samples.loc[samples.tissue_type.eq("Tumor"), "sample_id"].tolist()
    symbols = axis.gene_name.to_numpy()
    by_symbol = frame[tumor_cols].groupby(symbols, sort=False).sum(min_count=1)
    by_symbol.index.name = "gene_symbol"
    by_symbol.to_csv(directory / SYMBOLS, sep="\t", compression={"method": "gzip", "mtime": 0, "compresslevel": 2})
    coding = sorted(set(axis.loc[axis.gene_type == "protein_coding", "gene_name"]) - {""})
    pd.DataFrame({"gene_symbol": coding}).to_csv(directory / CODING, sep="\t", index=False)


def write_features(base, stage, cfg, axis):
    old = pd.read_parquet(base / "FEATURE_ALIGNMENT.parquet")
    genes, symbols = reference_genes(pinned(cfg["gencode"]))
    resolver = ProteinResolver(genes, symbols, read_tsv(pinned(cfg["hgnc"])),
                               pd.read_parquet(pinned(cfg["protein_gene"])))
    rows = []
    for row in axis.itertuples():
        rec = feature_record("rna", row.Index, resolver)
        rec.update(source_gene_symbol=row.gene_name, source_gene_type=row.gene_type)
        rows.append(rec)
    features = pd.concat([old.loc[old.layer.ne("rna")], pd.DataFrame(rows)], ignore_index=True)
    features.to_parquet(stage / "FEATURE_ALIGNMENT.parquet", index=False)


def write_tables(base, stage, target, matches, metadata, n_genes):
    samples = read_tsv(base / "CPTAC_SAMPLE_ASSAY_MAP.tsv")
    obs = read_tsv(base / "CPTAC_ASSAY_COVERAGE.tsv")
    selected = matches.loc[matches.match_status.eq("MATCHED")].copy()
    old_names = set(samples.loc[samples.layer.eq("rna"), "source_file"])
    selected["source_file"] = selected.gdc_file_id.map(lambda u: metadata[u]["file_name"])
    selected["native_sample_id"] = selected.gdc_sample_submitter_id
    mapping_fields = [field for field in ["gdc_file_id", "gdc_aliquot_id", "match_basis", "strict_pdc_sample_match"]
                      if field in selected and field not in samples.columns]
    updated = pd.concat([samples.loc[samples.layer.ne("rna")], selected[[*samples.columns, *mapping_fields]]],
                        ignore_index=True)
    updated.to_csv(stage / "CPTAC_SAMPLE_ASSAY_MAP.tsv", sep="\t", index=False)
    selected["features"] = n_genes
    selected["observed_features"] = n_genes
    selected["source_unit"] = "log2(TPM+1)"
    selected["matrix_path"] = [str(target / "matrices" / c / MATRIX) for c in selected.source_cohort]
    obs = obs.loc[obs.layer.ne("rna")].copy()
    obs["matrix_path"] = [str(target / "matrices" / c / Path(p).name)
                          for c, p in zip(obs.source_cohort, obs.matrix_path)]
    pd.concat([obs, selected[obs.columns]]).to_csv(stage / "CPTAC_ASSAY_COVERAGE.tsv", sep="\t", index=False)
    coverage = read_tsv(base / "COHORT_COVERAGE.tsv")
    for i, row in coverage.loc[coverage.layer.eq("rna")].iterrows():
        count = selected.loc[selected.source_cohort.eq(row.source_cohort) & selected.tissue_type.eq("Tumor"),
                             "patient_id"].nunique()
        coverage.loc[i, ["observed_patients", "coverage_percent"]] = str(count), str(100 * count / int(row.patients))
    coverage.to_csv(stage / "COHORT_COVERAGE.tsv", sep="\t", index=False)
    for name in ["SOURCE_INVALID_RNA_VALUES.tsv", "RNA_SCORING_EXCLUDED_GENES.tsv",
                 "SOURCE_NONFINITE_MEASUREMENTS.tsv", "CPTAC_EXCLUDED_SAMPLES.tsv"]:
        records = read_tsv(base / name)
        if "source_file" in records:
            records = records.loc[~records.source_file.isin(old_names)]
        else:
            records = records.iloc[:0]
        if name == "CPTAC_EXCLUDED_SAMPLES.tsv":
            records = records.loc[~records.reason.str.startswith("GDC_STAR_V36_")]
            unresolved = matches.loc[matches.match_status.ne("MATCHED")].copy()
            unresolved["reason"] = "GDC_STAR_V36_" + unresolved.match_status + ":" + unresolved.match_basis
            records = pd.concat([records, unresolved[records.columns]])
        records.to_csv(stage / name, sep="\t", index=False)
    matches.to_csv(stage / "GDC_SAMPLE_MATCH.tsv", sep="\t", index=False)


def materialize_gdc(baseline, matches_path, intake, output, published_root=None):
    base, matches_path, intake, out = map(lambda p: Path(p).resolve(), [baseline, matches_path, intake, output])
    if out.exists() or base in out.parents:
        raise ValueError("Choose an unused staging path outside the original dataset")
    source = read_json(base / "COMPLETE.json")
    if source["format"] != FORMAT or source["status"] != "COMPLETE":
        raise ValueError("Invalid baseline alignment")
    metadata_path = intake / "gdc_files.json"
    before = {p: sha(p) for p in [metadata_path, matches_path, base / "COMPLETE.json"]}
    hits = read_json(metadata_path)
    metadata = {h["file_id"]: h for h in hits}
    if len(metadata) != len(hits):
        raise ValueError("Duplicate file UUID in frozen GDC metadata")
    sample_map = base / ("GDC_SAMPLE_MATCH.tsv" if "rna_quantification" in source else "CPTAC_SAMPLE_ASSAY_MAP.tsv")
    metadata_inputs = {sample_map.name, "CPTAC_SAMPLE_ASSAY_MAP.tsv", "CPTAC_ASSAY_COVERAGE.tsv",
                       "FEATURE_ALIGNMENT.parquet", "COHORT_COVERAGE.tsv", "RNA_SCORING_EXCLUDED_GENES.tsv",
                       "SOURCE_INVALID_RNA_VALUES.tsv", "SOURCE_NONFINITE_MEASUREMENTS.tsv", "CPTAC_EXCLUDED_SAMPLES.tsv"}
    for name in metadata_inputs:
        if sha(base / name) != source["files"][name]["sha256"]:
            raise ValueError(f"Baseline alignment metadata changed: {name}")
    rows = read_tsv(matches_path)
    matches = check_match_axis(read_tsv(sample_map), rows.loc[rows.source_cohort.isin(source["cohorts"])])
    eligible = matches.loc[matches.match_status.eq("MATCHED")]
    if eligible.empty:
        raise ValueError("No verified GDC files")
    first = eligible.iloc[0]
    axis, _ = read_star(intake / "files" / (first.gdc_file_id + ".tsv"), metadata[first.gdc_file_id])
    counts = matches.groupby(["source_cohort", "match_status"]).size().unstack(fill_value=0)
    target = Path(published_root).resolve() if published_root else out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=out.name + ".staging.") as tmp:
        stage = Path(tmp)
        for c in source["cohorts"]:
            subset = matches.loc[matches.source_cohort.eq(c)]
            write_cohort(stage / "matrices" / c, subset, metadata, intake / "files", axis)
            print(f"{c}: {sum(subset.match_status == 'MATCHED')}/{len(subset)} GDC RNA columns", flush=True)
        write_features(base, stage, source["config"], axis)
        write_tables(base, stage, target, matches, metadata, len(axis))
        if any(sha(p) != h for p, h in before.items()):
            raise ValueError("GDC source manifests changed during materialization")
        bindings = {"workflow_type": "STAR - Counts", "gene_model": GENE_MODEL, "metric": TPM,
                    "baseline_complete_sha256": before[base / "COMPLETE.json"],
                    "match_sha256": before[matches_path], "metadata_sha256": before[metadata_path],
                    "cohort_counts": counts.to_dict(orient="index")}
        atomic_json(stage / "LAUNCH_PREFLIGHT.json", {"action": "CPTAC_HARMONIZED_RNA", **bindings})
        for rel, rec in source["files"].items():
            old, new = base / rel, stage / rel
            if not old.resolve().is_relative_to(base) or not new.resolve().is_relative_to(stage):
                raise ValueError(f"Unsafe baseline path: {rel}")
            if new.exists():
                continue
            new.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(old, new)
            except OSError:
                shutil.copy2(old, new)
            if new.stat().st_size != rec["bytes"] or sha(new) != rec["sha256"]:
                raise ValueError(f"Baseline payload mismatch: {rel}")
        files = {p.relative_to(stage).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in stage.rglob("*") if p.is_file()}
        receipt = {**source, "rna_quantification": bindings, "files": files,
                   "sample_authority": "Published CPTAC sample scope; GDC_STAR_MATCH resolves RNA case/tissue/aliquot"}
        atomic_json(stage / "COMPLETE.json", receipt)
        stage.rename(out)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline", "matches", "intake", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--published-root")
    args = parser.parse_args()
    materialize_gdc(args.baseline, args.matches, args.intake, args.output, args.published_root)


if __name__ == "__main__":
    main()
