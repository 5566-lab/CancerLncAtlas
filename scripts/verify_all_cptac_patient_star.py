"""Verify every CPTAC STAR v36 RNA value and every tumor lncRNA patient value.

Run with the GDC intake, the validated specimen match table, and both completed
source and patient matrix releases. The published receipt contains cohort counts
and cryptographic digests, without sample identifiers.
"""

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_positions(requested, available):
    if not (
        requested.is_unique and available.is_unique and len(requested) == len(available)
    ):
        raise ValueError("Expected two unique and equally sized gene axes")
    order = available.get_indexer(requested)
    if np.any(order < 0):
        raise ValueError("RNA gene axes differ")
    return order


def compare(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite value in a measured file or expression matrix")
    delta = np.abs(actual - expected)
    return int(np.count_nonzero(delta)), float(delta.max())


def verify_cohort(job):
    cohort, samples, source_root, common_root, intake, hits = job
    source = pd.read_parquet(
        source_root / "matrices" / cohort / "rna_TumorNormal.parquet"
    ).set_index("feature_id")
    lncs = pd.read_parquet(
        common_root / "aligned" / ("CPTAC_" + cohort) / "LNC_LOGTPM_FULL.parquet"
    )
    if not source.columns.is_unique or set(source.columns) != set(samples.sample_id):
        raise ValueError(cohort + ": source specimen axis differs from match authority")
    tumors = samples.loc[samples.tissue_type.eq("Tumor"), "patient_id"]
    if (
        not tumors.is_unique
        or not lncs.index.is_unique
        or not tumors.isin(lncs.index).all()
    ):
        raise ValueError(
            cohort + ": tumor RNA patient axis differs from match authority"
        )
    unmeasured = lncs.loc[~lncs.index.isin(tumors)]
    if not unmeasured.isna().all().all():
        raise ValueError(cohort + ": unmatched patient has finite lncRNA values")
    if not lncs.columns.str.startswith("LNC:").all():
        raise ValueError(cohort + ": unexpected patient lncRNA identifier")
    lnc_ids = lncs.columns.str.removeprefix("LNC:")
    previous_ids = previous_types = None
    source_differences = lnc_differences = 0
    source_max = lnc_max = 0.0
    for row in samples.itertuples(index=False):
        hit = hits[row.gdc_file_id]
        data = (intake / "files" / (row.gdc_file_id + ".tsv")).read_bytes()
        if (
            hit["analysis"]["workflow_type"] != "STAR - Counts"
            or hit["access"] != "open"
            or hit["data_type"] != "Gene Expression Quantification"
            or len(data) != int(hit["file_size"])
            or hashlib.md5(data).hexdigest() != hit["md5sum"]
            or hit["md5sum"] != row.gdc_md5
            or data.split(b"\n", 1)[0].strip() != b"# gene-model: GENCODE v36"
        ):
            raise ValueError(
                cohort + ": native file identity, gene model or workflow mismatch"
            )
        native = pd.read_csv(
            io.BytesIO(data),
            comment="#",
            sep="\t",
            usecols=["gene_id", "gene_type", "tpm_unstranded"],
        )
        ids = native.gene_id.to_numpy()
        types = native.gene_type.fillna("").to_numpy()
        if (
            previous_ids is None
            or not np.array_equal(ids, previous_ids)
            or not np.array_equal(types, previous_types)
        ):
            # The STAR summary rows have no expression units. Keep the complete
            # v36 source universe, including pseudoautosomal copies, in the source view.
            gene_rows = np.flatnonzero(native.gene_id.str.startswith("ENSG"))
            stable = pd.Index(
                [re.sub(r"\.\d+(?=_PAR_Y$|$)", "", ids[i]) for i in gene_rows]
            )
            source_order = gene_rows[index_positions(source.index, stable)]
            pos = np.flatnonzero(
                (types[gene_rows] == "lncRNA") & ~stable.str.endswith("_PAR_Y")
            )
            lnc_order = gene_rows[pos[index_positions(lnc_ids, stable[pos])]]
            previous_ids, previous_types = ids, types
        linear = native.tpm_unstranded.to_numpy()
        native_logged = np.log2(1.0 + linear)
        mismatch, maximum = compare(
            source[row.sample_id].to_numpy(), native_logged[source_order]
        )
        source_differences += mismatch
        source_max = max(source_max, maximum)
        if row.tissue_type == "Tumor":
            mismatch, maximum = compare(
                lncs.loc[row.patient_id].to_numpy(), native_logged[lnc_order]
            )
            lnc_differences += mismatch
            lnc_max = max(lnc_max, maximum)
    result = {
        "tissues": dict(Counter(samples.tissue_type)),
        "samples": len(samples),
        "source_genes": len(source),
        "source_values": len(source) * len(samples),
        "source_differences": source_differences,
        "source_max_abs_diff": source_max,
        "patients_with_tumor_rna": len(tumors),
        "unmeasured_patient_rows": len(unmeasured),
        "patient_lncrnas": len(lnc_ids),
        "patient_lncrna_values": len(tumors) * len(lnc_ids),
        "patient_lncrna_differences": lnc_differences,
        "patient_lncrna_max_abs_diff": lnc_max,
    }
    print(cohort, result, flush=True)
    return cohort, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("intake", "matches", "source", "common", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    paths = {
        "matches": args.matches,
        "gdc_metadata": args.intake / "gdc_files.json",
        "source": args.source / "COMPLETE.json",
        "patient": args.common / "COMPLETE.json",
    }
    hashes = {k: sha256(p) for k, p in paths.items()}
    raw = json.loads(paths["source"].read_text())
    patient = json.loads(paths["patient"].read_text())
    if (
        raw["status"] != "COMPLETE"
        or patient["status"] != "COMPLETE"
        or patient["cptac_source"]["sha256"] != hashes["source"]
        or raw["rna_quantification"]["match_sha256"] != hashes["matches"]
        or raw["rna_quantification"]["metadata_sha256"] != hashes["gdc_metadata"]
    ):
        raise ValueError("Input files do not belong to the same GDC generation")
    samples = pd.read_csv(args.matches, sep="\t", dtype=str, keep_default_na=False)
    if (
        not samples.match_status.eq("MATCHED").all()
        or not samples.gdc_file_id.is_unique
        or not samples.sample_id.is_unique
        or not samples.tissue_type.isin(["Tumor", "Normal"]).all()
        or set(samples.source_cohort) != set(raw["cohorts"])
    ):
        raise ValueError(
            "Expected one unique quantified file for every RNA specimen in every source cohort"
        )
    hits = {r["file_id"]: r for r in json.loads(paths["gdc_metadata"].read_text())}
    jobs = [
        (
            c,
            g,
            args.source,
            args.common,
            args.intake,
            {k: hits[k] for k in g.gdc_file_id},
        )
        for c, g in samples.groupby("source_cohort")
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        cohorts = dict(pool.map(verify_cohort, jobs))
    if any(hashes[k] != sha256(p) for k, p in paths.items()):
        raise ValueError("Input generation changed during the audit")
    totals = {
        k: sum(r[k] for r in cohorts.values())
        for k in (
            "samples",
            "source_values",
            "source_differences",
            "patients_with_tumor_rna",
            "patient_lncrna_values",
            "patient_lncrna_differences",
            "unmeasured_patient_rows",
        )
    }
    for k in ("source_max_abs_diff", "patient_lncrna_max_abs_diff"):
        totals[k] = max(r[k] for r in cohorts.values())
    status = (
        "PASS"
        if totals["source_differences"] == 0
        and totals["patient_lncrna_differences"] == 0
        else "REVIEW"
    )
    out = {
        "status": status,
        "gene_model": "GENCODE v36",
        "workflow": "STAR - Counts",
        "assay": "log2(1 + tpm_unstranded)",
        "inputs_sha256": hashes,
        "totals": totals,
        "cohorts": cohorts,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
