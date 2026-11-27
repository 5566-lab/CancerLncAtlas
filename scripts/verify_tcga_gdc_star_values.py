"""Reconcile every TCGA expression column with its exact GDC aliquot and native counts."""

import argparse
import csv
import hashlib
import io
import json
import socket
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3.util.connection


def fetch_metadata(project, out):
    path = out / (project + "_gdc.json")
    if path.is_file():
        return json.loads(path.read_text())
    conditions = {
        "cases.project.project_id": "TCGA-" + project,
        "data_type": "Gene Expression Quantification",
        "analysis.workflow_type": "STAR - Counts",
        "access": "open",
    }
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": k, "value": [v]}}
            for k, v in conditions.items()
        ],
    }
    result = []
    with requests.Session() as s:
        s.trust_env = False
        while True:
            args = {
                "fields": "file_id,file_name,file_size,md5sum,associated_entities.entity_submitter_id",
                "size": 1500,
                "from": len(result),
                "sort": "file_id:asc",
                "filters": json.dumps(filters),
            }
            res = s.get(
                "https://api.gdc.cancer.gov/files", params=args, timeout=(8, 70)
            )
            res.raise_for_status()
            page = res.json()["data"]
            result.extend(page["hits"])
            if len(result) >= page["pagination"]["total"]:
                break
            if not page["hits"]:
                raise ValueError("Empty GDC page")
    if len({r["file_id"] for r in result}) != len(result):
        raise ValueError("Metadata returned overlapping pages")
    path.write_text(json.dumps(result))
    return result


def check_cohort(cohort, receipt, axis, root, raw_root):
    out = root / (cohort + ".json")
    qt = Path(receipt["path"]) / "GENE_TPM.tsv.gz"
    gd = fetch_metadata(cohort, root)
    by_id = defaultdict(list)
    for hit in gd:
        for e in hit.get("associated_entities", []):
            by_id[e.get("entity_submitter_id")].append(hit)
    table = pd.read_csv(qt, sep="\t", index_col="gene_id")
    axis_map = {r["patient_id"]: r["sample_id"] for r in axis}
    if (
        len(axis_map) != len(axis)
        or set(axis_map) != set(table.columns)
        or not table.index.is_unique
    ):
        raise ValueError("TCGA export and sample authority axes disagree")
    gene_axis = table.index.to_numpy()
    issues = []
    passed = []
    expected_native_axis = None
    native_order = None
    for patient, aliquot in axis_map.items():
        candidates = by_id[aliquot]
        if len(candidates) != 1:
            issues.append(
                {
                    "aliquot": aliquot,
                    "reason": "NON_UNIQUE_NATIVE_FILE",
                    "candidates": len(candidates),
                }
            )
            continue
        hit = candidates[0]
        native = (
            raw_root
            / ("TCGA-" + cohort)
            / "Transcriptome_Profiling"
            / "Gene_Expression_Quantification"
            / hit["file_id"]
            / hit["file_name"]
        )
        if not native.is_file():
            issues.append(
                {"file_id": hit["file_id"], "reason": "NATIVE_FILE_NOT_CACHED"}
            )
            continue
        data = native.read_bytes()
        if (
            len(data) != hit["file_size"]
            or hashlib.md5(data).hexdigest() != hit["md5sum"]
            or data.split(b"\n", 1)[0].strip() != b"# gene-model: GENCODE v36"
        ):
            issues.append(
                {"file_id": hit["file_id"], "reason": "SOURCE_CHECKSUM_OR_MODEL_FAILED"}
            )
            continue
        native_values = pd.read_csv(
            io.BytesIO(data),
            sep="\t",
            comment="#",
            usecols=["gene_id", "tpm_unstranded"],
        )
        this_axis = native_values.gene_id.to_numpy()
        if expected_native_axis is None or not np.array_equal(
            this_axis, expected_native_axis
        ):
            ens = native_values.gene_id
            keep = ens.str.startswith("ENSG") & ~ens.str.endswith("_PAR_Y")
            id_map = ens[keep].str.split(".", n=1, regex=False).str[0]
            if id_map.duplicated().any() or set(id_map) != set(gene_axis):
                issues.append(
                    {"file_id": hit["file_id"], "reason": "INCOMPATIBLE_GENE_AXIS"}
                )
                continue
            native_order = (
                pd.Series(id_map.index.to_numpy(), index=id_map.to_numpy())
                .loc[gene_axis]
                .to_numpy()
            )
            expected_native_axis = this_axis.copy()
        diff = np.abs(
            native_values.tpm_unstranded.to_numpy()[native_order]
            - table[patient].to_numpy()
        )
        if np.any(diff != 0):
            issues.append(
                {
                    "file_id": hit["file_id"],
                    "reason": "EXPRESSION_MISMATCH",
                    "n": int(np.sum(diff != 0)),
                    "max_abs_diff": float(np.nanmax(diff)),
                }
            )
            continue
        passed.append(hit["file_id"])
    result = {
        "cohort": cohort,
        "expected": len(axis),
        "passed": len(passed),
        "genes": len(gene_axis),
        "issues": issues,
        "verified_file_ids": passed,
    }
    out.write_text(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--patient-map",
        type=Path,
        default=Path(
            "/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/"
            "patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/dsk2/user/xumj/ShuJuKu/data/01_bulk/GDCdata"),
    )
    parser.add_argument(
        "--tcga-export",
        type=Path,
        default=Path("/dsk2/user/dengsc/cla_functional_y_20261109_r1/full_tcga_tpm_r1"),
    )
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True, parents=True)
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
    auth = defaultdict(list)
    with args.patient_map.open() as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            auth[r["cancer_id"]].append(r)
    sources = json.loads((args.tcga_export / "COMPLETE.json").read_text())["records"]
    if not auth or set(auth) != set(sources):
        raise ValueError("TCGA cancer scope differs from its sample authority")
    results = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(check_cohort, c, sources[c], rows, args.root, args.raw_root): c
            for c, rows in auth.items()
        }
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                r = fut.result()
                results[c] = {k: r[k] for k in ["expected", "passed", "genes"]}
            except (OSError, ValueError, AssertionError) as exc:
                results[c] = {"error": repr(exc)}
            print(c, results[c], flush=True)
    complete = all(
        "passed" in r and r["expected"] == r["passed"] for r in results.values()
    )
    (args.root / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "PASS" if complete else "REVIEW", "cohorts": results}, indent=2
        )
    )
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
