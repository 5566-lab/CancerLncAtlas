"""Fetch public CPTAC GDC STAR Counts, checking every native TSV before promotion."""

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import local

import requests
import urllib3.util.connection

BASE = "https://api.gdc.cancer.gov"
BUCKETS = {
    "CPTAC-2": "gdc-cptac-2-phs000892-2-open",
    "CPTAC-3": "gdc-cptac-phs001287-2-open",
}
FIELDS = (
    "file_id,file_name,file_size,md5sum,data_type,access,analysis.workflow_type,"
    "associated_entities.entity_id,associated_entities.entity_submitter_id,associated_entities.entity_type,"
    "cases.case_id,cases.submitter_id,cases.project.project_id,cases.samples.sample_type,"
    "cases.samples.sample_id,cases.samples.submitter_id,"
    "cases.samples.portions.analytes.aliquots.aliquot_id,cases.samples.portions.analytes.aliquots.submitter_id"
)
state = local()
trust_env = False


def session():
    if not hasattr(state, "client"):
        state.client = requests.Session()
        state.client.trust_env = trust_env
        state.client.headers["User-Agent"] = "CancerLncAtlas/STAR-v36-quantification"
    return state.client


def write_json(dest, data):
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dest.parent, delete=False
    ) as stream:
        temp = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False)
        stream.write("\n")
    try:
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def collect():
    results = []
    for project in ["CPTAC-2", "CPTAC-3"]:
        constraints = {
            "cases.project.project_id": project,
            "data_type": "Gene Expression Quantification",
            "analysis.workflow_type": "STAR - Counts",
            "access": "open",
        }
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": k, "value": [v]}}
                for k, v in constraints.items()
            ],
        }
        start = 0
        while True:
            params = {
                "filters": json.dumps(filters),
                "size": 500,
                "from": start,
                "fields": FIELDS,
                "sort": "file_id:asc",
            }
            for i in range(5):
                try:
                    response = session().get(
                        BASE + "/files", params=params, timeout=(15, 120)
                    )
                    response.raise_for_status()
                    page = response.json()["data"]
                    break
                except (OSError, ValueError) as exc:
                    if i == 4:
                        raise
                    print(
                        "retry metadata", project, start, type(exc).__name__, flush=True
                    )
            results.extend(page["hits"])
            start += len(page["hits"])
            if start >= page["pagination"]["total"]:
                break
            if not page["hits"]:
                raise OSError("Unexpected empty metadata page")
    if len({r["file_id"] for r in results}) != len(results):
        raise ValueError("Overlapping result pages")
    return results


def check_file(path, metadata):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != b"# gene-model: GENCODE v36":
            raise ValueError("Unexpected gene model")
        header = handle.readline().rstrip(b"\r\n").split(b"\t")
        if b"tpm_unstranded" not in header or b"unstranded" not in header:
            raise ValueError("Unexpected STAR Counts fields")
        handle.seek(0)
        while chunk := handle.read(1024 * 1024):
            md5.update(chunk)
            sha.update(chunk)
    if (
        path.stat().st_size != metadata["file_size"]
        or md5.hexdigest() != metadata["md5sum"]
    ):
        raise ValueError("Native GDC file size/checksum mismatch")
    return {
        "file_id": metadata["file_id"],
        "file_name": metadata["file_name"],
        "file_size": path.stat().st_size,
        "md5sum": md5.hexdigest(),
        "sha256": sha.hexdigest(),
        "gene_model": "GENCODE v36",
        "tsv": str(path),
    }


def download(item, root, template):
    fid = item["file_id"]
    final = root / "files" / (fid + ".tsv")
    receipt = root / "receipts" / (fid + ".json")
    if final.exists():
        try:
            verified = check_file(final, item)
        except ValueError:
            pass  # Fetch and verify the replacement before replacing the cached bytes.
        else:
            write_json(receipt, verified)
            return
    for attempt in range(4):
        tmp = None
        try:
            project = item["cases"][0]["project"]["project_id"]
            url = template.format(
                file_id=fid,
                file_name=item["file_name"],
                project=project,
                bucket=BUCKETS[project],
            )
            with session().get(url, stream=True, timeout=(15, 50)) as response:
                response.raise_for_status()
                with NamedTemporaryFile(
                    dir=final.parent, suffix=".partial", delete=False
                ) as dest:
                    tmp = Path(dest.name)
                    for chunk in response.iter_content(128 * 1024):
                        dest.write(chunk)
            verified = check_file(tmp, item)
            verified["tsv"] = str(final)
            os.replace(tmp, final)
            write_json(receipt, verified)
            return
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--proxy-env",
        action="store_true",
        help="Honor HTTP proxy environment variables.",
    )
    parser.add_argument(
        "--template", default="https://{bucket}.s3.amazonaws.com/{file_id}/{file_name}"
    )
    parser.add_argument(
        "--axis",
        type=Path,
        default=Path(
            "/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/aligned_r4/CPTAC_SAMPLE_ASSAY_MAP.tsv"
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    global trust_env
    trust_env = args.proxy_env
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gdc_files.json"
    if not path.exists():
        write_json(path, collect())
    if args.metadata:
        return
    metadata = json.loads(path.read_text())
    with args.axis.open() as handle:
        patients = {
            r["native_case_id"]
            for r in csv.DictReader(handle, delimiter="\t")
            if r["layer"] == "rna"
        }
    selected = [
        f for f in metadata if any(c["submitter_id"] in patients for c in f["cases"])
    ]
    if not selected:
        raise ValueError("No GDC STAR Counts files matched the supplied RNA case axis")
    if len({f["file_id"] for f in selected}) != len(selected):
        raise ValueError("Duplicate file UUID in cached metadata")
    (root / "files").mkdir(exist_ok=True)
    (root / "receipts").mkdir(exist_ok=True)
    write_json(root / "selected_files.json", selected)
    print(
        "candidate files",
        len(selected),
        "total bytes",
        sum(q["file_size"] for q in selected),
        flush=True,
    )
    completed = 0
    errors = {}
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = {
            pool.submit(download, q, root, args.template): q["file_id"]
            for q in selected
        }
        for task in as_completed(tasks):
            try:
                task.result()
                completed += 1
            except (OSError, ValueError) as exc:
                errors[tasks[task]] = repr(exc)
            if (completed + len(errors)) % 25 == 0:
                print("success", completed, "errors", len(errors), flush=True)
    write_json(
        root / "download_status.json",
        {"selected": len(selected), "verified": completed, "errors": errors},
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
