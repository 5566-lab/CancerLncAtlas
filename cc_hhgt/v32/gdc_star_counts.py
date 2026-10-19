"""Pinned GDC STAR acquisition and exact count assembly; no normalization."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from .cnv_input_sources import require_host
from .cnv_only_training import TCGA_CANCERS
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record

API = "https://api.gdc.cancer.gov"
QUERY = "GDC_STAR_COUNTS_QUERY_V1"
INPUT = "GDC_STAR_COUNT_ASSEMBLY_INPUT_V1"
FORMAT = "GDC_STAR_COUNT_ASSEMBLY_V1"
FIELDS = "file_id,file_name,md5sum,file_size,state,access,data_category,data_type,data_format,experimental_strategy,analysis.workflow_type,associated_entities.case_id,associated_entities.entity_type,associated_entities.entity_id,associated_entities.entity_submitter_id,cases.case_id,cases.submitter_id,cases.project.project_id,cases.samples.sample_id,cases.samples.submitter_id,cases.samples.portions.analytes.aliquots.aliquot_id,cases.samples.portions.analytes.aliquots.submitter_id"
ROOT = Path(__file__).resolve().parents[2]
R_SCRIPT = ROOT / "scripts/assemble_gdc_star_counts.R"


def filters(cancers):
    return {"op": "and", "content": [{"op": "in", "content": {"field": k, "value": v}} for k, v in (
        ("cases.project.project_id", ["TCGA-" + c for c in cancers]),
        ("data_type", ["Gene Expression Quantification"]), ("analysis.workflow_type", ["STAR - Counts"]),
        ("experimental_strategy", ["RNA-Seq"]), ("access", ["open"]), ("state", ["released"]))]}


def post(payload):
    request = urllib.request.Request(API + "/files", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response: return json.load(response)


def uuid(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise ValueError("Invalid GDC UUID")
    return value


def sample_record(hit, cancers):
    expected = {"access": "open", "state": "released", "data_type": "Gene Expression Quantification",
        "data_category": "Transcriptome Profiling", "experimental_strategy": "RNA-Seq", "data_format": "TSV"}
    if any(hit.get(k) != v for k, v in expected.items()) or hit.get("analysis", {}).get("workflow_type") != "STAR - Counts":
        raise ValueError("GDC metadata is not released open STAR count data")
    file_id = uuid(hit["file_id"]); name = hit["file_name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.tsv", name) or name in {".", ".."}:
        raise ValueError("Invalid STAR file name")
    md5 = hit["md5sum"]
    if not re.fullmatch(r"[a-fA-F0-9]{32}", md5) or type(hit["file_size"]) is not int or hit["file_size"] <= 0:
        raise ValueError("Missing GDC size/checksum")
    if len(hit.get("cases", [])) != 1 or len(hit.get("associated_entities", [])) != 1:
        raise ValueError("Ambiguous GDC case/aliquot association")
    case = hit["cases"][0]; entity = hit["associated_entities"][0]
    project = case["project"]["project_id"]; patient = case["submitter_id"]
    if project not in {"TCGA-" + c for c in cancers} or not re.fullmatch(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}", patient):
        raise ValueError("GDC patient/project identity differs")
    case_id = uuid(case["case_id"])
    if entity.get("entity_type") != "aliquot" or entity.get("case_id") != case_id:
        raise ValueError("GDC associated entity is not this case's aliquot")
    matches = []
    for sample in case.get("samples", []):
        for portion in sample.get("portions", []):
            for analyte in portion.get("analytes", []):
                for aliquot in analyte.get("aliquots", []):
                    if aliquot.get("aliquot_id") == entity.get("entity_id"):
                        matches.append((sample, aliquot))
    if len(matches) != 1: raise ValueError("GDC aliquot does not map to exactly one sample")
    sample, aliquot = matches[0]; barcode = aliquot["submitter_id"]
    if (barcode != entity.get("entity_submitter_id")
            or not re.fullmatch(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-[0-9]{2}[A-Za-z]-[0-9]{2}[A-Za-z]-[A-Za-z0-9]{4}-[0-9]{2}", barcode)
            or barcode[:12] != patient or barcode[:16] != sample.get("submitter_id")):
        raise ValueError("GDC aliquot/sample/patient barcodes disagree")
    return {"file_id": file_id, "file_name": name, "md5sum": md5.lower(), "file_size": hit["file_size"],
        "cancer_id": project[5:], "source_dataset": project, "patient_id": patient, "sample_id": barcode,
        "specimen_sample_id": sample["submitter_id"], "sample_uuid": uuid(sample["sample_id"]),
        "case_uuid": case_id, "aliquot_uuid": uuid(aliquot["aliquot_id"])}


def replay_pages(value, root):
    cancers = value.get("cancers")
    if (value.get("format") != QUERY or value.get("target_host") != "149" or value.get("training_started") is not False
            or not cancers or cancers != sorted(set(cancers)) or not set(cancers) <= set(TCGA_CANCERS)):
        raise ValueError("Invalid GDC query declaration")
    records = []; total = None
    for source in value.get("pages", []):
        page = read(verified_path(source, root)); request = page["request"]; data = page["response"]["data"]
        if (request != {"filters": filters(cancers), "fields": FIELDS, "format": "JSON", "sort": "file_id:asc",
                        "from": len(records), "size": 500} or page.get("endpoint") != API + "/files"):
            raise ValueError("GDC page query scope differs")
        pagination = data["pagination"]; hits = data["hits"]
        if (type(pagination.get("total")) is not int or pagination["total"] <= 0
                or pagination.get("from") != len(records) or pagination.get("count") != len(hits) or not hits
                or (total is not None and total != pagination["total"])):
            raise ValueError("GDC pagination closure differs")
        total = pagination["total"]
        records.extend(sample_record(hit, cancers) for hit in hits)
    if (len(records) != total or len({r["file_id"] for r in records}) != len(records)
            or len({r["sample_id"] for r in records}) != len(records)
            or len({r["aliquot_uuid"] for r in records}) != len(records)
            or {r["cancer_id"] for r in records} != set(cancers)):
        raise ValueError("Missing, duplicate or ambiguous GDC file/sample coverage")
    # A UUID must not silently identify different people/specimens in two
    # otherwise individually valid file records. Additional aliquots remain
    # valid when their shared case/sample mappings agree.
    for keys, values in ((["case_uuid"], ["patient_id", "source_dataset"]),
                         (["source_dataset", "patient_id"], ["case_uuid"]),
                         (["sample_uuid"], ["case_uuid", "specimen_sample_id"]),
                         (["case_uuid", "specimen_sample_id"], ["sample_uuid"])):
        mapping = {}
        for item in records:
            key, identity = tuple(item[k] for k in keys), tuple(item[k] for k in values)
            if key in mapping and mapping[key] != identity:
                raise ValueError("Conflicting GDC identity mapping across files")
            mapping[key] = identity
    return sorted(records, key=lambda r: (r["cancer_id"], r["sample_id"]))


def query(cancers, output, *, request=post):
    require_host(); output = Path(output).resolve()
    cancers = sorted(cancers)
    if not cancers or len(set(cancers)) != len(cancers) or not set(cancers) <= set(TCGA_CANCERS):
        raise ValueError("Invalid requested TCGA cancers")
    if output.exists(): raise FileExistsError("Use a new GDC metadata output")
    output.mkdir(parents=True)
    write(output / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False, "operation": "GDC_STAR_METADATA_QUERY"})
    pages = []; offset = 0
    while True:
        payload = {"filters": filters(cancers), "fields": FIELDS, "format": "JSON", "sort": "file_id:asc", "from": offset, "size": 500}
        response = request(payload); hits = response["data"]["hits"]
        path = output / f"page_{len(pages):04}.json"
        write(path, {"endpoint": API + "/files", "request": payload, "response": response,
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
        pages.append(record(path)); offset += len(hits)
        if not hits or offset >= response["data"]["pagination"]["total"]: break
    value = {"format": QUERY, "target_host": "149", "training_started": False, "cancers": cancers, "pages": pages}
    value["files"] = replay_pages(value, output)
    write(output / "QUERY.json", value)
    return output / "QUERY.json"


def digest_md5(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "md5").hexdigest()


def check_raw(path, item):
    if Path(path).stat().st_size != item["file_size"] or digest_md5(path) != item["md5sum"]:
        raise ValueError("STAR raw file size/MD5 differs from GDC metadata")


def acquire(item, cache, *, download=False, opener=urllib.request.urlopen):
    path = Path(cache).resolve() / item["file_id"] / item["file_name"]
    if not path.is_file():
        if not download: raise FileNotFoundError("Verified STAR download missing: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        if temporary.exists(): raise FileExistsError("Unfinished STAR download needs inspection")
        try:
            with opener(API + "/data/" + item["file_id"], timeout=90) as response, temporary.open("xb") as handle:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > item["file_size"]: raise ValueError("STAR download exceeds metadata size")
                    handle.write(block)
            check_raw(temporary, item); temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    check_raw(path, item)
    return record(path)


def r_call(*args):
    require_host()
    result = subprocess.run(["/usr/bin/Rscript", str(R_SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=600)
    if result.returncode: raise ValueError("STAR count assembly/replay failed: " + result.stderr[-2500:])


def materialize(manifest, cancer, cache, output, *, download=False):
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve()
    value = read(manifest); rows = replay_pages(value, manifest.parent)
    if value.get("files") != rows: raise ValueError("GDC query file mapping differs from archived API pages")
    rows = [r for r in rows if r["cancer_id"] == cancer]
    if not rows: raise ValueError("Cancer is absent from declared GDC query")
    if output.exists(): raise FileExistsError("Use a new STAR count assembly output")
    output.mkdir(parents=True)
    write(output / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False, "operation": "GDC_STAR_COUNT_ASSEMBLY"})
    files = [{**r, "raw": acquire(r, cache, download=download)} for r in rows]
    spec = {"format": INPUT, "target_host": "149", "training_started": False, "cancer_id": cancer,
        "query": record(manifest), "files": files, "implementations": [record(__file__), record(R_SCRIPT)]}
    inputs = output / "COUNT_INPUTS.json"; write(inputs, spec)
    counts = output / ("TCGA-" + cancer + ".RDS")
    r_call(inputs, counts)
    samples = output / "SAMPLE_SOURCES.tsv"
    pd.DataFrame([{**{k: v for k, v in r.items() if k != "raw"}, "raw_file_sha256": r["raw"]["sha256"]} for r in files]).to_csv(samples, sep="\t", index=False)
    receipt = {"format": FORMAT, "target_host": "149", "training_started": False, "cancer_id": cancer,
        "inputs": record(inputs), "output": record(counts), "samples": record(samples),
        "assays": ["unstranded", "stranded_first", "stranded_second"], "normalization_performed": False,
        "upstream_alignment_verified": False}
    path = Path(str(counts) + ".sources.json"); write(path, receipt)
    validate(record(path), output, counts, replay=True, _allow_unfinished=True)
    write(output / "SUCCESS.json", {"status": "PASS", "target_host": "149", "training_started": False, "source": record(path)})
    return counts


def validate(item, root, counts, *, replay=False, _allow_unfinished=False):
    path = verified_path(item, root); value = read(path)
    if not _allow_unfinished:
        success_path = path.parent / "SUCCESS.json"
        if not success_path.is_file(): raise ValueError("STAR count assembly has no completed SUCCESS")
        success = read(success_path)
        if (success.get("status") != "PASS" or success.get("target_host") != "149"
                or success.get("training_started") is not False or success.get("source") != record(path)):
            raise ValueError("STAR count completed SUCCESS binding differs")
    if (value.get("format") != FORMAT or value.get("target_host") != "149" or value.get("training_started") is not False
            or value.get("normalization_performed") is not False or value.get("upstream_alignment_verified") is not False
            or value.get("assays") != ["unstranded", "stranded_first", "stranded_second"]
            or verified_path(value["output"], path.parent) != Path(counts).resolve()):
        raise ValueError("Invalid STAR count source binding")
    inputs = verified_path(value["inputs"], path.parent); spec = read(inputs)
    if (spec.get("format") != INPUT or spec.get("target_host") != "149" or spec.get("training_started") is not False
            or spec.get("cancer_id") != value["cancer_id"] or len(spec.get("implementations", [])) != 2):
        raise ValueError("Invalid STAR count input declaration")
    query_path = verified_path(spec["query"], inputs.parent); query_value = read(query_path)
    expected = replay_pages(query_value, query_path.parent)
    if query_value.get("files") != expected: raise ValueError("STAR query mapping changed")
    expected = [r for r in expected if r["cancer_id"] == value["cancer_id"]]
    if not expected or [{k: v for k, v in r.items() if k != "raw"} for r in spec["files"]] != expected:
        raise ValueError("STAR count samples differ from GDC metadata")
    for implementation in spec["implementations"]: verified_path(implementation, inputs.parent)
    samples = verified_path(value["samples"], path.parent)
    wanted = pd.DataFrame([{**r, "raw_file_sha256": s["raw"]["sha256"]} for r, s in zip(expected, spec["files"])])
    try: pd.testing.assert_frame_equal(pd.read_csv(samples, sep="\t", keep_default_na=False), wanted, check_dtype=False)
    except AssertionError as error: raise ValueError("STAR sample inventory differs") from error
    if replay:
        for source in spec["files"]: check_raw(verified_path(source["raw"], inputs.parent), source)
        r_call("--verify", inputs, counts)
    return path, value, spec


def inspect(inventory, normalization, root, model, role, fold, *, phase="FEATURE_CONSTRUCTION"):
    item = normalization.get("counts_assembly")
    counts = normalization["inputs"]["counts"]
    if item is None:
        inventory.gap(model, role, "EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED", fold, counts["sha256"])
        return
    key = ("gdc_star_counts", item["sha256"])
    path, value, spec = validate(item, root, verified_path(counts, root), replay=key not in inventory.visited)
    inventory.visited.add(key)
    inventory.bind(item, root, model, role + ".star_counts", fold)
    for name in ("inputs", "output", "samples"): inventory.bind(value[name], path.parent, model, role + ".star." + name, fold)
    inputs = Path(value["inputs"]["path"])
    query_path = inventory.bind(spec["query"], inputs.parent, model, role + ".star.query", fold)
    for i, source in enumerate(read(query_path)["pages"]): inventory.bind(source, query_path.parent, model, role + ".star.api." + str(i), fold)
    for i, source in enumerate(spec["implementations"]): inventory.bind(source, inputs.parent, model, role + ".star.implementation." + str(i), fold)
    for i, source in enumerate(spec["files"]): inventory.bind(source["raw"], inputs.parent, model, role + ".star.raw." + str(i), fold)
    # validate() above replays archived API metadata, raw STAR bytes, RDS values
    # and this exact sample table. Do not append a generic unknown-table ancestry
    # gap to a transformation that has just been reconstructed.
    from .current_auxiliary_model_use import source_metadata
    source_metadata(inventory, verified_path(value["samples"],path.parent), model,
                    role + ".star.sample_exposure", fold, phase=phase)
    inventory.gap(model, role, "GDC_UPSTREAM_ALIGNMENT_AND_BIOSPECIMEN_VERACITY_UNVERIFIED", fold, counts["sha256"])
