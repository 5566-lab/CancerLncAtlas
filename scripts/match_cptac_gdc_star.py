"""Bind processed CPTAC RNA columns to open GDC STAR files by native case and tissue."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


GDC_TYPE = {"Tumor": "Primary Tumor", "Normal": "Solid Tissue Normal"}
PROJECTS = ["CPTAC-2", "CPTAC-3"]
# Discovery studies used to check the source material; confirmation cohorts
# can share case identifiers and must not determine the RNA file choice.
PDC_STUDIES = {
    "BRCA": (["120"], ["121"]), "COAD": (["109", "116"], ["117"]),
    "OV": (["110", "118"], ["119"]), "CCRCC": (["127"], ["128"]),
    "GBM": (["204"], ["205"]), "HNSCC": (["221"], ["222"]),
    "LSCC": (["234"], ["232"]), "LUAD": (["153"], ["149"]),
    "PDAC": (["270"], ["271"]), "UCEC": (["125"], ["126"]),
}
FIELDS = [
    "file_id", "file_name", "file_size", "md5sum", "access", "data_type",
    "analysis.workflow_type", "analysis.workflow_version", "cases.case_id",
    "cases.project.project_id", "cases.submitter_id", "cases.samples.sample_type",
    "cases.samples.submitter_id", "cases.samples.sample_id",
    "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.samples.portions.analytes.aliquots.submitter_id",
    "associated_entities.entity_id", "associated_entities.entity_submitter_id",
    "associated_entities.entity_type",
]


def http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.trust_env = False
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=0.75, status_forcelist=(429, 500, 502, 503, 504))))
    return session


def fetch_files():
    criteria = {
        "cases.project.project_id": PROJECTS,
        "analysis.workflow_type": ["STAR - Counts"],
        "data_type": ["Gene Expression Quantification"],
        "access": ["open"],
    }
    query = {"op": "and", "content": [
        {"op": "in", "content": {"field": key, "value": vals}}
        for key, vals in criteria.items()
    ]}
    session = http_session()
    results = []
    offset = 0
    total = None
    while total is None or offset < total:
        response = session.get("https://api.gdc.cancer.gov/files", params={
            "filters": json.dumps(query), "fields": ",".join(FIELDS),
            "size": 500, "from": offset, "sort": "file_id:asc",
        }, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()["data"]
        if total is None:
            total = data["pagination"]["total"]
        assert total == data["pagination"]["total"] and data["hits"]
        results.extend(data["hits"])
        offset += len(data["hits"])
    assert len({row["file_id"] for row in results}) == total
    return results


def fetch_pdc():
    session = http_session()
    studies = sorted({"PDC000" + number for pair in PDC_STUDIES.values()
                      for assay in pair for number in assay})
    result = {}
    for study in studies:
        query = (
            '{ biospecimenPerStudy(pdc_study_id: "' + study + '") { '
            'aliquot_id aliquot_submitter_id sample_id sample_submitter_id case_id '
            'case_submitter_id sample_type pool project_name } }')
        response = session.get("https://pdc.cancer.gov/graphql",
                               params={"query": query}, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()
        if data.get("errors") or not data.get("data", {}).get("biospecimenPerStudy"):
            raise ValueError(f"Unable to freeze PDC discovery study {study}")
        result[study] = data["data"]["biospecimenPerStudy"]
    return result


def load_snapshot(path, fetch):
    if not path.is_file():
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(fetch(), separators=(",", ":")), encoding="utf-8")
        partial.replace(path)
    return json.loads(path.read_text(encoding="utf-8"))


def index_samples(files):
    samples = defaultdict(list)
    for row in files:
        if (row["access"] != "open" or row["analysis"]["workflow_type"] != "STAR - Counts"
                or row["data_type"] != "Gene Expression Quantification"):
            raise ValueError(f"Unexpected GDC file type {row['file_id']}")
        case, = row["cases"]
        entity, = row["associated_entities"]
        if case["project"]["project_id"] not in PROJECTS or entity["entity_type"] != "aliquot":
            raise ValueError(f"Unexpected GDC project or association {row['file_id']}")
        source = []
        for sample in case["samples"]:
            aliquot_ids = {a["aliquot_id"] for p in sample["portions"] for n in p["analytes"]
                           for a in n["aliquots"]}
            if entity["entity_id"] in aliquot_ids:
                source.append(sample)
        if not source:
            raise ValueError(f"GDC aliquot has no parent specimen {row['file_id']}")
        tissue_types = {s.get("sample_type") for s in source}
        if len(tissue_types) == 1 and tissue_types <= set(GDC_TYPE.values()):
            samples[case["submitter_id"], next(iter(tissue_types))].append((row, case, source, entity))
    return samples


def index_pdc(records):
    index = defaultdict(list)
    for cohort, (protein, phospho) in PDC_STUDIES.items():
        for assay, studies in (("protein", protein), ("phospho", phospho)):
            for number in studies:
                study = "PDC000" + number
                for x in records[study]:
                    if x["pool"] != "No":
                        continue
                    members = frozenset(v.strip() for v in x["sample_submitter_id"].split(","))
                    index[cohort, x["case_submitter_id"], x["sample_type"]].append(
                        (members, study, assay))
    return index


def bind(column, by_case, pdc):
    case_id = column["native_case_id"]
    tissue = GDC_TYPE[column["tissue_type"]]
    choices = by_case.get((case_id, tissue), [])
    studies = pdc.get((column["source_cohort"], case_id, tissue), [])
    scored = []
    for choice in choices:
        parents = frozenset(s["submitter_id"] for s in choice[2])
        exact = [(study, assay) for ids, study, assay in studies if ids == parents]
        overlap = [(study, assay) for ids, study, assay in studies if ids & parents]
        rank = 2 if exact else int(bool(overlap))
        scored.append((choice, rank, exact, overlap))
    best = [item for item in scored if item[1] == max(v[1] for v in scored)]
    keys = {
        "match_status": "MISSING", "match_basis": "NO_COMPATIBLE_CASE_TISSUE",
        "gdc_file_id": "", "gdc_md5": "", "gdc_sample_id": "", "gdc_sample_submitter_id": "",
        "gdc_aliquot_id": "", "gdc_aliquot_submitter_id": "", "gdc_project_id": "",
        "gdc_case_id": "", "gdc_sample_type": "", "candidate_count": len(choices),
        "candidate_file_ids": "|".join(sorted(x[0]["file_id"] for x in choices)),
        "strict_pdc_sample_match": 0, "strict_pdc_protein_match": 0,
        "strict_pdc_phospho_match": 0, "pdc_supporting_studies": "",
    }
    if len(best) > 1:
        keys.update(match_status="AMBIGUOUS", match_basis="MULTIPLE_COMPATIBLE_GDC_FILES_OR_SPECIMENS")
    elif best:
        (row, case, parents, aliquot), rank, exact, overlap = best[0]
        reasons = {0: "UNIQUE_NATIVE_CASE_AND_TISSUE", 1: "PDC_PARTIAL_SOURCE_SPECIMEN_OVERLAP",
                   2: "PDC_EXACT_SOURCE_SPECIMEN_SET"}
        keys.update(
            match_status="MATCHED", match_basis=reasons[rank], gdc_sample_type=tissue,
            gdc_file_id=row["file_id"], gdc_md5=row["md5sum"],
            gdc_sample_id="|".join(sorted({s["sample_id"] for s in parents})),
            gdc_sample_submitter_id="|".join(sorted({s["submitter_id"] for s in parents})),
            gdc_project_id=case["project"]["project_id"], gdc_case_id=case["case_id"],
            gdc_aliquot_id=aliquot["entity_id"], gdc_aliquot_submitter_id=aliquot["entity_submitter_id"],
            strict_pdc_sample_match=int(bool(exact)),
            strict_pdc_protein_match=int(any(assay == "protein" for _, assay in exact)),
            strict_pdc_phospho_match=int(any(assay == "phospho" for _, assay in exact)),
            pdc_supporting_studies="|".join(sorted({s for s, _ in exact or overlap})),
        )
    return {**column, **keys}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assay-map", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pdc-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    by_case = index_samples(load_snapshot(args.metadata, fetch_files))
    pdc = index_pdc(load_snapshot(args.pdc_metadata, fetch_pdc))
    with args.assay_map.open(encoding="utf-8", newline="") as fh:
        columns = [r for r in csv.DictReader(fh, delimiter="\t") if r["layer"] == "rna"]
    if not columns or len({r["sample_id"] for r in columns}) != len(columns):
        raise ValueError("Invalid RNA sample axis")
    matches = [bind(row, by_case, pdc) for row in columns]
    matched = [r["gdc_file_id"] for r in matches if r["match_status"] == "MATCHED"]
    if len(set(matched)) != len(matched):
        raise ValueError("GDC RNA file would be assigned to multiple source columns")
    dest = args.output / "GDC_SAMPLE_MATCH.tsv"
    tmp = dest.with_suffix(".partial")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(matches[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(matches)
    tmp.replace(dest)
    counts = Counter((r["source_cohort"], r["tissue_type"], r["match_status"], r["match_basis"])
                     for r in matches)
    with (args.output / "MATCH_COUNTS.tsv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["cohort", "tissue", "status", "basis", "n"])
        writer.writerows([*k, n] for k,n in sorted(counts.items()))
    receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (args.assay_map, args.metadata, args.pdc_metadata, dest)}
    receipt.update(n_source_rna_columns=len(matches), n_matched=len(matched),
                   basis=dict(Counter(r["match_basis"] for r in matches)))
    (args.output/"MATCH_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
