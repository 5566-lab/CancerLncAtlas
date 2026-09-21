"""Phase 5: build the ENCODE download manifest.

Records every requested accession, its authoritative assembly, and -- because the
project annotation is GRCh38 while every requested resource is hg19 -- the reason
each one was NOT downloaded.  Nothing is downloaded under an assembly mismatch.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1/manifests")
OUT.mkdir(parents=True, exist_ok=True)

REQUESTED = {
    "ENCSR456FVU": ("eCLIP reproducible peaks", "primary"),
    "ENCSR369TWP": ("RBP knockdown RNA-seq (HepG2)", "kd"),
    "ENCSR795JHH": ("RBP knockdown RNA-seq (K562)", "kd"),
    "ENCSR413YAF": ("processed DESeq/rMATS/MISO/Cuffdiff", "kd_secondary"),
    "ENCSR870OLK": ("batch-corrected expression/splicing", "kd_secondary"),
    "ENCSR876DCD": ("RBNS sequence preference", "rbns"),
}

PROJECT_ASSEMBLY = "GRCh38"

FIELDS = [
    "resource_type", "publication_set_accession", "experiment_accession",
    "file_accession", "RBP", "cell_line", "assembly", "file_format",
    "output_type", "source_url", "download_date", "sha256", "bytes", "status",
    "decision_reason",
]


def get(url: str) -> dict:
    raw = subprocess.run(
        ["curl", "-sSL", "--max-time", "120", "-H", "Accept: application/json", url],
        capture_output=True, text=True,
    ).stdout
    try:
        return json.loads(raw)
    except Exception:
        return {}


def sample_files(experiment: dict, limit: int = 40) -> list[dict]:
    out = []
    for fid in (experiment.get("files") or [])[:limit]:
        f = get(f"https://www.encodeproject.org{fid}?format=json")
        if f:
            out.append(f)
    return out


def label(value) -> str:
    if isinstance(value, dict):
        return str(value.get("label") or value.get("title") or value.get("term_name") or "")
    return str(value or "")


today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
rows: list[dict] = []
audit: dict[str, dict] = {}

for accession, (resource_type, kind) in REQUESTED.items():
    exp = get(f"https://www.encodeproject.org/experiments/{accession}/?format=json")
    assemblies = exp.get("assembly") or []
    files = sample_files(exp)
    file_assemblies = sorted({str(f.get("assembly")) for f in files if f.get("assembly")})
    peaks = [f for f in files
             if f.get("output_type") == "peaks" and f.get("status") == "released"]

    audit[accession] = {
        "resource_type": resource_type,
        "experiment_assembly": assemblies,
        "sampled_files": len(files),
        "file_assemblies_observed": file_assemblies,
        "released_peak_files_sampled": len(peaks),
        "target": label(exp.get("target")),
        "biosample": label(exp.get("biosample_ontology")),
        "status": exp.get("status"),
        "description": (exp.get("description") or "")[:200],
    }

    has_project_assembly = any(
        str(a).lower() in {"grch38", "hg38"} for a in assemblies
    ) or any(str(a).lower() in {"grch38", "hg38"} for a in file_assemblies)

    if not has_project_assembly:
        rows.append({
            "resource_type": resource_type,
            "publication_set_accession": label(exp.get("publication_set")),
            "experiment_accession": accession,
            "file_accession": "",
            "RBP": label(exp.get("target")),
            "cell_line": label(exp.get("biosample_ontology")),
            "assembly": "|".join(str(a) for a in assemblies),
            "file_format": "",
            "output_type": "",
            "source_url": f"https://www.encodeproject.org/experiments/{accession}/",
            "download_date": "",
            "sha256": "",
            "bytes": "",
            "status": "NOT_DOWNLOADED",
            "decision_reason": (
                f"ASSEMBLY_MISMATCH: project annotation is {PROJECT_ASSEMBLY}; "
                f"resource provides {assemblies or 'unknown'}; silent liftOver is forbidden"
            ),
        })
    else:
        for f in peaks:
            href = f.get("href") or ""
            rows.append({
                "resource_type": resource_type,
                "publication_set_accession": label(exp.get("publication_set")),
                "experiment_accession": accession,
                "file_accession": f.get("accession"),
                "RBP": label(exp.get("target")),
                "cell_line": label(exp.get("biosample_ontology")),
                "assembly": str(f.get("assembly")),
                "file_format": str(f.get("file_format")),
                "output_type": str(f.get("output_type")),
                "source_url": f"https://www.encodeproject.org{href}" if href else "",
                "download_date": today,
                "sha256": "",
                "bytes": str(f.get("file_size") or ""),
                "status": "PENDING_DOWNLOAD",
                "decision_reason": "assembly matches project annotation",
            })

tsv = OUT / "ENCODE_DOWNLOAD_MANIFEST.tsv"
with tsv.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "project_assembly": PROJECT_ASSEMBLY,
    "generated": datetime.now(timezone.utc).isoformat(),
    "requested_accessions": list(REQUESTED),
    "audit": audit,
    "manifest_rows": len(rows),
    "downloaded": sum(1 for r in rows if r["status"] == "DOWNLOADED"),
    "not_downloaded": sum(1 for r in rows if r["status"] == "NOT_DOWNLOADED"),
}
(OUT / "ENCODE_DOWNLOAD_MANIFEST.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print(f"project assembly : {PROJECT_ASSEMBLY}")
print(f"manifest rows    : {len(rows)}  -> {tsv}")
print()
for accession, info in audit.items():
    verdict = "ASSEMBLY MISMATCH -> NOT DOWNLOADED" \
        if not any(str(a).lower() in {"grch38", "hg38"} for a in info["experiment_assembly"]) \
        else "assembly OK"
    print(f"  {accession:12s} experiment_asm={info['experiment_assembly']} "
          f"observed={info['file_assemblies_observed']}  {verdict}")
