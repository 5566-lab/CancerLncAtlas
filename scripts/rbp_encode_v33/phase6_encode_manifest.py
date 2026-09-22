"""Phase 6A: file-level ENCODE manifest.

Two independent routes are combined, because the plan's manifest only had the
first one and stopped at experiment level:

Route 1 - the plan's six accessions.  They are **not** experiments.  A search for
``type=Experiment&accession=ENCSR456FVU`` answers HTTP 404 / total 0, while the
control experiment ``ENCSR720BJU`` answers total 1, proving the query form is
valid.  ``/publication-data/<accession>/`` resolves each of the six to a
``PublicationData`` FileSet.  Their embedded ``files`` lists are therefore the
faithful, complete spine of the plan's own input request.

Route 2 - the eCLIP narrowPeak resource itself.  A file-level search over
``assay_title=eCLIP`` returns 1,431 released narrowPeak files carrying a
per-file ``assembly``; 756 are GRCh38-native and 675 are hg19.  The
experiment-level manifest could only report the FileSet's dominant assembly
(hg19) and therefore parked the whole resource as ASSEMBLY_MISMATCH.

Nothing is downloaded.  Every file row is written with the assembly policy that
governs it, so the coordinate question is answered in the manifest rather than
improvised later.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
OUT = WORK / "inputs" / "encode"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
MAN = WORK / "manifests"
MAN.mkdir(parents=True, exist_ok=True)

PROXY = "socks5h://127.0.0.1:1080"
BASE = "https://www.encodeproject.org"
BATCH = 120

#: The six accessions named by the plan, with the role each one plays.
PLAN_SETS = {
    "ENCSR456FVU": ("eclip_peaks_fileset", "Gene Yeo, UCSD"),
    "ENCSR369TWP": ("kd_rnaseq_hepg2", "Brenton Graveley, UConn"),
    "ENCSR795JHH": ("kd_rnaseq_k562", "Brenton Graveley, UConn"),
    "ENCSR413YAF": ("kd_processed_hepg2_k562", "Brenton Graveley, UConn"),
    "ENCSR870OLK": ("kd_batch_corrected", "Brenton Graveley, UConn"),
    "ENCSR876DCD": ("rbns", "Chris Burge, MIT"),
}

FIELDS = [
    "accession", "assembly", "target.label", "target.organism.scientific_name",
    "biosample_ontology.term_name", "biosample_ontology.classification",
    "output_type", "file_format", "file_format_type", "href", "md5sum",
    "file_size", "dataset", "biological_replicates", "technical_replicates",
    "status", "date_created", "lab.title",
]
FIELD_Q = "&field=" + "&field=".join(FIELDS)

#: A control accession whose Experiment-ness is not in doubt.  If this stops
#: resolving, the 404s above stop meaning anything.
CONTROL_EXPERIMENT = "ENCSR720BJU"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(name: str, url: str, expect_http: int = 200) -> tuple[dict, dict]:
    """Fetch a URL to disk, then parse.  The raw bytes are pinned first so a
    schema change cannot erase the evidence of what was returned."""
    target = RAW / f"{name}.json"
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        return payload, {"name": name, "url": url, "cached": True,
                         "file": str(target), "sha256": sha256_file(target)}
    result = subprocess.run(
        ["curl", "-sSL", "--max-time", "600", "--retry", "3", "--proxy", PROXY,
         "-o", str(target), "-w", "%{http_code}", url],
        capture_output=True, text=True,
    )
    code = result.stdout.strip()
    if result.returncode != 0 or code != str(expect_http):
        raise SystemExit(f"FAIL-CLOSED: {name} http={code} (wanted {expect_http}) {result.stderr[:160]}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload, {"name": name, "url": url, "http": int(code),
                     "file": str(target), "sha256": sha256_file(target)}


def normalise(row: dict) -> dict:
    target = row.get("target") or {}
    biosample = row.get("biosample_ontology") or {}
    organism = target.get("organism") or {}
    dataset = str(row.get("dataset") or "")
    lab = row.get("lab")
    return {
        "file_accession": row.get("accession") or "",
        "experiment_accession": dataset.rstrip("/").rsplit("/", 1)[-1] if dataset else "",
        "assembly": row.get("assembly") or "",
        "rbp": target.get("label") or "",
        "organism": organism.get("scientific_name") or "",
        "biosample": biosample.get("term_name") or "",
        "biosample_classification": biosample.get("classification") or "",
        "output_type": row.get("output_type") or "",
        "file_format": row.get("file_format") or "",
        "file_format_type": row.get("file_format_type") or "",
        "href": row.get("href") or "",
        "md5sum": row.get("md5sum") or "",
        "file_size": row.get("file_size") or 0,
        "biological_replicates": ",".join(str(x) for x in (row.get("biological_replicates") or [])),
        "technical_replicates": ",".join(str(x) for x in (row.get("technical_replicates") or [])),
        "status": row.get("status") or "",
        "date_created": row.get("date_created") or "",
        "lab": lab.get("title") if isinstance(lab, dict) else "",
    }


print("=== proving the plan's accessions are not experiments ===", flush=True)
probe = {}
for accession in list(PLAN_SETS) + [CONTROL_EXPERIMENT]:
    payload, meta = get_json(
        f"PROBE_EXPERIMENT_{accession}",
        f"{BASE}/search/?type=Experiment&accession={accession}&format=json&limit=1",
        expect_http=404 if accession != CONTROL_EXPERIMENT else 200,
    )
    probe[accession] = {"http": meta.get("http"), "total": payload.get("total")}
    print(f"  /search/?type=Experiment&accession={accession:<12} http={meta.get('http')} total={payload.get('total')}")
if probe[CONTROL_EXPERIMENT]["total"] != 1:
    raise SystemExit("FAIL-CLOSED: the control experiment no longer resolves; the 404s prove nothing")
if any(probe[a]["total"] for a in PLAN_SETS):
    raise SystemExit("FAIL-CLOSED: a plan accession now resolves as an Experiment; re-read the manifest design")
print(f"  control {CONTROL_EXPERIMENT} resolves => the zero totals above are real, not a bad query")

print("\n=== route 1: the six PublicationData FileSets ===", flush=True)
sets: dict[str, dict] = {}
membership: dict[str, list[str]] = {}
for accession, (role, lab) in PLAN_SETS.items():
    payload, meta = get_json(f"PUBLICATION_DATA_{accession}", f"{BASE}/publication-data/{accession}/?format=json")
    types = payload.get("@type", [])
    if "FileSet" not in types:
        raise SystemExit(f"FAIL-CLOSED: {accession} is not a FileSet ({types})")
    accessions = [str(x).rstrip("/").rsplit("/", 1)[-1] for x in (payload.get("files") or [])]
    sets[accession] = {
        "role": role, "lab": lab, "object_type": types,
        "status": payload.get("status"), "assembly_field": payload.get("assembly"),
        "n_files": len(accessions), "raw": meta,
    }
    membership[accession] = accessions
    print(f"  {accession}  {role:24} files={len(accessions):>5}  assembly_field={payload.get('assembly')}")

all_accessions = sorted({a for accs in membership.values() for a in accs})
print(f"\n  union of the six sets: {len(all_accessions):,} distinct files")

print("\n=== route 1b: batched per-file metadata ===", flush=True)
metadata: dict[str, dict] = {}
n_batches = (len(all_accessions) + BATCH - 1) // BATCH
batch_meta = []
for i in range(0, len(all_accessions), BATCH):
    chunk = all_accessions[i:i + BATCH]
    query = "".join(f"&accession={a}" for a in chunk)
    name = f"FILES_BATCH_{i // BATCH:04d}"
    payload, meta = get_json(name, f"{BASE}/search/?type=File{query}&format=json&limit=all{FIELD_Q}")
    batch_meta.append({"batch": i // BATCH, "n_requested": len(chunk),
                       "total": payload.get("total"), "raw_sha256": meta.get("sha256")})
    for record in payload.get("@graph", []):
        flat = normalise(record)
        metadata[flat["file_accession"]] = flat
    if (i // BATCH) % 20 == 0 or i + BATCH >= len(all_accessions):
        print(f"    batch {i // BATCH + 1:>3}/{n_batches}  resolved={len(metadata):,}")
missing = [a for a in all_accessions if a not in metadata]
print(f"  resolved {len(metadata):,} / {len(all_accessions):,}; unresolved={len(missing)}")

print("\n=== route 2: the eCLIP narrowPeak resource ===", flush=True)
eclip, eclip_meta = get_json(
    "SEARCH_ECLIP_NARROWPEAK",
    f"{BASE}/search/?type=File&file_format=bed&file_format_type=narrowPeak"
    f"&assay_title=eCLIP&status=released&format=json&limit=all{FIELD_Q}")
eclip_rows = [normalise(r) for r in eclip.get("@graph", [])]
print(f"  released eCLIP narrowPeak files: {len(eclip_rows):,} (api total {eclip.get('total')})")

rbns, rbns_meta = get_json(
    "SEARCH_RBNS",
    f"{BASE}/search/?type=File&assay_title=RNA+Bind-n-Seq&status=released"
    f"&format=json&limit=all{FIELD_Q}")
rbns_rows = [normalise(r) for r in rbns.get("@graph", [])]
print(f"  released RNA Bind-n-Seq files:   {len(rbns_rows):,} (api total {rbns.get('total')})")

print("\n=== assembling the manifest ===", flush=True)
rows: list[dict] = []
for accession, accessions in membership.items():
    role = PLAN_SETS[accession][0]
    for file_accession in accessions:
        flat = dict(metadata.get(file_accession) or {"file_accession": file_accession})
        flat["resource"] = role
        flat["plan_fileset"] = accession
        rows.append(flat)

#: The eCLIP narrowPeak resource is larger than the plan's FileSet intersection:
#: the FileSet carries the hg19 files, while the resource also releases 756
#: GRCh38-native files that no FileSet in the plan names.  Both are needed, so the
#: manifest is the union keyed by file accession.
eclip_index = {r["file_accession"] for r in eclip_rows}
seen = {r["file_accession"] for r in rows}
added_from_resource = 0
for flat in eclip_rows:
    if flat["file_accession"] in seen:
        continue
    flat["resource"] = "eclip_narrowpeak_resource"
    flat["plan_fileset"] = ""
    rows.append(flat)
    added_from_resource += 1
print(f"  rows from the plan's six FileSets : {len(seen):,}")
print(f"  rows added from the eCLIP resource: {added_from_resource:,}")
print(f"  manifest rows                     : {len(rows):,}")

for row in rows:
    row["in_eclip_narrowpeak_search"] = row["file_accession"] in eclip_index
    row["download_status"] = "PENDING_DOWNLOAD"

#: Assembly policy.  Stated per file, never inferred at use time.
def policy(row: dict) -> str:
    fmt = row.get("file_format") or ""
    if fmt not in ("bed", "tsv"):
        return "not_needed_non_interval_format"
    assembly = row.get("assembly") or ""
    if assembly == "GRCh38":
        return "native_GRCh38"
    if assembly == "hg19":
        return "explicit_liftover_hg19_to_GRCh38"
    return f"UNSUPPORTED_ASSEMBLY:{assembly or 'missing'}"


for row in rows:
    row["assembly_policy"] = policy(row)

COLUMNS = ["resource", "plan_fileset", "file_accession", "experiment_accession", "assembly",
           "assembly_policy", "rbp", "organism", "biosample", "biosample_classification",
           "output_type", "file_format", "file_format_type", "href", "md5sum", "file_size",
           "biological_replicates", "technical_replicates", "status", "date_created", "lab",
           "in_eclip_narrowpeak_search", "download_status"]

rows.sort(key=lambda r: (r["resource"], r.get("assembly", ""), r.get("rbp", ""),
                         r.get("biosample", ""), r["file_accession"]))
tsv = MAN / "ENCODE_FILE_MANIFEST.tsv"
with tsv.open("w", encoding="utf-8", newline="") as handle:
    handle.write("\t".join(COLUMNS) + "\n")
    for row in rows:
        handle.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")

# --- the surface we will actually consume ------------------------------------
#: Every released eCLIP narrowPeak file, regardless of which FileSet names it.
#: Assembly is carried per row, so the GRCh38-native half needs no liftOver and
#: the hg19 half is explicitly lifted.
def surface(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("file_format") != "bed" or row.get("in_eclip_narrowpeak_search") is not True:
            continue
        if row.get("status") != "released":
            continue
        key = f"{row.get('assembly')}|{row.get('biosample')}"
        b = out.setdefault(key, {"files": 0, "bytes": 0, "rbps": set()})
        b["files"] += 1
        b["bytes"] += int(row.get("file_size") or 0)
        b["rbps"].add(row.get("rbp") or "")
    return out


print("\n=== eCLIP narrowPeak surface (the files Phase 6B will fetch) ===")
surf = surface(rows)
total_files = total_bytes = 0
print(f"{'assembly':10} {'biosample':10} {'files':>6} {'MiB':>9} {'RBPs':>5}")
for key, b in sorted(surf.items()):
    assembly, biosample = key.split("|")
    print(f"{assembly:10} {biosample:10} {b['files']:>6} {b['bytes']/1048576:>9.1f} {len(b['rbps']):>5}")
    total_files += b["files"]; total_bytes += b["bytes"]
print(f"{'TOTAL':10} {'':10} {total_files:>6} {total_bytes/1048576:>9.1f}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "api": "ENCODE REST",
    "proxy": PROXY,
    "plan_accession_audit": {
        "finding": (
            "The six accessions named by the plan are PublicationData FileSets, not "
            "Experiments. A type=Experiment search returns total=0 for all six while "
            "the control experiment resolves, so the plan's manifest could only ever "
            "report the FileSet's dominant assembly."
        ),
        "probe": probe,
        "control_experiment": CONTROL_EXPERIMENT,
    },
    "plan_filesets": {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in sets.items()},
    "plan_fileset_raw": {k: v["raw"] for k, v in sets.items()},
    "fileset_union_files": len(all_accessions),
    "fileset_union_resolved": len(metadata),
    "fileset_union_unresolved": missing[:50],
    "batch_queries": batch_meta,
    "eclip_narrowpeak_search": {"total": eclip.get("total"), "raw": eclip_meta},
    "rbns_search": {"total": rbns.get("total"), "raw": rbns_meta},
    "manifest_rows": len(rows),
    "manifest_columns": COLUMNS,
    "manifest_tsv": str(tsv),
    "manifest_sha256": sha256_file(tsv),
    "eclip_narrowpeak_surface": {
        key: {"files": b["files"], "bytes": b["bytes"], "n_rbps": len(b["rbps"])}
        for key, b in sorted(surf.items())
    },
    "eclip_narrowpeak_surface_files": total_files,
    "eclip_narrowpeak_surface_bytes": total_bytes,
    "downloaded": 0,
    "assembly_policy_values": sorted({r["assembly_policy"] for r in rows}),
}
receipt_path = MAN / "ENCODE_FILE_MANIFEST.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {tsv}")
print(f"written: {receipt_path}")
print(f"manifest sha256 = {receipt['manifest_sha256']}")
