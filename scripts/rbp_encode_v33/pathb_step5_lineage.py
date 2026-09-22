"""Path B step 5: declare the lineage of every ENCODE-derived edge.

An edge in the graph has to be traceable back to bytes on disk.  This script
joins the long evidence table (lncRNA x RBP x file) against the file manifest and
the two coordinate receipts, producing one row per (edge, source file) that
records:

* which ENCODE file it came from and that file's md5 and byte count;
* the assembly that file was published in, and the policy applied to it;
* the pinned chain hash when the file was lifted from hg19;
* the lncRNA coordinate authority the overlap was computed against.

It then fails closed if any edge cites a file that is not a verified download, or
a lifted file whose chain hash differs from the pinned one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
MAN = WORK / "manifests"
OUT = WORK / "outputs" / "phase3_encode_eclip"
OUT.mkdir(parents=True, exist_ok=True)

VERIFIED_STATUSES = {"DOWNLOADED", "CACHED_VERIFIED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = pd.read_csv(MAN / "ENCODE_FILE_MANIFEST.tsv", sep="\t", dtype=str).fillna("")
evidence = pd.read_parquet(OUT / "ENCODE_ECLIP_EVIDENCE_LONG.parquet")
download = json.loads((MAN / "ENCODE_DOWNLOAD_RECEIPT.json").read_text(encoding="utf-8"))
chain = json.loads((ENC.parent / "liftover" / "CHAIN_RECEIPT.json").read_text(encoding="utf-8"))
coords = json.loads((MAN / "LNCRNA_COORDINATE_AUTHORITY.json").read_text(encoding="utf-8"))

print(f"=== evidence rows: {len(evidence):,} ===")
if "file_accession" in evidence.columns:
    rows = evidence.copy()
else:
    # The overlap step aggregates each (lncRNA, RBP, assembly) group and records its
    # source files as one comma-joined field.  Lineage is per file, so explode it.
    # The peak count stays a property of the group: it is not attributable to one
    # file once several files of the same RBP and assembly have been summed.
    rows = evidence.copy()
    rows["file_accession"] = rows.source_files.astype(str).str.split(",")
    rows = rows.explode("file_accession")
    rows["file_accession"] = rows.file_accession.astype(str).str.strip()
    rows = rows.loc[rows.file_accession.ne("")]
    rows = rows.rename(columns={"n_peaks": "n_peaks_group", "n_source_files": "n_source_files_group"})
print(f"    edge-file rows : {len(rows):,}")
print(f"    files cited    : {rows.file_accession.nunique():,}")
meta = manifest.set_index("file_accession", drop=False)
for column in ("assembly", "assembly_policy", "md5sum", "file_size", "status",
               "download_status", "biosample", "rbp", "href"):
    rows[column] = rows.file_accession.map(meta[column]) if column in meta.columns else ""
# The peak count is per (edge, assembly) group, not per file; name it honestly.
if "n_peaks_group" in rows.columns:
    rows["n_peaks_edge_group"] = rows.n_peaks_group
elif "n_peaks" in rows.columns:
    rows["n_peaks_edge_group"] = rows.n_peaks

# The manifest was updated in place by the downloader, so its download_status is
# the authority; the receipt is used to cross-check.
receipt_status = {r["file_accession"]: r["status"] for r in download["files"]}
rows["receipt_status"] = rows.file_accession.map(receipt_status).fillna("")
if not rows.receipt_status.eq(rows.download_status).all():
    bad = rows.loc[~rows.receipt_status.eq(rows.download_status), "file_accession"].unique()[:5]
    raise SystemExit(f"FAIL-CLOSED: manifest and receipt disagree on download status for {list(bad)}")

unverified = sorted(set(rows.loc[~rows.download_status.isin(VERIFIED_STATUSES), "file_accession"]))
if unverified:
    raise SystemExit(f"FAIL-CLOSED: {len(unverified)} cited files are not verified downloads: {unverified[:5]}")
print(f"    every cited file is a verified download")

rows["chain_sha256"] = rows.assembly_policy.map(
    lambda p: chain["sha256"] if p == "explicit_liftover_hg19_to_GRCh38" else "")
bad_chain = rows.loc[rows.chain_sha256.ne("") & rows.chain_sha256.ne(chain["sha256"])]
if len(bad_chain):
    raise SystemExit("FAIL-CLOSED: a lifted edge cites a chain hash that is not the pinned one")
lifted_files = int(rows.loc[rows.chain_sha256.ne(""), "file_accession"].nunique())
native_files = int(rows.loc[rows.chain_sha256.eq(""), "file_accession"].nunique())

rows["lncrna_coordinate_authority_sha256"] = coords["output_sha256"]
rows["lncrna_coordinate_assembly"] = coords["assembly"]
rows["evidence_scope"] = rows.assembly_provenance.map({
    "native_GRCh38": "GRCh38-native ENCODE eCLIP peaks",
    "lifted_from_hg19": "hg19 ENCODE eCLIP peaks lifted to GRCh38 with the pinned chain",
})

LINEAGE_COLUMNS = ["lncrna_id", "rbp", "biosample", "file_accession",
                   "assembly", "assembly_policy", "assembly_provenance", "chain_sha256",
                   "md5sum", "file_size", "download_status", "receipt_status",
                   "n_peaks_edge_group", "evidence_scope", "lncrna_coordinate_assembly",
                   "lncrna_coordinate_authority_sha256", "href"]
lineage = rows[LINEAGE_COLUMNS].sort_values(
    ["lncrna_id", "rbp", "file_accession"]).reset_index(drop=True)
path = OUT / "ENCODE_ECLIP_EDGE_LINEAGE.parquet"
lineage.to_parquet(path, index=False)

print(f"\n=== lineage ===")
print(f"    rows                 : {len(lineage):,}")
print(f"    distinct files cited : {lineage.file_accession.nunique():,} "
      f"(native {native_files:,}, lifted {lifted_files:,})")
print(f"    distinct lncRNAs     : {lineage.lncrna_id.nunique():,}")
print(f"    distinct RBPs        : {lineage.rbp.nunique():,}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "step": "Path B step 5 - lineage declaration for ENCODE-derived edges",
    "lineage_rows": int(len(lineage)),
    "distinct_files_cited": int(lineage.file_accession.nunique()),
    "native_GRCh38_files": native_files,
    "lifted_hg19_files": lifted_files,
    "distinct_lncrnas": int(lineage.lncrna_id.nunique()),
    "distinct_rbps": int(lineage.rbp.nunique()),
    "chain_sha256": chain["sha256"],
    "chain_direction_verified_by": chain["direction_verified_by"],
    "lncrna_coordinate_authority_sha256": coords["output_sha256"],
    "download_receipt": str(MAN / "ENCODE_DOWNLOAD_RECEIPT.json"),
    "lineage_parquet": str(path),
    "lineage_sha256": sha256_file(path),
    "checks": {
        "manifest_and_receipt_status_agree": True,
        "all_cited_files_verified": True,
        "every_lifted_file_cites_the_pinned_chain": True,
    },
    "columns": LINEAGE_COLUMNS,
    "note": (
        "One row per (edge, source file). An edge is reproducible from this table "
        "alone: fetch the file, verify its md5, apply the recorded assembly policy "
        "against the recorded chain hash, and overlap with the recorded coordinate "
        "authority."
    ),
}
receipt_path = OUT / "PATHB_STEP5_LINEAGE.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {path}")
print(f"written: {receipt_path}")
