"""Step A: bring `preferred_default` into the manifest and decide the reproducible subset.

The approved admission rule starts with "ENCODE released/reproducible peak".  The
first manifest never asked for `preferred_default`, so all 1,431 files were treated
alike.  This step fetches the field, tests what it actually means against the
replicate structure, and records the coverage cost of restricting to it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
RAW = ENC / "raw"
MAN = WORK / "manifests"
PROXY = "socks5h://127.0.0.1:1080"
BASE = "https://www.encodeproject.org"

FIELDS = ["accession", "assembly", "preferred_default", "output_type",
          "file_format_type", "biological_replicates", "technical_replicates",
          "biosample_ontology.term_name", "target.label", "dataset", "status",
          "file_size", "md5sum"]

url = (f"{BASE}/search/?type=File&file_format=bed&file_format_type=narrowPeak"
       f"&assay_title=eCLIP&status=released&format=json&limit=all"
       "&field=" + "&field=".join(FIELDS))
target = RAW / "ENCODE_SEARCH_ECLIP_PREFERRED_DEFAULT.json"
result = subprocess.run(
    ["curl", "-sSL", "--max-time", "600", "--retry", "3", "--proxy", PROXY,
     "-o", str(target), "-w", "%{http_code}", url],
    capture_output=True, text=True)
if result.returncode != 0 or result.stdout.strip() != "200":
    raise SystemExit(f"FAIL-CLOSED: http={result.stdout.strip()} {result.stderr[:160]}")
payload = json.loads(target.read_text(encoding="utf-8"))
print(f"=== fetched {payload.get('total')} eCLIP narrowPeak records ===")

rows = []
for record in payload["@graph"]:
    target_obj = record.get("target") or {}
    biosample = record.get("biosample_ontology") or {}
    dataset = str(record.get("dataset") or "")
    rows.append({
        "file_accession": record.get("accession") or "",
        "assembly": record.get("assembly") or "",
        "preferred_default": record.get("preferred_default"),
        "biosample": biosample.get("term_name") or "",
        "rbp": target_obj.get("label") or "",
        "experiment_accession": dataset.rstrip("/").rsplit("/", 1)[-1],
        "n_biological_replicates": len(record.get("biological_replicates") or []),
        "biological_replicates": ",".join(str(x) for x in (record.get("biological_replicates") or [])),
        "technical_replicates": ",".join(str(x) for x in (record.get("technical_replicates") or [])),
        "file_size": record.get("file_size") or 0,
        "md5sum": record.get("md5sum") or "",
    })
pd_frame = pd.DataFrame(rows)
print(f"  preferred_default=True : {int(pd_frame.preferred_default.eq(True).sum()):,}")
print(f"  preferred_default unset: {int(pd_frame.preferred_default.isna().sum()):,}")

# --- what does the flag actually mean? ----------------------------------------
print("\n=== preferred_default x number of biological replicates ===")
ct = pd.crosstab(pd_frame.preferred_default.fillna("unset"),
                 pd_frame.n_biological_replicates)
print(ct.to_string())
consistent = (
    pd_frame.loc[pd_frame.preferred_default.eq(True), "n_biological_replicates"].min() >= 2
    and pd_frame.loc[pd_frame.preferred_default.isna(), "n_biological_replicates"].max() <= 1
)
print(f"\n  flag is exactly '>=2 biological replicates' : {consistent}")

# --- what would restricting to it cost? ---------------------------------------
pref = pd_frame.loc[pd_frame.preferred_default.eq(True)]
print(f"\n=== coverage of the reproducible subset ({len(pref)} files) ===")
print(f"{'assembly':10} {'biosample':14} {'files':>7} {'MiB':>9} {'RBPs':>6}")
for (assembly, biosample), block in pref.groupby(["assembly", "biosample"]):
    print(f"{assembly:10} {biosample:14} {len(block):>7} "
          f"{block.file_size.sum() / 1048576:>9.1f} {block.rbp.nunique():>6}")
print(f"\n  all files      : {len(pd_frame):,} files, {pd_frame.file_size.sum() / 1048576:.1f} MiB, "
      f"{pd_frame.rbp.nunique()} RBPs")
print(f"  reproducible   : {len(pref):,} files, {pref.file_size.sum() / 1048576:.1f} MiB, "
      f"{pref.rbp.nunique()} RBPs")
print(f"  coverage cost  : {len(pref) / len(pd_frame):.1%} of files, "
      f"{pref.rbp.nunique() / pd_frame.rbp.nunique():.1%} of RBPs")

# --- do the already-downloaded files cover the reproducible subset? -----------
existing = {p.name.replace(".bed.gz", "") for p in (ENC / "files").glob("*.bed.gz")}
lifted = {p.name.replace(".GRCh38.narrowPeak.gz", "") for p in (ENC / "lifted").glob("*.GRCh38.narrowPeak.gz")}
have = existing | lifted
print(f"\n=== do we already have the bytes? ===")
print(f"  downloaded files                     : {len(have):,}")
print(f"  reproducible files already downloaded: "
      f"{int(pref.file_accession.isin(have).sum()):,} / {len(pref):,}")
missing = sorted(set(pref.file_accession) - have)
print(f"  reproducible files still missing     : {len(missing):,}")

# --- write the augmented manifest ---------------------------------------------
manifest_path = MAN / "ENCODE_FILE_MANIFEST.tsv"
manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
lookup = pd_frame.set_index("file_accession")
manifest["preferred_default"] = manifest.file_accession.map(
    lambda a: lookup.preferred_default.get(a))
manifest["n_biological_replicates"] = manifest.file_accession.map(
    lambda a: lookup.n_biological_replicates.get(a))
manifest["reproducible_peak"] = manifest.preferred_default.map(
    lambda v: "True" if v is True else "False")
manifest.to_csv(manifest_path, sep="\t", index=False)
print(f"\nupdated manifest: {manifest_path}")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "step": "Step A - reproducible peak designation",
    "source_query": url,
    "raw_file": str(target),
    "raw_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "api_total": payload.get("total"),
    "preferred_default_true": int(pd_frame.preferred_default.eq(True).sum()),
    "preferred_default_unset": int(pd_frame.preferred_default.isna().sum()),
    "flag_means_two_or_more_biological_replicates": bool(consistent),
    "crosstab": ct.to_dict(),
    "coverage": {
        "all_files": int(len(pd_frame)),
        "reproducible_files": int(len(pref)),
        "file_fraction": len(pref) / len(pd_frame),
        "all_rbps": int(pd_frame.rbp.nunique()),
        "reproducible_rbps": int(pref.rbp.nunique()),
        "rbp_fraction": pref.rbp.nunique() / pd_frame.rbp.nunique(),
        "all_mib": float(pd_frame.file_size.sum() / 1048576),
        "reproducible_mib": float(pref.file_size.sum() / 1048576),
        "already_downloaded": int(pref.file_accession.isin(have).sum()),
        "still_missing": len(missing),
    },
    "decision": (
        "Restrict the ENCODE eCLIP layer to preferred_default=True, which is exactly "
        "the >=2-biological-replicate set ENCODE designates as the default peak call. "
        "The non-preferred files are single-replicate (pseudoreplicated) calls and the "
        "approved rule asks for reproducible peaks."
    ),
    "manifest_updated": str(manifest_path),
}
path = MAN / "ENCODE_REPRODUCIBLE_PEAK_AUDIT.json"
path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
print(f"written: {path}")
