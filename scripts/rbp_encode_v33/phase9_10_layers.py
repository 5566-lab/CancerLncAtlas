"""Phases 9-10: declare the knockdown and RBNS evidence layers, and gate them off.

Two ENCODE resources in the plan are *not* binding measurements and must never be
promoted to lncRNA functional truth:

``RBP knockdown RNA-seq`` (ENCSR369TWP / ENCSR795JHH / ENCSR413YAF / ENCSR870OLK)
    A knockdown says "gene X changed abundance when RBP Y was depleted".  It is
    ``expression_or_abundance`` evidence about genes, not a physical lncRNA-RBP
    interaction, and it is not a functional truth about an lncRNA.

``RBNS`` (ENCSR876DCD)
    An in-vitro k-mer enrichment experiment.  It yields a sequence preference,
    which is ``predicted`` evidence by construction.

Both therefore become *declared layers*: fully inventoried at file level, with an
explicit policy, an explicit switch defaulted **off**, and a stated prerequisite
before any future materialisation.  Neither contributes an edge to the main graph
under the default configuration.

The script also downloads the small RBNS products (enrichment tsv and peak bed)
when asked, so the layer is not merely described but present and hash-verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC = WORK / "inputs" / "encode"
MAN = WORK / "manifests"
LAYERS = ENC / "layers"
LAYERS.mkdir(parents=True, exist_ok=True)
OUT = WORK / "outputs" / "phase9_10_layers"
OUT.mkdir(parents=True, exist_ok=True)

PROXY = os.environ.get("ENCODE_PROXY", "socks5h://127.0.0.1:1080")
BASE = "https://www.encodeproject.org"
FETCH_RBNS = os.environ.get("RBP_FETCH_RBNS", "1") == "1"
WORKERS = int(os.environ.get("ENCODE_DOWNLOAD_WORKERS", "4"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = pd.read_csv(MAN / "ENCODE_FILE_MANIFEST.tsv", sep="\t", dtype=str).fillna("")
manifest["file_size"] = pd.to_numeric(manifest.file_size, errors="coerce").fillna(0).astype("int64")

KD_RESOURCES = ["kd_rnaseq_hepg2", "kd_rnaseq_k562", "kd_processed_hepg2_k562", "kd_batch_corrected"]
kd = manifest.loc[manifest.resource.isin(KD_RESOURCES)]
rbns = manifest.loc[manifest.resource.eq("rbns")]


def inventory(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    if frame.empty:
        return []
    grouped = frame.groupby(keys, dropna=False).agg(
        files=("file_accession", "nunique"), bytes=("file_size", "sum"))
    return [
        {**dict(zip(keys, index if isinstance(index, tuple) else (index,))),
         "files": int(row.files), "bytes": int(row.bytes), "mib": round(row.bytes / 1048576, 1)}
        for index, row in grouped.iterrows()
    ]


kd_inventory = inventory(kd, ["resource", "output_type", "file_format", "biosample"])
rbns_inventory = inventory(rbns, ["output_type", "file_format", "assembly"])

print("=== knockdown layer inventory ===")
print(f"  files {kd.file_accession.nunique():,}   "
      f"size {kd.file_size.sum() / 1024 ** 3:.2f} GiB")
for row in kd_inventory:
    print(f"    {row['resource']:28} {str(row['output_type'])[:38]:40} {row['file_format']:6} "
          f"{row['files']:>5} files  {row['mib']:>10.1f} MiB")

print("\n=== RBNS layer inventory ===")
print(f"  files {rbns.file_accession.nunique():,}   "
      f"size {rbns.file_size.sum() / 1024 ** 3:.2f} GiB")
for row in rbns_inventory:
    print(f"    {str(row['output_type']):12} {row['file_format']:6} {str(row['assembly']):8} "
          f"{row['files']:>5} files  {row['mib']:>10.1f} MiB")

# --- fetch the small RBNS products -------------------------------------------------
rbns_small = rbns.loc[rbns.file_format.isin(["tsv", "bed"])].copy()
print(f"\n=== RBNS small products: {len(rbns_small)} files, "
      f"{rbns_small.file_size.sum() / 1048576:.1f} MiB ===")

fetched: list[dict] = []
if FETCH_RBNS and len(rbns_small):

    def fetch(row: pd.Series) -> dict:
        target = LAYERS / f"{row.file_accession}.{row.file_format}.gz" if False else LAYERS / f"{row.file_accession}.{row.file_format}"
        expected_md5, expected_size = row.md5sum, int(row.file_size)
        url = BASE + row.href
        if target.exists() and target.stat().st_size == expected_size and md5_file(target) == expected_md5:
            return {"file_accession": row.file_accession, "status": "CACHED_VERIFIED",
                    "bytes": expected_size, "md5": expected_md5,
                    "output_type": row.output_type, "file_format": row.file_format}
        result = subprocess.run(
            ["curl", "-sSL", "--max-time", "600", "--retry", "3", "--retry-delay", "2",
             "--proxy", PROXY, "-o", str(target), "-w", "%{http_code}", url],
            capture_output=True, text=True)
        if result.returncode != 0 or result.stdout.strip() != "200":
            if target.exists():
                target.unlink()
            return {"file_accession": row.file_accession, "status": "FAILED_HTTP",
                    "http": result.stdout.strip(), "output_type": row.output_type,
                    "file_format": row.file_format}
        size = target.stat().st_size
        observed = md5_file(target)
        if size != expected_size or observed != expected_md5:
            target.unlink()
            return {"file_accession": row.file_accession, "status": "FAILED_VERIFY",
                    "bytes": size, "expected_bytes": expected_size,
                    "md5": observed, "expected_md5": expected_md5,
                    "output_type": row.output_type, "file_format": row.file_format}
        return {"file_accession": row.file_accession, "status": "DOWNLOADED", "bytes": size,
                "md5": observed, "output_type": row.output_type, "file_format": row.file_format}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, row) for _, row in rbns_small.iterrows()]
        for done, future in enumerate(as_completed(futures), 1):
            fetched.append(future.result())
            if done % 50 == 0 or done == len(futures):
                ok = sum(1 for f in fetched if f["status"] in ("DOWNLOADED", "CACHED_VERIFIED"))
                print(f"    {done:>4}/{len(futures)}  verified={ok}", flush=True)

    counts: dict[str, int] = {}
    for f in fetched:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    print(f"  status: {counts}")
    failures = [f for f in fetched if f["status"].startswith("FAILED")]
    if failures:
        raise SystemExit(f"FAIL-CLOSED: {len(failures)} RBNS files failed verification")
else:
    print("  (skipped: RBP_FETCH_RBNS is not 1)")

declaration = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "phases": "Phase 9 (RBP knockdown) and Phase 10 (RBNS)",
    "manifest": str(MAN / "ENCODE_FILE_MANIFEST.tsv"),
    "manifest_sha256": sha256_file(MAN / "ENCODE_FILE_MANIFEST.tsv"),
    "layers": {
        "rbp_knockdown": {
            "graph_admitted": False,
            "default_switch": "include_knockdown_evidence = False",
            "evidence_family": "expression_or_abundance",
            "graph_assay_class_if_ever_admitted": "experimental_unspecified",
            "why_not_admitted": (
                "A knockdown measures abundance change after depletion. Treating it as "
                "lncRNA functional truth would present a perturbation response as a "
                "physical interaction, which the plan forbids."
            ),
            "resources": KD_RESOURCES,
            "files": int(kd.file_accession.nunique()),
            "bytes": int(kd.file_size.sum()),
            "inventory": kd_inventory,
            "download_status": "NOT_DOWNLOADED_BY_DESIGN",
            "download_rationale": (
                f"The smallest useful product (differential expression quantifications) "
                f"is still {kd.loc[kd.output_type.eq('differential expression quantifications'), 'file_size'].sum() / 1024 ** 3:.2f} GiB. "
                "Downloading it would buy no graph edge under the default policy. The "
                "file-level inventory above is complete, so the layer can be fetched "
                "later with one command without re-querying ENCODE."
            ),
            "prerequisite_before_admission": [
                "an explicit decision that knockdown evidence may enter the graph",
                "a relation that encodes perturbation rather than physical binding",
            ],
        },
        "rbns": {
            "graph_admitted": False,
            "default_switch": "include_predicted_evidence = False",
            "evidence_family": "computational_prediction",
            "graph_assay_class_if_ever_admitted": "predicted",
            "why_not_admitted": (
                "RBNS reports an in-vitro k-mer preference. Any lncRNA-level statement "
                "built from it is a prediction, and predictions are excluded from the "
                "main graph by the conservative default."
            ),
            "files": int(rbns.file_accession.nunique()),
            "bytes": int(rbns.file_size.sum()),
            "inventory": rbns_inventory,
            "small_products": {
                "files": len(rbns_small),
                "bytes": int(rbns_small.file_size.sum()),
                "downloaded": len(fetched),
                "directory": str(LAYERS),
                "status_counts": {s: sum(1 for f in fetched if f["status"] == s)
                                  for s in sorted({f["status"] for f in fetched})} if fetched else {},
                "detail": fetched,
            },
            "prerequisite_before_admission": [
                "lncRNA transcript sequences, which are not part of the project's frozen inputs",
                "a k-mer scoring step whose output is labelled predicted, never measured",
            ],
        },
    },
    "node_type_created": False,
    "knockdown_used_as_ground_truth": False,
    "predicted_evidence_in_main_graph": False,
}
receipt_path = OUT / "PHASE9_10_LAYER_DECLARATION.json"
receipt_path.write_text(json.dumps(declaration, indent=2), encoding="utf-8")
print(f"\nwritten: {receipt_path}")
print(f"  knockdown: {declaration['layers']['rbp_knockdown']['files']:,} files declared, none admitted to the graph")
print(f"  rbns     : {declaration['layers']['rbns']['files']:,} files declared, none admitted to the graph")
