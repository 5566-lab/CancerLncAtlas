"""Path B step 1: acquire and pin the hg19 -> hg38 chain file.

The plan forbids *silent* liftOver.  An explicit one therefore starts by pinning
the exact coordinate mapping as a first-class, hash-bound artifact: nothing is
lifted until the chain file's own SHA256 is recorded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
OUT = WORK / "inputs" / "liftover"
OUT.mkdir(parents=True, exist_ok=True)

#: UCSC's canonical hg19 -> hg38 over.chain, as named by UCSC itself.
CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
CHAIN_NAME = "hg19ToHg38.over.chain.gz"
PROXY = "socks5h://127.0.0.1:1080"

chain_path = OUT / CHAIN_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("=== fetching the pinned chain file ===", flush=True)
if chain_path.exists() and chain_path.stat().st_size > 1_000_000:
    print(f"    already present ({chain_path.stat().st_size:,} bytes)")
else:
    result = subprocess.run(
        ["curl", "-sSL", "--max-time", "600", "--retry", "3",
         "--proxy", PROXY, "-o", str(chain_path), CHAIN_URL],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    curl failed: {result.stderr[:200]}")
        raise SystemExit(1)
    print(f"    downloaded {chain_path.stat().st_size:,} bytes")

chain_sha = sha256(chain_path)
print(f"    sha256 = {chain_sha}")

# A chain file is gzip-compressed text.  Verify it actually decompresses and that
# it contains hg19 -> hg38 records rather than trusting the filename.
print("\n=== verifying the payload is a usable chain file ===", flush=True)
head = subprocess.run(
    ["bash", "-c", f"zcat {chain_path} | head -40"],
    capture_output=True, text=True,
).stdout
lines = [line for line in head.splitlines() if line.strip()]
chain_header = next((line for line in lines if line.startswith("chain ")), "")
score_lines = [line for line in head.splitlines() if line.startswith("chain ")]
print(f"    first chain header: {chain_header[:90]}")
if not chain_header:
    raise SystemExit("FAIL-CLOSED: the downloaded file has no chain header")

parts = chain_header.split()
# UCSC chain header:
#   chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
# For ``hg19ToHg38.over.chain`` the TARGET is the source genome and the QUERY is
# the destination.  Both assemblies name their chromosomes identically ("chr1"),
# so verifying by NAME is ambiguous.  Chromosome SIZES are decisive, so the gate
# compares them against a declared table.
t_name, t_size = parts[2], int(parts[3])
q_name, q_size = parts[7], int(parts[8])
print(f"    target = {t_name}  size={t_size:,}")
print(f"    query  = {q_name}  size={q_size:,}")

#: chr1 sizes, GRCh37(hg19) vs GRCh38(hg38).  These differ enough to be decisive.
HG19_CHR1 = 249_250_621
HG38_CHR1 = 248_956_422

if (t_size, q_size) != (HG19_CHR1, HG38_CHR1):
    raise SystemExit(
        "FAIL-CLOSED: chain does not map hg19 -> hg38. "
        f"expected target size {HG19_CHR1:,} (hg19 chr1) and query size "
        f"{HG38_CHR1:,} (hg38 chr1); got target {t_size:,} and query {q_size:,}"
    )
print("    direction verified by chromosome size: hg19 -> hg38")

# Count chains and total aligned bases: a truncated download would show up here.
stats = subprocess.run(
    ["bash", "-c",
     f"zcat {chain_path} | awk '/^chain/ {{n++}} END {{print n}}'"],
    capture_output=True, text=True,
).stdout.strip()
print(f"    chain records: {stats}")

receipt = {
    "resource": "UCSC hg19 -> hg38 liftOver chain",
    "url": CHAIN_URL,
    "file": str(chain_path),
    "bytes": chain_path.stat().st_size,
    "sha256": chain_sha,
    "chain_header_sample": chain_header,
    "direction_verified_by": "chromosome size (hg19 chr1 249250621 -> hg38 chr1 248956422)",
    "target_assembly": "hg19 (GRCh37)",
    "query_assembly": "hg38 (GRCh38)",
    "chain_records": int(stats) if stats.isdigit() else None,
    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    "direction_verified": True,
    "silent_liftover_forbidden": True,
    "note": (
        "This chain file is the pinned coordinate authority for the explicit "
        "hg19 -> hg38 liftOver. Every lifted peak must record its mapping status "
        "against this file's SHA256, and unmapped peaks are quarantined rather "
        "than dropped."
    ),
}
receipt_path = OUT / "CHAIN_RECEIPT.json"
receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {receipt_path}")
