"""Phase 3D: gate the ENCODE gene-body overlap layer off the main graph.

The overlap ran, the edges exist, and the enrichment study then failed to justify
them.  This script records that outcome as a first-class decision instead of
leaving an ambiguous artifact on disk:

* the edge sets are kept, with their exact counts;
* they are **not** merged into the typed binding layer;
* the switch that would admit them defaults to off;
* the criterion that would have to be met first is stated.

The temptation this refuses is real: merging 428,949 edges into the eCLIP relation
would raise the typed binding layer from 98,873 to roughly 527,000 edges and look
like a triumph.  The peak-level measurement says those edges are not enriched over
a uniform null, so merging them would present positional coincidence as binding.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
ENC_DIR = WORK / "outputs" / "phase3_encode_eclip"
MAN = WORK / "manifests"
TYPED_DIR = WORK / "outputs" / "phase3_typed_binding"

GLOBAL = ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet"
CONTEXT = ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_CONTEXT.parquet"
SENSITIVITY = ENC_DIR / "PHASE3C_THRESHOLD_SENSITIVITY.json"
LEGACY = TYPED_DIR / "lnc_protein_binding_typed.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


sensitivity = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
global_edges = pd.read_parquet(GLOBAL)
context_edges = pd.read_parquet(CONTEXT)
legacy = pd.read_parquet(LEGACY)
legacy_eclip = legacy.loc[legacy.graph_assay_class.astype(str).eq("eclip")]

groups = sensitivity["measured_enrichment"]["groups"]
enrichment = {g["assembly"]: g["enrichment_over_null"] for g in groups}
peak_level_depleted = all(v is not None and v < 1.0 for v in enrichment.values())

print("=== the decision ===")
print(f"  ENCODE global edges built        : {len(global_edges):,}")
print(f"  ENCODE context edges built       : {len(context_edges):,}")
print(f"  peak-level enrichment vs uniform : "
      + ", ".join(f"{k}={v:.2f}x" for k, v in enrichment.items()))
print(f"  per-pair Poisson says            : "
      f"{sensitivity['conflicting_nulls']['per_pair']['pairs_at_two_fold_or_more_fraction']:.1%} "
      "of pairs at >=2x expectation")
print(f"  peak-level says                  : "
      f"{'DEPLETION' if peak_level_depleted else 'enrichment'}")
if not peak_level_depleted:
    raise SystemExit(
        "FAIL-CLOSED: the peak-level enrichment is no longer a depletion; re-read "
        "the study before keeping this layer gated off"
    )

projected = len(legacy) + len(global_edges)
print(f"\n  legacy typed binding rows        : {len(legacy):,}")
print(f"  would become                     : {projected:,} (+{len(global_edges):,})")
print(f"  legacy eCLIP rows                : {len(legacy_eclip):,}")
print("  -> NOT merged")

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "Phase 3D - ENCODE gene-body overlap layer gate",
    "decision": "GATED OFF - built but not admitted to the main graph",
    "switch": {
        "name": "include_encode_eclip_overlap",
        "default": False,
        "where": "cc_hhgt.v32.rbp_evidence_config.RbpEvidenceConfig",
    },
    "edges_built": {
        "global": int(len(global_edges)),
        "context": int(len(context_edges)),
        "lncrnas": int(global_edges.lncrna_id.nunique()),
        "proteins": int(global_edges.protein_id.nunique()),
    },
    "why": {
        "peak_level_measurement": enrichment,
        "peak_level_verdict": "depletion, not enrichment",
        "null_used": sensitivity["positional_null"],
        "conflicting_per_pair_view": sensitivity["conflicting_nulls"]["per_pair"],
        "explanation": (
            "Two nulls disagree: peaks land in lncRNA gene bodies less often than "
            "uniform placement predicts (0.71-0.74x), while a per-pair Poisson test "
            "calls 89.6% of pairs enriched because it is conditioned on the pair "
            "existing and assumes peaks are uniform. Neither null preserves the "
            "clustering of eCLIP peaks, and until one does, gene-body overlap "
            "cannot be told apart from position."
        ),
    },
    "counterfactual_if_merged": {
        "legacy_typed_binding_rows": int(len(legacy)),
        "would_become": projected,
        "legacy_eclip_rows": int(len(legacy_eclip)),
        "note": (
            "Merging would multiply the eCLIP evidence roughly 14-fold and look "
            "like a large gain. That is precisely why the measurement had to be "
            "made before merging rather than after."
        ),
    },
    "criterion_for_admission": [
        "a null model that preserves the genome-wide clustering of eCLIP peaks "
        "(for example a per-RBP peak-density background computed from the same "
        "files, or a dinucleotide-matched shuffle)",
        "an enrichment measured against that null that exceeds 1 with a stated "
        "confidence interval",
        "a peak-level criterion rather than a gene-body one, ideally requiring the "
        "peak to fall in an annotated exon of the lncRNA",
    ],
    "artifacts_kept": {
        "global": {"path": str(GLOBAL), "sha256": sha256_file(GLOBAL)},
        "context": {"path": str(CONTEXT), "sha256": sha256_file(CONTEXT)},
        "sensitivity": {"path": str(SENSITIVITY), "sha256": sha256_file(SENSITIVITY)},
        "evidence_long": {
            "path": str(ENC_DIR / "ENCODE_ECLIP_EVIDENCE_LONG.parquet"),
            "sha256": sha256_file(ENC_DIR / "ENCODE_ECLIP_EVIDENCE_LONG.parquet"),
        },
    },
    "merged_into_graph": False,
    "hepg2_broadcast_to_33_cancers": False,
    "k562_forced_to_LAML": False,
}
path = ENC_DIR / "PHASE3D_ENCODE_LAYER_GATE.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
(MAN / "RBP_ENCODE_LAYER_GATE.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {path}")
print(f"written: {MAN / 'RBP_ENCODE_LAYER_GATE.json'}")
