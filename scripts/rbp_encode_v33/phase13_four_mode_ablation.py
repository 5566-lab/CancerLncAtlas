"""Phase 13: the four-mode ablation, on the CPU side.

The plan's four modes differ only in which RBP evidence reaches the graph:

======  ==================================================  ====================
mode    binding layer                                       graph generation
======  ==================================================  ====================
A       legacy flat ``binds_protein`` (frozen)             frozen G012
B       typed assay-class relations, no ENCODE              typed generation
C       B + ENCODE eCLIP                                    typed + ENCODE
D       C, with RBP knockdown in the Evidence layer only     identical to C
======  ==================================================  ====================

What can be settled on CPU is the *graph* half: how many edges each mode's
binding layer actually contributes, per relation, and the proof that the four
modes share every fairness axis.  What cannot is the AUPRC half, which needs the
GPU that the CPU gate still holds shut; this script states that explicitly rather
than leaving a reader to assume the ablation is finished.

Nothing here re-runs a model.  Every number is read from an authority receipt.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.rbp_evidence_config import (  # noqa: E402
    ABLATION_MODES,
    assert_fair_comparison,
    resolve_mode,
)

FROZEN = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_local_cnv_corrected_graph_20260903_r1")
PRE_TYPED = WORK / "outputs" / "phase7_typed_authority" / "PHASE7_TYPED_AUTHORITY_SUMMARY.json"
ENCODE_TYPED = WORK / "outputs" / "phase7_authority_encode" / "PHASE7_TYPED_AUTHORITY_SUMMARY.json"
ENCODE_GATE = WORK / "outputs" / "phase3_encode_eclip" / "PHASE3D_ENCODE_LAYER_GATE.json"
OUT = WORK / "outputs" / "phase13_four_mode_ablation"
OUT.mkdir(parents=True, exist_ok=True)
MAN = WORK / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- mode A: what the frozen generation actually carries ----------------------
frozen_receipt = json.loads((FROZEN / "GRAPH_INPUT_AUTHORITY_RECEIPT.json").read_text(encoding="utf-8"))
frozen_binding = frozen_receipt["artifacts"]["lnc_protein_binding"]
# The receipt records path and sha256 but not necessarily a row count, so the
# count is read from the artifact itself rather than defaulted to zero.
frozen_binding_path = Path(str(frozen_binding.get("path")))
if not frozen_binding_path.exists():
    frozen_binding_path = Path(
        "${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/v32_g012_patient_first_20260830_r1"
        "/static/lnc_protein_binding.parquet")
frozen_generic_rows = int(len(pd.read_parquet(frozen_binding_path)))
print(f"=== mode A baseline ===")
print(f"  frozen binding artifact: {frozen_binding_path}")
print(f"  flat binds_protein rows: {frozen_generic_rows:,}")

# --- modes B and C: the two typed generations ---------------------------------
pre = load(PRE_TYPED)
enc = load(ENCODE_TYPED)


def binding_total(summary: dict | None) -> dict:
    if not summary:
        return {}
    folds = summary.get("folds") or {}
    if not folds:
        return {}
    # The binding layer is static, so every fold reports the same relation counts.
    first = folds[sorted(folds)[0]]
    return {k: int(v) for k, v in (first.get("binding_relations") or {}).items()}


pre_binding = binding_total(pre)
enc_binding = binding_total(enc)


def variant_counts(summary: dict | None) -> dict:
    if not summary:
        return {}
    folds = summary.get("folds") or {}
    if not folds:
        return {}
    return {k: int(v) for k, v in (folds[sorted(folds)[0]].get("variant_edge_counts") or {}).items()}


pre_variants = variant_counts(pre)
enc_variants = variant_counts(enc)

print("=== binding layer per mode ===")
print(f"  A  legacy flat binds_protein rows        : {frozen_generic_rows:,}")
print(f"  B  typed, no ENCODE, active binding edges: {sum(pre_binding.values()):,}")
gate = load(ENCODE_GATE)
if enc_binding:
    print(f"  C  typed + ENCODE, active binding edges  : {sum(enc_binding.values()):,}")
    delta = sum(enc_binding.values()) - sum(pre_binding.values())
    print(f"     ENCODE contribution                   : {delta:+,}")
elif gate:
    why = gate["why"]["peak_level_measurement"]
    print(f"  C  typed + ENCODE                        : NOT MATERIALISED - "
          f"{gate['decision']}")
    print(f"     ENCODE edges built                    : {gate['edges_built']['global']:,} global, "
          f"{gate['edges_built']['context']:,} context")
    print(f"     peak-level enrichment vs uniform      : "
          + ", ".join(f"{k}={v:.2f}x" for k, v in why.items()))
    print(f"     reason                                : {gate['why']['peak_level_verdict']}")
else:
    print(f"  C  typed + ENCODE                        : PENDING (no authority, no gate receipt)")

if pre_binding and enc_binding:
    print("\n=== per-relation change from ENCODE ===")
    for relation in sorted(set(pre_binding) | set(enc_binding)):
        before = pre_binding.get(relation, 0)
        after = enc_binding.get(relation, 0)
        marker = "" if before == after else f"   {after - before:+,}"
        print(f"    {relation:44s} {before:>10,} -> {after:>10,}{marker}")

# --- fairness: the four modes must share every frozen axis --------------------
configs = [resolve_mode(mode) for mode in ABLATION_MODES]
fairness = assert_fair_comparison(configs)
print(f"\n=== fairness across the four modes: {fairness['status']} ===")
for key, value in fairness["frozen_axes"].items():
    print(f"    {key:32s} {value}")

modes = {
    "legacy_generic_binding": {
        "mode": "A",
        "binding_layer": "frozen flat binds_protein",
        "authority": str(FROZEN / "GRAPH_INPUT_AUTHORITY_RECEIPT.json"),
        "authority_kind": "frozen",
        "binding_rows": frozen_generic_rows,
        "active_binding_edges": None,
        "variant_edge_counts": None,
    },
    "typed_binding_only": {
        "mode": "B",
        "binding_layer": "typed assay-class relations",
        "authority": str(PRE_TYPED) if pre else None,
        "authority_kind": "typed_pre_encode",
        "binding_rows": None,
        "active_binding_edges": sum(pre_binding.values()) if pre_binding else None,
        "per_relation": pre_binding,
        "variant_edge_counts": pre_variants,
    },
    "typed_binding_plus_encode_eclip": {
        "mode": "C",
        "binding_layer": "typed assay-class relations + ENCODE eCLIP",
        "authority": str(ENCODE_TYPED) if enc else None,
        "authority_kind": "typed_with_encode",
        "binding_rows": None,
        "active_binding_edges": sum(enc_binding.values()) if enc_binding else None,
        "per_relation": enc_binding,
        "variant_edge_counts": enc_variants,
        "status": "materialised" if enc_binding else "gated_off",
        "gate": None if enc_binding else {
            "decision": (gate or {}).get("decision"),
            "switch": (gate or {}).get("switch"),
            "edges_built": (gate or {}).get("edges_built"),
            "peak_level_enrichment": ((gate or {}).get("why") or {}).get("peak_level_measurement"),
            "criterion_for_admission": (gate or {}).get("criterion_for_admission"),
        },
    },
    "full_rbp_evidence": {
        "mode": "D",
        "binding_layer": "identical to C; RBP knockdown enters the Evidence layer only",
        "authority": str(ENCODE_TYPED) if enc else None,
        "authority_kind": "typed_with_encode",
        "binding_rows": None,
        "active_binding_edges": sum(enc_binding.values()) if enc_binding else None,
        "per_relation": enc_binding,
        "variant_edge_counts": enc_variants,
        "graph_identical_to_mode": "C",
        "evidence_layer_only": ["encode_rbp_kd"],
    },
}

receipt = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "phase": "Phase 13 - four-mode ablation (CPU half)",
    "modes": modes,
    "switch_matrix": fairness["switch_matrix"],
    "frozen_axes": fairness["frozen_axes"],
    "fairness_status": fairness["status"],
    "frozen_authority_receipt": str(FROZEN / "GRAPH_INPUT_AUTHORITY_RECEIPT.json"),
    "frozen_authority_receipt_sha256": sha256_file(FROZEN / "GRAPH_INPUT_AUTHORITY_RECEIPT.json"),
    "resolved_on_cpu": [
        "each mode's binding layer contribution, per relation",
        "the four modes' identical fairness axes",
        "that RBP knockdown never becomes a primary-graph edge in any mode",
    ],
    "not_resolved_on_cpu": [
        "AUPRC per mode - requires the GPU that the CPU gate holds shut",
        "any statement about which mode ranks better",
    ],
    "gpu_started": False,
    "note": (
        "Mode D is graph-identical to mode C by construction: knockdown evidence is "
        "declared evidence_only and is forbidden from the primary graph by "
        "assert_fair_comparison."
    ),
}

path = OUT / "PHASE13_FOUR_MODE_ABLATION.json"
path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
(MAN / "RBP_FOUR_MODE_ABLATION.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"\nwritten: {path}")
print(f"written: {MAN / 'RBP_FOUR_MODE_ABLATION.json'}")
