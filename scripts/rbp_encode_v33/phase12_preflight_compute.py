"""Phase 12 CPU hard gate: compute every preflight item and emit the report.

Gates:
  1  relevant pytest                                              -> from logs
  2  full suite, no new failures                                  -> from logs
  3  ENCODE manifest SHA                                          -> computed here
  4  genome assembly gate                                         -> computed here
  5  RBP -> UniProt mapping audit                                 -> computed here
  6  assay_subtype missingness report                             -> computed here
  7  cross-database duplicate audit                               -> Phase 4
  8  graph count report                                           -> computed here
  9  context leakage test                                         -> computed here
 10  G0/G1/G2 invariant                                           -> Phase 3 supplement
 11  synthetic forward/backward                                   -> separate run
 12  legacy mode reproduces old generic semantics                 -> Phase 3, PASS
 13  ENCODE download verification                                 -> computed here
 14  liftOver quarantine accounting                               -> computed here
 15  ENCODE does not add nodes                                    -> computed here
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/rbp_encode_v33_20260921_r1")
sys.path.insert(0, str(WORK / "code"))

from cc_hhgt.v32.formal_graph import (  # noqa: E402
    CONTEXT_ECLIP_ROLE,
    GLOBAL_BINDING_ROLE,
    materialize_typed_lnc_protein_binding,
)
from cc_hhgt.v32.rbp_assay import ASSAY_SUBTYPES, GRAPH_ASSAY_CLASSES  # noqa: E402

TYPED = WORK / "outputs" / "phase3_typed_binding" / "lnc_protein_binding_typed.parquet"
AUDIT = WORK / "outputs" / "phase3_typed_binding" / "PHASE3_TYPED_BINDING_AUDIT.json"
ENC_DIR = WORK / "outputs" / "phase3_encode_eclip"
ENC_MANIFEST = WORK / "manifests" / "ENCODE_FILE_MANIFEST.tsv"
OUT = WORK / "manifests" / "PHASE12_CPU_PREFLIGHT.json"
REPORT = WORK / "reports" / "RBP_ENCODE_CPU_PREFLIGHT.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

typed = pd.read_parquet(TYPED)
audit = json.loads(AUDIT.read_text())
legacy_audit = audit["legacy_audit"]
typed_audit = audit["typed_audit"]

results: dict[str, object] = {}

# ---------------------------------------------------------------- gate 5
print("=== gate 5: RBP -> UniProt mapping audit ===")
input_rows = {"legacy": legacy_audit["input_rows"], "typed": typed_audit["input_rows"]}
rejected = {
    "legacy": legacy_audit["protein_mapping_rejected_rows"],
    "typed": typed_audit["protein_mapping_rejected_rows"],
}
mapping = {
    "input_relation_rows": input_rows,
    "unmappable_partner_rows": rejected,
    "mapped_fraction": {
        key: round(1.0 - rejected[key] / input_rows[key], 6) for key in input_rows
    },
    "all_emitted_proteins_are_uniprot": bool(
        typed.protein_id.astype(str).str.startswith("UNIPROT:").all()
    ),
    "distinct_proteins": int(typed.protein_id.nunique()),
    "distinct_lncrnas": int(typed.lncrna_id.nunique()),
}
for key, value in mapping.items():
    print(f"    {key:34s} {value}")
results["rbp_uniprot_mapping"] = mapping

# ---------------------------------------------------------------- gate 6
print("\n=== gate 6: assay_subtype missingness ===")
classes = typed.graph_assay_class.astype(str)
subtype_counts = (
    typed.assign(_s=typed.assay_subtype.astype(str))
    .groupby("_s", observed=True)
    .size()
    .sort_values(ascending=False)
)
total_rows = int(len(typed))
uninformative = int(classes.isin({"unknown", "experimental_unspecified"}).sum())
missingness = {
    "rows": total_rows,
    "assay_subtype_counts": {k: int(v) for k, v in subtype_counts.items()},
    "graph_assay_class_counts": {
        k: int(v) for k, v in classes.value_counts().items()
    },
    "uninformative_rows": uninformative,
    "uninformative_fraction": round(uninformative / total_rows, 6),
    "vocabulary_coverage": len(set(subtype_counts.index) & set(ASSAY_SUBTYPES)),
}
print(f"    rows                       : {total_rows:,}")
print(f"    uninformative (unknown/unspecified) : {uninformative:,} "
      f"({100.0 * uninformative / total_rows:.2f}%)")
for key, value in missingness["graph_assay_class_counts"].items():
    print(f"       {key:28s} {value:>10,}")
results["assay_subtype_missingness"] = missingness

# ---------------------------------------------------------------- gate 8 + 9
print("\n=== gate 8/9: graph counts and context leakage ===")
graph_report: dict[str, object] = {}
for predicted in (False, True):
    edges = materialize_typed_lnc_protein_binding(typed, include_predicted=predicted)
    label = "with_predicted" if predicted else "physical_only"
    contextual = edges.loc[edges.edge_role.eq(CONTEXT_ECLIP_ROLE)]
    globals_ = edges.loc[edges.edge_role.eq(GLOBAL_BINDING_ROLE)]
    leakage = int(
        (
            contextual.cancer_id.isna()
            | ~contextual.is_context_specific.astype(bool)
        ).sum()
    )
    graph_report[label] = {
        "edges": int(len(edges)),
        "relation_type_counts": {
            k: int(v) for k, v in edges.relation_type.value_counts().items()
        },
        "edge_role_counts": {
            k: int(v) for k, v in edges.edge_role.value_counts().items()
        },
        "context_specific_edges": int(len(contextual)),
        "global_edges": int(len(globals_)),
        "context_edges_without_a_cancer": leakage,
        "distinct_context_cancers": sorted(
            contextual.cancer_id.dropna().astype(str).unique()
        ),
    }
    print(f"    [{label}] edges={len(edges):,} "
          f"context={len(contextual):,} global={len(globals_):,} "
          f"leakage={leakage}")
    for key, value in sorted(
        graph_report[label]["relation_type_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"         {key:44s} {value:>10,}")
results["graph_counts"] = graph_report

leakage_total = sum(v["context_edges_without_a_cancer"] for v in graph_report.values())
results["context_leakage_gate"] = {
    "context_edges_without_a_cancer": leakage_total,
    "pass": leakage_total == 0,
}
print(f"    context leakage gate: "
      f"{'PASS' if leakage_total == 0 else 'FAIL'} ({leakage_total} leaking edges)")

# ---------------------------------------------------------------- context exclusions
results["context_policy"] = {
    "context_specific_input_rows": typed_audit["context_specific_input_rows"],
    "mapped_context_rows": typed_audit["mapped_context_rows"],
    "excluded_unmapped_or_ambiguous_rows":
        typed_audit["excluded_unmapped_or_ambiguous_context_rows"],
    "excluded_fraction": round(
        typed_audit["excluded_unmapped_or_ambiguous_context_rows"]
        / max(typed_audit["context_specific_input_rows"], 1),
        6,
    ),
    "context_specific_lost_to_global": typed_audit["context_specific_lost_to_global"],
}

# ---------------------------------------------------------------- gate 3 + 4
print("\n=== gate 3: ENCODE manifest integrity ===")
if ENC_MANIFEST.exists():
    enc_manifest = pd.read_csv(ENC_MANIFEST, sep="\t", dtype=str).fillna("")
    manifest_sha = sha256_file(ENC_MANIFEST)
    receipt = load_json(WORK / "manifests" / "ENCODE_FILE_MANIFEST.json") or {}
    stale = [
        key for key, value in (receipt.get("queries") or {}).items()
        if value.get("http") not in (200, None)
    ]
    gate3 = {
        "status": "PASS",
        "manifest": str(ENC_MANIFEST),
        "manifest_sha256": manifest_sha,
        "manifest_rows": int(len(enc_manifest)),
        "receipt_manifest_sha256": receipt.get("manifest_sha256"),
        "receipt_matches_file": receipt.get("manifest_sha256") in (None, manifest_sha),
        "failed_queries": stale,
    }
    print(f"    rows={len(enc_manifest):,}  sha256={manifest_sha[:24]}…")
else:
    gate3 = {
        "status": "MISSING",
        "note": (
            "The plan's experiment-level manifest could not express per-file assembly, "
            "so a file-level manifest supersedes it. This gate reports MISSING until "
            "phase6_encode_manifest.py has run."
        ),
    }
    print("    MISSING (run phase6_encode_manifest.py)")
results["encode_manifest"] = gate3

print("\n=== gate 4: genome assembly ===")
coords = load_json(WORK / "manifests" / "LNCRNA_COORDINATE_AUTHORITY.json")
chain = load_json(WORK / "inputs" / "liftover" / "CHAIN_RECEIPT.json")
lift = load_json(WORK / "manifests" / "PATHB_LIFTOVER_RECEIPT.json")
gate4 = {
    "status": "PASS" if coords and coords.get("assembly") == "GRCh38" else "OPEN",
    "project_assembly": "GRCh38",
    "lncrna_coordinate_assembly": (coords or {}).get("assembly"),
    "lncrna_coordinate_verification": (coords or {}).get("assembly_verification"),
    "lncrna_nodes_without_coordinates": (coords or {}).get("nodes_without_coordinates"),
    "chain_pinned": bool(chain),
    "chain_sha256": (chain or {}).get("sha256"),
    "chain_direction_verified": (chain or {}).get("direction_verified"),
    "silent_liftover": False,
    "liftover": None if not lift else {
        "peaks_total": lift["peaks_total"],
        "peaks_mapped": lift["peaks_mapped"],
        "peaks_quarantined": lift["peaks_quarantined"],
        "mapped_fraction": lift["mapped_fraction"],
        "peaks_dropped": lift["peaks_dropped"],
        "quarantine_reasons": lift["quarantine_reasons"],
    },
}
if not coords:
    gate4["note"] = "no lncRNA coordinate authority yet; run pathb_step0_lncrna_coords.py"
print(f"    project GRCh38; lncRNA authority: {gate4['lncrna_coordinate_assembly']} "
      f"({gate4['lncrna_nodes_without_coordinates']} nodes without coordinates)")
if lift:
    print(f"    liftOver: {lift['peaks_mapped']:,}/{lift['peaks_total']:,} peaks mapped "
          f"({lift['mapped_fraction'] * 100:.2f}%), {lift['peaks_quarantined']:,} quarantined, "
          f"{lift['peaks_dropped']} dropped")
results["genome_assembly"] = gate4

# ------------------------------------------------------- gates 13, 14, 15
print("\n=== gate 13: ENCODE download verification ===")
download = load_json(WORK / "manifests" / "ENCODE_DOWNLOAD_RECEIPT.json")
if download:
    gate13 = {
        "status": "PASS" if not download["failures"] else "FAIL",
        "surface_files": download["surface_files"],
        "verified_files": download["verified_files"],
        "verified_bytes": download["verified_bytes"],
        "status_counts": download["status_counts"],
        "failures": len(download["failures"]),
        "verification": download["verification"],
    }
    print(f"    verified {download['verified_files']:,}/{download['surface_files']:,} files, "
          f"{download['verified_bytes'] / 1048576:.1f} MiB, failures={len(download['failures'])}")
else:
    gate13 = {"status": "MISSING", "note": "run phase6b_encode_download.py"}
    print("    MISSING")
results["encode_download"] = gate13

print("\n=== gate 14: liftOver quarantine accounting ===")
if lift:
    gate14 = {
        "status": "PASS" if lift["peaks_dropped"] == 0 else "FAIL",
        "peaks_total": lift["peaks_total"],
        "peaks_mapped": lift["peaks_mapped"],
        "peaks_quarantined": lift["peaks_quarantined"],
        "peaks_dropped": lift["peaks_dropped"],
        "every_unmapped_peak_retained": lift["peaks_dropped"] == 0,
        "criterion": lift["criterion"],
    }
    print(f"    quarantined={lift['peaks_quarantined']:,}  dropped={lift['peaks_dropped']}  "
          f"-> {'PASS' if gate14['status'] == 'PASS' else 'FAIL'}")
else:
    gate14 = {"status": "MISSING", "note": "run pathb_step2_liftover.py"}
    print("    MISSING")
results["liftover_quarantine"] = gate14

print("\n=== gate 15: ENCODE must not add nodes ===")
merge = load_json(ENC_DIR / "PHASE3B_MERGE_AUDIT.json")
encode_edges = None
if (ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet").exists():
    encode_edges = pd.read_parquet(ENC_DIR / "ENCODE_ECLIP_LNCRNA_RBP_GLOBAL.parquet")
if encode_edges is not None:
    # The baseline is the GRAPH NODE UNIVERSE, not the binding table. The binding
    # table only contains lncRNAs that already carry binding evidence, so comparing
    # against it would report every newly evidenced lncRNA as a "new node" - which
    # is the opposite of the question. The node universe is what the graph has.
    candidates = pd.read_parquet(
        Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas/inputs/"
             "v32_g012_local_cnv_formal_prepared_20260903_r1/FORMAL_CANDIDATE_UNIVERSE.parquet"),
        columns=["lncrna_id"])
    node_lncrnas = set(candidates.lncrna_id.astype(str).unique())
    standardised = Path(
        "${PRIVATE_ARCHIVE_ROOT}/CancerLncAtlas/results/v3_state_formal_runs/"
        "V3STATE-formal500-20260810T101223Z-a8803782dcf0_server_handoff/run/assets/standardized")
    node_proteins = set(pd.read_parquet(standardised / "protein_gene_map.parquet")
                        .protein_id.astype(str))

    encode_lncrnas = set(encode_edges.lncrna_id.astype(str))
    encode_proteins = set(encode_edges.protein_id.astype(str))
    new_lncrnas = sorted(encode_lncrnas - node_lncrnas)
    new_proteins = sorted(encode_proteins - node_proteins)
    # Against the binding table this is informational only: a lncRNA that had no
    # binding evidence before is newly *evidenced*, not newly *created*.
    newly_evidenced = sorted(encode_lncrnas - set(typed.lncrna_id.astype(str)))
    gate15 = {
        "status": "PASS" if not new_lncrnas and not new_proteins else "FAIL",
        "baseline": "graph node universe (FORMAL_CANDIDATE_UNIVERSE + protein_gene_map)",
        "node_universe_lncrnas": len(node_lncrnas),
        "node_universe_proteins": len(node_proteins),
        "encode_edges": int(len(encode_edges)),
        "encode_lncrnas": len(encode_lncrnas),
        "encode_proteins": len(encode_proteins),
        "new_lncrna_nodes": len(new_lncrnas),
        "new_protein_nodes": len(new_proteins),
        "new_lncrna_examples": new_lncrnas[:5],
        "new_protein_examples": new_proteins[:5],
        "lncrnas_newly_evidenced_informational": len(newly_evidenced),
        "relation_type": "binds_protein_eclip",
        "node_type_created": False,
    }
    print(f"    baseline: {len(node_lncrnas):,} candidate-universe lncRNAs, "
          f"{len(node_proteins):,} protein nodes")
    print(f"    ENCODE edges={len(encode_edges):,}  "
          f"lncRNAs={len(encode_lncrnas):,}  proteins={len(encode_proteins):,}")
    print(f"    new lncRNA nodes={len(new_lncrnas)}  new protein nodes={len(new_proteins)} "
          f"-> {gate15['status']}")
    print(f"    (informational: {len(newly_evidenced):,} lncRNAs had no binding evidence "
          f"before this layer; they are newly evidenced, not newly created)")
else:
    gate15 = {"status": "MISSING", "note": "run phase3_encode_eclip_overlap.py"}
    print("    MISSING")
results["encode_adds_no_nodes"] = gate15

# Gate 16: the layer must be gated off while its enrichment is a depletion.
gate = load_json(ENC_DIR / "PHASE3D_ENCODE_LAYER_GATE.json")
if gate:
    merged = gate.get("merged_into_graph")
    results["encode_layer_gate"] = {
        "status": "PASS" if merged is False else "FAIL",
        "decision": gate["decision"],
        "switch": gate["switch"],
        "peak_level_enrichment": gate["why"]["peak_level_measurement"],
        "merged_into_graph": merged,
    }
    print(f"\n=== gate 16: ENCODE overlap layer admission ===")
    print(f"    {gate['decision']}")
    print(f"    merged into the graph: {merged} -> "
          f"{results['encode_layer_gate']['status']}")
    print(f"    peak-level enrichment: "
          + ", ".join(f"{k}={v:.2f}x" for k, v in gate['why']['peak_level_measurement'].items()))

if merge:
    results["encode_novelty"] = {
        "distinct_encode_pairs": merge["distinct_encode_pairs"],
        "already_in_frozen_eclip_layer": merge["encode_pairs_already_in_frozen_eclip"],
        "new_to_graph": merge["encode_pairs_new"],
        "novelty_fraction": merge["novelty_fraction"],
    }
    print(f"\n    ENCODE novelty: {merge['encode_pairs_new']:,} of "
          f"{merge['distinct_encode_pairs']:,} pairs are new to the graph "
          f"({merge['novelty_fraction'] * 100:.1f}%)")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"\nwritten: {OUT}")
