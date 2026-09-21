#!/usr/bin/env python3
"""Create an immutable, fail-closed V2.9 release manifest and metadata bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cc_hhgt.common import load_config, read_table, write_json, write_table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_required(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize CC-HHGT V2.9 state-graph release")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    audit_path = cfg["_results"] / "reports" / "V2_9_FINAL_AUDIT.json"
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("V2.9 final audit is not PASS")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_root = cfg["_results"] / "releases" / timestamp
    deployed = release_root / "final_release"
    deployed.mkdir(parents=True, exist_ok=False)

    strict = cfg["_results"] / "strict_release"
    downstream = cfg["_results"] / "v2_9_downstream"
    full = downstream / "v2_9_release"
    graph = downstream / "graph_expert_stacking"
    event = downstream / "evidence_event_model"
    state = downstream / "lncrna_state_release"
    web = cfg["_results"] / "web_tables"
    files = {
        strict / "strict_cross_cancer_oof_prediction.parquet": deployed / "predictions" / "strict_cross_cancer_oof_prediction.parquet",
        strict / "strict_state_oof_prediction.parquet": deployed / "predictions" / "strict_state_oof_prediction.parquet",
        full / "final_expert_fusion_table.parquet": deployed / "predictions" / "final_expert_fusion_table.parquet",
        full / "full_expert_moe_oof_prediction.parquet": deployed / "predictions" / "full_expert_moe_oof_prediction.parquet",
        full / "candidate_union.parquet": deployed / "predictions" / "candidate_union.parquet",
        full / "cancer_specific_rescued_candidates.parquet": deployed / "predictions" / "cancer_specific_rescued_candidates.parquet",
        full / "candidate_union_unscored.parquet": deployed / "predictions" / "candidate_union_unscored.parquet",
        graph / "graph_expert_prediction.parquet": deployed / "experts" / "graph_expert_prediction.parquet",
        graph / "graph_expert_oof_prediction.parquet": deployed / "experts" / "graph_expert_oof_prediction.parquet",
        graph / "SUCCESS.json": deployed / "experts" / "GRAPH_EXPERT_SUCCESS.json",
        event / "evidence_transformer_prediction.parquet": deployed / "experts" / "evidence_transformer_prediction.parquet",
        event / "evidence_transformer_oof_prediction.parquet": deployed / "experts" / "evidence_transformer_oof_prediction.parquet",
        event / "EVIDENCE_TRANSFORMER_SUCCESS.json": deployed / "experts" / "EVIDENCE_TRANSFORMER_SUCCESS.json",
        state / "lncrna_state_final.parquet": deployed / "predictions" / "lncrna_state_final.parquet",
        state / "lncrna_state_significance_report.parquet": deployed / "predictions" / "lncrna_state_significance_report.parquet",
        state / "rnass_state_significance_report.parquet": deployed / "predictions" / "rnass_state_significance_report.parquet",
        state / "pancancer_lncrna_rnass_association.parquet": deployed / "predictions" / "pancancer_lncrna_rnass_association.parquet",
        full / "FULL_EXPERT_FUSION_SUCCESS.json": deployed / "reports" / "FULL_EXPERT_FUSION_SUCCESS.json",
        full / "FULL_MODEL_REQUIREMENTS_AUDIT.json": deployed / "reports" / "FULL_MODEL_REQUIREMENTS_AUDIT.json",
        state / "LNCRNA_STATE_AUDIT.json": deployed / "reports" / "LNCRNA_STATE_AUDIT.json",
        audit_path: deployed / "reports" / "V2_9_FINAL_AUDIT.json",
        cfg["_results"] / "tables" / "graph_state_asset_audit.json": deployed / "reports" / "graph_state_asset_audit.json",
    }
    for path in sorted(web.glob("web_*.parquet")):
        files[path] = deployed / "web_tables" / path.name
    for marker in ["V2_9_WEB_TABLES_SUCCESS.json", "V2_9_STATE_WEB_TABLES_SUCCESS.json"]:
        if (web / marker).exists():
            files[web / marker] = deployed / "web_tables" / marker
    for source, target in files.items():
        copy_required(source, target)

    # Copy trained lightweight gates and their metrics without copying all 279 checkpoints.
    optional_files = {
        full / "full_expert_moe_gate" / "discovery.pt": deployed / "models" / "full_expert_moe_gate" / "discovery.pt",
        full / "full_expert_moe_gate" / "confidence.pt": deployed / "models" / "full_expert_moe_gate" / "confidence.pt",
        full / "full_expert_moe_gate" / "crossfit_metrics.tsv": deployed / "models" / "full_expert_moe_gate" / "crossfit_metrics.tsv",
        graph / "graph_expert_stacker.pt": deployed / "models" / "graph_expert_stacker.pt",
        event / "evidence_transformer.pt": deployed / "models" / "evidence_transformer.pt",
    }
    for source, target in optional_files.items():
        if source.exists():
            copy_required(source, target)

    inventory_rows = []
    for path in sorted(deployed.rglob("*")):
        if not path.is_file():
            continue
        inventory_rows.append({
            "relative_path": str(path.relative_to(deployed)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    inventory = pd.DataFrame(inventory_rows)
    write_table(inventory, deployed / "V2_9_RELEASE_MANIFEST.tsv")

    card = f"""# CancerLncAtlas CC-HHGT V2.9 State-Graph Model Card

Generated: {datetime.now(timezone.utc).isoformat()}

## Core changes

- EXTEND, RNAss, DNAss and EREG.EXPss are formal `state` nodes.
- Signed gene/pathway/pathway-family/cancer-to-state relations participate in message passing.
- R-GCN, HGT and CC-HHGT-Strict are retrained for all LOCO folds and three seeds.
- Their OOF predictions are fused by a learned graph-expert stacker.
- Direct and indirect literature/experiment events are encoded separately by the Evidence Transformer.
- The final discovery and confidence probabilities use availability-masked OOF MoE gates.
- The strict encoder is trained jointly on lncRNA-pathway and auxiliary lncRNA-state tasks.
- EXTEND and DNAss are explicit Patient Adapter inputs.
- Cancer-specific lncRNA-RNAss associations use held-out patient effects and cancer/state-wide FDR.
- Pan-cancer lncRNA-RNAss associations use random-effects meta-analysis across cancers.

## Interpretation boundary

The model predicts reproducible association and network consistency. It does not establish that an lncRNA causally activates or suppresses RNAss, DNAss, EXTEND or another tumor state.

## Required audit

`V2_9_FINAL_AUDIT.json` is PASS. Missing state nodes, state embeddings, strict tasks, Patient Adapter state columns or state reports fail closed.
"""
    (deployed / "V2_9_MODEL_CARD.md").write_text(card, encoding="utf-8")

    # Rebuild inventory to include the model card and manifest itself.
    inventory_rows = []
    for path in sorted(deployed.rglob("*")):
        if path.is_file() and path.name != "V2_9_RELEASE_MANIFEST.tsv":
            inventory_rows.append({
                "relative_path": str(path.relative_to(deployed)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    inventory = pd.DataFrame(inventory_rows)
    write_table(inventory, deployed / "V2_9_RELEASE_MANIFEST.tsv")
    success = {
        "status": "COMPLETED",
        "version": "CC-HHGT_v2.9-state-graph",
        "analysis_version": cfg["analysis_version"],
        "release_root": str(release_root),
        "deployed_root": str(deployed),
        "n_manifest_files": int(len(inventory)),
        "state_nodes_in_strict_graph": True,
        "strict_models_retrained": ["R-GCN", "HGT", "CC-HHGT-Strict"],
    }
    write_json(success, deployed / "SUCCESS.json")
    write_json(success, cfg["_results"] / "V2_9_RELEASE_SUCCESS.json")
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
