#!/usr/bin/env python3
"""Fail-closed audit of the complete expert architecture and requested features."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(os.getenv("CC_HHGT_RELEASE_DIR", str(ROOT / "results" / "v2_8_cancer_native_moe_release")))
GRAPH = Path(os.getenv("CC_HHGT_GRAPH_STACK_ROOT", str(ROOT / "results" / "graph_expert_stacking")))
EVENT = Path(os.getenv("CC_HHGT_EVIDENCE_EVENT_ROOT", str(ROOT / "results" / "evidence_event_model")))
STATE = Path(os.getenv("CC_HHGT_STATE_RELEASE_ROOT", str(ROOT / "results" / "lncrna_state_release")))
ADAPTER_MODEL_ROOT = Path(os.getenv("CC_HHGT_ADAPTER_MODEL_ROOT", str(ROOT / "models" / "cancer_adapter")))
REQUIRE_DIRECT_EVENTS = os.getenv("CC_HHGT_REQUIRE_DIRECT_EVENTS", "1").strip().lower() not in {"0", "false", "no"}


def check_file(path: Path, columns: set[str] | None = None) -> dict:
    if not path.exists():
        return {"status": "FAIL", "detail": f"missing {path}"}
    if columns:
        observed = set(pq.read_schema(path).names)
        missing = sorted(columns - observed)
        if missing:
            return {"status": "FAIL", "detail": f"missing columns {missing}"}
    return {"status": "PASS", "detail": str(path)}


def checkpoint_feature_audit() -> dict:
    files = sorted(ADAPTER_MODEL_ROOT.glob("*/*/seed_*/best.pt"))
    mutation = set(); communication = set(); single_cell = set(); all_features = set()
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        columns = payload.get("patient_feature_columns", [])
        all_features.update(columns)
        mutation.update(c for c in columns if c.startswith("mutation_") or "mutation" in c.lower())
        communication.update(c for c in columns if c.startswith("communication_"))
        single_cell.update(c for c in columns if c.startswith(("sc_", "ucell_", "celltype_")))
    forbidden = sorted(c for c in all_features if c in {
        "candidate_from_evidence", "direct_evidence_available", "direct_pmid_count",
        "direct_event_count", "evidence_integrated_probability",
    } or c.startswith("direct_literature_"))
    return {
        "n_checkpoints": len(files),
        "n_patient_features": len(all_features),
        "mutation_features": sorted(mutation),
        "communication_features": sorted(communication),
        "single_cell_features": sorted(single_cell),
        "forbidden_discovery_features": forbidden,
    }


def main() -> int:
    checks = []
    def add(name: str, status: str, detail: object, required: bool = True):
        checks.append({"requirement": name, "status": status, "required_for_release": required, "detail": detail})

    graph_success = GRAPH / "SUCCESS.json"
    event_success = EVENT / "EVIDENCE_TRANSFORMER_SUCCESS.json"
    full_success = RELEASE / "FULL_EXPERT_FUSION_SUCCESS.json"
    for name, path in [
        ("graph_expert_stacking", graph_success),
        ("event_transformer", event_success),
        ("full_expert_fusion", full_success),
    ]:
        if not path.exists():
            add(name, "FAIL", f"missing {path}")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            add(name, "PASS" if payload.get("status") == "COMPLETED" else "FAIL", payload)

    graph_pred = GRAPH / "graph_expert_prediction.parquet"
    result = check_file(graph_pred, {
        "graph_ensemble_probability", "graph_weight_rgcn", "graph_weight_hgt", "graph_weight_cc_hhgt"
    })
    add("three_graph_experts_available", result["status"], result["detail"])
    if result["status"] == "PASS":
        frame = pd.read_parquet(graph_pred, columns=[
            "rgcn_probability", "hgt_probability", "cc_hhgt_probability",
            "graph_ensemble_probability", "graph_weight_rgcn", "graph_weight_hgt", "graph_weight_cc_hhgt",
        ])
        available_counts = {
            "rgcn": int(frame["rgcn_probability"].notna().sum()),
            "hgt": int(frame["hgt_probability"].notna().sum()),
            "cc_hhgt": int(frame["cc_hhgt_probability"].notna().sum()),
        }
        weights = frame[["graph_weight_rgcn", "graph_weight_hgt", "graph_weight_cc_hhgt"]].sum(axis=1)
        add("graph_stacker_weight_normalization", "PASS" if np.allclose(weights[frame["graph_ensemble_probability"].notna()], 1, atol=1e-4) else "FAIL", available_counts)

    event_pred = EVENT / "evidence_transformer_prediction.parquet"
    result = check_file(event_pred, {
        "indirect_mechanism_probability", "direct_evidence_probability", "neural_evidence_probability",
        "indirect_evidence_available", "direct_evidence_available",
    })
    add("direct_indirect_event_separation", result["status"], result["detail"])
    event_build = EVENT / "EVENT_BUILD_SUCCESS.json"
    if event_build.exists():
        payload = json.loads(event_build.read_text(encoding="utf-8"))
        direct_count = int(payload.get("direct_events", 0))
        direct_status = "PASS" if direct_count > 0 else ("FAIL" if REQUIRE_DIRECT_EVENTS else "PARTIAL")
        add("explicit_direct_literature_events_present", direct_status, payload, required=REQUIRE_DIRECT_EVENTS)

    final_path = RELEASE / "final_expert_fusion_table.parquet"
    result = check_file(final_path, {
        "discovery_ranking_probability", "fused_confidence_probability",
        "graph_ensemble_probability", "cancer_native_probability",
        "indirect_mechanism_probability", "direct_evidence_probability",
        "direct_evidence_used_in_discovery",
    })
    add("final_full_expert_table", result["status"], result["detail"])
    if result["status"] == "PASS":
        final = pd.read_parquet(final_path)
        direct_leak = final["direct_evidence_used_in_discovery"].fillna(False).astype(bool).any()
        add("direct_evidence_excluded_from_discovery", "FAIL" if direct_leak else "PASS", {"leaking_rows": int(direct_leak)})
        discovery_cols = [c for c in final if c.startswith("discovery_weight_")]
        confidence_cols = [c for c in final if c.startswith("confidence_weight_")]
        dvalid = final["discovery_ranking_probability"].notna()
        cvalid = final["fused_confidence_probability"].notna()
        add("discovery_weight_normalization", "PASS" if discovery_cols and np.allclose(final.loc[dvalid, discovery_cols].sum(axis=1), 1, atol=1e-4) else "FAIL", discovery_cols)
        add("confidence_weight_normalization", "PASS" if confidence_cols and np.allclose(final.loc[cvalid, confidence_cols].sum(axis=1), 1, atol=1e-4) else "FAIL", confidence_cols)

    discovery_gate = RELEASE / "full_expert_moe_gate" / "discovery.pt"
    confidence_gate = RELEASE / "full_expert_moe_gate" / "confidence.pt"
    if discovery_gate.exists() and confidence_gate.exists():
        dckpt = torch.load(discovery_gate, map_location="cpu", weights_only=False)
        cckpt = torch.load(confidence_gate, map_location="cpu", weights_only=False)
        dexperts = list(dckpt.get("expert_columns", []))
        dquality = list(dckpt.get("quality_columns", []))
        forbidden = [
            value for value in dexperts + dquality
            if value == "evidence_integrated_probability"
            or (
                "direct" in value.lower()
                and not value.lower().startswith("indirect")
            )
        ]
        add("discovery_gate_has_no_direct_evidence_inputs", "PASS" if not forbidden else "FAIL", {
            "experts": dexperts, "quality": dquality, "forbidden": forbidden,
        })
        add("confidence_gate_contains_direct_expert", "PASS" if "direct_evidence_probability" in cckpt.get("expert_columns", []) else "FAIL", cckpt.get("expert_columns", []))
    else:
        add("discovery_gate_has_no_direct_evidence_inputs", "FAIL", "full gate checkpoints missing")
        add("confidence_gate_contains_direct_expert", "FAIL", "full gate checkpoints missing")

    state_success = STATE / "LNCRNA_STATE_MODEL_SUCCESS.json"
    state_audit = STATE / "LNCRNA_STATE_AUDIT.json"
    state_final = STATE / "lncrna_state_final.parquet"
    if state_success.exists() and state_audit.exists() and state_final.exists():
        state_payload = json.loads(state_success.read_text(encoding="utf-8"))
        state_audit_payload = json.loads(state_audit.read_text(encoding="utf-8"))
        add("direct_lncrna_tumor_state_model", "PASS" if state_payload.get("status") == "COMPLETED" else "FAIL", state_payload)
        add("direct_lncrna_tumor_state_audit", "PASS" if state_audit_payload.get("status") == "PASS" else "FAIL", state_audit_payload)
        state_frame = pd.read_parquet(state_final, columns=["state_id", "lncrna_state_probability", "replication_class"])
        state_names = sorted(state_frame["state_id"].astype(str).unique())
        state_text = "|".join(state_names).lower()
        add("telomerase_and_stemness_state_targets", "PASS" if (("extend" in state_text or "telomerase" in state_text) and any(token in state_text for token in ["rnass", "dnass", "stemness", "ereg.expss", "ereg_expss"])) else "FAIL", state_names)
    else:
        add("direct_lncrna_tumor_state_model", "FAIL", {"success": str(state_success), "audit": str(state_audit), "final": str(state_final)})
        add("direct_lncrna_tumor_state_audit", "FAIL", "lncRNA-state outputs missing")
        add("telomerase_and_stemness_state_targets", "FAIL", "lncRNA-state outputs missing")

    feature = checkpoint_feature_audit()
    add("patient_native_is_trained", "PASS" if feature["n_checkpoints"] > 0 else "FAIL", feature)
    add("patient_native_has_no_direct_evidence_leakage", "PASS" if not feature["forbidden_discovery_features"] else "FAIL", feature["forbidden_discovery_features"])
    add("mutation_features_enter_patient_model", "PASS" if feature["mutation_features"] else "PARTIAL", feature["mutation_features"], required=False)
    add("single_cell_features_enter_patient_model", "PASS" if feature["single_cell_features"] else "PARTIAL", feature["single_cell_features"], required=False)
    add("cell_communication_features_enter_patient_model", "PASS" if feature["communication_features"] else "NOT_IMPLEMENTED", feature["communication_features"], required=False)

    # Requirements that are intentionally outside this lncRNA target model.
    add("lncrna_variant_position_encoder", "NOT_IMPLEMENTED", "mutation aggregates may enter; chr/start/end/splice-distance/RBP-site encoding is absent", required=False)
    add("protein_coding_gene_as_prediction_subject", "NOT_IMPLEMENTED", "current candidate key and labels are lncRNA-specific", required=False)
    add("dynamic_crawler_auto_retraining", "NOT_IMPLEMENTED", "new events can be imported and evidence/fusion retrained, but scheduler/crawler ingestion is external", required=False)
    add("causal_regulation_claim", "OUT_OF_SCOPE", "association and evidence ranking do not establish causality", required=False)
    add("patient_individual_probability", "OUT_OF_SCOPE", "current patient crossfit predicts cancer-level relation reproducibility", required=False)

    required_failures = [row for row in checks if row["required_for_release"] and row["status"] != "PASS"]
    status = "PASS" if not required_failures else "FAIL"
    payload = {
        "generated_at": datetime.now().isoformat(), "status": status,
        "required_failures": required_failures, "checks": checks,
    }
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.tsv", sep="\t", index=False)
    lines = ["# CancerLncAtlas Full Model Requirements Audit", "", f"Status: **{status}**", "", "| Requirement | Status | Required | Detail |", "|---|---:|---:|---|"]
    for row in checks:
        detail = str(row["detail"]).replace("|", "/").replace("\n", " ")
        if len(detail) > 260: detail = detail[:257] + "..."
        lines.append(f"| {row['requirement']} | {row['status']} | {row['required_for_release']} | {detail} |")
    (RELEASE / "FULL_MODEL_REQUIREMENTS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("Required full-model audit checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
