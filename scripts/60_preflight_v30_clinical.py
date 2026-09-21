#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cc_hhgt.common import input_candidates, load_config, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.0 clinical extension preflight")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = cfg["clinical"]
    v29 = cfg["_root"] / settings["v29_result_root"]
    required_v29 = [
        v29 / "reports" / "V2_9_FINAL_AUDIT.json",
        v29 / "strict_release" / "strict_cross_cancer_oof_prediction.parquet",
        v29 / "v2_9_downstream" / "v2_9_release" / "final_expert_fusion_table.parquet",
        v29 / "v2_9_downstream" / "lncrna_state_release" / "lncrna_state_final.parquet",
        v29 / "tables" / "pathway_family_member.parquet",
    ]
    missing = [str(p) for p in required_v29 if not p.exists()]
    clinical_sources = [p for p in input_candidates(cfg, "clinical_survival") if p.exists()]
    fold_sources = [p for p in input_candidates(cfg, "patient_fold_manifest") if p.exists()]
    if not clinical_sources:
        missing.append("inputs.clinical_survival")
    if not fold_sources:
        missing.append("inputs.patient_fold_manifest")
    audit_ok = False
    audit_path = v29 / "reports" / "V2_9_FINAL_AUDIT.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_ok = audit.get("status") == "PASS"
        if not audit_ok:
            missing.append("V2_9_FINAL_AUDIT.json status != PASS")
    payload = {
        "status": "PASS" if not missing else "FAIL",
        "v29_root": str(v29),
        "v29_audit_pass": audit_ok,
        "clinical_sources": [str(p) for p in clinical_sources],
        "patient_fold_sources": [str(p) for p in fold_sources],
        "missing": missing,
        "strict_graph_retrained_by_v30": False,
        "discovery_probability_modified_by_clinical": False,
    }
    out = cfg["_results"] / "reports" / "V3_0_CLINICAL_PREFLIGHT.json"
    write_json(payload, out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if missing:
        raise RuntimeError("V3.0 clinical preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
