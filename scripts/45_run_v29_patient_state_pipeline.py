#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cc_hhgt.common import load_config, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2.9 patient adapter and direct lncRNA-state pipeline using V2.9 strict assets")
    parser.add_argument("--config", default="config/model_v2_9_state_graph.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    downstream = cfg["_results"] / "v2_9_downstream"
    env = os.environ.copy()
    env.update({
        "CC_HHGT_RESULT_ROOT": str(downstream),
        "CC_HHGT_STRICT_OOF_PATH": str(cfg["_results"] / "strict_release" / "strict_cross_cancer_oof_prediction.parquet"),
        "CC_HHGT_STRICT_EMBEDDING_ROOT": str(cfg["_results"] / "v2_9_strict"),
        "CC_HHGT_V29_STRICT_ROOT": str(cfg["_results"] / "v2_9_strict"),
        "CC_HHGT_ADAPTER_DATA_ROOT": str(downstream / "adapter_data"),
        "CC_HHGT_ADAPTER_MODEL_ROOT": str(downstream / "models" / "cancer_adapter"),
        "CC_HHGT_ADAPTER_RESULT_ROOT": str(downstream / "cancer_adapter"),
        "CC_HHGT_ADAPTER_RUN_ROOT": str(downstream / "cancer_adapter_run"),
        "CC_HHGT_STATE_DATA_ROOT": str(downstream / "lncrna_state_data"),
        "CC_HHGT_STATE_MODEL_ROOT": str(downstream / "models" / "lncrna_state_experts"),
        "CC_HHGT_STATE_RESULT_ROOT": str(downstream / "lncrna_state_experts"),
        "CC_HHGT_STATE_RUN_ROOT": str(downstream / "lncrna_state_run"),
        "CC_HHGT_STATE_RELEASE_ROOT": str(downstream / "lncrna_state_release"),
        "CC_HHGT_RELEASE_DIR": str(downstream / "v2_9_release"),
        "CC_HHGT_STATE_TARGETS": "EXTEND::published_score,stemness_rna::RNAss,stemness_dna::DNAss,stemness_rna::EREG.EXPss",
        "CC_HHGT_REQUIRE_TELOMERASE_STATE": "1",
        "CC_HHGT_REQUIRE_STEMNESS_STATE": "1",
    })
    commands = [
        [args.python, "scripts/07_build_patient_adapter_data.py", *( ["--resume"] if args.resume else [] )],
        [args.python, "scripts/08_train_patient_adapter.py", "--epochs", "80", "--patience", "10"],
        [args.python, "scripts/09_aggregate_adapter_and_probabilities.py"],
        [args.python, "scripts/10_audit_cancer_native_moe.py"],
        [args.python, "scripts/27_build_lncrna_state_data.py"],
        [args.python, "scripts/28_train_lncrna_state_experts.py", "--epochs", "60", "--patience", "8"],
        [args.python, "scripts/29_integrate_lncrna_state_moe.py"],
        [args.python, "scripts/30_audit_lncrna_state_model.py"],
    ]
    downstream.mkdir(parents=True, exist_ok=True)
    for command in commands:
        print("[run]", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    payload = {
        "status": "COMPLETED",
        "result_root": str(downstream),
        "strict_oof": env["CC_HHGT_STRICT_OOF_PATH"],
        "strict_embeddings": env["CC_HHGT_STRICT_EMBEDDING_ROOT"],
        "required_states": env["CC_HHGT_STATE_TARGETS"].split(","),
    }
    write_json(payload, downstream / "V2_9_PATIENT_STATE_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
