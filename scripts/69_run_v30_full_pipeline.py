#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cc_hhgt.common import load_config, write_json


def run(command: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(command)}; log={log}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run complete V3.0 clinical extension on frozen V2.9 assets")
    parser.add_argument("--config", default="config/model_v3_0_clinical.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = os.environ.copy()
    result = cfg["_results"]
    logs = result / "logs"
    stages = [
        ("60_preflight", [args.python, "scripts/60_preflight_v30_clinical.py", "--config", args.config]),
        ("61_standardize", [args.python, "scripts/61_standardize_v30_clinical_endpoints.py", "--config", args.config]),
        ("62_build_data", [args.python, "scripts/62_build_v30_clinical_data.py", "--config", args.config]),
        ("63_survival_expert", [args.python, "scripts/63_train_v30_survival_expert.py", "--config", args.config, *( ["--resume"] if args.resume else [] )]),
        ("64_clinical_associations", [args.python, "scripts/64_build_v30_clinical_associations.py", "--config", args.config, "--jobs", str(args.jobs)]),
        ("65_clinical_expert", [args.python, "scripts/65_train_and_integrate_v30_clinical_expert.py", "--config", args.config]),
        ("66_audit", [args.python, "scripts/66_audit_v30_clinical.py", "--config", args.config]),
        ("67_web", [args.python, "scripts/67_materialize_v30_clinical_web_tables.py", "--config", args.config]),
        ("68_finalize", [args.python, "scripts/68_finalize_v30_release.py", "--config", args.config]),
    ]
    completed = []
    for name, command in stages:
        run(command, env, logs / f"{name}.log")
        completed.append(name)
        write_json({"status": "RUNNING", "completed_stages": completed, "last_stage": name}, result / "V3_0_PIPELINE_STATUS.json")
    payload = {"status": "COMPLETED", "version": "CC-HHGT_v3.0-clinical", "completed_stages": completed, "result_root": str(result), "v29_strict_models_retrained": False, "discovery_probability_modified": False}
    write_json(payload, result / "V3_0_PIPELINE_SUCCESS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
