#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cc_hhgt.convergence_diagnostic import audit_convergence_task
from cc_hhgt.v30_integrity import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one isolated V3.1 convergence diagnostic")
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--model", default="cc_hhgt")
    parser.add_argument("--fold", default="LOCO_ACC")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epoch-cap", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_convergence_task(
        args.task_root,
        expected_model=args.model,
        expected_fold=args.fold,
        expected_seed=args.seed,
        expected_cap=args.epoch_cap,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS_CONVERGED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

