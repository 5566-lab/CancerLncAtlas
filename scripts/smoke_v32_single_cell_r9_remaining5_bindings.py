#!/usr/bin/env python3
"""Read-only smoke for the r9 remaining-five immutable upstream bindings."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor", required=True, type=Path)
    args = parser.parse_args()
    supervisor = args.supervisor.resolve()
    spec = importlib.util.spec_from_file_location("r9_remaining5_smoke", supervisor)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import supervisor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bindings = module.external_bindings()
    if set(bindings) != module.EXTERNAL_CANCERS or len(bindings) != 12:
        raise RuntimeError("fixed upstream binding set drift")
    if module.REMAINING_EXECUTION_ORDER != ("ACC", "UCEC", "READ", "GBM", "PCPG"):
        raise RuntimeError("remaining-five order drift")
    value = {
        "status": "PASS",
        "supervisor_path": str(supervisor),
        "supervisor_sha256": module.sha256_file(supervisor),
        "immutable_binding_count": len(bindings),
        "immutable_cancers": sorted(bindings),
        "r8_typed_failure_path": str(module.R8_TYPED_FAILURE),
        "r8_typed_failure_sha256": module.R8_TYPED_FAILURE_SHA256,
        "r8_typed_failure_contract_sha256": (
            module.R8_TYPED_FAILURE_CONTRACT_SHA256
        ),
        "r9_execution_order": list(module.REMAINING_EXECUTION_ORDER),
        "bindings": bindings,
        "production_deployed": False,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
