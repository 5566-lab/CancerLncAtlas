#!/usr/bin/env python3
"""Read-only isolated server smoke for the frozen r8 remaining-15 supervisor."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


SUPERVISOR = Path(
    "./data/CancerLncAtlas/runtime/tools/"
    "single_cell_r7_r8_remaining15_supervisor_20260829_r1/scripts/"
    "run_v32_single_cell_r7_r8_remaining15_cohort.py"
)
SUPERVISOR_SHA256 = (
    "84232c5ae60f67112cf6e1fef2f9fe308fd6b804a22018020d69fb4287a7f77a"
)
FORMAL_PATHS = (
    Path(
        "./data/CancerLncAtlas/results/model/"
        "v32_single_cell_r7_fresh_streaming_r8_remaining15_20260829_r1"
    ),
    Path(
        "./data/CancerLncAtlas/runtime/audits/"
        "single_cell_r7_r8_remaining15_20260829_r1"
    ),
    Path(
        "./data/CancerLncAtlas/runtime/launch_logs/"
        "single_cell_r7_r8_remaining15_20260829_r1.launch.log"
    ),
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert file_sha256(SUPERVISOR) == SUPERVISOR_SHA256
    spec = importlib.util.spec_from_file_location("remaining15_smoke", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FORMAL_CANCERS == (
        module.EXTERNAL_CANCERS | module.REMAINING_CANCERS
    )
    assert len(module.FORMAL_CANCERS) == 17
    assert len(module.REMAINING_EXECUTION_ORDER) == 15
    assert module.EXTERNAL_CANCERS == {"HNSC", "SARC"}
    assert all(not path.exists() and not path.is_symlink() for path in FORMAL_PATHS)

    with tempfile.TemporaryDirectory(
        dir="./data/CancerLncAtlas/runtime"
    ) as temporary:
        audit = Path(temporary) / "audit"
        audit.mkdir()
        base = module._load_base()
        module.configure_base(base, audit)
        ordered = [row["cancer_id"] for row in base.ordered_formal_cancers()]
        assert ordered == list(module.REMAINING_EXECUTION_ORDER)
        plan_argv = base.runner_args(
            mode="plan",
            cancer="SKCM",
            plan_path=Path(temporary) / "SKCM.plan.json",
            output_root=None,
        )
        block_index = plan_argv.index("--association-pathway-block")
        assert plan_argv[block_index + 1] == "32"
        assert str(module.TOOL_ROOT / "scripts/run_v32_single_cell_r7_streaming.py") in plan_argv
        try:
            base.runner_args(
                mode="plan",
                cancer="HNSC",
                plan_path=Path(temporary) / "forbidden.json",
                output_root=None,
            )
        except module.RemainingCohortError:
            pass
        else:
            raise AssertionError("HNSC run was not rejected")

        validation = base.validate_frozen_tool()
        assert (audit / "PINNED_PYTHON_IDENTITY.json").is_file()
        binding_audit = json.loads(
            (audit / "EXTERNAL_BINDINGS_AUDIT.json").read_text(encoding="utf-8")
        )
        assert set(binding_audit["bindings"]) == {"HNSC", "SARC"}
        assert binding_audit["sarc_r3_partial_reused"] is False

        per_cancer = []
        for cancer in module.REMAINING_EXECUTION_ORDER:
            per_cancer.append(
                {
                    "cancer_id": cancer,
                    "output_root": f"/new/cancer_id={cancer}",
                    "success_sha256": "a" * 64,
                    "contract_sha256": "b" * 64,
                    "lineage_sha256": "c" * 64,
                    "file_manifest_sha256": "d" * 64,
                    "association_evidence_rows": 1,
                    "cells": 1,
                    "donors": 1,
                    "lncrnas": 1,
                }
            )
        formal_path = audit / "COHORT_SUCCESS.json"
        base.exclusive_json(
            formal_path,
            {
                "format": "OLD",
                "status": "SUCCESS_17_OF_17_AUDITED",
                "per_cancer": per_cancer,
                "cohort_contract_sha256": "old",
            },
        )
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        assert formal["status"] == "SUCCESS_17_OF_17_BOUND_AND_AUDITED"
        assert formal["formal_bound_cancer_count"] == 17
        assert len(formal["formal_bindings"]) == 17
        assert {row["cancer_id"] for row in formal["formal_bindings"]} == module.FORMAL_CANCERS
        contract = formal.pop("cohort_contract_sha256")
        assert contract == module.canonical_sha256(formal)

    print(
        json.dumps(
            {
                "base_supervisor_sha256": module.BASE_SUPERVISOR_SHA256,
                "external_bindings": ["HNSC", "SARC"],
                "formal_paths_absent": True,
                "formal_universe_count": 17,
                "memory_limit_bytes": module.MEMORY_LIMIT_BYTES,
                "memory_poll_seconds": module.POLL_SECONDS,
                "remaining_execution_order": ordered,
                "runner_manifest_sha256": module.TOOL_MANIFEST_SHA256,
                "status": "PASS",
                "supervisor_sha256": SUPERVISOR_SHA256,
                "tool_files_verified": len(validation["runner_files"]),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
