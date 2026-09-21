#!/usr/bin/env python3
"""Seal the R11 formal-23 single-cell cohort after independent audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAL23 = {
    "ACC", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KIRC", "LAML", "LGG", "LUSC", "MESO", "OV", "PCPG",
    "READ", "SARC", "SKCM", "TGCT", "THYM", "UCEC", "UVM",
}
TYPED10 = {"BLCA", "KICH", "KIRP", "LIHC", "LUAD", "PAAD", "PRAD", "STAD", "THCA", "UCS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing or unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-status", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_status_path = args.run_status.resolve()
    binding_path = args.binding.resolve()
    audit_path = args.audit.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable formal-23 receipt exists: {output}")

    run_status = load_json(run_status_path)
    binding = load_json(binding_path)
    audit = load_json(audit_path)
    eligible = [str(value) for value in run_status.get("formal_eligible_cancers", [])]
    unavailable = run_status.get("typed_unavailable")
    bound = [str(value) for value in binding.get("formal_cancer_universe", [])]
    audited = [str(value) for value in audit.get("formal_cancer_universe", [])]
    if (
        run_status.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R11_RESCUED_RUN_STATUS_V1"
        or run_status.get("status") != "PASS_INPUT_CONTRACT_READY"
        or len(eligible) != 23
        or len(set(eligible)) != 23
        or set(eligible) != FORMAL23
        or not isinstance(unavailable, dict)
        or len(unavailable) != 10
        or set(unavailable) != TYPED10
        or any(not str(reason) for reason in unavailable.values())
        or set(eligible) & set(unavailable)
        or len(set(eligible) | set(unavailable)) != 33
    ):
        raise RuntimeError("R11 23+10 run-status contract drift")
    if (
        binding.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_BINDING_V1"
        or binding.get("status") != "BOUND_23_OF_23_PENDING_INDEPENDENT_AUDIT"
        or binding.get("formal_bound_cancer_count") != 23
        or binding.get("generated_in_this_run_count") != 10
        or binding.get("external_immutable_binding_count") != 13
        or set(bound) != set(eligible)
    ):
        raise RuntimeError("formal-23 binding contract drift")
    binding_sha = sha256(binding_path)
    if (
        audit.get("format")
        != "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_INDEPENDENT_AUDIT_V1"
        or audit.get("status") != "PASS_23_OF_23_INDEPENDENTLY_VERIFIED"
        or audit.get("formal_cancer_count") != 23
        or audit.get("generated_in_current_r11_run_count") != 10
        or audit.get("immutable_upstream_binding_count") != 13
        or set(audited) != set(eligible)
        or audit.get("cohort_success_sha256") != binding_sha
    ):
        raise RuntimeError("formal-23 independent audit does not attest binding")

    value = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_R11_FORMAL23_SUCCESS_V1",
        "status": "SUCCESS",
        "formal_eligible_cancer_count": 23,
        "typed_unavailable_cancer_count": 10,
        "formal_eligible_cancers": sorted(eligible),
        "typed_unavailable": dict(sorted(unavailable.items())),
        "formal_binding_path": str(binding_path),
        "formal_binding_sha256": binding_sha,
        "independent_audit_path": str(audit_path),
        "independent_audit_sha256": sha256(audit_path),
        "run_status_path": str(run_status_path),
        "run_status_sha256": sha256(run_status_path),
        "total_cells": int(audit["total_cells"]),
        "total_association_evidence_rows": int(
            audit["total_association_evidence_rows"]
        ),
        "historical_assets_relabelled_fresh": False,
        "typed_unavailable_rows_are_null": True,
        "typed_unavailable_changes_primary_score": False,
        "full_33_single_cell_coverage_claimed": False,
        "scientific_module_ready_for_binding": True,
        "website_bound": False,
        "production_deployed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({**value, "success_sha256": sha256(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
