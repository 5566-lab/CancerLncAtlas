#!/usr/bin/env python3
"""Freeze a static+semantic audit of formal V3.2 patient-fold consumers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


DIRECT = (
    "scripts/run_v32_genomic_training.py",
    "scripts/train_v32_partitioned_genomic_heads.py",
    "scripts/run_v32_streaming_segment_cnv.py",
    "scripts/prepare_v32_pancancer_gistic_cnv.py",
    "scripts/map_v32_candidate_lncrna_segment_cnv.py",
    "scripts/run_v32_atac_materialization.py",
    "scripts/run_v32_atac_training.py",
    "scripts/run_v32_clinical_training.py",
    "scripts/run_v32_clinical_entity_training.py",
    "cc_hhgt/v32/state_training.py",
    "scripts/run_v32_continuous_activity.py",
    "scripts/prepare_v32_formal.py",
    "scripts/materialize_v32_exact_release.py",
    "scripts/materialize_v32_state_release.py",
)
PREPARED = (
    "cc_hhgt/v32/primary_fold_views.py",
    "cc_hhgt/v32/hierarchical_candidate_preparation.py",
    "scripts/prepare_v32_hierarchical_training_authorization.py",
    "scripts/build_v32_release_companions.py",
)
LAUNCHERS = (
    "scripts/server_launch_v32_cnv_router_cpu.sh",
    "scripts/server_train_v32_partitioned_genomic_heads_r1.sh",
    "scripts/server_launch_v32_atac_fresh_oof_r3.sh",
    "scripts/server_prepare_v32_g012_patient_first_r2.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r5.sh",
    "scripts/server_launch_v32_hierarchical_gpu.sh",
)
HISTORICAL = (
    "scripts/server_launch_v32_atac_fresh_oof.sh",
    "scripts/server_launch_v32_atac_fresh_oof_r2.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r1.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r2.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r3.sh",
    "scripts/server_prepare_v32_routing_fair_inputs_r4.sh",
    "scripts/prepare_v32_pilot.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict[str, object]:
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.patient_fold_authority import (
        FROZEN_V32_RECEIPT_SHA256,
        FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        validate_frozen_v32_patient_fold_binding,
    )

    authority_root = root / "artifacts/v32_patient_fold_authority_20260829_r1"
    authority = validate_frozen_v32_patient_fold_binding(
        authority_root / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        authority_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    records = []
    for kind, paths, token in (
        ("direct_consumer", DIRECT, "validate_frozen_v32_patient_fold_binding"),
        ("prepared_payload_consumer", PREPARED, "validate_frozen_v32_prepared_fold_binding"),
        ("current_server_launcher", LAUNCHERS, "gate_v32_patient_fold_authority.py"),
    ):
        for relative in paths:
            path = root / relative
            source = path.read_text(encoding="utf-8")
            checks = {
                "shared_gate_present": token in source,
                "legacy_manifest_literal_absent": "PATIENT_FOLD_MANIFEST.tsv" not in source,
            }
            if kind == "current_server_launcher":
                checks.update(
                    {
                        "sample_patient_map_named": "SAMPLE_PATIENT_FOLD_MAP.tsv" in source,
                        "receipt_named": "PATIENT_FOLD_AUTHORITY_RECEIPT.json" in source,
                    }
                )
            records.append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": sha256(path),
                    "checks": checks,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                }
            )
    failed = [record["path"] for record in records if record["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Patient-fold consumer binding audit failed: {failed}")
    return {
        "format": "CC_HHGT_V3_2_PATIENT_FOLD_CONSUMER_BINDING_AUDIT_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": authority,
        "pinned_identity": {
            "seed": 20260726,
            "cancers": 33,
            "patients": 10432,
            "sample_patient_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "consumer_records": records,
        "historical_not_current": list(HISTORICAL),
        "legacy_manifest_accepted_by_current_formal_surface": False,
        "missing_receipt_accepted_by_current_formal_surface": False,
        "sample_id_patient_derivation_accepted": False,
        "training_started": False,
        "server_accessed": False,
        "production_8260_touched": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    output = Path(args.output_root)
    if not output.is_absolute():
        output = (root / output).resolve()
    if output.exists():
        raise RuntimeError(f"Binding audit refuses output reuse: {output}")
    payload = audit(root)
    output.mkdir(parents=True)
    report = output / "AUDIT.json"
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = {
        "status": "PASS",
        "audit": {"path": str(report), "sha256": sha256(report)},
        "training_started": False,
        "server_accessed": False,
        "production_8260_touched": False,
    }
    (output / "SUCCESS.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
