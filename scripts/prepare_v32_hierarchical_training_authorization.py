#!/usr/bin/env python3
"""Create hash-bound COMPUTE_HOST GPU task/approval artifacts after authorization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--endpoint-id",
        choices=("local4070", "local_gpu", "paid_gpu"),
        default="local_gpu",
    )
    parser.add_argument("--hardware-class")
    parser.add_argument(
        "--analysis-version",
        default="CancerLncAtlas_V3.2_HIERARCHICAL_ROUTED_CANDIDATE",
    )
    parser.add_argument("--explicitly-authorized", action="store_true")
    args = parser.parse_args(argv)
    if not args.explicitly_authorized:
        raise SystemExit("Refusing to create training approval without --explicitly-authorized")
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.orchestration import (
        build_task_manifest,
        extract_execution_policy,
        write_task_manifest,
    )
    from cc_hhgt.v32.training_guard import (
        APPROVAL_FORMAT,
        compute_artifact_hashes,
        load_structured_mapping,
    )
    from cc_hhgt.v32.multimodal_fusion import artifact_sha256
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
        validate_frozen_v32_prepared_fold_binding,
    )

    config = Path(args.config).resolve()
    prepared = Path(args.prepared_root).resolve()
    patient_folds = Path(args.patient_folds).resolve()
    patient_fold_receipt = Path(args.patient_fold_authority_receipt).resolve()
    patient_authority = validate_frozen_v32_patient_fold_binding(
        patient_folds, patient_fold_receipt
    )
    prepared_patient_authority = validate_frozen_v32_prepared_fold_binding(prepared)
    output = Path(args.output_root).resolve()
    if output.exists():
        raise RuntimeError(f"Authorization output already exists: {output}")
    output.mkdir(parents=True)
    payload = load_structured_mapping(config)
    policy = extract_execution_policy(payload)
    hardware_by_endpoint = {
        "local4070": "LOCAL_RTX_4070_TI_SUPER_16GB",
        "local_gpu": "COMPUTE_HOST_CUDA_GPU",
        "paid_gpu": "PAID_PREEMPTIBLE_GPU",
    }
    hardware_class = args.hardware_class or hardware_by_endpoint[args.endpoint_id]
    if hardware_class != hardware_by_endpoint[args.endpoint_id]:
        raise RuntimeError("Endpoint and hardware class are inconsistent")
    tasks = build_task_manifest(
        run_id=args.run_id,
        seed=args.seed,
        policy=policy,
        owner=args.endpoint_id,
        hardware_class=hardware_class,
        paid_task=args.endpoint_id == "paid_gpu",
    )
    task_manifest = write_task_manifest(output / "TASK_MANIFEST.tsv", tasks)
    fold_inputs = []
    for fold in range(5):
        path = prepared / f"PATIENT_FOLD_{fold}.pt"
        if not path.is_file():
            raise RuntimeError(f"Prepared hierarchical fold is missing: {path}")
        fold_inputs.append({"fold": fold, "path": str(path), "sha256": artifact_sha256(path)})
    input_manifest_payload = {
        "analysis_version": args.analysis_version,
        "candidate_only": True,
        "formal_v32_primary_unchanged": True,
        "patient_fold_authority": patient_authority,
        "prepared_patient_fold_authority": prepared_patient_authority,
        "fold_inputs": fold_inputs,
    }
    input_manifest = output / "INPUT_MANIFEST.json"
    input_manifest.write_text(json.dumps(input_manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = compute_artifact_hashes(
        repo_root=root,
        config_path=config,
        input_manifest_path=input_manifest,
        task_manifest_path=task_manifest,
    )
    approval = {
        "approval_format": APPROVAL_FORMAT,
        "training_authorized": True,
        "authorization_basis": "USER_EXPLICITLY_AUTHORIZED_FULL_RETRAIN",
        "run_id": args.run_id,
        "endpoint_id": args.endpoint_id,
        "hardware_class": hardware_class,
        "authorized_trainer": "cc_hhgt.v32.training:run_authorized_task",
        "patient_fold_authority": patient_authority,
        "approved_task_ids": [task["task_id"] for task in tasks],
        "artifact_hashes": hashes,
        "paid_enabled": policy.paid_enabled,
        "max_paid_hours": policy.max_paid_hours,
        "max_cost_cny": policy.max_cost_cny,
    }
    approval_path = output / "TRAINING_APPROVAL.json"
    approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "status": "AUTHORIZED_ARTIFACTS_READY",
        "run_id": args.run_id,
        "tasks": len(tasks),
        "task_manifest": str(task_manifest),
        "input_manifest": str(input_manifest),
        "approval": str(approval_path),
        "artifact_hashes": hashes,
        "endpoint_id": args.endpoint_id,
        "hardware_class": hardware_class,
    }
    (output / "SUCCESS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
