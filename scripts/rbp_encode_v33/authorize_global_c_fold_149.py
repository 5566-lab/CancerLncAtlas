#!/usr/bin/env python3
"""Authorize one final G2 fold from CPU-validated global-binding graph inputs."""
from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--template-auth", type=Path, required=True)
    p.add_argument("--parent-ready", type=Path, required=True)
    p.add_argument("--overlay-receipt", type=Path, required=True)
    p.add_argument("--cpu-receipt", type=Path, required=True)
    p.add_argument("--global-binding", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cloud-root", type=Path, required=True)
    p.add_argument("--code-archive", type=Path, required=True)
    p.add_argument("--wheel-archive", type=Path, required=True)
    p.add_argument("--wheel-requirements", type=Path, required=True)
    p.add_argument("--launcher", type=Path, required=True)
    p.add_argument("--max-paid-hours", type=float, required=True)
    p.add_argument("--max-cost-cny", type=float, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("Final G2 training authorization must run on host 149")
    if args.max_paid_hours <= 0 or args.max_cost_cny <= 0:
        raise RuntimeError("Positive paid time and cost limits are required")
    cloud = args.cloud_root
    output = args.output_root
    if not cloud.is_absolute() or not output.is_absolute() or output.exists():
        raise RuntimeError("Cloud/output roots must be absolute and output must be new")
    repo = args.repo_root.resolve(strict=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from cc_hhgt.v32.c_graph_overlay import OVERLAY_FORMAT
    from cc_hhgt.v32.training_guard import (
        APPROVAL_FORMAT, compute_artifact_hashes, guard_training_entry,
    )

    ready = json.loads(args.parent_ready.read_text(encoding="utf-8"))
    if ready.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256" or ready.get("target_host") != "149":
        raise RuntimeError("Historical parent-fold receipt is invalid")
    parents = [row for row in ready.get("folds", []) if row.get("fold") == args.fold]
    if len(parents) != 1:
        raise RuntimeError("Historical parent-fold receipt is not unique")
    parent = parents[0]
    source = Path(parent["source_path"]).resolve(strict=True)
    sidecar = json.loads(args.overlay_receipt.read_text(encoding="utf-8"))
    cpu = json.loads(args.cpu_receipt.read_text(encoding="utf-8"))
    overlay = Path(sidecar["path"]).resolve(strict=True)
    binding = args.global_binding.resolve(strict=True)
    if (sidecar.get("status") != "PASS_G2_GLOBAL_BINDING_SIDECAR"
            or sidecar.get("target_host") != "149"
            or sidecar.get("patient_fold") != args.fold
            or sidecar.get("graph_variant") != "G2"
            or sidecar.get("binding_context_policy") != "GLOBAL_PHYSICAL_BINDING"
            or sidecar.get("format") != OVERLAY_FORMAT
            or sidecar.get("global_binding") != str(binding)
            or sidecar.get("source_prepared_sha256") != parent["source_sha256"]
            or sidecar.get("graph_authority_receipt_sha256") != parent["graph_authority_receipt_sha256"]
            or source.stat().st_size != parent["source_bytes"]
            or overlay.stat().st_size != sidecar["bytes"]
            or cpu.get("status") != "PASS_G2_GLOBAL_C_CPU_MODEL_PREFLIGHT"
            or cpu.get("target_host") != "149" or cpu.get("fold") != args.fold
            or cpu.get("graph_overlay_path") != str(overlay)
            or cpu.get("graph_overlay_bytes") != sidecar["bytes"]
            or cpu.get("graph_edges") != sidecar["new_edges"]
            or cpu.get("binding_edges") != sidecar["new_binding_edges"]):
        raise RuntimeError("Final G2 graph/CPU authority is incomplete")

    template = args.template_auth.resolve(strict=True)
    prior_transfer = json.loads(
        (template / "CLOUD_TRANSFER_MANIFEST.json").read_text(encoding="utf-8")
    )
    prior_sources = [row for row in prior_transfer.get("files", [])
                     if row.get("role") == "source_prepared"]
    if (prior_transfer.get("format") != "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2"
            or prior_transfer.get("fold") != args.fold
            or len(prior_sources) != 1
            or prior_sources[0].get("source") != str(source)
            or prior_sources[0].get("bytes") != parent["source_bytes"]
            or prior_sources[0].get("sha256") != parent["source_sha256"]):
        raise RuntimeError("Historical cloud A fold transfer record is invalid")
    prior_cloud_source = Path(prior_sources[0]["destination"])
    if not prior_cloud_source.is_absolute():
        raise RuntimeError("Historical cloud A fold destination is not absolute")
    config = yaml.safe_load((template / "config.yaml").read_text(encoding="utf-8"))
    run_id = "v32-g012-g2-c-global-binding-20260928-r1"
    task_id = f"{run_id}|PATIENT_FOLD_{args.fold}|CC-HHGT|20260726"
    config["contract_version"] = "3.2.0-c-global-binding-20260928-r1"
    config["analysis_version"] = "CancerLncAtlas_V3.2_C_GLOBAL_BINDING_G2"
    config["execution_control"] = {
        **config["execution_control"], "execution_mode": "TRAINING",
        "training_authorized": True, "paid_enabled": True,
        "max_paid_hours": args.max_paid_hours,
        "max_cost_cny": args.max_cost_cny,
    }
    config["task_contract"] = {
        **config["task_contract"], "graph_variant": "G2",
        "rbp_evidence_mode": "C_GLOBAL_PHYSICAL_BINDING",
    }
    config["runtime_profile"] = {
        **config["runtime_profile"],
        "max_coverage_cycles": 120,
        "patience_coverage_cycles": 6,
        "require_early_stopping_before_cap": True,
    }
    config["training_io"] = {
        "prepared_fold_pattern": str(cloud / "inputs" / "G2" / "A_G2_PATIENT_FOLD_{fold}.pt"),
        "c_graph_overlay_pattern": str(cloud / "inputs" / "G2" / "G2_GLOBAL_FOLD_{fold}.pt"),
        "output_root": str(cloud / "results"),
    }
    source_cloud = cloud / "inputs" / "G2" / f"A_G2_PATIENT_FOLD_{args.fold}.pt"
    overlay_cloud = cloud / "inputs" / "G2" / f"G2_GLOBAL_FOLD_{args.fold}.pt"
    binding_cloud = cloud / "inputs" / "lnc_protein_binding_global.parquet"
    manifest = {
        "format": "CANCERLNCATLAS_V32_C_GRAPH_OVERLAY_INPUT_MANIFEST_V1",
        "analysis_version": config["analysis_version"],
        "graph_variant": "G2", "rbp_evidence_mode": "C_GLOBAL_PHYSICAL_BINDING",
        "fold_inputs": [{
            "fold": args.fold, "bytes": parent["source_bytes"],
            "corrected_local_cnv": True,
            "path": str(source_cloud), "sha256": parent["source_sha256"],
            "graph_overlay": {
                "format": OVERLAY_FORMAT, "path": str(overlay_cloud),
                "bytes": sidecar["bytes"],
                "verification_mode": "SIZE_AND_GRAPH_SEMANTICS_V1",
                "expected_graph_edges": sidecar["new_edges"],
                "expected_binding_edges": sidecar["new_binding_edges"],
                "global_binding_source_path": str(binding_cloud),
                "source_prepared_sha256": parent["source_sha256"],
                "graph_authority_receipt_sha256": parent["graph_authority_receipt_sha256"],
            },
        }],
        "source_prepared_receipt_path": str(source),
        "source_prepared_sha256": parent["source_sha256"],
        "c_graph_overlay_receipt_path": str(args.overlay_receipt.resolve(strict=True)),
        "c_graph_authority_receipt_sha256": parent["graph_authority_receipt_sha256"],
        "sealed_test_read": False,
    }
    task = {
        "task_id": task_id, "run_id": run_id,
        "task_type": "CC_HHGT_PATIENT_FOLD", "model": "CC-HHGT",
        "patient_fold": str(args.fold), "seed": "20260726", "owner": "paid_gpu",
        "hardware_class": "PAID_ON_DEMAND_GPU", "status": "PENDING",
        "blocked_reason": "", "paid_task": "true",
    }
    output.mkdir(parents=True)
    config_path = output / "config.yaml"
    input_path = output / "INPUT_MANIFEST.json"
    task_path = output / "TASK_MANIFEST.tsv"
    approval_path = output / "TRAINING_APPROVAL.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    input_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with task_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerow(task)
    # The existing trainer guard requires digests of small code/config files.
    # The 12.8 GB parent and 3.3 GB C sidecar are not rehashed.
    hashes = compute_artifact_hashes(
        repo_root=repo, config_path=config_path,
        input_manifest_path=input_path, task_manifest_path=task_path,
    )
    approval = {
        "approval_format": APPROVAL_FORMAT, "training_authorized": True,
        "run_id": run_id, "endpoint_id": "paid_gpu",
        "hardware_class": "PAID_ON_DEMAND_GPU", "approved_task_ids": [task_id],
        "authorized_trainer": "cc_hhgt.v32.training:run_authorized_task",
        "artifact_hashes": hashes, "paid_enabled": True,
        "max_paid_hours": args.max_paid_hours, "max_cost_cny": args.max_cost_cny,
        "authorization_basis": "CPU_VALIDATED_GLOBAL_G2_GRAPH",
        "comp_share_billing_mode": "Postpay",
        "large_input_integrity_policy": "USER_DIRECTED_SIZE_AND_GRAPH_SEMANTICS_NO_REHASH",
    }
    approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    guard_training_entry(
        allow_training=True, repo_root=repo, config_path=config_path,
        input_manifest_path=input_path, task_manifest_path=task_path,
        approval_path=approval_path, run_id=run_id, task_id=task_id,
        endpoint_id="paid_gpu", hardware_class="PAID_ON_DEMAND_GPU",
        trainer_specification="cc_hhgt.v32.training:run_authorized_task",
    )
    ready_path = output / "AUTH_READY.json"
    ready_path.write_text(json.dumps({
        "status": "PASS_CPU_ONLY_GLOBAL_G2_FOLD_AUTHORIZATION",
        "host": "149", "gpu_started": False, "fold": args.fold,
        "task_id": task_id, "cloud_root": str(cloud),
        "artifact_hashes": hashes,
        "source_prepared_sha256": parent["source_sha256"],
        "graph_overlay_bytes": sidecar["bytes"],
        "global_binding_source_bytes": binding.stat().st_size,
        "graph_edges": sidecar["new_edges"],
        "binding_edges": sidecar["new_binding_edges"],
        "c_graph_authority_receipt_sha256": parent["graph_authority_receipt_sha256"],
        "comp_share_billing_mode": "Postpay",
        "large_files_rehashed": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    files = [
        ("code_archive", args.code_archive.resolve(strict=True), cloud / "code" / "C_TRAIN_CODE.tar.gz"),
        ("wheel_archive", args.wheel_archive.resolve(strict=True), cloud / "wheels" / "CLOUD_WHEELHOUSE.tar"),
        ("wheel_requirements", args.wheel_requirements.resolve(strict=True), cloud / "wheels" / "requirements.txt"),
        ("launcher", args.launcher.resolve(strict=True), cloud / "launch_c_gpu.py"),
        ("source_prepared", source, source_cloud),
        ("c_graph_overlay", overlay, overlay_cloud),
        ("global_binding_source", binding, binding_cloud),
        *((name, output / name, cloud / "auth" / name) for name in (
            "config.yaml", "INPUT_MANIFEST.json", "TASK_MANIFEST.tsv",
            "TRAINING_APPROVAL.json", "AUTH_READY.json",
        )),
    ]
    transfer = {
        "format": "CANCERLNCATLAS_C_GLOBAL_G2_FOLD_CLOUD_TRANSFER_V1",
        "preparation_host": "149", "fold": args.fold,
        "gpu_target_root": str(cloud), "billing_mode": "Postpay",
        "max_paid_hours": args.max_paid_hours,
        "max_cost_cny": args.max_cost_cny,
        "files": [{"role": role, "source": str(src), "destination": str(dst),
                   "bytes": src.stat().st_size,
                   **({"sha256_reused": parent["source_sha256"],
                       "reuse_cloud_source": str(prior_cloud_source)}
                      if role == "source_prepared" else {})}
                  for role, src, dst in files],
    }
    (output / "CLOUD_TRANSFER_MANIFEST.json").write_text(
        json.dumps(transfer, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status":"PASS_CPU_ONLY_GLOBAL_G2_FOLD_AUTHORIZATION",
                      "target_host":"149", "fold":args.fold,
                      "output_root":str(output), "large_files_rehashed":False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
