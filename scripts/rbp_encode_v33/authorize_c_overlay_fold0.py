#!/usr/bin/env python3
"""Make one C fold training authority without duplicating the A payload.

Run on host 149 after the C graph sidecar and its SHA256 receipt exist.  This
creates only small config/manifest/approval files.  GPU creation is separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.c_graph_overlay import OVERLAY_FORMAT  # noqa: E402
from cc_hhgt.v32.training_guard import (  # noqa: E402
    APPROVAL_FORMAT, compute_artifact_hashes, guard_training_entry,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--old-config", type=Path, required=True)
    p.add_argument("--old-input-manifest", type=Path, required=True)
    p.add_argument("--overlay-receipt", type=Path, required=True)
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--parent-hash-receipt", type=Path)
    source_group.add_argument("--all-folds-sha-ready", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cloud-root", type=Path, required=True)
    p.add_argument("--code-archive", type=Path, required=True)
    p.add_argument("--wheel-archive", type=Path, required=True)
    p.add_argument("--wheel-requirements", type=Path, required=True)
    p.add_argument("--launcher", type=Path, required=True)
    p.add_argument("--billing-mode", choices=("Spot", "Postpay"), required=True)
    p.add_argument("--hardware-class", required=True)
    p.add_argument(
        "--mixed-precision", choices=("inherit", "bf16", "fp32"), default="inherit",
        help="Override runtime precision for the authorized GPU family.",
    )
    p.add_argument(
        "--reuse-existing-authority-no-rehash", action="store_true",
        help="Reuse the existing five-fold SHA receipt without rereading large inputs.",
    )
    p.add_argument("--max-paid-hours", type=float, required=True)
    p.add_argument("--max-cost-cny", type=float, required=True)
    args = p.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("C training authorization must run on host 149")
    if args.max_paid_hours <= 0 or args.max_cost_cny <= 0:
        raise RuntimeError("Positive paid time and cost limits are required")
    if not args.cloud_root.is_absolute():
        raise RuntimeError("Cloud runtime root must be absolute")
    expected_hardware = {
        "Spot": "PAID_PREEMPTIBLE_GPU",
        "Postpay": "PAID_ON_DEMAND_GPU",
    }[args.billing_mode]
    if args.hardware_class != expected_hardware:
        raise RuntimeError("Billing mode and hardware class differ")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training authorization: {output}")
    old_config = yaml.safe_load(args.old_config.read_text(encoding="utf-8"))
    old_manifest = json.loads(args.old_input_manifest.read_text(encoding="utf-8"))
    sidecar = json.loads(args.overlay_receipt.read_text(encoding="utf-8"))
    if (sidecar.get("format") != OVERLAY_FORMAT or sidecar.get("fold") != args.fold
            or sidecar.get("variant") != "G2" or sidecar.get("host") != "149"):
        raise RuntimeError(f"C sidecar receipt is not fold-{args.fold} G2 from host 149")
    overlay_path = Path(sidecar["path"]).resolve(strict=True)
    if overlay_path.stat().st_size != sidecar.get("bytes"):
        raise RuntimeError("C sidecar byte size differs from its receipt")
    if (not args.reuse_existing_authority_no_rehash
            and sha256(overlay_path) != sidecar.get("sha256")):
        raise RuntimeError("C sidecar bytes differ from its receipt")
    if args.reuse_existing_authority_no_rehash and args.all_folds_sha_ready is None:
        raise RuntimeError("No-rehash authorization requires the existing five-fold receipt")
    old_rows = old_manifest["fold_inputs"]
    parents = [row for row in old_rows if row.get("fold") == args.fold]
    if len(parents) != 1:
        raise RuntimeError(f"Old input manifest lacks one fold-{args.fold} A source")
    parent = parents[0]
    if (parent.get("sha256") != sidecar.get("source_prepared_sha256")
            or parent.get("path") != sidecar.get("source_prepared_path")):
        raise RuntimeError("C sidecar is bound to a different A source")
    parent_path = Path(parent["path"]).resolve(strict=True)
    if parent_path.stat().st_size != parent.get("bytes"):
        raise RuntimeError("A source size drift")
    if args.parent_hash_receipt is not None:
        digest_line = args.parent_hash_receipt.read_text(encoding="utf-8").strip().split()
        if len(digest_line) != 2 or digest_line != [parent["sha256"], str(parent_path)]:
            raise RuntimeError("A source SHA256 has not been independently verified on 149")
    else:
        ready = json.loads(args.all_folds_sha_ready.read_text(encoding="utf-8"))
        if (ready.get("status") != "PASS_C_ALL_FOLDS_INPUT_SHA256"
                or ready.get("target_host") != "149"):
            raise RuntimeError("Five-fold input SHA256 receipt is not ready on 149")
        matches = [row for row in ready.get("folds", []) if row.get("fold") == args.fold]
        if (len(matches) != 1
                or matches[0].get("source_path") != str(parent_path)
                or matches[0].get("source_sha256") != parent["sha256"]
                or matches[0].get("source_bytes") != parent_path.stat().st_size
                or matches[0].get("overlay_sha256") != sidecar["sha256"]):
            raise RuntimeError("Five-fold input SHA256 receipt differs from this C fold")
    receipt_sha = sidecar["graph_authority_receipt_sha256"]
    if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
        raise RuntimeError("C graph authority receipt SHA256 missing")

    cloud = args.cloud_root
    run_id = "v32-g012-g2-c-typed-encode-overlay-20260926-r1"
    task_id = f"{run_id}|PATIENT_FOLD_{args.fold}|CC-HHGT|20260726"
    owner = "paid_gpu"
    config = dict(old_config)
    config["contract_version"] = "3.2.0-c-typed-encode-overlay-20260926-r1"
    config["analysis_version"] = "CancerLncAtlas_V3.2_C_TYPED_ENCODE_V2_G2"
    config["execution_control"] = {
        **dict(old_config["execution_control"]),
        "execution_mode": "TRAINING", "training_authorized": True,
        "paid_enabled": True,
        "max_paid_hours": args.max_paid_hours,
        "max_cost_cny": args.max_cost_cny,
    }
    config["task_contract"] = {
        **dict(old_config["task_contract"]),
        "graph_variant": "G2",
        "rbp_evidence_mode": "C_TYPED_ENCODE_V2",
    }
    if args.mixed_precision != "inherit":
        config["runtime_profile"] = {
            **dict(old_config["runtime_profile"]),
            "mixed_precision": (
                "bf16" if args.mixed_precision == "bf16" else False
            ),
        }
    config["training_io"] = {
        "prepared_fold_pattern": str(cloud / "inputs" / "G2" / "A_G2_PATIENT_FOLD_{fold}.pt"),
        "c_graph_overlay_pattern": str(cloud / "inputs" / "G2" / "C_G2_PATIENT_FOLD_{fold}.pt"),
        "output_root": str(cloud / "results"),
    }
    source_cloud = cloud / "inputs" / "G2" / f"A_G2_PATIENT_FOLD_{args.fold}.pt"
    overlay_cloud = cloud / "inputs" / "G2" / f"C_G2_PATIENT_FOLD_{args.fold}.pt"
    manifest = {
        "format": "CANCERLNCATLAS_V32_C_GRAPH_OVERLAY_INPUT_MANIFEST_V1",
        "analysis_version": config["analysis_version"],
        "graph_variant": "G2", "rbp_evidence_mode": "C_TYPED_ENCODE_V2",
        "fold_inputs": [{
            **{key: value for key, value in parent.items() if key != "path"},
            "path": str(source_cloud),
            "graph_overlay": {
                "format": OVERLAY_FORMAT,
                "path": str(overlay_cloud),
                "bytes": sidecar["bytes"],
                "sha256": sidecar["sha256"],
                "source_prepared_sha256": parent["sha256"],
                "graph_authority_receipt_sha256": receipt_sha,
            },
        }],
        "source_prepared_receipt_path": str(parent_path),
        "source_prepared_sha256": parent["sha256"],
        "c_graph_overlay_receipt_path": str(args.overlay_receipt.resolve()),
        "c_graph_authority_receipt_sha256": receipt_sha,
        "sealed_test_read": False,
    }
    task = {
        "task_id": task_id, "run_id": run_id,
        "task_type": "CC_HHGT_PATIENT_FOLD", "model": "CC-HHGT",
        "patient_fold": str(args.fold), "seed": "20260726", "owner": owner,
        "hardware_class": args.hardware_class,
        "status": "PENDING", "blocked_reason": "", "paid_task": "true",
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
        writer.writeheader()
        writer.writerow(task)
    hashes = compute_artifact_hashes(
        repo_root=ROOT, config_path=config_path,
        input_manifest_path=input_path, task_manifest_path=task_path,
    )
    approval = {
        "approval_format": APPROVAL_FORMAT,
        "training_authorized": True,
        "run_id": run_id, "endpoint_id": owner,
        "hardware_class": args.hardware_class,
        "approved_task_ids": [task_id],
        "authorized_trainer": "cc_hhgt.v32.training:run_authorized_task",
        "artifact_hashes": hashes,
        "paid_enabled": True,
        "max_paid_hours": args.max_paid_hours,
        "max_cost_cny": args.max_cost_cny,
        "authorization_basis": "VERIFIED_C_GRAPH_OVERLAY_FOLD",
        "comp_share_billing_mode": args.billing_mode,
        "large_input_integrity_policy": (
            "REUSE_EXISTING_VERIFIED_RECEIPTS_SIZE_ONLY"
            if args.reuse_existing_authority_no_rehash else "REHASH"
        ),
    }
    approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    guard_training_entry(
        allow_training=True, repo_root=ROOT, config_path=config_path,
        input_manifest_path=input_path, task_manifest_path=task_path,
        approval_path=approval_path, run_id=run_id, task_id=task_id,
        endpoint_id=owner, hardware_class=args.hardware_class,
        trainer_specification="cc_hhgt.v32.training:run_authorized_task",
    )
    (output / "AUTH_READY.json").write_text(json.dumps({
        "status": "PASS_CPU_ONLY_C_FOLD_AUTHORIZATION",
        "host": "149", "gpu_started": False, "fold": args.fold,
        "task_id": task_id,
        "cloud_root": str(cloud), "artifact_hashes": hashes,
        "source_prepared_sha256": parent["sha256"],
        "c_graph_overlay_sha256": sidecar["sha256"],
        "c_graph_authority_receipt_sha256": receipt_sha,
        "comp_share_billing_mode": args.billing_mode,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    transfer_files = [
        ("code_archive", args.code_archive.resolve(strict=True), cloud / "code" / "C_TRAIN_CODE.tar.gz"),
        ("wheel_archive", args.wheel_archive.resolve(strict=True), cloud / "wheels" / "CLOUD_WHEELHOUSE.tar"),
        ("wheel_requirements", args.wheel_requirements.resolve(strict=True),
         cloud / "wheels" / "cloud_runtime_wheels_20260926.txt"),
        ("launcher", args.launcher.resolve(strict=True), cloud / "launch_c_gpu.py"),
        ("source_prepared", parent_path, source_cloud),
        ("c_graph_overlay", overlay_path, overlay_cloud),
        *((name, output / name, cloud / "auth" / name) for name in (
            "config.yaml", "INPUT_MANIFEST.json", "TASK_MANIFEST.tsv",
            "TRAINING_APPROVAL.json", "AUTH_READY.json",
        )),
    ]
    transfer = {
        "format": "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2",
        "preparation_host": "149", "fold": args.fold,
        "gpu_target_root": str(cloud),
        "billing_mode": args.billing_mode,
        "max_paid_hours": args.max_paid_hours,
        "max_cost_cny": args.max_cost_cny,
        "files": [
            {
                "role": role, "source": str(source), "destination": str(destination),
                "sha256": (
                    parent["sha256"] if role == "source_prepared"
                    else sidecar["sha256"] if role == "c_graph_overlay"
                    else sha256(source)
                ),
                "bytes": source.stat().st_size,
            }
            for role, source, destination in transfer_files
        ],
    }
    (output / "CLOUD_TRANSFER_MANIFEST.json").write_text(
        json.dumps(transfer, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "PASS_CPU_ONLY_C_FOLD_AUTHORIZATION",
                      "host": "149", "fold": args.fold, "task_id": task_id,
                      "output_root": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
