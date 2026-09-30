#!/usr/bin/env python3
"""Verify one C fold on a CompShare GPU, then run the guarded trainer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(status: str, **values: object) -> None:
    print(json.dumps({"status": status, **values}, sort_keys=True), flush=True)


def unpack(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extracted runtime: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            path = Path(member.name)
            if (path.is_absolute() or ".." in path.parts
                    or not (member.isfile() or member.isdir())):
                raise RuntimeError(f"Unsafe cloud archive member: {member.name}")
        handle.extractall(destination)


def telemetry(stop: threading.Event) -> None:
    sequence = 0
    while not stop.is_set():
        sequence += 1
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
        )
        if result.returncode:
            emit("GPU_TELEMETRY_FAILED", sequence=sequence, error=result.stderr[-500:])
        else:
            emit("GPU_TELEMETRY", sequence=sequence, measurements=result.stdout.strip())
        stop.wait(60)


def run(args: argparse.Namespace) -> int:
    if socket.gethostname() == "149":
        raise RuntimeError("GPU trainer must not run on CPU preparation host 149")
    root = args.runtime_root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    if manifest_path != root / "auth" / "CLOUD_TRANSFER_MANIFEST.json":
        raise RuntimeError("Cloud transfer manifest path is outside the launch contract")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "CANCERLNCATLAS_C_FOLD_CLOUD_TRANSFER_V2"
            or manifest.get("preparation_host") != "149"
            or manifest.get("gpu_target_root") != str(root)
            or manifest.get("billing_mode") != args.billing_mode
            or type(manifest.get("fold")) is not int
            or manifest["fold"] not in range(5)):
        raise RuntimeError("Cloud transfer scope or billing mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Cloud transfer manifest lacks files")
    records = {row["role"]: row for row in files}
    required = {
        "code_archive", "wheel_archive", "wheel_requirements", "launcher",
        "source_prepared", "c_graph_overlay", "config.yaml", "INPUT_MANIFEST.json",
        "TASK_MANIFEST.tsv", "TRAINING_APPROVAL.json", "AUTH_READY.json",
    }
    if set(records) != required or len(files) != len(required):
        raise RuntimeError("Cloud transfer manifest has missing or duplicate roles")
    no_rehash = args.staged_no_rehash_receipt is not None
    if no_rehash:
        staged_path = args.staged_no_rehash_receipt.resolve(strict=True)
        if staged_path != root / "auth" / "CLOUD_STAGED_NO_REHASH.json":
            raise RuntimeError("No-rehash receipt path is outside the launch contract")
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        if (staged.get("format") != "C_GRAPH_CLOUD_STAGED_NO_REHASH_V1"
                or staged.get("integrity_policy")
                != "USER_DIRECTED_REUSE_VERIFIED_RECEIPTS_SIZE_ONLY"
                or staged.get("instance_id") != args.instance_id
                or staged.get("fold") != manifest["fold"]
                or staged.get("gpu_target_root") != str(root)
                or set(staged.get("verified_roles", [])) != required):
            raise RuntimeError("No-rehash staging receipt differs from this launch")
    else:
        if not args.manifest_sha256 or sha256(manifest_path) != args.manifest_sha256:
            raise RuntimeError("Cloud transfer manifest SHA256 mismatch")
    for role, row in records.items():
        path = Path(row["destination"]).resolve(strict=True)
        if not path.is_relative_to(root) or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Cloud transfer file path or size drift: {role}")
        if not no_rehash and sha256(path) != row["sha256"]:
            raise RuntimeError(f"Cloud transfer SHA256 mismatch: {role}")
    emit(
        "CLOUD_INPUT_SIZE_VERIFIED_NO_REHASH" if no_rehash else "CLOUD_INPUT_SHA256_VERIFIED",
        instance_id=args.instance_id, files=len(files), preparation_host="149",
        gpu_host=socket.gethostname(),
    )

    auth = root / "auth"
    ready = json.loads((auth / "AUTH_READY.json").read_text(encoding="utf-8"))
    if (ready.get("status") != "PASS_CPU_ONLY_C_FOLD_AUTHORIZATION"
            or ready.get("host") != "149" or ready.get("cloud_root") != str(root)
            or ready.get("comp_share_billing_mode") != args.billing_mode
            or ready.get("fold") != manifest["fold"]):
        raise RuntimeError("C fold training authorization scope mismatch")
    if ready.get("source_prepared_sha256") != records["source_prepared"]["sha256"]:
        raise RuntimeError("A source authorization SHA256 mismatch")
    if ready.get("c_graph_overlay_sha256") != records["c_graph_overlay"]["sha256"]:
        raise RuntimeError("C graph authorization SHA256 mismatch")
    code = root / "code" / "live"
    wheels = root / "wheels" / "files"
    runtime_ready = root / "NO_GPU_RUNTIME_READY.json"
    if runtime_ready.exists():
        prior = json.loads(runtime_ready.read_text(encoding="utf-8"))
        if (prior.get("status") != "PASS_NO_GPU_RUNTIME_PREPARATION"
                or prior.get("instance_id") != args.instance_id
                or prior.get("runtime_root") != str(root)
                or not code.is_dir() or not wheels.is_dir()):
            raise RuntimeError("Existing no-GPU runtime receipt is invalid")
    else:
        unpack(Path(records["code_archive"]["destination"]), code)
        unpack(Path(records["wheel_archive"]["destination"]), wheels)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-index", "--disable-pip-version-check",
            "--find-links", str(wheels), "-r", records["wheel_requirements"]["destination"],
            "torch-geometric==2.8.0",
        ], check=True)
        runtime_ready.write_text(json.dumps({
            "status": "PASS_NO_GPU_RUNTIME_PREPARATION",
            "instance_id": args.instance_id,
            "runtime_root": str(root),
            "gpu_required": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.prepare_only:
        emit("NO_GPU_RUNTIME_PREPARED", instance_id=args.instance_id)
        return 0

    import torch
    import torch_geometric

    if not torch.__version__.startswith("2.5.1") or torch_geometric.__version__ != "2.8.0":
        raise RuntimeError("Cloud image Torch/PyG version mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CompShare GPU is unavailable to PyTorch")
    probe = torch.tensor([1.0, 2.0], device="cuda")
    if float((probe * 2).sum().item()) != 6.0:
        raise RuntimeError("CompShare CUDA arithmetic probe failed")
    subprocess.run(["nvidia-smi", "-L"], check=True, timeout=15)
    emit("CLOUD_CUDA_PREFLIGHT_PASSED", instance_id=args.instance_id,
         torch=torch.__version__, torch_geometric=torch_geometric.__version__)

    task_id = str(ready["task_id"])
    with (auth / "TASK_MANIFEST.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if (len(rows) != 1 or rows[0].get("task_id") != task_id
            or rows[0].get("patient_fold") != str(manifest["fold"])):
        raise RuntimeError("Authorized task manifest is not exactly one fold")
    row = rows[0]
    if row.get("owner") != "paid_gpu":
        raise RuntimeError("C fold owner mismatch")
    launch_receipt = root / "LAUNCH_PREFLIGHT.json"
    if launch_receipt.exists():
        raise FileExistsError("GPU launch receipt already exists; refusing duplicate training")
    launch_receipt.write_text(json.dumps({
        "status": "PASS_GPU_LAUNCH_PREFLIGHT", "preparation_host": "149",
        "gpu_host": socket.gethostname(), "gpu_instance_id": args.instance_id,
        "billing_mode": args.billing_mode, "task_id": task_id,
        "fold": manifest["fold"],
        "transfer_manifest_sha256": args.manifest_sha256,
        "integrity_policy": (
            "USER_DIRECTED_REUSE_VERIFIED_RECEIPTS_SIZE_ONLY"
            if no_rehash else "SHA256"
        ),
        "source_prepared_sha256": ready["source_prepared_sha256"],
        "c_graph_overlay_sha256": ready["c_graph_overlay_sha256"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        sys.executable, str(code / "scripts" / "v32_pipeline.py"), "run-shard",
        "--allow-training", "--repo-root", str(code),
        "--config", str(auth / "config.yaml"),
        "--input-manifest", str(auth / "INPUT_MANIFEST.json"),
        "--task-manifest", str(auth / "TASK_MANIFEST.tsv"),
        "--approval", str(auth / "TRAINING_APPROVAL.json"),
        "--run-id", str(row["run_id"]), "--task-id", task_id,
        "--endpoint-id", str(row["owner"]),
        "--hardware-class", str(row["hardware_class"]),
        "--trainer", "cc_hhgt.v32.training:run_authorized_task",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code) + os.pathsep + environment.get("PYTHONPATH", "")
    if no_rehash:
        environment["CANCERLNCATLAS_REUSE_VERIFIED_INPUTS_NO_REHASH"] = "1"
    stop = threading.Event()
    thread = threading.Thread(target=telemetry, args=(stop,), daemon=True)
    thread.start()
    try:
        result = subprocess.run(command, env=environment)
    finally:
        stop.set()
        thread.join(timeout=20)
    if result.returncode:
        raise RuntimeError(f"Guarded C trainer exited with code {result.returncode}")
    emit("TRAINING_SUCCEEDED", instance_id=args.instance_id, task_id=task_id)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256")
    p.add_argument("--instance-id", required=True)
    p.add_argument("--billing-mode", choices=("Spot", "Postpay"), required=True)
    p.add_argument("--staged-no-rehash-receipt", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:
        emit("LAUNCH_FAILED", instance_id=args.instance_id, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
