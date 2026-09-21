#!/usr/bin/env python3
"""Measure Linux mmap loading of one hash-authorized prepared fold.

This is a CPU-only readiness probe.  It deliberately uses the production
handle-bound loader so a successful result covers both SHA-256 authorization
and lazy Torch storage mapping through ``/proc/self/fd``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from cc_hhgt.v32 import training


def _proc_memory_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, separator, remainder = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            amount, unit = remainder.strip().split()[:2]
            if unit != "kB":
                raise RuntimeError(f"unexpected /proc memory unit for {key}: {unit}")
            values[key] = int(amount) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise RuntimeError("/proc/self/status did not expose VmRSS and VmHWM")
    return values


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for child in value.values():
            observed = _first_tensor(child)
            if observed is not None:
                return observed
    elif isinstance(value, (list, tuple)):
        for child in value:
            observed = _first_tensor(child)
            if observed is not None:
                return observed
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        raise RuntimeError("probe requires Linux /proc/self/fd")
    os.environ["V32_PREPARED_ARTIFACT_MMAP"] = "1"
    prepared = args.prepared.resolve(strict=True)
    before = _proc_memory_bytes()
    started = time.monotonic()
    payload, digest = training.load_prepared_artifact_from_authorized_handle(
        torch,
        args.input_manifest.resolve(strict=True),
        fold=args.fold,
        prepared_path=prepared,
    )
    registered_hashes = payload.get("artifact_hashes", {})
    if not isinstance(registered_hashes, dict):
        raise RuntimeError("prepared artifact_hashes is not a mapping")
    training._validate_prepared(
        payload,
        fold=args.fold,
        artifact_hashes=registered_hashes,
    )
    elapsed = time.monotonic() - started
    tensor = _first_tensor(payload)
    if tensor is None or tensor.numel() == 0:
        raise RuntimeError("prepared payload contains no non-empty tensor")
    sample_value = tensor.reshape(-1)[0].item()
    after = _proc_memory_bytes()
    prepared_bytes = prepared.stat().st_size
    peak_growth = max(0, after["VmHWM"] - before["VmHWM"])
    # A lazy mmap load must not duplicate most of the multi-GB file in RSS.
    max_peak_growth = max(1024**3, prepared_bytes // 3)
    if peak_growth >= max_peak_growth:
        raise RuntimeError(
            "prepared mmap load exceeded the lazy-load memory bound: "
            f"growth={peak_growth}, bound={max_peak_growth}"
        )
    result = {
        "format": "CC_HHGT_V3_2_PREPARED_MMAP_LINUX_PROBE_V1",
        "status": "PASS",
        "cpu_only": not torch.cuda.is_available(),
        "fold": args.fold,
        "prepared_path": str(prepared),
        "prepared_bytes": prepared_bytes,
        "prepared_sha256": digest,
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
        "memory_before_bytes": before,
        "memory_after_bytes": after,
        "peak_rss_growth_bytes": peak_growth,
        "max_allowed_peak_growth_bytes": max_peak_growth,
        "sample_tensor_dtype": str(tensor.dtype),
        "sample_tensor_shape": list(tensor.shape),
        "sample_value": sample_value,
        "prepared_validation_pass": True,
        "mmap_environment": os.environ["V32_PREPARED_ARTIFACT_MMAP"],
    }
    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
