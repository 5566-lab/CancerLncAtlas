#!/usr/bin/env python3
"""Fail-closed CUDA preflight for the V3.1 formal multi-GPU run."""
from __future__ import annotations

import json

import torch


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    rows = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        capability = torch.cuda.get_device_capability(device)
        left = torch.randn((1024, 1024), device=device)
        right = torch.randn((1024, 1024), device=device)
        result = left @ right
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all().item():
            raise RuntimeError(f"Non-finite CUDA result on {device}")
        rows.append(
            {
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(device).total_memory
                ),
                "matmul_checksum": float(result.float().mean().cpu()),
            }
        )
    payload = {
        "status": "PASS",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
