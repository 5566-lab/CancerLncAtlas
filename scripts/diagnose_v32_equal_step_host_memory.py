#!/usr/bin/env python3
"""Read-only stage-by-stage host-memory diagnosis for one prepared fold."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
from pathlib import Path

import torch

from cc_hhgt.v32 import training
from cc_hhgt.v32.equal_step_candidate_pilot import (
    _training_identity,
    _validation_identity,
)
from cc_hhgt.v32.group_shared_encoder_oracle import _runtime_value_sha256
from cc_hhgt.v32.gpu_backward_probe import _build_formal_model


def memory(stage: str) -> None:
    status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(":")
        if sep and key in {"VmRSS", "VmHWM", "VmSize", "VmData"}:
            status[key] = int(value.strip().split()[0]) * 1024
    print(
        json.dumps(
            {
                "stage": stage,
                "memory": status,
                "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * 1024,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Linux is required")
    memory("before_load")
    payload = torch.load(
        args.prepared.resolve(strict=True),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    memory("after_mmap_load")
    training._validate_prepared(
        payload,
        fold=args.fold,
        artifact_hashes=payload.get("artifact_hashes", {}),
    )
    memory("after_validate")
    validation = _validation_identity(
        payload, runtime_value_sha256=_runtime_value_sha256
    )
    memory("after_validation_identity")
    validation.pop("proxy_label", None)
    del validation
    gc.collect()
    memory("after_validation_identity_release")
    training_identity = _training_identity(
        payload, runtime_value_sha256=_runtime_value_sha256
    )
    memory("after_training_identity")
    config = training._load_mapping(args.config.resolve(strict=True))
    model, architecture_id = _build_formal_model(
        training, payload, config, device="cpu"
    )
    memory("after_model_build_cpu")
    chunks, _weights = training._runtime_chunk_contract(payload["bundle"])
    memory("after_runtime_chunk_contract")
    graph = training._runtime_graph_for_chunk(
        payload["bundle"], int(chunks[0]), device="cpu"
    )
    memory("after_first_runtime_graph_cpu")
    del graph, model
    gc.collect()
    memory("after_model_graph_release")
    print(
        json.dumps(
            {
                "status": "PASS_DIAGNOSTIC",
                "training_rows": training_identity["candidate_rows"],
                "architecture_id": architecture_id,
                "training_identity_sha256": training_identity[
                    "ordered_identity_sha256"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
