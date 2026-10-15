#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_batched_lasso import (  # noqa: E402
    BatchedLassoPolicy,
    fit_batched_lasso,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if socket.gethostname() != "149" or preflight.get("target_host") != "149":
        raise SystemExit("WRONG_HOST")
    if preflight.get("cpu_only") is not True or preflight.get("paid_compute") is not False:
        raise SystemExit("BENCHMARK_PREFLIGHT_INVALID")

    rng = np.random.default_rng(20261005)
    sample_count, feature_count, response_count = 300, 1000, 1000
    donors = np.repeat([f"donor_{index:02d}" for index in range(12)], 25)
    coding = rng.lognormal(mean=0.0, sigma=0.7, size=(sample_count, feature_count))
    truth = np.zeros((feature_count, response_count), dtype=np.float64)
    for response_index in range(response_count):
        genes = rng.choice(feature_count, size=3, replace=False)
        truth[genes, response_index] = rng.normal(0, 0.6, size=3)
    responses = coding @ truth + rng.normal(
        scale=0.35, size=(sample_count, response_count)
    )
    policy = BatchedLassoPolicy(
        alphas=(0.3, 0.1, 0.03, 0.01),
        max_genes=feature_count,
        min_donors=4,
        min_detection=0.01,
        inner_folds=3,
        bootstraps=4,
        min_bootstrap_successes=2,
        max_iter=1000,
        tolerance=1e-5,
        kkt_check_interval=10,
        response_batch_size=250,
        seed=20261005,
    )
    started = time.perf_counter()
    result = fit_batched_lasso(
        coding,
        responses,
        donors,
        source_donors=np.unique(donors),
        excluded_donors=(),
        policy=policy,
    )
    elapsed = time.perf_counter() - started
    module_path = ROOT / "cc_hhgt" / "v32" / "single_cell_batched_lasso.py"
    receipt = {
        "format": "CANCERLNCATLAS_V32_SC_BATCHED_LASSO_BENCHMARK_V1",
        "state": "PASS",
        "target_host": "149",
        "cpu_only": True,
        "paid_compute": False,
        "shape": {"observations": sample_count, "genes": feature_count, "responses": response_count},
        "policy": {
            "alphas": list(policy.alphas),
            "inner_folds": policy.inner_folds,
            "bootstraps": policy.bootstraps,
            "max_iter": policy.max_iter,
            "tolerance": policy.tolerance,
            "response_batch_size": policy.response_batch_size,
            "top_k_per_response": policy.top_k_per_response,
        },
        "elapsed_seconds": elapsed,
        "maximum_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "available_responses": int(result.available.sum()),
        "unavailable_responses": int(result.unavailable.sum()),
        "signed_edges": int(len(result.coefficient)),
        "positive_edges": int((result.coefficient > 0).sum()),
        "negative_edges": int((result.coefficient < 0).sum()),
        "module_sha256": sha256(module_path),
        "numpy_version": np.__version__,
        "pid": os.getpid(),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

