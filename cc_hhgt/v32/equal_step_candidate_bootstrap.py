"""No-Torch assembler for the paid equal-step candidate comparison.

This module is intentionally tiny and side-effect free until all three
component approvals have passed the generic training guard.  It must stay
free of a top-level import of the pilot because that module is part of the
untrusted staged overlay and the external static verifier inspects this file
with ``ast`` before a GPU is started.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PILOT_ID = "v32-g012-equal-step-candidate-paid-gpu-20260901-r1"
TRAINER = (
    "cc_hhgt.v32.equal_step_candidate_pilot:"
    "run_authorized_equal_step_candidate_pilot"
)
VARIANTS = ("G0", "G1", "G2")
SEED = 20260726
FOLD = 0
ENDPOINT_ID = "paid_gpu"
HARDWARE_CLASS = "PAID_PREEMPTIBLE_GPU"


def _paths(authorization_root: Path, variant: str) -> tuple[Path, Path, Path]:
    key = variant.lower()
    return (
        authorization_root / key / "config.json",
        authorization_root / key / "INPUT_MANIFEST.json",
        authorization_root / key / "TASK_MANIFEST.tsv",
    )


def assemble_guarded_contexts(
    *, repo_root: Path, authorization_root: Path
):
    """Guard G0/G1/G2 separately and only then assemble one composite."""

    if "torch" in sys.modules:
        raise RuntimeError("EQUAL_STEP_BOOTSTRAP_TORCH_IMPORTED_BEFORE_GUARDS")
    from .training_guard import guard_training_entry

    contexts = []
    for variant in VARIANTS:
        config, input_manifest, task_manifest = _paths(
            authorization_root, variant
        )
        # The production trainer binds the graph variant to the conventional
        # v32-g012-{variant}- prefix.  Keep the candidate suffix after that
        # binding so the candidate gate and the training guard agree.
        run_id = (
            f"v32-g012-{variant.lower()}-equal-step-candidate-paid-gpu-"
            "20260901-r1"
        )
        task_id = f"{run_id}|PATIENT_FOLD_{FOLD}|CC-HHGT|{SEED}"
        approval = authorization_root / variant.lower() / "TRAINER_APPROVAL.json"
        contexts.append(
            guard_training_entry(
                allow_training=True,
                repo_root=repo_root,
                config_path=config,
                input_manifest_path=input_manifest,
                task_manifest_path=task_manifest,
                approval_path=approval,
                run_id=run_id,
                task_id=task_id,
                endpoint_id=ENDPOINT_ID,
                hardware_class=HARDWARE_CLASS,
                trainer_specification=TRAINER,
            )
        )
        if "torch" in sys.modules:
            raise RuntimeError(
                f"EQUAL_STEP_BOOTSTRAP_TORCH_IMPORTED_DURING_{variant}_GUARD"
            )
    from .equal_step_candidate_pilot import build_equal_step_pilot_authorization

    composite = build_equal_step_pilot_authorization(
        contexts, pilot_id=PILOT_ID
    )
    if "torch" in sys.modules:
        raise RuntimeError("EQUAL_STEP_BOOTSTRAP_TORCH_IMPORTED_DURING_ASSEMBLY")
    return composite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, required=True)
    args = parser.parse_args(argv)
    composite = assemble_guarded_contexts(
        repo_root=args.repo_root.resolve(),
        authorization_root=args.authorization_root.resolve(),
    )
    # Import the CUDA-capable callable only after the exact three-way
    # composite exists.  The callable re-runs each guard before importing
    # Torch, providing a second independent pre-CUDA verification boundary.
    from .equal_step_candidate_pilot import (
        run_authorized_equal_step_candidate_pilot,
    )

    return int(run_authorized_equal_step_candidate_pilot(composite))


if __name__ == "__main__":  # pragma: no cover - exercised by server launcher
    raise SystemExit(main())
