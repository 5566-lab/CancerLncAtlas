#!/usr/bin/env python3
"""Exercise actual CNV-head input/feature/label preparation on 149 without fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cc_hhgt.v32.cnv_input_sources import require_host
from cc_hhgt.v32.cnv_only_training_v2 import prepare_directional_inputs, collect_directional_examples
from cc_hhgt.v32.genomic_training import GenomicTrainingConfig, load_fold_core_embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "patient-folds", "directional-root", "association-root", "core-manifest", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    defaults = GenomicTrainingConfig()
    for name in ("seed", "min_pair_callable", "max_train_rows", "max_validation_rows"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(defaults, name))
    args = parser.parse_args(argv)
    require_host()
    if args.output.exists(): raise FileExistsError("Use a new CNV-head preflight output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"target_host": "149", "status": "CHECKING_INPUTS", "training_started": False})+"\n")
    try:
        return prepare(args)
    except Exception as error:
        args.output.write_text(json.dumps({"target_host": "149", "status": "FAILED", "training_started": False,
            "error_type": type(error).__name__, "error": str(error)})+"\n")
        raise


def prepare(args):
    sampling = {name: getattr(args, name) for name in ("seed", "min_pair_callable", "max_train_rows", "max_validation_rows")}
    settings = GenomicTrainingConfig(**sampling); settings.validate()
    prepared = prepare_directional_inputs(candidates_path=args.candidates, folds_path=args.patient_folds,
        directional_root=args.directional_root, association_root=args.association_root, core_manifest_path=args.core_manifest)
    rows = []
    for fold in range(5):
        core = load_fold_core_embeddings(args.core_manifest, prepared["manifest"], fold)
        values = collect_directional_examples(prepared, core, fold, settings)
        rows.append({"fold_id": fold, "train_rows": len(values[2]), "validation_rows": len(values[5]),
            "train_classes": np.unique(values[2]).tolist(), "validation_classes": np.unique(values[5]).tolist(),
            "core_parameter_sha256": core.core_parameter_sha256, "checkpoint_sha256": core.checkpoint_sha256,
            "input_hashes": dict(core.input_hashes)})
    result = {"format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_PREFLIGHT_V1", "status": "PASS_READY",
        "target_host": "149", "training_started": False, "candidate_rows": len(prepared["candidates"]),
        "cancers": list(prepared["cancers"]), "cohort": prepared["cohort"],
        "association_run_id": prepared["authority"]["run_id"], "patient_folds": 5,
        "sampling_config": sampling,
        "core_folds": rows, "input_hashes": prepared["input_hashes"],
        "consumed_inputs": list(prepared["capture"].records.values()),
        "fresh_head_required": True, "old_checkpoint_allowed": False,
        "mutation_features_allowed": False, "sealed_test_read": False}
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "core_folds": 5, "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
