#!/usr/bin/env python3
"""Exercise the actual five-fold ATAC input/features/targets without fitting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
from cc_hhgt.v32.cnv_input_sources import require_host
from cc_hhgt.v32.formal_expression_sources import write
from cc_hhgt.v32.atac_training import AtacTrainingConfig, prepare_atac_inputs, collect_atac_examples, file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "patient-folds", "pathway-membership", "materialization-success", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    defaults = AtacTrainingConfig()
    controls = ("seed", "min_pathway_promoter_genes", "min_inner_train_patients", "min_outer_train_patients", "max_training_rows")
    for name in controls: parser.add_argument("--"+name.replace("_", "-"), type=int, default=getattr(defaults, name))
    args = parser.parse_args(argv); require_host()
    if args.output.exists(): raise FileExistsError("Use a new ATAC preflight output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = {"target_host": "149", "training_started": False, "status": "CHECKING_INPUTS"}
    write(args.output, base)
    try:
        config = AtacTrainingConfig(**{name: getattr(args, name) for name in controls})
        prepared = prepare_atac_inputs(candidates_path=args.candidates, patient_folds_path=args.patient_folds,
            pathway_membership_path=args.pathway_membership, materialization_success_path=args.materialization_success, config=config)
        examples = collect_atac_examples(prepared); rows = []
        for fold, (x, score, audit) in examples.items():
            classes, counts = np.unique(score > np.median(score), return_counts=True)
            if len(x) < 100 or len(classes) != 2:
                raise ValueError(f"ATAC fold {fold} requires >=100 sampled rows and both persistence classes")
            rows.append({"fold": fold, "feature_rows": len(x), "features": x.shape[1],
                "persistence_classes": classes.tolist(), "class_counts": counts.tolist(), "sample_audit": audit})
        for name, path in prepared["paths"].items():
            if file_sha256(path) != prepared["input_hashes"][name]: raise ValueError("ATAC preflight input changed")
        write(args.output, {**base, "status": "PASS_READY", "format": "ATAC_NO_FIT_PREFLIGHT_V1",
            "cohort": prepared["cohort"], "cancers": list(prepared["cancers"]), "candidate_rows": prepared["total_rows"],
            "input_hashes": prepared["input_hashes"], "folds": rows,
            "sampling_config": {name: getattr(config, name) for name in controls}})
    except Exception as error:
        write(args.output, {**base, "status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__": raise SystemExit(main())
