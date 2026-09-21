#!/usr/bin/env python3
"""Read-only preflight for the directional CNV head before formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.genomic_training import _validate_core_manifest, load_fold_core_embeddings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--patient-folds", required=True, type=Path)
    parser.add_argument("--directional-root", required=True, type=Path)
    parser.add_argument("--association-root", required=True, type=Path)
    parser.add_argument("--core-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = pd.read_parquet(args.candidates, columns=["cancer_id", "lncrna_id", "pathway_id"])
    folds = pd.read_csv(args.patient_folds, sep="\t")
    if len(candidates) != 3_300_000 or candidates[["cancer_id", "lncrna_id", "pathway_id"]].duplicated().any():
        raise RuntimeError("candidate closure drift")
    if folds[["cancer_id", "patient_id"]].duplicated().any() or set(pd.to_numeric(folds.patient_fold_id)) != set(range(5)):
        raise RuntimeError("patient fold closure drift")
    directional = json.loads((args.directional_root / "SUCCESS.json").read_text())
    if directional.get("status") != "SUCCESS" or int(directional.get("cancer_count", -1)) != 33:
        raise RuntimeError("directional CNV closure drift")
    manifest, manifest_sha, composite = _validate_core_manifest(args.core_manifest)
    fold_rows = []
    for fold in range(5):
        core = load_fold_core_embeddings(args.core_manifest, manifest, fold)
        fold_rows.append({
            "fold_id": fold, "lncrna_rows": len(core.lncrna.ids),
            "pathway_rows": len(core.pathway.ids), "embedding_width": core.lncrna.values.shape[1],
            "core_parameter_sha256": core.core_parameter_sha256,
            "checkpoint_sha256": core.checkpoint_sha256, "input_hashes": dict(core.input_hashes),
        })
    association_ready = (args.association_root / "SUCCESS.json").is_file()
    payload = {
        "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_PREFLIGHT_V1",
        "status": "PASS_READY" if association_ready else "PASS_STATIC_INPUTS_WAITING_LOCAL_CNV_AUDIT",
        "candidate_rows": len(candidates), "cancers": candidates.cancer_id.nunique(),
        "patient_rows": len(folds), "patient_folds": 5,
        "candidate_sha256": sha256(args.candidates), "patient_fold_sha256": sha256(args.patient_folds),
        "directional_success_sha256": sha256(args.directional_root / "SUCCESS.json"),
        "core_manifest_sha256": manifest_sha, "core_parameter_composite_sha256": composite,
        "core_folds": fold_rows, "association_audit_ready": association_ready,
        "fresh_head_required": True, "old_checkpoint_allowed": False,
        "mutation_features_allowed": False, "router_used": False, "sealed_test_read": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "core_folds": 5}, sort_keys=True))


if __name__ == "__main__":
    main()
