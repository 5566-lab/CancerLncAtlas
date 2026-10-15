#!/usr/bin/env python3
"""Materialize actual drug pharmacology and fold-local design features; no fitting."""
import argparse
import json
import socket
from pathlib import Path

import pandas as pd

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.current_core_export import sha256, verified_path
from cc_hhgt.v32.drug_pharmacology import load_pharmacology, FEATURES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("Drug pharmacology preparation requires host 149")
    manifest_path = args.manifest.resolve(); digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "DRUG_PHARMACOLOGY_PREPARATION_V1": raise ValueError("Unknown pharmacology preparation manifest")
    paths = {role:verified_path(manifest[role], manifest_path.parent) for role in
             ("pharmacology_manifest", "drug_gene_target", "raw_drug_response", "cell_line_map")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149", "training_started":False, "manifest_sha256":digest}))
    targets = base.normalise_drug_targets(base._read_table(paths["drug_gene_target"]))
    response = base.normalise_raw_response(base._read_table(paths["raw_drug_response"]))
    mapping = base.assign_cell_line_folds(base.normalise_cell_line_map(base._read_table(paths["cell_line_map"])),
                                        int(manifest.get("seed", 20260726)))
    inputs = load_pharmacology(paths["pharmacology_manifest"], targets=targets, response=response, mapping=mapping)
    contexts = response.merge(mapping[["dataset_id", "cell_line_id", "cancer_id"]],
        on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_one")[["cancer_id", "drug_id"]].drop_duplicates()
    parts = []
    for fold in range(5):
        for split in ("train", "validation", "test"):
            models = base._split_models(mapping, fold, split)
            context = contexts.merge(inputs.design_table(models), on=["cancer_id", "drug_id"], how="left", validate="one_to_one")
            for column in ("dose_available", "positive_dose_available", "exposure_available"):
                context[column] = context[column].eq(True)
            context["protocol_metadata_available"] = context.log1p_protocol_count.notna()
            parts.append(context.assign(cell_line_fold=fold, split=split,
                context_semantics="OBSERVED_PROTOCOL_DISTRIBUTION_NOT_DOSE_SPECIFIC_RESPONSE"))
    tables = {"drug_static_features":inputs._static, "target_annotation_measurements":inputs.targets,
              "assay_design_fold_context":pd.concat(parts, ignore_index=True)}
    outputs = {}
    for name, frame in tables.items():
        path = args.output/(name+".parquet"); frame.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {"path":str(path.resolve()), "sha256":sha256(path), "rows":len(frame)}
    receipt = inputs.receipt(); inputs.verify_sources()
    if sha256(manifest_path) != digest: raise ValueError("Pharmacology preparation manifest changed")
    for role,path in paths.items():
        if sha256(path) != manifest[role]["sha256"]: raise ValueError("Pharmacology preparation source changed")
    audit = {"format":"DRUG_PHARMACOLOGY_PREPARATION_AUDIT_V1", "status":"PASS", "target_host":"149",
        "training_started":False, "input_manifest":{"path":str(manifest_path), "sha256":digest},
        "outputs":outputs, "feature_receipt":receipt, "additional_domain_features":len(FEATURES),
        "response_values_used_as_features":False, "independent_validation_claimed":False,
        "drug_training_entry":"scripts/run_v32_drug_training.py --pharmacology-manifest "+str(paths["pharmacology_manifest"])}
    path = args.output/"AUDIT.json"; path.write_text(json.dumps(audit, indent=2)+"\n")
    (args.output/"SUCCESS.json").write_text(json.dumps({"status":"PREPARED_NOT_TRAINED", "audit_sha256":sha256(path)}))
    print(json.dumps({"status":"PREPARED_NOT_TRAINED", "outputs":outputs, "additional_domain_features":len(FEATURES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
