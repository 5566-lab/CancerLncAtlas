#!/usr/bin/env python3
"""Run hash-bound target/pathway ablation using five frozen current drug heads.

Manifest DRUG_MODEL_ATTRIBUTION_INPUT_V1 has core_run_id, drug_run_id and
artifacts: core_embedding_manifest, drug_lineage, checkpoint_manifest, config,
exact_candidates, exact_pathway_membership, raw_drug_response,
raw_lncrna_expression, cell_line_map, drug_gene_target, queries. Every artifact
is {path,sha256}. Paths are relative to the manifest unless absolute.
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import pandas as pd
import torch

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.current_core_export import validate_current_export
from cc_hhgt.v32.drug_attribution import ablate_drug_targets
from cc_hhgt.v32.drug_training_streaming import StreamingDrugTrainingConfig
from cc_hhgt.v32.integrated_model import build_private_auxiliary_head
from cc_hhgt.v32.input_lineage import artifact_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-chunk-rows", type=int, default=1000)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("Drug attribution requires target host 149")
    if args.query_chunk_rows < 1: raise ValueError("Query chunk rows must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "DRUG_MODEL_ATTRIBUTION_INPUT_V1": raise ValueError("Unknown attribution manifest")
    paths = {}
    for role, record in manifest["artifacts"].items():
        path = Path(record["path"])
        path = path.resolve() if path.is_absolute() else (args.manifest.resolve().parent/path).resolve()
        if artifact_sha256(path) != record["sha256"]: raise ValueError(f"Attribution input hash mismatch: {role}")
        paths[role] = path
    lineage = json.loads(paths["drug_lineage"].read_text())
    if (lineage.get("training_run_id") != manifest["drug_run_id"] or lineage.get("training_status") != "SUCCESS"
            or lineage.get("checkpoint_manifest_sha256") != artifact_sha256(paths["checkpoint_manifest"])
            or lineage.get("config_sha256") != artifact_sha256(paths["config"])):
        raise ValueError("Selected drug run does not bind the supplied checkpoints/configuration")
    core_manifest, core_sha, _ = validate_current_export(paths["core_embedding_manifest"], expected_run_id=manifest["core_run_id"])
    for role in ["core_embedding_manifest", "exact_candidates", "exact_pathway_membership", "raw_drug_response",
                 "raw_lncrna_expression", "cell_line_map", "drug_gene_target"]:
        if lineage.get("input_hashes_start", {}).get(role) != manifest["artifacts"][role]["sha256"]:
            raise ValueError(f"Attribution input differs from the actual trained drug source: {role}")
    if lineage.get("input_hashes_start") != lineage.get("input_hashes_end"):
        raise ValueError("Drug training inputs changed during the selected run")
    config = StreamingDrugTrainingConfig(**json.loads(paths["config"].read_text())["training"])
    config.validate()
    records = json.loads(paths["checkpoint_manifest"].read_text())["records"]
    if len(records) != 5 or {r["cell_line_fold"] for r in records} != set(range(5)):
        raise ValueError("Attribution requires all five drug checkpoints")
    # Relocation is allowed via a per-fold override, but its checksum must be
    # the checksum in the actual training checkpoint manifest.
    checkpoints = {}
    for record in records:
        fold = record["cell_line_fold"]
        path = Path(manifest.get("checkpoint_paths", {}).get(str(fold), record["path"]))
        if not path.is_absolute(): path = args.manifest.resolve().parent/path
        if artifact_sha256(path) != record["sha256"] or record.get("optimizer_steps", 0) <= 0:
            raise ValueError("Drug checkpoint hash or completed optimizer evidence is invalid")
        checkpoints[fold] = path
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "core_run_id": manifest["core_run_id"], "drug_run_id": manifest["drug_run_id"]}, indent=2))
    response = base.normalise_raw_response(base._read_table(paths["raw_drug_response"]))
    expression = base.normalise_raw_expression(base._read_table(paths["raw_lncrna_expression"]))
    mapping = base.assign_cell_line_folds(base.normalise_cell_line_map(base._read_table(paths["cell_line_map"])), config.seed)
    targets = base.normalise_drug_targets(base._read_table(paths["drug_gene_target"]))
    exact = base._read_table(paths["exact_candidates"])
    membership = base._read_table(paths["exact_pathway_membership"])
    queries = base.normalise_drug_candidates(base._read_table(paths["queries"]), base.normalise_exact_candidates(exact))
    outputs = {}; audits = []
    for fold in range(5):
        checkpoint = torch.load(checkpoints[fold], map_location="cpu", weights_only=False)
        expected_checkpoint_sha = next(record["sha256"] for record in records if record["cell_line_fold"] == fold)
        if artifact_sha256(checkpoints[fold]) != expected_checkpoint_sha:
            raise ValueError("Drug checkpoint changed while loading")
        core = base.load_fold_core_embeddings(paths["core_embedding_manifest"], core_manifest, fold)
        if (checkpoint.get("checkpoint_format") != base.PRIVATE_CHECKPOINT_FORMAT
                or checkpoint.get("cell_line_fold") != fold
                or checkpoint.get("core_parameter_sha256") != core.core_parameter_sha256
                or checkpoint.get("core_checkpoint_sha256") != core.checkpoint_sha256
                or checkpoint.get("domain_features") != list(base._DOMAIN_FEATURES)):
            raise ValueError("Drug head is not bound to the selected fold/core/feature schema")
        head, _ = build_private_auxiliary_head("drug", core_features=core.lncrna.values.shape[1]*3,
            domain_features=len(base._DOMAIN_FEATURES), hidden_features=config.hidden_features,
            dropout=config.dropout, seed=config.seed+fold*101)
        head.load_state_dict(checkpoint["model_state"], strict=True)
        _, counts, availability = base._drug_target_vectors(targets, core)
        for start in range(0, len(queries), args.query_chunk_rows):
            local = queries.iloc[start:start+args.query_chunk_rows].reset_index(drop=True)
            stats = base._statistics_for_split(local, expression, response, mapping, base._split_models(mapping, fold, "test"),
                                               counts, availability, config)
            baseline, attribution, audit = ablate_drug_targets(head, core, local, stats.domain, stats.assay_available,
                targets, exact, membership, domain_mean=checkpoint["domain_mean"], domain_scale=checkpoint["domain_scale"])
            audits.append({"fold": fold, "query_start": start, **audit})
            for label, table in [("baseline", baseline), ("attribution", attribution)]:
                if table.empty: continue
                # Flat file names avoid Hive partition/type clashes.
                path = args.output/f"{label}_fold{fold}_part{start}.parquet"
                table.to_parquet(path, index=False, compression="zstd")
                outputs[path.name] = {"path": str(path.resolve()), "sha256": artifact_sha256(path), "rows": len(table)}
    for role, path in paths.items():
        if artifact_sha256(path) != manifest["artifacts"][role]["sha256"]:
            raise ValueError(f"Attribution source changed during inference: {role}")
    audit = {"format": "DRUG_MODEL_TARGET_ABLATION_V1", "status": "PASS", "target_host": "149", "training_started": False,
        "core_run_id": manifest["core_run_id"], "drug_run_id": manifest["drug_run_id"],
        "input_manifest_sha256": artifact_sha256(args.manifest), "outputs": outputs, "fold_audits": audits,
        "score_estimand": "ASSOCIATION_STRENGTH_NOT_SENSITIVITY_DIRECTION_OR_TREATMENT_BENEFIT",
        "pathway_ablation": "REMOVE_ALL_ANNOTATED_TARGETS_IN_THE_EXACT_PATHWAY_AT_ONCE",
        "attributions_additive": False, "biological_causality_claimed": False}
    (args.output/"AUDIT.json").write_text(json.dumps(audit, indent=2)+"\n")
    (args.output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "audit_sha256": artifact_sha256(args.output/"AUDIT.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
