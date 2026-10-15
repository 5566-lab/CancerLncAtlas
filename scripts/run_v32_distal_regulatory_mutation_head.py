#!/usr/bin/env python3
"""Train and publish the two-stage V3.2 distal-regulatory mutation head."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.distal_topology import require_shared_predictive_topology

from cc_hhgt.v32.distal_regulatory_mutation_head import (
    DistalRegulatoryMutationHeadConfig,
    crossfit_distal_regulatory_pathway_head,
)
from cc_hhgt.v32.epigenetic_expression_head import (
    EpigeneticExpressionConfig,
    nested_outer_crossfit_epigenetic_expression,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-input-root", type=Path, required=True)
    parser.add_argument("--pathway-activity-root", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-alpha-expression", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-pathway", type=float, default=1.0)
    parser.add_argument("--min-expression-train-patients", type=int, default=20)
    parser.add_argument("--min-pathway-train-patients", type=int, default=20)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149":
        raise RuntimeError("Distal head execution must run on target host 149")

    head_input = args.head_input_root.resolve()
    activity_root = args.pathway_activity_root.resolve()
    candidate_path = args.candidate_universe.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Distal head output reuse refused: {output}")
    if not (head_input / "AUDIT.json").is_file():
        raise FileNotFoundError(head_input / "AUDIT.json")
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    input_audit = json.loads((head_input / "AUDIT.json").read_text(encoding="utf-8"))
    if (input_audit.get("format")!="CANCERLNCATLAS_V32_DISTAL_REGULATORY_HEAD_INPUT_V1"
            or input_audit.get("status")!="PASS" or input_audit.get("target_host")!="149"):
        raise ValueError("Distal head requires completed input preparation on host 149")
    lineage = require_shared_predictive_topology(input_audit.get("topology_lineage", {}))
    input_hashes = {Path(item["path"]).parent.name: item["sha256"] for item in input_audit.get("outputs", [])}
    if len(input_hashes)!=len(input_audit.get("outputs",[])):raise ValueError("Duplicate distal input partitions")
    source_record=lambda path:{"path":str(path.resolve()),"sha256":sha256(path)}
    sources={"head_input_audit":source_record(head_input/"AUDIT.json"),"candidates":source_record(candidate_path)}
    partition_sources=[]
    expression_config=EpigeneticExpressionConfig(signal_features=("distal_mutation_burden",),
        nuisance_features=("atac_distal_accessibility","promoter_methylation_beta","distal_methylation_beta","local_cnv_log2","tumor_purity"),
        ridge_alpha=args.ridge_alpha_expression,min_train_patients=args.min_expression_train_patients)
    pathway_config=DistalRegulatoryMutationHeadConfig(ridge_alpha=args.ridge_alpha_pathway,min_train_patients=args.min_pathway_train_patients)
    output.mkdir(parents=True)
    (output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps({
        "target_host": "149", "observed_host": socket.gethostname(),
        "stage": "DISTAL_NESTED_OOF", "topology_lineage": lineage}, indent=2), encoding="utf-8")
    component_root = output / "patient_oof_expression_component"
    nested_component_root = output / "nested_outer_expression_components"
    pathway_root = output / "candidate_oof_pathway_head"
    component_root.mkdir()
    nested_component_root.mkdir()
    pathway_root.mkdir()

    candidates = pd.read_parquet(candidate_path)
    keys=["cancer_id","lncrna_id","pathway_id"]
    if (candidates.empty or not set(keys).issubset(candidates) or candidates[keys].isna().any().any()
            or candidates.duplicated(keys).any() or not candidates.cancer_id.astype(str).str.fullmatch(r"[A-Z0-9_]+").all()):
        raise ValueError("Distal candidate universe needs unique complete canonical keys")
    cancers = sorted(candidates.cancer_id.astype(str).str.upper().unique())
    cancer_audits = []
    artifacts = []
    for cancer in cancers:
        input_path = head_input / f"cancer_id={cancer}" / "part-0.parquet"
        activity_path = activity_root / f"cancer_id={cancer}" / "part-0.parquet"
        if not input_path.is_file() or not activity_path.is_file():
            raise FileNotFoundError(f"Missing formal input for {cancer}")
        if input_hashes.get(input_path.parent.name) != sha256(input_path):
            raise ValueError(f"Assembled distal input differs from its bound audit: {cancer}")
        partition_sources.append({"cancer_id":cancer,"head_input":source_record(input_path),"activity":source_record(activity_path)})
        frame = pd.read_parquet(input_path)
        component, nested_component, expression_audit = (
            nested_outer_crossfit_epigenetic_expression(
            frame,
            config=expression_config,
            )
        )
        rename_columns = {
            "regulatory_expression_component_z": "mutation_regulatory_expression_component_z",
            "epigenetic_expression_available": "mutation_regulatory_expression_component_available",
            "epigenetic_expression_unavailable_reason": "mutation_regulatory_expression_component_unavailable_reason",
        }
        component = component.rename(columns=rename_columns)
        nested_component = nested_component.rename(
            columns=rename_columns
        )
        component_dir = component_root / f"cancer_id={cancer}"
        component_dir.mkdir()
        component_path = component_dir / "part-0.parquet"
        atomic_parquet(component, component_path)
        nested_component_dir = nested_component_root / f"cancer_id={cancer}"
        nested_component_dir.mkdir()
        nested_component_path = nested_component_dir / "part-0.parquet"
        atomic_parquet(nested_component, nested_component_path)

        activity = pd.read_parquet(
            activity_path,
            columns=["cancer_id", "patient_id", "pathway_id", "activity_score"],
        )
        local_candidates = candidates.loc[
            candidates.cancer_id.astype(str).str.upper().eq(cancer),
            ["cancer_id", "lncrna_id", "pathway_id"],
        ].copy()
        head, pathway_audit = crossfit_distal_regulatory_pathway_head(
            nested_component,
            activity,
            local_candidates,
            config=pathway_config,
        )
        head_dir = pathway_root / f"cancer_id={cancer}"
        head_dir.mkdir()
        head_path = head_dir / "part-0.parquet"
        atomic_parquet(head, head_path)
        artifacts.extend(
            [
                {"path": str(component_path), "bytes": component_path.stat().st_size, "sha256": sha256(component_path)},
                {"path": str(nested_component_path), "bytes": nested_component_path.stat().st_size, "sha256": sha256(nested_component_path)},
                {"path": str(head_path), "bytes": head_path.stat().st_size, "sha256": sha256(head_path)},
            ]
        )
        cancer_audits.append(
            {
                "cancer_id": cancer,
                "expression_status": expression_audit["status"],
                "expression_rows": expression_audit["rows"],
                "expression_available_rows": expression_audit["own_fold_audit"]["available_rows"],
                "nested_expression_rows": expression_audit["nested_rows"],
                "outer_heldout_expression_used_for_any_component_fit": expression_audit[
                    "outer_heldout_expression_used_for_any_component_fit"
                ],
                "pathway_status": pathway_audit["status"],
                "candidate_rows": pathway_audit["candidate_rows"],
                "candidate_available_rows": pathway_audit["available_candidate_rows"],
            }
        )

    for item in [*sources.values(),*(r[key] for r in partition_sources for key in ("head_input","activity"))]:
        if sha256(Path(item["path"]))!=item["sha256"]:raise ValueError("Distal input changed during execution")
    from cc_hhgt.v32 import epigenetic_expression_head,distal_regulatory_mutation_head
    audit = {
        "format": "CANCERLNCATLAS_V32_DISTAL_REGULATORY_MUTATION_HEAD_NESTED_OOF_V2",
        "status": "PASS",
        "target_host": "149",
        "topology_lineage": lineage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_v32_training": True,
        "old_model_prediction_or_score_reused": False,
        "patient_folds": 5,
        "mutation_feature": "distal_mutation_burden",
        "expression_component": "FULL_MUTATION_PLUS_COVARIATES_MINUS_COVARIATES_ONLY",
        "pathway_target": "formal activity_score",
        "heldout_lncrna_expression_used_for_own_fit": False,
        "outer_heldout_lncrna_expression_used_for_any_training_component_fit": False,
        "outer_training_expression_components_are_inner_oof": True,
        "heldout_pathway_activity_used_for_own_fit": False,
        "missing_assay_assumed_zero": False,
        "causal_claimed": False,
        "head_input_audit_sha256": sha256(head_input / "AUDIT.json"),
        "candidate_universe_sha256": sha256(candidate_path),
        "sources":sources,"partition_sources":partition_sources,
        "expression_config":asdict(expression_config),"pathway_config":asdict(pathway_config),
        "implementation_sha256":{"scripts/run_v32_distal_regulatory_mutation_head.py":sha256(Path(__file__)),
            "cc_hhgt/v32/epigenetic_expression_head.py":sha256(Path(epigenetic_expression_head.__file__)),
            "cc_hhgt/v32/distal_regulatory_mutation_head.py":sha256(Path(distal_regulatory_mutation_head.__file__))},
        "statistical_analysis_executed":True,"neural_optimizer_started":False,
        "cancer_audits": cancer_audits,
        "artifacts": artifacts,
    }
    audit_path = output / "AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success = {
        "status": "PASS",
        "audit_sha256": sha256(audit_path),
        "success_written_last": True,
    }
    (output / "SUCCESS.json").write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(success, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
