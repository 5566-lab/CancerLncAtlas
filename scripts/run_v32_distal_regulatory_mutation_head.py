#!/usr/bin/env python3
"""Train and publish the two-stage V3.2 distal-regulatory mutation head."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-input-root", type=Path, required=True)
    parser.add_argument("--pathway-activity-root", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-alpha-expression", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-pathway", type=float, default=1.0)
    parser.add_argument("--min-expression-train-patients", type=int, default=20)
    parser.add_argument("--min-pathway-train-patients", type=int, default=20)
    args = parser.parse_args()

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
    output.mkdir(parents=True)
    component_root = output / "patient_oof_expression_component"
    nested_component_root = output / "nested_outer_expression_components"
    pathway_root = output / "candidate_oof_pathway_head"
    component_root.mkdir()
    nested_component_root.mkdir()
    pathway_root.mkdir()

    candidates = pd.read_parquet(candidate_path)
    cancers = sorted(candidates.cancer_id.astype(str).str.upper().unique())
    cancer_audits = []
    artifacts = []
    for cancer in cancers:
        input_path = head_input / f"cancer_id={cancer}" / "part-0.parquet"
        activity_path = activity_root / f"cancer_id={cancer}" / "part-0.parquet"
        if not input_path.is_file() or not activity_path.is_file():
            raise FileNotFoundError(f"Missing formal input for {cancer}")
        frame = pd.read_parquet(input_path)
        component, nested_component, expression_audit = (
            nested_outer_crossfit_epigenetic_expression(
            frame,
            config=EpigeneticExpressionConfig(
                signal_features=("distal_mutation_burden",),
                nuisance_features=(
                    "atac_distal_accessibility",
                    "promoter_methylation_beta",
                    "distal_methylation_beta",
                    "local_cnv_log2",
                    "tumor_purity",
                ),
                ridge_alpha=args.ridge_alpha_expression,
                min_train_patients=args.min_expression_train_patients,
            ),
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
            config=DistalRegulatoryMutationHeadConfig(
                ridge_alpha=args.ridge_alpha_pathway,
                min_train_patients=args.min_pathway_train_patients,
            ),
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

    audit = {
        "format": "CANCERLNCATLAS_V32_DISTAL_REGULATORY_MUTATION_HEAD_NESTED_OOF_V2",
        "status": "PASS",
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
