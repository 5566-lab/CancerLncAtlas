#!/usr/bin/env python3
"""Independent closure audit for the V3.2 distal-regulatory mutation head."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EFFECT_COLUMNS = (
    "mean_oof_slope",
    "oof_rmse",
    "oof_baseline_rmse",
    "oof_delta_mse",
    "oof_r2",
    "oof_prediction_correlation",
)
FEATURES = (
    "distal_mutation_burden",
    "atac_distal_accessibility",
    "promoter_methylation_beta",
    "distal_methylation_beta",
    "local_cnv_log2",
    "tumor_purity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--head-input-root", type=Path, required=True)
    parser.add_argument("--methylation-root", type=Path, required=True)
    parser.add_argument("--distal-mutation-root", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    head_root = args.head_root.resolve()
    input_root = args.head_input_root.resolve()
    methylation_root = args.methylation_root.resolve()
    mutation_root = args.distal_mutation_root.resolve()
    candidate_path = args.candidate_universe.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Closure output reuse refused: {output}")
    output.mkdir(parents=True)

    head_audit_path = head_root / "AUDIT.json"
    head_success_path = head_root / "SUCCESS.json"
    input_audit_path = input_root / "AUDIT.json"
    methylation_audit_path = methylation_root / "AUDIT.json"
    mutation_audit_path = mutation_root / "AUDIT.json"
    head = read_json(head_audit_path)
    success = read_json(head_success_path)
    input_audit = read_json(input_audit_path)
    methylation = read_json(methylation_audit_path)
    mutation = read_json(mutation_audit_path)

    require(head.get("status") == "PASS", "Head audit is not PASS")
    require(success.get("status") == "PASS", "Head success marker is not PASS")
    require(success.get("audit_sha256") == sha256(head_audit_path), "Head audit hash drift")
    require(head.get("fresh_v32_training") is True, "Fresh V3.2 training not proven")
    require(head.get("old_model_prediction_or_score_reused") is False, "Old score reuse detected")
    require(head.get("heldout_lncrna_expression_used_for_own_fit") is False, "Expression leakage flag")
    require(
        head.get(
            "outer_heldout_lncrna_expression_used_for_any_training_component_fit"
        )
        is False,
        "Outer-heldout expression entered a training component fit",
    )
    require(
        head.get("outer_training_expression_components_are_inner_oof") is True,
        "Nested inner-OOF expression components are not proven",
    )
    require(head.get("heldout_pathway_activity_used_for_own_fit") is False, "Pathway leakage flag")
    require(head.get("missing_assay_assumed_zero") is False, "Missing assay zero-fill flag")
    require(head.get("causal_claimed") is False, "Unsupported causal claim")
    require(input_audit.get("status") == "PASS", "Head-input audit is not PASS")
    require(
        methylation.get("status") == "PASS_PATIENT_LNCRNA_METHYLATION_FEATURES",
        "Methylation feature audit is not PASS",
    )
    require(methylation.get("raw_download_delete_ready") is True, "Methylation raw deletion not ready")
    require(mutation.get("status") == "PASS_DISTAL_MUTATION_FEATURES", "Distal mutation audit not PASS")
    require(mutation.get("locus_callability_claimed") is False, "Invalid locus-callability claim")
    require(mutation.get("missing_assay_assumed_zero") is False, "Mutation missingness zero-filled")

    for artifact in head.get("artifacts", []):
        path = Path(str(artifact.get("path", ""))).resolve()
        require(path.is_file(), f"Head artifact missing: {path}")
        require(path.stat().st_size == int(artifact.get("bytes", -1)), f"Head bytes drift: {path}")
        require(sha256(path) == artifact.get("sha256"), f"Head sha256 drift: {path}")

    candidates = pd.read_parquet(candidate_path)
    candidates["cancer_id"] = candidates.cancer_id.astype(str).str.upper()
    expected = candidates.groupby("cancer_id", observed=True).size().to_dict()
    require(len(expected) == 33, "Candidate universe is not full33")
    require(int(sum(expected.values())) == 3_300_000, "Candidate universe is not 3.3M")
    cancer_records = []
    total_available = 0
    for cancer in sorted(expected):
        head_path = head_root / "candidate_oof_pathway_head" / f"cancer_id={cancer}" / "part-0.parquet"
        component_path = head_root / "patient_oof_expression_component" / f"cancer_id={cancer}" / "part-0.parquet"
        nested_component_path = head_root / "nested_outer_expression_components" / f"cancer_id={cancer}" / "part-0.parquet"
        input_path = input_root / f"cancer_id={cancer}" / "part-0.parquet"
        for path in (head_path, component_path, nested_component_path, input_path):
            require(path.is_file(), f"Required cancer artifact missing: {path}")
        score = pd.read_parquet(head_path)
        require(len(score) == int(expected[cancer]), f"Candidate row count mismatch: {cancer}")
        require(
            not score.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any(),
            f"Candidate key duplication: {cancer}",
        )
        available = score["distal_regulatory_mutation_pathway_available"]
        require(not available.isna().any(), f"Head availability null: {cancer}")
        available = available.astype(bool).to_numpy()
        effects = score.loc[:, list(EFFECT_COLUMNS)].to_numpy(float)
        require(np.isfinite(effects[available]).all(), f"Available head effect not finite: {cancer}")
        require(np.isnan(effects[~available]).all(), f"Unavailable head effect is not typed null: {cancer}")
        reasons = score["distal_regulatory_mutation_pathway_unavailable_reason"]
        require(reasons.loc[available].isna().all(), f"Available head row has reason: {cancer}")
        require(reasons.loc[~available].notna().all(), f"Unavailable head row lacks reason: {cancer}")

        component = pd.read_parquet(component_path)
        component_available = component[
            "mutation_regulatory_expression_component_available"
        ].astype(bool).to_numpy()
        component_value = pd.to_numeric(
            component["mutation_regulatory_expression_component_z"], errors="coerce"
        ).to_numpy(float)
        require(
            np.array_equal(np.isfinite(component_value), component_available),
            f"Component typed-null violation: {cancer}",
        )
        require(
            not component.duplicated(["cancer_id", "patient_id", "lncrna_id"]).any(),
            f"Component key duplication: {cancer}",
        )
        nested_component = pd.read_parquet(nested_component_path)
        require(
            len(nested_component) == 5 * len(component),
            f"Nested component row count mismatch: {cancer}",
        )
        require(
            set(nested_component.outer_fold_id.astype(int)) == set(range(5)),
            f"Nested outer folds are not exact 0..4: {cancer}",
        )
        require(
            not nested_component.duplicated(
                ["cancer_id", "patient_id", "lncrna_id", "outer_fold_id"]
            ).any(),
            f"Nested component key duplication: {cancer}",
        )
        nested_available = nested_component[
            "mutation_regulatory_expression_component_available"
        ].astype(bool).to_numpy()
        nested_value = pd.to_numeric(
            nested_component["mutation_regulatory_expression_component_z"],
            errors="coerce",
        ).to_numpy(float)
        require(
            np.array_equal(np.isfinite(nested_value), nested_available),
            f"Nested component typed-null violation: {cancer}",
        )

        assembled = pd.read_parquet(input_path)
        require(
            not assembled.duplicated(["cancer_id", "patient_id", "lncrna_id"]).any(),
            f"Head input key duplication: {cancer}",
        )
        feature_coverage = {}
        for feature in FEATURES:
            flag = assembled[f"{feature}__available"]
            require(not flag.isna().any(), f"Feature availability null: {cancer}/{feature}")
            flag_value = flag.astype(bool).to_numpy()
            value = pd.to_numeric(assembled[feature], errors="coerce").to_numpy(float)
            require(
                np.array_equal(np.isfinite(value), flag_value),
                f"Feature typed-null violation: {cancer}/{feature}",
            )
            feature_coverage[feature] = int(flag_value.sum())
        available_count = int(available.sum())
        total_available += available_count
        cancer_records.append(
            {
                "cancer_id": cancer,
                "candidate_rows": int(len(score)),
                "candidate_available_rows": available_count,
                "component_rows": int(len(component)),
                "component_available_rows": int(component_available.sum()),
                "nested_component_rows": int(len(nested_component)),
                "nested_component_available_rows": int(nested_available.sum()),
                "feature_available_rows": feature_coverage,
            }
        )

    require(total_available > 0, "No distal-regulatory mutation head row is available")
    raw_target = Path(str(methylation.get("raw_download_delete_target", ""))).resolve()
    require(raw_target.is_dir(), "Methylation raw delete target is not an existing directory")
    closure = {
        "format": "CANCERLNCATLAS_V32_DISTAL_REGULATORY_MUTATION_CLOSURE_NESTED_OOF_V2",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "full33": True,
        "candidate_rows": int(sum(expected.values())),
        "candidate_available_rows": total_available,
        "fresh_v32_training": True,
        "old_model_prediction_or_score_reused": False,
        "patient_oof_expression": True,
        "nested_outer_patient_oof_expression": True,
        "outer_heldout_expression_used_for_any_training_component_fit": False,
        "patient_oof_pathway": True,
        "missing_assay_assumed_zero": False,
        "causal_claimed": False,
        "raw_methylation_delete_authorized": True,
        "raw_methylation_delete_target": str(raw_target),
        "retained_audits": {
            "head": {"path": str(head_audit_path), "sha256": sha256(head_audit_path)},
            "head_input": {"path": str(input_audit_path), "sha256": sha256(input_audit_path)},
            "methylation": {"path": str(methylation_audit_path), "sha256": sha256(methylation_audit_path)},
            "distal_mutation": {"path": str(mutation_audit_path), "sha256": sha256(mutation_audit_path)},
            "candidate_universe": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        },
        "cancer_records": cancer_records,
    }
    closure_path = output / "CLOSURE_AUDIT.json"
    closure_path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {"status": "PASS", "closure_audit_sha256": sha256(closure_path)}
    (output / "SUCCESS.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
