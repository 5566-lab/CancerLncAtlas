#!/usr/bin/env python3
"""Write an immutable, local-only readiness audit for the routed comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--patient-folds",
        default="artifacts/v32_patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv",
    )
    parser.add_argument(
        "--patient-fold-authority-receipt",
        default="artifacts/v32_patient_fold_authority_20260829_r1/PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    output = Path(args.output_root)
    if not output.is_absolute():
        output = (root / output).resolve()
    if output.exists():
        raise RuntimeError(f"Readiness audit refuses output reuse: {output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(root))

    from cc_hhgt.v32.cancer_modality_router import validate_patient_oof_lineage
    from cc_hhgt.v32.multimodal_fusion import artifact_sha256
    from cc_hhgt.v32.patient_fold_authority import (
        TCGA_CANCERS,
        validate_frozen_v32_patient_fold_binding,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    paths = {
        "external_router_source": root / "cc_hhgt/v32/cancer_modality_router.py",
        "hierarchical_preparation_source": root / "cc_hhgt/v32/hierarchical_candidate_preparation.py",
        "hierarchical_inference_source": root / "cc_hhgt/v32/hierarchical_candidate_inference.py",
        "fair_comparator_source": root / "cc_hhgt/v32/routed_fair_comparison.py",
        "cpu_launcher": root / "scripts/server_launch_v32_cnv_router_cpu.sh",
        "gpu_launcher": root / "scripts/server_launch_v32_hierarchical_gpu.sh",
        "formal_candidate_universe": root / "artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parquet",
        "patient_fold_manifest": resolve(args.patient_folds),
        "patient_fold_authority_receipt": resolve(args.patient_fold_authority_receipt),
        "mutation_cnv_predictions": root / "artifacts/v32_full_multitask/genomic_fresh_rerun1/mutation_cnv_typed_predictions.parquet",
        "mutation_cnv_lineage": root / "artifacts/v32_full_multitask/genomic_fresh_rerun1/LINEAGE.json",
        "cnv_full33_raw_manifest": root / "artifacts/v32_pancancer_segment_cnv_manifest_20260829_r3_full33_reconciled/MANIFEST.json",
        "atac_readiness": root / "artifacts/v32_atac_readiness_audit_20260829_r1/ATAC_READINESS.json",
        "external_router_success": root / "artifacts/v32_routed_candidate/external_router/SUCCESS.json",
        "hierarchical_oof_success": root / "artifacts/v32_routed_candidate/hierarchical_oof/SUCCESS.json",
        "fair_comparison_success": root / "artifacts/v32_routed_candidate/fair_comparison/SUCCESS.json",
    }
    evidence = {name: _evidence(path) for name, path in paths.items()}

    patient_authority_valid = False
    patient_authority_error = None
    patient_authority_audit: dict[str, Any] = {}
    try:
        patient_authority_audit = validate_frozen_v32_patient_fold_binding(
            paths["patient_fold_manifest"],
            paths["patient_fold_authority_receipt"],
        )
        patient_authority_valid = True
    except Exception as exc:
        patient_authority_error = f"{type(exc).__name__}: {exc}"

    genomic_valid = False
    genomic_error = None
    genomic_lineage: dict[str, Any] = {}
    if paths["mutation_cnv_predictions"].is_file() and paths["mutation_cnv_lineage"].is_file():
        try:
            genomic_lineage = json.loads(paths["mutation_cnv_lineage"].read_text(encoding="utf-8"))
            validate_patient_oof_lineage(
                genomic_lineage,
                predictions_sha256=artifact_sha256(paths["mutation_cnv_predictions"]),
            )
            genomic_valid = True
        except Exception as exc:  # audit records the closed gate instead of hiding it
            genomic_error = f"{type(exc).__name__}: {exc}"

    atac = (
        json.loads(paths["atac_readiness"].read_text(encoding="utf-8"))
        if paths["atac_readiness"].is_file()
        else {}
    )
    cnv_full = (
        json.loads(paths["cnv_full33_raw_manifest"].read_text(encoding="utf-8"))
        if paths["cnv_full33_raw_manifest"].is_file()
        else {}
    )
    cnv_oof_cancers = sorted(genomic_lineage.get("cnv_cancers_with_calls", []))
    all_cancers = sorted(genomic_lineage.get("cnv_cancers_with_calls", []) + genomic_lineage.get("cnv_cancers_null_with_reason", []))
    cnv_missing_oof = sorted(set(all_cancers) - set(cnv_oof_cancers))

    payload = {
        "status": "BLOCKED_ON_REQUIRED_OOF_AND_ROUTED_OUTPUTS",
        "analysis_version": "CancerLncAtlas_V3.2_ROUTING_FAIR_COMPARE_READINESS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "LOCAL_READ_ONLY_AUDIT_NO_TRAINING_NO_8260_NO_SERVER_ACCESS",
        "formal_comparison_complete": False,
        "implementation": {
            "external_nested_router": True,
            "hierarchical_end_to_end_gate": True,
            "same_candidate_and_fold_comparator": True,
            "same_label_and_primary_row_check": True,
            "same_modality_availability_callability_row_check": True,
            "typed_null_row_check": True,
            "fresh_lineage_sha256_binding_for_both_architectures": True,
            "per_cancer_positive_increment_gate": {
                "modalities": ["mutation", "cnv", "atac"],
                "selection_scope": "INNER_VALIDATION_ONLY",
                "logloss_improvement_required": True,
                "brier_noninferiority_required": True,
                "heldout_reverse_selection_forbidden": True,
            },
        },
        "data_readiness": {
            "patient_fold_authority": {
                "valid": patient_authority_valid,
                "error": patient_authority_error,
                "audit": patient_authority_audit,
                "patient_id_source": "EXPLICIT_PATIENT_ID_COLUMN",
                "sample_id_fallback_used": False,
                "required_before_any_routed_comparison": True,
            },
            "mutation": {
                "fresh_v32_patient_fold_oof": genomic_valid,
                "lineage_error": genomic_error,
                "next_step": "RUN_EXTERNAL_AND_HIERARCHICAL_CANDIDATES_THEN_APPLY_PER_CANCER_INCREMENT_GATE",
            },
            "cnv": {
                "fresh_v32_patient_fold_oof": "PARTIAL_3_CANCERS" if genomic_valid else False,
                "oof_cancers": cnv_oof_cancers,
                "missing_oof_cancers": cnv_missing_oof,
                "full33_raw_manifest_present": bool(cnv_full.get("queried_cancers")),
                "full33_raw_files_download_required": int(cnv_full.get("selected_download_required_files", 0)),
                "next_step": "MATERIALIZE_FULL_CALLABILITY_AND_TRAIN_FRESH_5_FOLD_OOF",
            },
            "atac": {
                "fresh_v32_patient_fold_oof": bool(atac.get("fresh_v32_atac_predictions_generated", False)),
                "fresh_oof_cancers": atac.get("fresh_v32_atac_contribution_estimated_cancers", []),
                "raw_covered_cancers": atac.get("summary", {}).get("raw_covered_cancers", 0),
                "true_raw_gap_cancers": atac.get("true_gaps", {}).get("raw_absent_cancers", []),
                "next_step": "IMPLEMENT_MATERIALIZER_TRAINER_AND_FRESH_5_FOLD_OOF_BEFORE_ANY_INCREMENT_CLAIM",
            },
        },
        "processing_gaps_fixable_without_new_oof": [
            "stage one canonical primary fusion frame for both architectures",
            "run the external router once required modality inputs are fixed",
            "run the hierarchical candidate and materialize its OOF after training authorization",
            "execute the fail-closed comparison on the two OOF tables",
        ],
        "must_wait_for_new_oof": [
            "CNV cancers outside the existing BRCA/COAD/KIRP fresh OOF scope",
            "ATAC for every cancer because no fresh V3.2 ATAC OOF exists",
        ],
        "required_before_formal_success": [
            "canonical primary OOF staged",
            "receipt-bound explicit patient-first five-fold authority",
            "external nested OOF SUCCESS",
            "hierarchical end-to-end OOF SUCCESS",
            "equal comparison contracts",
            "row-level candidate/fold/label/primary/modality parity",
            "per-cancer inner-validation positive increment for each enabled modality",
            "untouched external validation before promotion",
        ],
        "evidence": evidence,
        "formal_v32_primary_unchanged": True,
        "training_started_by_audit": False,
        "server_accessed_by_audit": False,
        "output_reuse_forbidden": True,
    }
    report = output / "READINESS.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "status": payload["status"],
        "readiness_path": str(report),
        "readiness_sha256": _sha256(report),
        "formal_comparison_complete": False,
        "success_marker_intentionally_absent": True,
    }
    (output / "BLOCKED.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
