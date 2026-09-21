#!/usr/bin/env python3
"""Run the hash-bound full V3.2 exact-pathway secondary fusion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_fusion_adapter import (  # noqa: E402
    BINDING_FORMAT as EVIDENCE_FUSION_BINDING_FORMAT,
)
from cc_hhgt.v32.evidence_semantic_wrapper import (  # noqa: E402
    SEMANTIC_WRAPPER_FORMAT,
    SEMANTIC_WRAPPER_STATUS,
)
from cc_hhgt.v32.fusion_expert_policy import (  # noqa: E402
    EXACT_PATHWAY_FUSION_EXPERTS,
    experts_for_endpoint,
    validate_exact_pathway_fusion_expert_policy,
)
from cc_hhgt.v32.multimodal_fusion import (  # noqa: E402
    ANALYSIS_VERSION,
    ResidualFusionConfig,
    artifact_sha256,
    build_pair_aggregated_fusion_frame_from_parquet,
    materialize_multimodal_fusion_release,
    train_pair_blocked_crossfit_fusion,
)
from cc_hhgt.v32.single_cell_fusion_binding import (  # noqa: E402
    BINDING_FORMAT as SINGLE_CELL_BINDING_FORMAT,
)
from cc_hhgt.v32.single_cell_fusion_post_audit import (  # noqa: E402
    BINDING_FORMAT as SINGLE_CELL_POST_AUDIT_BINDING_FORMAT,
)


PRIMARY = ROOT / "artifacts/v32_full_multitask/exact_pathway_release_r2/exact_pathway_five_fold_ensemble.parquet"
FOLD_ROOT = ROOT / "artifacts/formal_release_predictions_1seed"
GENOMIC = ROOT / "artifacts/v32_full_multitask/genomic_fresh_rerun1/mutation_cnv_typed_predictions.parquet"
GENOMIC_LINEAGE = ROOT / "artifacts/v32_full_multitask/genomic_fresh_rerun1/LINEAGE.json"
FORMAL_SOURCE_HASHES = {
    PRIMARY: "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1",
    GENOMIC: "a0b4d4bfe1399dc7d34968dced86ed62d76ea76b5638236ad11694277a3a24a5",
    FOLD_ROOT / "FOLD_0_PREDICTIONS.parquet": "0a8f3f7661a6a7781c68a7614d5da37659a2f1e497db7d0cdfb2bb09648e2f7b",
    FOLD_ROOT / "FOLD_1_PREDICTIONS.parquet": "11a1c141042436764b37e73641410e787f3ed4ff0bfa0a8b2bc2f90dad8b7f61",
    FOLD_ROOT / "FOLD_2_PREDICTIONS.parquet": "1bfb9bb4dcabec5c8bc59f21aa30d61a1e82e1b382a3bbd2f9574a6a7dfb4943",
    FOLD_ROOT / "FOLD_3_PREDICTIONS.parquet": "eb3d2dc4e21f810ee2959e7f8215d685fce635399e2da276a7afd24cee7deecb",
    FOLD_ROOT / "FOLD_4_PREDICTIONS.parquet": "2350bbc2e6672aadb7fb5fdfc59918170229d45d7b0a7307f0862aa4d16e8fab",
}
FORMAL_CANDIDATE_ROWS = 3_300_000
EVIDENCE_FUSION_POST_AUDIT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_EVIDENCE_FUSION_INDEPENDENT_POST_AUDIT_BINDING_V1"
)


class FormalFusionRunnerError(RuntimeError):
    """Raised when an upstream binding or formal source fails closed."""


def _read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FormalFusionRunnerError(f"{role} is missing or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalFusionRunnerError(f"{role} is invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise FormalFusionRunnerError(f"{role} must contain a JSON object")
    return source, value


def _read_bound_json(
    path: str | Path, expected_sha256: str, role: str
) -> tuple[Path, dict[str, Any]]:
    source, value = _read_json(path, role)
    if artifact_sha256(source) != str(expected_sha256).lower():
        raise FormalFusionRunnerError(f"{role} SHA256 drift")
    return source, value


def _require(payload: Mapping[str, Any], expected: Mapping[str, Any], role: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FormalFusionRunnerError(
                f"{role}.{key} drift: observed={payload.get(key)!r}, expected={value!r}"
            )


def _bound_record(record: Any, role: str, *, rows: int | None = None) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise FormalFusionRunnerError(f"{role} record is missing")
    source = Path(str(record.get("path", ""))).resolve()
    expected = str(record.get("sha256", "")).lower()
    if not source.is_file() or source.is_symlink() or artifact_sha256(source) != expected:
        raise FormalFusionRunnerError(f"{role} path/SHA256 drift")
    if rows is not None and int(record.get("rows", -1)) != int(rows):
        raise FormalFusionRunnerError(f"{role} row-count drift")
    return source, expected


def _bound_explicit_file(path: Any, sha256: Any, role: str) -> tuple[Path, str]:
    source = Path(str(path or "")).resolve()
    expected = str(sha256 or "").lower()
    if not source.is_file() or source.is_symlink() or artifact_sha256(source) != expected:
        raise FormalFusionRunnerError(f"{role} path/SHA256 drift")
    return source, expected


def _validate_formal_sources() -> list[Path]:
    for path, expected in FORMAL_SOURCE_HASHES.items():
        if not path.is_file() or path.is_symlink() or artifact_sha256(path) != expected:
            raise FormalFusionRunnerError(f"Pinned formal source drift: {path}")
    _, lineage = _read_json(GENOMIC_LINEAGE, "genomic lineage")
    _require(
        lineage,
        {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "mutation_cnv",
            "training_status": "SUCCESS",
            "folds": 5,
            "prediction_rows": FORMAL_CANDIDATE_ROWS,
            "prediction_sha256": FORMAL_SOURCE_HASHES[GENOMIC],
            "private_head_trained_from_scratch": True,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
            "missing_mutation_assumed_wildtype": False,
            "missing_cnv_assumed_neutral": False,
        },
        "genomic lineage",
    )
    if Path(str(lineage.get("prediction_path", ""))).resolve() != GENOMIC.resolve():
        raise FormalFusionRunnerError("Genomic lineage prediction path drift")
    return [FOLD_ROOT / f"FOLD_{fold}_PREDICTIONS.parquet" for fold in range(5)]


def _resolve_single_cell(
    path: str | Path,
    expected_sha256: str,
    post_audit_binding_path: str | Path,
    expected_post_audit_binding_sha256: str,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    binding_path, binding = _read_bound_json(path, expected_sha256, "single-cell fusion binding")
    _require(
        binding,
        {
            "format": SINGLE_CELL_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_HASH_BOUND_FUSION_INPUT",
            "candidate_rows": FORMAL_CANDIDATE_ROWS,
            "available_rows": 954_541,
            "unavailable_rows": 2_345_459,
            "fusion_input_eligible": True,
            "direct_target_evidence": False,
            "affects_discovery": True,
            "affects_confidence": True,
            "changes_primary_ranking": False,
            "family_to_exact_broadcast": False,
            "unavailable_encoding": "null_with_reason",
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "release_ready": False,
            "production_deployed": False,
        },
        "single-cell fusion binding",
    )
    prediction, prediction_sha = _bound_record(
        binding.get("prediction"), "single-cell fusion prediction", rows=FORMAL_CANDIDATE_ROWS
    )
    candidate = binding.get("candidate_authority")
    if not isinstance(candidate, Mapping) or candidate.get("sha256") != (
        "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
    ):
        raise FormalFusionRunnerError("Single-cell candidate authority drift")
    _bound_record(binding.get("adapter_audit"), "single-cell adapter audit")
    _bound_record(binding.get("independent_audit_binding"), "single-cell independent audit binding")
    _bound_record(binding.get("independent_audit_report"), "single-cell independent audit report")
    post_path, post = _read_bound_json(
        post_audit_binding_path,
        expected_post_audit_binding_sha256,
        "single-cell post-materialization audit binding",
    )
    _require(
        post,
        {
            "format": SINGLE_CELL_POST_AUDIT_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "fail_closed": True,
            "fusion_binding_sha256": artifact_sha256(binding_path),
            "prediction_sha256": prediction_sha,
        },
        "single-cell post-materialization audit binding",
    )
    post_decision = post.get("release_decision")
    if not isinstance(post_decision, Mapping):
        raise FormalFusionRunnerError("Single-cell post-audit lacks release_decision")
    _require(
        post_decision,
        {
            "eligible_for_multimodal_fusion_input": True,
            "fusion_input_accepted": True,
            "direct_target_evidence": False,
            "family_to_exact_broadcast": False,
            "primary_ranking_may_be_changed": False,
            "production_deployed": False,
        },
        "single-cell post-audit release_decision",
    )
    _bound_explicit_file(
        post.get("post_materialization_audit_path"),
        post.get("post_materialization_audit_sha256"),
        "single-cell post-materialization audit report",
    )
    declarations = [
        {
            "path": str(prediction),
            "sha256": prediction_sha,
            "source_role": "current_v32_oof_prediction",
            "generation": "current_v32",
        },
        {
            "path": str(binding_path),
            "sha256": artifact_sha256(binding_path),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
        {
            "path": str(post_path),
            "sha256": artifact_sha256(post_path),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
    ]
    return prediction, binding_path, declarations


def _resolve_evidence(
    path: str | Path,
    expected_sha256: str,
    post_audit_binding_path: str | Path,
    expected_post_audit_binding_sha256: str,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    binding_path, binding = _read_bound_json(
        path, expected_sha256, "canonical Evidence fusion binding"
    )
    _require(
        binding,
        {
            "format": EVIDENCE_FUSION_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_HASH_BOUND_CANONICAL_FUSION_INPUT",
            "candidate_rows": FORMAL_CANDIDATE_ROWS,
            "available_rows": 825_753,
            "unavailable_rows": 2_474_247,
            "release_ready": False,
            "production_deployed": False,
            "normalized_bijection_proved": True,
            "fusion_input_eligible": True,
            "direct_target_evidence": True,
            "unavailable_encoding": "null_with_reason",
            "confidence_only": True,
            "changes_primary_ranking": False,
            "affects_discovery": False,
            "affects_confidence": True,
            "family_to_exact_broadcast": False,
            "raw_prediction_direct_fusion_allowed": False,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
        },
        "canonical Evidence fusion binding",
    )
    prediction, prediction_sha = _bound_record(
        binding.get("prediction"),
        "canonical Evidence fusion prediction",
        rows=FORMAL_CANDIDATE_ROWS,
    )
    candidate = binding.get("candidate_authority")
    if not isinstance(candidate, Mapping) or candidate.get("sha256") != (
        "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
    ):
        raise FormalFusionRunnerError("Evidence candidate authority drift")
    _bound_record(binding.get("adapter_audit"), "Evidence canonical adapter audit")
    wrapper_path, _ = _bound_record(
        binding.get("semantic_wrapper"), "Evidence semantic wrapper"
    )
    _, wrapper = _read_json(wrapper_path, "Evidence semantic wrapper")
    _require(
        wrapper,
        {
            "format": SEMANTIC_WRAPPER_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": SEMANTIC_WRAPPER_STATUS,
            "release_ready": False,
            "production_deployed": False,
            "confidence_only": True,
            "affects_discovery": False,
            "raw_prediction_direct_fusion_allowed": False,
            "canonical_candidate_adapter_required": True,
        },
        "Evidence semantic wrapper",
    )
    r2_path, _ = _bound_record(
        binding.get("r2_evidence_binding"), "R2 Evidence binding"
    )
    _bound_record(binding.get("independent_post_audit"), "Evidence independent post-audit")
    post_path, post = _read_bound_json(
        post_audit_binding_path,
        expected_post_audit_binding_sha256,
        "Evidence canonical adapter independent post-audit binding",
    )
    _require(
        post,
        {
            "format": EVIDENCE_FUSION_POST_AUDIT_BINDING_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "fusion_input_eligible": True,
            "primary_ranking_unchanged": True,
            "rowwise_reconstruction_difference": 0,
            "failed_checks": [],
        },
        "Evidence canonical adapter independent post-audit binding",
    )
    post_adapter, post_adapter_sha = _bound_record(
        post.get("adapter_binding"), "post-audited Evidence adapter binding"
    )
    if post_adapter != binding_path or post_adapter_sha != artifact_sha256(binding_path):
        raise FormalFusionRunnerError("Evidence post-audit references another adapter binding")
    post_prediction, post_prediction_sha = _bound_record(
        post.get("prediction"),
        "post-audited Evidence canonical prediction",
        rows=FORMAL_CANDIDATE_ROWS,
    )
    if post_prediction != prediction or post_prediction_sha != prediction_sha:
        raise FormalFusionRunnerError("Evidence post-audit references another prediction")
    _bound_record(
        post.get("candidate_authority"),
        "post-audited Evidence candidate authority",
        rows=FORMAL_CANDIDATE_ROWS,
    )
    _bound_record(
        post.get("raw_evidence_prediction"),
        "post-audited raw Evidence prediction",
        rows=FORMAL_CANDIDATE_ROWS,
    )
    _bound_record(post.get("semantic_wrapper"), "post-audited Evidence semantic wrapper")
    _bound_record(post.get("r2_evidence_binding"), "post-audited R2 Evidence binding")
    _bound_record(post.get("report"), "Evidence canonical adapter independent audit report")
    declarations = [
        {
            "path": str(prediction),
            "sha256": prediction_sha,
            "source_role": "current_v32_oof_prediction",
            "generation": "current_v32",
        },
        {
            "path": str(binding_path),
            "sha256": artifact_sha256(binding_path),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
        {
            "path": str(wrapper_path),
            "sha256": artifact_sha256(wrapper_path),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
        {
            "path": str(r2_path),
            "sha256": artifact_sha256(r2_path),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
        {
            "path": str(post_path),
            "sha256": artifact_sha256(post_path),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
    ]
    return prediction, binding_path, declarations


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-cell-binding", type=Path, required=True)
    parser.add_argument("--single-cell-binding-sha256", required=True)
    parser.add_argument("--single-cell-post-audit-binding", type=Path, required=True)
    parser.add_argument("--single-cell-post-audit-binding-sha256", required=True)
    parser.add_argument("--evidence-fusion-binding", type=Path, required=True)
    parser.add_argument("--evidence-fusion-binding-sha256", required=True)
    parser.add_argument("--evidence-fusion-post-audit-binding", type=Path, required=True)
    parser.add_argument("--evidence-fusion-post-audit-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run-id", default="V32-FUSION-FORMAL-20260826-R1")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FormalFusionRunnerError(f"Refusing to reuse formal fusion output: {output}")
    validate_exact_pathway_fusion_expert_policy()
    fold_paths = _validate_formal_sources()
    single_cell, single_binding, single_declarations = _resolve_single_cell(
        args.single_cell_binding,
        args.single_cell_binding_sha256,
        args.single_cell_post_audit_binding,
        args.single_cell_post_audit_binding_sha256,
    )
    evidence, evidence_binding, evidence_declarations = _resolve_evidence(
        args.evidence_fusion_binding,
        args.evidence_fusion_binding_sha256,
        args.evidence_fusion_post_audit_binding,
        args.evidence_fusion_post_audit_binding_sha256,
    )

    specs = tuple(EXACT_PATHWAY_FUSION_EXPERTS)
    expert_paths = {
        next(spec for spec in specs if spec.expert_id == "genomic"): GENOMIC,
        next(spec for spec in specs if spec.expert_id == "single_cell"): single_cell,
        next(spec for spec in specs if spec.expert_id == "evidence_transformer"): evidence,
    }
    frame, input_audit = build_pair_aggregated_fusion_frame_from_parquet(
        PRIMARY,
        fold_paths,
        expert_paths,
        expected_rows=FORMAL_CANDIDATE_ROWS,
        temp_directory=output.parent / f".{output.name}.duckdb_tmp",
    )
    input_audit.update(
        {
            "formal_primary_sha256": FORMAL_SOURCE_HASHES[PRIMARY],
            "formal_primary_fold_sha256": {
                str(index): FORMAL_SOURCE_HASHES[path]
                for index, path in enumerate(fold_paths)
            },
            "genomic_sha256": FORMAL_SOURCE_HASHES[GENOMIC],
            "single_cell_binding_path": str(single_binding),
            "single_cell_binding_sha256": artifact_sha256(single_binding),
            "evidence_canonical_fusion_binding_path": str(evidence_binding),
            "evidence_canonical_fusion_binding_sha256": artifact_sha256(evidence_binding),
            "evidence_canonical_fusion_post_audit_binding_path": str(
                args.evidence_fusion_post_audit_binding.resolve()
            ),
            "evidence_canonical_fusion_post_audit_binding_sha256": (
                args.evidence_fusion_post_audit_binding_sha256.lower()
            ),
            "raw_evidence_prediction_direct_fusion_allowed": False,
            "discovery_experts": [
                spec.expert_id for spec in experts_for_endpoint("discovery")
            ],
            "confidence_experts": [
                spec.expert_id for spec in experts_for_endpoint("confidence")
            ],
            "drug_in_exact_pathway_fusion": False,
            "primary_score_immutable": True,
        }
    )
    config = ResidualFusionConfig(
        seed=20260826,
        learning_rate=0.05,
        l2=1.0e-3,
        max_steps=600,
        patience=60,
        tolerance=1.0e-7,
        max_train_rows=2_000_000,
        max_validation_rows=500_000,
        batch_size=65_536,
        evaluation_interval=10,
        increment_logloss_delta=1.0e-4,
    )
    discovery = train_pair_blocked_crossfit_fusion(
        frame,
        specs,
        endpoint="discovery",
        config=config,
    )
    confidence = train_pair_blocked_crossfit_fusion(
        frame,
        specs,
        endpoint="confidence",
        config=config,
    )
    source_declarations: list[dict[str, Any]] = [
        {
            "path": str(PRIMARY),
            "sha256": FORMAL_SOURCE_HASHES[PRIMARY],
            "source_role": "current_v32_oof_prediction",
            "generation": "current_v32",
        },
        *[
            {
                "path": str(path),
                "sha256": FORMAL_SOURCE_HASHES[path],
                "source_role": "current_v32_oof_prediction",
                "generation": "current_v32",
            }
            for path in fold_paths
        ],
        {
            "path": str(GENOMIC),
            "sha256": FORMAL_SOURCE_HASHES[GENOMIC],
            "source_role": "current_v32_oof_prediction",
            "generation": "current_v32",
        },
        {
            "path": str(GENOMIC_LINEAGE),
            "sha256": artifact_sha256(GENOMIC_LINEAGE),
            "source_role": "provenance_audited_standardized_source",
            "generation": "current_v32",
        },
        *single_declarations,
        *evidence_declarations,
    ]
    result = materialize_multimodal_fusion_release(
        output,
        frame,
        discovery,
        confidence,
        source_declarations=source_declarations,
        training_run_id=args.training_run_id,
        input_audit=input_audit,
    )
    receipt_path = output / "FORMAL_RUN_RECEIPT.json"
    receipt = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_FORMAL_RUN_RECEIPT_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_SECONDARY_FUSION",
        "training_run_id": args.training_run_id,
        "runner_sha256": artifact_sha256(Path(__file__)),
        "binding_path": result["binding_path"],
        "binding_sha256": result["binding_sha256"],
        "public_rows": result["public_rows"],
        "discovery_performance_outcome": result["discovery_performance_outcome"],
        "confidence_performance_outcome": result["confidence_performance_outcome"],
        "primary_ranking_unchanged": True,
        "drug_actionability_separate": True,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {**result, "formal_run_receipt_sha256": artifact_sha256(receipt_path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
