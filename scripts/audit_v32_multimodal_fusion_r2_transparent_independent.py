"""Independent final audit for the transparent V3.2 multimodal fusion R2.

The common R1 auditor supplies the full source-hash, exact-universe, target,
pair-fold, formula, OOF-metric, privacy, and fresh/no-old checks.  This R2
auditor replaces the obsolete R1 fallback semantics and independently checks
the transparent native expert columns against their bound inputs.

Neither auditor imports or modifies the fusion implementation or release.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import duckdb

import audit_v32_multimodal_fusion_formal_independent as shared


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
EXPECTED_BINDING_SHA256 = "0d2b5a34d0464016438725ef1e2fcea1ff04db611c9a03e571b01f87b780b059"
EXPECTED_TRAINING_RUN_ID = "V32-FUSION-FORMAL-20260826-R2-TRANSPARENT"
EXPECTED_ROWS = 3_300_000
EXPECTED_NATIVE_AVAILABLE = {
    "genomic": 747_408,
    "single_cell": 954_541,
    "evidence_transformer": 825_753,
}
OBSOLETE_SHARED_CHECKS = {
    "formal_identity_contract",
    "discovery_oof_fallback_flag_violation",
    "discovery_final_fallback_flag_violation",
    "confidence_oof_fallback_flag_violation",
    "confidence_final_fallback_flag_violation",
}


class Checks:
    def __init__(self, initial: list[dict[str, Any]]) -> None:
        self.items = list(initial)

    def add(
        self,
        check_id: str,
        passed: bool,
        requirement: str,
        observed: Any,
        expected: Any,
    ) -> None:
        self.items.append(
            {
                "check_id": check_id,
                "status": "PASS" if bool(passed) else "FAIL",
                "severity": "ERROR",
                "requirement": requirement,
                "observed": observed,
                "expected": expected,
            }
        )

    @property
    def failed(self) -> list[str]:
        return [str(item["check_id"]) for item in self.items if item["status"] == "FAIL"]

    @property
    def pass_count(self) -> int:
        return sum(item["status"] == "PASS" for item in self.items)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def locate_source(binding: Mapping[str, Any], suffix: str) -> Path:
    matches = [
        Path(str(item["path"])).resolve()
        for item in binding["source_inputs"]
        if str(item["path"]).replace("\\", "/").endswith(suffix.replace("\\", "/"))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one bound source ending {suffix!r}, found {matches}")
    return matches[0]


def total_contribution_audit(
    connection: duckdb.DuckDBPyConnection,
    view: str,
    endpoint: str,
    stage: str,
    experts: list[str],
) -> dict[str, Any]:
    prefix = "" if stage == "oof" else f"{endpoint}_"
    score = "discovery_adjusted_probability" if endpoint == "discovery" else "fused_confidence_probability"
    total = " + ".join(
        f"coalesce(o.{prefix}{expert}_logit_contribution, 0.0)" for expert in experts
    )
    count = " + ".join(
        f"CASE WHEN o.{prefix}{expert}_available THEN 1 ELSE 0 END" for expert in experts
    )
    no_available = f"o.{prefix}no_available_auxiliary_expert"
    zero_total = f"o.{prefix}zero_total_contribution"
    fallback = f"o.{prefix}primary_fallback"
    primary_logit = shared.logit_expression("o.primary_probability")
    expected_nonzero = f"1.0 / (1.0 + exp(-(({primary_logit}) + ({total}))))"
    row = connection.execute(
        f"""
        SELECT
          count(*) AS row_count,
          count_if(({count}) = 0) AS independently_no_available_rows,
          count_if(({total}) = 0.0) AS independently_zero_total_rows,
          count_if(({total}) = 0.0 AND ({count}) > 0) AS zero_total_but_available_rows,
          count_if({no_available} IS DISTINCT FROM (({count}) = 0)) AS no_available_flag_mismatch,
          count_if({zero_total} IS DISTINCT FROM (({total}) = 0.0)) AS zero_total_flag_mismatch,
          count_if({fallback} IS DISTINCT FROM (({total}) = 0.0)) AS primary_fallback_semantic_mismatch,
          count_if(({total}) = 0.0 AND o.{score} IS DISTINCT FROM o.primary_probability) AS zero_total_nonexact_primary_rows,
          count_if(({count}) = 0 AND o.{score} IS DISTINCT FROM o.primary_probability) AS no_available_nonexact_primary_rows,
          count_if(({total}) <> 0.0 AND abs(o.{score} - ({expected_nonzero})) > {shared.FORMULA_TOLERANCE}) AS nonzero_formula_mismatch,
          max(CASE WHEN ({total}) <> 0.0 THEN abs(o.{score} - ({expected_nonzero})) ELSE NULL END) AS max_nonzero_formula_difference,
          count_if({fallback}) AS primary_fallback_rows,
          count_if({no_available} AND NOT {zero_total}) AS no_available_not_zero_rows
        FROM {view} o
        """
    ).fetchdf().iloc[0].to_dict()
    return {
        key: (int(value) if key != "max_nonzero_formula_difference" else float(value or 0.0))
        for key, value in row.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--binding-sha256", default=EXPECTED_BINDING_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = args.formal_dir.resolve()
    output = args.output.resolve()
    if not formal.is_dir():
        raise RuntimeError(f"R2 formal directory is missing: {formal}")
    if output.exists():
        raise RuntimeError(f"Refusing to reuse final audit output: {output}")

    script_path = Path(__file__).resolve()
    shared_script = script_path.with_name("audit_v32_multimodal_fusion_formal_independent.py")
    with tempfile.TemporaryDirectory(prefix="v32_r2_transparent_audit_", dir=str(output.parent)) as temporary:
        shared_output = Path(temporary) / "shared_audit"
        completed = subprocess.run(
            [
                sys.executable,
                str(shared_script),
                "--formal-dir",
                str(formal),
                "--binding-sha256",
                args.binding_sha256.lower(),
                "--output",
                str(shared_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Shared independent audit execution failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        shared_report_path = shared_output / "INDEPENDENT_POST_AUDIT_REPORT.json"
        shared_report = shared.read_json(shared_report_path)

    shared_failed = set(map(str, shared_report.get("failed_checks", [])))
    retained_shared_checks = [
        item for item in shared_report["checks"] if item["check_id"] not in OBSOLETE_SHARED_CHECKS
    ]
    checks = Checks(retained_shared_checks)
    checks.add(
        "shared_audit_only_expected_r1_semantic_drifts",
        shared_failed == OBSOLETE_SHARED_CHECKS,
        "The full shared audit has no R2 failure except the five intentionally replaced R1 identity/fallback checks",
        sorted(shared_failed),
        sorted(OBSOLETE_SHARED_CHECKS),
    )

    binding_path = formal / "MULTIMODAL_FUSION_BINDING.json"
    checkpoint_path = formal / "FUSION_CHECKPOINT.json"
    lineage_path = formal / "MODULE_LINEAGE.json"
    receipt_path = formal / "FORMAL_RUN_RECEIPT.json"
    success_path = formal / "SUCCESS.json"
    binding = shared.read_json(binding_path)
    checkpoint = shared.read_json(checkpoint_path)
    lineage = shared.read_json(lineage_path)
    receipt = shared.read_json(receipt_path)
    success = shared.read_json(success_path)
    binding_sha = shared.sha256(binding_path)

    identity = {
        "binding_sha256": binding_sha,
        "binding_training_run_id": binding.get("training_run_id"),
        "checkpoint_training_run_id": checkpoint.get("training_run_id"),
        "lineage_training_run_id": lineage.get("training_run_id"),
        "receipt_training_run_id": receipt.get("training_run_id"),
        "success_training_run_id": success.get("training_run_id"),
        "analysis_versions": sorted(
            {
                binding.get("analysis_version"),
                checkpoint.get("analysis_version"),
                lineage.get("analysis_version"),
                receipt.get("analysis_version"),
                success.get("analysis_version"),
            }
        ),
    }
    checks.add(
        "r2_transparent_formal_identity",
        binding_sha == args.binding_sha256.lower()
        and all(
            payload.get("training_run_id") == EXPECTED_TRAINING_RUN_ID
            for payload in (binding, checkpoint, lineage, receipt, success)
        )
        and identity["analysis_versions"] == [ANALYSIS_VERSION],
        "All R2 manifests identify the requested transparent run and exact binding SHA",
        identity,
        {
            "binding_sha256": args.binding_sha256.lower(),
            "training_run_id": EXPECTED_TRAINING_RUN_ID,
            "analysis_version": ANALYSIS_VERSION,
        },
    )

    public_path = Path(binding["artifacts"]["secondary_scores"]["path"]).resolve()
    discovery_oof_path = Path(binding["artifacts"]["discovery_oof_private"]["path"]).resolve()
    confidence_oof_path = Path(binding["artifacts"]["confidence_oof_private"]["path"]).resolve()
    genomic_path = locate_source(
        binding, "artifacts/v32_full_multitask/genomic_fresh_rerun1/mutation_cnv_typed_predictions.parquet"
    )
    single_cell_path = locate_source(
        binding, "artifacts/v32_single_cell_fusion_adapter_20260826_r1/single_cell_exact_fusion_expert.parquet"
    )
    evidence_path = locate_source(
        binding, "artifacts/v32_evidence_fusion_adapter_20260826_r1/evidence_exact_fusion_expert.parquet"
    )

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute(
        f"CREATE VIEW pub AS SELECT * FROM read_parquet('{shared.sql_path(public_path)}')"
    )
    connection.execute(
        f"CREATE VIEW doof AS SELECT * FROM read_parquet('{shared.sql_path(discovery_oof_path)}')"
    )
    connection.execute(
        f"CREATE VIEW coof AS SELECT * FROM read_parquet('{shared.sql_path(confidence_oof_path)}')"
    )
    connection.execute(
        f"CREATE VIEW g AS SELECT cancer_id,lncrna_id,pathway_id,mutation_cnv_context_probability AS source_probability,genomic_available AS source_available FROM read_parquet('{shared.sql_path(genomic_path)}')"
    )
    connection.execute(
        f"CREATE VIEW s AS SELECT cancer_id,lncrna_id,pathway_id,single_cell_replication_probability AS source_probability,single_cell_available AS source_available FROM read_parquet('{shared.sql_path(single_cell_path)}')"
    )
    connection.execute(
        f"CREATE VIEW e AS SELECT cancer_id,lncrna_id,pathway_id,evidence_confidence_probability AS source_probability,availability AS source_available FROM read_parquet('{shared.sql_path(evidence_path)}')"
    )

    checkpoint_experts = {
        "discovery": list(checkpoint["discovery"]["expert_ids"]),
        "confidence": list(checkpoint["confidence"]["expert_ids"]),
    }
    fallback_details: dict[str, Any] = {}
    for endpoint, view in (("discovery", "doof"), ("confidence", "coof")):
        fallback_details[f"{endpoint}_oof"] = total_contribution_audit(
            connection, view, endpoint, "oof", checkpoint_experts[endpoint]
        )
        fallback_details[f"{endpoint}_final"] = total_contribution_audit(
            connection, "pub", endpoint, "final", checkpoint_experts[endpoint]
        )

    fallback_fail_fields = (
        "no_available_flag_mismatch",
        "zero_total_flag_mismatch",
        "primary_fallback_semantic_mismatch",
        "zero_total_nonexact_primary_rows",
        "no_available_nonexact_primary_rows",
        "nonzero_formula_mismatch",
        "no_available_not_zero_rows",
    )
    for stage, details in fallback_details.items():
        checks.add(
            f"{stage}_transparent_fallback_contract",
            details["row_count"] == EXPECTED_ROWS
            and all(details[field] == 0 for field in fallback_fail_fields)
            and details["primary_fallback_rows"] == details["independently_zero_total_rows"],
            "No-availability, zero-total-contribution, primary_fallback, exact primary assignment, and nonzero formula have distinct consistent semantics",
            details,
            {
                "row_count": EXPECTED_ROWS,
                **{field: 0 for field in fallback_fail_fields},
                "primary_fallback_rows_equals_independently_zero_total_rows": True,
            },
        )
    zero_exact_failures = {
        stage: int(details["zero_total_nonexact_primary_rows"])
        for stage, details in fallback_details.items()
    }
    checks.add(
        "all_zero_total_contribution_rows_bitwise_primary",
        all(value == 0 for value in zero_exact_failures.values()),
        "Across both final endpoints and both OOF endpoints, every zero-total-contribution row is bit-for-bit primary",
        zero_exact_failures,
        {key: 0 for key in zero_exact_failures},
    )

    public_columns = set(shared.parquet_columns(connection, public_path))
    expected_native_columns = {
        f"{expert}_native_{suffix}"
        for expert in EXPECTED_NATIVE_AVAILABLE
        for suffix in ("probability", "available")
    }
    checks.add(
        "transparent_native_columns_complete",
        expected_native_columns.issubset(public_columns),
        "All three native expert probability/availability column pairs are public",
        sorted(expected_native_columns & public_columns),
        sorted(expected_native_columns),
    )

    native_details: dict[str, Any] = {}
    source_views = {"genomic": "g", "single_cell": "s", "evidence_transformer": "e"}
    for expert, source_view in source_views.items():
        row = connection.execute(
            f"""
            SELECT
              count(*) AS row_count,
              count_if(pub.{expert}_native_available) AS available_rows,
              count_if(pub.{expert}_native_available IS DISTINCT FROM src.source_available) AS availability_mismatch,
              count_if(pub.{expert}_native_probability IS DISTINCT FROM src.source_probability) AS probability_mismatch,
              count_if(pub.{expert}_native_available AND (pub.{expert}_native_probability IS NULL OR NOT isfinite(pub.{expert}_native_probability) OR pub.{expert}_native_probability<0 OR pub.{expert}_native_probability>1)) AS bad_available_probability,
              count_if(NOT pub.{expert}_native_available AND pub.{expert}_native_probability IS NOT NULL) AS unavailable_probability_not_null
            FROM pub JOIN {source_view} src USING(cancer_id,lncrna_id,pathway_id)
            """
        ).fetchdf().iloc[0].to_dict()
        details = {key: int(value) for key, value in row.items()}
        native_details[expert] = details
        checks.add(
            f"{expert}_native_rowwise_source_identity_and_typed_null",
            details
            == {
                "row_count": EXPECTED_ROWS,
                "available_rows": EXPECTED_NATIVE_AVAILABLE[expert],
                "availability_mismatch": 0,
                "probability_mismatch": 0,
                "bad_available_probability": 0,
                "unavailable_probability_not_null": 0,
            },
            "Public native probability/availability equals the exact bound input row-for-row and preserves typed nulls",
            details,
            {
                "row_count": EXPECTED_ROWS,
                "available_rows": EXPECTED_NATIVE_AVAILABLE[expert],
                "availability_mismatch": 0,
                "probability_mismatch": 0,
                "bad_available_probability": 0,
                "unavailable_probability_not_null": 0,
            },
        )

    route_consistency = connection.execute(
        """
        SELECT
          count_if(genomic_native_available IS DISTINCT FROM discovery_genomic_available OR genomic_native_available IS DISTINCT FROM confidence_genomic_available) genomic_route_mismatch,
          count_if(single_cell_native_available IS DISTINCT FROM discovery_single_cell_available OR single_cell_native_available IS DISTINCT FROM confidence_single_cell_available) single_cell_route_mismatch,
          count_if(evidence_transformer_native_available IS DISTINCT FROM confidence_evidence_transformer_available) evidence_route_mismatch
        FROM pub
        """
    ).fetchdf().iloc[0].to_dict()
    route_consistency = {key: int(value) for key, value in route_consistency.items()}
    checks.add(
        "native_and_fusion_availability_route_consistency",
        all(value == 0 for value in route_consistency.values()),
        "Native availability is identical to every endpoint route that consumes that expert",
        route_consistency,
        {key: 0 for key in route_consistency},
    )

    all_models = [checkpoint["discovery"], checkpoint["confidence"]]
    for endpoint in ("discovery", "confidence"):
        all_models.extend(checkpoint["fold_models"][endpoint])
    zero_weight_audit = {
        "single_cell_weights": [
            float(model["weights"][model["expert_ids"].index("single_cell")])
            for model in all_models
            if "single_cell" in model["expert_ids"]
        ],
        "evidence_transformer_weights": [
            float(model["weights"][model["expert_ids"].index("evidence_transformer")])
            for model in all_models
            if "evidence_transformer" in model["expert_ids"]
        ],
        "single_cell_native_available_rows": native_details["single_cell"]["available_rows"],
        "evidence_transformer_native_available_rows": native_details["evidence_transformer"]["available_rows"],
    }
    checks.add(
        "zero_weight_single_cell_and_evidence_native_scores_retained",
        all(value == 0.0 for value in zero_weight_audit["single_cell_weights"])
        and all(value == 0.0 for value in zero_weight_audit["evidence_transformer_weights"])
        and zero_weight_audit["single_cell_native_available_rows"] == EXPECTED_NATIVE_AVAILABLE["single_cell"]
        and zero_weight_audit["evidence_transformer_native_available_rows"] == EXPECTED_NATIVE_AVAILABLE["evidence_transformer"],
        "Zero learned residual weight does not delete native single-cell or Evidence probabilities",
        zero_weight_audit,
        {
            "all_single_cell_weights": 0.0,
            "all_evidence_transformer_weights": 0.0,
            "single_cell_native_available_rows": EXPECTED_NATIVE_AVAILABLE["single_cell"],
            "evidence_transformer_native_available_rows": EXPECTED_NATIVE_AVAILABLE["evidence_transformer"],
        },
    )

    metadata = {
        "binding_native_expert_probabilities_public": binding.get("native_expert_probabilities_public"),
        "binding_public_native_experts": binding.get("public_native_experts"),
        "binding_zero_total_fallback": binding.get("zero_total_contribution_exact_primary_fallback"),
        "lineage_native_expert_probabilities_public": lineage.get("native_expert_probabilities_public"),
        "lineage_public_native_experts": lineage.get("public_native_experts"),
        "lineage_zero_total_fallback": lineage.get("zero_total_contribution_exact_primary_fallback"),
        "checkpoint_model_zero_total_contracts": [
            model.get("zero_total_contribution_fallback") for model in all_models
        ],
    }
    checks.add(
        "transparent_release_metadata_contract",
        metadata["binding_native_expert_probabilities_public"] is True
        and metadata["lineage_native_expert_probabilities_public"] is True
        and metadata["binding_public_native_experts"] == ["genomic", "single_cell", "evidence_transformer"]
        and metadata["lineage_public_native_experts"] == ["genomic", "single_cell", "evidence_transformer"]
        and metadata["binding_zero_total_fallback"] is True
        and metadata["lineage_zero_total_fallback"] is True
        and all(value == "EXACT_PRIMARY_PROBABILITY" for value in metadata["checkpoint_model_zero_total_contracts"]),
        "Binding, lineage, and all 12 checkpoints explicitly bind transparent native output and exact zero-total fallback",
        metadata,
        "all transparent/fallback metadata exact",
    )

    status = "PASS" if not checks.failed else "FAIL"
    accepted_for_api = status == "PASS"
    report = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_R2_TRANSPARENT_INDEPENDENT_POST_AUDIT_REPORT_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "fail_closed": True,
        "audited_directory": str(formal),
        "audited_binding_path": str(binding_path),
        "audited_binding_sha256": binding_sha,
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "pass_count": checks.pass_count,
        "fail_count": len(checks.failed),
        "failed_checks": checks.failed,
        "checks": checks.items,
        "shared_full_audit": {
            "auditor_path": str(shared_script),
            "auditor_sha256": shared.sha256(shared_script),
            "original_failed_checks": sorted(shared_failed),
            "replaced_obsolete_checks": sorted(OBSOLETE_SHARED_CHECKS),
            "universe_statistics": shared_report.get("universe_statistics"),
            "expert_statistics": shared_report.get("expert_statistics"),
            "formula_reconstruction": shared_report.get("formula_reconstruction"),
            "oof_outcomes": shared_report.get("oof_outcomes"),
            "direct_source_hash_results": shared_report.get("direct_source_hash_results"),
            "nested_source_file_references_checked": shared_report.get("nested_source_file_references_checked"),
        },
        "fallback_reconstruction": fallback_details,
        "native_expert_reconstruction": native_details,
        "native_route_consistency": route_consistency,
        "zero_weight_native_retention": zero_weight_audit,
        "primary_score_preserved": shared_report.get("primary_score_preserved") is True,
        "discovery_performance_outcome": shared_report["oof_outcomes"]["discovery"]["recomputed_outcome"],
        "confidence_performance_outcome": shared_report["oof_outcomes"]["confidence"]["recomputed_outcome"],
        "scientific_status": {
            "discovery": shared_report["oof_outcomes"]["discovery"]["declared_status"],
            "confidence": shared_report["oof_outcomes"]["confidence"]["declared_status"],
        },
        "release_decision": {
            "accepted_for_api_integration": accepted_for_api,
            "requires_zero_failed_checks": True,
            "production_deployed": False,
        },
        "audit_code": {"path": str(script_path), "sha256": shared.sha256(script_path)},
    }

    output.mkdir(parents=True)
    report_path = output / "INDEPENDENT_POST_AUDIT_REPORT.json"
    atomic_json(report_path, report)
    report_sha = shared.sha256(report_path)
    audit_binding = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_R2_TRANSPARENT_INDEPENDENT_POST_AUDIT_BINDING_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "fail_closed": True,
        "formal_binding": {"path": str(binding_path), "sha256": binding_sha},
        "report": {"path": str(report_path), "sha256": report_sha},
        "audit_code": {"path": str(script_path), "sha256": shared.sha256(script_path)},
        "shared_audit_code": {"path": str(shared_script), "sha256": shared.sha256(shared_script)},
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "exact_candidate_rows": EXPECTED_ROWS,
        "pair_blocked_folds": 5,
        "pass_count": checks.pass_count,
        "fail_count": len(checks.failed),
        "failed_checks": checks.failed,
        "zero_total_contribution_exact_primary_fallback": all(
            value == 0 for value in zero_exact_failures.values()
        ),
        "native_expert_probabilities_public": True,
        "public_native_experts": ["genomic", "single_cell", "evidence_transformer"],
        "discovery_performance_outcome": report["discovery_performance_outcome"],
        "confidence_performance_outcome": report["confidence_performance_outcome"],
        "no_increment_capabilities_retained": True,
        "accepted_for_api_integration": accepted_for_api,
        "production_deployed": False,
    }
    audit_binding_path = output / "INDEPENDENT_POST_AUDIT_BINDING.json"
    atomic_json(audit_binding_path, audit_binding)
    audit_binding_sha = shared.sha256(audit_binding_path)
    completion = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_R2_TRANSPARENT_INDEPENDENT_POST_AUDIT_COMPLETION_V1",
        "status": status,
        "binding_path": str(audit_binding_path),
        "binding_sha256": audit_binding_sha,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "pass_count": checks.pass_count,
        "fail_count": len(checks.failed),
        "failed_checks": checks.failed,
        "accepted_for_api_integration": accepted_for_api,
    }
    atomic_json(output / "AUDIT_COMPLETE.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
