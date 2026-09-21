"""Independent, fail-closed audit of the V3.2 artifact remediation release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb


FORMAT = "CC_HHGT_V3_2_HISTORICAL_ARTIFACT_REMEDIATION_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_root.resolve()
    audit = args.audit_root.resolve()
    if audit.exists():
        raise RuntimeError(f"Audit output reuse refused: {audit}")
    audit.mkdir(parents=True)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        passed = observed == expected
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    success_path = release / "SUCCESS.json"
    binding_path = release / "HISTORICAL_ARTIFACT_REMEDIATION_BINDING.json"
    success = read_json(success_path)
    binding = read_json(binding_path)
    check("binding_format", binding.get("format"), FORMAT)
    check(
        "materialization_status",
        success.get("status"),
        "SUCCESS_MATERIALIZATION_WITH_GAPS_PRESERVED",
    )
    check("binding_hash", sha256_file(binding_path), success.get("binding_sha256"))
    check("production_not_deployed", binding.get("production_deployed"), False)
    check("parity_not_overclaimed", success.get("parity_release_ready"), False)
    check("family_broadcast_forbidden", binding.get("family_broadcast"), False)
    for filename, record in sorted(binding.get("files", {}).items()):
        path = release / filename
        check(f"file_exists::{filename}", path.is_file() and path.stat().st_size > 0, True)
        if path.is_file():
            check(f"file_sha256::{filename}", sha256_file(path), record.get("sha256"))
    required_logical_ids = {
        "clinical": {
            "v32_clinical_endpoint_summary",
            "v32_survival_associations",
            "v32_patient_risk_oof",
            "v32_translational_priority",
            "v32_clinical_report_manifest",
        },
        "mutation": {
            "v32_mutation_context",
            "v32_mutation_subgroup_predictions",
            "v32_mutation_lncrna_exact_pathway",
        },
        "cnv": {"v32_cnv_context", "v32_cnv_lncrna_exact_pathway", "v32_cnv_coverage"},
        "mixed_lncrna_protein_pathway_query": {
            "v32_mixed_set_encoder",
            "v32_custom_gene_set_encoder",
            "v32_protein_set_encoder",
            "v32_identifier_crosswalk",
            "v32_exact_pathway_ora_reference",
        },
        "single_cell": {
            "v32_sc_lncrna_celltype",
            "v32_sc_lncrna_exact_pathway",
            "v32_sc_activity",
            "v32_sc_ucell",
            "v32_sc_pseudotime",
            "v32_sc_figure_manifest",
        },
        "evidence_transformer": {
            "v32_evidence_transformer_model",
            "v32_neural_evidence_probability",
            "v32_fused_confidence_probability",
            "v32_evidence_direction_probability",
        },
    }
    index = binding.get("logical_artifact_index", {})
    check(
        "logical_artifact_index_capabilities",
        sorted(index),
        sorted(required_logical_ids),
    )
    permitted_absent = {
        ("single_cell", "v32_sc_pseudotime"),
    }
    for capability_id, expected_ids in sorted(required_logical_ids.items()):
        observed = index.get(capability_id, {})
        check(
            f"logical_artifact_ids::{capability_id}",
            sorted(observed),
            sorted(expected_ids),
        )
        for artifact_id in sorted(expected_ids):
            record = observed[artifact_id]
            absent = record.get("present") is False
            check(
                f"logical_artifact_absence_allowed::{capability_id}::{artifact_id}",
                absent,
                (capability_id, artifact_id) in permitted_absent,
            )
            if not absent:
                path = Path(record["path"])
                check(
                    f"logical_artifact_exists::{capability_id}::{artifact_id}",
                    path.is_file() and path.stat().st_size > 0,
                    True,
                )
                if path.is_file():
                    check(
                        f"logical_artifact_hash::{capability_id}::{artifact_id}",
                        sha256_file(path),
                        record["sha256"],
                    )

    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute("SET preserve_insertion_order=false")
    try:
        # Clinical: recompute values and rank independently from the V3.2 entity table.
        clinical_manifest = read_json(release / "CLINICAL_RELEASE_MANIFEST.json")
        clinical_source = Path(
            clinical_manifest["sources"]["clinical_entity_associations"]["path"]
        )
        clinical_priority = release / "clinical_translational_priority.parquet"
        clinical_counts = con.execute(
            """
            SELECT count(*), count_if(availability), count_if(NOT availability),
                   count(DISTINCT cancer_id), count(DISTINCT clinical_endpoint),
                   count_if(NOT availability AND translational_priority_score IS NOT NULL),
                   count_if(changes_primary_ranking)
            FROM read_parquet(?)
            """,
            [str(clinical_priority)],
        ).fetchone()
        check("clinical_priority_rows", int(clinical_counts[0]), 884_520)
        check("clinical_priority_cancers", int(clinical_counts[3]), 33)
        check("clinical_priority_endpoints", int(clinical_counts[4]), 6)
        check("clinical_unavailable_score_null", int(clinical_counts[5]), 0)
        check("clinical_primary_unchanged", int(clinical_counts[6]), 0)
        clinical_value_mismatch = con.execute(
            f"""
            WITH expected AS (
              SELECT cancer_id,subject_type,subject_id,clinical_endpoint,
                     CASE WHEN availability THEN clinical_relevance_probability ELSE NULL END
                       AS translational_priority_score,
                     availability,
                     CASE WHEN availability THEN '' ELSE failure_reason END AS unavailable_reason
              FROM read_parquet('{sql_path(clinical_source)}')
            ), observed AS (
              SELECT cancer_id,subject_type,subject_id,clinical_endpoint,
                     translational_priority_score,availability,unavailable_reason
              FROM read_parquet('{sql_path(clinical_priority)}')
            )
            SELECT count(*) FROM (
              (SELECT * FROM expected EXCEPT ALL SELECT * FROM observed)
              UNION ALL
              (SELECT * FROM observed EXCEPT ALL SELECT * FROM expected)
            )
            """
        ).fetchone()[0]
        check("clinical_value_exact_recompute_mismatches", int(clinical_value_mismatch), 0)
        clinical_rank_mismatch = con.execute(
            f"""
            WITH expected AS (
              SELECT cancer_id,subject_type,subject_id,clinical_endpoint,
                     CASE WHEN availability THEN CAST(row_number() OVER (
                       PARTITION BY cancer_id,clinical_endpoint,subject_type
                       ORDER BY CASE WHEN availability THEN 0 ELSE 1 END,
                                clinical_relevance_probability DESC NULLS LAST,
                                fdr ASC NULLS LAST,subject_id ASC
                     ) AS BIGINT) ELSE NULL END AS expected_rank
              FROM read_parquet('{sql_path(clinical_source)}')
            ), observed AS (
              SELECT cancer_id,subject_type,subject_id,clinical_endpoint,
                     translational_priority_rank
              FROM read_parquet('{sql_path(clinical_priority)}')
            )
            SELECT count(*) FROM expected e JOIN observed o
              USING(cancer_id,subject_type,subject_id,clinical_endpoint)
            WHERE e.expected_rank IS DISTINCT FROM o.translational_priority_rank
            """
        ).fetchone()[0]
        check("clinical_rank_exact_recompute_mismatches", int(clinical_rank_mismatch), 0)
        endpoint_counts = con.execute(
            """
            SELECT count(*),count(DISTINCT cancer_id),count(DISTINCT clinical_endpoint),
                   count_if(clinical_endpoint='DFS' AND
                     endpoint_unavailable_policy!='NULL_WITH_NO_DISTINCT_DFS_SOURCE'),
                   count_if(endpoint_substituted)
            FROM read_parquet(?)
            """,
            [str(release / "clinical_endpoint_summary.parquet")],
        ).fetchone()
        check("clinical_endpoint_summary_rows", int(endpoint_counts[0]), 198)
        check("clinical_endpoint_summary_cancers", int(endpoint_counts[1]), 33)
        check("clinical_endpoint_summary_endpoints", int(endpoint_counts[2]), 6)
        check("clinical_dfs_policy_mismatch", int(endpoint_counts[3]), 0)
        check("clinical_endpoint_substitution_rows", int(endpoint_counts[4]), 0)

        # Genomic: recompute mutation logical projection and CNV coverage from source.
        genomic_manifest = read_json(release / "GENOMIC_LOGICAL_RELEASE_MANIFEST.json")
        genomic_source = Path(genomic_manifest["sources"]["typed_predictions"]["path"])
        mutation = release / "mutation_subgroup_predictions.parquet"
        mutation_counts = con.execute(
            """
            SELECT count(*),count_if(availability),count_if(NOT availability),
                   count_if(NOT availability AND mutation_subgroup_probability IS NOT NULL),
                   count_if(family_broadcast),count_if(missing_assumed_wildtype),
                   count_if(changes_primary_ranking),
                   count(*)-count(DISTINCT (cancer_id,lncrna_id,pathway_id))
            FROM read_parquet(?)
            """,
            [str(mutation)],
        ).fetchone()
        check("mutation_rows", int(mutation_counts[0]), 3_300_000)
        check("mutation_available_rows", int(mutation_counts[1]), 698_302)
        check("mutation_unavailable_rows", int(mutation_counts[2]), 2_601_698)
        check("mutation_unavailable_probability_nonnull", int(mutation_counts[3]), 0)
        check("mutation_family_broadcast_rows", int(mutation_counts[4]), 0)
        check("mutation_missing_as_wildtype_rows", int(mutation_counts[5]), 0)
        check("mutation_primary_changed_rows", int(mutation_counts[6]), 0)
        check("mutation_duplicate_exact_keys", int(mutation_counts[7]), 0)
        mutation_mismatch = con.execute(
            f"""
            WITH expected AS (
              SELECT cancer_id,lncrna_id,pathway_id,
                     CASE WHEN mutation_available THEN mutation_context_probability ELSE NULL END
                       AS mutation_subgroup_probability,
                     mutation_available AS availability,
                     CASE WHEN mutation_available THEN '' ELSE mutation_unavailable_reason END
                       AS unavailable_reason,
                     mutation_patient_folds_with_prediction
              FROM read_parquet('{sql_path(genomic_source)}')
            ), observed AS (
              SELECT cancer_id,lncrna_id,pathway_id,mutation_subgroup_probability,
                     availability,unavailable_reason,mutation_patient_folds_with_prediction
              FROM read_parquet('{sql_path(mutation)}')
            )
            SELECT count(*) FROM (
              (SELECT * FROM expected EXCEPT ALL SELECT * FROM observed)
              UNION ALL
              (SELECT * FROM observed EXCEPT ALL SELECT * FROM expected)
            )
            """
        ).fetchone()[0]
        check("mutation_projection_exact_recompute_mismatches", int(mutation_mismatch), 0)
        cnv = release / "cnv_coverage.parquet"
        cnv_mismatch = con.execute(
            f"""
            WITH expected AS (
              SELECT cancer_id,count(*) AS formal_exact_key_rows,
                     count_if(cnv_available) AS cnv_available_rows,
                     count_if(NOT cnv_available) AS cnv_unavailable_rows,
                     count_if(cnv_available)::DOUBLE/count(*) AS cnv_coverage_fraction
              FROM read_parquet('{sql_path(genomic_source)}') GROUP BY cancer_id
            ), observed AS (
              SELECT cancer_id,formal_exact_key_rows,cnv_available_rows,
                     cnv_unavailable_rows,cnv_coverage_fraction
              FROM read_parquet('{sql_path(cnv)}')
            )
            SELECT count(*) FROM expected e FULL JOIN observed o USING(cancer_id)
            WHERE e.cancer_id IS NULL OR o.cancer_id IS NULL
               OR e.formal_exact_key_rows IS DISTINCT FROM o.formal_exact_key_rows
               OR e.cnv_available_rows IS DISTINCT FROM o.cnv_available_rows
               OR e.cnv_unavailable_rows IS DISTINCT FROM o.cnv_unavailable_rows
               OR abs(e.cnv_coverage_fraction-o.cnv_coverage_fraction)>1e-15
            """
        ).fetchone()[0]
        check("cnv_coverage_exact_recompute_mismatches", int(cnv_mismatch), 0)
        cnv_counts = con.execute(
            """
            SELECT count(*),sum(cnv_available_rows),sum(cnv_unavailable_rows),
                   count_if(family_broadcast),count_if(missing_assumed_neutral),
                   count_if(unavailable_fill_value IS NOT NULL)
            FROM read_parquet(?)
            """,
            [str(cnv)],
        ).fetchone()
        check("cnv_coverage_cancers", int(cnv_counts[0]), 33)
        check("cnv_available_rows", int(cnv_counts[1]), 86_461)
        check("cnv_unavailable_rows", int(cnv_counts[2]), 3_213_539)
        check("cnv_family_broadcast_rows", int(cnv_counts[3]), 0)
        check("cnv_missing_as_neutral_rows", int(cnv_counts[4]), 0)
        check("cnv_unavailable_fill_rows", int(cnv_counts[5]), 0)

        # Three encoder files must bind real deterministic code and exact assets.
        mixed = read_json(release / "MIXED_ENCODER_RELEASE_MANIFEST.json")
        encoder_ids = (
            "v32_mixed_set_encoder",
            "v32_custom_gene_set_encoder",
            "v32_protein_set_encoder",
        )
        check("mixed_encoder_artifact_count", len([x for x in encoder_ids if x in mixed["artifacts"]]), 3)
        for encoder_id in encoder_ids:
            record = mixed["artifacts"][encoder_id]
            path = Path(record["path"])
            check(f"mixed_encoder_hash::{encoder_id}", sha256_file(path), record["sha256"])
            encoder = read_json(path)
            check(
                f"mixed_encoder_status::{encoder_id}",
                encoder.get("status"),
                "READY_HASH_BOUND_DETERMINISTIC_ENCODER",
            )
            check(f"mixed_encoder_no_learned_params::{encoder_id}", encoder.get("learnable_encoder_parameters"), False)
            check(f"mixed_encoder_family_broadcast::{encoder_id}", encoder.get("family_broadcast"), False)
            code = Path(encoder["query_implementation"]["path"])
            check(f"mixed_encoder_code_hash::{encoder_id}", sha256_file(code), encoder["query_implementation"]["sha256"])
            for asset_id, asset in encoder["assets"].items():
                check(
                    f"mixed_encoder_asset_hash::{encoder_id}::{asset_id}",
                    sha256_file(Path(asset["path"])),
                    asset["sha256"],
                )

        # Single-cell gaps are evidence-backed, not empty artifacts presented as ready.
        single_cell = read_json(release / "SINGLE_CELL_FUNCTIONAL_BINDING.json")
        sc_activity = Path(single_cell["artifacts"]["v32_sc_activity"]["path"])
        sc_counts = con.execute(
            """
            SELECT count(*),count_if(activity_available),
                   count_if(mean_pseudotime IS NOT NULL AND isfinite(mean_pseudotime)),
                   count(DISTINCT cancer_id)
            FROM read_parquet(?)
            """,
            [str(sc_activity)],
        ).fetchone()
        check("sc_activity_rows", int(sc_counts[0]), 4_354_871)
        check("sc_activity_available_rows", int(sc_counts[1]), 4_320_711)
        check("sc_pseudotime_numeric_rows", int(sc_counts[2]), 0)
        check("sc_activity_cancers", int(sc_counts[3]), 33)
        check("sc_overall_partial", single_cell.get("status"), "PARTIAL_WITH_TYPED_GAPS")
        check("sc_artifact_gate_not_overclaimed", single_cell.get("capability_artifact_gate_closed"), False)
        check(
            "sc_ucell_scope",
            single_cell["artifacts"]["v32_sc_ucell"].get("scope"),
            "HNSC_ONLY_PILOT",
        )
        check(
            "sc_ucell_formal_cancer_count",
            single_cell["artifacts"]["v32_sc_ucell"].get("formal_cancers_covered"),
            1,
        )
        check(
            "sc_pseudotime_typed_gap",
            single_cell["artifacts"]["v32_sc_pseudotime"].get("status"),
            "GAP_TYPED_UNAVAILABLE",
        )
        check(
            "sc_figure_files",
            single_cell["artifacts"]["v32_sc_figure_manifest"].get("available_figure_files"),
            0,
        )

        # Evidence fused confidence and separately audited direction probabilities exist.
        evidence = read_json(release / "EVIDENCE_SEMANTIC_RELEASE.json")
        native = Path(evidence["artifacts"]["v32_neural_evidence_probability"]["path"])
        fused = Path(evidence["artifacts"]["v32_fused_confidence_probability"]["path"])
        evidence_counts = con.execute(
            """
            SELECT count(*),count_if(availability),count_if(NOT availability),
                   count_if(family_to_exact_broadcast),count_if(affects_discovery),
                   count_if(NOT confidence_only),
                   count_if(availability AND direction NOT IN ('negative','neutral','positive'))
            FROM read_parquet(?)
            """,
            [str(native)],
        ).fetchone()
        check("evidence_native_rows", int(evidence_counts[0]), 3_300_000)
        check("evidence_native_available_rows", int(evidence_counts[1]), 825_753)
        check("evidence_native_unavailable_rows", int(evidence_counts[2]), 2_474_247)
        check("evidence_family_broadcast_rows", int(evidence_counts[3]), 0)
        check("evidence_affects_discovery_rows", int(evidence_counts[4]), 0)
        check("evidence_not_confidence_only_rows", int(evidence_counts[5]), 0)
        check("evidence_bad_direction_class_rows", int(evidence_counts[6]), 0)
        fused_counts = con.execute(
            """
            SELECT count(*),count_if(abs(fused_confidence_probability-primary_probability)>1e-12),
                   count_if(abs(confidence_evidence_transformer_fusion_weight)>1e-15),
                   count_if(abs(coalesce(confidence_evidence_transformer_logit_contribution,0))>1e-15),
                   count_if(used_for_primary_release),count_if(NOT primary_ranking_unchanged),
                   count_if(historical_predictions_used),count_if(historical_rankings_used)
            FROM read_parquet(?)
            """,
            [str(fused)],
        ).fetchone()
        check("fused_confidence_rows", int(fused_counts[0]), 3_300_000)
        check(
            "fused_rows_different_from_primary",
            int(fused_counts[1]),
            int(evidence["counts"]["fused_confidence_rows_different_from_primary"]),
        )
        check("evidence_nonzero_fusion_weight_rows", int(fused_counts[2]), 0)
        check("evidence_nonzero_fusion_contribution_rows", int(fused_counts[3]), 0)
        check("fused_used_for_primary_rows", int(fused_counts[4]), 0)
        check("fused_primary_ranking_changed_rows", int(fused_counts[5]), 0)
        check("fused_historical_prediction_rows", int(fused_counts[6]), 0)
        check("fused_historical_ranking_rows", int(fused_counts[7]), 0)
        direction_record = evidence["artifacts"]["v32_evidence_direction_probability"]
        direction = Path(direction_record["path"])
        direction_counts = con.execute(
            """
            SELECT count(*),count_if(direction_probability_available),
                   count_if(NOT direction_probability_available),
                   count(DISTINCT cancer_id),
                   count_if(direction_probability_available AND
                     (direction_negative_probability IS NULL OR
                      direction_neutral_probability IS NULL OR
                      direction_positive_probability IS NULL OR
                      abs(direction_negative_probability+
                          direction_neutral_probability+
                          direction_positive_probability-1.0)>1e-5)),
                   count_if(NOT direction_probability_available AND
                     (direction_negative_probability IS NOT NULL OR
                      direction_neutral_probability IS NOT NULL OR
                      direction_positive_probability IS NOT NULL)),
                   count_if(changes_primary_ranking),
                   count_if(changes_discovery_ranking),
                   count_if(used_for_fusion),
                   count_if(historical_checkpoint_used),
                   count_if(historical_prediction_used)
            FROM read_parquet(?)
            """,
            [str(direction)],
        ).fetchone()
        check("evidence_direction_probability_status", direction_record.get("status"), "SUCCESS_THREE_CLASS_PROBABILITIES_HASH_PINNED")
        check("evidence_direction_probability_present", direction_record.get("present"), True)
        check("evidence_direction_probability_rows", int(direction_counts[0]), 3_300_000)
        check("evidence_direction_probability_available_rows", int(direction_counts[1]), 825_753)
        check("evidence_direction_probability_typed_null_rows", int(direction_counts[2]), 2_474_247)
        check("evidence_direction_probability_cancers", int(direction_counts[3]), 33)
        check("evidence_direction_probability_bad_available_rows", int(direction_counts[4]), 0)
        check("evidence_direction_probability_bad_typed_null_rows", int(direction_counts[5]), 0)
        check("evidence_direction_probability_changes_primary", int(direction_counts[6]), 0)
        check("evidence_direction_probability_changes_discovery", int(direction_counts[7]), 0)
        check("evidence_direction_probability_used_for_fusion", int(direction_counts[8]), 0)
        check("evidence_direction_probability_historical_checkpoint", int(direction_counts[9]), 0)
        check("evidence_direction_probability_historical_prediction", int(direction_counts[10]), 0)
        for binding_key in ("release_binding", "independent_audit_binding"):
            declaration = direction_record[binding_key]
            bound_path = Path(declaration["path"])
            check(
                f"evidence_direction::{binding_key}::hash",
                sha256_file(bound_path),
                declaration["sha256"],
            )
    finally:
        con.close()

    failed = [item for item in checks if item["status"] != "PASS"]
    report = {
        "format": "CC_HHGT_V3_2_HISTORICAL_ARTIFACT_REMEDIATION_INDEPENDENT_AUDIT_V1",
        "status": "PASS" if not failed else "FAIL",
        "audit_interpretation": (
            "PASS means the remediation truthfully binds/derives available V3.2 artifacts and "
            "preserves declared gaps; it does not mean all 25 capability gates pass."
        ),
        "release_root": str(release),
        "check_count": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": len(failed),
        "checks": checks,
        "fully_closed_artifact_logic": binding.get("fully_closed_artifact_logic"),
        "partial_artifact_logic": binding.get("partial_artifact_logic"),
        "parity_release_ready": False,
        "production_deployed": False,
    }
    report_path = audit / "INDEPENDENT_AUDIT.json"
    write_json(report_path, report)
    binding_value = {
        "status": "PASS_HASH_BOUND" if not failed else "FAIL",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "release_binding_path": str(binding_path),
        "release_binding_sha256": sha256_file(binding_path),
        "checks": len(checks),
        "failed_checks": len(failed),
        "parity_release_ready": False,
    }
    audit_binding_path = audit / "INDEPENDENT_AUDIT_BINDING.json"
    write_json(audit_binding_path, binding_value)
    write_json(
        audit / "SUCCESS.json",
        {
            "status": (
                "SUCCESS_AUDIT_WITH_GAPS_PRESERVED" if not failed else "AUDIT_FAILED"
            ),
            "audit_binding_path": str(audit_binding_path),
            "audit_binding_sha256": sha256_file(audit_binding_path),
            "failed_checks": len(failed),
            "parity_release_ready": False,
        },
    )
    print(json.dumps(binding_value, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
