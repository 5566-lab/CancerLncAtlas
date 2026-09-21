#!/usr/bin/env python
"""Independent post-materialisation audit of the experiment/Evidence bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


EXPECTED_FORMAT = "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_BINDING_V1"
EXPECTED_STATUS = "ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER"
EXPECTED_EXPERIMENT_BINDING_SHA = (
    "ec85aff0debb2d374a00efcb009b0b65c7606598fae4e47fb57338f317bcbb2f"
)
EXPECTED_EVIDENCE_BINDING_SHA = (
    "444b5d779615ff94c93890d56ab8e807fd1b9ab1592a2bba9e610a0e44b06748"
)
EXPECTED_FUSION_BINDING_SHA = (
    "0d2b5a34d0464016438725ef1e2fcea1ff04db611c9a03e571b01f87b780b059"
)
EXPECTED_FUSION_AUDIT_SHA = (
    "9bcedbdc186f044bb70eef96feb8d1ab04fd1b5382b0cfeeaa50758d8664c3cd"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sql_path(path: str | Path) -> str:
    return "'" + Path(path).resolve().as_posix().replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-binding", required=True, type=Path)
    parser.add_argument("--expected-bridge-binding-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"Audit output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    binding_path = args.bridge_binding.resolve()
    observed_binding_sha = sha256(binding_path)
    check(
        "bridge_binding_sha256",
        observed_binding_sha == args.expected_bridge_binding_sha256,
        observed_binding_sha,
        args.expected_bridge_binding_sha256,
    )
    binding = load_json(binding_path)
    check("binding_format", binding.get("binding_format") == EXPECTED_FORMAT, binding.get("binding_format"), EXPECTED_FORMAT)
    check("route_status", binding.get("route_determination") == EXPECTED_STATUS, binding.get("route_determination"), EXPECTED_STATUS)
    check("fully_absorbed", binding.get("fully_absorbed_by_evidence_transformer") is True, binding.get("fully_absorbed_by_evidence_transformer"), True)
    check("separate_fusion_forbidden", binding.get("separate_fusion_forbidden") is True, binding.get("separate_fusion_forbidden"), True)
    check("production_not_deployed", binding.get("production_deployed") is False, binding.get("production_deployed"), False)

    expected_authorities = {
        "experiment_binding": EXPECTED_EXPERIMENT_BINDING_SHA,
        "evidence_binding": EXPECTED_EVIDENCE_BINDING_SHA,
        "transparent_fusion_binding": EXPECTED_FUSION_BINDING_SHA,
        "transparent_fusion_independent_audit_binding": EXPECTED_FUSION_AUDIT_SHA,
    }
    for role, expected in expected_authorities.items():
        declaration = binding["authorities"][role]
        observed = sha256(declaration["path"])
        check(f"authority_{role}", declaration.get("sha256") == expected and observed == expected, {"declared": declaration.get("sha256"), "observed": observed}, expected)

    for role, declaration in binding["artifacts"].items():
        path = Path(declaration["path"])
        observed_hash = sha256(path)
        check(f"artifact_hash_{role}", observed_hash == declaration["sha256"], observed_hash, declaration["sha256"])
    for role in ("audit", "manifest"):
        declaration = binding[role]
        observed_hash = sha256(declaration["path"])
        check(f"{role}_hash", observed_hash == declaration["sha256"], observed_hash, declaration["sha256"])
    for fold, declaration in binding["checkpoints"].items():
        observed_hash = sha256(declaration["checkpoint_path"])
        check(f"checkpoint_hash_fold_{fold}", observed_hash == declaration["checkpoint_sha256"], observed_hash, declaration["checkpoint_sha256"])
        check(f"same_fold_checkpoint_fold_{fold}", declaration["checkpoint_payload_patient_fold"] == int(fold) and declaration["same_fold_checkpoint"] is True and declaration["retrained"] is False and declaration["fold_switched"] is False, declaration, {"fold": int(fold), "same_fold": True, "retrained": False, "switched": False})

    experiment_binding = load_json(binding["authorities"]["experiment_binding"]["path"])
    experiment_manifest = load_json(experiment_binding["manifest"]["path"])
    raw_path = experiment_manifest["input_artifacts"]["evidence_event"]["path"]
    events_path = experiment_binding["artifacts"]["v32_experiment_events"]["path"]
    exact_path = experiment_binding["artifacts"]["v32_perturbation_exact_pathway"]["path"]
    rejected_path = experiment_binding["artifacts"]["v32_experiment_rejected"]["path"]
    candidates_path = experiment_manifest["input_artifacts"]["candidate_universe"]["path"]
    evidence_binding = load_json(binding["authorities"]["evidence_binding"]["path"])
    lineage_path = evidence_binding["artifacts"]["event_lineage"]["path"]
    physical_path = evidence_binding["artifacts"]["physical_facts"]["path"]
    bridge_path = binding["artifacts"]["lineage_bridge"]["path"]
    query_path = binding["artifacts"]["exact_query"]["path"]
    ablation_path = binding["artifacts"]["source_removal_ablation"]["path"]
    metrics_path = binding["artifacts"]["source_removal_metrics"]["path"]

    connection = duckdb.connect(database=":memory:")
    try:
        raw = connection.execute(
            f"""
            SELECT count(*) AS raw_rows,
              count_if(regexp_matches(lower(regexp_replace(coalesce(experiment_family,''),'[^a-z0-9]+','_','g')), '(^|_)perturbation(_|$)')) AS selected_rows
            FROM read_parquet({sql_path(raw_path)})
            """
        ).fetchone()
        check("raw_input_rows", raw[0] == 479238, int(raw[0]), 479238)
        check("raw_functional_perturbation_selected", raw[1] == 2155, int(raw[1]), 2155)
        source_counts = connection.execute(
            f"""
            SELECT count(*),count_if(mapping_available),count_if(NOT mapping_available),
                   count_if(family_broadcast_used),
                   count_if(independent_probability_generated),
                   count_if(release_ready)
            FROM read_parquet({sql_path(events_path)})
            """
        ).fetchone()
        check("source_event_accounting", tuple(map(int, source_counts[:3])) == (2155,1301,854), tuple(map(int, source_counts[:3])), (2155,1301,854))
        check("source_no_broadcast_probability_release", tuple(map(int, source_counts[3:])) == (0,0,0), tuple(map(int, source_counts[3:])), (0,0,0))
        rejected_rows = connection.execute(f"SELECT count(*) FROM read_parquet({sql_path(rejected_path)})").fetchone()[0]
        check("rejected_rows", rejected_rows == 854, int(rejected_rows), 854)
        exact_counts = connection.execute(
            f"""
            SELECT count(*),count(DISTINCT (cancer_id,lncrna_id,pathway_id)),
                   count(DISTINCT cancer_id),count(DISTINCT lncrna_id),count(DISTINCT pathway_id),
                   count_if(family_broadcast_used),count_if(changes_primary_ranking),
                   count_if(changes_discovery_ranking),count_if(independent_probability_generated)
            FROM read_parquet({sql_path(exact_path)})
            """
        ).fetchone()
        check("exact_materialisation_counts", tuple(map(int, exact_counts[:5])) == (32461,14452,33,234,1456), tuple(map(int, exact_counts[:5])), (32461,14452,33,234,1456))
        check("exact_no_broadcast_or_ranking_change", tuple(map(int, exact_counts[5:])) == (0,0,0,0), tuple(map(int, exact_counts[5:])), (0,0,0,0))
        pan_canonical = experiment_manifest["row_audit"]["canonical_pan_cancer_exact_event_rows"]
        check("canonical_pan_event_rows", pan_canonical == 24527, pan_canonical, 24527)
        candidate_mismatch = connection.execute(
            f"""
            SELECT count(*) FROM (
              SELECT DISTINCT e.cancer_id,e.lncrna_id,e.pathway_id
              FROM read_parquet({sql_path(exact_path)}) e
              ANTI JOIN read_parquet({sql_path(candidates_path)}) c
                USING(cancer_id,lncrna_id,pathway_id)
            )
            """
        ).fetchone()[0]
        check("candidate_key_mismatches", candidate_mismatch == 0, int(candidate_mismatch), 0)
        raw_identity_mismatch = connection.execute(
            f"""
            WITH selected AS (
              SELECT evidence_event_id FROM read_parquet({sql_path(raw_path)})
              WHERE regexp_matches(lower(regexp_replace(coalesce(experiment_family,''),'[^a-z0-9]+','_','g')), '(^|_)perturbation(_|$)')
            ), materialized AS (
              SELECT experiment_event_id AS evidence_event_id FROM read_parquet({sql_path(events_path)})
            )
            SELECT
              (SELECT count(*) FROM (SELECT * FROM selected EXCEPT SELECT * FROM materialized)),
              (SELECT count(*) FROM (SELECT * FROM materialized EXCEPT SELECT * FROM selected))
            """
        ).fetchone()
        check("raw_selected_identity_exact", tuple(map(int,raw_identity_mismatch)) == (0,0), tuple(map(int,raw_identity_mismatch)), (0,0))

        overlap = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count_if(x.perturbation_event_id=e.event_id) AS id_matches,
                   count_if(x.cancer_id=e.cancer_id
                         AND replace(x.lncrna_id,'LNC:','')=e.lncrna_id
                         AND x.evidence_pathway_id=e.pathway_id
                         AND x.partner_id=e.partner_id
                         AND x.source_record_id=e.source_record_id) AS signature_matches,
                   count_if(e.is_physical) AS physical_rows,
                   count(DISTINCT e.source_record_id) FILTER (WHERE e.is_physical)
                     AS physical_source_records,
                   count(DISTINCT e.physical_fact_id) FILTER (WHERE e.is_physical)
                     AS physical_facts,
                   count_if(e.family_broadcast_used) AS broadcast_rows
            FROM read_parquet({sql_path(exact_path)}) x
            LEFT JOIN read_parquet({sql_path(lineage_path)}) e
              ON x.perturbation_event_id=e.event_id
            """
        ).fetchone()
        check("evidence_event_overlap", tuple(map(int,overlap[:3])) == (32461,32461,32461), tuple(map(int,overlap[:3])), (32461,32461,32461))
        check(
            "physical_overlap_and_no_broadcast",
            tuple(map(int, overlap[3:])) == (460, 37, 37, 0),
            tuple(map(int, overlap[3:])),
            (460, 37, 37, 0),
        )
        physical_missing = connection.execute(
            f"""
            SELECT count(*) FROM read_parquet({sql_path(bridge_path)}) b
            LEFT JOIN read_parquet({sql_path(physical_path)}) p USING(physical_fact_id)
            WHERE b.is_physical AND p.physical_fact_id IS NULL
            """
        ).fetchone()[0]
        check("physical_fact_lineage_complete", physical_missing == 0, int(physical_missing), 0)
        bridge_counts = connection.execute(
            f"""
            SELECT count(*),count(DISTINCT (cancer_id,lncrna_id,pathway_id)),
                   count_if(experiment_native_event_id<>evidence_event_id),
                   count_if(absorption_status<>'{EXPECTED_STATUS}'),
                   count_if(NOT separate_fusion_forbidden OR used_for_separate_fusion
                     OR family_broadcast_used OR changes_primary_ranking OR changes_discovery_ranking)
            FROM read_parquet({sql_path(bridge_path)})
            """
        ).fetchone()
        check("bridge_rows_keys_and_identity", tuple(map(int,bridge_counts[:3])) == (32461,14452,0), tuple(map(int,bridge_counts[:3])), (32461,14452,0))
        check("bridge_antidoublecount_flags", tuple(map(int,bridge_counts[3:])) == (0,0), tuple(map(int,bridge_counts[3:])), (0,0))
        query_counts = connection.execute(
            f"""
            SELECT count(*),count_if(NOT experiment_event_available),
                   count_if(NOT evidence_native_available),
                   count_if(NOT evidence_transformer_native_available),
                   count_if(abs(evidence_confidence_probability-evidence_transformer_native_probability)>1e-12),
                   count_if(NOT separate_fusion_forbidden OR used_for_separate_fusion
                     OR used_for_primary_release OR changes_primary_ranking OR changes_discovery_ranking)
            FROM read_parquet({sql_path(query_path)})
            """
        ).fetchone()
        check("query_key_coverage", tuple(map(int,query_counts[:4])) == (14452,0,0,0), tuple(map(int,query_counts[:4])), (14452,0,0,0))
        check("query_probability_and_antidoublecount", tuple(map(int,query_counts[4:])) == (0,0), tuple(map(int,query_counts[4:])), (0,0))
        ablation_columns = {row[0] for row in connection.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_path(ablation_path)})").fetchall()}
        check("private_target_not_exported", "fusion_target_private" not in ablation_columns and "fusion_target" not in ablation_columns, sorted(ablation_columns & {"fusion_target_private","fusion_target"}), [])
        ablation_counts = connection.execute(
            f"""
            SELECT count(*),count_if(checkpoint_patient_fold<>evidence_leakage_fold),
                   count_if(NOT same_fold_checkpoint OR retrained OR fold_switched),
                   count_if(source_removed_event_count<=0),
                   count_if(experiment_selected_in_full_top64_count>0),
                   sum(experiment_event_count),sum(all_evidence_event_count),sum(source_removed_event_count)
            FROM read_parquet({sql_path(ablation_path)})
            """
        ).fetchone()
        check("same_fold_source_removal_ablation", tuple(map(int,ablation_counts[:4])) == (14452,0,0,0), tuple(map(int,ablation_counts[:4])), (14452,0,0,0))
        check("selected_experiment_bag_coverage", int(ablation_counts[4]) == 13257, int(ablation_counts[4]), 13257)
        check("affected_event_accounting", tuple(map(int,ablation_counts[5:])) == (32461,2257931,2225470), tuple(map(int,ablation_counts[5:])), (32461,2257931,2225470))
        metrics = connection.execute(
            f"SELECT * FROM read_parquet({sql_path(metrics_path)}) WHERE evaluation_group='ALL_PAIR_BLOCKED_OOF'"
        ).fetchdf()
        check("aggregate_metric_singleton", len(metrics) == 1 and int(metrics.iloc[0].evaluated_exact_keys) == 14452, {"rows":len(metrics),"keys":int(metrics.iloc[0].evaluated_exact_keys) if len(metrics) else None}, {"rows":1,"keys":14452})
        finite_metrics = all(
            math.isfinite(float(metrics.iloc[0][column]))
            for column in (
                "full_logloss","source_removed_logloss","full_brier","source_removed_brier",
                "mean_absolute_probability_delta",
            )
        ) if len(metrics) else False
        check("aggregate_metrics_finite", finite_metrics, finite_metrics, True)
    finally:
        connection.close()

    failed = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "audit_format": "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_INDEPENDENT_AUDIT_V1",
        "status": "PASS" if not failed else "FAIL",
        "accepted_for_api_integration": not failed,
        "bridge_binding": str(binding_path),
        "bridge_binding_sha256": observed_binding_sha,
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "checks": checks,
        "route_determination": EXPECTED_STATUS,
        "separate_fusion_forbidden": True,
        "production_deployed": False,
    }
    report_path = output / "INDEPENDENT_AUDIT_REPORT.json"
    write_json(report_path, report)
    audit_binding = {
        "format": "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_INDEPENDENT_AUDIT_BINDING_V1",
        "status": report["status"],
        "accepted_for_api_integration": report["accepted_for_api_integration"],
        "bridge_binding": {"path": str(binding_path), "sha256": observed_binding_sha},
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
        "auditor": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "production_deployed": False,
    }
    audit_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    write_json(audit_binding_path, audit_binding)
    complete = {
        "status": report["status"],
        "accepted_for_api_integration": report["accepted_for_api_integration"],
        "report_sha256": sha256(report_path),
        "audit_binding_sha256": sha256(audit_binding_path),
        "checks_passed": report["checks_passed"],
        "checks_failed": report["checks_failed"],
    }
    write_json(output / "AUDIT_COMPLETE.json", complete)
    print(json.dumps(complete, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
