#!/usr/bin/env python
"""Independently audit the hash-bound V3.2 Gene Set parity release.

This file intentionally does not import the materializer.  It re-hashes every
bound source/output and recomputes all 54,380 descriptive enrichment rows from
the member parts, fusion scores, and physical-interaction release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_GENE_SET_PARITY_RELEASE_V1"
AUDIT_FORMAT = "CC_HHGT_V3_2_GENE_SET_PARITY_INDEPENDENT_AUDIT_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_GENE_SET_PARITY_INDEPENDENT_AUDIT_BINDING_V1"
GLOBAL_SCOPE = "GLOBAL_NOT_CANCER_SPECIFIC"
ENRICHMENT_FORMAT = "CC_HHGT_V3_2_GENE_SET_EVIDENCE_SUMMARY_V2"
NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE = (
    "NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE"
)
EXPECTED_ROLES = {
    "v32_gene_set_catalog",
    "v32_gene_set_member_matrix",
    "v32_gene_set_enrichment",
    "v32_gene_set_gmt",
    "v32_gene_set_report_manifest",
    "v32_gene_set_cancer_coverage",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class AuditFailure(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    if requested.is_symlink():
        raise AuditFailure(f"{role} is a symlink")
    source = requested.resolve()
    if not source.is_file():
        raise AuditFailure(f"{role} is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"{role} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"{role} must be an object")
    return source, value


def literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def relation(path: str | Path) -> str:
    return f"read_parquet({literal(Path(path).resolve())}, hive_partitioning=false)"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_record(record: Any, role: str, *, base: Path | None = None) -> Path:
    if not isinstance(record, dict):
        raise AuditFailure(f"Missing artifact record: {role}")
    raw = Path(str(record.get("path", "")))
    path = (base / raw).resolve() if base is not None and not raw.is_absolute() else raw.resolve()
    expected = str(record.get("sha256", "")).lower()
    if not _SHA.fullmatch(expected) or path.is_symlink() or not path.is_file():
        raise AuditFailure(f"Invalid artifact declaration: {role}")
    if sha256(path) != expected:
        raise AuditFailure(f"Artifact SHA256 drift: {role}")
    return path


def add_check(
    checks: list[dict[str, Any]], name: str, observed: Any, expected: Any
) -> None:
    status = "PASS" if observed == expected else "FAIL"
    checks.append(
        {"name": name, "observed": observed, "expected": expected, "status": status}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_manifest_sha = str(args.manifest_sha256).lower()
    if not _SHA.fullmatch(expected_manifest_sha):
        raise AuditFailure("Manifest SHA256 pin is invalid")
    manifest_path, manifest = read_json(args.manifest, "Gene Set release manifest")
    if sha256(manifest_path) != expected_manifest_sha:
        raise AuditFailure("Gene Set release manifest SHA256 drift")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise AuditFailure(f"Refusing audit output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    add_check(checks, "manifest_format", manifest.get("format"), RELEASE_FORMAT)
    add_check(checks, "manifest_analysis_version", manifest.get("analysis_version"), ANALYSIS_VERSION)
    add_check(checks, "manifest_module", manifest.get("module_id"), "gene_set")
    add_check(
        checks,
        "manifest_status",
        manifest.get("status"),
        "SUCCESS_NEWLY_MATERIALIZED_V32",
    )
    for field, expected in {
        "primary_score_preserved": True,
        "ranking_uses_auxiliary_evidence": False,
        "family_to_exact_broadcast": False,
        "old_checkpoints_used": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "release_ready": False,
        "production_deployed": False,
    }.items():
        add_check(checks, f"manifest_{field}", manifest.get(field), expected)

    required = manifest.get("required_artifact_ids")
    if not isinstance(required, dict):
        raise AuditFailure("Manifest lacks required_artifact_ids")
    add_check(checks, "artifact_roles", sorted(required), sorted(EXPECTED_ROLES))
    catalog = resolve_record(required.get("v32_gene_set_catalog"), "catalog")
    enrichment = resolve_record(required.get("v32_gene_set_enrichment"), "enrichment")
    gmt = resolve_record(required.get("v32_gene_set_gmt"), "GMT")
    report_path = resolve_record(required.get("v32_gene_set_report_manifest"), "report")
    coverage = resolve_record(required.get("v32_gene_set_cancer_coverage"), "coverage")
    _, report = read_json(report_path, "Gene Set report")
    add_check(checks, "report_status", report.get("status"), "SUCCESS_NEWLY_MATERIALIZED_V32")
    add_check(checks, "report_circular_ora_false", report.get("circular_self_ora_performed"), False)
    add_check(
        checks,
        "report_auxiliary_not_ranking",
        report.get("auxiliary_evidence_used_for_member_selection_or_ranking"),
        False,
    )

    member_record = required.get("v32_gene_set_member_matrix")
    if not isinstance(member_record, dict):
        raise AuditFailure("Member matrix record is missing")
    part_records = member_record.get("part_artifacts")
    if not isinstance(part_records, list) or len(part_records) != 33:
        raise AuditFailure("Member matrix must contain 33 bound partitions")
    parts = [resolve_record(item, f"member part {index}") for index, item in enumerate(part_records)]
    tree = hashlib.sha256(
        "\n".join(
            f"{part.name}\t{sha256(part)}" for part in parts
        ).encode()
    ).hexdigest()
    add_check(checks, "member_tree_sha256", tree, member_record.get("sha256_tree"))
    add_check(checks, "member_parts", len(parts), int(member_record.get("parts", -1)))
    member_relation = "read_parquet([" + ",".join(literal(part) for part in parts) + "], hive_partitioning=false)"

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise AuditFailure("Manifest lacks sources")
    primary = resolve_record(sources.get("primary_exact_pathway"), "primary exact pathway")
    fusion_binding_path = resolve_record(sources.get("fusion_binding"), "fusion binding")
    physical_manifest_path = resolve_record(sources.get("physical_manifest"), "physical manifest")
    _, fusion_binding = read_json(fusion_binding_path, "fusion binding")
    fusion_artifacts = fusion_binding.get("artifacts")
    if not isinstance(fusion_artifacts, dict):
        raise AuditFailure("Fusion binding lacks artifacts")
    fusion_scores = resolve_record(fusion_artifacts.get("secondary_scores"), "fusion scores")
    _, physical_manifest = read_json(physical_manifest_path, "physical manifest")
    physical_artifacts = physical_manifest.get("artifacts")
    if not isinstance(physical_artifacts, dict):
        raise AuditFailure("Physical manifest lacks artifacts")
    physical_scores = resolve_record(
        physical_artifacts.get("interaction_exact_pathway_enrichment.parquet"),
        "physical enrichment",
        base=physical_manifest_path.parent,
    )

    success_path, success = read_json(manifest_path.parent / "SUCCESS.json", "release success")
    add_check(checks, "success_manifest_sha", success.get("manifest_sha256"), expected_manifest_sha)
    add_check(checks, "success_release_ready_false", success.get("release_ready"), False)
    add_check(checks, "success_production_false", success.get("production_deployed"), False)
    add_check(checks, "success_manifest_name", success.get("manifest"), manifest_path.name)

    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        master_rel = relation(catalog)
        output_rel = relation(enrichment)
        primary_rel = relation(primary)
        fusion_rel = relation(fusion_scores)
        physical_rel = relation(physical_scores)
        master_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id), count(DISTINCT cancer_id),
                   count_if(pathway_target_level <> 'exact_pathway'),
                   count_if(ranking_uses_regulatory_evidence IS NOT FALSE),
                   count_if(member_count <= 0)
            FROM {master_rel}
            """
        ).fetchone()
        member_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id),
                   count(*) - count(DISTINCT (geneset_id, lncrna_id)),
                   count_if(pathway_target_level <> 'exact_pathway')
            FROM {member_relation}
            """
        ).fetchone()
        output_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id), count(DISTINCT cancer_id),
                   count_if(used_for_primary_ranking OR changes_primary_ranking OR
                            family_to_exact_broadcast),
                   count_if(enrichment_scope <> 'DESCRIPTIVE_MEMBER_SUPPORT_NOT_CIRCULAR_ORA'),
                   count_if(statistical_test <> 'NOT_APPLICABLE_SELECTION_DEFINED_BY_PRIMARY'),
                   count_if(mean_membership_probability < 0 OR mean_membership_probability > 1),
                   count_if(independent_support_channel_fraction IS NOT NULL AND
                            independent_support_channel_fraction NOT BETWEEN 0 AND 1),
                   count_if(
                     (independent_support_available_member_channel_count = 0 AND (
                        independent_support_channel_fraction IS NOT NULL OR
                        independent_support_availability IS NOT FALSE OR
                        independent_support_unavailable_reason IS DISTINCT FROM
                          {literal(NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE)}
                     )) OR
                     (independent_support_available_member_channel_count > 0 AND (
                        independent_support_channel_fraction IS NULL OR
                        independent_support_availability IS NOT TRUE OR
                        independent_support_unavailable_reason IS NOT NULL
                     ))
                   ),
                   count_if(independent_support_channel_hits >
                            independent_support_available_member_channel_count),
                   count_if(independent_support_total_member_channel_count <> 3 * member_count OR
                            independent_support_unavailable_member_channel_count <>
                              independent_support_total_member_channel_count -
                                independent_support_available_member_channel_count),
                   count_if(independent_support_channel_coverage_fraction NOT BETWEEN 0 AND 1),
                   count_if(enrichment_format <> {literal(ENRICHMENT_FORMAT)})
            FROM {output_rel}
            """
        ).fetchone()
        expected_counts = {
            "master_rows": 54_380,
            "master_unique": 54_380,
            "available_cancers": 31,
            "member_rows": 2_217_719,
            "member_genesets": 54_380,
            "output_rows": 54_380,
            "output_unique": 54_380,
            "output_cancers": 31,
        }
        observed_counts = {
            "master_rows": int(master_stats[0]),
            "master_unique": int(master_stats[1]),
            "available_cancers": int(master_stats[2]),
            "member_rows": int(member_stats[0]),
            "member_genesets": int(member_stats[1]),
            "output_rows": int(output_stats[0]),
            "output_unique": int(output_stats[1]),
            "output_cancers": int(output_stats[2]),
        }
        for key, expected in expected_counts.items():
            add_check(checks, key, observed_counts[key], expected)
        add_check(checks, "master_non_exact", int(master_stats[3]), 0)
        add_check(checks, "master_evidence_ranked", int(master_stats[4]), 0)
        add_check(checks, "master_empty", int(master_stats[5]), 0)
        add_check(checks, "duplicate_members", int(member_stats[2]), 0)
        add_check(checks, "member_non_exact", int(member_stats[3]), 0)
        add_check(checks, "output_semantic_violations", int(output_stats[3]), 0)
        add_check(checks, "output_scope_violations", int(output_stats[4]), 0)
        add_check(checks, "output_test_violations", int(output_stats[5]), 0)
        add_check(checks, "output_probability_violations", int(output_stats[6]), 0)
        add_check(checks, "output_support_fraction_violations", int(output_stats[7]), 0)
        add_check(checks, "output_missingness_semantic_violations", int(output_stats[8]), 0)
        add_check(checks, "output_numerator_denominator_violations", int(output_stats[9]), 0)
        add_check(checks, "output_denominator_identity_violations", int(output_stats[10]), 0)
        add_check(checks, "output_coverage_fraction_violations", int(output_stats[11]), 0)
        add_check(checks, "output_enrichment_format_violations", int(output_stats[12]), 0)

        coverage_rows = con.execute(
            f"""
            SELECT cancer_id, coverage_status, publishable_genesets
            FROM read_csv_auto({literal(coverage)}, delim='\\t', header=true)
            ORDER BY cancer_id
            """
        ).fetchall()
        unavailable = [
            str(row[0])
            for row in coverage_rows
            if str(row[1]) == "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS"
        ]
        bad_coverage = sum(
            1
            for _, status, count in coverage_rows
            if (str(status) == "AVAILABLE" and int(count) <= 0)
            or (
                str(status) == "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS"
                and int(count) != 0
            )
            or str(status)
            not in {"AVAILABLE", "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS"}
        )
        add_check(checks, "coverage_cancers", len(coverage_rows), 33)
        add_check(checks, "typed_unavailable_cancers", unavailable, ["CHOL", "UCS"])
        add_check(checks, "coverage_semantic_violations", bad_coverage, 0)

        source_closure = con.execute(
            f"""
            SELECT
              count_if(p.lncrna_id IS NULL),
              count_if(f.lncrna_id IS NULL),
              count_if(p.lncrna_id IS NOT NULL AND abs(
                CAST(m.association_membership_probability AS DOUBLE) -
                CAST(p.association_membership_probability AS DOUBLE)) > 1e-7),
              count_if(f.lncrna_id IS NOT NULL AND abs(
                CAST(m.association_membership_probability AS DOUBLE) -
                CAST(f.primary_probability AS DOUBLE)) > 1e-7)
            FROM {member_relation} m
            LEFT JOIN {primary_rel} p USING (cancer_id, lncrna_id, pathway_id)
            LEFT JOIN {fusion_rel} f USING (cancer_id, lncrna_id, pathway_id)
            """
        ).fetchone()
        for index, name in enumerate(
            ["missing_primary", "missing_fusion", "primary_drift", "fusion_primary_drift"]
        ):
            add_check(checks, name, int(source_closure[index]), 0)

        recomputation = f"""
        WITH member_fusion AS (
          SELECT m.geneset_id, m.lncrna_id,
            CAST(m.association_membership_probability AS DOUBLE) primary_probability,
            CAST(m.fold_selection_frequency AS DOUBLE) fold_selection_frequency,
            CAST(m.direction_stability AS DOUBLE) direction_stability,
            CAST(f.discovery_adjusted_probability AS DOUBLE) discovery_probability,
            CAST(f.fused_confidence_probability AS DOUBLE) confidence_probability,
            f.genomic_native_available,
            CAST(f.genomic_native_probability AS DOUBLE) genomic_native_probability,
            f.single_cell_native_available,
            CAST(f.single_cell_native_probability AS DOUBLE) single_cell_native_probability,
            f.evidence_transformer_native_available,
            CAST(f.evidence_transformer_native_probability AS DOUBLE)
              evidence_transformer_native_probability
          FROM {member_relation} m
          JOIN {fusion_rel} f USING (cancer_id, lncrna_id, pathway_id)
        ), fusion_summary AS (
          SELECT geneset_id, count(*) member_count,
            avg(primary_probability) mean_membership_probability,
            max(primary_probability) max_membership_probability,
            min(primary_probability) min_membership_probability,
            avg(fold_selection_frequency) mean_fold_selection_frequency,
            avg(direction_stability) mean_direction_stability,
            avg(discovery_probability) mean_discovery_probability,
            avg(confidence_probability) mean_confidence_probability,
            count(*) FILTER (WHERE genomic_native_available) genomic_available_member_count,
            avg(genomic_native_probability) FILTER (WHERE genomic_native_available)
              mean_genomic_native_probability,
            count(*) FILTER (WHERE single_cell_native_available)
              single_cell_available_member_count,
            avg(single_cell_native_probability) FILTER (WHERE single_cell_native_available)
              mean_single_cell_native_probability,
            count(*) FILTER (WHERE evidence_transformer_native_available)
              evidence_available_member_count,
            avg(evidence_transformer_native_probability)
              FILTER (WHERE evidence_transformer_native_available)
              mean_evidence_native_probability,
            count(*) FILTER (WHERE genomic_native_available OR single_cell_native_available
              OR evidence_transformer_native_available)
              any_native_auxiliary_available_member_count
          FROM member_fusion GROUP BY geneset_id
        ), physical_available_candidates AS (
          SELECT DISTINCT m.geneset_id, m.lncrna_id
          FROM {member_relation} m JOIN {physical_rel} p
            ON p.lncrna_id=m.lncrna_id
           AND p.availability IS TRUE AND p.cancer_scope={literal(GLOBAL_SCOPE)}
          UNION
          SELECT DISTINCT m.geneset_id, m.lncrna_id
          FROM {member_relation} m JOIN {physical_rel} p
            ON p.lncrna_id=m.lncrna_id
           AND p.availability IS TRUE AND p.cancer_scope=m.cancer_id
        ), physical_availability_summary AS (
          SELECT geneset_id, count(*) physical_available_member_count
          FROM physical_available_candidates GROUP BY geneset_id
        ), physical_candidates AS (
          SELECT m.geneset_id, m.lncrna_id, CAST(p.ora_fdr AS DOUBLE) fdr,
                 CAST(p.fold_enrichment AS DOUBLE) fold_enrichment
          FROM {member_relation} m JOIN {physical_rel} p
            ON p.lncrna_id=m.lncrna_id AND p.pathway_id=m.pathway_id
           AND p.availability IS TRUE AND p.cancer_scope={literal(GLOBAL_SCOPE)}
          UNION ALL
          SELECT m.geneset_id, m.lncrna_id, CAST(p.ora_fdr AS DOUBLE) fdr,
                 CAST(p.fold_enrichment AS DOUBLE) fold_enrichment
          FROM {member_relation} m JOIN {physical_rel} p
            ON p.lncrna_id=m.lncrna_id AND p.pathway_id=m.pathway_id
           AND p.availability IS TRUE AND p.cancer_scope=m.cancer_id
        ), member_physical AS (
          SELECT geneset_id, lncrna_id, min(fdr) best_physical_ora_fdr,
                 max(fold_enrichment) max_physical_fold_enrichment
          FROM physical_candidates GROUP BY geneset_id, lncrna_id
        ), physical_support_summary AS (
          SELECT geneset_id, count(*) physical_supported_member_count,
                 min(best_physical_ora_fdr) best_physical_ora_fdr,
                 max(max_physical_fold_enrichment) max_physical_fold_enrichment
          FROM member_physical GROUP BY geneset_id
        ), support_summary AS (
        SELECT s.*,
          coalesce(a.physical_available_member_count,0) physical_available_member_count,
          coalesce(p.physical_supported_member_count,0) physical_supported_member_count,
          p.best_physical_ora_fdr, p.max_physical_fold_enrichment,
          s.single_cell_available_member_count+s.evidence_available_member_count+
            coalesce(p.physical_supported_member_count,0) independent_support_channel_hits,
          s.single_cell_available_member_count+s.evidence_available_member_count+
            coalesce(a.physical_available_member_count,0)
            independent_support_available_member_channel_count
        FROM fusion_summary s
        LEFT JOIN physical_availability_summary a USING (geneset_id)
        LEFT JOIN physical_support_summary p USING (geneset_id)
        )
        SELECT s.*,
          3*s.member_count independent_support_total_member_channel_count,
          3*s.member_count-s.independent_support_available_member_channel_count
            independent_support_unavailable_member_channel_count,
          CASE WHEN s.independent_support_available_member_channel_count=0 THEN NULL
            ELSE CAST(s.independent_support_channel_hits AS DOUBLE)/
              CAST(s.independent_support_available_member_channel_count AS DOUBLE)
          END independent_support_channel_fraction,
          CAST(s.independent_support_channel_hits AS DOUBLE)/(3*s.member_count)
            independent_support_channel_coverage_fraction,
          s.independent_support_available_member_channel_count>0
            independent_support_availability,
          CASE WHEN s.independent_support_available_member_channel_count=0
            THEN {literal(NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE)} ELSE NULL END
            independent_support_unavailable_reason
        FROM support_summary s
        """
        integer_fields = [
            "member_count", "genomic_available_member_count",
            "single_cell_available_member_count", "evidence_available_member_count",
            "any_native_auxiliary_available_member_count",
            "physical_available_member_count", "physical_supported_member_count",
            "independent_support_channel_hits",
            "independent_support_available_member_channel_count",
            "independent_support_total_member_channel_count",
            "independent_support_unavailable_member_channel_count",
        ]
        float_fields = [
            "mean_membership_probability", "max_membership_probability",
            "min_membership_probability", "mean_fold_selection_frequency",
            "mean_direction_stability", "mean_discovery_probability",
            "mean_confidence_probability", "mean_genomic_native_probability",
            "mean_single_cell_native_probability", "mean_evidence_native_probability",
            "best_physical_ora_fdr", "max_physical_fold_enrichment",
            "independent_support_channel_fraction",
            "independent_support_channel_coverage_fraction",
        ]
        exact_fields = [
            "independent_support_availability",
            "independent_support_unavailable_reason",
        ]
        integer_predicates = [f"o.{field} IS DISTINCT FROM e.{field}" for field in integer_fields]
        float_predicates = [
            f"((o.{field} IS NULL) <> (e.{field} IS NULL) OR "
            f"(o.{field} IS NOT NULL AND abs(o.{field}-e.{field}) > 1e-12))"
            for field in float_fields
        ]
        exact_predicates = [
            f"o.{field} IS DISTINCT FROM e.{field}" for field in exact_fields
        ]
        comparison = con.execute(
            f"""
            WITH expected AS ({recomputation})
            SELECT
              count_if(e.geneset_id IS NULL OR o.geneset_id IS NULL) closure_errors,
              count_if({' OR '.join(integer_predicates)}) integer_mismatches,
              count_if({' OR '.join(float_predicates)}) float_mismatches,
              count_if({' OR '.join(exact_predicates)}) exact_mismatches
            FROM {output_rel} o FULL OUTER JOIN expected e USING (geneset_id)
            """
        ).fetchone()
        add_check(checks, "recomputed_key_closure", int(comparison[0]), 0)
        add_check(checks, "recomputed_integer_mismatches", int(comparison[1]), 0)
        add_check(checks, "recomputed_float_mismatches", int(comparison[2]), 0)
        add_check(checks, "recomputed_exact_mismatches", int(comparison[3]), 0)
    finally:
        con.close()

    # The GMT is intentionally large; line count is a cheap independent closure
    # against the catalog and avoids interpreting member identifiers as genes.
    with gmt.open("r", encoding="utf-8") as handle:
        gmt_lines = sum(1 for line in handle if line.strip())
    add_check(checks, "gmt_nonempty_lines", gmt_lines, 54_380)

    fail_count = sum(item["status"] == "FAIL" for item in checks)
    pass_count = len(checks) - fail_count
    report_out = {
        "format": AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS" if fail_count == 0 else "FAIL",
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "manifest": {"path": str(manifest_path), "sha256": expected_manifest_sha},
        "checks": checks,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "accepted_for_registry_integration": fail_count == 0,
        "release_ready": False,
        "production_deployed": False,
    }
    report_out_path = output / "INDEPENDENT_AUDIT_REPORT.json"
    atomic_json(report_out_path, report_out)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": report_out["status"],
        "manifest": report_out["manifest"],
        "report": {"path": str(report_out_path), "sha256": sha256(report_out_path)},
        "pass_count": pass_count,
        "fail_count": fail_count,
        "accepted_for_registry_integration": fail_count == 0,
        "independent_of_materializer_implementation": True,
        "release_ready": False,
        "production_deployed": False,
    }
    binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    atomic_json(binding_path, binding)
    success = {
        "status": binding["status"],
        "binding": binding_path.name,
        "binding_sha256": sha256(binding_path),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "release_ready": False,
        "production_deployed": False,
    }
    atomic_json(output / "SUCCESS.json", success)
    print(json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True))
    if fail_count:
        raise AuditFailure(f"Independent audit failed {fail_count}/{len(checks)} checks")


if __name__ == "__main__":
    main()
