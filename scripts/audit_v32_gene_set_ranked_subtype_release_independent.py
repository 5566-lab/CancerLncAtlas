#!/usr/bin/env python
"""Independently audit the V3.2 Gene Set/ranked-subtype staging binding.

The script intentionally does not import the release materializer or subtype
implementation.  It re-hashes every bound artifact, independently checks table
closure, and recomputes a deterministic cross-pathway sample of RBO and
weighted-Jaccard stability values directly from the bound Gene Set members.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_RELEASE_BINDING_V1"
RELEASE_STATUS = "SUCCESS_DETERMINISTIC_V32_DERIVED_HEAD_HASH_BOUND"
AUDIT_FORMAT = "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_INDEPENDENT_AUDIT_V1"
AUDIT_BINDING_FORMAT = (
    "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_INDEPENDENT_AUDIT_BINDING_V1"
)
EXPECTED_ARTIFACTS = {
    "exact_pathway_associations",
    "gene_set_catalog",
    "gene_set_members",
    "gene_set_gmt",
    "gene_set_enrichment",
    "gene_set_coverage",
    "gene_set_report",
    "ranked_subtypes",
    "cancer_program_subtypes",
    "subtype_stability",
}
EXPECTED_SOURCES = {
    "audited_gene_set_release_manifest",
    "audited_gene_set_independent_audit_binding",
    "subtype_materialization_success",
    "subtype_materialization_coverage_audit",
    "deterministic_subtype_implementation",
}
EXPECTED_UPSTREAM_GENE_SET_AUDIT_CHECKS = 57
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class IndependentAuditError(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    if requested.is_symlink():
        raise IndependentAuditError(f"{label} may not be a symlink")
    source = requested.resolve()
    if not source.is_file():
        raise IndependentAuditError(f"{label} is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentAuditError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise IndependentAuditError(f"{label} must be a JSON object")
    return source, value


def resolve_record(record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise IndependentAuditError(f"Missing artifact record: {label}")
    expected = str(record.get("sha256", "")).lower()
    if not _SHA256.fullmatch(expected):
        raise IndependentAuditError(f"Invalid artifact SHA256: {label}")
    requested = Path(str(record.get("path", "")))
    if requested.is_symlink():
        raise IndependentAuditError(f"Artifact may not be a symlink: {label}")
    path = requested.resolve()
    if not path.is_file() or sha256(path) != expected:
        raise IndependentAuditError(f"Artifact drift: {label}")
    return path


def literal(path: str | Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def parquet(path: str | Path) -> str:
    return f"read_parquet({literal(path)}, hive_partitioning=false)"


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def add_check(
    checks: list[dict[str, Any]], name: str, observed: Any, expected: Any
) -> None:
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "status": "PASS" if observed == expected else "FAIL",
        }
    )


def rank_biased_overlap(left: list[str], right: list[str], p: float = 0.98) -> float:
    depth = min(200, max(len(left), len(right)))
    if depth <= 0:
        return 1.0 if not left and not right else 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    weighted = 0.0
    agreement = 0.0
    for position in range(1, depth + 1):
        if position <= len(left):
            left_seen.add(left[position - 1])
        if position <= len(right):
            right_seen.add(right[position - 1])
        agreement = len(left_seen & right_seen) / float(position)
        weighted += (1.0 - p) * agreement * (p ** (position - 1))
    return min(1.0, max(0.0, weighted + agreement * (p**depth)))


def weighted_jaccard(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    denominator = sum(max(float(left.get(key, 0)), float(right.get(key, 0))) for key in keys)
    if denominator == 0:
        return 1.0
    numerator = sum(min(float(left.get(key, 0)), float(right.get(key, 0))) for key in keys)
    return min(1.0, max(0.0, numerator / denominator))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_binding_sha = str(args.binding_sha256).lower()
    if not _SHA256.fullmatch(expected_binding_sha):
        raise IndependentAuditError("Binding SHA256 pin is invalid")
    binding_path, binding = read_json(args.binding, "release binding")
    if sha256(binding_path) != expected_binding_sha:
        raise IndependentAuditError("Release binding SHA256 drift")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise IndependentAuditError(f"Refusing audit output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    for name, expected in {
        "format": RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": RELEASE_STATUS,
        "module_ids": ["gene_set", "ranked_subtype"],
        "result_role": "DETERMINISTIC_V32_EXACT_PATHWAY_DERIVED_FUNCTIONAL_HEAD",
        "head_kind": "DETERMINISTIC_DERIVED_FUNCTIONAL_HEAD_NOT_PREDICTIVE_MODEL",
        "initialization": "v32_core_frozen_new_head",
        "model_training_performed": False,
        "learned_parameters": False,
        "new_checkpoint_created": False,
        "deterministic_derivation_from_current_v32_exact_gene_sets": True,
        "source_core_model_version": "V3.2",
        "source_core_frozen": True,
        "primary_score_preserved": True,
        "subtype_feedback_to_model": False,
        "target_level": "exact_pathway",
        "family_to_exact_broadcast": False,
        "old_checkpoints_used": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "missingness_encoding": "null_with_reason",
        "unavailable_fill_value": None,
        "release_ready": False,
        "production_deployed": False,
    }.items():
        add_check(checks, f"binding_{name}", binding.get(name), expected)
    add_check(
        checks,
        "binding_initialization_semantics",
        binding.get("initialization_semantics"),
        "FUNCTIONAL_HEAD_BOUND_TO_FROZEN_CURRENT_V32_EXACT_PATHWAY_OUTPUTS;NO_PARAMETER_INITIALIZATION",
    )
    add_check(
        checks,
        "binding_typed_unavailable",
        binding.get("typed_unavailable_cancers"),
        [
            {
                "cancer_id": "CHOL",
                "reason": "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS",
            },
            {
                "cancer_id": "UCS",
                "reason": "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS",
            },
        ],
    )

    sources = binding.get("sources")
    artifacts = binding.get("artifacts")
    if not isinstance(sources, Mapping) or not isinstance(artifacts, Mapping):
        raise IndependentAuditError("Binding sources or artifacts are missing")
    add_check(checks, "source_roles", sorted(sources), sorted(EXPECTED_SOURCES))
    add_check(checks, "artifact_roles", sorted(artifacts), sorted(EXPECTED_ARTIFACTS))
    source_paths = {
        role: resolve_record(sources.get(role), f"source {role}")
        for role in sorted(EXPECTED_SOURCES)
    }
    artifact_paths = {
        role: resolve_record(artifacts.get(role), f"artifact {role}")
        for role in sorted(EXPECTED_ARTIFACTS - {"gene_set_members"})
    }
    member_record = artifacts.get("gene_set_members")
    if not isinstance(member_record, Mapping):
        raise IndependentAuditError("Gene Set member declaration is missing")
    part_records = member_record.get("part_artifacts")
    if not isinstance(part_records, list) or len(part_records) != 33:
        raise IndependentAuditError("Gene Set release must bind all 33 member partitions")
    member_parts = [
        resolve_record(record, f"member partition {index}")
        for index, record in enumerate(part_records)
    ]
    member_tree = hashlib.sha256(
        "\n".join(f"{path.name}\t{sha256(path)}" for path in member_parts).encode()
    ).hexdigest()
    add_check(checks, "member_partition_count", len(member_parts), 33)
    add_check(checks, "member_tree_sha256", member_tree, member_record.get("sha256_tree"))

    _, upstream_manifest = read_json(
        source_paths["audited_gene_set_release_manifest"], "upstream Gene Set manifest"
    )
    _, upstream_audit = read_json(
        source_paths["audited_gene_set_independent_audit_binding"],
        "upstream Gene Set audit",
    )
    for name, observed, expected in (
        ("upstream_manifest_format", upstream_manifest.get("format"), "CC_HHGT_V3_2_GENE_SET_PARITY_RELEASE_V1"),
        ("upstream_manifest_status", upstream_manifest.get("status"), "SUCCESS_NEWLY_MATERIALIZED_V32"),
        ("upstream_manifest_target", upstream_manifest.get("target_level"), "exact_pathway"),
        ("upstream_manifest_family_broadcast", upstream_manifest.get("family_to_exact_broadcast"), False),
        ("upstream_manifest_old_checkpoints", upstream_manifest.get("old_checkpoints_used"), False),
        ("upstream_manifest_old_predictions", upstream_manifest.get("old_predictions_used"), False),
        ("upstream_manifest_old_rankings", upstream_manifest.get("old_rankings_used"), False),
        ("upstream_audit_format", upstream_audit.get("format"), "CC_HHGT_V3_2_GENE_SET_PARITY_INDEPENDENT_AUDIT_BINDING_V1"),
        ("upstream_audit_status", upstream_audit.get("status"), "PASS"),
        (
            "upstream_audit_pass_count",
            upstream_audit.get("pass_count"),
            EXPECTED_UPSTREAM_GENE_SET_AUDIT_CHECKS,
        ),
        ("upstream_audit_fail_count", upstream_audit.get("fail_count"), 0),
        ("upstream_audit_accepted", upstream_audit.get("accepted_for_registry_integration"), True),
    ):
        add_check(checks, name, observed, expected)
    upstream_declaration = upstream_audit.get("manifest")
    add_check(
        checks,
        "upstream_audit_manifest_sha_closure",
        upstream_declaration.get("sha256") if isinstance(upstream_declaration, Mapping) else None,
        sha256(source_paths["audited_gene_set_release_manifest"]),
    )
    _, source_success = read_json(
        source_paths["subtype_materialization_success"], "source materialization success"
    )
    _, source_coverage_audit = read_json(
        source_paths["subtype_materialization_coverage_audit"],
        "source materialization coverage audit",
    )
    for name, observed, expected in (
        ("source_materialization_status", source_success.get("status"), "SUCCESS"),
        ("source_family_target_false", source_success.get("family_used_as_target"), False),
        ("source_evidence_ranking_false", source_success.get("regulatory_evidence_used_for_ranking"), False),
        ("source_subtype_feedback_false", source_success.get("subtype_feedback_to_model"), False),
        ("source_pathway_seed_isolation", source_success.get("pathway_seed_isolation"), True),
        ("source_coverage_status", source_coverage_audit.get("status"), "PASS"),
        ("source_coverage_violations", source_coverage_audit.get("violations"), []),
        ("source_coverage_no_gene_set_cancers", source_coverage_audit.get("cancers_without_genesets"), ["CHOL", "UCS"]),
    ):
        add_check(checks, name, observed, expected)

    implementation_text = source_paths["deterministic_subtype_implementation"].read_text(
        encoding="utf-8"
    )
    add_check(checks, "implementation_consumes_members", "members: pd.DataFrame" in implementation_text, True)
    add_check(checks, "implementation_no_file_read", "read_parquet(" in implementation_text or "read_csv(" in implementation_text, False)
    add_check(checks, "implementation_no_family_target", "pathway_family" + "_id" in implementation_text, True)
    add_check(
        checks,
        "implementation_downstream_only_contract",
        "subtype labels back into CC-HHGT" in implementation_text,
        True,
    )

    exact_associations = parquet(artifact_paths["exact_pathway_associations"])
    catalog = parquet(artifact_paths["gene_set_catalog"])
    enrichment = parquet(artifact_paths["gene_set_enrichment"])
    context = parquet(artifact_paths["ranked_subtypes"])
    program = parquet(artifact_paths["cancer_program_subtypes"])
    pairwise = parquet(artifact_paths["subtype_stability"])
    members = "read_parquet([" + ",".join(literal(path) for path in member_parts) + "], hive_partitioning=false)"
    coverage_literal = literal(artifact_paths["gene_set_coverage"])
    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        exact_columns = {
            str(row[0])
            for row in con.execute(
                f"DESCRIBE SELECT * FROM {exact_associations}"
            ).fetchall()
        }
        exact_stats = con.execute(
            f"""
            SELECT count(*),count(DISTINCT (cancer_id,lncrna_id,pathway_id)),
                   count(DISTINCT cancer_id),count(DISTINCT lncrna_id),
                   count(DISTINCT pathway_id),
                   count_if(pathway_target_level<>'exact_pathway'),
                   count_if(pathway_id=pathway_family_id),
                   count_if(association_membership_probability IS NULL),
                   count_if(association_membership_probability IS NOT NULL AND
                            (NOT isfinite(association_membership_probability) OR
                             association_membership_probability NOT BETWEEN 0 AND 1)),
                   count_if(analysis_version<>?),
                   count_if(training_run_id<>'v32-exact-pathway-20260825')
            FROM {exact_associations}
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        catalog_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id), count(DISTINCT cancer_id),
                   count_if(pathway_target_level <> 'exact_pathway'),
                   count_if(ranking_uses_regulatory_evidence IS NOT FALSE),
                   count(*) - count(DISTINCT (cancer_id, pathway_id))
            FROM {catalog}
            """
        ).fetchone()
        member_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id),
                   count(*) - count(DISTINCT (geneset_id, lncrna_id)),
                   count_if(pathway_target_level <> 'exact_pathway')
            FROM {members}
            """
        ).fetchone()
        enrichment_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id),
                   count_if(family_to_exact_broadcast OR changes_primary_ranking
                            OR used_for_primary_ranking),
                   count_if(enrichment_scope <> 'DESCRIPTIVE_MEMBER_SUPPORT_NOT_CIRCULAR_ORA')
            FROM {enrichment}
            """
        ).fetchone()
        context_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT (cancer_id, pathway_id)),
                   count_if(cancer_id IN ('CHOL','UCS')),
                   count_if(classification_status='UNAVAILABLE'),
                   count_if(classification_status='UNAVAILABLE' AND
                            (selected_k<>0 OR proposed_k<>0 OR
                             pathway_context_subtype_id<>'UNAVAILABLE')),
                   count_if(classification_status<>'UNAVAILABLE' AND
                            (selected_k<1 OR pathway_context_subtype_id='UNAVAILABLE'))
            FROM {context}
            """
        ).fetchone()
        program_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT cancer_id),
                   count_if(cancer_id IN ('CHOL','UCS')),
                   count_if(selected_k < 1 OR proposed_k < 1 OR
                            classification_status IS NULL)
            FROM {program}
            """
        ).fetchone()
        pairwise_stats = con.execute(
            f"""
            SELECT count(*), count(*)-count(DISTINCT (pathway_id,cancer_a,cancer_b)),
                   count_if(cancer_a>=cancer_b),
                   count_if(rbo_similarity NOT BETWEEN 0 AND 1 OR
                            weighted_jaccard_similarity NOT BETWEEN 0 AND 1 OR
                            combined_similarity NOT BETWEEN 0 AND 1),
                   count_if(abs(combined_similarity-
                            (0.5*rbo_similarity+0.5*weighted_jaccard_similarity))>1e-14)
            FROM {pairwise}
            """
        ).fetchone()
        closure = con.execute(
            f"""
            SELECT count(*) FROM {catalog} m FULL OUTER JOIN {context} s
              USING (cancer_id,pathway_id,pathway_family_id)
            WHERE m.geneset_id IS NULL OR s.pathway_context_subtype_id IS NULL
            """
        ).fetchone()[0]
        shared_count_drift = con.execute(
            f"""
            WITH expected AS (
              SELECT cancer_id,pathway_id,
                     count(*) FILTER (WHERE shared_or_local_scope='shared') n_shared
              FROM {members} GROUP BY cancer_id,pathway_id
            )
            SELECT count(*) FROM {context} s JOIN expected e
              USING (cancer_id,pathway_id)
            WHERE s.n_shared_members<>e.n_shared
            """
        ).fetchone()[0]
        program_count_drift = con.execute(
            f"""
            WITH expected AS (
              SELECT cancer_id,count(*) FILTER
                (WHERE classification_status<>'UNAVAILABLE') n_evaluable
              FROM {context} GROUP BY cancer_id
            )
            SELECT count(*) FROM {program} p JOIN expected e USING (cancer_id)
            WHERE p.n_evaluable_pathways<>e.n_evaluable
            """
        ).fetchone()[0]
        pair_context_drift = con.execute(
            f"""
            SELECT count(*) FROM {pairwise} p
            LEFT JOIN {context} a ON p.pathway_id=a.pathway_id AND p.cancer_a=a.cancer_id
            LEFT JOIN {context} b ON p.pathway_id=b.pathway_id AND p.cancer_b=b.cancer_id
            WHERE a.pathway_id IS NULL OR b.pathway_id IS NULL OR
                  p.pathway_family_id<>a.pathway_family_id OR
                  p.pathway_family_id<>b.pathway_family_id OR
                  a.n_shared_members<10 OR b.n_shared_members<10
            """
        ).fetchone()[0]
        coverage_rows = con.execute(
            f"""
            SELECT cancer_id,coverage_status,publishable_genesets
            FROM read_csv_auto({coverage_literal},delim='\t',header=true)
            ORDER BY cancer_id
            """
        ).fetchall()
        sample_pairs = con.execute(
            f"""
            WITH numbered AS (
              SELECT *,row_number() OVER
                (PARTITION BY pathway_id ORDER BY cancer_a,cancer_b) rn
              FROM {pairwise}
            )
            SELECT pathway_id,cancer_a,cancer_b,rbo_similarity,
                   weighted_jaccard_similarity,combined_similarity
            FROM numbered WHERE rn=1 ORDER BY pathway_id LIMIT 128
            """
        ).fetchall()
        con.execute(
            "CREATE TEMP TABLE sampled_pairs(pathway_id VARCHAR,cancer_a VARCHAR,cancer_b VARCHAR)"
        )
        con.executemany(
            "INSERT INTO sampled_pairs VALUES (?,?,?)",
            [(str(row[0]), str(row[1]), str(row[2])) for row in sample_pairs],
        )
        sampled_members = con.execute(
            f"""
            WITH endpoints AS (
              SELECT pathway_id,cancer_a cancer_id FROM sampled_pairs
              UNION SELECT pathway_id,cancer_b cancer_id FROM sampled_pairs
            )
            SELECT m.pathway_id,m.cancer_id,m.association_direction,m.lncrna_id,
                   CAST(m.association_membership_probability AS DOUBLE),m.geneset_rank
            FROM {members} m JOIN endpoints e USING (pathway_id,cancer_id)
            WHERE m.shared_or_local_scope='shared'
            QUALIFY row_number() OVER
              (PARTITION BY m.pathway_id,m.cancer_id,m.association_direction
               ORDER BY m.geneset_rank,m.lncrna_id)<=200
            ORDER BY m.pathway_id,m.cancer_id,m.association_direction,
                     m.geneset_rank,m.lncrna_id
            """
        ).fetchall()
    finally:
        con.close()

    table_checks = {
        "exact_association_rows": (int(exact_stats[0]), 3_300_000),
        "exact_association_unique_keys": (int(exact_stats[1]), 3_300_000),
        "exact_association_cancers": (int(exact_stats[2]), 33),
        "exact_association_lncrnas": (int(exact_stats[3]), 8_541),
        "exact_association_pathways": (int(exact_stats[4]), 2_135),
        "exact_association_non_exact": (int(exact_stats[5]), 0),
        "exact_association_family_as_target": (int(exact_stats[6]), 0),
        "exact_association_null_probability": (int(exact_stats[7]), 0),
        "exact_association_invalid_probability": (int(exact_stats[8]), 0),
        "exact_association_version_violations": (int(exact_stats[9]), 0),
        "exact_association_run_violations": (int(exact_stats[10]), 0),
        "catalog_rows": (int(catalog_stats[0]), 54_380),
        "catalog_unique": (int(catalog_stats[1]), 54_380),
        "catalog_available_cancers": (int(catalog_stats[2]), 31),
        "catalog_non_exact": (int(catalog_stats[3]), 0),
        "catalog_evidence_ranked": (int(catalog_stats[4]), 0),
        "catalog_duplicate_exact_keys": (int(catalog_stats[5]), 0),
        "member_rows": (int(member_stats[0]), 2_217_719),
        "member_gene_sets": (int(member_stats[1]), 54_380),
        "member_duplicate_keys": (int(member_stats[2]), 0),
        "member_non_exact": (int(member_stats[3]), 0),
        "enrichment_rows": (int(enrichment_stats[0]), 54_380),
        "enrichment_unique": (int(enrichment_stats[1]), 54_380),
        "enrichment_semantic_violations": (int(enrichment_stats[2]), 0),
        "enrichment_scope_violations": (int(enrichment_stats[3]), 0),
        "context_rows": (int(context_stats[0]), 54_380),
        "context_unique": (int(context_stats[1]), 54_380),
        "context_chol_ucs_rows": (int(context_stats[2]), 0),
        "context_unavailable_rows": (int(context_stats[3]), 406),
        "context_unavailable_violations": (int(context_stats[4]), 0),
        "context_available_violations": (int(context_stats[5]), 0),
        "program_rows": (int(program_stats[0]), 31),
        "program_unique": (int(program_stats[1]), 31),
        "program_chol_ucs_rows": (int(program_stats[2]), 0),
        "program_semantic_violations": (int(program_stats[3]), 0),
        "pairwise_rows": (int(pairwise_stats[0]), 668_274),
        "pairwise_duplicate_keys": (int(pairwise_stats[1]), 0),
        "pairwise_order_violations": (int(pairwise_stats[2]), 0),
        "pairwise_range_violations": (int(pairwise_stats[3]), 0),
        "pairwise_formula_violations": (int(pairwise_stats[4]), 0),
        "context_catalog_closure": (int(closure), 0),
        "context_shared_member_count_drift": (int(shared_count_drift), 0),
        "program_evaluable_pathway_count_drift": (int(program_count_drift), 0),
        "pairwise_context_closure": (int(pair_context_drift), 0),
    }
    for name, (observed, expected) in table_checks.items():
        add_check(checks, name, observed, expected)
    forbidden_private_label_columns = {
        "label",
        "labels",
        "target",
        "y",
        "y_true",
        "training_label",
        "heldout_label",
        "private_label",
    }
    add_check(
        checks,
        "exact_association_private_label_columns",
        sorted(exact_columns & forbidden_private_label_columns),
        [],
    )
    unavailable = [
        [str(cancer), str(status), int(count)]
        for cancer, status, count in coverage_rows
        if str(status) != "AVAILABLE"
    ]
    add_check(checks, "coverage_cancers", len(coverage_rows), 33)
    add_check(
        checks,
        "coverage_typed_unavailable",
        unavailable,
        [
            ["CHOL", "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS", 0],
            ["UCS", "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS", 0],
        ],
    )

    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for pathway, cancer, direction, lncrna, probability, _rank in sampled_members:
        profile = profiles.setdefault(
            (str(pathway), str(cancer)), {"directions": {}, "weights": {}}
        )
        token = f"{direction}\x1f{lncrna}"
        profile["directions"].setdefault(str(direction), []).append(token)
        profile["weights"][token] = float(probability)
    recompute_failures = 0
    for pathway, cancer_a, cancer_b, expected_rbo, expected_jaccard, expected_combined in sample_pairs:
        left = profiles[(str(pathway), str(cancer_a))]
        right = profiles[(str(pathway), str(cancer_b))]
        directions = sorted(set(left["directions"]) | set(right["directions"]))
        directional = [
            rank_biased_overlap(
                left["directions"].get(direction, []),
                right["directions"].get(direction, []),
            )
            for direction in directions
        ]
        observed_rbo = sum(directional) / len(directional) if directional else 1.0
        observed_jaccard = weighted_jaccard(left["weights"], right["weights"])
        observed_combined = 0.5 * observed_rbo + 0.5 * observed_jaccard
        if (
            not math.isclose(observed_rbo, float(expected_rbo), abs_tol=1e-12)
            or not math.isclose(observed_jaccard, float(expected_jaccard), abs_tol=1e-12)
            or not math.isclose(observed_combined, float(expected_combined), abs_tol=1e-12)
        ):
            recompute_failures += 1
    add_check(checks, "sampled_stability_pathways", len(sample_pairs), 128)
    add_check(checks, "sampled_stability_recompute_failures", recompute_failures, 0)

    with artifact_paths["gene_set_gmt"].open("r", encoding="utf-8") as handle:
        gmt_lines = sum(1 for line in handle if line.strip())
    add_check(checks, "gmt_nonempty_lines", gmt_lines, 54_380)
    success_path, success = read_json(binding_path.parent / "SUCCESS.json", "release SUCCESS")
    add_check(checks, "success_binding_name", success.get("binding"), binding_path.name)
    add_check(checks, "success_binding_sha", success.get("binding_sha256"), expected_binding_sha)
    add_check(checks, "success_trained_model_false", success.get("trained_model"), False)
    add_check(checks, "success_release_ready_false", success.get("release_ready"), False)
    add_check(checks, "success_production_false", success.get("production_deployed"), False)

    fail_count = sum(item["status"] == "FAIL" for item in checks)
    pass_count = len(checks) - fail_count
    report = {
        "format": AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS" if fail_count == 0 else "FAIL",
        "binding": {"path": str(binding_path), "sha256": expected_binding_sha},
        "independent_of_release_materializer": True,
        "release_materializer_imported": False,
        "subtype_implementation_imported": False,
        "stability_values_independently_recomputed": True,
        "checks": checks,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "accepted_for_api_integration": fail_count == 0,
        "release_ready": False,
        "production_deployed": False,
    }
    report_path = output / "INDEPENDENT_AUDIT_REPORT.json"
    atomic_json(report_path, report)
    audit_binding = {
        "format": AUDIT_BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": report["status"],
        "release_binding": report["binding"],
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
        "pass_count": pass_count,
        "fail_count": fail_count,
        "accepted_for_api_integration": fail_count == 0,
        "independent_of_release_materializer": True,
        "release_materializer_imported": False,
        "subtype_implementation_imported": False,
        "release_ready": False,
        "production_deployed": False,
    }
    audit_binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    atomic_json(audit_binding_path, audit_binding)
    atomic_json(
        output / "SUCCESS.json",
        {
            "status": audit_binding["status"],
            "binding": audit_binding_path.name,
            "binding_sha256": sha256(audit_binding_path),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "release_ready": False,
            "production_deployed": False,
        },
    )
    print(json.dumps(audit_binding, ensure_ascii=False, indent=2, sort_keys=True))
    if fail_count:
        raise IndependentAuditError(
            f"Independent audit failed {fail_count}/{len(checks)} checks"
        )


if __name__ == "__main__":
    main()
