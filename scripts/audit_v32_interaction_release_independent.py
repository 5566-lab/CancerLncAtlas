#!/usr/bin/env python
"""Independently audit a formal local V3.2 physical-interaction release.

This verifier deliberately does not import the interaction materialiser.  It
rehashes every declared input/output, reconstructs the fact-to-relationship
and relationship-to-exact-pathway key closures with DuckDB, and independently
checks deterministic identifiers and ORA/BH statistics on deterministic
samples.  It writes only a new local audit directory after every gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
AUDIT_FORMAT = "CC_HHGT_V3_2_PHYSICAL_INTERACTION_INDEPENDENT_AUDIT_V1"
RELEASE_FORMAT = "CC_HHGT_V3_2_PHYSICAL_INTERACTION_RELEASE_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_EVIDENCE_OUTPUT_BINDING_V1"
WRAPPER_FORMAT = "CC_HHGT_V3_2_EVIDENCE_SEMANTIC_WRAPPER_V1"
GLOBAL_SCOPE = "GLOBAL_NOT_CANCER_SPECIFIC"
FORMAL_FACT_ROWS = 1_926_537
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_CANCERS = 33
FORMAL_PATHWAYS = 2_135
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ARTIFACTS = {
    "physical_interaction_relationships.parquet": "rel",
    "relationship_evidence.parquet": "ev",
    "interaction_exact_pathway_enrichment.parquet": "enrich",
    "rejected_physical_facts.parquet": "rejected",
    "unmapped_physical_partners.parquet": "unmapped",
}


class AuditError(RuntimeError):
    """Raised when an independent physical-interaction gate fails."""


def artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise AuditError(f"Missing or unsafe file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    if requested.is_symlink():
        raise AuditError(f"{role} is missing or unsafe: {requested}")
    source = requested.resolve()
    if not source.is_file():
        raise AuditError(f"{role} is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{role} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{role} must be a JSON object")
    return source, payload


def sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--evidence-binding", required=True, type=Path)
    parser.add_argument("--evidence-binding-sha256", required=True)
    parser.add_argument("--semantic-wrapper", required=True, type=Path)
    parser.add_argument("--semantic-wrapper-sha256", required=True)
    parser.add_argument("--physical-facts", required=True, type=Path)
    parser.add_argument("--physical-facts-sha256", required=True)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--membership-sha256", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--candidates-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def bh_with_total(p_values: np.ndarray, total: int) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = ranked * float(total) / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def require(name: str, condition: bool, observed: Any, expected: Any) -> None:
        record = {
            "name": name,
            "status": "PASS" if bool(condition) else "FAIL",
            "observed": observed,
            "expected": expected,
        }
        checks.append(record)
        if not condition:
            raise AuditError(
                f"{name} failed: observed={observed!r}, expected={expected!r}"
            )

    expected_hashes = {
        "release_manifest": args.release_manifest_sha256.lower(),
        "evidence_binding": args.evidence_binding_sha256.lower(),
        "semantic_wrapper": args.semantic_wrapper_sha256.lower(),
        "physical_facts": args.physical_facts_sha256.lower(),
        "membership": args.membership_sha256.lower(),
        "candidates": args.candidates_sha256.lower(),
    }
    for role, digest in expected_hashes.items():
        require(f"{role}_expected_sha256_well_formed", bool(SHA256.fullmatch(digest)), digest, "64 lowercase hex")

    manifest_path, manifest = read_json(args.release_manifest, "release manifest")
    binding_path, binding = read_json(args.evidence_binding, "Evidence binding")
    wrapper_path, wrapper = read_json(args.semantic_wrapper, "Evidence semantic wrapper")
    facts_path = args.physical_facts.resolve()
    membership_path = args.membership.resolve()
    candidates_path = args.candidates.resolve()
    source_paths = {
        "release_manifest": manifest_path,
        "evidence_binding": binding_path,
        "semantic_wrapper": wrapper_path,
        "physical_facts": facts_path,
        "membership": membership_path,
        "candidates": candidates_path,
    }
    observed_hashes = {role: artifact_sha256(path) for role, path in source_paths.items()}
    for role, digest in observed_hashes.items():
        require(f"{role}_sha256_pinned", digest == expected_hashes[role], digest, expected_hashes[role])

    required_manifest = {
        "format": RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "physical_interaction_release",
        "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
        "formal_authority_enforced": True,
        "release_ready": False,
        "production_deployed": False,
        "source_generation": "CURRENT_V3.2_RAW_FACTS_PLUS_EXACT_STATIC_MEMBERSHIP",
        "old_interaction_pathway_support_loaded": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "old_checkpoints_used": False,
        "family_to_exact_broadcast": False,
        "global_facts_broadcast_to_cancers": False,
    }
    for key, expected in required_manifest.items():
        require(f"manifest_{key}", manifest.get(key) == expected, manifest.get(key), expected)

    success_path, success = read_json(manifest_path.parent / "SUCCESS.json", "release success marker")
    require("success_status", success.get("status") == manifest["status"], success.get("status"), manifest["status"])
    require("success_release_ready_false", success.get("release_ready") is False, success.get("release_ready"), False)
    require("success_manifest_name", success.get("manifest") == manifest_path.name, success.get("manifest"), manifest_path.name)
    require("success_manifest_sha256", success.get("manifest_sha256") == observed_hashes["release_manifest"], success.get("manifest_sha256"), observed_hashes["release_manifest"])

    require("binding_format", binding.get("format") == BINDING_FORMAT, binding.get("format"), BINDING_FORMAT)
    require("binding_status", binding.get("status") == "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND", binding.get("status"), "SUCCESS_FRESH_EVIDENCE_OUTPUTS_HASH_BOUND")
    require("binding_five_fresh_heads", binding.get("five_fresh_private_heads_verified") is True, binding.get("five_fresh_private_heads_verified"), True)
    require("binding_interaction_eligible", binding.get("interaction_materialization_input_eligible") is True, binding.get("interaction_materialization_input_eligible"), True)
    require("binding_family_broadcast_false", binding.get("family_to_exact_broadcast") is False, binding.get("family_to_exact_broadcast"), False)
    require("binding_historical_outputs_false", not any(bool(binding.get(key)) for key in ("historical_predictions_used", "historical_rankings_used", "historical_checkpoints_used")), {key: binding.get(key) for key in ("historical_predictions_used", "historical_rankings_used", "historical_checkpoints_used")}, "all false")
    require("binding_optimizer_steps_positive", int(binding.get("optimizer_steps_total", 0)) > 0, binding.get("optimizer_steps_total"), "> 0")
    require("binding_five_checkpoint_keys", set(binding.get("checkpoints", {})) == {str(i) for i in range(5)}, sorted(binding.get("checkpoints", {})), [str(i) for i in range(5)])

    required_wrapper = {
        "format": WRAPPER_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS_HASH_BOUND_EVIDENCE_SEMANTICS",
        "release_ready": False,
        "production_deployed": False,
        "direct_target_evidence": True,
        "direct_target_scope": "LNCRNA_PARTNER_MAPPED_VIA_EXACT_MEMBER",
        "direct_exact_pathway_assertion": False,
        "confidence_only": True,
        "changes_primary_ranking": False,
        "affects_discovery": False,
        "affects_primary_ranking": False,
        "raw_prediction_direct_fusion_allowed": False,
        "canonical_candidate_adapter_required": True,
        "five_fresh_private_heads_verified": True,
    }
    for key, expected in required_wrapper.items():
        require(f"wrapper_{key}", wrapper.get(key) == expected, wrapper.get(key), expected)
    require("wrapper_binding_sha", wrapper.get("r2_evidence_binding", {}).get("sha256") == observed_hashes["evidence_binding"], wrapper.get("r2_evidence_binding", {}).get("sha256"), observed_hashes["evidence_binding"])
    require("wrapper_training_run_matches_binding", wrapper.get("evidence_training_run_id") == binding.get("evidence_training_run_id"), wrapper.get("evidence_training_run_id"), binding.get("evidence_training_run_id"))
    require("wrapper_optimizer_steps_match_binding", wrapper.get("optimizer_steps_total") == binding.get("optimizer_steps_total"), wrapper.get("optimizer_steps_total"), binding.get("optimizer_steps_total"))

    fresh_binding = manifest.get("fresh_evidence_output_binding", {})
    fresh_wrapper = manifest.get("fresh_evidence_semantic_wrapper", {})
    require("manifest_binding_sha", fresh_binding.get("sha256") == observed_hashes["evidence_binding"], fresh_binding.get("sha256"), observed_hashes["evidence_binding"])
    require("manifest_physical_facts_sha", fresh_binding.get("physical_facts_sha256") == observed_hashes["physical_facts"], fresh_binding.get("physical_facts_sha256"), observed_hashes["physical_facts"])
    require("manifest_wrapper_sha", fresh_wrapper.get("sha256") == observed_hashes["semantic_wrapper"], fresh_wrapper.get("sha256"), observed_hashes["semantic_wrapper"])
    require("manifest_wrapper_confidence_only", fresh_wrapper.get("confidence_only") is True and fresh_wrapper.get("affects_discovery") is False and fresh_wrapper.get("affects_primary_ranking") is False, {key: fresh_wrapper.get(key) for key in ("confidence_only", "affects_discovery", "affects_primary_ranking")}, {"confidence_only": True, "affects_discovery": False, "affects_primary_ranking": False})

    input_map = manifest.get("inputs", {})
    expected_inputs = {
        "physical_facts": (facts_path, observed_hashes["physical_facts"]),
        "exact_membership": (membership_path, observed_hashes["membership"]),
        "exact_candidates": (candidates_path, observed_hashes["candidates"]),
        "evidence_output_binding": (binding_path, observed_hashes["evidence_binding"]),
        "evidence_semantic_wrapper": (wrapper_path, observed_hashes["semantic_wrapper"]),
    }
    require("manifest_input_roles_exact", set(input_map) == set(expected_inputs), sorted(input_map), sorted(expected_inputs))
    for role, (expected_path, expected_sha) in expected_inputs.items():
        declaration = input_map.get(role, {})
        require(f"input_{role}_path", Path(str(declaration.get("path", ""))).resolve() == expected_path, str(Path(str(declaration.get("path", ""))).resolve()), str(expected_path))
        require(f"input_{role}_sha", declaration.get("sha256") == expected_sha, declaration.get("sha256"), expected_sha)

    output = args.output.resolve()
    if output.exists():
        raise AuditError(f"Audit output reuse is forbidden: {output}")
    output.mkdir(parents=True)
    temp_dir = output / ".duckdb_tmp"
    temp_dir.mkdir()
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"SET temp_directory={sql_path(temp_dir)}")
        declarations = manifest.get("artifacts", {})
        require("artifact_roles_exact", set(declarations) == set(EXPECTED_ARTIFACTS), sorted(declarations), sorted(EXPECTED_ARTIFACTS))
        artifact_paths: dict[str, Path] = {}
        artifact_counts: dict[str, int] = {}
        release_root = manifest_path.parent.resolve()
        for name, view in EXPECTED_ARTIFACTS.items():
            declaration = declarations[name]
            relative = Path(str(declaration.get("path", "")))
            require(f"artifact_{view}_path_relative", not relative.is_absolute(), str(relative), "relative")
            path = (release_root / relative).resolve()
            try:
                path.relative_to(release_root)
                contained = True
            except ValueError:
                contained = False
            require(f"artifact_{view}_contained", contained, str(path), str(release_root))
            require(f"artifact_{view}_role_name", path.name == name, path.name, name)
            digest = artifact_sha256(path)
            require(f"artifact_{view}_sha", digest == declaration.get("sha256"), digest, declaration.get("sha256"))
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet({sql_path(path)})")
            rows = int(con.execute(f"SELECT count(*) FROM {view}").fetchone()[0])
            require(f"artifact_{view}_rows", rows == int(declaration.get("rows", -1)), rows, declaration.get("rows"))
            artifact_paths[view] = path
            artifact_counts[view] = rows

        con.execute(f"CREATE VIEW facts_raw AS SELECT * FROM read_parquet({sql_path(facts_path)})")
        con.execute(f"CREATE VIEW membership_raw AS SELECT * FROM read_parquet({sql_path(membership_path)})")
        con.execute(f"CREATE VIEW candidates_raw AS SELECT * FROM read_parquet({sql_path(candidates_path)})")
        candidate_summary = con.execute(
            """
            SELECT count(*) AS row_count, count(DISTINCT upper(trim(cancer_id))) AS cancer_count,
                   count(DISTINCT trim(pathway_id)) pathways,
                   count(DISTINCT regexp_replace(upper(trim(lncrna_id)), '\\.[0-9]+$', '')) lncrnas,
                   count(*) - count(DISTINCT (upper(trim(cancer_id)), regexp_replace(upper(trim(lncrna_id)), '\\.[0-9]+$', ''), trim(pathway_id))) duplicate_keys
            FROM candidates_raw
            """
        ).fetchone()
        for name, observed, expected in (
            ("candidate_rows", int(candidate_summary[0]), FORMAL_CANDIDATE_ROWS),
            ("candidate_cancers", int(candidate_summary[1]), FORMAL_CANCERS),
            ("candidate_pathways", int(candidate_summary[2]), FORMAL_PATHWAYS),
            ("candidate_duplicate_keys", int(candidate_summary[4]), 0),
        ):
            require(name, observed == expected, observed, expected)
        require("candidate_lncrnas_match_manifest", int(candidate_summary[3]) == int(manifest["authority"]["candidate_lncrnas"]), int(candidate_summary[3]), manifest["authority"]["candidate_lncrnas"])

        stable_gene = "regexp_replace(regexp_replace(upper(trim(CAST(gene_id AS VARCHAR))), '^(GENE:|PROTEIN:|LNC:|LNCRNA:)', ''), '\\.[0-9]+$', '')"
        con.execute("CREATE TEMP VIEW candidate_pathways AS SELECT DISTINCT trim(pathway_id) pathway_id FROM candidates_raw")
        con.execute(
            f"""
            CREATE TEMP TABLE membership AS
            SELECT DISTINCT trim(m.pathway_id) pathway_id, {stable_gene} gene_id
            FROM membership_raw m JOIN candidate_pathways p ON trim(m.pathway_id)=p.pathway_id
            WHERE {stable_gene} <> ''
            """
        )
        membership_summary = con.execute("SELECT count(*), count(DISTINCT pathway_id), count(DISTINCT gene_id) FROM membership").fetchone()
        require("membership_edges", int(membership_summary[0]) == int(manifest["authority"]["exact_membership_edges"]), int(membership_summary[0]), manifest["authority"]["exact_membership_edges"])
        require("membership_pathways", int(membership_summary[1]) == FORMAL_PATHWAYS, int(membership_summary[1]), FORMAL_PATHWAYS)
        require("manifest_authority_rows", int(manifest["authority"]["candidate_rows"]) == FORMAL_CANDIDATE_ROWS, manifest["authority"]["candidate_rows"], FORMAL_CANDIDATE_ROWS)
        require("manifest_authority_cancers", int(manifest["authority"]["candidate_cancers"]) == FORMAL_CANCERS, manifest["authority"]["candidate_cancers"], FORMAL_CANCERS)

        facts_summary = con.execute(
            """
            SELECT count(*) AS row_count, count(DISTINCT trim(physical_fact_id)) AS unique_ids,
                   count(*) FILTER (WHERE physical_fact_id IS NULL OR trim(physical_fact_id)='') invalid_ids,
                   count(*) FILTER (WHERE coalesce(is_prediction, true)) predicted_rows
            FROM facts_raw
            """
        ).fetchone()
        require("physical_fact_rows", int(facts_summary[0]) == FORMAL_FACT_ROWS, int(facts_summary[0]), FORMAL_FACT_ROWS)
        require("physical_fact_ids_unique", int(facts_summary[1]) == FORMAL_FACT_ROWS, int(facts_summary[1]), FORMAL_FACT_ROWS)
        require("physical_fact_invalid_ids_zero", int(facts_summary[2]) == 0, int(facts_summary[2]), 0)
        require("physical_fact_predictions_zero", int(facts_summary[3]) == 0, int(facts_summary[3]), 0)
        facts_columns = {row[0] for row in con.execute("DESCRIBE facts_raw").fetchall()}
        require("physical_facts_have_no_pathway_assertion", "pathway_id" not in facts_columns, sorted(facts_columns), "pathway_id absent")

        require("relationship_evidence_full_fact_rows", artifact_counts["ev"] == FORMAL_FACT_ROWS, artifact_counts["ev"], FORMAL_FACT_ROWS)
        require("rejected_facts_zero", artifact_counts["rejected"] == 0, artifact_counts["rejected"], 0)
        output_bad = con.execute(
            """
            SELECT
              (SELECT count(*) FROM rel WHERE NOT coalesce(availability,false) OR coalesce(is_prediction,true) OR analysis_version<>?) bad_rel,
              (SELECT count(*) FROM ev WHERE coalesce(is_prediction,true) OR analysis_version<>? OR evidence_type<>'experimental_physical_interaction') bad_ev,
              (SELECT count(*) FROM enrich WHERE NOT coalesce(availability,false) OR coalesce(family_to_exact_broadcast,true) OR analysis_version<>? OR NOT isfinite(ora_p_value) OR NOT isfinite(ora_fdr) OR NOT isfinite(fold_enrichment) OR ora_p_value<0 OR ora_p_value>1 OR ora_fdr<0 OR ora_fdr>1) bad_enrich
            """,
            [ANALYSIS_VERSION, ANALYSIS_VERSION, ANALYSIS_VERSION],
        ).fetchone()
        require("relationship_semantics", int(output_bad[0]) == 0, int(output_bad[0]), 0)
        require("drilldown_semantics", int(output_bad[1]) == 0, int(output_bad[1]), 0)
        require("enrichment_semantics", int(output_bad[2]) == 0, int(output_bad[2]), 0)

        con.execute(
            f"""
            CREATE TEMP VIEW facts AS
            SELECT trim(physical_fact_id) physical_fact_id,
                   CASE WHEN cancer_id IS NULL OR upper(trim(CAST(cancer_id AS VARCHAR))) IN ('', 'NAN', 'NONE', 'NA', 'GLOBAL', 'PAN_CANCER')
                        THEN '{GLOBAL_SCOPE}' ELSE upper(trim(CAST(cancer_id AS VARCHAR))) END expected_scope,
                   'LNC:' || regexp_replace(regexp_replace(upper(trim(CAST(lncrna_id AS VARCHAR))), '^(GENE:|PROTEIN:|LNC:|LNCRNA:)', ''), '\\.[0-9]+$', '') expected_lnc,
                   regexp_replace(regexp_replace(upper(trim(CAST(partner_id AS VARCHAR))), '^(GENE:|PROTEIN:|LNC:|LNCRNA:)', ''), '\\.[0-9]+$', '') expected_partner
            FROM facts_raw
            """
        )
        fact_closure = con.execute(
            """
            SELECT
              count(*) FILTER (WHERE f.physical_fact_id IS NULL) extra_drilldown,
              count(*) FILTER (WHERE e.physical_fact_id IS NULL) missing_drilldown,
              count(*) FILTER (WHERE f.physical_fact_id IS NOT NULL AND e.physical_fact_id IS NOT NULL AND
                 (e.cancer_scope IS DISTINCT FROM f.expected_scope OR e.lncrna_id IS DISTINCT FROM f.expected_lnc OR e.partner_gene_id IS DISTINCT FROM f.expected_partner)) mapping_mismatch
            FROM facts f FULL OUTER JOIN ev e USING (physical_fact_id)
            """
        ).fetchone()
        require("fact_to_drilldown_extra_zero", int(fact_closure[0]) == 0, int(fact_closure[0]), 0)
        require("fact_to_drilldown_missing_zero", int(fact_closure[1]) == 0, int(fact_closure[1]), 0)
        require("fact_to_drilldown_canonical_mapping_zero", int(fact_closure[2]) == 0, int(fact_closure[2]), 0)
        ev_duplicates = int(con.execute("SELECT count(*)-count(DISTINCT physical_fact_id) FROM ev").fetchone()[0])
        require("drilldown_physical_fact_ids_unique", ev_duplicates == 0, ev_duplicates, 0)

        rel_key_audit = con.execute(
            """
            SELECT count(*)-count(DISTINCT relationship_id) duplicate_ids,
                   count(*)-count(DISTINCT (cancer_scope,lncrna_id,partner_gene_id)) duplicate_keys
            FROM rel
            """
        ).fetchone()
        require("relationship_ids_unique", int(rel_key_audit[0]) == 0, int(rel_key_audit[0]), 0)
        require("relationship_keys_unique", int(rel_key_audit[1]) == 0, int(rel_key_audit[1]), 0)
        con.execute(
            """
            CREATE TEMP TABLE expected_rel AS
            SELECT relationship_id, cancer_scope, lncrna_id, partner_gene_id,
                   count(DISTINCT physical_fact_id) physical_fact_count,
                   sum(source_occurrence_count) source_occurrence_count,
                   count(DISTINCT CASE WHEN trim(pmid)<>'' THEN pmid END) independent_pmid_count,
                   count(DISTINCT CASE WHEN trim(source_database)<>'' THEN source_database END) independent_source_database_count,
                   count(DISTINCT CASE WHEN trim(source_record_id)<>'' THEN source_record_id END) independent_source_record_count
            FROM ev GROUP BY ALL
            """
        )
        rel_closure = int(
            con.execute(
                """
                SELECT count(*) FROM expected_rel e FULL OUTER JOIN rel r USING (relationship_id)
                WHERE e.relationship_id IS NULL OR r.relationship_id IS NULL
                   OR e.cancer_scope IS DISTINCT FROM r.cancer_scope
                   OR e.lncrna_id IS DISTINCT FROM r.lncrna_id
                   OR e.partner_gene_id IS DISTINCT FROM r.partner_gene_id
                   OR e.physical_fact_count IS DISTINCT FROM r.physical_fact_count
                   OR e.source_occurrence_count IS DISTINCT FROM r.source_occurrence_count
                   OR e.independent_pmid_count IS DISTINCT FROM r.independent_pmid_count
                   OR e.independent_source_database_count IS DISTINCT FROM r.independent_source_database_count
                   OR e.independent_source_record_count IS DISTINCT FROM r.independent_source_record_count
                """
            ).fetchone()[0]
        )
        require("drilldown_to_relationship_group_closure", rel_closure == 0, rel_closure, 0)
        require("relationship_count_matches_manifest", artifact_counts["rel"] == int(manifest["counts"]["relationship_rows"]), artifact_counts["rel"], manifest["counts"]["relationship_rows"])

        rel_id_sample = con.execute("SELECT relationship_id,cancer_scope,lncrna_id,partner_gene_id FROM rel ORDER BY hash(relationship_id) LIMIT 50000").fetchall()
        bad_rel_ids = sum(
            relationship_id != "REL32:" + hashlib.sha256(f"{scope}\0{lnc}\0{partner}".encode("utf-8")).hexdigest()[:24]
            for relationship_id, scope, lnc, partner in rel_id_sample
        )
        require("relationship_id_deterministic_sample_50000", bad_rel_ids == 0, bad_rel_ids, 0)

        con.execute(
            """
            CREATE TEMP TABLE expected_enrich AS
            SELECT r.cancer_scope, r.lncrna_id, m.pathway_id,
                   count(DISTINCT r.partner_gene_id) overlap_target_count,
                   count(DISTINCT r.relationship_id) supporting_relationship_count
            FROM rel r JOIN membership m ON r.partner_gene_id=m.gene_id
            GROUP BY ALL
            """
        )
        expected_enrich_rows = int(con.execute("SELECT count(*) FROM expected_enrich").fetchone()[0])
        require("relationship_membership_to_enrichment_rows", expected_enrich_rows == artifact_counts["enrich"], expected_enrich_rows, artifact_counts["enrich"])
        enrich_key_audit = con.execute(
            """
            SELECT count(*)-count(DISTINCT enrichment_id) duplicate_ids,
                   count(*)-count(DISTINCT (cancer_scope,lncrna_id,pathway_id)) duplicate_keys,
                   count(DISTINCT pathway_id) pathways
            FROM enrich
            """
        ).fetchone()
        require("enrichment_ids_unique", int(enrich_key_audit[0]) == 0, int(enrich_key_audit[0]), 0)
        require("enrichment_keys_unique", int(enrich_key_audit[1]) == 0, int(enrich_key_audit[1]), 0)
        require("enrichment_exact_pathway_coverage", int(enrich_key_audit[2]) == FORMAL_PATHWAYS, int(enrich_key_audit[2]), FORMAL_PATHWAYS)
        enrich_closure = int(
            con.execute(
                """
                SELECT count(*) FROM expected_enrich x FULL OUTER JOIN enrich e
                  USING (cancer_scope,lncrna_id,pathway_id)
                WHERE x.pathway_id IS NULL OR e.pathway_id IS NULL
                   OR x.overlap_target_count IS DISTINCT FROM e.overlap_target_count
                   OR x.supporting_relationship_count IS DISTINCT FROM e.supporting_relationship_count
                """
            ).fetchone()[0]
        )
        require("enrichment_exact_key_and_count_closure", enrich_closure == 0, enrich_closure, 0)
        foreign_pathways = int(con.execute("SELECT count(*) FROM enrich e ANTI JOIN candidate_pathways p USING(pathway_id)").fetchone()[0])
        require("enrichment_foreign_pathways_zero", foreign_pathways == 0, foreign_pathways, 0)

        expected_unmapped = int(con.execute("SELECT count(*) FROM rel r ANTI JOIN (SELECT DISTINCT gene_id FROM membership) m ON r.partner_gene_id=m.gene_id").fetchone()[0])
        require("unmapped_expected_rows", expected_unmapped == artifact_counts["unmapped"], expected_unmapped, artifact_counts["unmapped"])
        unmapped_closure = int(
            con.execute(
                """
                WITH expected AS (
                  SELECT relationship_id FROM rel r
                  ANTI JOIN (SELECT DISTINCT gene_id FROM membership) m ON r.partner_gene_id=m.gene_id
                )
                SELECT count(*) FROM expected e FULL OUTER JOIN unmapped u USING(relationship_id)
                WHERE e.relationship_id IS NULL OR u.relationship_id IS NULL
                   OR u.mapping_status<>'PARTNER_OUTSIDE_EXACT_MEMBERSHIP_UNIVERSE'
                """
            ).fetchone()[0]
        )
        require("unmapped_relationship_key_closure", unmapped_closure == 0, unmapped_closure, 0)

        enrichment_sample = con.execute(
            """
            SELECT enrichment_id,cancer_scope,lncrna_id,pathway_id,overlap_target_count,
                   pathway_member_count,physical_target_count_in_universe,
                   membership_gene_universe_count,ora_p_value
            FROM enrich ORDER BY hash(enrichment_id) LIMIT 50000
            """
        ).fetchdf()
        expected_ids = [
            "INTPATH32:" + hashlib.sha256(f"{scope}\0{lnc}\0{pathway}".encode("utf-8")).hexdigest()[:24]
            for scope, lnc, pathway in enrichment_sample[["cancer_scope", "lncrna_id", "pathway_id"]].itertuples(index=False, name=None)
        ]
        bad_enrichment_ids = int((enrichment_sample.enrichment_id.to_numpy() != np.asarray(expected_ids)).sum())
        require("enrichment_id_deterministic_sample_50000", bad_enrichment_ids == 0, bad_enrichment_ids, 0)
        expected_p = hypergeom.sf(
            enrichment_sample.overlap_target_count.to_numpy(np.int64) - 1,
            enrichment_sample.membership_gene_universe_count.to_numpy(np.int64),
            enrichment_sample.pathway_member_count.to_numpy(np.int64),
            enrichment_sample.physical_target_count_in_universe.to_numpy(np.int64),
        )
        p_diff = float(np.max(np.abs(expected_p - enrichment_sample.ora_p_value.to_numpy(float))))
        require("ora_hypergeometric_sample_50000", bool(np.allclose(expected_p, enrichment_sample.ora_p_value.to_numpy(float), rtol=1e-10, atol=1e-14)), p_diff, "rtol<=1e-10, atol<=1e-14")

        audit_groups = con.execute(
            """
            SELECT cancer_scope,lncrna_id
            FROM (SELECT DISTINCT cancer_scope,lncrna_id FROM enrich)
            ORDER BY hash(cancer_scope || '|' || lncrna_id) LIMIT 64
            """
        ).fetchdf()
        con.register("audit_groups", audit_groups)
        fdr_sample = con.execute(
            """
            SELECT e.cancer_scope,e.lncrna_id,e.pathway_id,e.ora_p_value,e.ora_fdr
            FROM enrich e JOIN audit_groups g USING(cancer_scope,lncrna_id)
            ORDER BY e.cancer_scope,e.lncrna_id,e.pathway_id
            """
        ).fetchdf()
        max_fdr_diff = 0.0
        for _, part in fdr_sample.groupby(["cancer_scope", "lncrna_id"], sort=True):
            expected_fdr = bh_with_total(part.ora_p_value.to_numpy(float), FORMAL_PATHWAYS)
            max_fdr_diff = max(max_fdr_diff, float(np.max(np.abs(expected_fdr - part.ora_fdr.to_numpy(float)))))
        require("bh_fdr_64_deterministic_groups", max_fdr_diff <= 1e-12, max_fdr_diff, "<=1e-12")

        forbidden_columns: dict[str, list[str]] = {}
        for view in EXPECTED_ARTIFACTS.values():
            columns = {row[0] for row in con.execute(f"DESCRIBE {view}").fetchall()}
            bad = sorted(
                column for column in columns
                if column.lower() in {
                    "primary_probability", "primary_rank", "main_rank", "ranking",
                    "model_prediction", "evidence_confidence_probability",
                }
            )
            if bad:
                forbidden_columns[view] = bad
        require("no_primary_or_model_score_columns", not forbidden_columns, forbidden_columns, {})
    except Exception:
        con.close()
        shutil.rmtree(output, ignore_errors=True)
        raise
    else:
        con.close()

    shutil.rmtree(temp_dir, ignore_errors=True)
    audit = {
        "format": AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS",
        "release_ready": False,
        "production_deployed": False,
        "independent_of_materializer_implementation": True,
        "materializer_functions_imported": False,
        "main_ranking_modified": False,
        "prediction_role": "PHYSICAL_RELATIONSHIP_AND_EXACT_PATHWAY_ORA_SIDECAR",
        "direct_exact_pathway_assertion": False,
        "family_to_exact_broadcast": False,
        "global_facts_broadcast_to_cancers": False,
        "release_manifest": {"path": str(manifest_path), "sha256": observed_hashes["release_manifest"]},
        "release_success": {"path": str(success_path), "sha256": artifact_sha256(success_path)},
        "evidence_binding": {"path": str(binding_path), "sha256": observed_hashes["evidence_binding"]},
        "semantic_wrapper": {"path": str(wrapper_path), "sha256": observed_hashes["semantic_wrapper"]},
        "physical_facts": {"path": str(facts_path), "sha256": observed_hashes["physical_facts"], "rows": FORMAL_FACT_ROWS},
        "authority": dict(manifest["authority"]),
        "counts": dict(manifest["counts"]),
        "independent_statistics": {
            "membership_gene_universe_count": int(membership_summary[2]),
            "expected_enrichment_rows": expected_enrich_rows,
            "expected_unmapped_relationship_rows": expected_unmapped,
            "ora_sample_rows": int(len(enrichment_sample)),
            "bh_sample_groups": int(len(audit_groups)),
            "maximum_ora_absolute_difference": p_diff,
            "maximum_bh_fdr_absolute_difference": max_fdr_diff,
        },
        "checks": checks,
        "pass_count": len(checks),
        "fail_count": 0,
        "audit_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": artifact_sha256(Path(__file__).resolve()),
        },
    }
    audit_path = output / "PHYSICAL_INTERACTION_INDEPENDENT_AUDIT.json"
    temporary = output / ".PHYSICAL_INTERACTION_INDEPENDENT_AUDIT.json.tmp"
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    success_payload = {
        "status": "PASS",
        "release_ready": False,
        "audit": audit_path.name,
        "audit_sha256": artifact_sha256(audit_path),
        "pass_count": len(checks),
        "fail_count": 0,
    }
    success_output = output / "SUCCESS.json"
    success_output.write_text(json.dumps(success_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(success_payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
