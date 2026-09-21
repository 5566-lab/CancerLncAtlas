#!/usr/bin/env python
"""Independent full-closure audit of the unified V3.2 network release.

This auditor intentionally does not import ``network_release`` or
``network_query``.  It reconstructs expected nodes and edges directly from
the pinned current-V3.2 authorities and compares the public projection with
DuckDB set/semantic checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
from typing import Any


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_UNIFIED_NETWORK_RELEASE_V1"
RELEASE_STATUS = "SUCCESS_HASH_BOUND_CURRENT_V32_NETWORK"
AUDIT_FORMAT = "CC_HHGT_V3_2_UNIFIED_NETWORK_INDEPENDENT_AUDIT_V1"
AUDIT_BINDING_FORMAT = "CC_HHGT_V3_2_UNIFIED_NETWORK_INDEPENDENT_AUDIT_BINDING_V1"
MODEL = "MODEL_EXACT_PATHWAY_ASSOCIATION"
MEMBERSHIP = "EXACT_PATHWAY_GENE_MEMBERSHIP"
PHYSICAL = "PHYSICAL_INTERACTION"

PINS = {
    "primary": "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1",
    "exact_lineage": "892f13c7a0109d644f93806fefc5b254117cef0329285ad53c8d98107459ca08",
    "fusion_scores": "23690313401adaaea84a1f32d5d16590d5b7bae68b8f7ae0a436899afbecee0c",
    "fusion_binding": "0d2b5a34d0464016438725ef1e2fcea1ff04db611c9a03e571b01f87b780b059",
    "fusion_audit": "9bcedbdc186f044bb70eef96feb8d1ab04fd1b5382b0cfeeaa50758d8664c3cd",
    "membership": "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef",
    "candidate": "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f",
    "physical_relationships": "33e8f705477471fd4ed1f303b22d7e6a25d0d82fc9cff769e3e7fe178c7e3528",
    "physical_manifest": "30dbb6bb27c9f58ba21472d42635db0eeb95ed55c45fc8ae24e481e917f59f5c",
    "physical_audit": "90d1ab6221a8594084e248289910bce7c78d04d6c22731b9b73f6c27c73fe858",
    "experiment_bridge": "4716fac045bc4bf55f34751514dc5c262bdc7a90d71b79fe3540e4287693f9a8",
    "experiment_audit": "54e89c8ff9aa3fc06771d7c00702139d32c970d5bceaee81468bb59f2938f778",
}
EXPECTED_NODE_COUNTS = {"lncRNA": 10_465, "exact_pathway": 2_135, "protein_gene": 26_717}
EXPECTED_EDGE_COUNTS = {MODEL: 3_300_000, MEMBERSHIP: 352_205, PHYSICAL: 1_761_002}
EXPECTED_NATIVE_AVAILABLE = {
    "genomic": 747_408,
    "single_cell": 954_541,
    "evidence_transformer": 825_753,
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory-limit", default="8GB")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def add(name: str, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": expected,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )

    manifest_path = args.manifest.resolve()
    expected_manifest_sha = args.expected_manifest_sha256.lower()
    add("expected_manifest_sha256_well_formed", bool(_SHA.fullmatch(expected_manifest_sha)), True)
    add("manifest_is_regular_file", manifest_path.is_file() and not manifest_path.is_symlink(), True)
    manifest_sha = sha256(manifest_path)
    add("manifest_sha256_pinned", manifest_sha, expected_manifest_sha)
    manifest = load_json(manifest_path)
    expected_flags = {
        "format": RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "unified_network",
        "status": RELEASE_STATUS,
        "release_ready": False,
        "production_deployed": False,
        "formal_authority_enforced": True,
        "source_generation": "CURRENT_V3.2_ONLY",
        "primary_frozen": True,
        "primary_ranking_unchanged": True,
        "secondary_scores_remain_secondary": True,
        "native_expert_probabilities_public": True,
        "native_missingness_encoding": "availability_boolean_plus_nullable_probability",
        "missing_native_probability_imputed_to_zero": False,
        "family_to_exact_broadcast": False,
        "global_physical_facts_broadcast_to_cancers": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "experiment_route": "SUPPORT_ATTRIBUTE_ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER",
        "experiment_separate_fusion_created": False,
        "experiment_changes_primary_ranking": False,
    }
    for key, expected in expected_flags.items():
        add(f"manifest_{key}", manifest.get(key), expected)
    add("manifest_node_types", sorted(manifest.get("node_types", [])), sorted(EXPECTED_NODE_COUNTS))
    add("manifest_edge_types", sorted(manifest.get("edge_types", [])), sorted(EXPECTED_EDGE_COUNTS))
    counts = manifest.get("counts", {})
    add("manifest_node_rows", counts.get("nodes"), sum(EXPECTED_NODE_COUNTS.values()))
    add("manifest_edge_rows", counts.get("edges"), sum(EXPECTED_EDGE_COUNTS.values()))
    add("manifest_node_type_rows", counts.get("node_type_rows"), EXPECTED_NODE_COUNTS)
    add("manifest_edge_type_rows", counts.get("edge_type_rows"), EXPECTED_EDGE_COUNTS)

    inputs = manifest.get("inputs", {})
    add("manifest_input_roles", sorted(inputs), sorted([*PINS, "experiment_exact_query"]))
    input_paths: dict[str, Path] = {}
    for role, expected_sha in PINS.items():
        declaration = inputs.get(role, {})
        path = Path(str(declaration.get("path", ""))).resolve()
        input_paths[role] = path
        add(f"input_{role}_declared_sha", declaration.get("sha256"), expected_sha)
        add(f"input_{role}_regular_file", path.is_file() and not path.is_symlink(), True)
        add(f"input_{role}_rehash", sha256(path), expected_sha)
    experiment_query_decl = inputs.get("experiment_exact_query", {})
    experiment_query_path = Path(str(experiment_query_decl.get("path", ""))).resolve()
    input_paths["experiment_exact_query"] = experiment_query_path
    add("experiment_query_regular_file", experiment_query_path.is_file() and not experiment_query_path.is_symlink(), True)
    add("experiment_query_rehash", sha256(experiment_query_path), experiment_query_decl.get("sha256"))

    root = manifest_path.parent.resolve()
    artifacts = manifest.get("artifacts", {})
    artifact_paths: dict[str, Path] = {}
    for name in ("v32_network_nodes.parquet", "v32_network_edges.parquet"):
        declaration = artifacts.get(name, {})
        relative = Path(str(declaration.get("path", "")))
        add(f"artifact_{name}_relative", relative.is_absolute(), False)
        path = (root / relative).resolve()
        try:
            contained = path.relative_to(root) is not None
        except ValueError:
            contained = False
        add(f"artifact_{name}_contained", contained, True)
        add(f"artifact_{name}_regular_file", path.is_file() and not path.is_symlink(), True)
        add(f"artifact_{name}_rehash", sha256(path), declaration.get("sha256"))
        artifact_paths[name] = path

    success_path = root / "SUCCESS.json"
    success = load_json(success_path)
    add("success_status", success.get("status"), RELEASE_STATUS)
    add("success_release_ready", success.get("release_ready"), False)
    add("success_production_deployed", success.get("production_deployed"), False)
    add("success_manifest", success.get("manifest"), "NETWORK_RELEASE_MANIFEST.json")
    add("success_manifest_sha256", success.get("manifest_sha256"), manifest_sha)

    exact_lineage = load_json(input_paths["exact_lineage"])
    add("exact_lineage_status", exact_lineage.get("training_status"), "SUCCESS")
    add("exact_lineage_trained_from_scratch", exact_lineage.get("trained_from_scratch"), True)
    add("exact_lineage_five_folds", exact_lineage.get("folds"), 5)
    add("exact_lineage_prediction_sha", exact_lineage.get("prediction_sha256"), PINS["primary"])
    add("exact_lineage_old_checkpoint", exact_lineage.get("old_checkpoint_loaded"), False)
    add("exact_lineage_old_predictions", exact_lineage.get("old_predictions_used_as_features"), False)
    add("exact_lineage_old_rankings", exact_lineage.get("old_rankings_used_as_outputs"), False)
    fusion_binding = load_json(input_paths["fusion_binding"])
    add("fusion_binding_status", fusion_binding.get("status"), "SUCCESS_SECONDARY_FUSION")
    add("fusion_binding_primary_score_preserved", fusion_binding.get("primary_score_preserved"), True)
    add("fusion_binding_primary_ranking_unchanged", fusion_binding.get("primary_ranking_unchanged"), True)
    add("fusion_binding_native_scores_public", fusion_binding.get("native_expert_probabilities_public"), True)
    fusion_audit = load_json(input_paths["fusion_audit"])
    add("fusion_independent_audit_status", fusion_audit.get("status"), "PASS")
    add("fusion_independent_audit_passes", fusion_audit.get("pass_count"), 91)
    add("fusion_independent_audit_failures", fusion_audit.get("fail_count"), 0)
    add("fusion_api_accepted", fusion_audit.get("accepted_for_api_integration"), True)
    physical_manifest = load_json(input_paths["physical_manifest"])
    add("physical_manifest_status", physical_manifest.get("status"), "SUCCESS_NEWLY_MATERIALIZED_V32")
    add("physical_manifest_family_broadcast", physical_manifest.get("family_to_exact_broadcast"), False)
    add("physical_manifest_cancer_broadcast", physical_manifest.get("global_facts_broadcast_to_cancers"), False)
    physical_audit = load_json(input_paths["physical_audit"])
    add("physical_independent_audit_status", physical_audit.get("status"), "PASS")
    add("physical_independent_audit_passes", physical_audit.get("pass_count"), 137)
    add("physical_independent_audit_failures", physical_audit.get("fail_count"), 0)
    experiment_bridge = load_json(input_paths["experiment_bridge"])
    add("experiment_bridge_status", experiment_bridge.get("status"), "SUCCESS_HASH_BOUND_ABSORPTION_BRIDGE")
    add("experiment_bridge_absorbed", experiment_bridge.get("fully_absorbed_by_evidence_transformer"), True)
    add("experiment_bridge_separate_fusion_forbidden", experiment_bridge.get("separate_fusion_forbidden"), True)
    experiment_audit_binding = load_json(input_paths["experiment_audit"])
    experiment_report_path = Path(str(experiment_audit_binding.get("report", {}).get("path", ""))).resolve()
    add("experiment_audit_binding_status", experiment_audit_binding.get("status"), "PASS")
    add("experiment_audit_report_rehash", sha256(experiment_report_path), experiment_audit_binding.get("report", {}).get("sha256"))
    experiment_report = load_json(experiment_report_path)
    experiment_checks = experiment_report.get("checks", [])
    add("experiment_independent_audit_check_count", len(experiment_checks), 49)
    add("experiment_independent_audit_all_pass", sum(item.get("status") == "PASS" for item in experiment_checks), 49)

    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required") from exc
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"audit output target already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise RuntimeError(f"audit staging target already exists: {staging}")
    staging.mkdir()
    temp = staging / ".duckdb_tmp"
    temp.mkdir()
    con = duckdb.connect(str(staging / ".audit.duckdb"))
    try:
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute(f"SET temp_directory={sql_path(temp)}")
        con.execute("SET preserve_insertion_order=false")
        views = {
            "nodes": artifact_paths["v32_network_nodes.parquet"],
            "edges": artifact_paths["v32_network_edges.parquet"],
            "primary_scores": input_paths["primary"],
            "fusion": input_paths["fusion_scores"],
            "raw_membership": input_paths["membership"],
            "candidate": input_paths["candidate"],
            "physical": input_paths["physical_relationships"],
            "experiment": input_paths["experiment_exact_query"],
        }
        for name, path in views.items():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})")
        con.execute(
            """
            CREATE TEMP VIEW exact_membership AS
            SELECT DISTINCT trim(m.pathway_id) pathway_id,
              regexp_replace(regexp_replace(upper(trim(m.gene_id)),
                '^(GENE:|PROTEIN:)', ''), '\\.[0-9]+$', '') gene_id
            FROM raw_membership m
            JOIN (SELECT DISTINCT pathway_id FROM candidate) p
              ON trim(m.pathway_id)=p.pathway_id
            WHERE trim(m.pathway_id)<>'' AND trim(m.gene_id)<>''
            """
        )
        con.execute(
            """
            CREATE TEMP VIEW expected_nodes AS
            SELECT entity_id node_id, entity_id, 'lncRNA' node_type FROM (
              SELECT lncrna_id entity_id FROM candidate
              UNION SELECT lncrna_id FROM physical
            ) GROUP BY entity_id
            UNION ALL
            SELECT 'PATHWAY:'||pathway_id, pathway_id, 'exact_pathway'
              FROM candidate GROUP BY pathway_id
            UNION ALL
            SELECT 'GENE:'||entity_id, entity_id, 'protein_gene' FROM (
              SELECT gene_id entity_id FROM exact_membership
              UNION SELECT partner_gene_id FROM physical
            ) GROUP BY entity_id
            """
        )
        simple = con.execute(
            """
            SELECT
              (SELECT count(*) FROM nodes), (SELECT count(*) FROM edges),
              (SELECT count(*)-count(DISTINCT node_id) FROM nodes),
              (SELECT count(*)-count(DISTINCT edge_id) FROM edges),
              (SELECT count(*) FROM expected_nodes),
              (SELECT count(*) FROM (SELECT node_id,entity_id,node_type FROM expected_nodes
                 EXCEPT SELECT node_id,entity_id,node_type FROM nodes)),
              (SELECT count(*) FROM (SELECT node_id,entity_id,node_type FROM nodes
                 EXCEPT SELECT node_id,entity_id,node_type FROM expected_nodes)),
              (SELECT count(*) FROM edges e LEFT JOIN nodes n
                 ON e.source_node_id=n.node_id WHERE n.node_id IS NULL),
              (SELECT count(*) FROM edges e LEFT JOIN nodes n
                 ON e.target_node_id=n.node_id WHERE n.node_id IS NULL),
              (SELECT count(*) FROM nodes WHERE analysis_version<>?
                 OR node_id IS NULL OR trim(node_id)=''),
              (SELECT count(*) FROM edges WHERE analysis_version<>?
                 OR family_to_exact_broadcast OR changes_primary_ranking),
              (SELECT count(*) FROM edges WHERE edge_type NOT IN (?,?,?)),
              (SELECT count(*) FROM edges WHERE edge_type=? AND NOT is_prediction),
              (SELECT count(*) FROM edges WHERE edge_type<>? AND is_prediction),
              (SELECT count(*) FROM edges WHERE edge_type=? AND
                 (NOT primary_ranking_unchanged OR NOT adjusted_ranking_is_secondary
                   OR used_for_primary_release)),
              (SELECT count(*) FROM edges WHERE edge_type=? AND cancer_id IS NOT NULL),
              (SELECT count(*) FROM edges WHERE edge_type=? AND cancer_id IS NOT NULL)
            """,
            [ANALYSIS_VERSION, ANALYSIS_VERSION, MODEL, MEMBERSHIP, PHYSICAL,
             MODEL, MODEL, MODEL, MEMBERSHIP, PHYSICAL],
        ).fetchone()
        names = (
            "node_rows", "edge_rows", "duplicate_node_ids", "duplicate_edge_ids",
            "expected_node_rows", "missing_expected_nodes", "extra_nodes",
            "orphan_source_nodes", "orphan_target_nodes", "bad_node_semantics",
            "bad_edge_semantics", "unknown_edge_types", "model_not_prediction",
            "nonmodel_predictions", "bad_model_ranking_flags",
            "membership_has_cancer", "physical_has_cancer",
        )
        expected = (
            sum(EXPECTED_NODE_COUNTS.values()), sum(EXPECTED_EDGE_COUNTS.values()),
            0, 0, sum(EXPECTED_NODE_COUNTS.values()), 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        )
        for name, observed, wanted in zip(names, map(int, simple), expected):
            add(name, observed, wanted)
        node_counts = {row[0]: int(row[1]) for row in con.execute(
            "SELECT node_type,count(*) FROM nodes GROUP BY node_type").fetchall()}
        edge_counts = {row[0]: int(row[1]) for row in con.execute(
            "SELECT edge_type,count(*) FROM edges GROUP BY edge_type").fetchall()}
        add("node_type_counts", node_counts, EXPECTED_NODE_COUNTS)
        add("edge_type_counts", edge_counts, EXPECTED_EDGE_COUNTS)
        authority = con.execute(
            """
            SELECT
              (SELECT count(*) FROM candidate),
              (SELECT count(DISTINCT cancer_id) FROM candidate),
              (SELECT count(DISTINCT lncrna_id) FROM candidate),
              (SELECT count(DISTINCT pathway_id) FROM candidate),
              (SELECT count(*)-count(DISTINCT (cancer_id,lncrna_id,pathway_id)) FROM candidate),
              (SELECT count(*) FROM exact_membership),
              (SELECT count(DISTINCT pathway_id) FROM exact_membership),
              (SELECT count(DISTINCT gene_id) FROM exact_membership),
              (SELECT count(*) FROM physical),
              (SELECT count(*)-count(DISTINCT relationship_id) FROM physical),
              (SELECT count(*) FROM experiment),
              (SELECT count(*)-count(DISTINCT (cancer_id,lncrna_id,pathway_id)) FROM experiment)
            """
        ).fetchone()
        authority_names = (
            "candidate_rows", "candidate_cancers", "candidate_lncrnas",
            "candidate_pathways", "candidate_duplicate_keys", "exact_membership_rows",
            "exact_membership_pathways", "exact_membership_genes", "physical_rows",
            "physical_duplicate_relationship_ids", "experiment_supported_exact_keys",
            "experiment_duplicate_keys",
        )
        authority_expected = (
            3_300_000, 33, 8_541, 2_135, 0, 352_205, 2_135, 20_439,
            1_761_002, 0, 14_452, 0,
        )
        for name, observed, wanted in zip(authority_names, map(int, authority), authority_expected):
            add(name, observed, wanted)
        native = con.execute(
            """
            SELECT
              count(*) FILTER(WHERE edge_type=? AND genomic_native_available),
              count(*) FILTER(WHERE edge_type=? AND single_cell_native_available),
              count(*) FILTER(WHERE edge_type=? AND evidence_transformer_native_available),
              count(*) FILTER(WHERE edge_type=? AND (
                (genomic_native_available AND genomic_native_probability IS NULL)
                OR (NOT genomic_native_available AND genomic_native_probability IS NOT NULL)
                OR (single_cell_native_available AND single_cell_native_probability IS NULL)
                OR (NOT single_cell_native_available AND single_cell_native_probability IS NOT NULL)
                OR (evidence_transformer_native_available AND evidence_transformer_native_probability IS NULL)
                OR (NOT evidence_transformer_native_available AND evidence_transformer_native_probability IS NOT NULL))),
              count(*) FILTER(WHERE edge_type<>? AND (
                genomic_native_available IS NOT NULL OR genomic_native_probability IS NOT NULL
                OR single_cell_native_available IS NOT NULL OR single_cell_native_probability IS NOT NULL
                OR evidence_transformer_native_available IS NOT NULL
                OR evidence_transformer_native_probability IS NOT NULL))
            FROM edges
            """,
            [MODEL, MODEL, MODEL, MODEL, MODEL],
        ).fetchone()
        native_names = (
            "genomic_native_available_rows", "single_cell_native_available_rows",
            "evidence_native_available_rows", "native_missingness_violations",
            "native_scores_on_nonmodel_edges",
        )
        native_expected = (
            EXPECTED_NATIVE_AVAILABLE["genomic"], EXPECTED_NATIVE_AVAILABLE["single_cell"],
            EXPECTED_NATIVE_AVAILABLE["evidence_transformer"], 0, 0,
        )
        for name, observed, wanted in zip(native_names, map(int, native), native_expected):
            add(name, observed, wanted)
        model_closure = con.execute(
            """
            SELECT
              (SELECT count(*) FROM (
                SELECT cancer_id,lncrna_id,pathway_id FROM fusion
                EXCEPT SELECT cancer_id,lncrna_id,pathway_id FROM edges WHERE edge_type=?)),
              (SELECT count(*) FROM (
                SELECT cancer_id,lncrna_id,pathway_id FROM edges WHERE edge_type=?
                EXCEPT SELECT cancer_id,lncrna_id,pathway_id FROM fusion)),
              (SELECT count(*) FROM edges e JOIN fusion f USING(cancer_id,lncrna_id,pathway_id)
                WHERE e.edge_type=? AND (
                  e.primary_probability IS DISTINCT FROM f.primary_probability
                  OR e.discovery_adjusted_probability IS DISTINCT FROM f.discovery_adjusted_probability
                  OR e.fused_confidence_probability IS DISTINCT FROM f.fused_confidence_probability
                  OR e.genomic_native_available IS DISTINCT FROM f.genomic_native_available
                  OR e.genomic_native_probability IS DISTINCT FROM f.genomic_native_probability
                  OR e.single_cell_native_available IS DISTINCT FROM f.single_cell_native_available
                  OR e.single_cell_native_probability IS DISTINCT FROM f.single_cell_native_probability
                  OR e.evidence_transformer_native_available IS DISTINCT FROM f.evidence_transformer_native_available
                  OR e.evidence_transformer_native_probability IS DISTINCT FROM f.evidence_transformer_native_probability)),
              (SELECT count(*) FROM edges e JOIN primary_scores p USING(cancer_id,lncrna_id,pathway_id)
                WHERE e.edge_type=? AND e.primary_probability IS DISTINCT FROM
                  CAST(p.association_membership_probability AS DOUBLE)),
              (SELECT count(*) FROM edges WHERE edge_type=? AND edge_id <>
                'NET32:M:'||substr(sha256(cancer_id||chr(0)||lncrna_id||chr(0)||pathway_id),1,24))
            """,
            [MODEL, MODEL, MODEL, MODEL, MODEL],
        ).fetchone()
        for name, observed in zip(
            ("model_missing_keys", "model_extra_keys", "model_fusion_value_mismatch",
             "model_primary_value_mismatch", "model_edge_id_mismatch"),
            map(int, model_closure),
        ):
            add(name, observed, 0)
        membership_closure = con.execute(
            """
            SELECT
              (SELECT count(*) FROM (
                SELECT pathway_id,gene_id protein_gene_id FROM exact_membership
                EXCEPT SELECT pathway_id,protein_gene_id FROM edges WHERE edge_type=?)),
              (SELECT count(*) FROM (
                SELECT pathway_id,protein_gene_id FROM edges WHERE edge_type=?
                EXCEPT SELECT pathway_id,gene_id FROM exact_membership)),
              (SELECT count(*) FROM edges WHERE edge_type=? AND edge_id <>
                'NET32:G:'||substr(sha256(pathway_id||chr(0)||protein_gene_id),1,24)),
              (SELECT count(*) FROM edges WHERE edge_type=? AND (
                source_node_id<>'PATHWAY:'||pathway_id
                OR target_node_id<>'GENE:'||protein_gene_id OR NOT directed))
            """,
            [MEMBERSHIP, MEMBERSHIP, MEMBERSHIP, MEMBERSHIP],
        ).fetchone()
        for name, observed in zip(
            ("membership_missing_edges", "membership_extra_edges",
             "membership_edge_id_mismatch", "membership_endpoint_mismatch"),
            map(int, membership_closure),
        ):
            add(name, observed, 0)
        physical_closure = con.execute(
            """
            SELECT
              (SELECT count(*) FROM (
                SELECT relationship_id FROM physical
                EXCEPT SELECT substr(edge_id,9) FROM edges WHERE edge_type=?)),
              (SELECT count(*) FROM (
                SELECT substr(edge_id,9) relationship_id FROM edges WHERE edge_type=?
                EXCEPT SELECT relationship_id FROM physical)),
              (SELECT count(*) FROM edges e JOIN physical p
                ON e.edge_type=? AND e.edge_id='NET32:P:'||p.relationship_id
                WHERE e.source_node_id<>p.lncrna_id
                  OR e.target_node_id<>'GENE:'||p.partner_gene_id
                  OR e.cancer_scope IS DISTINCT FROM p.cancer_scope
                  OR e.physical_fact_count IS DISTINCT FROM p.physical_fact_count
                  OR e.source_occurrence_count IS DISTINCT FROM p.source_occurrence_count
                  OR e.independent_pmid_count IS DISTINCT FROM p.independent_pmid_count
                  OR e.independent_source_database_count IS DISTINCT FROM p.independent_source_database_count
                  OR e.independent_source_record_count IS DISTINCT FROM p.independent_source_record_count
                  OR e.directed OR e.is_prediction OR NOT e.availability),
              (SELECT count(*) FROM edges WHERE edge_type=? AND edge_id<>
                'NET32:P:'||substr(edge_id,9)),
              (SELECT count(DISTINCT cancer_scope) FROM edges WHERE edge_type=?),
              (SELECT count(*) FROM edges WHERE edge_type=?
                AND cancer_scope<>'GLOBAL_NOT_CANCER_SPECIFIC')
            """,
            [PHYSICAL, PHYSICAL, PHYSICAL, PHYSICAL, PHYSICAL, PHYSICAL],
        ).fetchone()
        physical_names = (
            "physical_missing_relationships", "physical_extra_relationships",
            "physical_value_or_endpoint_mismatch", "physical_edge_id_mismatch",
            "physical_scope_count", "physical_non_global_rows",
        )
        physical_expected = (0, 0, 0, 0, 1, 0)
        for name, observed, wanted in zip(physical_names, map(int, physical_closure), physical_expected):
            add(name, observed, wanted)
        experiment_closure = con.execute(
            """
            SELECT
              count(*) FILTER(WHERE edge_type=? AND experiment_support_available),
              count(*) FILTER(WHERE edge_type=? AND experiment_support_available AND (
                experiment_event_count IS NULL OR experiment_event_count<=0
                OR experiment_source_record_count IS NULL OR experiment_source_record_count<=0
                OR experiment_support_role<>'ABSORBED_IN_EVIDENCE_TRANSFORMER_NO_SEPARATE_FUSION')),
              count(*) FILTER(WHERE NOT experiment_support_available AND (
                experiment_event_count IS NOT NULL OR experiment_source_record_count IS NOT NULL
                OR experiment_support_role IS NOT NULL)),
              (SELECT count(*) FROM experiment x LEFT JOIN edges e
                ON e.edge_type=? AND e.cancer_id=x.cancer_id AND e.lncrna_id=x.lncrna_id
                  AND e.pathway_id=x.pathway_id
                WHERE e.edge_id IS NULL OR NOT e.experiment_support_available
                  OR e.experiment_event_count IS DISTINCT FROM x.experiment_event_count
                  OR e.experiment_source_record_count IS DISTINCT FROM x.source_record_count),
              (SELECT count(*) FROM edges e LEFT JOIN experiment x
                ON e.cancer_id=x.cancer_id AND e.lncrna_id=x.lncrna_id
                  AND e.pathway_id=x.pathway_id
                WHERE e.experiment_support_available AND x.cancer_id IS NULL)
            FROM edges
            """,
            [MODEL, MODEL, MODEL],
        ).fetchone()
        for name, observed, wanted in zip(
            ("experiment_support_rows", "bad_experiment_support_semantics",
             "unsupported_rows_with_experiment_values", "experiment_bridge_value_mismatch",
             "experiment_support_extra_keys"),
            map(int, experiment_closure),
            (14_452, 0, 0, 0, 0),
        ):
            add(name, observed, wanted)
        edge_columns = {row[0] for row in con.execute("DESCRIBE SELECT * FROM edges").fetchall()}
        forbidden_experiment_score_columns = sorted(
            column for column in edge_columns
            if "experiment" in column.lower()
            and any(token in column.lower() for token in ("probability", "logit", "fusion_weight"))
        )
        add("no_separate_experiment_score_columns", forbidden_experiment_score_columns, [])
    finally:
        con.close()
        (staging / ".audit.duckdb").unlink(missing_ok=True)
        shutil.rmtree(temp, ignore_errors=True)

    fail_count = sum(item["status"] == "FAIL" for item in checks)
    pass_count = len(checks) - fail_count
    report = {
        "format": AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS" if fail_count == 0 else "FAIL",
        "independent_of_materializer_implementation": True,
        "materializer_modules_imported": False,
        "release_ready": False,
        "production_deployed": False,
        "accepted_for_api_integration": fail_count == 0,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "checks": checks,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "counts": {
            "nodes": sum(EXPECTED_NODE_COUNTS.values()),
            "edges": sum(EXPECTED_EDGE_COUNTS.values()),
            "node_type_rows": EXPECTED_NODE_COUNTS,
            "edge_type_rows": EXPECTED_EDGE_COUNTS,
            "experiment_support_rows": 14_452,
        },
        "invariants": {
            "primary_probability_bitwise_preserved": True,
            "primary_ranking_modified": False,
            "native_missingness_to_zero": False,
            "family_to_exact_broadcast": False,
            "global_physical_facts_broadcast_to_cancers": False,
            "experiment_separate_fusion_created": False,
            "experiment_double_counted": False,
        },
    }
    report_path = staging / "NETWORK_INDEPENDENT_AUDIT_REPORT.json"
    write_json(report, report_path)
    auditor_path = Path(__file__).resolve()
    binding = {
        "format": AUDIT_BINDING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": report["status"],
        "release_ready": False,
        "production_deployed": False,
        "accepted_for_api_integration": fail_count == 0,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "network_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "report": {"path": str(output / report_path.name), "sha256": sha256(report_path)},
        "auditor": {"path": str(auditor_path), "sha256": sha256(auditor_path)},
        "primary_probability_bitwise_preserved": fail_count == 0,
        "native_missingness_to_zero": False,
        "family_to_exact_broadcast": False,
        "experiment_separate_fusion_created": False,
    }
    binding_path = staging / "NETWORK_INDEPENDENT_AUDIT_BINDING.json"
    write_json(binding, binding_path)
    if fail_count:
        os.replace(staging, output)
        raise SystemExit(f"network independent audit failed: {fail_count}/{len(checks)}")
    success_payload = {
        "status": "PASS",
        "release_ready": False,
        "production_deployed": False,
        "binding": binding_path.name,
        "binding_sha256": sha256(binding_path),
        "report": report_path.name,
        "report_sha256": sha256(report_path),
        "pass_count": pass_count,
        "fail_count": 0,
    }
    write_json(success_payload, staging / "SUCCESS.json")
    os.replace(staging, output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "pass_count": pass_count,
                "fail_count": 0,
                "report_sha256": sha256(output / report_path.name),
                "binding_sha256": sha256(output / binding_path.name),
                "success_sha256": sha256(output / "SUCCESS.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
