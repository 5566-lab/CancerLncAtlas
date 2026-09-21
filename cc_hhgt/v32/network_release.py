"""Hash-bound unified V3.2 network release.

The network is a read-only projection of current V3.2 authorities.  It does
not train, recalibrate or overwrite the exact-pathway primary result.  Model
association edges retain native genomic, single-cell and Evidence Transformer
probabilities with explicit availability booleans; unavailable probabilities
remain null.  Experiment perturbation is exposed only as support already
absorbed by the Evidence Transformer and never as a second fusion head.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .input_lineage import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_UNIFIED_NETWORK_RELEASE_V1"
RELEASE_STATUS = "SUCCESS_HASH_BOUND_CURRENT_V32_NETWORK"
NETWORK_NODE_FILE = "v32_network_nodes.parquet"
NETWORK_EDGE_FILE = "v32_network_edges.parquet"
MANIFEST_FILE = "NETWORK_RELEASE_MANIFEST.json"
SUCCESS_FILE = "SUCCESS.json"

MODEL_EDGE_TYPE = "MODEL_EXACT_PATHWAY_ASSOCIATION"
MEMBERSHIP_EDGE_TYPE = "EXACT_PATHWAY_GENE_MEMBERSHIP"
PHYSICAL_EDGE_TYPE = "PHYSICAL_INTERACTION"
EDGE_TYPES = (MODEL_EDGE_TYPE, MEMBERSHIP_EDGE_TYPE, PHYSICAL_EDGE_TYPE)
NODE_TYPES = ("lncRNA", "exact_pathway", "protein_gene")

FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_CANCER_COUNT = 33
FORMAL_EXACT_PATHWAY_COUNT = 2_135
FORMAL_EXACT_MEMBERSHIP_ROWS = 352_205
FORMAL_PHYSICAL_RELATIONSHIP_ROWS = 1_761_002

FORMAL_PRIMARY_SHA256 = (
    "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1"
)
FORMAL_EXACT_LINEAGE_SHA256 = (
    "892f13c7a0109d644f93806fefc5b254117cef0329285ad53c8d98107459ca08"
)
FORMAL_FUSION_SCORES_SHA256 = (
    "23690313401adaaea84a1f32d5d16590d5b7bae68b8f7ae0a436899afbecee0c"
)
FORMAL_FUSION_BINDING_SHA256 = (
    "0d2b5a34d0464016438725ef1e2fcea1ff04db611c9a03e571b01f87b780b059"
)
FORMAL_FUSION_AUDIT_SHA256 = (
    "9bcedbdc186f044bb70eef96feb8d1ab04fd1b5382b0cfeeaa50758d8664c3cd"
)
FORMAL_PHYSICAL_MANIFEST_SHA256 = (
    "30dbb6bb27c9f58ba21472d42635db0eeb95ed55c45fc8ae24e481e917f59f5c"
)
FORMAL_PHYSICAL_AUDIT_SHA256 = (
    "90d1ab6221a8594084e248289910bce7c78d04d6c22731b9b73f6c27c73fe858"
)
FORMAL_PHYSICAL_RELATIONSHIPS_SHA256 = (
    "33e8f705477471fd4ed1f303b22d7e6a25d0d82fc9cff769e3e7fe178c7e3528"
)
FORMAL_MEMBERSHIP_SHA256 = (
    "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
)
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_EXPERIMENT_BRIDGE_SHA256 = (
    "4716fac045bc4bf55f34751514dc5c262bdc7a90d71b79fe3540e4287693f9a8"
)
FORMAL_EXPERIMENT_AUDIT_SHA256 = (
    "54e89c8ff9aa3fc06771d7c00702139d32c970d5bceaee81468bb59f2938f778"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NetworkReleaseError(RuntimeError):
    """Raised when a network input, projection or release contract fails."""


def _json(path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NetworkReleaseError(f"{role} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NetworkReleaseError(f"{role} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise NetworkReleaseError(f"{role} must be a JSON object: {path}")
    return value


def _file(path: str | Path, role: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise NetworkReleaseError(f"{role} is missing or unsafe: {source}")
    return source


def _bound_file(
    path: str | Path,
    role: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = _file(path, role)
    observed = artifact_sha256(source)
    if expected_sha256 is not None and observed != expected_sha256:
        raise NetworkReleaseError(
            f"{role} SHA256 mismatch: {observed} != {expected_sha256}"
        )
    return {"path": str(source), "sha256": observed}


def _check(payload: Mapping[str, Any], expected: Mapping[str, Any], role: str) -> None:
    for key, wanted in expected.items():
        observed = payload.get(key)
        if observed != wanted:
            raise NetworkReleaseError(
                f"{role} has invalid {key}: {observed!r} != {wanted!r}"
            )


def _declared_artifact(
    declaration: Any,
    *,
    expected_path: Path,
    expected_sha256: str,
    role: str,
    base: Path | None = None,
) -> None:
    if not isinstance(declaration, Mapping):
        raise NetworkReleaseError(f"{role} declaration is missing")
    declared_path = Path(str(declaration.get("path", "")))
    if not declared_path.is_absolute() and base is not None:
        declared_path = base / declared_path
    declared_path = declared_path.resolve()
    if declared_path != expected_path.resolve():
        raise NetworkReleaseError(
            f"{role} path mismatch: {declared_path} != {expected_path.resolve()}"
        )
    if declaration.get("sha256") != expected_sha256:
        raise NetworkReleaseError(f"{role} declared SHA256 mismatch")


def _validate_formal_bindings(paths: Mapping[str, Path]) -> None:
    exact = _json(paths["exact_lineage"], "exact-pathway lineage")
    _check(
        exact,
        {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "exact_pathway",
            "training_status": "SUCCESS",
            "trained_from_scratch": True,
            "five_fold_ensemble": True,
            "folds": 5,
            "prediction_rows": FORMAL_CANDIDATE_ROWS,
            "prediction_sha256": FORMAL_PRIMARY_SHA256,
            "old_checkpoint_loaded": False,
            "old_predictions_used_as_features": False,
            "old_rankings_used_as_outputs": False,
        },
        "exact-pathway lineage",
    )
    if Path(str(exact.get("prediction_path", ""))).resolve() != paths["primary"]:
        raise NetworkReleaseError("exact lineage points to a different primary table")

    fusion = _json(paths["fusion_binding"], "multimodal fusion binding")
    _check(
        fusion,
        {
            "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_SECONDARY_FUSION",
            "release_ready": False,
            "production_deployed": False,
            "primary_score_preserved": True,
            "primary_ranking_unchanged": True,
            "adjusted_ranking_is_secondary": True,
            "zero_total_contribution_exact_primary_fallback": True,
            "native_expert_probabilities_public": True,
            "public_rows": FORMAL_CANDIDATE_ROWS,
        },
        "multimodal fusion binding",
    )
    _declared_artifact(
        fusion.get("artifacts", {}).get("secondary_scores")
        if isinstance(fusion.get("artifacts"), Mapping)
        else None,
        expected_path=paths["fusion_scores"],
        expected_sha256=FORMAL_FUSION_SCORES_SHA256,
        role="fusion secondary scores",
    )

    fusion_audit = _json(paths["fusion_audit"], "multimodal fusion audit")
    _check(
        fusion_audit,
        {
            "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_R2_TRANSPARENT_INDEPENDENT_POST_AUDIT_BINDING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "fail_count": 0,
            "pass_count": 91,
            "accepted_for_api_integration": True,
            "native_expert_probabilities_public": True,
            "zero_total_contribution_exact_primary_fallback": True,
            "production_deployed": False,
        },
        "multimodal fusion audit",
    )
    formal_binding = fusion_audit.get("formal_binding")
    _declared_artifact(
        formal_binding,
        expected_path=paths["fusion_binding"],
        expected_sha256=FORMAL_FUSION_BINDING_SHA256,
        role="audited fusion binding",
    )

    physical = _json(paths["physical_manifest"], "physical interaction manifest")
    _check(
        physical,
        {
            "format": "CC_HHGT_V3_2_PHYSICAL_INTERACTION_RELEASE_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
            "formal_authority_enforced": True,
            "release_ready": False,
            "production_deployed": False,
            "family_to_exact_broadcast": False,
            "global_facts_broadcast_to_cancers": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "old_checkpoints_used": False,
        },
        "physical interaction manifest",
    )
    physical_artifacts = physical.get("artifacts")
    _declared_artifact(
        physical_artifacts.get("physical_interaction_relationships.parquet")
        if isinstance(physical_artifacts, Mapping)
        else None,
        expected_path=paths["physical_relationships"],
        expected_sha256=FORMAL_PHYSICAL_RELATIONSHIPS_SHA256,
        role="physical relationships",
        base=paths["physical_manifest"].parent,
    )
    physical_inputs = physical.get("inputs")
    if not isinstance(physical_inputs, Mapping):
        raise NetworkReleaseError("physical interaction manifest lacks inputs")
    if physical_inputs.get("exact_membership", {}).get("sha256") != FORMAL_MEMBERSHIP_SHA256:
        raise NetworkReleaseError("physical release uses a different membership authority")
    if physical_inputs.get("exact_candidates", {}).get("sha256") != FORMAL_CANDIDATE_SHA256:
        raise NetworkReleaseError("physical release uses a different candidate authority")

    physical_audit = _json(paths["physical_audit"], "physical interaction audit")
    _check(
        physical_audit,
        {
            "format": "CC_HHGT_V3_2_PHYSICAL_INTERACTION_INDEPENDENT_AUDIT_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS",
            "pass_count": 137,
            "fail_count": 0,
            "release_ready": False,
            "production_deployed": False,
            "family_to_exact_broadcast": False,
            "global_facts_broadcast_to_cancers": False,
            "main_ranking_modified": False,
        },
        "physical interaction audit",
    )
    _declared_artifact(
        physical_audit.get("release_manifest"),
        expected_path=paths["physical_manifest"],
        expected_sha256=FORMAL_PHYSICAL_MANIFEST_SHA256,
        role="audited physical manifest",
    )

    experiment = _json(paths["experiment_bridge"], "experiment bridge")
    _check(
        experiment,
        {
            "binding_format": "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_BINDING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_HASH_BOUND_ABSORPTION_BRIDGE",
            "route_determination": "ALREADY_ABSORBED_IN_EVIDENCE_TRANSFORMER",
            "fully_absorbed_by_evidence_transformer": True,
            "confidence_route": "EXISTING_EVIDENCE_TRANSFORMER_ONLY",
            "separate_fusion_forbidden": True,
            "production_deployed": False,
        },
        "experiment bridge",
    )
    invariants = experiment.get("invariants")
    if not isinstance(invariants, Mapping):
        raise NetworkReleaseError("experiment bridge lacks invariants")
    _check(
        invariants,
        {
            "duplicate_fusion_created": False,
            "independent_experiment_probability_head_created": False,
            "family_broadcast_used": False,
            "changes_primary_ranking": False,
            "changes_discovery_ranking": False,
            "private_targets_written_to_public_artifacts": False,
            "production_deployed": False,
        },
        "experiment bridge invariants",
    )
    experiment_artifacts = experiment.get("artifacts")
    exact_query = (
        experiment_artifacts.get("exact_query")
        if isinstance(experiment_artifacts, Mapping)
        else None
    )
    if not isinstance(exact_query, Mapping):
        raise NetworkReleaseError("experiment bridge lacks exact-query artifact")
    if Path(str(exact_query.get("path", ""))).resolve() != paths["experiment_exact_query"]:
        raise NetworkReleaseError("experiment exact-query path differs from binding")
    if artifact_sha256(paths["experiment_exact_query"]) != exact_query.get("sha256"):
        raise NetworkReleaseError("experiment exact-query SHA256 drift")

    experiment_audit = _json(paths["experiment_audit"], "experiment bridge audit")
    _check(
        experiment_audit,
        {
            "format": "CC_HHGT_V3_2_EXPERIMENT_EVIDENCE_BRIDGE_INDEPENDENT_AUDIT_BINDING_V1",
            "status": "PASS",
            "accepted_for_api_integration": True,
            "production_deployed": False,
        },
        "experiment bridge audit",
    )
    _declared_artifact(
        experiment_audit.get("bridge_binding"),
        expected_path=paths["experiment_bridge"],
        expected_sha256=FORMAL_EXPERIMENT_BRIDGE_SHA256,
        role="audited experiment bridge",
    )
    audit_report = experiment_audit.get("report")
    if not isinstance(audit_report, Mapping):
        raise NetworkReleaseError("experiment audit lacks report")
    report_path = Path(str(audit_report.get("path", ""))).resolve()
    if artifact_sha256(report_path) != audit_report.get("sha256"):
        raise NetworkReleaseError("experiment independent audit report SHA256 drift")
    report = _json(report_path, "experiment independent audit report")
    checks = report.get("checks")
    if (
        report.get("status") != "PASS"
        or not isinstance(checks, list)
        or len(checks) != 49
        or any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in checks)
    ):
        raise NetworkReleaseError("experiment independent audit is not 49/49 PASS")


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _duckdb() -> Any:
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise NetworkReleaseError("duckdb is required for network materialisation") from exc
    return duckdb


def _table_columns(con: Any, view: str) -> set[str]:
    return {str(row[0]) for row in con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()}


def _require_columns(con: Any, view: str, required: set[str]) -> None:
    missing = sorted(required - _table_columns(con, view))
    if missing:
        raise NetworkReleaseError(f"{view} lacks columns: {missing}")


def materialize_network_tables(
    *,
    primary_path: str | Path,
    fusion_scores_path: str | Path,
    membership_path: str | Path,
    candidate_path: str | Path,
    physical_relationships_path: str | Path,
    experiment_exact_query_path: str | Path,
    output_root: str | Path,
    expected_candidate_rows: int | None = FORMAL_CANDIDATE_ROWS,
    expected_cancers: int | None = FORMAL_CANCER_COUNT,
    expected_pathways: int | None = FORMAL_EXACT_PATHWAY_COUNT,
    expected_membership_rows: int | None = FORMAL_EXACT_MEMBERSHIP_ROWS,
    expected_physical_rows: int | None = FORMAL_PHYSICAL_RELATIONSHIP_ROWS,
    memory_limit: str = "8GB",
) -> dict[str, Any]:
    """Stream the two immutable parquet projections with DuckDB."""

    paths = {
        "primary": _file(primary_path, "primary scores"),
        "fusion": _file(fusion_scores_path, "fusion scores"),
        "membership": _file(membership_path, "exact membership"),
        "candidate": _file(candidate_path, "candidate authority"),
        "physical": _file(physical_relationships_path, "physical relationships"),
        "experiment": _file(experiment_exact_query_path, "experiment exact query"),
    }
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    nodes_path = output / NETWORK_NODE_FILE
    edges_path = output / NETWORK_EDGE_FILE
    if nodes_path.exists() or edges_path.exists():
        raise NetworkReleaseError("network table target already exists")
    temp_root = output / ".duckdb_tmp"
    temp_root.mkdir(exist_ok=False)
    con = _duckdb().connect(str(output / ".network_build.duckdb"))
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET temp_directory={_sql_path(temp_root)}")
        con.execute("SET preserve_insertion_order=false")
        view_names = {key: ("primary_scores" if key == "primary" else key) for key in paths}
        for name, path in paths.items():
            con.execute(
                f"CREATE VIEW {view_names[name]} AS SELECT * FROM read_parquet({_sql_path(path)})"
            )
        _require_columns(
            con,
            "primary_scores",
            {
                "cancer_id", "lncrna_id", "pathway_id",
                "association_membership_probability", "analysis_version",
                "changes_primary_ranking",
            },
        )
        _require_columns(
            con,
            "fusion",
            {
                "cancer_id", "lncrna_id", "pathway_id", "primary_probability",
                "discovery_adjusted_probability", "fused_confidence_probability",
                "genomic_native_probability", "genomic_native_available",
                "single_cell_native_probability", "single_cell_native_available",
                "evidence_transformer_native_probability",
                "evidence_transformer_native_available", "analysis_version",
                "primary_ranking_unchanged", "adjusted_ranking_is_secondary",
                "used_for_primary_release", "historical_predictions_used",
                "historical_rankings_used",
            },
        )
        _require_columns(con, "membership", {"pathway_id", "gene_id"})
        _require_columns(con, "candidate", {"cancer_id", "lncrna_id", "pathway_id"})
        _require_columns(
            con,
            "physical",
            {
                "relationship_id", "cancer_scope", "lncrna_id", "partner_gene_id",
                "physical_fact_count", "source_occurrence_count",
                "independent_pmid_count", "independent_source_database_count",
                "independent_source_record_count", "availability", "is_prediction",
                "analysis_version",
            },
        )
        _require_columns(
            con,
            "experiment",
            {
                "cancer_id", "lncrna_id", "pathway_id", "experiment_event_count",
                "source_record_count", "absorption_status", "confidence_route",
                "separate_fusion_forbidden", "used_for_separate_fusion",
                "changes_primary_ranking", "changes_discovery_ranking",
            },
        )

        stats = con.execute(
            """
            SELECT
              (SELECT count(*) FROM candidate) AS candidate_rows,
              (SELECT count(DISTINCT cancer_id) FROM candidate) AS cancers,
              (SELECT count(DISTINCT pathway_id) FROM candidate) AS pathways,
              (SELECT count(*) - count(DISTINCT (cancer_id, lncrna_id, pathway_id))
                 FROM candidate) AS duplicate_candidates,
              (SELECT count(*) FROM fusion) AS fusion_rows,
              (SELECT count(*) FROM primary_scores) AS primary_rows,
              (SELECT count(*) FROM physical) AS physical_rows,
              (SELECT count(*) - count(DISTINCT relationship_id) FROM physical)
                AS duplicate_relationships,
              (SELECT count(*) - count(DISTINCT (cancer_id, lncrna_id, pathway_id))
                 FROM experiment) AS duplicate_experiment_keys
            """
        ).fetchone()
        named = dict(
            zip(
                (
                    "candidate_rows", "cancers", "pathways", "duplicate_candidates",
                    "fusion_rows", "primary_rows", "physical_rows",
                    "duplicate_relationships", "duplicate_experiment_keys",
                ),
                map(int, stats),
            )
        )
        expectations = {
            "candidate_rows": expected_candidate_rows,
            "cancers": expected_cancers,
            "pathways": expected_pathways,
            "fusion_rows": expected_candidate_rows,
            "primary_rows": expected_candidate_rows,
            "physical_rows": expected_physical_rows,
        }
        for key, wanted in expectations.items():
            if wanted is not None and named[key] != int(wanted):
                raise NetworkReleaseError(f"{key} mismatch: {named[key]} != {wanted}")
        if any(named[key] for key in ("duplicate_candidates", "duplicate_relationships", "duplicate_experiment_keys")):
            raise NetworkReleaseError(f"duplicate source keys: {named}")

        closure = con.execute(
            """
            SELECT
              (SELECT count(*) FROM (
                 SELECT cancer_id, lncrna_id, pathway_id FROM candidate
                 EXCEPT SELECT cancer_id, lncrna_id, pathway_id FROM fusion)) AS candidate_missing_fusion,
              (SELECT count(*) FROM (
                 SELECT cancer_id, lncrna_id, pathway_id FROM fusion
                 EXCEPT SELECT cancer_id, lncrna_id, pathway_id FROM candidate)) AS fusion_extra_candidate,
              (SELECT count(*) FROM (
                 SELECT cancer_id, lncrna_id, pathway_id FROM primary_scores
                 EXCEPT SELECT cancer_id, lncrna_id, pathway_id FROM fusion)) AS primary_missing_fusion,
              (SELECT count(*) FROM (
                 SELECT cancer_id, lncrna_id, pathway_id FROM fusion
                 EXCEPT SELECT cancer_id, lncrna_id, pathway_id FROM primary_scores)) AS fusion_extra_primary,
              (SELECT count(*) FROM fusion f JOIN primary_scores p USING(cancer_id, lncrna_id, pathway_id)
                 WHERE f.primary_probability IS DISTINCT FROM
                       CAST(p.association_membership_probability AS DOUBLE)) AS primary_value_mismatch,
              (SELECT count(*) FROM fusion WHERE analysis_version <> ?
                 OR NOT primary_ranking_unchanged OR NOT adjusted_ranking_is_secondary
                 OR used_for_primary_release OR historical_predictions_used
                 OR historical_rankings_used) AS bad_fusion_semantics,
              (SELECT count(*) FROM primary_scores WHERE analysis_version <> ?
                 OR NOT changes_primary_ranking) AS bad_primary_semantics,
              (SELECT count(*) FROM physical WHERE analysis_version <> ?
                 OR NOT availability OR is_prediction) AS bad_physical_semantics,
              (SELECT count(*) FROM fusion WHERE
                 (genomic_native_available AND genomic_native_probability IS NULL)
                 OR (NOT genomic_native_available AND genomic_native_probability IS NOT NULL)
                 OR (single_cell_native_available AND single_cell_native_probability IS NULL)
                 OR (NOT single_cell_native_available AND single_cell_native_probability IS NOT NULL)
                 OR (evidence_transformer_native_available AND evidence_transformer_native_probability IS NULL)
                 OR (NOT evidence_transformer_native_available AND evidence_transformer_native_probability IS NOT NULL)
              ) AS native_missingness_mismatch,
              (SELECT count(*) FROM experiment WHERE NOT separate_fusion_forbidden
                 OR used_for_separate_fusion OR changes_primary_ranking
                 OR changes_discovery_ranking
                 OR confidence_route <> 'EXISTING_EVIDENCE_TRANSFORMER_ONLY') AS bad_experiment_semantics,
              (SELECT count(*) FROM experiment e LEFT JOIN candidate c
                 USING(cancer_id, lncrna_id, pathway_id)
                 WHERE c.cancer_id IS NULL) AS experiment_outside_authority
            """,
            [ANALYSIS_VERSION, ANALYSIS_VERSION, ANALYSIS_VERSION],
        ).fetchone()
        if any(int(value) for value in closure):
            raise NetworkReleaseError(
                "source closure/semantic validation failed: "
                + repr(tuple(map(int, closure)))
            )

        con.execute(
            """
            CREATE TEMP VIEW exact_membership AS
            SELECT DISTINCT
              trim(m.pathway_id) AS pathway_id,
              regexp_replace(
                regexp_replace(upper(trim(m.gene_id)), '^(GENE:|PROTEIN:)', ''),
                '\\.[0-9]+$', ''
              ) AS gene_id
            FROM membership m
            JOIN (SELECT DISTINCT pathway_id FROM candidate) p
              ON trim(m.pathway_id) = p.pathway_id
            WHERE trim(m.pathway_id) <> '' AND trim(m.gene_id) <> ''
            """
        )
        membership_rows = int(con.execute("SELECT count(*) FROM exact_membership").fetchone()[0])
        if expected_membership_rows is not None and membership_rows != int(expected_membership_rows):
            raise NetworkReleaseError(
                f"exact membership row mismatch: {membership_rows} != {expected_membership_rows}"
            )

        node_sql = """
            WITH
            lnc AS (
              SELECT lncrna_id AS entity_id, true AS in_model, false AS in_membership,
                     false AS in_physical
              FROM candidate GROUP BY lncrna_id
              UNION ALL
              SELECT lncrna_id, false, false, true FROM physical GROUP BY lncrna_id
            ),
            lnc_grouped AS (
              SELECT entity_id, bool_or(in_model) in_model,
                     bool_or(in_membership) in_membership,
                     bool_or(in_physical) in_physical
              FROM lnc GROUP BY entity_id
            ),
            pathway AS (
              SELECT c.pathway_id AS entity_id, true AS in_model,
                     count(m.gene_id) > 0 AS in_membership, false AS in_physical
              FROM (SELECT DISTINCT pathway_id FROM candidate) c
              LEFT JOIN exact_membership m USING(pathway_id)
              GROUP BY c.pathway_id
            ),
            gene AS (
              SELECT gene_id AS entity_id, false AS in_model, true AS in_membership,
                     false AS in_physical FROM exact_membership GROUP BY gene_id
              UNION ALL
              SELECT partner_gene_id, false, false, true
              FROM physical GROUP BY partner_gene_id
            ),
            gene_grouped AS (
              SELECT entity_id, bool_or(in_model) in_model,
                     bool_or(in_membership) in_membership,
                     bool_or(in_physical) in_physical
              FROM gene GROUP BY entity_id
            )
            SELECT entity_id AS node_id, entity_id, 'lncRNA' AS node_type,
                   entity_id AS display_label, in_model AS in_model_candidate_authority,
                   in_membership AS in_exact_membership,
                   in_physical AS in_physical_interaction,
                   ? AS analysis_version, 'CURRENT_V3.2' AS generation
            FROM lnc_grouped
            UNION ALL
            SELECT 'PATHWAY:' || entity_id, entity_id, 'exact_pathway', entity_id,
                   in_model, in_membership, in_physical, ?, 'CURRENT_V3.2'
            FROM pathway
            UNION ALL
            SELECT 'GENE:' || entity_id, entity_id, 'protein_gene', entity_id,
                   in_model, in_membership, in_physical, ?, 'CURRENT_V3.2'
            FROM gene_grouped
        """
        node_copy = (
            f"COPY ({node_sql}) TO {_sql_path(nodes_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        con.execute(node_copy, [ANALYSIS_VERSION] * 3)

        edge_sql = f"""
            SELECT
              'NET32:M:' || substr(sha256(f.cancer_id || chr(0) || f.lncrna_id || chr(0) || f.pathway_id), 1, 24) AS edge_id,
              '{MODEL_EDGE_TYPE}' AS edge_type,
              f.lncrna_id AS source_node_id,
              'PATHWAY:' || f.pathway_id AS target_node_id,
              true AS directed,
              f.cancer_id,
              CAST(NULL AS VARCHAR) AS cancer_scope,
              f.lncrna_id,
              f.pathway_id,
              CAST(NULL AS VARCHAR) AS protein_gene_id,
              f.primary_probability,
              f.discovery_adjusted_probability,
              f.fused_confidence_probability,
              f.genomic_native_available,
              f.genomic_native_probability,
              f.single_cell_native_available,
              f.single_cell_native_probability,
              f.evidence_transformer_native_available,
              f.evidence_transformer_native_probability,
              e.cancer_id IS NOT NULL AS experiment_support_available,
              CASE WHEN e.cancer_id IS NOT NULL THEN CAST(e.experiment_event_count AS BIGINT) END AS experiment_event_count,
              CASE WHEN e.cancer_id IS NOT NULL THEN CAST(e.source_record_count AS BIGINT) END AS experiment_source_record_count,
              CASE WHEN e.cancer_id IS NOT NULL THEN 'ABSORBED_IN_EVIDENCE_TRANSFORMER_NO_SEPARATE_FUSION' END AS experiment_support_role,
              CAST(NULL AS BIGINT) AS physical_fact_count,
              CAST(NULL AS BIGINT) AS source_occurrence_count,
              CAST(NULL AS BIGINT) AS independent_pmid_count,
              CAST(NULL AS BIGINT) AS independent_source_database_count,
              CAST(NULL AS BIGINT) AS independent_source_record_count,
              true AS availability,
              true AS is_prediction,
              f.primary_ranking_unchanged,
              f.adjusted_ranking_is_secondary,
              f.used_for_primary_release,
              false AS family_to_exact_broadcast,
              false AS changes_primary_ranking,
              f.analysis_version,
              'CURRENT_V3.2_FRESH_MODEL' AS generation
            FROM fusion f
            LEFT JOIN experiment e USING(cancer_id, lncrna_id, pathway_id)

            UNION ALL

            SELECT
              'NET32:G:' || substr(sha256(pathway_id || chr(0) || gene_id), 1, 24),
              '{MEMBERSHIP_EDGE_TYPE}',
              'PATHWAY:' || pathway_id,
              'GENE:' || gene_id,
              true,
              CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR),
              pathway_id, gene_id,
              CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS BIGINT), CAST(NULL AS BIGINT),
              CAST(NULL AS VARCHAR),
              CAST(NULL AS BIGINT), CAST(NULL AS BIGINT), CAST(NULL AS BIGINT),
              CAST(NULL AS BIGINT), CAST(NULL AS BIGINT),
              true, false, true, false, false, false, false,
              ?, 'CURRENT_V3.2_STATIC_EXACT_MEMBERSHIP'
            FROM exact_membership

            UNION ALL

            SELECT
              'NET32:P:' || relationship_id,
              '{PHYSICAL_EDGE_TYPE}',
              lncrna_id,
              'GENE:' || partner_gene_id,
              false,
              CAST(NULL AS VARCHAR), cancer_scope, lncrna_id,
              CAST(NULL AS VARCHAR), partner_gene_id,
              CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS DOUBLE),
              CAST(NULL AS BOOLEAN), CAST(NULL AS BIGINT), CAST(NULL AS BIGINT),
              CAST(NULL AS VARCHAR),
              physical_fact_count, source_occurrence_count, independent_pmid_count,
              independent_source_database_count, independent_source_record_count,
              availability, is_prediction, true, false, false, false, false,
              analysis_version, 'CURRENT_V3.2_PHYSICAL_FACTS'
            FROM physical
        """
        edge_copy = (
            f"COPY ({edge_sql}) TO {_sql_path(edges_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        con.execute(edge_copy, [ANALYSIS_VERSION])

        node_rows = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(nodes_path)]).fetchone()[0])
        edge_counts = {
            str(row[0]): int(row[1])
            for row in con.execute(
                "SELECT edge_type, count(*) FROM read_parquet(?) GROUP BY edge_type",
                [str(edges_path)],
            ).fetchall()
        }
        node_counts = {
            str(row[0]): int(row[1])
            for row in con.execute(
                "SELECT node_type, count(*) FROM read_parquet(?) GROUP BY node_type",
                [str(nodes_path)],
            ).fetchall()
        }
        expected_edge_counts = {
            MODEL_EDGE_TYPE: named["candidate_rows"],
            MEMBERSHIP_EDGE_TYPE: membership_rows,
            PHYSICAL_EDGE_TYPE: named["physical_rows"],
        }
        if edge_counts != expected_edge_counts:
            raise NetworkReleaseError(
                f"network edge-type count mismatch: {edge_counts} != {expected_edge_counts}"
            )
        semantic_bad = con.execute(
            """
            WITH n AS (SELECT * FROM read_parquet(?)), e AS (SELECT * FROM read_parquet(?))
            SELECT
              (SELECT count(*) - count(DISTINCT node_id) FROM n) AS duplicate_nodes,
              (SELECT count(*) - count(DISTINCT edge_id) FROM e) AS duplicate_edges,
              (SELECT count(*) FROM e x LEFT JOIN n ON x.source_node_id=n.node_id
                 WHERE n.node_id IS NULL) AS orphan_sources,
              (SELECT count(*) FROM e x LEFT JOIN n ON x.target_node_id=n.node_id
                 WHERE n.node_id IS NULL) AS orphan_targets,
              (SELECT count(*) FROM e WHERE family_to_exact_broadcast
                 OR changes_primary_ranking) AS forbidden_semantics,
              (SELECT count(*) FROM e WHERE edge_type=? AND (
                 (genomic_native_available AND genomic_native_probability IS NULL)
                 OR (NOT genomic_native_available AND genomic_native_probability IS NOT NULL)
                 OR (single_cell_native_available AND single_cell_native_probability IS NULL)
                 OR (NOT single_cell_native_available AND single_cell_native_probability IS NOT NULL)
                 OR (evidence_transformer_native_available AND evidence_transformer_native_probability IS NULL)
                 OR (NOT evidence_transformer_native_available AND evidence_transformer_native_probability IS NOT NULL)
              )) AS native_missingness_mismatch
            """,
            [str(nodes_path), str(edges_path), MODEL_EDGE_TYPE],
        ).fetchone()
        if any(int(value) for value in semantic_bad):
            raise NetworkReleaseError(
                "materialized network semantic validation failed: "
                + repr(tuple(map(int, semantic_bad)))
            )
    finally:
        con.close()
        (output / ".network_build.duckdb").unlink(missing_ok=True)
        shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "nodes_path": str(nodes_path),
        "nodes_rows": node_rows,
        "node_type_rows": node_counts,
        "edges_path": str(edges_path),
        "edges_rows": sum(edge_counts.values()),
        "edge_type_rows": edge_counts,
        "membership_rows": membership_rows,
    }


def materialize_network_release(
    *,
    primary_path: str | Path,
    exact_lineage_path: str | Path,
    fusion_scores_path: str | Path,
    fusion_binding_path: str | Path,
    fusion_audit_path: str | Path,
    membership_path: str | Path,
    candidate_path: str | Path,
    physical_relationships_path: str | Path,
    physical_manifest_path: str | Path,
    physical_audit_path: str | Path,
    experiment_bridge_path: str | Path,
    experiment_audit_path: str | Path,
    output_root: str | Path,
    execution_code_paths: Sequence[str | Path] = (),
    strict_formal_authority: bool = True,
    memory_limit: str = "8GB",
) -> dict[str, Any]:
    """Validate all bindings, materialise, then atomically publish a release."""

    bridge_path = _file(experiment_bridge_path, "experiment bridge")
    bridge_payload = _json(bridge_path, "experiment bridge")
    bridge_artifacts = bridge_payload.get("artifacts")
    if not isinstance(bridge_artifacts, Mapping) or not isinstance(
        bridge_artifacts.get("exact_query"), Mapping
    ):
        raise NetworkReleaseError("experiment bridge lacks exact-query declaration")
    experiment_exact_query = _file(
        str(bridge_artifacts["exact_query"].get("path", "")),
        "experiment exact query",
    )
    raw_paths = {
        "primary": primary_path,
        "exact_lineage": exact_lineage_path,
        "fusion_scores": fusion_scores_path,
        "fusion_binding": fusion_binding_path,
        "fusion_audit": fusion_audit_path,
        "membership": membership_path,
        "candidate": candidate_path,
        "physical_relationships": physical_relationships_path,
        "physical_manifest": physical_manifest_path,
        "physical_audit": physical_audit_path,
        "experiment_bridge": bridge_path,
        "experiment_audit": experiment_audit_path,
        "experiment_exact_query": experiment_exact_query,
    }
    paths = {key: _file(value, key) for key, value in raw_paths.items()}
    formal_hashes = {
        "primary": FORMAL_PRIMARY_SHA256,
        "exact_lineage": FORMAL_EXACT_LINEAGE_SHA256,
        "fusion_scores": FORMAL_FUSION_SCORES_SHA256,
        "fusion_binding": FORMAL_FUSION_BINDING_SHA256,
        "fusion_audit": FORMAL_FUSION_AUDIT_SHA256,
        "membership": FORMAL_MEMBERSHIP_SHA256,
        "candidate": FORMAL_CANDIDATE_SHA256,
        "physical_relationships": FORMAL_PHYSICAL_RELATIONSHIPS_SHA256,
        "physical_manifest": FORMAL_PHYSICAL_MANIFEST_SHA256,
        "physical_audit": FORMAL_PHYSICAL_AUDIT_SHA256,
        "experiment_bridge": FORMAL_EXPERIMENT_BRIDGE_SHA256,
        "experiment_audit": FORMAL_EXPERIMENT_AUDIT_SHA256,
    }
    if strict_formal_authority:
        for role, digest in formal_hashes.items():
            observed = artifact_sha256(paths[role])
            if observed != digest:
                raise NetworkReleaseError(
                    f"formal {role} SHA256 mismatch: {observed} != {digest}"
                )
        _validate_formal_bindings(paths)

    output = Path(output_root).resolve()
    if output.exists():
        raise NetworkReleaseError(f"network release target already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise NetworkReleaseError(f"network staging target already exists: {staging}")
    staging.mkdir()
    try:
        tables = materialize_network_tables(
            primary_path=paths["primary"],
            fusion_scores_path=paths["fusion_scores"],
            membership_path=paths["membership"],
            candidate_path=paths["candidate"],
            physical_relationships_path=paths["physical_relationships"],
            experiment_exact_query_path=paths["experiment_exact_query"],
            output_root=staging,
            expected_candidate_rows=(FORMAL_CANDIDATE_ROWS if strict_formal_authority else None),
            expected_cancers=(FORMAL_CANCER_COUNT if strict_formal_authority else None),
            expected_pathways=(FORMAL_EXACT_PATHWAY_COUNT if strict_formal_authority else None),
            expected_membership_rows=(FORMAL_EXACT_MEMBERSHIP_ROWS if strict_formal_authority else None),
            expected_physical_rows=(FORMAL_PHYSICAL_RELATIONSHIP_ROWS if strict_formal_authority else None),
            memory_limit=memory_limit,
        )
        inputs = {
            role: {
                "path": str(path),
                "sha256": artifact_sha256(path),
            }
            for role, path in paths.items()
        }
        code = {
            Path(item).name: _bound_file(item, f"execution code {Path(item).name}")
            for item in execution_code_paths
        }
        nodes = staging / NETWORK_NODE_FILE
        edges = staging / NETWORK_EDGE_FILE
        manifest: dict[str, Any] = {
            "format": RELEASE_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "unified_network",
            "status": RELEASE_STATUS,
            "release_ready": False,
            "production_deployed": False,
            "formal_authority_enforced": bool(strict_formal_authority),
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
            "edge_types": list(EDGE_TYPES),
            "node_types": list(NODE_TYPES),
            "counts": {
                "nodes": tables["nodes_rows"],
                "edges": tables["edges_rows"],
                "node_type_rows": tables["node_type_rows"],
                "edge_type_rows": tables["edge_type_rows"],
            },
            "inputs": inputs,
            "execution_code": code,
            "experiment_query_link": {
                "path": str(paths["experiment_exact_query"]),
                "sha256": artifact_sha256(paths["experiment_exact_query"]),
                "bridge_path": str(paths["experiment_bridge"]),
                "bridge_sha256": artifact_sha256(paths["experiment_bridge"]),
                "separate_probability_head": False,
                "double_counted_in_network_score": False,
            },
            "artifacts": {
                NETWORK_NODE_FILE: {
                    "path": NETWORK_NODE_FILE,
                    "rows": tables["nodes_rows"],
                    "sha256": artifact_sha256(nodes),
                },
                NETWORK_EDGE_FILE: {
                    "path": NETWORK_EDGE_FILE,
                    "rows": tables["edges_rows"],
                    "sha256": artifact_sha256(edges),
                },
            },
        }
        manifest_path = staging / MANIFEST_FILE
        _write_json(manifest, manifest_path)
        success = {
            "status": RELEASE_STATUS,
            "analysis_version": ANALYSIS_VERSION,
            "release_ready": False,
            "production_deployed": False,
            "manifest": MANIFEST_FILE,
            "manifest_sha256": artifact_sha256(manifest_path),
            "nodes_rows": tables["nodes_rows"],
            "edges_rows": tables["edges_rows"],
        }
        _write_json(success, staging / SUCCESS_FILE)
        os.replace(staging, output)
    except Exception:
        # Deliberately retain a failed staging directory as evidence; a caller
        # must choose a new target or explicitly inspect/remove it.
        raise

    manifest_path = output / MANIFEST_FILE
    return {
        "status": RELEASE_STATUS,
        "release_root": str(output),
        "manifest_path": str(manifest_path),
        "manifest_sha256": artifact_sha256(manifest_path),
        "success_path": str(output / SUCCESS_FILE),
        "success_sha256": artifact_sha256(output / SUCCESS_FILE),
        "nodes_rows": tables["nodes_rows"],
        "edges_rows": tables["edges_rows"],
        "node_type_rows": tables["node_type_rows"],
        "edge_type_rows": tables["edge_type_rows"],
        "release_ready": False,
        "production_deployed": False,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "EDGE_TYPES",
    "FORMAL_CANDIDATE_SHA256",
    "FORMAL_MEMBERSHIP_SHA256",
    "MANIFEST_FILE",
    "MEMBERSHIP_EDGE_TYPE",
    "MODEL_EDGE_TYPE",
    "NETWORK_EDGE_FILE",
    "NETWORK_NODE_FILE",
    "NODE_TYPES",
    "PHYSICAL_EDGE_TYPE",
    "RELEASE_FORMAT",
    "RELEASE_STATUS",
    "SUCCESS_FILE",
    "NetworkReleaseError",
    "materialize_network_release",
    "materialize_network_tables",
]
