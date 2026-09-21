"""Hash-bound staging release for V3.2 Gene Sets and ranked subtypes.

This module does not train a model.  It binds the already audited V3.2 exact-
pathway Gene Sets to their deterministic ranked-subtype derivatives and records
the distinction between a functional head and a learned predictive head.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
SOURCE_MATERIALIZATION_VERSION = "V3_2_ONESEED_FORMAL_RELEASE"
RELEASE_FORMAT = "CC_HHGT_V3_2_GENE_SET_RANKED_SUBTYPE_RELEASE_BINDING_V1"
RELEASE_STATUS = "SUCCESS_DETERMINISTIC_V32_DERIVED_HEAD_HASH_BOUND"
GENE_SET_RELEASE_FORMAT = "CC_HHGT_V3_2_GENE_SET_PARITY_RELEASE_V1"
GENE_SET_AUDIT_FORMAT = (
    "CC_HHGT_V3_2_GENE_SET_PARITY_INDEPENDENT_AUDIT_BINDING_V1"
)
GENE_SET_AUDIT_CHECK_COUNT = 57
TYPED_UNAVAILABLE = {
    "CHOL": "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS",
    "UCS": "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GeneSetSubtypeReleaseError(RuntimeError):
    """Raised when the deterministic release cannot be safely bound."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise GeneSetSubtypeReleaseError(f"{label} may not be a symlink")
    resolved = requested.resolve()
    if not resolved.is_file():
        raise GeneSetSubtypeReleaseError(f"{label} is missing: {resolved}")
    return resolved


def _safe_directory(path: str | Path, label: str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise GeneSetSubtypeReleaseError(f"{label} may not be a symlink")
    resolved = requested.resolve()
    if not resolved.is_dir():
        raise GeneSetSubtypeReleaseError(f"{label} is missing: {resolved}")
    return resolved


def _read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _safe_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneSetSubtypeReleaseError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise GeneSetSubtypeReleaseError(f"{label} must be a JSON object")
    return source, value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact(path: str | Path, *, rows: int | None = None) -> dict[str, Any]:
    source = _safe_file(path, "artifact")
    result: dict[str, Any] = {
        "path": str(source),
        "sha256": artifact_sha256(source),
        "bytes": source.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _resolve_record(record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise GeneSetSubtypeReleaseError(f"Missing artifact declaration: {label}")
    expected = str(record.get("sha256", "")).lower()
    if not _SHA256.fullmatch(expected):
        raise GeneSetSubtypeReleaseError(f"Invalid SHA256 declaration: {label}")
    path = _safe_file(str(record.get("path", "")), label)
    if artifact_sha256(path) != expected:
        raise GeneSetSubtypeReleaseError(f"Artifact SHA256 drift: {label}")
    return path


def _sql_literal(value: str | Path) -> str:
    return "'" + str(Path(value).resolve()).replace("'", "''") + "'"


def _parquet(path: str | Path) -> str:
    return f"read_parquet({_sql_literal(path)}, hive_partitioning=false)"


def _member_relation(parts: list[Path]) -> str:
    values = ",".join(_sql_literal(path) for path in parts)
    return f"read_parquet([{values}], hive_partitioning=false)"


def _validate_gene_set_authority(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    audit_path: Path,
    audit: Mapping[str, Any],
    audit_sha256: str,
) -> tuple[dict[str, Path], list[Path]]:
    required_manifest = {
        "format": GENE_SET_RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "gene_set",
        "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
        "target_level": "exact_pathway",
        "primary_score_preserved": True,
        "ranking_uses_auxiliary_evidence": False,
        "family_to_exact_broadcast": False,
        "old_checkpoints_used": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "release_ready": False,
        "production_deployed": False,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise GeneSetSubtypeReleaseError(
                f"Gene Set authority has invalid {key}: {manifest.get(key)!r}"
            )
    required_audit = {
        "format": GENE_SET_AUDIT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "PASS",
        "pass_count": GENE_SET_AUDIT_CHECK_COUNT,
        "fail_count": 0,
        "accepted_for_registry_integration": True,
        "independent_of_materializer_implementation": True,
        "release_ready": False,
        "production_deployed": False,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise GeneSetSubtypeReleaseError(
                f"Gene Set audit has invalid {key}: {audit.get(key)!r}"
            )
    declaration = audit.get("manifest")
    if (
        not isinstance(declaration, Mapping)
        or Path(str(declaration.get("path", ""))).resolve() != manifest_path
        or declaration.get("sha256") != manifest_sha256
    ):
        raise GeneSetSubtypeReleaseError(
            "Gene Set independent audit is tied to a different manifest"
        )
    report_path = _resolve_record(audit.get("report"), "Gene Set audit report")
    _, report = _read_json(report_path, "Gene Set audit report")
    checks = report.get("checks")
    if (
        report.get("status") != "PASS"
        or report.get("pass_count") != GENE_SET_AUDIT_CHECK_COUNT
        or report.get("fail_count") != 0
        or report.get("accepted_for_registry_integration") is not True
        or not isinstance(checks, list)
        or len(checks) != GENE_SET_AUDIT_CHECK_COUNT
        or any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in checks)
    ):
        raise GeneSetSubtypeReleaseError(
            "Gene Set independent audit did not pass all "
            f"{GENE_SET_AUDIT_CHECK_COUNT} checks"
        )
    artifacts = manifest.get("required_artifact_ids")
    if not isinstance(artifacts, Mapping):
        raise GeneSetSubtypeReleaseError("Gene Set authority lacks artifacts")
    role_map = {
        "catalog": "v32_gene_set_catalog",
        "enrichment": "v32_gene_set_enrichment",
        "gmt": "v32_gene_set_gmt",
        "coverage": "v32_gene_set_cancer_coverage",
        "report": "v32_gene_set_report_manifest",
    }
    paths = {
        role: _resolve_record(artifacts.get(artifact_id), f"Gene Set {role}")
        for role, artifact_id in role_map.items()
    }
    source_records = manifest.get("sources")
    if not isinstance(source_records, Mapping):
        raise GeneSetSubtypeReleaseError("Gene Set authority lacks source declarations")
    paths["exact_pathway_associations"] = _resolve_record(
        source_records.get("primary_exact_pathway"),
        "current V3.2 exact-pathway associations",
    )
    member_record = artifacts.get("v32_gene_set_member_matrix")
    if not isinstance(member_record, Mapping):
        raise GeneSetSubtypeReleaseError("Gene Set member matrix is missing")
    part_records = member_record.get("part_artifacts")
    if not isinstance(part_records, list) or not part_records:
        raise GeneSetSubtypeReleaseError("Gene Set member partitions are missing")
    parts = [
        _resolve_record(record, f"Gene Set member part {index}")
        for index, record in enumerate(part_records)
    ]
    observed_tree = hashlib.sha256(
        "\n".join(
            f"{path.name}\t{artifact_sha256(path)}" for path in parts
        ).encode()
    ).hexdigest()
    if observed_tree != member_record.get("sha256_tree"):
        raise GeneSetSubtypeReleaseError("Gene Set member tree SHA256 drift")
    paths["gene_set_manifest"] = manifest_path
    paths["gene_set_audit"] = audit_path
    return paths, parts


def materialize_gene_set_ranked_subtype_release(
    *,
    gene_set_manifest_path: str | Path,
    expected_gene_set_manifest_sha256: str,
    gene_set_audit_binding_path: str | Path,
    expected_gene_set_audit_binding_sha256: str,
    subtype_root: str | Path,
    subtype_implementation_path: str | Path,
    output_root: str | Path,
    strict_formal_authority: bool = True,
) -> dict[str, Any]:
    """Bind audited Gene Sets and deterministic ranked-subtype tables."""

    manifest_path, gene_set_manifest = _read_json(
        gene_set_manifest_path, "Gene Set release manifest"
    )
    expected_manifest_sha = str(expected_gene_set_manifest_sha256).lower()
    if (
        not _SHA256.fullmatch(expected_manifest_sha)
        or artifact_sha256(manifest_path) != expected_manifest_sha
    ):
        raise GeneSetSubtypeReleaseError("Gene Set release manifest SHA256 drift")
    audit_path, gene_set_audit = _read_json(
        gene_set_audit_binding_path, "Gene Set independent-audit binding"
    )
    expected_audit_sha = str(expected_gene_set_audit_binding_sha256).lower()
    if (
        not _SHA256.fullmatch(expected_audit_sha)
        or artifact_sha256(audit_path) != expected_audit_sha
    ):
        raise GeneSetSubtypeReleaseError("Gene Set audit binding SHA256 drift")
    gene_paths, member_parts = _validate_gene_set_authority(
        manifest_path=manifest_path,
        manifest=gene_set_manifest,
        manifest_sha256=expected_manifest_sha,
        audit_path=audit_path,
        audit=gene_set_audit,
        audit_sha256=expected_audit_sha,
    )

    subtype_dir = _safe_directory(subtype_root, "ranked-subtype source root")
    subtype_paths = {
        "cancer_program_subtype": _safe_file(
            subtype_dir / "CANCER_PROGRAM_SUBTYPE.parquet",
            "cancer program subtype",
        ),
        "pathway_context_subtype": _safe_file(
            subtype_dir / "PATHWAY_CONTEXT_SUBTYPE.parquet",
            "pathway context subtype",
        ),
        "pathway_pairwise_similarity": _safe_file(
            subtype_dir / "PATHWAY_PAIRWISE_SIMILARITY.parquet",
            "pathway pairwise stability",
        ),
        "materialization_success": _safe_file(
            subtype_dir / "MATERIALIZATION_SUCCESS.json",
            "subtype materialization SUCCESS",
        ),
        "materialization_coverage_audit": _safe_file(
            subtype_dir / "MATERIALIZATION_COVERAGE_AUDIT.json",
            "subtype materialization coverage audit",
        ),
        "implementation": _safe_file(
            subtype_implementation_path, "deterministic subtype implementation"
        ),
    }
    _, success = _read_json(
        subtype_paths["materialization_success"], "subtype materialization SUCCESS"
    )
    _, source_audit = _read_json(
        subtype_paths["materialization_coverage_audit"],
        "subtype materialization coverage audit",
    )
    required_success = {
        "status": "SUCCESS",
        "analysis_version": SOURCE_MATERIALIZATION_VERSION,
        "family_used_as_target": False,
        "regulatory_evidence_used_for_ranking": False,
        "subtype_feedback_to_model": False,
        "pathway_seed_isolation": True,
    }
    for key, expected in required_success.items():
        if success.get(key) != expected:
            raise GeneSetSubtypeReleaseError(
                f"Subtype materialization has invalid {key}: {success.get(key)!r}"
            )
    if (
        source_audit.get("status") != "PASS"
        or source_audit.get("family_used_as_target") is not False
        or source_audit.get("regulatory_evidence_used_for_ranking") is not False
        or source_audit.get("violations") != []
    ):
        raise GeneSetSubtypeReleaseError("Subtype materialization coverage audit failed")

    catalog = _parquet(gene_paths["catalog"])
    members = _member_relation(member_parts)
    enrichment = _parquet(gene_paths["enrichment"])
    program = _parquet(subtype_paths["cancer_program_subtype"])
    context = _parquet(subtype_paths["pathway_context_subtype"])
    pairwise = _parquet(subtype_paths["pathway_pairwise_similarity"])
    exact_associations = _parquet(gene_paths["exact_pathway_associations"])
    coverage_literal = _sql_literal(gene_paths["coverage"])
    con = duckdb.connect()
    try:
        catalog_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id), count(DISTINCT cancer_id),
                   count_if(pathway_target_level <> 'exact_pathway'),
                   count_if(ranking_uses_regulatory_evidence IS NOT FALSE),
                   count(*) - count(DISTINCT (cancer_id, pathway_id))
            FROM {catalog}
            """
        ).fetchone()
        exact_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT (cancer_id,lncrna_id,pathway_id)),
                   count(DISTINCT cancer_id), count(DISTINCT lncrna_id),
                   count(DISTINCT pathway_id),
                   count_if(pathway_target_level <> 'exact_pathway'),
                   count_if(pathway_id = pathway_family_id),
                   count_if(association_membership_probability IS NULL),
                   count_if(association_membership_probability IS NOT NULL AND
                            (NOT isfinite(association_membership_probability) OR
                             association_membership_probability NOT BETWEEN 0 AND 1)),
                   count_if(analysis_version <> ?),
                   count_if(training_run_id <> 'v32-exact-pathway-20260825')
            FROM {exact_associations}
            """,
            [ANALYSIS_VERSION],
        ).fetchone()
        member_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id),
                   count_if(pathway_target_level <> 'exact_pathway')
            FROM {members}
            """
        ).fetchone()
        enrichment_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT geneset_id),
                   count_if(family_to_exact_broadcast OR changes_primary_ranking
                            OR used_for_primary_ranking)
            FROM {enrichment}
            """
        ).fetchone()
        program_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT cancer_id),
                   count_if(cancer_id IN ('CHOL', 'UCS')),
                   count_if(classification_status IS NULL OR selected_k < 0
                            OR proposed_k < 0)
            FROM {program}
            """
        ).fetchone()
        context_stats = con.execute(
            f"""
            SELECT count(*), count(DISTINCT (cancer_id, pathway_id)),
                   count_if(cancer_id IN ('CHOL', 'UCS')),
                   count_if(classification_status = 'UNAVAILABLE' AND
                            (selected_k <> 0 OR proposed_k <> 0 OR
                             pathway_context_subtype_id <> 'UNAVAILABLE')),
                   count_if(classification_status <> 'UNAVAILABLE' AND
                            (selected_k < 1 OR pathway_context_subtype_id = 'UNAVAILABLE'))
            FROM {context}
            """
        ).fetchone()
        context_closure = con.execute(
            f"""
            SELECT count(*) FROM {catalog} m
            FULL OUTER JOIN {context} s
              USING (cancer_id, pathway_id, pathway_family_id)
            WHERE m.geneset_id IS NULL OR s.pathway_context_subtype_id IS NULL
            """
        ).fetchone()[0]
        pairwise_stats = con.execute(
            f"""
            SELECT count(*),
                   count(*) - count(DISTINCT (p.pathway_id, p.cancer_a, p.cancer_b)),
                   count_if(p.cancer_a >= p.cancer_b),
                   count_if(p.rbo_similarity NOT BETWEEN 0 AND 1 OR
                            p.weighted_jaccard_similarity NOT BETWEEN 0 AND 1 OR
                            p.combined_similarity NOT BETWEEN 0 AND 1),
                   count_if(a.pathway_id IS NULL OR b.pathway_id IS NULL OR
                            p.pathway_family_id <> a.pathway_family_id OR
                            p.pathway_family_id <> b.pathway_family_id)
            FROM {pairwise} p
            LEFT JOIN {context} a
              ON p.pathway_id=a.pathway_id AND p.cancer_a=a.cancer_id
            LEFT JOIN {context} b
              ON p.pathway_id=b.pathway_id AND p.cancer_b=b.cancer_id
            """
        ).fetchone()
        coverage_rows = con.execute(
            f"""
            SELECT cancer_id, coverage_status, publishable_genesets
            FROM read_csv_auto({coverage_literal}, delim='\t', header=true)
            ORDER BY cancer_id
            """
        ).fetchall()
    finally:
        con.close()

    unavailable = {
        str(cancer): str(status)
        for cancer, status, _ in coverage_rows
        if str(status) != "AVAILABLE"
    }
    observed = {
        "exact_association_rows": int(exact_stats[0]),
        "exact_association_unique_keys": int(exact_stats[1]),
        "exact_association_cancers": int(exact_stats[2]),
        "exact_association_lncrnas": int(exact_stats[3]),
        "exact_association_pathways": int(exact_stats[4]),
        "exact_association_non_exact_rows": int(exact_stats[5]),
        "exact_association_family_as_target_rows": int(exact_stats[6]),
        "exact_association_null_probability_rows": int(exact_stats[7]),
        "exact_association_invalid_probability_rows": int(exact_stats[8]),
        "exact_association_version_violations": int(exact_stats[9]),
        "exact_association_run_violations": int(exact_stats[10]),
        "gene_sets": int(catalog_stats[0]),
        "unique_gene_sets": int(catalog_stats[1]),
        "available_cancers": int(catalog_stats[2]),
        "catalog_non_exact_rows": int(catalog_stats[3]),
        "catalog_evidence_ranked_rows": int(catalog_stats[4]),
        "catalog_duplicate_exact_keys": int(catalog_stats[5]),
        "member_rows": int(member_stats[0]),
        "member_gene_sets": int(member_stats[1]),
        "member_non_exact_rows": int(member_stats[2]),
        "enrichment_rows": int(enrichment_stats[0]),
        "enrichment_gene_sets": int(enrichment_stats[1]),
        "enrichment_semantic_violations": int(enrichment_stats[2]),
        "cancer_program_rows": int(program_stats[0]),
        "cancer_program_unique_cancers": int(program_stats[1]),
        "cancer_program_typed_unavailable_rows": int(program_stats[2]),
        "cancer_program_semantic_violations": int(program_stats[3]),
        "pathway_context_rows": int(context_stats[0]),
        "pathway_context_unique_keys": int(context_stats[1]),
        "pathway_context_typed_unavailable_rows": int(context_stats[2]),
        "pathway_context_unavailable_violations": int(context_stats[3]),
        "pathway_context_available_violations": int(context_stats[4]),
        "pathway_context_catalog_closure_errors": int(context_closure),
        "pathway_pairwise_rows": int(pairwise_stats[0]),
        "pathway_pairwise_duplicate_keys": int(pairwise_stats[1]),
        "pathway_pairwise_order_violations": int(pairwise_stats[2]),
        "pathway_pairwise_range_violations": int(pairwise_stats[3]),
        "pathway_pairwise_context_closure_errors": int(pairwise_stats[4]),
        "formal_cancers": len(coverage_rows),
        "typed_unavailable_cancers": unavailable,
    }
    zero_fields = [
        key
        for key in observed
        if key.endswith(("_violations", "_errors"))
        or key in {
            "exact_association_non_exact_rows",
            "exact_association_family_as_target_rows",
            "exact_association_null_probability_rows",
            "exact_association_invalid_probability_rows",
            "catalog_non_exact_rows",
            "catalog_evidence_ranked_rows",
            "catalog_duplicate_exact_keys",
            "member_non_exact_rows",
            "cancer_program_typed_unavailable_rows",
        }
    ]
    if any(observed[key] != 0 for key in zero_fields):
        raise GeneSetSubtypeReleaseError(
            f"Gene Set/ranked-subtype semantic audit failed: {observed}"
        )
    if unavailable != TYPED_UNAVAILABLE:
        raise GeneSetSubtypeReleaseError(
            f"Typed unavailable cancers drifted: {unavailable}"
        )
    if strict_formal_authority:
        expected_counts = {
            "exact_association_rows": 3_300_000,
            "exact_association_unique_keys": 3_300_000,
            "exact_association_cancers": 33,
            "exact_association_lncrnas": 8_541,
            "exact_association_pathways": 2_135,
            "gene_sets": 54_380,
            "unique_gene_sets": 54_380,
            "available_cancers": 31,
            "member_rows": 2_217_719,
            "member_gene_sets": 54_380,
            "enrichment_rows": 54_380,
            "enrichment_gene_sets": 54_380,
            "cancer_program_rows": 31,
            "cancer_program_unique_cancers": 31,
            "pathway_context_rows": 54_380,
            "pathway_context_unique_keys": 54_380,
            "pathway_pairwise_rows": 668_274,
            "formal_cancers": 33,
        }
        mismatch = {
            key: (observed[key], expected)
            for key, expected in expected_counts.items()
            if observed[key] != expected
        }
        if mismatch:
            raise GeneSetSubtypeReleaseError(
                f"Formal Gene Set/ranked-subtype counts drifted: {mismatch}"
            )

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise GeneSetSubtypeReleaseError(f"Refusing output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "GENE_SET_RANKED_SUBTYPE_RELEASE_BINDING.json"
    success_path = output / "SUCCESS.json"
    member_records = [_artifact(path) for path in member_parts]
    binding: dict[str, Any] = {
        "format": RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "source_materialization_version": SOURCE_MATERIALIZATION_VERSION,
        "status": RELEASE_STATUS,
        "module_ids": ["gene_set", "ranked_subtype"],
        "result_role": "DETERMINISTIC_V32_EXACT_PATHWAY_DERIVED_FUNCTIONAL_HEAD",
        "head_kind": "DETERMINISTIC_DERIVED_FUNCTIONAL_HEAD_NOT_PREDICTIVE_MODEL",
        "initialization": "v32_core_frozen_new_head",
        "initialization_semantics": (
            "FUNCTIONAL_HEAD_BOUND_TO_FROZEN_CURRENT_V32_EXACT_PATHWAY_OUTPUTS;"
            "NO_PARAMETER_INITIALIZATION"
        ),
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
        "typed_unavailable_cancers": [
            {"cancer_id": cancer, "reason": reason}
            for cancer, reason in sorted(TYPED_UNAVAILABLE.items())
        ],
        "sources": {
            "audited_gene_set_release_manifest": _artifact(manifest_path),
            "audited_gene_set_independent_audit_binding": _artifact(audit_path),
            "subtype_materialization_success": _artifact(
                subtype_paths["materialization_success"]
            ),
            "subtype_materialization_coverage_audit": _artifact(
                subtype_paths["materialization_coverage_audit"]
            ),
            "deterministic_subtype_implementation": _artifact(
                subtype_paths["implementation"]
            ),
        },
        "artifacts": {
            "exact_pathway_associations": _artifact(
                gene_paths["exact_pathway_associations"],
                rows=observed["exact_association_rows"],
            ),
            "gene_set_catalog": _artifact(
                gene_paths["catalog"], rows=observed["gene_sets"]
            ),
            "gene_set_members": {
                "path": str(Path(member_parts[0]).parent),
                "parts": len(member_records),
                "rows": observed["member_rows"],
                "sha256_tree": hashlib.sha256(
                    "\n".join(
                        f"{Path(item['path']).name}\t{item['sha256']}"
                        for item in member_records
                    ).encode()
                ).hexdigest(),
                "part_artifacts": member_records,
            },
            "gene_set_gmt": _artifact(
                gene_paths["gmt"], rows=observed["gene_sets"]
            ),
            "gene_set_enrichment": _artifact(
                gene_paths["enrichment"], rows=observed["enrichment_rows"]
            ),
            "gene_set_coverage": _artifact(
                gene_paths["coverage"], rows=observed["formal_cancers"]
            ),
            "gene_set_report": _artifact(gene_paths["report"]),
            "ranked_subtypes": _artifact(
                subtype_paths["pathway_context_subtype"],
                rows=observed["pathway_context_rows"],
            ),
            "cancer_program_subtypes": _artifact(
                subtype_paths["cancer_program_subtype"],
                rows=observed["cancer_program_rows"],
            ),
            "subtype_stability": _artifact(
                subtype_paths["pathway_pairwise_similarity"],
                rows=observed["pathway_pairwise_rows"],
            ),
        },
        "counts": observed,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(binding_path, binding)
    success_marker = {
        "format": RELEASE_FORMAT,
        "status": RELEASE_STATUS,
        "binding": binding_path.name,
        "binding_sha256": artifact_sha256(binding_path),
        "deterministic_derived_head": True,
        "trained_model": False,
        "typed_unavailable_cancers": sorted(TYPED_UNAVAILABLE),
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(success_path, success_marker)
    return binding


__all__ = [
    "ANALYSIS_VERSION",
    "GENE_SET_AUDIT_FORMAT",
    "GENE_SET_RELEASE_FORMAT",
    "GeneSetSubtypeReleaseError",
    "RELEASE_FORMAT",
    "RELEASE_STATUS",
    "SOURCE_MATERIALIZATION_VERSION",
    "TYPED_UNAVAILABLE",
    "artifact_sha256",
    "materialize_gene_set_ranked_subtype_release",
]
