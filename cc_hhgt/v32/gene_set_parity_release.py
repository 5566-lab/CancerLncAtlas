"""Hash-bound V3.2 Gene Set parity release.

The ranked Gene Sets already exist as a deterministic post-processing view of
the fresh V3.2 exact-pathway ensemble.  This module closes the historical
``geneset_enrichment`` gap without refitting or changing those rankings.  The
new enrichment table is a descriptive, per-Gene-Set summary of the public
primary score and independently available V3.2 auxiliary evidence.

It is deliberately *not* an ORA test of a Gene Set against the pathway that
defined it.  Such a test would be circular.  User-supplied mixed lists retain
their separate exact-pathway ORA/learned-query endpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
RELEASE_FORMAT = "CC_HHGT_V3_2_GENE_SET_PARITY_RELEASE_V1"
REPORT_FORMAT = "CC_HHGT_V3_2_GENE_SET_REPORT_MANIFEST_V1"
ENRICHMENT_FORMAT = "CC_HHGT_V3_2_GENE_SET_EVIDENCE_SUMMARY_V2"
FORMAL_GENESETS = 54_380
FORMAL_MEMBER_ROWS = 2_217_719
FORMAL_CANCERS = 33
FORMAL_AVAILABLE_CANCERS = 31
FORMAL_UNAVAILABLE_CANCERS = ("CHOL", "UCS")
FORMAL_PRIMARY_ROWS = 3_300_000
FORMAL_PRIMARY_SHA256 = (
    "4259ecc7453f0c636200087e8393db59a73824414f64c78a9fbc8e3bcb47c3a1"
)
GLOBAL_SCOPE = "GLOBAL_NOT_CANCER_SPECIFIC"
NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE = (
    "NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE"
)


class GeneSetParityReleaseError(RuntimeError):
    """Raised when a Gene Set source or parity output violates its contract."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: str | Path, role: str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise GeneSetParityReleaseError(f"{role} is an unsafe symlink: {requested}")
    source = requested.resolve()
    if not source.is_file():
        raise GeneSetParityReleaseError(f"{role} is missing: {source}")
    return source


def _safe_directory(path: str | Path, role: str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise GeneSetParityReleaseError(f"{role} is an unsafe symlink: {requested}")
    source = requested.resolve()
    if not source.is_dir():
        raise GeneSetParityReleaseError(f"{role} is missing: {source}")
    return source


def _read_json(path: str | Path, role: str) -> tuple[Path, dict[str, Any]]:
    source = _safe_file(path, role)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneSetParityReleaseError(f"{role} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise GeneSetParityReleaseError(f"{role} must contain an object")
    return source, value


def _read_bound_json(
    path: str | Path, expected_sha256: str, role: str
) -> tuple[Path, dict[str, Any]]:
    expected = str(expected_sha256 or "").lower()
    if not _SHA256.fullmatch(expected):
        raise GeneSetParityReleaseError(f"{role} expected SHA256 is invalid")
    source, value = _read_json(path, role)
    if artifact_sha256(source) != expected:
        raise GeneSetParityReleaseError(f"{role} SHA256 drift")
    return source, value


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _parquet_relation(path: Path) -> str:
    return f"read_parquet({_sql_literal(path)}, hive_partitioning=false)"


def _member_relation(member_root: Path) -> tuple[str, list[Path]]:
    parts = sorted(member_root.glob("*.parquet"))
    if not parts or any(part.is_symlink() for part in parts):
        raise GeneSetParityReleaseError("Gene Set member parts are missing or unsafe")
    quoted = ", ".join(_sql_literal(part) for part in parts)
    return f"read_parquet([{quoted}], hive_partitioning=false)", parts


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _copy_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        connection.execute(
            f"COPY ({query}) TO {_sql_literal(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": artifact_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _bound_artifact(
    container: Mapping[str, Any], role: str, *, base: Path | None = None
) -> tuple[Path, int | None]:
    record = container.get(role)
    if not isinstance(record, Mapping):
        raise GeneSetParityReleaseError(f"Binding lacks artifact {role}")
    raw = Path(str(record.get("path", "")))
    source = (base / raw).resolve() if base is not None and not raw.is_absolute() else raw.resolve()
    expected = str(record.get("sha256", "")).lower()
    if not _SHA256.fullmatch(expected) or artifact_sha256(_safe_file(source, role)) != expected:
        raise GeneSetParityReleaseError(f"Bound artifact {role} SHA256 drift")
    rows = record.get("rows")
    return source, int(rows) if rows is not None else None


def _validate_source_contracts(
    *,
    materialization_success_path: Path,
    coverage_path: Path,
    master_path: Path,
    member_root: Path,
    gmt_path: Path,
    primary_path: Path,
    fusion_binding_path: Path,
    fusion_binding: Mapping[str, Any],
    fusion_post_audit: Mapping[str, Any],
    physical_manifest_path: Path,
    physical_manifest: Mapping[str, Any],
    physical_post_audit: Mapping[str, Any],
    strict_formal_authority: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    _, success = _read_json(materialization_success_path, "Gene Set materialization success")
    if success.get("status") != "SUCCESS" or success.get("family_used_as_target") is not False:
        raise GeneSetParityReleaseError("Gene Set materialization is not a successful exact run")
    if success.get("regulatory_evidence_used_for_ranking") is not False:
        raise GeneSetParityReleaseError("Auxiliary evidence changed ranked Gene Set membership")

    required_fusion = {
        "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_SECONDARY_FUSION",
        "primary_score_preserved": True,
        "primary_ranking_unchanged": True,
        "native_expert_probabilities_public": True,
        "release_ready": False,
        "production_deployed": False,
    }
    for key, expected in required_fusion.items():
        if fusion_binding.get(key) != expected:
            raise GeneSetParityReleaseError(
                f"Fusion binding {key} drift: {fusion_binding.get(key)!r}"
            )
    public_experts = set(map(str, fusion_binding.get("public_native_experts", [])))
    if public_experts != {"genomic", "single_cell", "evidence_transformer"}:
        raise GeneSetParityReleaseError("Fusion binding lacks the three public native experts")
    fusion_artifacts = fusion_binding.get("artifacts")
    if not isinstance(fusion_artifacts, Mapping):
        raise GeneSetParityReleaseError("Fusion binding lacks artifacts")
    fusion_scores, fusion_rows = _bound_artifact(fusion_artifacts, "secondary_scores")
    if fusion_rows is None:
        raise GeneSetParityReleaseError("Fusion binding lacks public row count")
    if fusion_post_audit.get("accepted_for_api_integration") is not True:
        raise GeneSetParityReleaseError("Fusion independent post-audit is not accepted")
    if fusion_post_audit.get("status") != "PASS" or int(
        fusion_post_audit.get("fail_count", -1)
    ) != 0:
        raise GeneSetParityReleaseError("Fusion independent post-audit contains failures")

    required_physical = {
        "format": "CC_HHGT_V3_2_PHYSICAL_INTERACTION_RELEASE_V1",
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
        "family_to_exact_broadcast": False,
        "old_checkpoints_used": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "release_ready": False,
        "production_deployed": False,
    }
    for key, expected in required_physical.items():
        if physical_manifest.get(key) != expected:
            raise GeneSetParityReleaseError(
                f"Physical manifest {key} drift: {physical_manifest.get(key)!r}"
            )
    physical_artifacts = physical_manifest.get("artifacts")
    if not isinstance(physical_artifacts, Mapping):
        raise GeneSetParityReleaseError("Physical manifest lacks artifacts")
    physical_scores, _ = _bound_artifact(
        physical_artifacts,
        "interaction_exact_pathway_enrichment.parquet",
        base=physical_manifest_path.parent,
    )
    if physical_post_audit.get("status") not in {"PASS", "PASS_INDEPENDENT_AUDIT"}:
        raise GeneSetParityReleaseError("Physical independent audit is not PASS")
    if int(physical_post_audit.get("fail_count", -1)) != 0:
        raise GeneSetParityReleaseError("Physical independent audit contains failures")

    if strict_formal_authority:
        expected = {
            "genesets": FORMAL_GENESETS,
            "subtype_member_rows": FORMAL_MEMBER_ROWS,
            "cancers": FORMAL_CANCERS,
        }
        for key, value in expected.items():
            if int(success.get(key, -1)) != value:
                raise GeneSetParityReleaseError(f"Formal Gene Set {key} drift")
        if artifact_sha256(primary_path) != FORMAL_PRIMARY_SHA256:
            raise GeneSetParityReleaseError("Formal primary exact-pathway SHA256 drift")
        if fusion_rows != FORMAL_PRIMARY_ROWS:
            raise GeneSetParityReleaseError("Formal fusion row count drift")

    source_summary = {
        "materialization_success": _artifact(materialization_success_path),
        "cancer_coverage": _artifact(coverage_path, rows=FORMAL_CANCERS if strict_formal_authority else None),
        "geneset_master": _artifact(master_path),
        "geneset_gmt": _artifact(gmt_path),
        "primary_exact_pathway": _artifact(primary_path),
        "fusion_binding": _artifact(fusion_binding_path),
        "physical_manifest": _artifact(physical_manifest_path),
    }
    return fusion_scores, physical_scores, source_summary


def _audit_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    master: str,
    members: str,
    primary: str,
    fusion: str,
    coverage_path: Path,
    strict_formal_authority: bool,
) -> dict[str, Any]:
    master_summary = connection.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT geneset_id) AS unique_ids,
               count(DISTINCT cancer_id) AS cancers,
               count_if(pathway_target_level <> 'exact_pathway') AS non_exact,
               count_if(ranking_uses_regulatory_evidence IS NOT FALSE) AS evidence_ranked,
               count_if(member_count <= 0) AS empty_sets
        FROM {master}
        """
    ).fetchone()
    member_summary = connection.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT geneset_id) AS genesets,
               count_if(pathway_target_level <> 'exact_pathway') AS non_exact,
               count_if(association_membership_probability IS NULL OR
                        association_membership_probability < 0 OR
                        association_membership_probability > 1) AS bad_probability,
               count(*) - count(DISTINCT (geneset_id, lncrna_id)) AS duplicate_members
        FROM {members}
        """
    ).fetchone()
    binding_summary = connection.execute(
        f"""
        SELECT
          count_if(p.lncrna_id IS NULL) AS missing_primary,
          count_if(f.lncrna_id IS NULL) AS missing_fusion,
          count_if(p.lncrna_id IS NOT NULL AND
                   abs(CAST(m.association_membership_probability AS DOUBLE) -
                       CAST(p.association_membership_probability AS DOUBLE)) > 1e-7)
            AS primary_probability_drift,
          count_if(f.lncrna_id IS NOT NULL AND
                   abs(CAST(m.association_membership_probability AS DOUBLE) -
                       CAST(f.primary_probability AS DOUBLE)) > 1e-7)
            AS fusion_primary_drift
        FROM {members} m
        LEFT JOIN {primary} p USING (cancer_id, lncrna_id, pathway_id)
        LEFT JOIN {fusion} f USING (cancer_id, lncrna_id, pathway_id)
        """
    ).fetchone()
    member_count_drift = int(
        connection.execute(
            f"""
            SELECT count(*) FROM (
              SELECT x.geneset_id
              FROM {master} x
              LEFT JOIN (
                SELECT geneset_id, count(*) AS observed
                FROM {members} GROUP BY geneset_id
              ) y USING (geneset_id)
              WHERE coalesce(y.observed, 0) <> x.member_count
            )
            """
        ).fetchone()[0]
    )
    coverage_rows = connection.execute(
        f"""
        SELECT cancer_id, coverage_status, publishable_genesets
        FROM read_csv_auto({_sql_literal(coverage_path)}, delim='\\t', header=true)
        ORDER BY cancer_id
        """
    ).fetchall()
    if not coverage_rows:
        raise GeneSetParityReleaseError("Gene Set cancer coverage table is empty")
    coverage_cancers = [str(row[0]) for row in coverage_rows]
    unavailable_cancers = [
        str(row[0])
        for row in coverage_rows
        if str(row[1]) == "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS"
    ]
    invalid_coverage_rows = sum(
        1
        for _, status, genesets in coverage_rows
        if (
            str(status) == "AVAILABLE" and int(genesets) <= 0
        ) or (
            str(status) == "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS"
            and int(genesets) != 0
        ) or str(status) not in {
            "AVAILABLE", "UNAVAILABLE_NO_GENESET_PASSING_THRESHOLDS"
        }
    )
    audit = {
        "master_rows": int(master_summary[0]),
        "master_unique_ids": int(master_summary[1]),
        "cancers": int(master_summary[2]),
        "master_non_exact_rows": int(master_summary[3]),
        "evidence_ranked_gene_sets": int(master_summary[4]),
        "empty_gene_sets": int(master_summary[5]),
        "member_rows": int(member_summary[0]),
        "member_genesets": int(member_summary[1]),
        "member_non_exact_rows": int(member_summary[2]),
        "bad_member_probabilities": int(member_summary[3]),
        "duplicate_members": int(member_summary[4]),
        "member_count_drift": member_count_drift,
        "missing_primary_rows": int(binding_summary[0]),
        "missing_fusion_rows": int(binding_summary[1]),
        "primary_probability_drift_rows": int(binding_summary[2]),
        "fusion_primary_drift_rows": int(binding_summary[3]),
        "formal_cancer_universe": len(coverage_cancers),
        "typed_unavailable_cancers": unavailable_cancers,
        "invalid_coverage_rows": invalid_coverage_rows,
    }
    zero_fields = [
        key for key in audit
        if key not in {
            "master_rows", "master_unique_ids", "cancers", "member_rows",
            "member_genesets", "formal_cancer_universe", "typed_unavailable_cancers"
        }
    ]
    if any(audit[key] != 0 for key in zero_fields):
        raise GeneSetParityReleaseError(f"Gene Set source audit failed: {audit}")
    if audit["master_rows"] != audit["master_unique_ids"] or (
        audit["master_rows"] != audit["member_genesets"]
    ):
        raise GeneSetParityReleaseError("Gene Set master/member closure failed")
    if strict_formal_authority and (
        audit["master_rows"] != FORMAL_GENESETS
        or audit["member_rows"] != FORMAL_MEMBER_ROWS
        or audit["cancers"] != FORMAL_AVAILABLE_CANCERS
        or audit["formal_cancer_universe"] != FORMAL_CANCERS
        or tuple(audit["typed_unavailable_cancers"]) != FORMAL_UNAVAILABLE_CANCERS
    ):
        raise GeneSetParityReleaseError("Formal Gene Set row/count authority drift")
    return audit


def materialize_gene_set_parity_release(
    *,
    master_path: str | Path,
    member_root: str | Path,
    gmt_path: str | Path,
    materialization_success_path: str | Path,
    coverage_path: str | Path,
    primary_path: str | Path,
    fusion_binding_path: str | Path,
    expected_fusion_binding_sha256: str,
    fusion_post_audit_path: str | Path,
    expected_fusion_post_audit_sha256: str,
    physical_manifest_path: str | Path,
    expected_physical_manifest_sha256: str,
    physical_post_audit_path: str | Path,
    expected_physical_post_audit_sha256: str,
    output_root: str | Path,
    strict_formal_authority: bool = True,
) -> dict[str, Any]:
    """Materialise the missing V3.2 Gene Set enrichment/report contract."""

    master_path = _safe_file(master_path, "Gene Set master")
    member_root = _safe_directory(member_root, "Gene Set member root")
    gmt_path = _safe_file(gmt_path, "Gene Set GMT")
    materialization_success_path = _safe_file(
        materialization_success_path, "Gene Set materialization success"
    )
    coverage_path = _safe_file(coverage_path, "Gene Set cancer coverage")
    primary_path = _safe_file(primary_path, "V3.2 primary exact-pathway predictions")
    fusion_binding_path, fusion_binding = _read_bound_json(
        fusion_binding_path, expected_fusion_binding_sha256, "fusion binding"
    )
    fusion_post_audit_path, fusion_post_audit = _read_bound_json(
        fusion_post_audit_path,
        expected_fusion_post_audit_sha256,
        "fusion independent post-audit",
    )
    physical_manifest_path, physical_manifest = _read_bound_json(
        physical_manifest_path,
        expected_physical_manifest_sha256,
        "physical interaction manifest",
    )
    physical_post_audit_path, physical_post_audit = _read_bound_json(
        physical_post_audit_path,
        expected_physical_post_audit_sha256,
        "physical interaction independent post-audit",
    )
    fusion_scores, physical_scores, source_summary = _validate_source_contracts(
        materialization_success_path=materialization_success_path,
        coverage_path=coverage_path,
        master_path=master_path,
        member_root=member_root,
        gmt_path=gmt_path,
        primary_path=primary_path,
        fusion_binding_path=fusion_binding_path,
        fusion_binding=fusion_binding,
        fusion_post_audit=fusion_post_audit,
        physical_manifest_path=physical_manifest_path,
        physical_manifest=physical_manifest,
        physical_post_audit=physical_post_audit,
        strict_formal_authority=strict_formal_authority,
    )

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise GeneSetParityReleaseError(f"Refusing output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)
    enrichment_path = output / "gene_set_enrichment.parquet"
    report_path = output / "GENE_SET_REPORT_MANIFEST.json"
    release_path = output / "GENE_SET_RELEASE_MANIFEST.json"
    success_path = output / "SUCCESS.json"

    member_relation, member_parts = _member_relation(member_root)
    master_relation = _parquet_relation(master_path)
    primary_relation = _parquet_relation(primary_path)
    fusion_relation = _parquet_relation(fusion_scores)
    physical_relation = _parquet_relation(physical_scores)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        audit = _audit_tables(
            connection,
            master=master_relation,
            members=member_relation,
            primary=primary_relation,
            fusion=fusion_relation,
            coverage_path=coverage_path,
            strict_formal_authority=strict_formal_authority,
        )
        query = f"""
        WITH member_fusion AS (
          SELECT
            m.geneset_id,
            m.lncrna_id,
            CAST(m.association_membership_probability AS DOUBLE) AS primary_probability,
            CAST(m.fold_selection_frequency AS DOUBLE) AS fold_selection_frequency,
            CAST(m.direction_stability AS DOUBLE) AS direction_stability,
            CAST(f.discovery_adjusted_probability AS DOUBLE) AS discovery_probability,
            CAST(f.fused_confidence_probability AS DOUBLE) AS confidence_probability,
            f.genomic_native_available,
            CAST(f.genomic_native_probability AS DOUBLE) AS genomic_native_probability,
            f.single_cell_native_available,
            CAST(f.single_cell_native_probability AS DOUBLE) AS single_cell_native_probability,
            f.evidence_transformer_native_available,
            CAST(f.evidence_transformer_native_probability AS DOUBLE)
              AS evidence_transformer_native_probability
          FROM {member_relation} m
          JOIN {fusion_relation} f USING (cancer_id, lncrna_id, pathway_id)
        ),
        fusion_summary AS (
          SELECT
            geneset_id,
            count(*) AS member_count,
            avg(primary_probability) AS mean_membership_probability,
            max(primary_probability) AS max_membership_probability,
            min(primary_probability) AS min_membership_probability,
            avg(fold_selection_frequency) AS mean_fold_selection_frequency,
            avg(direction_stability) AS mean_direction_stability,
            avg(discovery_probability) AS mean_discovery_probability,
            avg(confidence_probability) AS mean_confidence_probability,
            count(*) FILTER (WHERE genomic_native_available) AS genomic_available_member_count,
            avg(genomic_native_probability) FILTER (WHERE genomic_native_available)
              AS mean_genomic_native_probability,
            count(*) FILTER (WHERE single_cell_native_available)
              AS single_cell_available_member_count,
            avg(single_cell_native_probability) FILTER (WHERE single_cell_native_available)
              AS mean_single_cell_native_probability,
            count(*) FILTER (WHERE evidence_transformer_native_available)
              AS evidence_available_member_count,
            avg(evidence_transformer_native_probability)
              FILTER (WHERE evidence_transformer_native_available)
              AS mean_evidence_native_probability,
            count(*) FILTER (
              WHERE genomic_native_available OR single_cell_native_available
                    OR evidence_transformer_native_available
            ) AS any_native_auxiliary_available_member_count
          FROM member_fusion
          GROUP BY geneset_id
        ),
        physical_available_candidates AS (
          SELECT DISTINCT
            m.geneset_id,
            m.lncrna_id
          FROM {member_relation} m
          JOIN {physical_relation} p
            ON p.lncrna_id = m.lncrna_id
           AND p.availability IS TRUE
           AND p.cancer_scope = {_sql_literal(GLOBAL_SCOPE)}
          UNION
          SELECT DISTINCT
            m.geneset_id,
            m.lncrna_id
          FROM {member_relation} m
          JOIN {physical_relation} p
            ON p.lncrna_id = m.lncrna_id
           AND p.availability IS TRUE
           AND p.cancer_scope = m.cancer_id
        ),
        physical_availability_summary AS (
          SELECT
            geneset_id,
            count(*) AS physical_available_member_count
          FROM physical_available_candidates
          GROUP BY geneset_id
        ),
        member_physical_candidates AS (
          SELECT
            m.geneset_id,
            m.lncrna_id,
            CAST(p.ora_fdr AS DOUBLE) AS physical_ora_fdr,
            CAST(p.fold_enrichment AS DOUBLE) AS physical_fold_enrichment
          FROM {member_relation} m
          JOIN {physical_relation} p
            ON p.lncrna_id = m.lncrna_id
           AND p.pathway_id = m.pathway_id
           AND p.availability IS TRUE
           AND p.cancer_scope = {_sql_literal(GLOBAL_SCOPE)}
          UNION ALL
          SELECT
            m.geneset_id,
            m.lncrna_id,
            CAST(p.ora_fdr AS DOUBLE) AS physical_ora_fdr,
            CAST(p.fold_enrichment AS DOUBLE) AS physical_fold_enrichment
          FROM {member_relation} m
          JOIN {physical_relation} p
            ON p.lncrna_id = m.lncrna_id
           AND p.pathway_id = m.pathway_id
           AND p.cancer_scope = m.cancer_id
           AND p.availability IS TRUE
        ),
        member_physical AS (
          SELECT
            geneset_id,
            lncrna_id,
            min(physical_ora_fdr) AS best_physical_ora_fdr,
            max(physical_fold_enrichment) AS max_physical_fold_enrichment
          FROM member_physical_candidates
          GROUP BY geneset_id, lncrna_id
        ),
        physical_support_summary AS (
          SELECT
            geneset_id,
            count(*) FILTER (WHERE best_physical_ora_fdr IS NOT NULL)
              AS physical_supported_member_count,
            min(best_physical_ora_fdr) AS best_physical_ora_fdr,
            max(max_physical_fold_enrichment) AS max_physical_fold_enrichment
          FROM member_physical
          GROUP BY geneset_id
        ),
        support_summary AS (
          SELECT
            s.*,
            coalesce(a.physical_available_member_count, 0)
              AS physical_available_member_count,
            coalesce(p.physical_supported_member_count, 0)
              AS physical_supported_member_count,
            p.best_physical_ora_fdr,
            p.max_physical_fold_enrichment,
            (s.single_cell_available_member_count + s.evidence_available_member_count
               + coalesce(p.physical_supported_member_count, 0))
              AS independent_support_channel_hits,
            (s.single_cell_available_member_count + s.evidence_available_member_count
               + coalesce(a.physical_available_member_count, 0))
              AS independent_support_available_member_channel_count
          FROM fusion_summary s
          LEFT JOIN physical_availability_summary a USING (geneset_id)
          LEFT JOIN physical_support_summary p USING (geneset_id)
        )
        SELECT
          g.geneset_id,
          g.geneset_name,
          g.cancer_id,
          g.pathway_id,
          g.pathway_family_id,
          g.direction,
          s.* EXCLUDE (geneset_id),
          (3 * s.member_count) AS independent_support_total_member_channel_count,
          (3 * s.member_count - s.independent_support_available_member_channel_count)
            AS independent_support_unavailable_member_channel_count,
          CASE
            WHEN s.independent_support_available_member_channel_count = 0 THEN NULL
            ELSE CAST(s.independent_support_channel_hits AS DOUBLE)
              / CAST(s.independent_support_available_member_channel_count AS DOUBLE)
          END AS independent_support_channel_fraction,
          CAST(s.independent_support_channel_hits AS DOUBLE)
            / CAST(3 * s.member_count AS DOUBLE)
            AS independent_support_channel_coverage_fraction,
          s.independent_support_available_member_channel_count > 0
            AS independent_support_availability,
          CASE
            WHEN s.independent_support_available_member_channel_count = 0
              THEN {_sql_literal(NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE)}
            ELSE NULL
          END AS independent_support_unavailable_reason,
          {_sql_literal(ENRICHMENT_FORMAT)} AS enrichment_format,
          'DESCRIPTIVE_MEMBER_SUPPORT_NOT_CIRCULAR_ORA' AS enrichment_scope,
          'NOT_APPLICABLE_SELECTION_DEFINED_BY_PRIMARY' AS statistical_test,
          FALSE AS used_for_primary_ranking,
          FALSE AS changes_primary_ranking,
          FALSE AS family_to_exact_broadcast,
          {_sql_literal(ANALYSIS_VERSION)} AS analysis_version,
          'V3.2_CURRENT_PRIMARY_PLUS_AUDITED_NATIVE_AUXILIARY_EVIDENCE' AS generation
        FROM {master_relation} g
        JOIN support_summary s USING (geneset_id)
        ORDER BY g.cancer_id, g.pathway_id, g.direction, g.geneset_id
        """
        _copy_query(connection, query, enrichment_path)
        output_summary = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT geneset_id) AS unique_ids,
                   count(DISTINCT cancer_id) AS cancers,
                   count_if(member_count <= 0) AS empty_sets,
                   count_if(mean_membership_probability < 0 OR
                            mean_membership_probability > 1) AS bad_probability,
                   count_if(used_for_primary_ranking OR changes_primary_ranking OR
                            family_to_exact_broadcast) AS semantic_violations,
                   count_if(
                     (independent_support_available_member_channel_count = 0 AND (
                        independent_support_channel_fraction IS NOT NULL OR
                        independent_support_availability IS NOT FALSE OR
                        independent_support_unavailable_reason IS DISTINCT FROM
                          {_sql_literal(NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE)}
                     )) OR
                     (independent_support_available_member_channel_count > 0 AND (
                        independent_support_channel_fraction IS NULL OR
                        independent_support_availability IS NOT TRUE OR
                        independent_support_unavailable_reason IS NOT NULL
                     ))
                   ) AS missingness_semantic_violations,
                   count_if(
                     independent_support_channel_hits >
                       independent_support_available_member_channel_count
                   ) AS numerator_denominator_violations,
                   count_if(
                     independent_support_total_member_channel_count <> 3 * member_count OR
                     independent_support_unavailable_member_channel_count < 0 OR
                     independent_support_unavailable_member_channel_count <>
                       independent_support_total_member_channel_count -
                         independent_support_available_member_channel_count
                   ) AS denominator_identity_violations,
                   count_if(
                     independent_support_channel_fraction IS NOT NULL AND
                     independent_support_channel_fraction NOT BETWEEN 0 AND 1
                   ) AS support_fraction_violations,
                   count_if(
                     independent_support_channel_coverage_fraction NOT BETWEEN 0 AND 1
                   ) AS coverage_fraction_violations,
                   count_if(enrichment_format <> {_sql_literal(ENRICHMENT_FORMAT)})
                     AS enrichment_format_violations,
                   count_if(
                     independent_support_available_member_channel_count = 0
                   ) AS zero_available_denominator_rows
            FROM {_parquet_relation(enrichment_path)}
            """
        ).fetchone()
    finally:
        connection.close()

    output_audit = {
        "rows": int(output_summary[0]),
        "unique_genesets": int(output_summary[1]),
        "cancers": int(output_summary[2]),
        "empty_sets": int(output_summary[3]),
        "bad_probability_rows": int(output_summary[4]),
        "semantic_violations": int(output_summary[5]),
        "missingness_semantic_violations": int(output_summary[6]),
        "numerator_denominator_violations": int(output_summary[7]),
        "denominator_identity_violations": int(output_summary[8]),
        "support_fraction_violations": int(output_summary[9]),
        "coverage_fraction_violations": int(output_summary[10]),
        "enrichment_format_violations": int(output_summary[11]),
        "zero_available_denominator_rows": int(output_summary[12]),
    }
    if (
        output_audit["rows"] != audit["master_rows"]
        or output_audit["unique_genesets"] != audit["master_rows"]
        or any(
            output_audit[key]
            for key in (
                "empty_sets",
                "bad_probability_rows",
                "semantic_violations",
                "missingness_semantic_violations",
                "numerator_denominator_violations",
                "denominator_identity_violations",
                "support_fraction_violations",
                "coverage_fraction_violations",
                "enrichment_format_violations",
            )
        )
    ):
        raise GeneSetParityReleaseError(f"Gene Set enrichment output audit failed: {output_audit}")

    part_records = [_artifact(part) for part in member_parts]
    member_tree_digest = hashlib.sha256(
        "\n".join(f"{Path(item['path']).name}\t{item['sha256']}" for item in part_records).encode()
    ).hexdigest()
    report = {
        "format": REPORT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
        "gene_sets": audit["master_rows"],
        "member_rows": audit["member_rows"],
        "cancers": audit["cancers"],
        "formal_cancer_universe": audit["formal_cancer_universe"],
        "typed_unavailable_cancers": audit["typed_unavailable_cancers"],
        "enrichment_rows": output_audit["rows"],
        "enrichment_semantics": "descriptive independent-support summary",
        "independent_support_missingness": (
            "NULL_PLUS_TYPED_REASON_WHEN_NO_MEMBER_CHANNEL_IS_AVAILABLE"
        ),
        "independent_support_fraction_denominator": (
            "AVAILABLE_SINGLE_CELL_EVIDENCE_AND_PHYSICAL_MEMBER_CHANNELS"
        ),
        "independent_support_coverage_fraction_denominator": (
            "THREE_TIMES_GENE_SET_MEMBER_COUNT"
        ),
        "circular_self_ora_performed": False,
        "mixed_list_exact_pathway_ora_remains_separate": True,
        "auxiliary_evidence_used_for_member_selection_or_ranking": False,
        "changes_primary_ranking": False,
        "family_to_exact_broadcast": False,
        "old_checkpoints_used": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "release_ready": False,
        "production_deployed": False,
        "source_audit": audit,
        "output_audit": output_audit,
    }
    _atomic_json(report_path, report)
    release = {
        "format": RELEASE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "gene_set",
        "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
        "new_v32_head": True,
        "initialization": "v32_core_frozen_new_head",
        "target_level": "exact_pathway",
        "required_artifact_ids": {
            "v32_gene_set_catalog": _artifact(master_path, rows=audit["master_rows"]),
            "v32_gene_set_member_matrix": {
                "path": str(member_root),
                "sha256_tree": member_tree_digest,
                "parts": len(part_records),
                "rows": audit["member_rows"],
                "part_artifacts": part_records,
            },
            "v32_gene_set_enrichment": _artifact(
                enrichment_path, rows=output_audit["rows"]
            ),
            "v32_gene_set_gmt": _artifact(gmt_path, rows=audit["master_rows"]),
            "v32_gene_set_report_manifest": _artifact(report_path),
            "v32_gene_set_cancer_coverage": _artifact(
                coverage_path, rows=audit["formal_cancer_universe"]
            ),
        },
        "sources": source_summary,
        "fusion_post_audit": _artifact(fusion_post_audit_path),
        "physical_post_audit": _artifact(physical_post_audit_path),
        "source_audit": audit,
        "output_audit": output_audit,
        "primary_score_preserved": True,
        "ranking_uses_auxiliary_evidence": False,
        "family_to_exact_broadcast": False,
        "old_checkpoints_used": False,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(release_path, release)
    success = {
        "status": release["status"],
        "format": RELEASE_FORMAT,
        "manifest": release_path.name,
        "manifest_sha256": artifact_sha256(release_path),
        "gene_sets": output_audit["rows"],
        "release_ready": False,
        "production_deployed": False,
    }
    _atomic_json(success_path, success)
    return release


__all__ = [
    "ANALYSIS_VERSION",
    "ENRICHMENT_FORMAT",
    "GeneSetParityReleaseError",
    "NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE",
    "RELEASE_FORMAT",
    "REPORT_FORMAT",
    "artifact_sha256",
    "materialize_gene_set_parity_release",
]
