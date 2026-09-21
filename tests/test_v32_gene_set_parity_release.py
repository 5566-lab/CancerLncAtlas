from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.gene_set_parity_release import (
    ANALYSIS_VERSION,
    ENRICHMENT_FORMAT,
    GeneSetParityReleaseError,
    NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE,
    artifact_sha256,
    materialize_gene_set_parity_release,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    sources = tmp_path / "sources"
    sources.mkdir()
    members = sources / "members"
    members.mkdir()
    master = sources / "master.parquet"
    member_part = members / "cancer_id=BRCA.parquet"
    primary = sources / "primary.parquet"
    fusion = sources / "fusion.parquet"
    physical = sources / "physical.parquet"
    gmt = sources / "genesets.gmt"
    materialization_success = sources / "MATERIALIZATION_SUCCESS.json"
    coverage = sources / "CANCER_GENESET_COVERAGE.tsv"
    fusion_binding = sources / "FUSION_BINDING.json"
    fusion_audit = sources / "FUSION_AUDIT.json"
    physical_manifest = sources / "PHYSICAL_MANIFEST.json"
    physical_audit = sources / "PHYSICAL_AUDIT.json"

    pd.DataFrame(
        [
            {
                "geneset_id": "GS1",
                "geneset_name": "BRCA__P1__POSITIVE",
                "geneset_type": "cancer_exact_pathway_ranked",
                "cancer_id": "BRCA",
                "pathway_id": "P1",
                "pathway_family_id": "PF1",
                "direction": "positive",
                "member_count": 2,
                "shared_member_count": 2,
                "local_member_count": 0,
                "analysis_version": "V3_2_ONESEED_FORMAL_RELEASE",
                "pathway_target_level": "exact_pathway",
                "ranking_uses_regulatory_evidence": False,
            }
        ]
    ).to_parquet(master, index=False)
    member_rows = []
    for rank, lnc, probability in [(1, "LNC:ENSG1", 0.9), (2, "LNC:ENSG2", 0.7)]:
        member_rows.append(
            {
                "geneset_id": "GS1",
                "geneset_rank": rank,
                "cancer_id": "BRCA",
                "lncrna_id": lnc,
                "pathway_id": "P1",
                "pathway_family_id": "PF1",
                "association_membership_probability": probability,
                "association_direction": "positive",
                "l1_probability": probability,
                "ridge_probability": probability,
                "graph_residual": 0.1,
                "graph_gate": 0.2,
                "fold_rank_percentile": 0.8,
                "fold_selection_frequency": 1.0,
                "shared_or_local_scope": "shared",
                "regulatory_evidence_confidence": 0.5,
                "direction_stability": 1.0,
                "n_folds_available": 5,
                "pathway_target_level": "exact_pathway",
                "analysis_version": "V3_2_ONESEED_FORMAL_RELEASE",
            }
        )
    pd.DataFrame(member_rows).to_parquet(member_part, index=False)
    pd.DataFrame(
        [
            {
                "cancer_id": row["cancer_id"],
                "lncrna_id": row["lncrna_id"],
                "pathway_id": row["pathway_id"],
                "association_membership_probability": row[
                    "association_membership_probability"
                ],
            }
            for row in member_rows
        ]
    ).to_parquet(primary, index=False)
    pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG1",
                "pathway_id": "P1",
                "primary_probability": 0.9,
                "discovery_adjusted_probability": 0.88,
                "fused_confidence_probability": 0.92,
                "genomic_native_available": True,
                "genomic_native_probability": 0.8,
                "single_cell_native_available": True,
                "single_cell_native_probability": 0.7,
                "evidence_transformer_native_available": False,
                "evidence_transformer_native_probability": None,
            },
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG2",
                "pathway_id": "P1",
                "primary_probability": 0.7,
                "discovery_adjusted_probability": 0.69,
                "fused_confidence_probability": 0.71,
                "genomic_native_available": False,
                "genomic_native_probability": None,
                "single_cell_native_available": False,
                "single_cell_native_probability": None,
                "evidence_transformer_native_available": True,
                "evidence_transformer_native_probability": 0.6,
            },
        ]
    ).to_parquet(fusion, index=False)
    pd.DataFrame(
        [
            {
                "cancer_scope": "GLOBAL_NOT_CANCER_SPECIFIC",
                "lncrna_id": "LNC:ENSG1",
                "pathway_id": "P1",
                "ora_fdr": 0.01,
                "fold_enrichment": 3.0,
                "availability": True,
            }
        ]
    ).to_parquet(physical, index=False)
    gmt.write_text("BRCA__P1__POSITIVE\tfixture\tLNC:ENSG1\tLNC:ENSG2\n", encoding="utf-8")
    _write_json(
        materialization_success,
        {
            "status": "SUCCESS",
            "family_used_as_target": False,
            "regulatory_evidence_used_for_ranking": False,
            "genesets": 1,
            "subtype_member_rows": 2,
            "cancers": 1,
        },
    )
    coverage.write_text(
        "cancer_id\tpublishable_genesets\texact_pathways\tmember_rows\t"
        "positive_genesets\tnegative_genesets\tcoverage_status\n"
        "BRCA\t1\t1\t2\t1\t0\tAVAILABLE\n",
        encoding="utf-8",
    )
    _write_json(
        fusion_binding,
        {
            "format": "CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_SECONDARY_FUSION",
            "primary_score_preserved": True,
            "primary_ranking_unchanged": True,
            "native_expert_probabilities_public": True,
            "public_native_experts": [
                "genomic",
                "single_cell",
                "evidence_transformer",
            ],
            "release_ready": False,
            "production_deployed": False,
            "artifacts": {
                "secondary_scores": {
                    "path": str(fusion),
                    "sha256": artifact_sha256(fusion),
                    "rows": 2,
                }
            },
        },
    )
    _write_json(
        fusion_audit,
        {"status": "PASS", "fail_count": 0, "accepted_for_api_integration": True},
    )
    _write_json(
        physical_manifest,
        {
            "format": "CC_HHGT_V3_2_PHYSICAL_INTERACTION_RELEASE_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_NEWLY_MATERIALIZED_V32",
            "family_to_exact_broadcast": False,
            "old_checkpoints_used": False,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "release_ready": False,
            "production_deployed": False,
            "artifacts": {
                "interaction_exact_pathway_enrichment.parquet": {
                    "path": physical.name,
                    "sha256": artifact_sha256(physical),
                    "rows": 1,
                }
            },
        },
    )
    _write_json(physical_audit, {"status": "PASS", "fail_count": 0})
    return {
        "master_path": master,
        "member_root": members,
        "gmt_path": gmt,
        "materialization_success_path": materialization_success,
        "coverage_path": coverage,
        "primary_path": primary,
        "fusion_binding_path": fusion_binding,
        "expected_fusion_binding_sha256": artifact_sha256(fusion_binding),
        "fusion_post_audit_path": fusion_audit,
        "expected_fusion_post_audit_sha256": artifact_sha256(fusion_audit),
        "physical_manifest_path": physical_manifest,
        "expected_physical_manifest_sha256": artifact_sha256(physical_manifest),
        "physical_post_audit_path": physical_audit,
        "expected_physical_post_audit_sha256": artifact_sha256(physical_audit),
    }


def _make_native_support_unavailable(
    args: dict[str, Path | str], *, physical_lncrna: str, physical_pathway: str
) -> None:
    fusion_binding_path = Path(args["fusion_binding_path"])
    fusion_binding = json.loads(fusion_binding_path.read_text(encoding="utf-8"))
    fusion_path = Path(fusion_binding["artifacts"]["secondary_scores"]["path"])
    fusion = pd.read_parquet(fusion_path)
    for prefix in ("genomic", "single_cell", "evidence_transformer"):
        fusion[f"{prefix}_native_available"] = False
        fusion[f"{prefix}_native_probability"] = None
    fusion.to_parquet(fusion_path, index=False)
    fusion_binding["artifacts"]["secondary_scores"]["sha256"] = artifact_sha256(
        fusion_path
    )
    _write_json(fusion_binding_path, fusion_binding)
    args["expected_fusion_binding_sha256"] = artifact_sha256(fusion_binding_path)

    physical_manifest_path = Path(args["physical_manifest_path"])
    physical_manifest = json.loads(
        physical_manifest_path.read_text(encoding="utf-8")
    )
    physical_path = physical_manifest_path.parent / physical_manifest["artifacts"][
        "interaction_exact_pathway_enrichment.parquet"
    ]["path"]
    pd.DataFrame(
        [
            {
                "cancer_scope": "GLOBAL_NOT_CANCER_SPECIFIC",
                "lncrna_id": physical_lncrna,
                "pathway_id": physical_pathway,
                "ora_fdr": 0.02,
                "fold_enrichment": 2.0,
                "availability": True,
            }
        ]
    ).to_parquet(physical_path, index=False)
    physical_manifest["artifacts"][
        "interaction_exact_pathway_enrichment.parquet"
    ]["sha256"] = artifact_sha256(physical_path)
    _write_json(physical_manifest_path, physical_manifest)
    args["expected_physical_manifest_sha256"] = artifact_sha256(
        physical_manifest_path
    )


def test_materializes_missing_enrichment_without_changing_primary(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    output = tmp_path / "release"
    release = materialize_gene_set_parity_release(
        **args, output_root=output, strict_formal_authority=False
    )
    enrichment = pd.read_parquet(output / "gene_set_enrichment.parquet")
    assert release["status"] == "SUCCESS_NEWLY_MATERIALIZED_V32"
    assert set(release["required_artifact_ids"]) == {
        "v32_gene_set_catalog",
        "v32_gene_set_member_matrix",
        "v32_gene_set_enrichment",
        "v32_gene_set_gmt",
        "v32_gene_set_report_manifest",
        "v32_gene_set_cancer_coverage",
    }
    assert len(enrichment) == 1
    assert enrichment.iloc[0].mean_membership_probability == pytest.approx(0.8)
    assert enrichment.iloc[0].single_cell_available_member_count == 1
    assert enrichment.iloc[0].evidence_available_member_count == 1
    assert enrichment.iloc[0].physical_available_member_count == 1
    assert enrichment.iloc[0].physical_supported_member_count == 1
    assert enrichment.iloc[0].independent_support_channel_hits == 3
    assert enrichment.iloc[0].independent_support_available_member_channel_count == 3
    assert enrichment.iloc[0].independent_support_total_member_channel_count == 6
    assert enrichment.iloc[0].independent_support_unavailable_member_channel_count == 3
    assert enrichment.iloc[0].independent_support_channel_fraction == pytest.approx(1.0)
    assert enrichment.iloc[0].independent_support_channel_coverage_fraction == pytest.approx(0.5)
    assert enrichment.iloc[0].independent_support_availability == True  # noqa: E712
    assert pd.isna(enrichment.iloc[0].independent_support_unavailable_reason)
    assert enrichment.iloc[0].enrichment_format == ENRICHMENT_FORMAT
    assert enrichment.iloc[0].changes_primary_ranking == False  # noqa: E712
    assert enrichment.iloc[0].statistical_test == "NOT_APPLICABLE_SELECTION_DEFINED_BY_PRIMARY"
    with pytest.raises(GeneSetParityReleaseError, match="output reuse"):
        materialize_gene_set_parity_release(
            **args, output_root=output, strict_formal_authority=False
        )


def test_all_independent_channels_unavailable_is_null_with_reason(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    _make_native_support_unavailable(
        args, physical_lncrna="LNC:OUTSIDE", physical_pathway="P2"
    )
    output = tmp_path / "release"
    release = materialize_gene_set_parity_release(
        **args, output_root=output, strict_formal_authority=False
    )
    row = pd.read_parquet(output / "gene_set_enrichment.parquet").iloc[0]
    assert row.independent_support_channel_hits == 0
    assert row.independent_support_available_member_channel_count == 0
    assert row.independent_support_total_member_channel_count == 6
    assert row.independent_support_unavailable_member_channel_count == 6
    assert pd.isna(row.independent_support_channel_fraction)
    assert row.independent_support_channel_coverage_fraction == pytest.approx(0.0)
    assert row.independent_support_availability == False  # noqa: E712
    assert (
        row.independent_support_unavailable_reason
        == NO_INDEPENDENT_SUPPORT_CHANNEL_AVAILABLE
    )
    assert release["output_audit"]["zero_available_denominator_rows"] == 1


def test_available_physical_channel_with_no_pathway_overlap_is_real_zero(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    _make_native_support_unavailable(
        args, physical_lncrna="LNC:ENSG1", physical_pathway="P2"
    )
    output = tmp_path / "release"
    materialize_gene_set_parity_release(
        **args, output_root=output, strict_formal_authority=False
    )
    row = pd.read_parquet(output / "gene_set_enrichment.parquet").iloc[0]
    assert row.physical_available_member_count == 1
    assert row.physical_supported_member_count == 0
    assert row.independent_support_channel_hits == 0
    assert row.independent_support_available_member_channel_count == 1
    assert row.independent_support_channel_fraction == pytest.approx(0.0)
    assert row.independent_support_availability == True  # noqa: E712
    assert pd.isna(row.independent_support_unavailable_reason)


def test_primary_probability_drift_fails_closed(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    primary = Path(args["primary_path"])
    frame = pd.read_parquet(primary)
    frame.loc[0, "association_membership_probability"] = 0.1
    frame.to_parquet(primary, index=False)
    with pytest.raises(GeneSetParityReleaseError, match="source audit failed"):
        materialize_gene_set_parity_release(
            **args,
            output_root=tmp_path / "release",
            strict_formal_authority=False,
        )
