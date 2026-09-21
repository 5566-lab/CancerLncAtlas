from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.single_cell_input_builder import (
    ANALYSIS_VERSION,
    EXACT_PATHWAY_COUNT,
    EXPECTED_CANCERS,
    SingleCellInputBuildConfig,
    SingleCellInputBuildError,
    build_lnc_celltype_summary,
    build_v32_single_cell_inputs,
    compute_exact_pathway_activity,
    compute_fresh_associations,
    validate_dataset_manifest,
    validate_exact_candidates,
    validate_exact_membership,
    validate_expression_facts,
    validate_membership_provenance,
)
from cc_hhgt.v32.single_cell_training import (
    normalise_dataset_manifest as trainer_normalise_manifest,
    normalise_single_cell_associations,
)


LEGACY_CANCERS = {
    "BLCA", "BRCA", "COAD", "LIHC", "LUAD", "OV", "PAAD", "PRAD", "STAD", "THCA"
}
SOURCE_NORMALIZED_CANCERS = {"HNSC", "LGG"}


def _manifest_frame() -> pd.DataFrame:
    rows = []
    for cancer in sorted(EXPECTED_CANCERS):
        if cancer in LEGACY_CANCERS:
            tier = "legacy counts"
            scale = "counts"
            generation = "SOURCE_LEGACY_COUNTS_REAUDITED"
        elif cancer in SOURCE_NORMALIZED_CANCERS:
            tier = "source-normalized"
            scale = "log_normalized"
            generation = "SOURCE_PUBLICATION_NORMALIZED_EXPRESSION"
        else:
            tier = "raw_counts"
            scale = "counts"
            generation = "SOURCE_RAW_EXPRESSION_COUNTS"
        feature_count = {"CESC": 32, "UCS": 39, "UVM": 446}.get(cancer, 2_500)
        rows.append(
            {
                "dataset_id": f"SC_{cancer}",
                "cancer_id": cancer,
                "expression_source_tier": tier,
                "measurement_scale": scale,
                "source_generation": generation,
                "formal_eligible": cancer not in {"CESC", "UCS", "UVM"},
                "quality_status": "PASS",
                "model_derived": False,
                "outcome_derived": False,
                "lncrna_feature_universe_count": feature_count,
                "donor_metadata_available": True,
            }
        )
    return pd.DataFrame(rows)


def _membership_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pathway_id": [f"PW:{index:04d}" for index in range(EXACT_PATHWAY_COUNT)],
            "gene_id": ["ENSGP1"] * EXACT_PATHWAY_COUNT,
        }
    )


def _candidate_frame() -> pd.DataFrame:
    pathways = [f"PW:{index:04d}" for index in range(EXACT_PATHWAY_COUNT)]
    rows = [
        {"cancer_id": "ACC", "lncrna_id": "LNC:ENSG_L1", "pathway_id": pathway}
        for pathway in pathways
    ]
    rows.extend(
        {
            "cancer_id": cancer,
            "lncrna_id": "LNC:ENSG_L1",
            "pathway_id": pathways[0],
        }
        for cancer in sorted(EXPECTED_CANCERS - {"ACC"})
    )
    return pd.DataFrame(rows)


def _expression_frame() -> pd.DataFrame:
    rows = []
    lnc_values = [0, 1, 8, 9]
    p1_values = [1, 2, 9, 10]
    p2_values = [10, 9, 2, 1]
    manifest = validate_dataset_manifest(_manifest_frame())
    source = manifest.set_index("cancer_id").expression_source_tier
    for cancer in sorted(EXPECTED_CANCERS):
        for donor, (lnc, p1, p2) in enumerate(
            zip(lnc_values, p1_values, p2_values, strict=True), start=1
        ):
            values = (
                [("ENSG_L1", "lncRNA", lnc), ("ENSGP1", "protein_coding", p1),
                 ("ENSGP2", "protein_coding", p2)]
            )
            for gene, gene_type, value in values:
                # Source-normalized values need not be integer, but integer fixtures
                # remain valid and keep the cross-tier expected correlations identical.
                rows.append(
                    {
                        "dataset_id": f"SC_{cancer}",
                        "cancer_id": cancer,
                        "donor_id": f"D{donor}",
                        "cell_type": "Malignant",
                        "cell_state": "overall",
                        "gene_id": gene,
                        "gene_type": gene_type,
                        "expression": value,
                        "n_cells": 10,
                        "n_detected_cells": min(int(value), 10),
                        "fixture_source_tier": source[cancer],
                    }
                )
    return pd.DataFrame(rows).drop(columns="fixture_source_tier")


def _write_membership_provenance(path, membership_path, *, sha: str | None = None) -> None:
    membership_sha = sha or artifact_sha256(membership_path)
    path.write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "pathway_target_level": "exact_pathway",
                "membership_restricted_to_current_v32_exact_pathways": True,
                "pathway_family_broadcast": False,
                "static_annotation_only_no_predictions_or_rankings": True,
                "historical_checkpoints_used": False,
                "historical_predictions_used": False,
                "historical_rankings_used": False,
                "artifacts": {
                    "mixed_query_exact_pathway_membership": {
                        "filename": membership_path.name,
                        "sha256": membership_sha,
                    }
                },
                "rows": {
                    "exact_pathway_membership": EXACT_PATHWAY_COUNT,
                    "pathway_metadata": EXACT_PATHWAY_COUNT,
                },
            }
        ),
        encoding="utf-8",
    )


def test_manifest_has_exact_33_source_tiers_and_quality_limitations() -> None:
    manifest = validate_dataset_manifest(_manifest_frame())
    assert set(manifest.cancer_id) == EXPECTED_CANCERS
    assert set(manifest.expression_source_tier) == {
        "raw_counts", "source_normalized", "legacy_counts"
    }
    limited = manifest.set_index("cancer_id").loc[["CESC", "UCS", "UVM"]]
    assert limited.feature_universe_status.eq("LIMITED").all()
    assert not limited.formal_eligible.any()
    assert limited.quality_flags.str.contains(
        "KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION"
    ).all()
    assert limited.quality_flags.str.contains("LOW_LNCRNA_FEATURE_UNIVERSE").all()
    assert manifest.loc[manifest.expression_source_tier.eq("raw_counts"), "source_tier"].eq(
        "primary_raw_count"
    ).all()
    assert manifest.loc[
        manifest.expression_source_tier.isin({"source_normalized", "legacy_counts"}),
        "source_tier",
    ].eq("approved").all()

    incomplete = _manifest_frame().loc[lambda frame: frame.cancer_id.ne("ACC")]
    with pytest.raises(SingleCellInputBuildError, match="exactly the 33"):
        validate_dataset_manifest(incomplete)


def test_expression_facts_reject_results_and_invalid_count_semantics() -> None:
    manifest = validate_dataset_manifest(_manifest_frame())
    expression = _expression_frame().loc[lambda frame: frame.cancer_id.eq("ACC")]
    clean = validate_expression_facts(
        expression, manifest, require_all_manifest_datasets=False
    )
    assert set(clean.gene_class) == {"lncRNA", "protein_coding"}
    assert set(clean.loc[clean.gene_class.eq("lncRNA"), "gene_id"]) == {"LNC:ENSG_L1"}

    with pytest.raises(SingleCellInputBuildError, match="historical/result-like"):
        validate_expression_facts(
            expression.assign(predicted_probability=0.7),
            manifest,
            require_all_manifest_datasets=False,
        )
    invalid = expression.copy()
    invalid["expression"] = invalid.expression.astype(float)
    invalid.loc[invalid.index[0], "expression"] = 0.25
    with pytest.raises(SingleCellInputBuildError, match="integer-valued"):
        validate_expression_facts(invalid, manifest, require_all_manifest_datasets=False)


def test_fresh_activity_and_association_use_expression_and_exact_ids_only() -> None:
    manifest = validate_dataset_manifest(_manifest_frame())
    expression = validate_expression_facts(
        _expression_frame().loc[lambda frame: frame.cancer_id.eq("ACC")],
        manifest,
        require_all_manifest_datasets=False,
    )
    membership = validate_exact_membership(_membership_frame())
    candidates = _candidate_frame().loc[lambda frame: frame.cancer_id.eq("ACC")]
    activity = compute_exact_pathway_activity(expression, membership)
    assert activity.pathway_id.nunique() == EXACT_PATHWAY_COUNT
    assert activity.activity_method.eq(
        "V3.2_PROTEIN_ONLY_WITHIN_PSEUDOBULK_RANK_MEAN_V1"
    ).all()
    lnc_summary = build_lnc_celltype_summary(expression)
    assert lnc_summary.lncrna_id.tolist() == ["LNC:ENSG_L1"]

    association = compute_fresh_associations(
        expression, activity, candidates,
        min_observations=3,
        chunk_size=257,
    )
    assert association.pathway_id.nunique() == EXACT_PATHWAY_COUNT
    assert association.generation.eq("V3.2_FRESH_FROM_EXPRESSION_FACTS").all()
    assert association.association_method.str.contains("FRESH_DONOR_PSEUDOBULK").all()
    assert association.rho.abs().gt(0).all()
    assert association.fdr.between(0, 1).all()
    assert "pathway_family_id" not in association


def test_membership_requires_exact_count_and_hash_bound_v32_provenance(tmp_path) -> None:
    membership_path = tmp_path / "exact_pathway_membership.parquet"
    provenance_path = tmp_path / "ASSET_MANIFEST.json"
    membership = _membership_frame()
    membership.to_parquet(membership_path, index=False)
    validate_exact_membership(membership)
    _write_membership_provenance(provenance_path, membership_path)
    result = validate_membership_provenance(
        membership_path, provenance_path, membership_rows=len(membership)
    )
    assert result["status"] == "PASS"
    assert result["membership_sha256"] == artifact_sha256(membership_path)

    _write_membership_provenance(provenance_path, membership_path, sha="0" * 64)
    with pytest.raises(SingleCellInputBuildError, match="not uniquely pinned"):
        validate_membership_provenance(
            membership_path, provenance_path, membership_rows=len(membership)
        )
    with pytest.raises(SingleCellInputBuildError, match="exactly 2135"):
        validate_exact_membership(membership.iloc[:-1])


def test_end_to_end_handoff_is_fresh_and_current_trainer_compatible(tmp_path) -> None:
    manifest_path = tmp_path / "dataset_manifest.tsv"
    expression_path = tmp_path / "gene_expression_facts.parquet"
    membership_path = tmp_path / "exact_pathway_membership.parquet"
    provenance_path = tmp_path / "ASSET_MANIFEST.json"
    candidates_path = tmp_path / "exact_candidates.parquet"
    output = tmp_path / "built_inputs"
    _manifest_frame().to_csv(manifest_path, sep="\t", index=False)
    _expression_frame().to_parquet(expression_path, index=False)
    _membership_frame().to_parquet(membership_path, index=False)
    _candidate_frame().to_parquet(candidates_path, index=False)
    _write_membership_provenance(provenance_path, membership_path)

    result = build_v32_single_cell_inputs(
        dataset_manifest_path=manifest_path,
        expression_facts_path=expression_path,
        exact_membership_path=membership_path,
        membership_provenance_path=provenance_path,
        exact_candidates_path=candidates_path,
        output_root=output,
        build_run_id="unit-fresh-33c",
        config=SingleCellInputBuildConfig(
            min_association_observations=3,
            association_chunk_size=257,
        ),
    )
    assert result["status"] == "SUCCESS_STANDARD_INPUTS_BUILT"
    assert result["counts"]["cancers"] == 33
    assert result["counts"]["exact_pathways"] == EXACT_PATHWAY_COUNT
    assert result["fresh_activity_recomputed"] is True
    assert result["fresh_association_recomputed"] is True
    assert result["association_33c_coverage_complete"] is True
    assert result["activity_33c_coverage_complete"] is True
    assert result["historical_single_cell_pathway_outputs_used"] is False
    assert result["formal_33c_training_ready"] is False
    assert result["release_ready"] is False
    assert result["production_deployed"] is False

    handoff = json.loads((output / "TRAINING_HANDOFF.json").read_text())
    assert handoff["schemas_are_consumable_by_single_cell_training"] is True
    assert handoff["boundary_report"]["boundary_expansion_required"] is False
    assert handoff["boundary_report"]["boundary_policy"] == (
        "V3.2_EXACT_33_CANCER_AUTHORITY"
    )
    assert handoff["formal_33c_authority_does_not_override_quality_gates"] is True
    recompute = json.loads((output / "RECOMPUTE_REQUIREMENTS.json").read_text())
    assert recompute["still_requires_cell_level_fresh_recompute"]["ucell"]["required"]
    assert not recompute["still_requires_cell_level_fresh_recompute"]["ucell"][
        "historical_sc_trajectory_output_allowed"
    ]

    trainer_manifest = trainer_normalise_manifest(
        pd.read_parquet(output / "dataset_manifest.current_trainer.parquet")
    )
    associations = normalise_single_cell_associations(
        pd.read_parquet(output / "single_cell_association.parquet"),
        trainer_manifest,
    )
    assert associations.formal_row.any()
    assert set(pd.read_parquet(output / "activity.parquet").pathway_id).issubset(
        set(pd.read_parquet(output / "exact_pathway_membership.parquet").pathway_id)
    )
    build_file = json.loads((output / "BUILD_SUCCESS.json").read_text())
    assert all(str(output.resolve()) in row["path"] for row in build_file["artifacts"].values())

    with pytest.raises(SingleCellInputBuildError, match="refuses output reuse"):
        build_v32_single_cell_inputs(
            dataset_manifest_path=manifest_path,
            expression_facts_path=expression_path,
            exact_membership_path=membership_path,
            membership_provenance_path=provenance_path,
            exact_candidates_path=candidates_path,
            output_root=output,
            build_run_id="unit-reuse-forbidden",
            config=SingleCellInputBuildConfig(min_association_observations=3),
        )

    forbidden_expression = tmp_path / "historical_pathway_predictions.parquet"
    _expression_frame().to_parquet(forbidden_expression, index=False)
    with pytest.raises(SingleCellInputBuildError, match="historical/model output"):
        build_v32_single_cell_inputs(
            dataset_manifest_path=manifest_path,
            expression_facts_path=forbidden_expression,
            exact_membership_path=membership_path,
            membership_provenance_path=provenance_path,
            exact_candidates_path=candidates_path,
            output_root=tmp_path / "must_not_build",
            build_run_id="unit-historical-input-forbidden",
            config=SingleCellInputBuildConfig(min_association_observations=3),
        )


def test_candidate_validation_refuses_predictions_and_nonexact_namespace() -> None:
    membership = validate_exact_membership(_membership_frame())
    candidates = _candidate_frame()
    validated = validate_exact_candidates(candidates, membership)
    assert validated.pathway_id.nunique() == EXACT_PATHWAY_COUNT
    with pytest.raises(SingleCellInputBuildError, match="historical/result-like"):
        validate_exact_candidates(candidates.assign(probability=0.9), membership)
    with pytest.raises(SingleCellInputBuildError, match="do not exactly match"):
        validate_exact_candidates(candidates.loc[candidates.pathway_id.ne("PW:0001")], membership)
