from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.external_validation_query import (
    ExternalValidationQueryAssetError,
    ExternalValidationQueryInputError,
    ExternalValidationReleaseQuery,
)
from cc_hhgt.v32.external_validation_release import (
    ExternalValidationReleaseError,
    artifact_sha256,
    materialize_external_validation_release,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "materialize_v32_external_validation_release.py"


def _write_raw_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    specifications = {
        "lnc2cancer": pd.DataFrame(
            {"name": ["L1"], "cancer type": ["breast"], "methods": ["qPCR"],
             "pubmed id": ["111111"]}
        ),
        "lncrnadisease": pd.DataFrame(
            {"ncRNA Symbol": ["L4"], "Disease Name": ["lung cancer"],
             "Validated Method//Prediction Method": ["experiment"],
             "PubMed ID": ["333333"]}
        ),
        "rnadisease_experimental": pd.DataFrame(
            {"RDID": ["RD1"], "RNA Symbol": ["L5"],
             "Disease Name": ["colon cancer"], "PMID": ["222222"]}
        ),
        "rnadisease_predicted": pd.DataFrame(
            {"RDID": ["RD2"], "RNA_symbol": ["L2"],
             "disease_name": ["breast cancer"], "method_name": ["RF"]}
        ),
        "gse85011": pd.DataFrame(
            {"dataset_accession": ["GSE85011"], "gsm": ["GSM1"],
             "cell_line": ["MCF7"], "target_raw": ["L1"], "pmid": ["666666"]}
        ),
    }
    paths: dict[str, Path] = {}
    for role, frame in specifications.items():
        path = root / f"{role}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[role] = path
    return paths


def _task_row(
    source: str,
    row_id: str,
    lnc: str | None,
    cancer: str | None,
    pmid: str | None,
    *,
    experimental: bool,
    predicted: bool,
) -> dict[str, object]:
    return {
        "source_database": source,
        "source_row_id": row_id,
        "lncrna_raw": lnc,
        "disease_raw": cancer,
        "pmid": pmid,
        "evidence_tier": "experimental" if experimental else "predicted",
        "is_experimental": experimental,
        "is_predicted": predicted,
        "dataset_accession": "GSE85011" if source == "GSE85011" else None,
        "cell_line": "MCF7" if source == "GSE85011" else None,
        "growth_modifier_hit": source == "GSE85011",
        "lncrna_id": lnc,
        "lncrna_mapping_status": "MAPPED" if lnc else "UNMAPPED",
        "cancer_id": cancer,
        "cancer_mapping_status": "MAPPED" if cancer else "UNMAPPED",
        # Deliberately poisoned legacy-derived values. The new materializer must
        # never read any of these columns.
        "training_pmid_overlap": pmid not in {"222222", "555555"},
        "validation_role": "POISONED_LEGACY_ROLE",
        "primary_validation_eligible": False,
        "consistency_validation_eligible": False,
        "external_evidence_id": "OLD-" + row_id,
        "independent_event_hash": "OLD-HASH-" + row_id,
        "analysis_version": "CancerLncAtlas_V2.3_STALE",
    }


def _build_fixture(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    raw = _write_raw_sources(root / "raw")
    task_rows = [
        _task_row("Lnc2Cancer", "1", "LNC:L1", "BRCA", "111111",
                  experimental=True, predicted=False),
        _task_row("Lnc2Cancer", "2", "LNC:L1", "BRCA", "111112",
                  experimental=True, predicted=False),
        _task_row("Lnc2Cancer", "3", "LNC:L2", "BRCA", "222222",
                  experimental=True, predicted=False),
        _task_row("LncRNADisease", "4", "LNC:L3", "LUAD", None,
                  experimental=True, predicted=False),
        _task_row("LncRNADisease", "5", "LNC:L4", "LUAD", "333333",
                  experimental=True, predicted=False),
        _task_row("LncRNADisease", "6", "LNC:L5", "COAD", "222222",
                  experimental=True, predicted=False),
        _task_row("RNADisease", "7", "LNC:L2", "BRCA", None,
                  experimental=False, predicted=True),
        _task_row("RNADisease", "8", "LNC:L4", "LUAD", None,
                  experimental=False, predicted=True),
        _task_row("RNADisease", "9", "LNC:L5", "COAD", "555555",
                  experimental=False, predicted=True),
        _task_row("GSE85011", "10", "LNC:L1", "BRCA", "666666",
                  experimental=True, predicted=False),
        _task_row("Lnc2Cancer", "11", None, None, "777777",
                  experimental=True, predicted=False),
    ]
    task = root / "external_validation_evidence.parquet"
    pd.DataFrame(task_rows).to_parquet(task, index=False)

    evidence = root / "evidence_event.parquet"
    interaction = root / "interaction_relation.parquet"
    pd.DataFrame({"pmid": ["PMID:222222", None]}).to_parquet(evidence, index=False)
    pd.DataFrame({"pmid": ["555555", "PMID 222222"]}).to_parquet(
        interaction, index=False
    )

    prediction_rows: list[dict[str, object]] = []
    probabilities = {
        ("BRCA", "LNC:L1"): [0.9, 0.5],
        ("BRCA", "LNC:L2"): [0.8, 0.7],
        ("BRCA", "LNC:L3"): [0.2, 0.1],
        ("LUAD", "LNC:L9"): [0.9, 0.8],
        ("COAD", "LNC:L10"): [0.5, 0.4],
    }
    for (cancer, lnc), values in probabilities.items():
        for index, probability in enumerate(values, start=1):
            prediction_rows.append(
                {
                    "cancer_id": cancer,
                    "lncrna_id": lnc,
                    "pathway_id": f"P{index}",
                    "association_membership_probability": probability,
                    "n_folds_available": 5,
                    "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                    "training_run_id": "FIXTURE-FIVE-FOLD",
                }
            )
    prediction = root / "five_fold_prediction.parquet"
    pd.DataFrame(prediction_rows).to_parquet(prediction, index=False)
    candidate = root / "candidate.parquet"
    pd.DataFrame(prediction_rows)[["cancer_id", "lncrna_id", "pathway_id"]].to_parquet(
        candidate, index=False
    )
    lineage = root / "MODULE_LINEAGE.json"
    lineage.write_text(
        json.dumps(
            {
                "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "five_fold_ensemble": True,
                "folds": 5,
                "trained_from_scratch": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "old_rankings_used_as_outputs": False,
                "prediction_rows": len(prediction_rows),
                "prediction_sha256": artifact_sha256(prediction),
                "training_run_id": "FIXTURE-FIVE-FOLD",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "raw": raw,
        "task": task,
        "evidence": evidence,
        "interaction": interaction,
        "prediction": prediction,
        "lineage": lineage,
        "candidate": candidate,
    }


def _materialize(root: Path):
    inputs = _build_fixture(root / "inputs")
    output = root / "release"
    result = materialize_external_validation_release(
        task_definition_path=inputs["task"],
        raw_source_paths=inputs["raw"],
        training_evidence_path=inputs["evidence"],
        training_interaction_path=inputs["interaction"],
        current_prediction_path=inputs["prediction"],
        current_prediction_lineage_path=inputs["lineage"],
        current_candidate_path=inputs["candidate"],
        output_root=output,
        runner_path=RUNNER,
        strict_formal_authority=False,
    )
    return inputs, output, result


def test_materializer_recomputes_overlap_and_rank_without_legacy_results(
    tmp_path: Path,
) -> None:
    _, output, result = _materialize(tmp_path)
    assert result["counts"] == {
        "task_records": 11,
        "training_union_distinct_pmids": 2,
        "external_overlap_records": 3,
        "primary_eligible_records": 4,
        "primary_unique_positives": 3,
        "primary_rank_available": 2,
        "secondary_unique_positives": 2,
        "secondary_rank_available": 1,
        "detail_rows": 5,
        "metric_rows": 28,
        "available_metric_rows": 12,
        "overlap_audit_rows": 10,
        "fresh_rank_rows": 5,
        "source_summary_rows": 4,
        "rank_cutoffs": 4,
    }
    details = pd.read_parquet(output / "external_validation_cohort_details.parquet")
    primary_l1 = details.loc[
        details.validation_role.eq("primary_known_positive")
        & details.source_database.eq("Lnc2Cancer")
        & details.lncrna_id.eq("LNC:L1")
    ].iloc[0]
    assert primary_l1.evidence_record_count == 2
    assert primary_l1.independent_pmid_count == 2
    assert primary_l1["rank"] == 1
    assert primary_l1.lncrna_rank_score == pytest.approx(0.84)
    assert bool(primary_l1.recovered_at_10)
    assert not details.lncrna_id.eq("LNC:L5").any()

    overlap = pd.read_parquet(output / "external_validation_overlap_audit.parquet")
    poisoned = overlap.loc[overlap.pmid_normalized.eq("222222")]
    assert poisoned.training_pmid_overlap.all()
    assert poisoned.excluded_from_primary.all()
    audit = json.loads((output / "TASK_DEFINITION_AUDIT.json").read_text())
    assert audit["old_external_result_artifacts_opened"] is False
    assert "training_pmid_overlap" not in audit["columns_consumed"]
    assert "training_pmid_overlap" in audit[
        "historical_derived_columns_explicitly_ignored"
    ]


def test_hash_bound_query_returns_cohorts_details_and_overlap(tmp_path: Path) -> None:
    _, output, result = _materialize(tmp_path)
    query = ExternalValidationReleaseQuery(
        output / "EXTERNAL_VALIDATION_BINDING.json",
        expected_binding_sha256=result["binding_sha256"],
        require_formal_authority=False,
    )
    metrics = query.query_metrics(
        validation_role="primary_known_positive",
        source_database="lnc2cancer",
        cancer_id="brca",
        k=100,
    )
    assert metrics["returned_rows"] == 1
    assert metrics["rows"][0]["hits_at_k"] == 1
    assert metrics["fresh_rank_recovery_calculation"] is True
    assert metrics["pmid_overlap_recomputed"] is True
    detail = query.query_lncrna(lncrna_id="l1.7", cancer_id="BRCA")
    assert detail["returned_rows"] == 2
    assert all(row["rank"] == 1 for row in detail["rows"])
    overlap = query.query_overlap(pmid="PMID:222222", training_pmid_overlap=True)
    assert overlap["returned_rows"] == 2
    assert overlap["provenance"]["historical_metrics_used"] is False
    assert overlap["provenance"]["historical_predictions_used"] is False


def test_query_and_materializer_fail_closed_on_hash_drift_or_overwrite(
    tmp_path: Path,
) -> None:
    inputs, output, result = _materialize(tmp_path)
    binding = output / "EXTERNAL_VALIDATION_BINDING.json"
    with pytest.raises(ExternalValidationQueryAssetError, match="SHA mismatch"):
        ExternalValidationReleaseQuery(
            binding,
            expected_binding_sha256="0" * 64,
            require_formal_authority=False,
        )
    with pytest.raises(ExternalValidationReleaseError, match="overwrite"):
        materialize_external_validation_release(
            task_definition_path=inputs["task"],
            raw_source_paths=inputs["raw"],
            training_evidence_path=inputs["evidence"],
            training_interaction_path=inputs["interaction"],
            current_prediction_path=inputs["prediction"],
            current_prediction_lineage_path=inputs["lineage"],
            current_candidate_path=inputs["candidate"],
            output_root=output,
            runner_path=RUNNER,
            strict_formal_authority=False,
        )
    raw = inputs["raw"]["lnc2cancer"]
    raw.write_text(raw.read_text(encoding="utf-8") + "L2\tbreast\tqPCR\t888888\n")
    with pytest.raises(ExternalValidationQueryAssetError, match="source SHA drift"):
        ExternalValidationReleaseQuery(
            binding,
            expected_binding_sha256=result["binding_sha256"],
            require_formal_authority=False,
        )


def test_strict_formal_authority_and_query_inputs_fail_closed(tmp_path: Path) -> None:
    inputs = _build_fixture(tmp_path / "strict-inputs")
    with pytest.raises(ExternalValidationReleaseError, match="raw source .* SHA256"):
        materialize_external_validation_release(
            task_definition_path=inputs["task"],
            raw_source_paths=inputs["raw"],
            training_evidence_path=inputs["evidence"],
            training_interaction_path=inputs["interaction"],
            current_prediction_path=inputs["prediction"],
            current_prediction_lineage_path=inputs["lineage"],
            current_candidate_path=inputs["candidate"],
            output_root=tmp_path / "strict-release",
            runner_path=RUNNER,
            strict_formal_authority=True,
        )
    _, output, result = _materialize(tmp_path / "query")
    query = ExternalValidationReleaseQuery(
        output / "EXTERNAL_VALIDATION_BINDING.json",
        expected_binding_sha256=result["binding_sha256"],
        require_formal_authority=False,
    )
    with pytest.raises(ExternalValidationQueryInputError, match="lncrna_id"):
        query.query_lncrna(lncrna_id="")
    with pytest.raises(ExternalValidationQueryInputError, match="validation_role"):
        query.query_metrics(validation_role="old_primary")
    with pytest.raises(ExternalValidationQueryInputError, match="k"):
        query.query_metrics(k=20)
    with pytest.raises(ExternalValidationQueryInputError, match="pmid"):
        query.query_overlap(pmid="not-a-pmid")
    with pytest.raises(ExternalValidationQueryInputError, match="limit"):
        query.query_metrics(limit=1_001)


def test_task_definition_with_historical_result_columns_is_rejected(
    tmp_path: Path,
) -> None:
    inputs = _build_fixture(tmp_path / "inputs")
    task = pd.read_parquet(inputs["task"])
    task["old_hits_at_100"] = 999
    task.to_parquet(inputs["task"], index=False)
    with pytest.raises(ExternalValidationReleaseError, match="forbidden historical"):
        materialize_external_validation_release(
            task_definition_path=inputs["task"],
            raw_source_paths=inputs["raw"],
            training_evidence_path=inputs["evidence"],
            training_interaction_path=inputs["interaction"],
            current_prediction_path=inputs["prediction"],
            current_prediction_lineage_path=inputs["lineage"],
            current_candidate_path=inputs["candidate"],
            output_root=tmp_path / "release",
            runner_path=RUNNER,
            strict_formal_authority=False,
        )


def test_same_source_context_can_have_primary_and_secondary_roles(
    tmp_path: Path,
) -> None:
    inputs = _build_fixture(tmp_path / "inputs")
    task = pd.read_parquet(inputs["task"])
    collision = _task_row(
        "Lnc2Cancer",
        "12",
        "LNC:L1",
        "BRCA",
        None,
        experimental=False,
        predicted=True,
    )
    task = pd.concat([task, pd.DataFrame([collision])], ignore_index=True)
    task.to_parquet(inputs["task"], index=False)
    output = tmp_path / "release"
    materialize_external_validation_release(
        task_definition_path=inputs["task"],
        raw_source_paths=inputs["raw"],
        training_evidence_path=inputs["evidence"],
        training_interaction_path=inputs["interaction"],
        current_prediction_path=inputs["prediction"],
        current_prediction_lineage_path=inputs["lineage"],
        current_candidate_path=inputs["candidate"],
        output_root=output,
        runner_path=RUNNER,
        strict_formal_authority=False,
    )
    details = pd.read_parquet(output / "external_validation_cohort_details.parquet")
    collision_rows = details.loc[
        details.source_database.eq("Lnc2Cancer")
        & details.cancer_id.eq("BRCA")
        & details.lncrna_id.eq("LNC:L1")
    ]
    assert set(collision_rows.validation_role) == {
        "primary_known_positive", "secondary_consistency"
    }
    secondary = collision_rows.loc[
        collision_rows.validation_role.eq("secondary_consistency")
    ].iloc[0]
    assert secondary.overlapping_records_excluded == 0
    assert secondary.missing_pmid_records_excluded == 0
