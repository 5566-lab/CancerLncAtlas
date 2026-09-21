from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

import cc_hhgt.v32.drug_mechanism as drug_mechanism
from cc_hhgt.v32.drug_mechanism import (
    ANALYSIS_VERSION,
    DrugMechanismError,
    build_structural_mechanisms,
    materialize_drug_mechanisms,
)
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.drug_mechanism_audit import (
    DrugMechanismAuditError,
    audit_drug_mechanism_release,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG00000000001",
                "drug_id": "DRUG:1",
                "drug_response_association_probability": 0.8,
                "availability": True,
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": "V32-DRUG-TEST",
                "prediction_fold_mask": 3,
                "cell_line_folds_with_prediction": 2,
            }
        ]
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("BRCA", "LNC:ENSG00000000001", "P1"),
            ("BRCA", "LNC:ENSG00000000001", "P2"),
        ],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    )


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [("P1", "GENE:ENSG00000000101"), ("P2", "ENSG00000000102")],
        columns=["pathway_id", "gene_id"],
    )


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drug_id": "DRUG:1",
                "gene_id": "GENE:ENSG00000000101.2",
                "target_gene_symbol": "G1",
                "drug_name": "Drug One",
                "mapping_method": "HGNC",
            },
            {
                "drug_id": "DRUG:1",
                "gene_id": "GENE:ENSG00000000999",
                "target_gene_symbol": "OUT",
                "drug_name": "Drug One",
                "mapping_method": "HGNC",
            },
        ]
    )


def test_builds_structural_not_causal_mechanism() -> None:
    result = build_structural_mechanisms(
        _predictions(), _candidates(), _membership(), _targets()
    )
    assert len(result) == 1
    assert result.pathway_id.tolist() == ["P1"]
    assert result.target_gene_id.tolist() == ["ENSG00000000101"]
    assert not result.causal_mechanism_claimed.any()
    assert not result.target_contribution_claimed.any()
    assert not result.family_to_exact_broadcast.any()
    assert result.mechanism_path_id.str.startswith("DRUGMECH32:").all()


def test_rejects_private_columns_and_old_analysis_version() -> None:
    private = _predictions().assign(held_out_rho=0.4)
    with pytest.raises(DrugMechanismError, match="Private"):
        build_structural_mechanisms(private, _candidates(), _membership(), _targets())
    old = _predictions().assign(analysis_version="CancerLncAtlas_V3.1")
    with pytest.raises(DrugMechanismError, match="current V3.2"):
        build_structural_mechanisms(old, _candidates(), _membership(), _targets())


def test_materializes_out_of_core_fixture_and_refuses_reuse(tmp_path: Path) -> None:
    prediction_root = tmp_path / "v32_predictions"
    prediction_root.mkdir()
    _predictions().to_parquet(prediction_root / "part.parquet", index=False)
    for frame, name in (
        (_candidates(), "candidates.parquet"),
        (_membership(), "membership.parquet"),
        (_targets(), "targets.parquet"),
    ):
        frame.to_parquet(tmp_path / name, index=False)
    output = tmp_path / "mechanism_release"
    manifest = materialize_drug_mechanisms(
        predictions_path=prediction_root,
        candidates_path=tmp_path / "candidates.parquet",
        membership_path=tmp_path / "membership.parquet",
        drug_targets_path=tmp_path / "targets.parquet",
        output_root=output,
        strict_formal_authority=False,
    )
    assert manifest["counts"]["mechanism_paths"] == 1
    assert manifest["native_target_keys"] == ["cancer_id", "lncrna_id", "drug_id"]
    assert manifest["actionability_separate_from_exact_pathway"] is True
    assert manifest["does_not_change_primary_pathway_ranking"] is True
    assert manifest["drug_response_association_probability_not_efficacy_or_direction"] is True
    assert manifest["signed_rho_private_only"] is True
    result = pd.read_parquet(output / "lncrna_exact_pathway_target_drug_mechanisms.parquet")
    assert len(result) == 1
    assert not result.causal_mechanism_claimed.any()
    with pytest.raises(DrugMechanismError, match="output reuse"):
        materialize_drug_mechanisms(
            predictions_path=prediction_root,
            candidates_path=tmp_path / "candidates.parquet",
            membership_path=tmp_path / "membership.parquet",
            drug_targets_path=tmp_path / "targets.parquet",
            output_root=output,
            strict_formal_authority=False,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("native_target_keys", ["cancer_id", "lncrna_id", "pathway_id"]),
        ("actionability_separate_from_exact_pathway", False),
        ("does_not_change_primary_pathway_ranking", False),
        ("drug_response_association_probability_not_efficacy_or_direction", False),
        ("signed_rho_private_only", False),
    ],
)
def test_independent_audit_fails_closed_on_score_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: object,
) -> None:
    paths = _write_formal_bundle(tmp_path)
    candidate_sha = artifact_sha256(paths["candidate"])
    membership_sha = artifact_sha256(paths["membership"])
    monkeypatch.setattr(drug_mechanism, "FORMAL_CANDIDATE_SHA256", candidate_sha)
    monkeypatch.setattr(drug_mechanism, "FORMAL_MEMBERSHIP_SHA256", membership_sha)
    output = tmp_path / "strict_mechanisms"
    materialize_drug_mechanisms(
        predictions_path=paths["prediction"],
        candidates_path=paths["candidate"],
        membership_path=paths["membership"],
        drug_targets_path=paths["target"],
        output_root=output,
    )
    manifest_path = output / "DRUG_MECHANISM_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = invalid
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DrugMechanismAuditError, match=f"invalid {field}"):
        audit_drug_mechanism_release(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
            expected_candidate_sha256=candidate_sha,
            expected_membership_sha256=membership_sha,
        )


def test_independent_audit_fails_closed_on_missing_score_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_formal_bundle(tmp_path)
    candidate_sha = artifact_sha256(paths["candidate"])
    membership_sha = artifact_sha256(paths["membership"])
    monkeypatch.setattr(drug_mechanism, "FORMAL_CANDIDATE_SHA256", candidate_sha)
    monkeypatch.setattr(drug_mechanism, "FORMAL_MEMBERSHIP_SHA256", membership_sha)
    output = tmp_path / "strict_mechanisms"
    materialize_drug_mechanisms(
        predictions_path=paths["prediction"],
        candidates_path=paths["candidate"],
        membership_path=paths["membership"],
        drug_targets_path=paths["target"],
        output_root=output,
    )
    manifest_path = output / "DRUG_MECHANISM_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("signed_rho_private_only")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DrugMechanismAuditError, match="invalid signed_rho_private_only"):
        audit_drug_mechanism_release(
            manifest_path,
            expected_manifest_sha256=artifact_sha256(manifest_path),
            expected_candidate_sha256=candidate_sha,
            expected_membership_sha256=membership_sha,
        )


def _write_formal_bundle(tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "fresh_drug_bundle"
    prediction_root = bundle / "drug_response_association"
    prediction_root.mkdir(parents=True)
    _predictions().to_parquet(prediction_root / "part.parquet", index=False)
    paths = {
        "bundle": bundle,
        "prediction": prediction_root,
        "candidate": tmp_path / "candidates.parquet",
        "membership": tmp_path / "membership.parquet",
        "target": tmp_path / "targets.parquet",
    }
    _candidates().to_parquet(paths["candidate"], index=False)
    _membership().to_parquet(paths["membership"], index=False)
    _targets().to_parquet(paths["target"], index=False)
    prediction_sha = artifact_sha256(prediction_root)
    lineage = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "drug",
        "training_run_id": "V32-DRUG-TEST",
        "training_status": "SUCCESS",
        "prediction_sha256": prediction_sha,
        "release_ready": True,
        "partial_not_publishable": False,
        "private_head_trained_from_scratch": True,
        "all_five_folds_have_optimizer_updates": True,
        "all_non_null_predictions_newly_trained_v32": True,
        "available_keys_unique": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_gdsc_prism_association_tables_used": False,
        "input_artifacts": [
            {
                "path": str(paths["target"].resolve()),
                "sha256": artifact_sha256(paths["target"]),
            }
        ],
    }
    lineage_path = bundle / "MODULE_LINEAGE.json"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    success = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "drug",
        "training_run_id": "V32-DRUG-TEST",
        "release_ready": True,
        "prediction_path": str(prediction_root.resolve()),
        "prediction_sha256": prediction_sha,
        "lineage_sha256": artifact_sha256(lineage_path),
    }
    (bundle / "SUCCESS.json").write_text(json.dumps(success), encoding="utf-8")
    return paths


def test_strict_materialization_binds_fresh_bundle_and_static_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_formal_bundle(tmp_path)
    monkeypatch.setattr(
        drug_mechanism, "FORMAL_CANDIDATE_SHA256", artifact_sha256(paths["candidate"])
    )
    monkeypatch.setattr(
        drug_mechanism, "FORMAL_MEMBERSHIP_SHA256", artifact_sha256(paths["membership"])
    )
    manifest = materialize_drug_mechanisms(
        predictions_path=paths["prediction"],
        candidates_path=paths["candidate"],
        membership_path=paths["membership"],
        drug_targets_path=paths["target"],
        output_root=tmp_path / "strict_mechanisms",
    )
    assert manifest["formal_drug_bundle_binding"]["training_run_id"] == "V32-DRUG-TEST"
    assert manifest["counts"]["source_available_predictions"] == 1
    assert manifest["native_target_keys"] == ["cancer_id", "lncrna_id", "drug_id"]
    assert manifest["actionability_separate_from_exact_pathway"] is True
    assert manifest["does_not_change_primary_pathway_ranking"] is True
    assert manifest["drug_response_association_probability_not_efficacy_or_direction"] is True
    assert manifest["signed_rho_private_only"] is True
    manifest_path = tmp_path / "strict_mechanisms" / "DRUG_MECHANISM_MANIFEST.json"
    audit = audit_drug_mechanism_release(
        manifest_path,
        expected_manifest_sha256=artifact_sha256(manifest_path),
        audit_output=tmp_path / "independent_mechanism_audit.json",
        expected_candidate_sha256=artifact_sha256(paths["candidate"]),
        expected_membership_sha256=artifact_sha256(paths["membership"]),
    )
    assert audit["status"] == "PASS"
    assert audit["semantic_declarations"]["native_target_keys"] == [
        "cancer_id", "lncrna_id", "drug_id"
    ]
    assert audit["semantic_declarations"]["actionability_separate_from_exact_pathway"] is True
    assert audit["semantic_declarations"]["does_not_change_primary_pathway_ranking"] is True
    assert audit["semantic_declarations"][
        "drug_response_association_probability_not_efficacy_or_direction"
    ] is True
    assert audit["semantic_declarations"]["signed_rho_private_only"] is True


@pytest.mark.parametrize("failure", ["prediction_hash", "target_binding"])
def test_strict_materialization_rejects_bundle_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    paths = _write_formal_bundle(tmp_path)
    monkeypatch.setattr(
        drug_mechanism, "FORMAL_CANDIDATE_SHA256", artifact_sha256(paths["candidate"])
    )
    monkeypatch.setattr(
        drug_mechanism, "FORMAL_MEMBERSHIP_SHA256", artifact_sha256(paths["membership"])
    )
    if failure == "prediction_hash":
        changed = _predictions().assign(drug_response_association_probability=0.7)
        changed.to_parquet(paths["prediction"] / "part.parquet", index=False)
    else:
        lineage_path = paths["bundle"] / "MODULE_LINEAGE.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        lineage["input_artifacts"][0]["sha256"] = "0" * 64
        lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
        success_path = paths["bundle"] / "SUCCESS.json"
        success = json.loads(success_path.read_text(encoding="utf-8"))
        success["lineage_sha256"] = artifact_sha256(lineage_path)
        success_path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(DrugMechanismError, match="SHA|hash-bound"):
        materialize_drug_mechanisms(
            predictions_path=paths["prediction"],
            candidates_path=paths["candidate"],
            membership_path=paths["membership"],
            drug_targets_path=paths["target"],
            output_root=tmp_path / f"rejected_{failure}",
        )


@pytest.mark.parametrize("corruption", ["old_version", "duplicate_key", "private_column"])
def test_out_of_core_contract_rejects_mixed_or_private_prediction_rows(
    tmp_path: Path, corruption: str
) -> None:
    prediction_root = tmp_path / "v32_predictions"
    prediction_root.mkdir()
    prediction = _predictions()
    if corruption == "old_version":
        prediction = pd.concat(
            [prediction, prediction.assign(drug_id="DRUG:2", analysis_version="CancerLncAtlas_V3.1")],
            ignore_index=True,
        )
    elif corruption == "duplicate_key":
        prediction = pd.concat([prediction, prediction], ignore_index=True)
    else:
        prediction["held_out_rho"] = 0.5
    prediction.to_parquet(prediction_root / "part.parquet", index=False)
    _candidates().to_parquet(tmp_path / "candidates.parquet", index=False)
    _membership().to_parquet(tmp_path / "membership.parquet", index=False)
    _targets().to_parquet(tmp_path / "targets.parquet", index=False)
    output = tmp_path / f"rejected_{corruption}"
    with pytest.raises(DrugMechanismError):
        materialize_drug_mechanisms(
            predictions_path=prediction_root,
            candidates_path=tmp_path / "candidates.parquet",
            membership_path=tmp_path / "membership.parquet",
            drug_targets_path=tmp_path / "targets.parquet",
            output_root=output,
            strict_formal_authority=False,
        )
    assert not output.exists()
