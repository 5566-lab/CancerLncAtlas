from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.drug_mechanism import (
    ANALYSIS_VERSION,
    FORMAL_CANDIDATE_SHA256,
    FORMAL_MEMBERSHIP_SHA256,
    materialize_drug_mechanisms,
)
from cc_hhgt.v32.drug_mechanism_query import (
    DrugMechanismQueryAssetError,
    DrugMechanismQueryInputError,
    DrugMechanismReleaseQuery,
)
from cc_hhgt.v32.release_registry import artifact_sha256


def _release(tmp_path: Path) -> tuple[Path, str]:
    prediction_root = tmp_path / "v32_predictions"
    prediction_root.mkdir(parents=True)
    pd.DataFrame(
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
    ).to_parquet(prediction_root / "part.parquet", index=False)
    candidates = pd.DataFrame(
        [("BRCA", "LNC:ENSG00000000001", "P1"), ("BRCA", "LNC:ENSG00000000001", "P2")],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    )
    membership = pd.DataFrame(
        [("P1", "GENE:ENSG00000000101"), ("P2", "ENSG00000000102")],
        columns=["pathway_id", "gene_id"],
    )
    targets = pd.DataFrame(
        [
            {
                "drug_id": "DRUG:1",
                "gene_id": "GENE:ENSG00000000101.2",
                "target_gene_symbol": "G1",
                "drug_name": "Drug One",
                "mapping_method": "HGNC",
            }
        ]
    )
    for frame, name in (
        (candidates, "candidates.parquet"),
        (membership, "membership.parquet"),
        (targets, "targets.parquet"),
    ):
        frame.to_parquet(tmp_path / name, index=False)
    output = tmp_path / "release"
    materialize_drug_mechanisms(
        predictions_path=prediction_root,
        candidates_path=tmp_path / "candidates.parquet",
        membership_path=tmp_path / "membership.parquet",
        drug_targets_path=tmp_path / "targets.parquet",
        output_root=output,
        strict_formal_authority=False,
    )
    manifest_path = output / "DRUG_MECHANISM_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_drug_bundle_binding"] = {
        "training_run_id": "V32-DRUG-TEST",
        "success_sha256": "1" * 64,
        "lineage_sha256": "2" * 64,
    }
    manifest["inputs"]["current_exact_candidates"]["sha256"] = FORMAL_CANDIDATE_SHA256
    manifest["inputs"]["current_exact_membership"]["sha256"] = FORMAL_MEMBERSHIP_SHA256
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, artifact_sha256(manifest_path)


def test_queries_structural_mechanisms_with_explicit_noncausal_semantics(
    tmp_path: Path,
) -> None:
    manifest, digest = _release(tmp_path)
    query = DrugMechanismReleaseQuery(manifest, expected_manifest_sha256=digest)
    result = query.query_mechanisms(
        lncrna_id="ENSG00000000001", cancer_id="brca", min_probability=0.7
    )
    assert result["returned_rows"] == 1
    assert result["rows"][0]["pathway_id"] == "P1"
    assert result["native_target_keys"] == ["cancer_id", "lncrna_id", "drug_id"]
    assert result["actionability_separate_from_exact_pathway"] is True
    assert result["does_not_change_primary_pathway_ranking"] is True
    assert result["drug_response_association_probability_not_efficacy_or_direction"] is True
    assert result["signed_rho_private_only"] is True
    assert result["mechanism_semantics"] == "STRUCTURAL_HYPOTHESIS_NOT_MODEL_ATTRIBUTION"
    assert result["causal_mechanism_claimed"] is False
    assert result["target_contribution_claimed"] is False
    assert result["provenance"]["historical_results_used"] is False


def test_query_requires_manifest_pin_and_formal_authorities(tmp_path: Path) -> None:
    manifest, digest = _release(tmp_path)
    with pytest.raises(DrugMechanismQueryAssetError, match="requires"):
        DrugMechanismReleaseQuery(manifest, expected_manifest_sha256=None)
    with pytest.raises(DrugMechanismQueryAssetError, match="mismatch"):
        DrugMechanismReleaseQuery(manifest, expected_manifest_sha256="0" * 64)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"]["current_exact_membership"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DrugMechanismQueryAssetError, match="formal authority"):
        DrugMechanismReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )
    assert digest != "0" * 64


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
def test_query_fails_closed_on_drug_score_semantic_drift(
    tmp_path: Path, field: str, invalid: object
) -> None:
    manifest, _ = _release(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = invalid
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DrugMechanismQueryAssetError, match=f"invalid {field}"):
        DrugMechanismReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )


def test_query_fails_closed_on_missing_drug_score_semantic(tmp_path: Path) -> None:
    manifest, _ = _release(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("signed_rho_private_only")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DrugMechanismQueryAssetError, match="invalid signed_rho_private_only"):
        DrugMechanismReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )


def test_query_rejects_artifact_hash_and_semantic_drift(tmp_path: Path) -> None:
    manifest, _ = _release(tmp_path)
    artifact = manifest.parent / "lncrna_exact_pathway_target_drug_mechanisms.parquet"
    frame = pd.read_parquet(artifact)
    frame.loc[0, "drug_response_association_probability"] = 0.1
    frame.to_parquet(artifact, index=False)
    with pytest.raises(DrugMechanismQueryAssetError, match="SHA drift"):
        DrugMechanismReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )

    manifest, _ = _release(tmp_path / "semantic")
    artifact = manifest.parent / "lncrna_exact_pathway_target_drug_mechanisms.parquet"
    frame = pd.read_parquet(artifact)
    frame.loc[0, "causal_mechanism_claimed"] = True
    frame.to_parquet(artifact, index=False)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = artifact_sha256(artifact)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DrugMechanismQueryAssetError, match="semantic"):
        DrugMechanismReleaseQuery(
            manifest, expected_manifest_sha256=artifact_sha256(manifest)
        )


def test_query_input_bounds_fail_closed(tmp_path: Path) -> None:
    manifest, digest = _release(tmp_path)
    query = DrugMechanismReleaseQuery(manifest, expected_manifest_sha256=digest)
    with pytest.raises(DrugMechanismQueryInputError, match="lncrna_id"):
        query.query_mechanisms(lncrna_id="")
    with pytest.raises(DrugMechanismQueryInputError, match="limit"):
        query.query_mechanisms(lncrna_id="ENSG1", limit=0)
    with pytest.raises(DrugMechanismQueryInputError, match="min_probability"):
        query.query_mechanisms(lncrna_id="ENSG1", min_probability=1.1)
