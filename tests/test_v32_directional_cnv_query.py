from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.directional_cnv_query import (
    ANALYSIS_VERSION,
    DirectionalCNVQueryAssetError,
    DirectionalCNVReleaseQuery,
)
from cc_hhgt.v32.release_registry import artifact_sha256


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)


def write_directional_cnv_binding(
    root: Path, *, invalid_available_null: bool = False
) -> tuple[Path, str, Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    coverage_rows = []
    for index, cancer in enumerate(CANCERS):
        available = cancer != "UCS"
        probability = None if not available else 0.1 + index / 100.0
        if invalid_available_null and cancer == "LUAD":
            probability = None
        prediction_rows.append(
            {
                "cancer_id": cancer,
                "lncrna_id": f"LNC:{cancer}",
                "pathway_id": "MSIGDB:REACTOME:P1",
                "cnv_context_probability": probability,
                "cnv_context_probability_sd": 0.01 if available else None,
                "cnv_patient_folds_with_prediction": 5 if available else 0,
                "cnv_patient_fold_count": 5,
                "cnv_available": available,
                "cnv_unavailable_reason": (
                    None if available else "INSUFFICIENT_CALLABLE_PATIENTS"
                ),
                "local_cnv_available_fold_count": 5 if available else 0,
                "pathway_cnv_available_fold_count": 5 if available else 0,
                "cnv_pair_callable_patients_across_oof": 50 if available else 0,
                "training_run_id": "V32_DIRECTIONAL_CNV_TEST",
                "analysis_version": ANALYSIS_VERSION,
                "module_id": "cnv",
                "target_level": "cancer_x_lncrna_x_exact_pathway_cnv_context",
                "prediction_format": (
                    "CC_HHGT_V3_2_DIRECTIONAL_CNV_TYPED_PREDICTIONS_V1"
                ),
                "changes_primary_ranking": False,
            }
        )
        coverage_rows.append(
            {
                "cancer_id": cancer,
                "candidate_rows": 1,
                "available_rows": int(available),
                "typed_unavailable_rows": int(not available),
                "mean_available_folds": 5.0 if available else 0.0,
                "mean_pair_callable_patients": 50.0 if available else 0.0,
            }
        )
    predictions = root / "directional_cnv_typed_predictions.parquet"
    coverage = root / "directional_cnv_coverage_33c.parquet"
    pd.DataFrame(prediction_rows).to_parquet(predictions, index=False)
    pd.DataFrame(coverage_rows).to_parquet(coverage, index=False)
    prediction_sha = artifact_sha256(predictions)
    coverage_sha = artifact_sha256(coverage)
    lineage = root / "LINEAGE.json"
    audit = root / "TRANSFORMATION_AUDIT.json"
    lineage.write_text(
        json.dumps(
            {
                "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_LINEAGE_V1",
                "status": "PASS",
                "mutation_features_used": False,
                "primary_ranking_changed": False,
            }
        ),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "format": (
                    "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_"
                    "TRANSFORMATION_AUDIT_V1"
                ),
                "status": "PASS",
                "checks": {
                    "source_prediction_hashes_recomputed": True,
                    "source_independent_audit_bound": True,
                    "signed_directional_source_only": True,
                    "mutation_features_used": False,
                    "superseded_combined_cnv_read": False,
                    "available_probability_finite": True,
                    "typed_unavailable_probability_null": True,
                    "primary_ranking_changed": False,
                },
                "prediction_sha256": prediction_sha,
                "coverage_sha256": coverage_sha,
            }
        ),
        encoding="utf-8",
    )
    binding = root / "DIRECTIONAL_CNV_WEBSITE_BINDING.json"
    binding.write_text(
        json.dumps(
            {
                "format": "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_BINDING_V1",
                "status": "PASS",
                "analysis_version": ANALYSIS_VERSION,
                "staging_queryable": True,
                "release_ready": False,
                "release_blocker": "CNV_ROUTER_INCREMENT_NOT_TESTED",
                "changes_primary_ranking": False,
                "artifacts": {
                    "predictions": {
                        "path": str(predictions),
                        "sha256": prediction_sha,
                        "rows": 33,
                    },
                    "coverage": {
                        "path": str(coverage),
                        "sha256": coverage_sha,
                        "rows": 33,
                    },
                    "lineage": {
                        "path": str(lineage),
                        "sha256": artifact_sha256(lineage),
                    },
                    "transformation_audit": {
                        "path": str(audit),
                        "sha256": artifact_sha256(audit),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    binding_sha = artifact_sha256(binding)
    independent_root = root / "independent_audit"
    independent_root.mkdir()
    independent_report = independent_root / "AUDIT.json"
    independent_report.write_text(
        json.dumps(
            {
                "format": (
                    "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_"
                    "INDEPENDENT_AUDIT_V1"
                ),
                "status": "PASS",
                "independent_of_materializer_implementation": True,
                "materializer_imported": False,
                "accepted_for_staging_api_integration": True,
                "checks": {
                    "artifact_hashes_recomputed": True,
                    "source_hashes_recomputed": True,
                    "coverage_rederived": True,
                    "signed_directional_source_only": True,
                    "mutation_columns_absent": True,
                    "typed_null_contract": True,
                    "probability_contract": True,
                    "five_fold_count_contract": True,
                    "superseded_combined_cnv_exposed": False,
                    "changes_primary_ranking": False,
                },
                "release_binding_sha256": binding_sha,
                "prediction_sha256": prediction_sha,
                "coverage_sha256": coverage_sha,
            }
        ),
        encoding="utf-8",
    )
    independent_binding = independent_root / "INDEPENDENT_AUDIT_BINDING.json"
    independent_binding.write_text(
        json.dumps(
            {
                "format": (
                    "CC_HHGT_V3_2_DIRECTIONAL_CNV_WEBSITE_"
                    "INDEPENDENT_AUDIT_BINDING_V1"
                ),
                "status": "PASS",
                "accepted_for_staging_api_integration": True,
                "release_binding": {"path": str(binding), "sha256": binding_sha},
                "report": {
                    "path": str(independent_report),
                    "sha256": artifact_sha256(independent_report),
                },
                "production_deployed": False,
                "release_ready": False,
            }
        ),
        encoding="utf-8",
    )
    return binding, binding_sha, independent_binding, artifact_sha256(independent_binding)


def test_directional_cnv_query_coverage_and_downloads_are_typed(tmp_path: Path) -> None:
    binding, digest, audit, audit_digest = write_directional_cnv_binding(tmp_path)
    query = DirectionalCNVReleaseQuery(
        binding,
        expected_sha256=digest,
        audit_binding_path=audit,
        expected_audit_sha256=audit_digest,
    )

    available = query.query(cancer_id="luad", availability=True)
    assert available["total_rows"] == 1
    assert available["results"][0]["availability"] is True
    assert available["results"][0]["context_probability"] == pytest.approx(0.26)
    unavailable = query.query(cancer_id="ucs", availability=False)
    assert unavailable["results"][0]["context_probability"] is None
    assert unavailable["results"][0]["unavailable_reason"]
    assert query.coverage()["count"] == 33
    assert query.provenance()["mutation_features_used"] is False
    assert query.provenance()["new_training_attestation"] is True
    assert query.provenance()["old_checkpoint_loaded"] is False
    assert query.provenance()["old_predictions_used_as_features"] is False
    assert query.provenance()["old_rankings_used_as_outputs"] is False

    entry = query.download_entry("cnv_context")
    resolved = query.resolve_download("cnv_associations")
    assert entry["status"] == "READY_FILE"
    assert entry["supersedes_historical_combined_cnv"] is True
    assert resolved["sha256"] == entry["file"]["sha256"]


def test_directional_cnv_query_rejects_hash_drift(tmp_path: Path) -> None:
    binding, digest, audit, audit_digest = write_directional_cnv_binding(tmp_path)
    query = DirectionalCNVReleaseQuery(
        binding,
        expected_sha256=digest,
        audit_binding_path=audit,
        expected_audit_sha256=audit_digest,
    )
    frame = pd.read_parquet(query.prediction_path)
    frame.loc[0, "cnv_context_probability"] = 0.99
    frame.to_parquet(query.prediction_path, index=False)
    with pytest.raises(DirectionalCNVQueryAssetError, match="artifact drifted"):
        query.query(cancer_id="ACC")


def test_directional_cnv_query_rejects_available_null_even_when_rehashed(
    tmp_path: Path,
) -> None:
    binding, digest, audit, audit_digest = write_directional_cnv_binding(
        tmp_path, invalid_available_null=True
    )
    with pytest.raises(DirectionalCNVQueryAssetError, match="content validation failed"):
        DirectionalCNVReleaseQuery(
            binding,
            expected_sha256=digest,
            audit_binding_path=audit,
            expected_audit_sha256=audit_digest,
        )
