from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.web_relationship_materialization import (
    ANALYSIS_VERSION,
    INPUT_MANIFEST_FORMAT,
    OUTPUT_MANIFEST_FORMAT,
    TCGA_CANCERS,
    WebRelationshipMaterializationError,
    build_cancer_overview,
    classify_fresh_v32_relationships,
    materialize_fresh_v32_web_relationships,
)


def _primary() -> pd.DataFrame:
    probabilities = [0.65, 0.95, 0.88, 0.55]
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 4,
            "lncrna_id": [f"LNC:{index}" for index in range(4)],
            "pathway_id": [f"KEGG:P{index}" for index in range(4)],
            "pathway_family_id": ["PF:0001"] * 4,
            "association_membership_probability": probabilities,
            "association_direction_probability": [0.7, 0.4, 0.8, 0.3],
            "association_direction": ["positive", "negative", "positive", "negative"],
            "n_folds_available": [5] * 4,
            "analysis_version": [ANALYSIS_VERSION] * 4,
            "old_checkpoint_loaded": [False] * 4,
            "old_predictions_used_as_features": [False] * 4,
        }
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 4,
            "lncrna_id": [f"LNC:{index}" for index in range(4)],
            "pathway_id": [f"KEGG:P{index}" for index in range(4)],
            "evidence_confidence_probability": [0.8, None, 0.5, None],
            "availability": [True, False, True, False],
            "direct_target_evidence": [True, False, True, False],
            "family_to_exact_broadcast": [False] * 4,
            "direction": ["negative", "", "positive", ""],
            "event_count": [5, 0, 3, 0],
            "analysis_version": [ANALYSIS_VERSION] * 4,
        }
    )


def test_classification_is_mutually_exclusive_and_overview_is_honest() -> None:
    classified = classify_fresh_v32_relationships(
        _primary(),
        _evidence(),
        prediction_uncertainty=pd.Series([0.1, 0.1, 0.1, 0.1]),
    )
    assert classified.relationship_class.tolist() == [
        "observed_core",
        "predicted_candidate",
        "model_supported",
        "exploratory",
    ]
    assert classified.final_direction.tolist() == [
        "negative",
        "negative",
        "positive",
        "negative",
    ]
    assert classified.loc[classified.evidence_available, "observed_evidence_probability"].notna().all()
    assert classified.loc[~classified.evidence_available, "observed_evidence_probability"].isna().all()
    overview = build_cancer_overview(classified)
    row = overview.iloc[0]
    assert row.predicted_candidate_relations == 1
    assert row.non_exploratory_relations == 3
    assert row.significant_lncRNA_pathway_relations == 3
    assert "not statistical significance" in row.significant_relation_semantics
    assert row.pathway_target_level == "exact_pathway"
    assert not bool(row.family_to_exact_broadcast)


def test_old_or_private_columns_fail_closed() -> None:
    primary = _primary()
    primary["held_out_proxy_label"] = 0
    with pytest.raises(ValueError, match="Held-out/private"):
        classify_fresh_v32_relationships(
            primary,
            _evidence(),
            prediction_uncertainty=pd.Series([0.1] * 4),
        )


def test_family_broadcast_and_missing_uncertainty_fail_closed() -> None:
    evidence = _evidence()
    evidence.loc[0, "family_to_exact_broadcast"] = True
    with pytest.raises(ValueError, match="broadcast"):
        classify_fresh_v32_relationships(
            _primary(), evidence, prediction_uncertainty=pd.Series([0.1] * 4)
        )
    with pytest.raises(ValueError, match="uncertainty"):
        classify_fresh_v32_relationships(_primary(), _evidence())


def test_duplicate_keys_fail_closed() -> None:
    primary = pd.concat([_primary(), _primary().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate exact relationship keys"):
        classify_fresh_v32_relationships(
            primary,
            _evidence(),
            prediction_uncertainty=pd.Series([0.1] * len(primary)),
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flags() -> dict[str, bool]:
    return {
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
    }


def _write_formal_inputs(root: Path) -> tuple[Path, str, dict]:
    root.mkdir()
    cancers = list(TCGA_CANCERS)
    primary_run = "PRIMARY-FRESH-V32-RUN"
    evidence_run = "EVIDENCE-FRESH-V32-RUN"
    selection_sha = "1" * 64
    universe_sha = "2" * 64
    keys = {
        "cancer_id": cancers,
        "lncrna_id": [f"LNC:{index:02d}" for index in range(33)],
        "pathway_id": [f"KEGG:P{index:02d}" for index in range(33)],
    }
    primary = pd.DataFrame(
        {
            **keys,
            "pathway_family_id": ["PF:0001"] * 33,
            "association_membership_probability": [0.95, 0.86, 0.65] + [0.5] * 30,
            "association_direction_probability": [0.7] * 33,
            "association_direction": ["positive"] * 33,
            "n_folds_available": [5] * 33,
            "analysis_version": [ANALYSIS_VERSION] * 33,
            "training_run_id": [primary_run] * 33,
            **{field: [value] * 33 for field, value in _flags().items()},
        }
    )
    evidence_probability = [None] * 33
    evidence_probability[1] = 0.5
    evidence_probability[2] = 0.8
    availability = [False] * 33
    availability[1] = True
    availability[2] = True
    evidence = pd.DataFrame(
        {
            **keys,
            "evidence_confidence_probability": evidence_probability,
            "availability": availability,
            "direct_target_evidence": availability,
            "family_to_exact_broadcast": [False] * 33,
            "direction": ["", "negative", "positive"] + [""] * 30,
            "event_count": [0, 4, 5] + [0] * 30,
            "analysis_version": [ANALYSIS_VERSION] * 33,
            "training_run_id": [evidence_run] * 33,
            **{field: [value] * 33 for field, value in _flags().items()},
        }
    )
    primary_path = root / "primary.parquet"
    evidence_path = root / "evidence.parquet"
    primary.to_parquet(primary_path, index=False)
    evidence.to_parquet(evidence_path, index=False)
    common = {
        "analysis_version": ANALYSIS_VERSION,
        "selection_sha256": selection_sha,
        "candidate_universe_sha256": universe_sha,
        **_flags(),
    }
    fold_declarations = []
    for fold_id, delta in enumerate((-0.02, -0.01, 0.0, 0.01, 0.02)):
        frame = pd.DataFrame(
            {
                **keys,
                "association_membership_probability": np.clip(
                    primary.association_membership_probability.to_numpy() + delta,
                    0,
                    1,
                ),
                "fold_id": [fold_id] * 33,
                "analysis_version": [ANALYSIS_VERSION] * 33,
                "training_run_id": [primary_run] * 33,
                **{field: [value] * 33 for field, value in _flags().items()},
            }
        )
        fold_path = root / f"fold_{fold_id}.parquet"
        frame.to_parquet(fold_path, index=False)
        fold_declarations.append(
            {
                "role": "independent_fold_prediction",
                "fold_id": fold_id,
                "path": fold_path.name,
                "sha256": _sha(fold_path),
                "training_run_id": primary_run,
                "checkpoint_sha256": f"{fold_id + 3:x}" * 64,
                "training_partition_sha256": f"{fold_id + 8:x}" * 64,
                **common,
            }
        )
    manifest = {
        "manifest_format": INPUT_MANIFEST_FORMAT,
        "status": "PASS",
        "analysis_version": ANALYSIS_VERSION,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "family_to_exact_broadcast": False,
        "primary_score_status": "FINAL_SELECTED",
        "selection_status": "PASS",
        "selection_split": "validation_only",
        "test_metrics_used_for_selection": False,
        "expected_cancers": 33,
        "expected_cancer_ids": list(TCGA_CANCERS),
        "required_folds": list(range(5)),
        "primary_training_run_id": primary_run,
        "evidence_training_run_id": evidence_run,
        "selection_sha256": selection_sha,
        "candidate_universe_sha256": universe_sha,
        **_flags(),
        "artifacts": {
            "primary": {
                "role": "final_selected_primary",
                "path": primary_path.name,
                "sha256": _sha(primary_path),
                "training_run_id": primary_run,
                **common,
            },
            "exact_evidence": {
                "role": "exact_target_evidence",
                "path": evidence_path.name,
                "sha256": _sha(evidence_path),
                "training_run_id": evidence_run,
                "family_to_exact_broadcast": False,
                **common,
            },
            "fold_predictions": fold_declarations,
        },
    }
    manifest_path = root / "INPUT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, _sha(manifest_path), manifest


def test_low_memory_materializer_computes_unlabelled_fold_std_and_stages_only(
    tmp_path: Path,
) -> None:
    manifest_path, manifest_sha, _ = _write_formal_inputs(tmp_path / "inputs")
    output = tmp_path / "staging"
    result = materialize_fresh_v32_web_relationships(
        input_manifest_path=manifest_path,
        expected_input_manifest_sha256=manifest_sha,
        output_root=output,
        memory_limit="128MB",
    )
    assert result["status"] == "PASS"
    assert result["production_deployed"] is False
    assert result["release_ready"] is False
    relationships = pd.read_parquet(output / "web_exact_pathway_relationships.parquet")
    assert len(relationships) == 33
    assert relationships.relationship_class.iloc[:3].tolist() == [
        "predicted_candidate",
        "model_supported",
        "observed_core",
    ]
    assert relationships.prediction_uncertainty.iloc[0] == pytest.approx(
        np.std([0.93, 0.94, 0.95, 0.96, 0.97], ddof=1)
    )
    assert not (set(relationships.columns) & {"label", "held_out_proxy_label"})
    assert not relationships.family_to_exact_broadcast.any()
    assert relationships.old_predictions_used_as_features.eq(False).all()
    assert relationships.loc[
        relationships.evidence_available, "observed_evidence_probability"
    ].notna().all()
    assert relationships.loc[
        ~relationships.evidence_available, "observed_evidence_probability"
    ].isna().all()
    selected = pd.read_parquet(output / "web_exact_pathway_selected.parquet")
    assert selected.relationship_class.tolist() == [
        "predicted_candidate",
        "model_supported",
        "observed_core",
    ]
    overview = pd.read_parquet(output / "web_cancer_overview.parquet")
    assert len(overview) == 33
    assert overview.significant_lncRNA_pathway_relations.sum() == 3
    materialization = json.loads(
        (output / "MATERIALIZATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    assert materialization["manifest_format"] == OUTPUT_MANIFEST_FORMAT
    assert materialization["staging_only"] is True
    assert materialization["production_deployed"] is False
    assert success["manifest_sha256"] == _sha(output / "MATERIALIZATION_MANIFEST.json")
    for declaration in materialization["artifacts"].values():
        assert _sha(output / declaration["path"]) == declaration["sha256"]


def test_materializer_rejects_key_drift_and_never_emits_success(tmp_path: Path) -> None:
    manifest_path, _, manifest = _write_formal_inputs(tmp_path / "inputs")
    fold_path = manifest_path.parent / "fold_4.parquet"
    fold = pd.read_parquet(fold_path)
    fold.loc[0, "pathway_id"] = "KEGG:WRONG"
    fold.to_parquet(fold_path, index=False)
    manifest["artifacts"]["fold_predictions"][4]["sha256"] = _sha(fold_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = tmp_path / "failed"
    with pytest.raises(WebRelationshipMaterializationError, match="Candidate-key mismatch"):
        materialize_fresh_v32_web_relationships(
            input_manifest_path=manifest_path,
            expected_input_manifest_sha256=_sha(manifest_path),
            output_root=output,
            memory_limit="128MB",
        )
    assert output.is_dir()
    assert not (output / "SUCCESS.json").exists()


def test_materializer_rejects_manifest_hash_old_reuse_and_output_reuse(
    tmp_path: Path,
) -> None:
    manifest_path, manifest_sha, manifest = _write_formal_inputs(tmp_path / "inputs")
    with pytest.raises(WebRelationshipMaterializationError, match="manifest SHA-256 mismatch"):
        materialize_fresh_v32_web_relationships(
            input_manifest_path=manifest_path,
            expected_input_manifest_sha256="f" * 64,
            output_root=tmp_path / "hash-failure",
            memory_limit="128MB",
        )
    manifest["historical_predictions_used"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(WebRelationshipMaterializationError, match="historical_predictions_used=false"):
        materialize_fresh_v32_web_relationships(
            input_manifest_path=manifest_path,
            expected_input_manifest_sha256=_sha(manifest_path),
            output_root=tmp_path / "old-failure",
            memory_limit="128MB",
        )
    # Recreate a clean declaration, then prove a pre-existing target is never reused.
    manifest["historical_predictions_used"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(WebRelationshipMaterializationError, match="Refusing to overwrite"):
        materialize_fresh_v32_web_relationships(
            input_manifest_path=manifest_path,
            expected_input_manifest_sha256=_sha(manifest_path),
            output_root=existing,
            memory_limit="128MB",
        )
