from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cc_hhgt.v32.continuous_activity import (
    CHECKPOINT_FORMAT,
    ContinuousActivityConfig,
    audit_frozen_lnc_embeddings,
    fit_fresh_pca_ridge_head,
)


def _synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(72, 18))
    coefficient = np.zeros((18, 7))
    coefficient[:5] = rng.normal(size=(5, 7))
    y = x @ coefficient + rng.normal(scale=0.08, size=(72, 7))
    return x[:44], y[:44], x[44:58], y[44:58], x[58:], y[58:]


def test_fresh_continuous_head_is_outer_test_safe_and_writes_checkpoint(tmp_path) -> None:
    values = _synthetic()
    config = ContinuousActivityConfig(
        n_components=8,
        ridge_alphas=(0.01, 0.1, 1.0, 10.0),
        top_attributions=4,
    )
    fit = fit_fresh_pca_ridge_head(
        *values,
        feature_ids=[f"LNC:{index}" for index in range(18)],
        pathway_ids=[f"PATH:{index}" for index in range(7)],
        seed=20260826,
        config=config,
    )
    altered = list(values)
    altered[-1] = values[-1] * -999.0
    replay = fit_fresh_pca_ridge_head(
        *altered,
        feature_ids=[f"LNC:{index}" for index in range(18)],
        pathway_ids=[f"PATH:{index}" for index in range(7)],
        seed=20260826,
        config=config,
    )
    np.testing.assert_allclose(fit.test_prediction, replay.test_prediction)
    assert fit.model.final_parameter_sha256 == replay.model.final_parameter_sha256
    assert fit.model.initial_parameter_sha256 != fit.model.final_parameter_sha256
    assert fit.model.selected_alpha in config.ridge_alphas
    assert fit.metrics.outer_test_used_for_tuning.eq(False).all()
    assert len(fit.attributions) == 7 * 4
    assert fit.attributions.groupby("pathway_id").attribution_rank.max().eq(4).all()

    checkpoint = tmp_path / "head.npz"
    fit.model.write(checkpoint)
    loaded = np.load(checkpoint)
    metadata = json.loads(str(loaded["metadata_json"]))
    assert metadata["checkpoint_format"] == CHECKPOINT_FORMAT
    assert metadata["new_parameters_from_scratch"] is True
    assert metadata["old_checkpoint_loaded"] is False
    assert loaded["ridge_coefficients"].shape == (8, 7)


def test_constant_frozen_lnc_embeddings_are_rejected_for_patient_projection() -> None:
    frame = pd.DataFrame(
        {
            "node_id": ["LNC:A", "LNC:B", "LNC:C"],
            "core_feature_000": [0.5, 0.5, 0.5],
            "core_feature_001": [0.0, 0.0, 0.0],
        }
    )
    audit = audit_frozen_lnc_embeddings(frame)
    assert audit["status"] == "UNUSABLE_FOR_PATIENT_PROJECTION"
    assert audit["unique_embedding_vectors"] == 1
    assert audit["usable_for_patient_expression_projection"] is False
    assert audit["fallback"] == "FRESH_FOLD_LOCAL_EXPRESSION_PCA"


def test_varying_frozen_lnc_embeddings_pass_usability_audit() -> None:
    frame = pd.DataFrame(
        {
            "node_id": ["LNC:A", "LNC:B", "LNC:C"],
            "core_feature_000": [0.5, 0.2, 0.9],
            "core_feature_001": [0.0, 1.0, 0.3],
        }
    )
    audit = audit_frozen_lnc_embeddings(frame)
    assert audit["status"] == "PASS"
    assert audit["unique_embedding_vectors"] == 3
    assert audit["usable_for_patient_expression_projection"] is True
