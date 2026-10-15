"""Native label replay and fusion development semantics; no optimizer is run."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_primary_fusion import (FORMAT, SCOPE, LABEL, AVAILABLE, REASON,
    extract_validation_labels, materialize_primary_fusion_inputs, load_primary_fusion_inputs)
from cc_hhgt.v32 import multimodal_fusion as fusion


def record(path): return {"path": str(path), "sha256": sha256(path)}
def read(path): return json.loads(Path(path).read_text())
def write(path, value): path.write_text(json.dumps(value)); return path


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*a, **kw): raise AssertionError("No optimizer or training is allowed in this suite")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(fusion, "fit_residual_fusion_model", forbidden)
    import scripts.run_v32_current_multimodal_fusion as runner
    monkeypatch.setattr(runner, "train_pair_blocked_crossfit_fusion", forbidden)


def payload_fixture():
    from test_v32_current_core_prediction import tiny_features
    payload, candidates = tiny_features()
    payload.update(patient_fold=2, input_scope={"candidate_rows_per_split": 2,
        "discovery_features_split": "train_patients_only", "replication_labels_splits": ["validation"],
        "held_out_effects_as_features": False, "sealed_test_accessed": False})
    batch = payload["validation_batches"][0]
    batch.update(proxy_label=torch.tensor([0., 0.]), association_available=torch.tensor([False, True]))
    return payload, candidates


def test_extract_preserves_real_zero_mask_and_candidate_mapping_without_reading_train_labels():
    payload, candidates = payload_fixture()
    frame = extract_validation_labels(payload, candidates, fold=2)
    assert frame[LABEL].iloc[0] == 0 and pd.isna(frame[LABEL].iloc[1])
    assert frame[AVAILABLE].tolist() == [True, False]
    assert pd.isna(frame[REASON].iloc[0]) and frame[REASON].iloc[1]
    assert frame.validation_patient_fold_id.eq(3).all()
    assert frame.fusion_supervision_scope.eq(SCOPE).all()


@pytest.mark.parametrize("fault", ["missing_mask", "float_mask", "mask_shape", "invalid_label", "sealed", "scope", "feature", "fold"])
def test_extract_rejects_invented_availability_or_heldout_scope(fault):
    payload, candidates = payload_fixture(); batch = payload["validation_batches"][0]
    if fault == "missing_mask": del batch["association_available"]
    if fault == "float_mask": batch["association_available"] = torch.ones(2)
    if fault == "mask_shape": batch["association_available"] = torch.tensor([True])
    if fault == "invalid_label": batch["proxy_label"][1] = float("nan")
    if fault == "sealed": payload["sealed_test_labels"] = object()
    if fault == "scope": payload["input_scope"]["replication_labels_splits"] = ["test"]
    if fault == "feature": batch["base_logit"][0] += 1
    if fault == "fold": payload["patient_fold"] = 1
    with pytest.raises(ValueError): extract_validation_labels(payload, candidates, fold=2)


def native_fixture(tmp_path, monkeypatch):
    from test_v32_current_core_prediction import current_fixture
    import test_v32_current_binding_expansion as fixtures
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    original = fixtures.verified_payload
    def payload(fold=0):
        value = original(fold)
        value["validation_batches"][0]["proxy_label"] = torch.tensor([float(fold % 2)])
        value["validation_batches"][0]["association_available"] = torch.tensor([fold != 1])
        return value
    monkeypatch.setattr(fixtures, "verified_payload", payload)
    manifest = current_fixture(tmp_path/"native")
    primary = materialize_current_predictions(manifest, tmp_path/"prediction", expected_run_id="CURRENT-V4-SYNTHETIC")
    from scripts.materialize_v32_current_primary_fusion import main
    assert main(["--binding", str(primary), "--expected-sha256", sha256(primary),
        "--expected-core-run-id", "CURRENT-V4-SYNTHETIC", "--output", str(tmp_path/"labels")]) == 0
    return tmp_path/"labels/CURRENT_PRIMARY_FUSION_BINDING.json"


def test_native_checkpoint_to_label_replay_and_current_fusion_cli(tmp_path, monkeypatch):
    path = native_fixture(tmp_path, monkeypatch)
    binding = load_primary_fusion_inputs(path, expected_sha=sha256(path), expected_core_run_id="CURRENT-V4-SYNTHETIC")
    assert binding["format"] == FORMAT and binding["supervision_scope"] == SCOPE
    assert not binding["sealed_test_labels_used"] and not binding["independent_validation_claimed"]
    for item in binding["folds"]:
        data = pd.read_parquet(item["artifact"]["path"])
        assert data.validation_patient_fold_id.eq((item["fold_id"]+1) % 5).all()
        assert bool(data[AVAILABLE].iloc[0]) == (item["fold_id"] != 1)
    keys = pd.read_parquet(binding["prediction"]["path"])[list(fusion.TARGET_KEYS)]
    expert_path = tmp_path/"expert.parquet"
    keys.assign(value=-.4, available=True, analysis_version=fusion.ANALYSIS_VERSION).to_parquet(expert_path, index=False)
    spec = fusion.ExpertSpec("signed", "value", "available", ("discovery", "confidence"), "donor", value_transform="signed_fisher")
    audit = write(tmp_path/"expert.json", {"status": "PASS", "run_id": "synthetic-donor-model", "estimand": "SIGNED_ASSOCIATION",
        "core_dependency": "INDEPENDENT_MODEL", "expert_specs": [asdict(spec)], "outputs": [record(expert_path)]})
    manifest = write(tmp_path/"fusion.json", {"format": "CURRENT_MULTIMODAL_FUSION_MANIFEST_V1",
        **read(path.parent/"FUSION_PRIMARY_INPUTS.json"),
        "experts": [{"spec": asdict(spec), "run_id": "synthetic-donor-model", "estimand": "SIGNED_ASSOCIATION",
            "binding_audit": record(audit), "prediction": record(expert_path)}]})
    import scripts.run_v32_current_multimodal_fusion as runner
    assert runner.main(["--manifest", str(manifest), "--expected-core-run-id", binding["run_id"],
        "--output", str(tmp_path/"prepared")]) == 0
    frame = pd.read_parquet(tmp_path/"prepared/PRIVATE_FUSION_INPUT.parquet")
    assert frame.fusion_target.iloc[0] == .25  # fold 1 was unavailable; not 2/5
    assert frame.fusion_target_available_fold_count.iloc[0] == 4
    audit = read(tmp_path/"prepared/INPUT_AUDIT.json")
    assert audit["primary_label_authority"] == "NATIVE_VALIDATION_REPLAY" and audit["target_partial_fold_rows"] == 1
    assert audit["native_label_authority_verified"] and not audit["eligible_for_current_fitting"]
    assert not (tmp_path/"prepared/SUCCESS.json").exists()
    # Even consistently rehashing an edited private label does not evade replay.
    item = binding["folds"][0]; table = Path(item["artifact"]["path"])
    changed = pd.read_parquet(table); changed[LABEL] = 1.; changed.to_parquet(table, index=False)
    item["artifact"] = record(table); write(path, binding)
    write(path.parent/"SUCCESS.json", {"status": "PASS", "binding_sha256": sha256(path)})
    with pytest.raises(ValueError, match="native validation replay"):
        load_primary_fusion_inputs(path, expected_sha=sha256(path), expected_core_run_id=binding["run_id"])


def parquet_fixture(tmp_path):
    keys = pd.DataFrame({"cancer_id": "BRCA", "lncrna_id": [f"L{i:03}" for i in range(80)], "pathway_id": "P1"})
    primary = tmp_path/"primary.parquet"
    keys.assign(association_membership_probability=.5, analysis_version=fusion.ANALYSIS_VERSION).to_parquet(primary, index=False)
    paths = []
    for fold in range(5):
        frame = keys.assign(patient_fold_id=fold, validation_patient_fold_id=(fold+1)%5,
            validation_proxy_label=float(fold % 2), validation_label_available=True,
            validation_label_unavailable_reason=None, fusion_supervision_scope=SCOPE)
        missing = frame.lncrna_id.eq("L000") | (frame.lncrna_id.eq("L001") & (fold != 0))
        frame.loc[missing, LABEL] = np.nan; frame.loc[missing, AVAILABLE] = False
        frame.loc[missing, REASON] = "NATIVE_ASSOCIATION_UNAVAILABLE_REASON_NOT_SERIALIZED"
        path = tmp_path/f"fold{fold}.parquet"; frame.to_parquet(path, index=False); paths.append(path)
    spec = fusion.ExpertSpec("signed", "value", "available", ("discovery", "confidence"), "donor", value_transform="signed_fisher")
    expert = tmp_path/"expert.parquet"
    keys.assign(value=np.linspace(-.8, .8, 80), available=True, analysis_version=fusion.ANALYSIS_VERSION).to_parquet(expert, index=False)
    return primary, paths, {spec: expert}


def test_nullable_aggregate_crossfit_filters_and_release_never_claims_independent_validation(tmp_path, monkeypatch):
    primary, paths, experts = parquet_fixture(tmp_path)
    frame, audit = fusion.build_pair_aggregated_fusion_frame_from_parquet(primary, paths, experts, native_validation_labels=True)
    absent = frame.loc[frame.lncrna_id.eq("L000")].iloc[0]
    zero = frame.loc[frame.lncrna_id.eq("L001")].iloc[0]
    assert pd.isna(absent.fusion_target) and absent.fusion_target_available_fold_count == 0
    assert pd.isna(absent.fusion_target_positive_folds)
    assert zero.fusion_target == 0 and zero.fusion_target_available_fold_count == 1 and zero.fusion_target_available
    assert audit["target_unavailable_rows"] == audit["target_partial_fold_rows"] == 1
    calls = []
    def frozen(train, validation, expert_ids, **kw):
        assert train.fusion_target.notna().all() and validation.fusion_target.notna().all()
        assert not train.lncrna_id.eq("L000").any() and not validation.lncrna_id.eq("L000").any()
        calls.append((set(train.lncrna_id), set(validation.lncrna_id)))
        return fusion.ResidualFusionModel(kw["endpoint"], tuple(expert_ids), (0.,), (.5,), 1, "0"*64, "1"*64, 0, 0,
                                         feature_transforms=kw["feature_transforms"])
    monkeypatch.setattr(fusion, "fit_residual_fusion_model", frozen)
    discovery = fusion.train_pair_blocked_crossfit_fusion(frame, tuple(experts), endpoint="discovery")
    confidence = fusion.train_pair_blocked_crossfit_fusion(frame, tuple(experts), endpoint="confidence")
    assert len(calls) == 12 and all(not a & b for a, b in calls)
    assert len(discovery.oof_prediction) == 80
    assert discovery.oof_prediction.loc[lambda f: f.lncrna_id.eq("L000"), "discovery_adjusted_probability"].notna().all()
    overall = discovery.fold_metrics.loc[lambda f: f.heldout_pair_fold.eq("ALL_OOF")].iloc[0]
    assert overall.test_rows == 79 and overall.predicted_rows == 80
    assert discovery.scientific_status == "DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED"
    assert discovery.performance_outcome.startswith("DEVELOPMENT_")
    declaration = {**record(primary), "source_role": "current_v32_oof_prediction", "generation": "current_v32"}
    result = fusion.materialize_multimodal_fusion_release(tmp_path/"release", frame, discovery, confidence,
        source_declarations=[declaration], training_run_id="V32-FUSION-NATIVE_SYNTHETIC", input_audit=audit)
    public = pd.read_parquet(tmp_path/"release/multimodal_secondary_scores.parquet")
    assert len(public) == 80 and not public.independent_validation_claimed.any()
    assert public.fusion_supervision_scope.eq(SCOPE).all() and LABEL not in public and "fusion_target" not in public
    lineage = read(tmp_path/"release/MODULE_LINEAGE.json")
    assert lineage["unavailable_labels_excluded_from_fitting_and_metrics"]
    # No caller may upgrade development-only results by replacing a status string.
    discovery.scientific_status = "SECONDARY_VALIDATED"
    with pytest.raises(fusion.MultimodalFusionError, match="independent"):
        fusion.materialize_multimodal_fusion_release(tmp_path/"invalid", frame, discovery, confidence,
            source_declarations=[declaration], training_run_id="V32-FUSION-NATIVE_SYNTHETIC", input_audit=audit)


@pytest.mark.parametrize("fault", ["filled_missing", "missing_reason", "available_null", "float_mask", "patient_fold", "scope"])
def test_builder_refuses_malformed_native_labels(tmp_path, fault):
    primary, paths, experts = parquet_fixture(tmp_path)
    data = pd.read_parquet(paths[0])
    if fault == "filled_missing": data.loc[0, LABEL] = 0
    if fault == "missing_reason": data.loc[0, REASON] = None
    if fault == "available_null": data.loc[1, LABEL] = np.nan
    if fault == "float_mask": data[AVAILABLE] = data[AVAILABLE].astype(float)
    if fault == "patient_fold": data["validation_patient_fold_id"] = 0
    if fault == "scope": data["fusion_supervision_scope"] = "SEALED_TEST"
    data.to_parquet(paths[0], index=False)
    with pytest.raises(fusion.MultimodalFusionError):
        fusion.build_pair_aggregated_fusion_frame_from_parquet(primary, paths, experts, native_validation_labels=True)


def test_no_fitting_begins_when_one_pair_fold_has_no_labels(tmp_path):
    primary, paths, experts = parquet_fixture(tmp_path)
    frame, _ = fusion.build_pair_aggregated_fusion_frame_from_parquet(primary, paths, experts, native_validation_labels=True)
    missing = frame.fusion_pair_fold.eq(0)
    frame.loc[missing, "fusion_target"] = np.nan
    frame.loc[missing, "fusion_target_available"] = False
    frame.loc[missing, "fusion_target_available_fold_count"] = 0
    frame.loc[missing, "fusion_target_unavailable_reason"] = "NO_AVAILABLE_NATIVE_VALIDATION_FOLD"
    with pytest.raises(fusion.MultimodalFusionError, match="before fitting"):
        fusion.train_pair_blocked_crossfit_fusion(frame, tuple(experts), endpoint="discovery")


def test_bare_hash_list_cannot_authorize_current_fusion_fit(tmp_path):
    import scripts.run_v32_current_multimodal_fusion as runner
    primary, paths, experts = parquet_fixture(tmp_path)
    binding = write(tmp_path/"bare.json", {"status": "PASS", "run_id": "current", "prediction": record(primary),
        "fold_prediction_sha256": [sha256(p) for p in paths]})
    manifest = write(tmp_path/"fusion.json", {"format": "CURRENT_MULTIMODAL_FUSION_MANIFEST_V1", "core_run_id": "current",
        "primary_prediction": record(primary), "primary_binding_audit": record(binding)})
    with pytest.raises(ValueError, match="native primary validation"):
        runner.main(["--manifest", str(manifest), "--expected-core-run-id", "current", "--fit", "--output", str(tmp_path/"out")])
    assert not (tmp_path/"out/SUCCESS.json").exists()
