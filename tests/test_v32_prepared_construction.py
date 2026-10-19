"""Actual native preparation and tensor reconstruction; no L1/neural fit."""
from copy import deepcopy
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.prepared_construction import replay, validate, AUTHORITY, FORMAT
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.preparation_sources import canonical_sha, AUTHORITY_KEY
from test_v32_native_core_workflow import completed, inputs, native_authority, scoped, normalized, controls


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*args, **kwargs): raise AssertionError("No fitting or neural construction in replay")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def payload(root, variant="G2", fold=0):
    path = root/"prepared"/variant/f"PATIENT_FOLD_{fold}.pt"
    return path, torch.load(path, map_location="cpu", weights_only=False)


def test_actual_fifteen_native_payloads_replay_and_independent_label_oracle(completed, tmp_path):
    for variant in ("G0", "G1", "G2"):
        for fold in range(5):
            path, value = payload(completed, variant, fold)
            result = replay(value, path.parent)
            assert result["tensor_fields_compared"] == 22 and result["linear_baseline_replayed"]
            assert not result["training_started"] and not result["parameter_fitting_replayed"]
            assert value["contains_trained_parameters"] is False
            assert set(value["feature_label_construction"]) == {"format", "path", "sha256"}
            # Oracle reads original statistical outputs, not the constructor's
            # reconstructed frame. Validation signs must be from replication.
            for split, prefix in (("train", "A1_local"), ("validation", "A1_validation")):
                batch = value[split+"_batches"][0]
                row = next(r for r in value["preparation_sources"]["source_reads"] if r["role"] == split+"_associations")
                raw = pd.read_parquet(row["path"]).set_index("lncrna_id")
                inverse = {v:k for k,v in value["bundle"].node_maps["lncRNA"].items()}
                ids = [inverse[int(n)] for n in batch["candidate_batch"]["l"]]
                raw = raw.loc[ids]; rho = raw["rho_"+prefix].to_numpy(float)
                available = raw.availability.eq("available").to_numpy()
                assert np.array_equal(batch["association_available"], available)
                assert np.array_equal(batch["proxy_label"], raw["label_"+prefix].isin(["strong_positive", "weak_positive"]))
                assert np.array_equal(batch["weak_positive"], raw["label_"+prefix].eq("weak_positive"))
                assert np.array_equal(batch["direction_available"], available & (np.abs(rho) >= .15))
                if split == "validation": assert np.array_equal(batch["direction_label"], rho >= 0)
    from scripts.audit_prepared_construction import main
    path, _ = payload(completed)
    output = tmp_path/"audit.json"
    assert main(["--prepared", str(path), "--expected-sha256", record(path)["sha256"], "--output", str(output)]) == 0
    assert read(output)["status"] == "PASS" and not read(output)["raw_study_ancestry_complete"]


def test_baseline_inspection_does_not_read_validation_associations(completed, monkeypatch):
    path, value = payload(completed)
    original = pd.read_parquet
    def guarded(p, *args, **kwargs):
        assert "validation_associations" not in str(p)
        return original(p, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    result = replay(value, path.parent, baseline_only=True)
    assert result["tensor_fields_compared"] == 11


def test_actual_producer_preserves_nonzero_frozen_linear_scores(completed, native_authority, scoped, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.prepare_v32_formal as producer
    from cc_hhgt.v32.baselines import build_safe_hashed_design
    from cc_hhgt.v32.native_core_workflow import resolve
    controls(monkeypatch, native_authority, scoped)
    expected = {}
    def frozen_nonzero(frames, **kwargs):
        coef = ((np.arange(4096)%17)-8)*.001; intercept = .13
        designs = {k: build_safe_hashed_design(v, n_features=4096) for k,v in frames.items()}
        for split, design in designs.items():
            # Separate scalar summation oracle, without model predict/fit.
            expected[split] = np.asarray([sum(float(v)*coef[int(j)] for j,v in zip(row.indices,row.data))+intercept
                                          for row in design], dtype=np.float32)
        model = SimpleNamespace(coef_=coef[None,:], intercept_=np.array([intercept]), classes_=np.array([0,1]),
            decision_function=lambda x: np.asarray(x @ coef).reshape(-1)+intercept)
        return model, {"synthetic_fixed_no_fit": True}, designs["train"], designs["validation"]
    monkeypatch.setattr(producer, "_fast_l1", frozen_nonzero)
    plan = read(completed/"LAUNCH_PREFLIGHT.json")
    argv = resolve(next(c for c in plan["commands"] if c["stage"] == "formal_preparation"), completed)[3:]
    target = tmp_path/"nonzero"
    argv[argv.index("--output")+1] = str(target)
    argv[argv.index("--fold-count")+1] = "1"
    assert producer.main(argv) == 0
    for variant in ("G0", "G1", "G2"):
        path = target/variant/"PATIENT_FOLD_0.pt"
        value = torch.load(path, weights_only=False)
        for split in ("train", "validation"):
            logits = torch.cat([b["base_logit"] for b in value[split+"_batches"]]).numpy()
            assert not np.all(logits == 0) and np.unique(logits).size > 1
            np.testing.assert_allclose(logits, expected[split], atol=1e-7, rtol=0)
        assert replay(value, path.parent)["tensor_fields_compared"] == 22


@pytest.mark.parametrize("field", ["base_logit", "conservation_context", "proxy_label", "weak_positive",
    "association_available", "direction_label", "direction_available", "graph_available", "candidate_batch", "missing_field", "missing_batch"])
def test_rehashed_payload_tensor_changes_are_rejected(completed, field):
    path, value = payload(completed)
    batch = value["validation_batches"][0]
    if field == "candidate_batch": batch[field]["l"][0] = (batch[field]["l"][0]+1)%3
    elif field == "missing_field": del batch["weak_positive"]
    elif field == "missing_batch": value["validation_batches"] = []
    elif batch[field].dtype == torch.bool: batch[field][0] = ~batch[field][0]
    else: batch[field][0] += .5
    # Re-saving a modified prepared file can change its outer SHA. This check
    # reconstructs tensor contents, so an updated container hash is insufficient.
    with pytest.raises(ValueError, match="tensor|fields|coverage"):
        replay(value, path.parent)


@pytest.mark.parametrize("fault", ["missing_reference", "removed_reference_and_hash", "state_weight", "fit_weight_hash", "fit_policy", "implementation"])
def test_construction_authority_and_fit_contract_cannot_be_relabelled(completed, tmp_path, fault):
    path, value = payload(completed)
    if fault.startswith("missing") or fault.startswith("removed"):
        value.pop("feature_label_construction")
        if fault.startswith("removed"):
            for key in ("artifact_hashes", "input_authority_hashes"): value.get(key, {}).pop(AUTHORITY, None)
    else:
        contract = read(value["feature_label_construction"]["path"])
        if fault == "state_weight": contract["baseline"]["intercept"] = .7
        elif fault == "fit_weight_hash": contract["fit_inputs"]["sample_weights_sha256"] = "1"*64
        elif fault == "fit_policy": contract["baseline"]["fit_split"] = "validation"
        else: contract["implementations"]["cc_hhgt/v32/prepared_construction.py"]["sha256"] = "2"*64
        target = tmp_path/"altered.json"; write(target, contract)
        value["feature_label_construction"] = {"format": FORMAT, **record(target)}
        for key in ("artifact_hashes", "input_authority_hashes"):
            if key in value: value[key][AUTHORITY] = record(target)["sha256"]
    with pytest.raises(ValueError): replay(value, path.parent)


def test_legacy_payloads_remain_explicitly_unreplayed(completed):
    path, value = payload(completed)
    value.pop("feature_label_construction"); value["preparation_sources"].pop("feature_construction_format")
    for key in ("artifact_hashes", "input_authority_hashes"):
        if key in value:
            value[key].pop(AUTHORITY, None); value[key][AUTHORITY_KEY] = canonical_sha(value["preparation_sources"])
    assert replay(value, path.parent) is None
