"""Original State statistics and completed frozen contexts; never fitting."""
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import state_training as native
from cc_hhgt.v32 import current_state_context as context
from cc_hhgt.v32.state_patient_use import Trace, TABLE
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_state_training import _fixture
from test_v32_current_state_context import current_sources, native_fixture


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*a, **k): raise AssertionError("State scope verification cannot fit")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(native, "train_fold_private_head", forbidden)
    monkeypatch.setattr(native, "train_state_models", forbidden)


def inputs():
    states, patients, expression, cores = _fixture()
    # Distinct actual pair masks; zero and constant observations still count.
    expression.loc[expression.sample_id.eq("S01") & expression.lncrna_id.eq("L_POS"), "logcpm"] = np.nan
    expression.loc[expression.sample_id.eq("S03") & expression.lncrna_id.eq("L_MIXED"), "logcpm"] = np.nan
    # Two aliquots, one analysis patient; raw authority order intentionally unsorted.
    alias = patients.iloc[[1]].copy(); alias.sample_id = "S01_A2"
    patients = pd.concat([alias, patients.iloc[::-1]], ignore_index=True)
    aliq = expression.loc[expression.sample_id.eq("S01")].copy(); aliq.sample_id = "S01_A2"
    expression = pd.concat([expression, aliq], ignore_index=True)
    for fold in cores: cores[fold] = cores[fold].loc[~cores[fold].node_id.eq("L_MIXED")]
    cfg = native.StateTrainingConfig(min_pairs=2, validation_offset=2)
    parents = {fold: {key: str(fold)*64 for key in ("core_embedding_sha256", "core_parameter_sha256",
                "v32_core_checkpoint_sha256")} for fold in range(5)}
    trace = Trace(patients, {"sha256": "a"*64}, cores, parents, cfg.validation_offset)
    aligned = native.prepare_state_inputs(states, patients, cores, expression, cfg)
    return patients, cfg, trace, aligned


def test_actual_masks_preserve_statistics_aliases_missing_and_core_eligibility():
    raw, cfg, trace, (states, expression, folds, cores, _) = inputs()
    for fold in range(5):
        split, ordinary = native.state_fold_statistics(states, expression, folds, fold, cfg)
        _, captured = native.state_fold_statistics(states, expression, folds, fold, cfg, patient_observer=trace.observer(fold))
        for role in ordinary: pd.testing.assert_frame_equal(ordinary[role], captured[role], check_exact=True)
        events = trace.table().loc[lambda d: d.patient_fold_id.eq(fold)]
        emitted = set()
        for event in events.to_dict("records"):
            rows = json.loads(event["patient_map_row_indices_json"])
            actual = raw.iloc[rows]
            assert actual.patient_id.nunique() == event["native_patient_count"]
            allowed = set(split.loc[split.split.eq(event["split_role"]), "sample_id"])
            assert set(actual.patient_id) <= allowed
            names = json.loads(event["lncrna_ids_json"])
            for name, available, eligible in zip(names, json.loads(event["statistic_available_json"]),
                                                json.loads(event["private_head_input_eligible_json"]), strict=True):
                row = captured[event["split_role"]].set_index(["lncrna_id", "state_id"]).loc[(name, event["state_id"])]
                assert row.n_pairs == event["native_patient_count"] and bool(row.available) == available
                assert eligible == (available and name != "L_MIXED" and event["split_role"] != "test")
                state_ids = set(states.loc[states.state_id.eq(event["state_id"]) & states.state_value.notna(), "sample_id"])
                expr_ids = set(expression.loc[expression.lncrna_id.eq(name) & expression.logcpm.notna(), "sample_id"])
                assert set(actual.patient_id) == allowed & state_ids & expr_ids
                emitted.add((event["split_role"], name, event["state_id"]))
        assert len(emitted) == sum(map(len, captured.values()))
    table = trace.table()
    assert table.native_patient_count.eq(0).any()
    assert any(len(json.loads(r.patient_map_row_indices_json)) > r.native_patient_count for r in table.itertuples())
    # Candidate cohorts differ, and constant-expression candidates retain their
    # measured patients despite having no estimable correlation/head input.
    selected = table.loc[table.patient_fold_id.eq(0) & table.split_role.eq("train")]
    assert selected.native_patient_count.nunique() >= 2
    assert any("L_CONST" in json.loads(r.lncrna_ids_json) and r.native_patient_count > 0 for r in selected.itertuples())


@pytest.mark.parametrize("fault", ["unknown", "cross_split", "duplicate", "count", "mask_shape"])
def test_scope_rejects_invalid_observer_alignment(fault):
    raw, cfg, trace, _ = inputs()
    ids = ["S01"]; mask = np.array([[True]]); count = np.array([1])
    if fault == "unknown": ids = ["UNKNOWN"]
    if fault == "cross_split": ids = ["S00"]
    if fault == "duplicate": ids *= 2; mask = np.ones((2, 1), dtype=bool); count = np.array([2])
    if fault == "count": count = np.array([2])
    if fault == "mask_shape": mask = np.ones((1, 2), dtype=bool)
    with pytest.raises(ValueError): trace.observer(0)("TEST", native.HISTORICAL_STATE_IDS[0], "train", ["L_POS"], ids, mask, count, [True])


def test_completed_native_replay_private_scope_and_publication(tmp_path, current_sources):
    manifest = native_fixture(tmp_path/"inputs", current_sources)
    data = read(manifest); source = Path(data["state_training_lineage"]["path"]).parent
    before = {str(p): (record(p)["sha256"], p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()}
    result = context.materialize_current_state_context(manifest, tmp_path/"context")
    path = Path(result["binding_path"]); binding = read(path)
    private = binding["private_outputs"][TABLE]
    assert private["private"] is True and TABLE not in binding["outputs"]
    table = pd.read_parquet(private["path"])
    assert len(table) == 5*3*len(native.HISTORICAL_STATE_IDS)
    oof = pd.read_parquet(source/"v32_state_oof_predictions.parquet")
    for event in table.to_dict("records"):
        if event["split_role"] == "validation": continue
        row = oof.loc[oof.patient_fold_id.eq(event["patient_fold_id"]) & oof.state_id.eq(event["state_id"])].iloc[0]
        assert event["native_patient_count"] == row[event["split_role"]+"_n_pairs"]
    reader = context.CurrentStateContextQuery(path, expected_sha=record(path)["sha256"],
        selected_analysis_sha=binding["selected_analysis"]["sha256"])
    with pytest.raises(ValueError): reader.download(TABLE)
    with pytest.raises(ValueError): reader.query(table=TABLE)
    assert before == {str(p): (record(p)["sha256"], p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()}
    # Rehashing a changed cohort, even with unchanged cohort size, is rejected by
    # the actual measurement/forward replay rather than by a checksum alone.
    index = table.index[table.native_patient_count.gt(0)][0]
    table.loc[index, "patient_map_row_indices_json"] = "[999]"
    table.to_parquet(private["path"], index=False)
    private.update(record(Path(private["path"]))); write(path, binding)
    with pytest.raises(ValueError, match="private patient scope replay"):
        context.audit_current_state_context(path, expected_sha=record(path)["sha256"])


@pytest.mark.parametrize("fault", ["missing", "false", "public_alias"])
def test_private_scope_cannot_be_registered_as_a_public_table(tmp_path, current_sources, fault):
    manifest = native_fixture(tmp_path/"inputs", current_sources)
    result = context.materialize_current_state_context(manifest, tmp_path/"context")
    path = Path(result["binding_path"]); binding = read(path)
    if fault == "missing": binding.pop("private_outputs")
    elif fault == "false": binding["private_outputs"][TABLE]["private"] = False
    else: binding["outputs"]["alias"] = binding["private_outputs"][TABLE]
    write(path, binding)
    with pytest.raises(ValueError, match="must remain private"):
        context.CurrentStateContextQuery(path, expected_sha=record(path)["sha256"],
            selected_analysis_sha=binding["selected_analysis"]["sha256"])
