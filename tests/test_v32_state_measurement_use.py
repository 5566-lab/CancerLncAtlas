"""Original State means and paired masks, including exact loaded aliquot rows."""
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import state_training as native, current_state_context as context
from cc_hhgt.v32.state_measurement_use import MeasurementTrace, TABLE, LINKS, PRIVATE_NAMES
from cc_hhgt.v32.state_patient_use import Trace
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_current_state_context import current_sources, native_fixture


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*a, **k): raise AssertionError("State aliquot tracing cannot fit")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(native, "train_fold_private_head", forbidden)
    monkeypatch.setattr(native, "train_state_models", forbidden)


def fixture():
    patients = []; states = []; expression = []
    state_ids = native.HISTORICAL_STATE_IDS[:2]
    for i in range(15):
        patient = f"TCGA-ZZ-{i:04d}"
        for aliquot in range(2 if i == 1 else 1):
            sample = f"{patient}-E{aliquot}"
            patients.append(("TEST", sample, patient, i % 5))
            for gene in ("L1", "L2"):
                number = i + 2*aliquot if gene == "L1" else 0
                if (i, aliquot, gene) == (1, 0, "L1"): number = np.nan
                if i == 2 and gene == "L2": number = np.nan
                expression.append(("TEST", sample, gene, number))
            # State samples differ from expression aliquots; State maps by its
            # explicit TCGA patient prefix, expression by the actual authority.
            for sid in state_ids:
                states.append((f"STATE_{i}_{aliquot}", "TEST", patient+"-01A", sid, float(i+4*aliquot)))
    patients = pd.DataFrame(patients, columns=["cancer_id", "sample_id", "patient_id", "patient_fold_id"]).iloc[::-1].reset_index(drop=True)
    states += [("outside", "TEST", "OUTSIDE", state_ids[0], 99), ("meta", None, None, "sample_type_code", 1)]
    expression += [("TEST", "outside", "L1", 88)]
    states = pd.DataFrame(states, columns=["sample_id", "cancer_id", "patient_id", "state_id", "state_value"]).iloc[::-1].reset_index(drop=True)
    expression = pd.DataFrame(expression, columns=["cancer_id", "sample_id", "lncrna_id", "logcpm"]).iloc[::-1].reset_index(drop=True)
    cores = {fold: pd.DataFrame({"node_id": ["L1", "L2"], "core_feature_0": [0., 1.]}) for fold in range(5)}
    cfg = native.StateTrainingConfig(min_pairs=2, validation_offset=2)
    pins = {role: {"sha256": digit*64} for role, digit in zip(
        ("tumor_state_long_measurement", "lncrna_expression", "v32_patient_fold_manifest"), "abc")}
    trace = MeasurementTrace(patients, pins)
    trace.register("tumor_state_long_measurement", states, "state_id")
    trace.register("lncrna_expression", expression, "lncrna_id")
    parents = {fold: {key: str(fold)*64 for key in ("core_embedding_sha256", "core_parameter_sha256", "v32_core_checkpoint_sha256")} for fold in range(5)}
    pairs = Trace(patients, pins["v32_patient_fold_manifest"], cores, parents, cfg.validation_offset)
    return states, patients, expression, cores, cfg, trace, pairs


def captured():
    states, patients, expression, cores, cfg, trace, pairs = fixture()
    ordinary = native.prepare_state_inputs(states, patients, cores, expression, cfg)
    observed = native.prepare_state_inputs(states, patients, cores, expression, cfg, alignment_observer=trace.observe)
    for a, b in zip(ordinary[:3], observed[:3], strict=True): pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert ordinary[4] == observed[4]
    a, b, folds, cores, _ = observed
    for fold in range(5):
        _, expected = native.state_fold_statistics(a, b, folds, fold, cfg)
        _, actual = native.state_fold_statistics(a, b, folds, fold, cfg, patient_observer=pairs.observer(fold))
        for split in expected: pd.testing.assert_frame_equal(expected[split], actual[split], check_exact=True)
    return states, patients, expression, trace, pairs.table()


def test_original_mean_contributors_and_pair_masks_are_exact():
    states, patients, expression, trace, scopes = captured()
    tables = trace.tables(scopes); groups = tables[TABLE]; links = tables[LINKS]
    assert len(groups) == 15*4 and len(links) == 2*len(scopes)
    assert len(set(groups.contribution_id)) == len(groups)
    for group in groups.itertuples():
        raw = states if group.role == "tumor_state_long_measurement" else expression
        positions = json.loads(group.source_row_indices_json)
        finite = json.loads(group.finite_source_row_indices_json); missing = json.loads(group.missing_source_row_indices_json)
        assert sorted(finite+missing) == positions and not set(finite) & set(missing)
        values = raw.iloc[positions][group.value_column]
        assert group.finite_value_count == values.notna().sum()
        assert group.native_patient_mean == pytest.approx(values.mean(), nan_ok=True)
        authority = patients.iloc[json.loads(group.patient_map_row_indices_json)]
        assert set(authority.patient_id) == {group.patient_id}
        if group.role == "lncrna_expression": assert set(raw.iloc[positions].sample_id) <= set(authority.sample_id)
    subset = groups.loc[groups.patient_id.eq("TCGA-ZZ-0001")]
    assert subset.loc[subset.role.eq("tumor_state_long_measurement"), "native_patient_mean"].eq(3).all()
    x = subset.loc[subset.feature_id.eq("L1")].iloc[0]
    assert x.native_patient_mean == 3 and x.finite_value_count == 1
    assert len(json.loads(x.missing_source_row_indices_json)) == 1
    assert groups.finite_value_count.eq(0).sum() == 1
    index = groups.set_index("contribution_id")
    for link in links.itertuples():
        selected = index.loc[json.loads(link.contribution_ids_json)]
        assert selected.patient_id.nunique() == link.native_patient_count
        assert selected.finite_value_count.gt(0).all()
        assert selected.role.eq(link.role).all()
        expected = {link.state_id} if link.role == "tumor_state_long_measurement" else set(json.loads(link.lncrna_ids_json))
        if len(selected): assert set(selected.feature_id) == expected
        event = scopes.loc[scopes.patient_fold_id.eq(link.patient_fold_id) & scopes.split_role.eq(link.split_role)
            & scopes.cohort_id.eq(link.cohort_id)].iloc[0]
        assert set(selected.patient_id) == set(patients.iloc[json.loads(event.patient_map_row_indices_json)].patient_id)


def test_nullable_measurement_dtype_retains_all_missing_mean():
    states, patients, expression, cores, cfg, trace, _ = fixture()
    states["state_value"] = states.state_value.astype("Float64")
    expression["logcpm"] = expression.logcpm.astype("Float64")
    native.prepare_state_inputs(states, patients, cores, expression, cfg, alignment_observer=trace.observe)
    missing = [r for r in trace.events if r["finite_value_count"] == 0]
    assert len(missing) == 1 and np.isnan(missing[0]["native_patient_mean"])


@pytest.mark.parametrize("fault", ["authority", "count", "position", "missing_group", "missing_source", "duplicate_group"])
def test_scope_cannot_claim_unobserved_contributions(fault):
    _, _, _, trace, scopes = captured()
    row = scopes.index[scopes.native_patient_count.gt(0)][0]
    if fault == "authority": scopes.loc[row, "patient_map_sha256"] = "0"*64
    if fault == "count": scopes.loc[row, "native_patient_count"] += 1
    if fault == "position": scopes.loc[row, "patient_map_row_indices_json"] = "[-1]"
    if fault == "missing_group": trace.events.clear()
    if fault == "missing_source": trace.observed.clear()
    if fault == "duplicate_group": trace.events.append(trace.events[0])
    with pytest.raises(ValueError, match="State"): trace.tables(scopes)


@pytest.mark.parametrize("fault", ["rows", "links", "public_alias", "missing_table"])
def test_actual_frozen_context_replays_private_contributions(tmp_path, current_sources, fault):
    manifest = native_fixture(tmp_path/"inputs", current_sources)
    result = context.materialize_current_state_context(manifest, tmp_path/"context")
    path = Path(result["binding_path"]); binding = read(path)
    assert PRIVATE_NAMES <= set(binding["private_outputs"])
    reader = context.CurrentStateContextQuery(path, expected_sha=record(path)["sha256"],
        selected_analysis_sha=binding["selected_analysis"]["sha256"])
    for name in PRIVATE_NAMES:
        with pytest.raises(ValueError): reader.download(name)
        with pytest.raises(ValueError): reader.query(table=name)
    if fault in {"rows", "links"}:
        item = binding["private_outputs"][TABLE if fault == "rows" else LINKS]
        frame = pd.read_parquet(item["path"])
        frame.loc[0, "finite_source_row_indices_json" if fault == "rows" else "contribution_ids_json"] = "[]"
        frame.to_parquet(item["path"], index=False); item.update(record(Path(item["path"])))
    elif fault == "public_alias": binding["outputs"]["alias"] = binding["private_outputs"][TABLE]
    else: del binding["private_outputs"][LINKS]
    write(path, binding)
    with pytest.raises(ValueError, match="private patient scope replay|must remain private"):
        context.audit_current_state_context(path, expected_sha=record(path)["sha256"])
