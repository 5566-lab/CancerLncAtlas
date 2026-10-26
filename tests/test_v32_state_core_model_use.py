"""Actual State -> core input traversal and explicit identities; no fitting."""
import json
from pathlib import Path
import shutil
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import current_state_context as state
from cc_hhgt.v32.current_state_model_use import PREFIX, PRIVATE_NAMES
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_current_state_context import current_sources, native_fixture


def forbidden(*a, **k): raise AssertionError("State/core source inspection cannot fit or rebuild the core")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)


@pytest.fixture(scope="module")
def parents(tmp_path_factory):
    import test_v32_current_binding_expansion as fixtures
    root = tmp_path_factory.mktemp("state_core_source")
    raw = root/"binding_source.parquet"
    pd.DataFrame({"lncrna_id": ["LNC:Z"]*2, "protein_id": ["UNIPROT:Z"]*2,
        "pmid": ["66778899"]*2, "patient_id": ["CORE_ALIAS", "P0_1"]}).to_parquet(raw, index=False)
    original = fixtures.verified_payload
    def payload(fold=0):
        result = original(fold)
        result["formal_graph_authority"]["static_artifacts"]["lnc_protein_binding"] = record(raw)
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixtures, "verified_payload", payload)
        parent = current_sources.__wrapped__(tmp_path_factory)
    return parent, raw


def fixture(tmp_path, parents, *, mapped=False):
    parent, raw = parents
    path = native_fixture(tmp_path/"inputs", parent)
    manifest = read(path); lineage = read(manifest["state_training_lineage"]["path"])
    patient = next(r for r in lineage["input_artifacts"] if r["role"] == "v32_patient_fold_manifest")
    if mapped:
        mapping = pd.read_parquet(patient["path"])
        rows = [(patient["sha256"], "patient_id", identity, identity) for identity in mapping.patient_id.unique()]
        # Equal text in different source files is explicitly a different person;
        # a differently spelled alias is explicitly the same person.
        rows += [(record(raw)["sha256"], "patient_id", "CORE_ALIAS", "P0_0"),
                 (record(raw)["sha256"], "patient_id", "P0_1", "UNRELATED_CORE_PERSON")]
        crosswalk = tmp_path/"crosswalk.parquet"
        pd.DataFrame(rows, columns=["source_file_sha256", "identity_column", "raw_id", "canonical_patient_id"]).to_parquet(crosswalk, index=False)
        manifest["source_inventory"] = {"patient_identity_crosswalk": {**record(crosswalk), "namespace": "SYNTHETIC_STATE_CORE"}}
        write(path, manifest)
    return path


def test_actual_parent_inputs_and_study_sources_are_followed_without_new_core_forward(tmp_path, parents, monkeypatch):
    path = fixture(tmp_path, parents)
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter, "_build_model", forbidden)
    context = state.state_sources(path)
    tables = context["input_use"]; deps = tables[PREFIX+"dependencies"]
    core = deps.loc[deps.dependency_context.eq("STATE_CORE")]
    assert set(core.loc[core.role.eq("checkpoint"), "patient_fold_id"]) == set(range(5))
    assert set(core.loc[core.role.eq("prepared"), "patient_fold_id"]) == set(range(5))
    usage = tables[PREFIX+"usage"]
    assert usage.loc[usage.dependency_context.eq("STATE_CORE"), "study_tokens_json"].str.contains("PMID:66778899").any()
    assert {"STATE_HEAD", "STATE_CORE"} == set(context["input_coverage"].dependency_context)
    assert not context["input_coverage"].inventory_complete.any()
    mapping = tables[PREFIX+"patient_mapping"]
    assert mapping.loc[mapping.observed_identity_count.gt(0), "unmapped_identity_count"].gt(0).all()
    assert not context["input_coverage"].registered_patient_mapping_complete.any()
    assert "STATE_RAW_MEASUREMENT_ANCESTRY_NOT_RECONSTRUCTED" in set(tables[PREFIX+"gaps"].reason)


def test_explicit_aliases_reach_actual_private_overlap_and_public_coverage(tmp_path, parents, monkeypatch):
    path = fixture(tmp_path, parents, mapped=True)
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter, "_build_model", forbidden)
    result = state.materialize_current_state_context(path, tmp_path/"context")
    binding_path = Path(result["binding_path"]); binding = read(binding_path)
    private = binding["private_outputs"]
    assert PRIVATE_NAMES <= set(private) and all(r["private"] is True for r in private.values())
    overlap = pd.read_parquet(private[PREFIX+"patient_context_overlap"]["path"])
    assert overlap.shared_patient_count.tolist() == [1]
    assert overlap.interpretation.str.contains("NOT_A_FOLD_LEAKAGE_ATTESTATION").all()
    summary = pd.read_parquet(binding["outputs"]["state_input_coverage"]["path"])
    assert summary.registered_patient_mapping_complete.all() and not summary.inventory_complete.any()
    reader = state.CurrentStateContextQuery(binding_path, expected_sha=record(binding_path)["sha256"],
        selected_analysis_sha=binding["selected_analysis"]["sha256"])
    public = json.dumps(reader.query(table="state_input_coverage")["rows"])
    assert not any(value in public for value in ("CORE_ALIAS", "P0_0", "P0_1", "UNRELATED_CORE_PERSON", "patient_token"))
    for name in PRIVATE_NAMES:
        with pytest.raises(ValueError): reader.download(name)
        with pytest.raises(ValueError): reader.query(table=name)
    # A freshly rehashed private inventory cannot omit the actual core inputs.
    item = private[PREFIX+"dependencies"]; frame = pd.read_parquet(item["path"])
    frame = frame.loc[~frame.role.eq("checkpoint")]
    frame.to_parquet(item["path"], index=False); item.update(record(Path(item["path"]))); item["rows"] = len(frame)
    write(binding_path, binding)
    with pytest.raises(ValueError, match="private patient scope replay"):
        state.audit_current_state_context(binding_path, expected_sha=record(binding_path)["sha256"])


def test_native_locator_is_consumed_and_unused_locator_is_rejected(tmp_path, parents):
    path = fixture(tmp_path, parents); manifest = read(path)
    relocated = tmp_path/"relocated.parquet"; shutil.copyfile(parents[1], relocated)
    manifest["source_inventory"] = {"artifact_locations": [record(relocated)]}; write(path, manifest)
    context = state.state_sources(path)
    deps = context["input_use"][PREFIX+"dependencies"]
    assert deps.loc[deps.sha256.eq(record(relocated)["sha256"]), "path"].eq(str(relocated)).all()
    unused = tmp_path/"unused.txt"; unused.write_text("unused")
    manifest["source_inventory"]["artifact_locations"].append(record(unused)); write(path, manifest)
    with pytest.raises(ValueError, match="Unused State model-use artifact locator"):
        state.state_sources(path)


@pytest.mark.parametrize("fault", ["changed_source", "unknown_option", "false_private", "public_alias"])
def test_state_core_source_and_private_contract_cannot_be_rehashed_away(tmp_path, parents, fault):
    path = fixture(tmp_path, parents)
    if fault == "unknown_option":
        manifest = read(path); manifest["source_inventory"] = {"all_sources_verified": True}; write(path, manifest)
        with pytest.raises(ValueError, match="source inventory options"): state.state_sources(path)
        return
    if fault == "changed_source":
        raw = parents[1]; original = raw.read_bytes()
        try:
            raw.write_bytes(original+b"changed")
            with pytest.raises(ValueError, match="SHA256 mismatch"): state.state_sources(path)
        finally: raw.write_bytes(original)
        return
    result = state.materialize_current_state_context(path, tmp_path/"context")
    binding_path = Path(result["binding_path"]); binding = read(binding_path)
    item = binding["private_outputs"][PREFIX+"patient_usage_private"]
    if fault == "false_private": item["private"] = False
    else: binding["outputs"]["alias"] = item
    write(binding_path, binding)
    with pytest.raises(ValueError, match="must remain private"):
        state.CurrentStateContextQuery(binding_path, expected_sha=record(binding_path)["sha256"],
            selected_analysis_sha=binding["selected_analysis"]["sha256"])


@pytest.mark.parametrize("extension,separator", [("csv", ","), ("tsv", "\t")])
def test_original_state_reader_preserves_file_scoped_numeric_identifiers(tmp_path, extension, separator):
    from cc_hhgt.v32.state_training import read_table, validate_fold_manifest
    from cc_hhgt.v32.patient_source_identity import IdentityMap, PatientUse
    file = tmp_path/("raw_patients."+extension)
    frame = pd.DataFrame({"cancer_id": ["01"]*5, "sample_id": ["001", "1", "02", "03", "04"],
        "patient_id": ["001", "1", "02", "03", "04"], "patient_fold_id": list(range(5))})
    frame.to_csv(file, sep=separator, index=False)
    actual = read_table(file, "patient folds")
    assert actual.patient_id.tolist() == frame.patient_id.tolist()
    assert actual.sample_id.tolist() == frame.sample_id.tolist()
    assert actual.cancer_id.tolist() == ["01"]*5
    assert validate_fold_manifest(actual).patient_id.nunique() == 5
    crosswalk = tmp_path/"crosswalk.parquet"
    pd.DataFrame([(record(file)["sha256"], "patient_id", raw, "PERSON_"+raw) for raw in frame.patient_id],
        columns=["source_file_sha256", "identity_column", "raw_id", "canonical_patient_id"]).to_parquet(crosswalk, index=False)
    use = PatientUse(IdentityMap({**record(crosswalk), "namespace": "NUMERIC_IDS"}, tmp_path))
    use.observe([actual[["patient_id"]]], digest=record(file)["sha256"], model="state", context="STATE_HEAD",
        role="paired", scope="NATIVE_STATE_STATISTICAL_PAIRED_PATIENT_ROWS")
    tables = use.tables()
    assert tables["patient_mapping"].mapped_identity_count.tolist() == [5]
    assert tables["patient_usage_private"].patient_token.nunique() == 5
