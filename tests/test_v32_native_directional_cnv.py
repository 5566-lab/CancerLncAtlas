"""Actual raw segment preparation, bounded reconstruction and CNV consumers."""
from copy import deepcopy
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import native_directional_cnv as native
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.current_model_use import Inventory
from cc_hhgt.v32.current_cnv_model_use import inspect_bound_sources
from test_v32_native_cnv_workflow import input_manifest
from test_v32_local_cnv_cohort import no_fit, ROOT, CANCERS, PATHWAY, LNCS


def inputs(root):
    workflow, args = input_manifest(root)
    patients = pd.read_csv(args.patient_map, sep="\t")
    intervals = root/"native_intervals.parquet"
    pd.DataFrame([("lncrna","LNC:A","1",10,20), ("lncrna","LNC:A","1",210,220),
        ("lncrna","LNC:B","1",30,40), ("lncrna","LNC:LOW","1",50,60), ("gene","G1","1",110,120)]
        + [("gene",f"G{i}","2",i*5,i*5+3) for i in range(2,11)],
        columns=["entity_type","entity_id","chromosome","start","end"]).to_parquet(intervals, index=False)
    membership = root/"native_membership.parquet"
    pd.DataFrame({"pathway_id":PATHWAY, "gene_id":[f"G{i}" for i in range(1,11)]}).to_parquet(membership, index=False)
    segments = []
    for cancer in CANCERS:
        for i, patient in enumerate(patients.loc[patients.cancer_id.eq(cancer), "patient_id"]):
            row = {"cancer_id":cancer, "patient_id":patient}
            if i == 119:
                segments.append({**row,"status":"typed_unavailable","unavailable_reason":"NO_SELECTED_SEGMENT_FILE"}); continue
            signal = np.sin(i*.3)+np.cos(i*.13)
            left, right = (.6, -.6) if i == 0 else (signal, signal)
            gene = .5 if i == 0 else .1 if i == 1 else -.5 if i == 2 else 0.
            raw = root/(cancer+"_"+str(i)+".seg.tsv")
            raw_rows = [("chr1",0,24,left), ("chr1",25,74,0.), ("chr1",75,150,gene), ("chr1",200,224,right)]
            if i > 2: raw_rows.append(("chr2",0,100,0.))
            pd.DataFrame(raw_rows, columns=["Chromosome","Start","End","Segment_Mean"]).assign(Sample=patient).to_csv(raw, sep="\t", index=False)
            segments.append({**row, "status":"available", "file":record(raw), "study_metadata":{"pmid":"99112233", "synthetic_fixture":True}})
    manifest = root/"native_directional.json"
    write(manifest, {"format":native.INPUT, "candidate_definition":record(root/"CANDIDATE_SOURCES.json"),
        "candidates":record(args.candidates), "patient_map":record(args.patient_map), "intervals":record(intervals),
        "membership":record(membership), "segments":segments, "event_threshold":.30, "max_segment_rows":100})
    return manifest, workflow, args


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    assert socket.gethostname() == "149"
    root = tmp_path_factory.mktemp("native_directional_inputs")/"inputs"
    manifest, workflow, args = inputs(root)
    output = root/"native_directional"
    result = subprocess.run([sys.executable,"-m","scripts.materialize_v32_directional_cnv","--phase","all",
        "--native-manifest",str(manifest),"--output-root",str(output)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return manifest, workflow, args, output


def test_actual_cli_signed_mapping_callability_and_missing_patients(completed, tmp_path):
    manifest, _, _, output = completed
    _, release = native.validate_release(record(output/"SUCCESS.json"), output, replay=True)
    assert set(release["cancers"]) == set(CANCERS) and release["cancer_count"] == 2
    assert read(output/"LAUNCH_PREFLIGHT.json")["target_host"] == "149"
    for cancer in CANCERS:
        part = output/("cancer="+cancer)
        info = read(part/"SUCCESS.json")
        assert info["processed_segment_patients"] == 119 and info["typed_unavailable_patients"] == 1
        assert set(info["files"]) == set((*native.IDS, *native.ARRAYS, "SOURCE_FILES.json"))
        lncs = read(part/"lncrna_ids.json"); a = lncs.index("LNC:A")
        signed = np.load(part/"lncrna_signed_mean.npy")
        assert signed[0,a] == 0.  # Two opposite intervals, not absolute burden.
        assert np.load(part/"lncrna_amplification.npy")[0,a] == 1
        assert np.load(part/"lncrna_deletion.npy")[0,a] == 1
        assert np.load(part/"lncrna_neutral.npy")[0,a] == 0
        assert np.isnan(signed[-1]).all() and not np.load(part/"lncrna_callable.npy")[-1].any()
        path_signed = np.load(part/"pathway_signed_mean.npy")
        np.testing.assert_allclose(path_signed[[0,2],0], [.5,-.5])
        assert np.isnan(path_signed[1,0])  # Neutral observed gene + nine missing members is unavailable.
        assert np.load(part/"pathway_amplification_fraction.npy")[0,0] == 1.
        assert np.load(part/"pathway_deletion_fraction.npy")[2,0] == 1.
        # Independent direct expected mean for every ordinary patient.
        index = np.arange(1,119)
        np.testing.assert_allclose(signed[index,a], np.sin(index*.3)+np.cos(index*.13), rtol=1e-6, atol=1e-7)
    from scripts.audit_native_directional_cnv import main
    audit = tmp_path/"audit.json"
    assert main(["--success",str(output/"SUCCESS.json"),"--expected-sha256",record(output/"SUCCESS.json")["sha256"],
                 "--output",str(audit)]) == 0
    assert read(audit)["arrays_compared"] == 24 and not read(audit)["raw_study_ancestry_complete"]


@pytest.mark.parametrize("name", native.ARRAYS)
def test_repinned_array_values_fail_numerical_reconstruction(completed, tmp_path, name):
    part = completed[3]/"cancer=BRCA"; value = deepcopy(read(part/"SUCCESS.json"))
    # Isolate altered output; all input/control/raw files remain their actual originals.
    for source in part.iterdir():
        if source.is_file(): (tmp_path/source.name).write_bytes(source.read_bytes())
    path = tmp_path/name; array = np.load(path)
    array[0,0] = .731 if np.issubdtype(array.dtype, np.floating) else 1-array[0,0]
    np.save(path, array, allow_pickle=False)
    value["files"][name] = record(path)["sha256"]; write(tmp_path/"SUCCESS.json", value)
    item = record(tmp_path/"SUCCESS.json")
    native.validate_partition(item, tmp_path)  # Ordinary hash validation passes.
    with pytest.raises(ValueError, match="numerical replay differs"):
        native.validate_partition(item, tmp_path, replay=True)


@pytest.mark.parametrize("fault", ["missing_patient", "duplicate_patient", "unknown_patient", "duplicate_file", "wrong_content_patient",
                                  "missing_reason", "threshold", "row_limit", "raw_hash", "fractional_interval"])
def test_invalid_input_contract_cannot_publish(completed, tmp_path, fault):
    spec = deepcopy(read(completed[0]))
    if fault == "missing_patient": spec["segments"].pop()
    elif fault == "duplicate_patient": spec["segments"].append(spec["segments"][0])
    elif fault == "unknown_patient": spec["segments"][0]["patient_id"] = "UNKNOWN"
    elif fault == "duplicate_file": spec["segments"][1]["file"] = spec["segments"][0]["file"]
    elif fault == "wrong_content_patient":
        source = tmp_path/"raw.tsv"; pd.read_csv(spec["segments"][0]["file"]["path"],sep="\t").assign(Sample="OTHER").to_csv(source,sep="\t",index=False)
        spec["segments"][0]["file"] = record(source)
    elif fault == "missing_reason": spec["segments"][-1].pop("unavailable_reason")
    elif fault == "threshold": spec["event_threshold"] = 0.
    elif fault == "row_limit": spec["max_segment_rows"] = 2
    elif fault == "raw_hash": spec["segments"][0]["file"]["sha256"] = "0"*64
    else:
        frame = pd.read_parquet(spec["intervals"]["path"]); frame["start"] = frame.start.astype(float); frame.loc[0,"start"] = 10.5
        target = tmp_path/"intervals.parquet"; frame.to_parquet(target,index=False); spec["intervals"] = record(target)
    manifest = tmp_path/"input.json"; write(manifest, spec)
    with pytest.raises((ValueError,RuntimeError)):
        native.materialize(record(manifest), tmp_path, tmp_path/"out", "BRCA")
    assert not (tmp_path/"out/cancer=BRCA").exists()


def test_native_partition_replay_enters_actual_source_inventory(completed, tmp_path):
    part = completed[3]/"cancer=BRCA"; item = record(part/"SUCCESS.json")
    inventory = Inventory([], tmp_path)
    inspect_bound_sources(inventory, [item], tmp_path, "core", "compact", 0)
    rows = pd.DataFrame(inventory.transform_replays)
    assert rows.loc[rows.format.eq(native.REPLAY),"outputs_compared"].tolist() == [12]
    assert "SEGMENT_ACQUISITION_AND_REFERENCE_ANCESTRY_UNVERIFIED" in {r["reason"] for r in inventory.gaps}
    usage = pd.DataFrame(inventory.usage)
    assert usage.study_tokens_json.str.contains("99112233").any()
    assert not any("PRIVATE_BRCA" in str(row) for row in inventory.usage)
    mapped = inventory.patient_use.tables()["patient_mapping"]
    assert mapped.loc[mapped.source_role.str.endswith("segment_patient"),"observed_identity_count"].sum() == 119


def test_cache_rechecks_raw_files_and_reference_inputs(completed, tmp_path):
    item = record(completed[3]/"cancer=BRCA/SUCCESS.json"); cache = set()
    native.validate_partition(item, tmp_path, replay=True, cache=cache)
    raw = Path(read(completed[0])["segments"][0]["file"]["path"]); original = raw.read_bytes()
    try:
        raw.write_bytes(original+b"changed")
        with pytest.raises(ValueError): native.validate_partition(item, tmp_path, replay=True, cache=cache)
    finally: raw.write_bytes(original)
    reference = Path(read(completed[0])["membership"]["path"]); original = reference.read_bytes()
    try:
        reference.write_bytes(original+b"changed")
        with pytest.raises(ValueError): native.validate_partition(item, tmp_path, replay=True, cache=cache)
    finally: reference.write_bytes(original)


def test_actual_native_cnv_workflow_consumes_raw_derived_arrays(completed, tmp_path, monkeypatch):
    from cc_hhgt.v32 import native_cnv_workflow as workflow
    spec = deepcopy(read(completed[1])); spec["directional_success"] = record(completed[3]/"SUCCESS.json")
    manifest = tmp_path/"cnv.json"; write(manifest, spec)
    output = tmp_path/"cnv"
    path = workflow.run(manifest, output)
    _, result = workflow.validate(record(path), output)
    assert result["corrected_inputs_available"] and not result["training_started"]
    plan = read(output/"LAUNCH_PREFLIGHT.json")
    assert plan["directional_materialization"] == spec["directional_success"]
    from cc_hhgt.v32.local_cnv_sources import partition_binding, inspect
    original = output/"association/pair_level/cancer=BRCA/fold=0/part.parquet"
    binding = partition_binding(original, output/"association/receipts/BRCA.json")
    inventory = Inventory([], tmp_path); inspect(inventory, binding, output, "core", "native", 0)
    from cc_hhgt.v32.local_cnv_source_replay import FORMAT as STATISTICS
    assert {native.REPLAY, STATISTICS} <= {r["format"] for r in inventory.transform_replays}
    # Feed the same actual raw-derived store into the existing head's input
    # preparation and five-fold collector. Embeddings are frozen serialization
    # fixtures; no head is fitted or used for predictions.
    from cc_hhgt.v32 import cnv_only_training_v2 as head
    from test_v32_current_core_export import export_fixture
    def forbidden(*a, **k): raise AssertionError("No auxiliary fitting or forward")
    monkeypatch.setattr(head,"_fit_head",forbidden); monkeypatch.setattr(head,"_predict_head",forbidden)
    core_root = tmp_path/"core"; core_root.mkdir(); core_path, core = export_fixture(core_root)
    for fold in core["folds"].values():
        for kind, ids in (("lncRNA", LNCS), ("pathway", [PATHWAY])):
            file = core_root/fold["exports"][kind]["path"]
            pd.DataFrame({"node_index":range(len(ids)),"node_id":ids,"core_feature_000":np.arange(len(ids),dtype=float),
                          "core_feature_001":1.}).to_parquet(file,index=False)
            fold["exports"][kind] = {**record(file),"rows":len(ids),"features":2}
    core["synthetic_serialization_no_optimizer"] = True; write(core_path,core)
    prepared = head.prepare_directional_inputs(candidates_path=completed[2].candidates, folds_path=completed[2].patient_map,
        directional_root=completed[3], association_root=output/"association", core_manifest_path=core_path)
    from test_v32_native_cnv_head_scope import collect_all
    collect_all(prepared)
    from cc_hhgt.v32.cnv_input_sources import validate_consumed_sources
    from cc_hhgt.v32.native_auxiliary_fusion import Sources
    captured = prepared["capture"].write(tmp_path/"CNV_CONSUMED.json",kind="cnv",run_id="SYNTHETIC-NO-FIT")
    source_path, consumed = validate_consumed_sources(Sources(), captured, tmp_path, kind="cnv", run_id="SYNTHETIC-NO-FIT",
        candidates=record(completed[2].candidates),folds=record(completed[2].patient_map),core=record(core_path))
    from cc_hhgt.v32.current_cnv_model_use import inspect_consumed_sources
    downstream = Inventory([], tmp_path)
    inspect_consumed_sources(downstream,{"consumed":(source_path,consumed),
        "patients":pd.read_csv(completed[2].patient_map,sep="\t")},"cnv","native")
    gaps = {r["reason"] for r in downstream.gaps}
    assert "CNV_SEGMENT_ACQUISITION_AND_CROSS_MODULE_PATIENT_CLOSURE_UNVERIFIED" in gaps
    assert "CNV_RAW_TRANSFORMATION_AND_CROSS_MODULE_PATIENT_CLOSURE_NOT_VERIFIED" not in gaps


def test_wrong_host_rejected_before_preparation_and_legacy_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"): native.load_manifest({},tmp_path)
    from scripts.materialize_v32_directional_cnv import main
    with pytest.raises(RuntimeError,match="149"): main([])


@pytest.mark.parametrize("fault", ["missing_array", "dtype", "identity", "source_files", "counts"])
def test_rehashed_partition_contract_cannot_hide_mismatched_inputs(completed, tmp_path, fault):
    part = completed[3]/"cancer=BRCA"
    for source in part.iterdir():
        if source.is_file(): (tmp_path/source.name).write_bytes(source.read_bytes())
    value = read(tmp_path/"SUCCESS.json")
    if fault == "missing_array": value["files"].pop("lncrna_callable.npy")
    elif fault == "counts": value["processed_segment_patients"] += 1
    else:
        if fault == "dtype":
            file = tmp_path/"lncrna_signed_mean.npy"; np.save(file,np.load(file).astype(np.float64))
        elif fault == "identity":
            file = tmp_path/"patient_ids.json"; ids = read(file); ids[0],ids[1] = ids[1],ids[0]; write(file,ids)
        else:
            file = tmp_path/"SOURCE_FILES.json"; raw = read(file); raw["records"].pop(); write(file,raw)
        value["files"][file.name] = record(file)["sha256"]
    write(tmp_path/"SUCCESS.json",value)
    with pytest.raises(ValueError): native.validate_partition(record(tmp_path/"SUCCESS.json"),tmp_path,replay=True)


@pytest.mark.parametrize("fault", ["partition_path", "cancer_coverage", "missing_file_record"])
def test_release_rejects_scope_drift(completed, tmp_path, fault):
    path = completed[3]/"SUCCESS.json"; original = path.read_bytes()
    try:
        value = read(path)
        if fault == "partition_path": value["cancers"]["BRCA"]["path"] = value["cancers"]["LUAD"]["path"]
        elif fault == "cancer_coverage": value["cancers"].pop("LUAD")
        else: value["cancers"]["BRCA"]["success_sha256"] = "0"*64
        write(path,value)
        with pytest.raises(ValueError): native.validate_release(record(path),tmp_path,replay=True)
    finally: path.write_bytes(original)


def test_immutable_outputs_and_failed_audit_receipt(completed, tmp_path):
    item = record(completed[0]); before = record(completed[3]/"cancer=BRCA/SUCCESS.json")
    with pytest.raises(FileExistsError): native.materialize(item,completed[0].parent,completed[3],"BRCA")
    assert record(completed[3]/"cancer=BRCA/SUCCESS.json") == before
    from scripts.audit_native_directional_cnv import main
    output = tmp_path/"audit.json"
    with pytest.raises(ValueError):
        main(["--success",before["path"],"--expected-sha256","0"*64,"--output",str(output)])
    assert read(output)["status"] == "FAILED" and not read(output)["training_started"]
