"""Actual native control validation on synthetic files; no optimizer or real cohort."""
from pathlib import Path
import copy
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_model_use import Inventory,inventory_tables,read,write,record
from cc_hhgt.v32.current_core_model_use import graph_sources
from cc_hhgt.v32.model_control_sources import inspect_patients,inspect_graph_receipt
from cc_hhgt.v32.model_use_completeness import is_complete


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**kw):raise AssertionError("No fitting during control inspection")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture
def patient_control(tmp_path,monkeypatch):
    from cc_hhgt.v32 import patient_fold_authority as patient,formal_graph_authority as graph
    import test_v32_formal_prepare_graph_authority as fixtures
    activity=tmp_path/"synthetic_activity";activity.mkdir()
    pd.DataFrame({"cancer_id":"BRCA","sample_id":[f"S{i}" for i in range(20)],
        "patient_id":[f"PRIVATE-PATIENT-{i}" for i in range(20)],"pathway_id":"P1","activity_score":1.}).to_parquet(activity/"part.parquet",index=False)
    root=tmp_path/"patients"
    patient.build_patient_fold_authority(activity_root=activity,output_root=root,expected_cancers=["BRCA"],expected_patients=20)
    manifest=root/"SAMPLE_PATIENT_FOLD_MAP.tsv";receipt=root/"PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    audit=patient.validate_patient_fold_authority(manifest,receipt,expected_cancers=["BRCA"])
    # Only frozen cohort identities/counts are synthetic. Both real validators
    # run unchanged against actual builder output; no validator is substituted.
    monkeypatch.setattr(patient,"TCGA_CANCERS",("BRCA",));monkeypatch.setattr(patient,"FROZEN_V32_PATIENTS",20)
    for module in (patient,graph,fixtures):
        monkeypatch.setattr(module,"FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256",record(manifest)["sha256"])
        monkeypatch.setattr(module,"FROZEN_V32_RECEIPT_SHA256",record(receipt)["sha256"])
    monkeypatch.setattr(patient,"FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256",audit["patient_authority_sha256"])
    monkeypatch.setattr(patient,"FROZEN_V32_SAMPLE_PATIENT_MAP_LOGICAL_SHA256",audit["sample_patient_map_sha256"])
    monkeypatch.setattr(fixtures,"PATIENT_MAP",manifest);monkeypatch.setattr(fixtures,"PATIENT_RECEIPT",receipt)
    binding={"format":patient.FORMAL_PREPARED_BINDING_FORMAT,"status":"PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
        "sample_id_patient_fallback_used":False,"legacy_patient_fold_manifest_used":False,
        "sample_patient_fold_map":record(manifest),"authority_receipt":record(receipt),
        "authority":patient.validate_frozen_v32_patient_fold_binding(manifest,receipt)}
    return binding


def test_real_patient_controls_are_dependencies_without_fabricated_study_or_fitted_patient_usage(patient_control,tmp_path):
    inv=Inventory([],tmp_path)
    known=inspect_patients(inv,patient_control,tmp_path,"core",0)
    tables=inventory_tables(inv,["core"])
    assert len(tables["dependencies"])==2 and not len(tables["gaps"])
    assert known=={patient_control[k]["sha256"] for k in ("sample_patient_fold_map","authority_receipt")}
    assert not len(tables["usage"]) and not len(tables["patient_usage_private"])
    assert not is_complete(tables["completeness"])  # Controls alone prove no study coverage.


def test_patient_metadata_without_files_remains_incomplete(patient_control,tmp_path):
    binding=copy.deepcopy(patient_control)
    for key in ("sample_patient_fold_map","authority_receipt"):binding[key].pop("path")
    inv=Inventory([],tmp_path);inspect_patients(inv,binding,tmp_path,"core",0)
    assert {"HASH_BOUND_AUTHORITY_NOT_LOCATED","PATIENT_CONTROL_FILES_NOT_VERIFIED"}<=set(r["reason"] for r in inv.gaps)
    assert not inv.usage


def test_patient_files_can_be_relocated_by_hash_but_changed_bytes_cannot(patient_control,tmp_path):
    binding=copy.deepcopy(patient_control);locations=[];directory=tmp_path/"relocated";directory.mkdir()
    for key in ("sample_patient_fold_map","authority_receipt"):
        item=binding[key];path=directory/Path(item["path"]).name;path.write_bytes(Path(item["path"]).read_bytes())
        item.pop("path");locations.append(record(path))
    inv=Inventory(locations,tmp_path);inspect_patients(inv,binding,tmp_path,"core",0)
    assert inv.used_locations==set(inv.locations) and not inv.gaps
    manifest=directory/"SAMPLE_PATIENT_FOLD_MAP.tsv";manifest.write_bytes(manifest.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="SHA256"):inspect_patients(Inventory(locations,tmp_path),binding,tmp_path,"core",0)


def test_fivefold_graph_receipts_are_checked_controls_while_raw_tables_remain_untraced(patient_control,tmp_path):
    from test_v32_formal_prepare_graph_authority import _fixture
    from scripts.prepare_v32_formal import build_graph
    from cc_hhgt.v32.formal_graph_authority import bind_formal_graph_variant
    root=tmp_path/"graph";root.mkdir();inputs,candidates,splits=_fixture(root)
    inv=Inventory([],root)
    for fold in range(5):
        bound=build_graph(inputs,outer_fold=fold,split_manifest=splits[fold],candidate_pairs=candidates)
        binding=bind_formal_graph_variant(bound.binding,bound.authority,"G2")
        graph_sources(inv,binding,root,"core",fold)
    tables=inventory_tables(inv,["core"])
    assert set(tables["dependencies"].loc[tables["dependencies"].role.eq("graph.receipt"),"patient_fold_id"])==set(range(5))
    assert not tables["gaps"].role.eq("graph.receipt").any()
    assert "RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED" in set(tables["gaps"].reason)
    assert not is_complete(tables["completeness"])
    assert not tables["usage"].source_file_sha256.eq(inputs.receipt_sha256).any()


@pytest.mark.parametrize("fault",["policy","static_hash","source_role","fold_hash","train_identity","missing_fold"])
def test_repinned_graph_control_cannot_disagree_with_consumed_inputs(patient_control,tmp_path,fault):
    from test_v32_formal_prepare_graph_authority import _fixture
    from scripts.prepare_v32_formal import build_graph
    from cc_hhgt.v32.formal_graph_authority import bind_formal_graph_variant
    root=tmp_path/"graph";root.mkdir();inputs,candidates,splits=_fixture(root)
    bound=build_graph(inputs,outer_fold=0,split_manifest=splits[0],candidate_pairs=candidates)
    binding=bind_formal_graph_variant(bound.binding,bound.authority,"G2")
    value=read(inputs.receipt_path)
    if fault=="policy":value["gates"]["outer_train_expression_only"]=False
    elif fault=="static_hash":value["artifacts"]["ppi"]["sha256"]="0"*64
    elif fault=="source_role":value["artifacts"]["ppi"]["outcome_derived"]=True
    elif fault=="fold_hash":value["fold_artifacts"]["0"]["train_expression"]["sha256"]="0"*64
    elif fault=="train_identity":value["fold_artifacts"]["0"]["train_expression"]["train_patient_sha256"]="0"*64
    else:value["fold_artifacts"].pop("4")
    path=tmp_path/"repinned.json";write(path,value);binding["receipt"]=record(path)
    with pytest.raises((ValueError,RuntimeError)):
        inspect_graph_receipt(Inventory([],tmp_path),binding,tmp_path,"core","graph.receipt",0)


def test_actual_expansion_loader_proves_logical_receipt_without_looking_for_a_file(tmp_path):
    from test_v32_current_binding_expansion import overlay_inputs
    from cc_hhgt.v32.current_binding_overlay import apply_binding_overlay
    payload,config,overlay=overlay_inputs();path=tmp_path/"overlay.pt";torch.save(overlay,path)
    source=record(path)
    actual=apply_binding_overlay(payload,overlay,fold=0,arm="experimental_predicted",overlay_sha=source["sha256"],
        comparison_sha=overlay["formal_graph_authority"]["comparison_sha256"],config=config)
    inv=Inventory([],tmp_path)
    known=graph_sources(inv,actual["formal_graph_authority"],tmp_path,"core",0,
        expansion_source=source,verified_expansion=actual["binding_expansion_authority"])
    assert actual["formal_graph_authority"]["receipt"]["sha256"] in known
    assert actual["formal_graph_authority"]["comparison_sha256"] in known
    assert not any(r["role"]=="graph.receipt" for r in inv.gaps)
    assert any(r["role"].startswith("graph.base.") for r in inv.gaps)  # Unavailable base ancestry remains.
    assert any(r["role"]=="graph.receipt.sidecar" for r in inv.dependencies)
    inv=Inventory([],tmp_path);graph_sources(inv,actual["formal_graph_authority"],tmp_path,"core",0)
    assert "EXPANSION_CONTROL_SIDECAR_NOT_VERIFIED" in {r["reason"] for r in inv.gaps}


@pytest.mark.parametrize("fault",["audit","logical_receipt","repinned_audit","repinned_arm_count"])
def test_actual_expansion_loader_rejects_a_rehashed_control_lie(tmp_path,fault):
    from test_v32_current_binding_expansion import overlay_inputs
    from cc_hhgt.v32.current_binding_overlay import apply_binding_overlay,canonical_json_sha
    payload,config,overlay=overlay_inputs()
    if fault in {"audit","repinned_audit"}:overlay["audit"]["appended_nodes"]+=1
    elif fault=="repinned_arm_count":overlay["audit"]["arm_edge_counts"]["experimental_predicted"]+=1
    else:overlay["formal_graph_authority"]["receipt"]["sha256"]="0"*64
    if fault.startswith("repinned"):
        overlay["formal_graph_authority"]["receipt"]["sha256"]=canonical_json_sha({
            "audit":overlay["audit"],"comparison":overlay["formal_graph_authority"]["comparison"]})
    path=tmp_path/"repinned.pt";torch.save(overlay,path)
    with pytest.raises(ValueError,match="control (receipt|audit)"):
        apply_binding_overlay(payload,overlay,fold=0,arm="experimental_predicted",overlay_sha=record(path)["sha256"],
            comparison_sha=overlay["formal_graph_authority"]["comparison_sha256"],config=config)


def test_actual_core_and_external_consumer_validate_patient_files_without_waiving_other_gaps(patient_control,tmp_path,monkeypatch):
    import test_v32_current_binding_expansion as fixtures
    import test_v32_training_objective as objective
    from cc_hhgt.v32 import patient_fold_authority as patient
    from test_v32_local_cnv_source_ancestry import test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs
    original=fixtures.verified_payload
    for module in (objective,):
        monkeypatch.setattr(module,"FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256",patient.FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256)
        monkeypatch.setattr(module,"FROZEN_V32_RECEIPT_SHA256",patient.FROZEN_V32_RECEIPT_SHA256)
    def payload(fold=0):
        result=original(fold);result["patient_fold_authority"]=copy.deepcopy(patient_control)
        return result
    monkeypatch.setattr(fixtures,"verified_payload",payload)
    test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs(tmp_path,monkeypatch)
    from cc_hhgt.v32.external_validation_current import CurrentExternalValidationQuery
    source=record(tmp_path/"evaluation/EXTERNAL_BINDING.json")
    query=CurrentExternalValidationQuery(source["path"],expected_sha=source["sha256"])
    rows=query.query(table="model_dependencies",filters={"role":"patient.sample_patient_fold_map"})["rows"]
    assert {r["patient_fold_id"] for r in rows}==set(range(5)) and all("path" not in r for r in rows)
    gaps=query.query(table="model_dependency_gaps",limit=1000)["rows"]
    assert not any(r["role"].startswith("patient.") for r in gaps)
    assert gaps and not query.query()["model_use_inventory_completeness_verified"]
