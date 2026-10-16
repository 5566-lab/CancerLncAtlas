"""Native core export -> private heads -> model-use isolation, no optimization."""
import copy
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_model_use import Inventory,record,read,write,derive_manifest,materialize_current_model_use
from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_core_model_use import inspect_export,prepared_candidates,graph_sources


def forbidden(*a,**kw):raise AssertionError("No optimizer or model construction during source inspection")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    import scripts.export_v32_current_core_embeddings as exporter
    import test_v32_current_binding_expansion as fixtures
    from test_v32_current_core_prediction import current_fixture
    from test_v32_preparation_sources import adjusted_capture
    from cc_hhgt.v32.preparation_sources import attach_sources
    from scripts.prepare_v32_formal import make_batches,attach_replication_labels
    root=tmp_path_factory.mktemp("core_export_sources")
    original=fixtures.verified_payload
    def payload(fold=0):
        result=original(fold)
        _,receipt,frames=adjusted_capture(root/f"sources{fold}",fold,train_pmid="55667788",validation_pmid="88990011")
        for frame in frames.values():
            frame["shared_or_local_scope"]="shared";frame["detection_rate"]=.8;frame["detected_cancers"]=5
        frames["validation"]=attach_replication_labels(frames["train"],frames["validation"])
        for split in ("train","validation"):
            result[split+"_batches"]=make_batches(frames[split],np.zeros(1),result["bundle"],split=split)
        result["input_scope"]["association_mode"]="local-cnv-adjusted"
        result["label_contract"].update(train_direction_label_source="train_local_cnv_adjusted_effect",
            validation_direction_label_source="validation_local_cnv_adjusted_replication_effect",
            association_target="A1_LOCAL",association_available_mask="association_available")
        attach_sources(result,receipt)
        graph_source=root/f"graph{fold}.parquet"
        pd.DataFrame({"lncrna_id":["LNC:Z"],"protein_id":["UNIPROT:Z"],"graph_assay_class":["eclip"],
            "pmid":["66778899"],"private_donor":["PRIVATE_CORE_DONOR"]}).to_parquet(graph_source,index=False)
        result["formal_graph_authority"]["static_artifacts"]["lnc_protein_binding"]=record(graph_source)
        return result
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        from sklearn.linear_model import LogisticRegression
        patch.setattr(LogisticRegression,"fit",forbidden)
        patch.setattr(fixtures,"verified_payload",payload)
        source=current_fixture(root/"inputs");inputs=read(source)
        inputs["format"]="CURRENT_CORE_EXPORT_INPUT_V1"
        # The input manifest uses relative paths in a different directory from
        # the output. Published source records must resolve to the actual files.
        for item in inputs["fold_inputs"]:
            for key,value in list(item.items()):
                if isinstance(value,dict) and "path" in value:
                    value["path"]=str(Path(value["path"]).relative_to(source.parent))
        write(source,inputs)
        assert exporter.main(["--manifest",str(source),"--expected-core-run-id",inputs["core_run_id"],
                              "--output",str(root/"export")])==0
    return root,root/"export/CORE_EMBEDDING_MANIFEST.json",inputs["core_run_id"]


def test_actual_export_retains_native_training_and_validation_sources(exported,monkeypatch):
    import scripts.export_v32_current_core_embeddings as exporter
    root,path,run=exported;monkeypatch.setattr(exporter,"_build_model",forbidden)
    observed=read(path)
    assert all(Path(r["path"]).is_absolute() for f in observed["folds"].values() for r in f["sources"].values())
    inventory=Inventory([],root)
    with inventory.context("FUSION_EXPERT:mutation"):
        assert inspect_export(inventory,path,"fusion","auxiliary.core")
    usage=pd.DataFrame(inventory.usage)
    assert {"TRAINING","MODEL_SELECTION","FEATURE_CONSTRUCTION"} <= set(usage.phase)
    assert usage.study_tokens_json.str.contains("55667788").any()
    assert usage.study_tokens_json.str.contains("88990011").any()
    graph=usage.loc[usage.study_tokens_json.str.contains("66778899")]
    assert set(graph.patient_fold_id)==set(range(5))
    assert not usage.astype(str).apply(lambda c:c.str.contains("PRIVATE_CORE_DONOR",regex=False)).any().any()
    assert not usage.duplicated(["model_id","source_event_id"]).any()
    assert usage.dependency_context.eq("FUSION_EXPERT:mutation").all()


@pytest.mark.parametrize("kind,pmid",[("genomic","55667788"),("cnv","88990011"),("cnv_event","66778899")])
def test_core_only_study_reaches_actual_auxiliary_fusion_and_external_isolation(exported,tmp_path,monkeypatch,kind,pmid):
    from test_v32_auxiliary_model_use import setup
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    import scripts.export_v32_current_core_embeddings as exporter
    root,core,run=exported;monkeypatch.setattr(exporter,"_build_model",forbidden)
    keys=pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]})
    path,_=setup(tmp_path/"auxiliary",kind,core_override=(core,keys,run))
    _,_,tables,_=derive_manifest(path)
    assert "AUXILIARY_CORE_TRAINING_ANCESTRY_NOT_FULLY_RECONSTRUCTED" not in set(tables["gaps"].reason)
    assert tables["usage"].study_tokens_json.str.contains("55667788").any()
    assert tables["usage"].study_tokens_json.str.contains("66778899").any()
    inventory=Path(materialize_current_model_use(path,tmp_path/"inventory")["binding_path"])
    x=tmp_path/"crosswalk.parquet";e=tmp_path/"external.parquet"
    pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","breast","BRCA"),("pathway","p","P1")],
        columns=["namespace","raw_id","canonical_id"]).to_parquet(x,index=False)
    pd.DataFrame({"record_id":["synthetic"],"lncrna":["new"],"cancer":["breast"],"pathway":["p"],"label":[1],
        "pmid":[pmid],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic assay"]}).to_parquet(e,index=False)
    source=tmp_path/"external.json"
    write(source,{"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":read(path)["models"],
        "model_use_inventory":record(inventory),"identity_crosswalk":record(x),
        "external_sources":[{"kind":"exact_assay","source_id":"core_only_study","table":record(e)}],
        "evaluation":{"primary_model":"fusion","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}})
    result=materialize_current_external(source,tmp_path/"external")
    query=CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    rows=query.query(table="isolation")["rows"]
    assert rows and all(r["study_overlap"] and not r["lncrna_overlap"] and not r["provenance_held_out"] for r in rows)
    assert all("path" not in r for r in query.query(table="model_dependencies")["rows"])


@pytest.mark.parametrize("fault",["state_hash","node_identity","missing_source","copied_receipt","source_manifest"])
def test_repinned_export_semantic_drift_rejected(exported,tmp_path,monkeypatch,fault):
    import scripts.export_v32_current_core_embeddings as exporter
    _,path,_=exported;monkeypatch.setattr(exporter,"_build_model",forbidden)
    export=read(path)
    for part in export["folds"].values():
        for item in [part["training_receipt"],*part["exports"].values()]:
            item["path"]=str(path.parent/item["path"])
    part=export["folds"]["0"]
    if fault=="state_hash":part["core_parameter_sha256"]="b"*64
    elif fault=="missing_source":part["sources"].pop("prepared")
    elif fault=="copied_receipt":part["sources"]["training_receipt"]["sha256"]="b"*64
    elif fault=="source_manifest":
        source=read(Path(export["source_inputs"]["path"]));source["core_run_id"]="another_run"
        changed=tmp_path/"export_input.json";write(changed,source);export["source_inputs"]=record(changed)
    else:
        data=pd.read_parquet(part["exports"]["lncRNA"]["path"]);data.loc[0,"node_id"]="LNC:WRONG"
        changed=tmp_path/"nodes.parquet";data.to_parquet(changed,index=False)
        part["exports"]["lncRNA"]={**part["exports"]["lncRNA"],**record(changed)}
    changed=tmp_path/"CORE.json";write(changed,export)
    with pytest.raises(ValueError,match="differ|incomplete"):
        inspect_export(Inventory([],tmp_path),changed,"fusion","core")


def test_changed_core_graph_source_is_rejected(exported,monkeypatch):
    import scripts.export_v32_current_core_embeddings as exporter
    root,path,_=exported;monkeypatch.setattr(exporter,"_build_model",forbidden)
    graph=root/"graph0.parquet";original=graph.read_bytes()
    try:
        graph.write_bytes(original+b"changed")
        with pytest.raises(ValueError,match="SHA256 mismatch"):
            inspect_export(Inventory([],root),path,"fusion","core")
    finally:graph.write_bytes(original)


def test_hash_bound_export_input_locator_preserves_source_identity(exported,tmp_path,monkeypatch):
    import scripts.export_v32_current_core_embeddings as exporter
    _,path,_=exported;monkeypatch.setattr(exporter,"_build_model",forbidden)
    source=Path(read(path)["source_inputs"]["path"])
    relocated=tmp_path/"relocated_input.json";relocated.write_bytes(source.read_bytes())
    inventory=Inventory([record(relocated)],tmp_path)
    assert inspect_export(inventory,path,"fusion","core")
    assert sha256(relocated) in inventory.used_locations
    assert any(r["role"]=="core.export_inputs" and r["path"]==str(relocated) for r in inventory.dependencies)


def test_missing_historical_sources_remain_explicit(tmp_path):
    from test_v32_current_core_export import export_fixture
    path,_=export_fixture(tmp_path);inventory=Inventory([],tmp_path)
    assert not inspect_export(inventory,path,"fusion","core")
    assert {r["patient_fold_id"] for r in inventory.gaps}==set(range(5))
    assert {r["reason"] for r in inventory.gaps}=={"CORE_EXPORT_TRAINING_SOURCES_NOT_BOUND"}


def test_expanded_graph_retains_base_source_metadata_and_fold_identity(tmp_path):
    path=tmp_path/"source.parquet";pd.DataFrame({"pmid":["11229988"]}).to_parquet(path,index=False)
    graph={"base_authority":{"static_artifacts":{"lnc_protein_binding":record(path)},"receipt":{"sha256":"a"*64}},
           "receipt":{"sha256":"b"*64}}
    inventory=Inventory([],tmp_path)
    graph_sources(inventory,graph,tmp_path,"primary",2)
    assert any("11229988" in r["study_tokens_json"] and r["patient_fold_id"]==2 for r in inventory.usage)
    assert any(r["role"]=="graph.base.static_artifacts.lnc_protein_binding" for r in inventory.dependencies)


def test_actual_input_keys_are_reconstructed_without_downstream_candidates():
    from test_v32_current_binding_expansion import verified_payload
    payload=verified_payload();keys=prepared_candidates(payload)
    assert keys.to_dict("records")==[{"lncrna_id":"LNC:Z","pathway_id":"P1","cancer_id":"BRCA"}]
    payload["train_batches"][0]["candidate_batch"]["l"][0]=99
    with pytest.raises(ValueError,match="unknown node"):prepared_candidates(payload)
