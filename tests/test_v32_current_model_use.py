"""Actual native input exposure and external consumption; no optimization."""
import json
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_model_use import (INPUT_FORMAT,SCOPE,record,read,write,
    materialize_current_model_use,load_current_model_use,derive_manifest)


def forbidden(*args,**kwargs):raise AssertionError("No model fitting/forwards during native model-use extraction")


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    import cc_hhgt.v32.multimodal_fusion as fusion
    monkeypatch.setattr(fusion,"fit_residual_fusion_model",forbidden)


@pytest.fixture(scope="module")
def native_inventory(tmp_path_factory):
    import test_v32_current_binding_expansion as fixtures
    from test_v32_current_core_prediction import current_fixture
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.selected_score import materialize_selected_scores
    root=tmp_path_factory.mktemp("native_model_use")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        original=fixtures.verified_payload
        def payload(fold=0):
            result=original(fold)
            result["train_batches"][0]["association_available"]=torch.tensor([fold!=2])
            result["validation_batches"][0]["association_available"]=torch.tensor([fold!=1])
            return result
        patch.setattr(fixtures,"verified_payload",payload)
        manifest=current_fixture(root/"inputs")
        primary=materialize_current_predictions(manifest,root/"prediction",expected_run_id="CURRENT-V4-SYNTHETIC")
        spec={"model_id":"primary","producer_kind":"current_core_release","producer_binding":record(primary),
            "run_id":"CURRENT-V4-SYNTHETIC","score_column":"association_membership_probability"}
        selected={"format":"CURRENT_SELECTED_ANALYSIS_INPUT_V1","selected_model":spec,"reference_exact_model":spec,
            "selection":{"kind":"EXPLICIT_SOURCE_REGISTRATION","rationale":"Synthetic unchanged parameters"}}
        selected_path=root/"selected.json";write(selected_path,selected)
        selected_binding=materialize_selected_scores(selected_path,root/"selected")
        models=[{"model_id":"selected","producer_kind":"selected_release","producer_binding":record(selected_binding),
            "run_id":"CURRENT-V4-SYNTHETIC","score_column":"selected_score"}]
        # The actual input-inspection path must not construct a model.
        import scripts.export_v32_current_core_embeddings as exporter
        patch.setattr(exporter,"_build_model",forbidden)
        source=root/"MODEL_USE_INPUT.json";write(source,{"format":INPUT_FORMAT,"models":models})
        result=materialize_current_model_use(source,root/"inventory")
    return root,models,Path(result["binding_path"])


def test_native_training_selection_and_graph_inputs_are_derived_with_masks(native_inventory,monkeypatch):
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter,"_build_model",forbidden)
    _,models,path=native_inventory
    binding,tables=load_current_model_use(path,expected_sha=sha256(path),models=models)
    usage=tables["usage"]
    train=usage.loc[usage.phase.eq("TRAINING")]
    validation=usage.loc[usage.phase.eq("MODEL_SELECTION")]
    assert set(train.patient_fold_id)=={0,1,3,4} and set(validation.patient_fold_id)=={0,2,3,4}
    assert train.lncrna_id.eq("LNC:Z").all() and train.candidate_rows.eq(1).all()
    features=usage.loc[usage.phase.eq("FEATURE_CONSTRUCTION")]
    assert set(features.patient_fold_id)==set(range(5)) and features.lncrna_id.eq("LNC:Z").any()
    assert set(features.use_view)=={"GRAPH_NODE_INPUT","CANDIDATE_FEATURES"}
    assert usage.pathway_id.eq("").all() and usage.entity_scope.eq("LNCRNA_EXPOSURE_ALL_PATHWAYS").all()
    assert usage.study_tokens_json.eq("[]").all() and "patient_id" not in usage
    assert not binding["model_use_inventory_completeness_verified"] and not binding["model_forwards_executed"]
    assert len(tables["dependencies"].query("role == 'checkpoint'"))==5
    assert "HASH_BOUND_AUTHORITY_NOT_LOCATED" in set(tables["gaps"].reason)
    assert "RAW_FEATURE_LABEL_STUDY_LINEAGE_NOT_RECONSTRUCTED" in set(tables["gaps"].reason)


def test_native_inventory_cannot_be_relabelled_for_another_score_or_claim_complete(native_inventory,tmp_path):
    _,models,path=native_inventory
    wrong=[{**models[0],"score_column":"other_score"}]
    with pytest.raises(ValueError,match="different evaluated models"):
        load_current_model_use(path,expected_sha=sha256(path),models=wrong)
    binding=read(path);binding["model_use_inventory_completeness_verified"]=True
    changed=tmp_path/"BINDING.json";write(changed,binding)
    with pytest.raises(ValueError,match="unsupported completeness"):
        load_current_model_use(changed,expected_sha=sha256(changed),replay=False)


def test_rehashed_native_use_omission_fails_actual_input_replay(native_inventory,tmp_path):
    _,_,path=native_inventory;binding=read(path)
    data=pd.read_parquet(binding["outputs"]["usage"]["path"])
    data=data.loc[~data.phase.eq("TRAINING")]
    table=tmp_path/"USAGE.parquet";data.to_parquet(table,index=False)
    binding["outputs"]["usage"]={**record(table),"rows":len(data)}
    changed=tmp_path/"BINDING.json";write(changed,binding)
    with pytest.raises(ValueError,match="recomputation differs"):
        load_current_model_use(changed,expected_sha=sha256(changed))


def test_external_v2_uses_actual_native_exposure_and_reports_unresolved_provenance(native_inventory,tmp_path):
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    _,models,path=native_inventory
    frames={"crosswalk":pd.DataFrame([("lncrna","Z","LNC:Z"),("lncrna","other","LNC:OTHER"),
        ("cancer","breast","BRCA"),("pathway","p1","P1")],columns=["namespace","raw_id","canonical_id"]),
        "declared":pd.DataFrame({"lncrna_id":["LNC:OTHER"],"pathway_id":["P1"],"cancer_id":["BRCA"],"pmid":["99999999"]}),
        "external":pd.DataFrame({"record_id":["synthetic1"],"lncrna":["Z"],"cancer":["breast"],"pathway":["p1"],
            "label":[1],"pmid":["12345678"],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic measured assay"]})}
    files={}
    for name,frame in frames.items():
        file=tmp_path/(name+".parquet");frame.to_parquet(file,index=False);files[name]=record(file)
    manifest={"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","identity_crosswalk":files["crosswalk"],"models":models,
        "model_use_inventory":record(path),"model_use_sources":[{"kind":"canonical_usage","source_id":"declared",
            "phase":"TRAINING","model_ids":["selected"],"table":files["declared"]}],
        "external_sources":[{"kind":"exact_assay","source_id":"external","table":files["external"]}],
        "evaluation":{"primary_model":"selected","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}}
    source=tmp_path/"INPUT.json";write(source,manifest)
    result=materialize_current_external(source,tmp_path/"evaluation")
    query=CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    rows=query.query(table="isolation")["rows"]
    assert rows[0]["lncrna_overlap"] and rows[0]["pair_overlap"] and rows[0]["descriptive_eligible"]
    assert not rows[0]["provenance_held_out"] and "NATIVE_DEPENDENCIES_INCOMPLETE" in rows[0]["isolation_reasons_json"]
    assert query.query()["isolation_scope"]==SCOPE
    assert "path" not in pd.read_parquet(query.paths["model_dependencies"]).columns
    assert len(pd.read_parquet(query.paths["model_dependency_gaps"]))>0
    # A manually declared disjoint source alone would have missed this exposure.
    from cc_hhgt.v32.external_validation_current import evaluate_manifest
    manifest["format"]="CURRENT_EXTERNAL_VALIDATION_INPUT_V1";manifest.pop("model_use_inventory");write(source,manifest)
    _,legacy,_=evaluate_manifest(source)
    assert not legacy["isolation"].lncrna_overlap.any() and legacy["isolation"].provenance_held_out.all()


def test_unused_locations_cannot_smuggle_external_data_into_the_dependency_inventory(native_inventory,tmp_path):
    _,models,_=native_inventory
    file=tmp_path/"unused.txt";file.write_text("not a native dependency")
    source=tmp_path/"INPUT.json";write(source,{"format":INPUT_FORMAT,"models":models,"artifact_locations":[record(file)]})
    with pytest.raises(ValueError,match="Unused model-use artifact locator"):derive_manifest(source)


def test_selected_l1_score_does_not_claim_reference_neural_graph_or_validation_use(native_inventory,tmp_path):
    from cc_hhgt.v32.selected_score import materialize_selected_scores
    root,_,_=native_inventory
    source=read(root/"selected.json")
    source["selected_model"]={**source["selected_model"],"score_column":"l1_probability"}
    path=tmp_path/"selected.json";write(path,source)
    selected=materialize_selected_scores(path,tmp_path/"selected")
    model={"model_id":"baseline","producer_kind":"selected_release","producer_binding":record(selected),
        "run_id":"CURRENT-V4-SYNTHETIC","score_column":"selected_score"}
    path=tmp_path/"INPUT.json";write(path,{"format":INPUT_FORMAT,"models":[model]})
    _,_,tables,_=derive_manifest(path)
    assert set(tables["usage"].use_view)=={"CANDIDATE_FEATURES"}
    assert set(tables["usage"].phase)=={"FEATURE_CONSTRUCTION"}
    assert not tables["dependencies"].role.str.startswith("graph.").any()
    assert set(tables["gaps"].reason)=={"BASELINE_FITTING_STUDY_DEPENDENCIES_NOT_RECONSTRUCTED"}


def test_directory_authority_uses_native_tree_hash_and_cannot_hide_a_changed_partition(tmp_path):
    from cc_hhgt.v32.current_model_use import Inventory,verified_source
    from cc_hhgt.v32.input_lineage import artifact_sha256
    source=tmp_path/"assays";source.mkdir();part=source/"part.parquet"
    pd.DataFrame({"value":[0.,1.]}).to_parquet(part,index=False)
    expected={"path":str(source),"sha256":artifact_sha256(source)}
    inventory=Inventory([],tmp_path);inventory.authority(expected,tmp_path,"model","graph.fold_artifacts.train_expression",0)
    assert inventory.dependencies[0]["status"]=="VERIFIED_DIRECTORY"
    assert inventory.gaps[0]["reason"]=="RAW_STUDY_LINEAGE_NOT_RECONSTRUCTED"
    pd.DataFrame({"value":[9.,1.]}).to_parquet(part,index=False)
    with pytest.raises(ValueError,match="SHA256 mismatch"):verified_source(expected,tmp_path)


def test_actual_fusion_inputs_include_zero_coefficient_experts_and_explicit_missing_ancestry(tmp_path):
    from test_v32_selected_analysis import exact_fixture,fusion_fixture
    spec=fusion_fixture(tmp_path/"fusion",exact_fixture(tmp_path/"exact"),development=True)
    source=tmp_path/"INPUT.json";write(source,{"format":INPUT_FORMAT,"models":[spec]})
    result=materialize_current_model_use(source,tmp_path/"inventory")
    binding,tables=load_current_model_use(result["binding_path"],expected_sha=result["binding_sha256"])
    assert set(tables["usage"].phase)=={"FEATURE_CONSTRUCTION","CALIBRATION"}
    assert {"fusion.expert.sc","fusion.expert.zero"}.issubset(set(tables["gaps"].role))
    assert "PRIMARY_NATIVE_BINDING_MISSING" in set(tables["gaps"].reason)
    assert not binding["model_use_inventory_completeness_verified"]
