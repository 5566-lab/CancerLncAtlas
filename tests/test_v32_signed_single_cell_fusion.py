from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.multimodal_fusion import (ANALYSIS_VERSION, PRIMARY_COLUMN, ExpertSpec,
    ResidualFusionModel, MultimodalFusionError, apply_residual_fusion_model,
    artifact_sha256, build_pair_aggregated_fusion_frame, build_pair_aggregated_fusion_frame_from_parquet)
from cc_hhgt.v32.single_cell_signed_fusion import materialize_signed_sc_fusion, SIGNED_SC_EXPERTS


def fixture(tmp_path):
    keys=pd.DataFrame({"cancer_id":["BRCA","KICH"],"lncrna_id":["L1","L1"],"pathway_id":["P1","P1"]})
    keys.to_parquet(tmp_path/"candidates.parquet",index=False)
    rows=[]
    donors=[]
    for donor in range(5):
        donors.append(dict(dataset_id="S1",query_donor_id=f"d{donor}",canonical_donor_id=f"person{donor}"))
        for cell,rho in [("tumour",.8),("immune",-.6)]:
            for arm in ["with_lasso","without_lasso"]:
                rows.append(dict(dataset_id="S1",cancer_id="BRCA",cell_key=cell,query_donor_id=f"d{donor}",
                    lncrna_id="L1",pathway_id="P1",outer_fold_id=donor,predicted_rho=rho if arm=="with_lasso" else 0,
                    arm=arm,rho=999)) # deliberately invalid observed target, not a fusion feature
    pd.DataFrame(rows).to_parquet(tmp_path/"predictions.parquet",index=False)
    pd.DataFrame(donors).to_parquet(tmp_path/"crosswalk.parquet",index=False)
    (tmp_path/"validation.json").write_text(json.dumps({"status":"PASS"}))
    bound=lambda p:{"path":str(p),"sha256":artifact_sha256(p)}
    pointer={"status":"PASS_ACTIVATED_SC_HHGT","estimand":"signed_within_donor_metacell_spearman_rho",
        "split_policy":"donor_blocked_outer_oof_with_nested_inner_validation","training_complete":True,"outer_folds":5,
        "primary_arm":"with_lasso","oof_predictions":bound(tmp_path/"predictions.parquet"),
        "independent_validation":bound(tmp_path/"validation.json")}
    (tmp_path/"pointer.json").write_text(json.dumps(pointer))
    return keys


def run(tmp_path,name="output"):
    return materialize_signed_sc_fusion(tmp_path/"pointer.json",tmp_path/"candidates.parquet",
                                        tmp_path/"crosswalk.parquet",tmp_path/name)


def test_signed_hhgt_reaches_actual_fusion_and_preserves_context(tmp_path):
    keys=fixture(tmp_path)
    audit=run(tmp_path)
    raw=pd.read_parquet(tmp_path/"output/DONOR_CELL_CONTEXT.parquet")
    context=pd.read_parquet(tmp_path/"output/CELL_CONTEXT_SUMMARY.parquet")
    expert=pd.read_parquet(tmp_path/"output/SINGLE_CELL_SIGNED_FUSION.parquet")
    assert len(raw)==10 and "rho" not in raw.columns
    assert context.set_index("cell_key").signed_rho_mean.to_dict()==pytest.approx({"tumour":.8,"immune":-.6})
    brca=expert.loc[expert.cancer_id.eq("BRCA")].iloc[0]
    assert brca.single_cell_positive_strength==pytest.approx(.4)
    assert brca.single_cell_negative_strength==pytest.approx(.3)
    assert brca.single_cell_signed_rho_mean==pytest.approx(.1)
    assert not expert.loc[expert.cancer_id.eq("KICH"),"single_cell_available"].any()
    primary=keys.assign(**{PRIMARY_COLUMN:.25,"analysis_version":ANALYSIS_VERSION})
    folds=[keys.assign(patient_fold_id=i,held_out_proxy_label=1) for i in range(5)]
    frame=build_pair_aggregated_fusion_frame(primary,folds,{spec:expert for spec in SIGNED_SC_EXPERTS})
    model=ResidualFusionModel(endpoint="discovery",expert_ids=tuple(s.expert_id for s in SIGNED_SC_EXPERTS),
        centres=(0.,0.),weights=(1.,1.),seed=1,initial_parameter_sha256="0"*64,final_parameter_sha256="1"*64,
        optimiser_steps=0,validation_logloss=0,feature_transforms=("identity_unit","identity_unit"))
    result=apply_residual_fusion_model(model,frame)
    assert result.loc[result.cancer_id.eq("BRCA"),"discovery_adjusted_probability"].iloc[0]>.25
    assert result.loc[result.cancer_id.eq("KICH"),"discovery_adjusted_probability"].iloc[0]==.25
    assert model.to_dict()["checkpoint_format"].endswith("V2")
    assert audit["raw_rho_treated_as_probability"] is False


@pytest.mark.parametrize("corruption",["cross_fold_donor","missing_crosswalk","duplicates","range"])
def test_signed_adapter_rejects_corrupt_donor_predictions(tmp_path,corruption):
    fixture(tmp_path)
    if corruption in ["cross_fold_donor","missing_crosswalk"]:
        path=tmp_path/"crosswalk.parquet";frame=pd.read_parquet(path)
        if corruption=="cross_fold_donor":frame.loc[1,"canonical_donor_id"]="person0"
        else:frame=frame.iloc[:-1]
        frame.to_parquet(path,index=False)
    else:
        path=tmp_path/"predictions.parquet";frame=pd.read_parquet(path)
        if corruption=="duplicates":frame=pd.concat([frame,frame.iloc[[0]]])
        else:frame.loc[0,"predicted_rho"]=1.5
        frame.to_parquet(path,index=False)
        pointer=json.loads((tmp_path/"pointer.json").read_text())
        pointer["oof_predictions"]["sha256"]=artifact_sha256(path)
        (tmp_path/"pointer.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError):run(tmp_path)
    assert not (tmp_path/"output/SUCCESS.json").exists()


def test_signed_transform_preserves_negative_values_and_parquet_builder_matches(tmp_path):
    # Enough distinct pairs to occupy all five fusion pair folds.
    keys=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":[f"L{i}" for i in range(80)],"pathway_id":"P1"})
    primary=keys.assign(**{PRIMARY_COLUMN:.5,"analysis_version":ANALYSIS_VERSION})
    primary.to_parquet(tmp_path/"primary.parquet",index=False)
    folds=[keys.assign(patient_fold_id=i,held_out_proxy_label=i%2) for i in range(5)]
    paths=[]
    for i,frame in enumerate(folds):
        path=tmp_path/f"fold{i}.parquet";frame.to_parquet(path,index=False);paths.append(path)
    spec=ExpertSpec("signed_sc","rho","available",("discovery",),"donor",value_transform="signed_fisher")
    expert=keys.assign(rho=np.linspace(-.8,.8,80),available=True,analysis_version=ANALYSIS_VERSION)
    expert.to_parquet(tmp_path/"expert.parquet",index=False)
    memory=build_pair_aggregated_fusion_frame(primary,folds,{spec:expert})
    disk,_=build_pair_aggregated_fusion_frame_from_parquet(tmp_path/"primary.parquet",paths,{spec:tmp_path/"expert.parquet"})
    np.testing.assert_allclose(memory.signed_sc_probability,disk.signed_sc_probability)
    model=ResidualFusionModel("discovery",("signed_sc",),(0.,),(1.,),1,"0"*64,"1"*64,0,0,
                              feature_transforms=("signed_fisher",))
    result=apply_residual_fusion_model(model,memory)
    assert (result.signed_sc_input_value<0).any()
    negative=result.signed_sc_input_value<0
    assert (result.loc[negative,"discovery_adjusted_probability"]<.5).all()
    with pytest.raises(MultimodalFusionError,match="outside"):
        apply_residual_fusion_model(replace(model,feature_transforms=("probability_logit",)),memory)


@pytest.mark.parametrize("binding_fault",[None,"run_id","estimand","spec","core"])
def test_current_runner_consumes_signed_adapter_without_starting_training(tmp_path,monkeypatch,binding_fault):
    import scripts.run_v32_current_multimodal_fusion as runner
    fixture(tmp_path)
    keys=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":[f"L{i}" for i in range(80)],"pathway_id":"P1"})
    keys.to_parquet(tmp_path/"candidates.parquet",index=False)
    adapter_audit=run(tmp_path)
    primary=keys.assign(**{PRIMARY_COLUMN:.5,"analysis_version":ANALYSIS_VERSION})
    primary.to_parquet(tmp_path/"primary.parquet",index=False)
    record=lambda p:{"path":str(p),"sha256":artifact_sha256(p)}
    folds=[]
    for fold in range(5):
        path=tmp_path/f"primary_fold{fold}.parquet"
        keys.assign(patient_fold_id=fold,held_out_proxy_label=fold%2).to_parquet(path,index=False)
        folds.append({"fold_id":fold,"artifact":record(path)})
    binding_path=tmp_path/"core_binding.json"
    binding_path.write_text(json.dumps({"status":"PASS","run_id":"G2-CURRENT-FIXTURE",
        "prediction":record(tmp_path/"primary.parquet"),
        "fold_prediction_sha256":[value["artifact"]["sha256"] for value in folds]}))
    manifest={"format":"CURRENT_MULTIMODAL_FUSION_MANIFEST_V1","core_run_id":"G2-CURRENT-FIXTURE",
        "primary_prediction":record(tmp_path/"primary.parquet"),"primary_folds":folds,
        "primary_binding_audit":record(binding_path),
        "experts":[{"spec":asdict(spec),"prediction":record(tmp_path/"output/SINGLE_CELL_SIGNED_FUSION.parquet"),
                    "binding_audit":record(tmp_path/"output/AUDIT.json"),"run_id":adapter_audit["run_id"],
                    "estimand":"DONOR_BALANCED_SIGNED_ASSOCIATION_STRENGTH"} for spec in SIGNED_SC_EXPERTS]}
    if binding_fault in {"run_id","estimand"}:manifest["experts"][0][binding_fault]="WRONG_SOURCE"
    if binding_fault=="spec":manifest["experts"][0]["spec"]["value_transform"]="probability_logit"
    if binding_fault=="core":
        adapter_audit.update(core_dependency="CURRENT_CORE_EMBEDDINGS",core_run_id="OLD_CORE")
        (tmp_path/"output/AUDIT.json").write_text(json.dumps(adapter_audit))
        for expert in manifest["experts"]:expert["binding_audit"]=record(tmp_path/"output/AUDIT.json")
    path=tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    def forbidden(*a,**kw):raise AssertionError("Preparation must not start a training optimizer")
    monkeypatch.setattr(runner,"train_pair_blocked_crossfit_fusion",forbidden)
    if binding_fault:
        with pytest.raises(ValueError):
            runner.main(["--manifest",str(path),"--expected-core-run-id","G2-CURRENT-FIXTURE",
                         "--output",str(tmp_path/"prepared")])
        assert not (tmp_path/"prepared/PREPARED.json").exists()
        return
    assert runner.main(["--manifest",str(path),"--expected-core-run-id","G2-CURRENT-FIXTURE",
                        "--output",str(tmp_path/"prepared")])==0
    receipt=json.loads((tmp_path/"prepared/PREPARED.json").read_text())
    assert receipt["status"]=="PREPARED_NOT_TRAINED"
    assert not receipt["training_started"]
    result=pd.read_parquet(tmp_path/"prepared/PRIVATE_FUSION_INPUT.parquet")
    assert result.loc[result.lncrna_id.eq("L1"),"single_cell_positive_probability"].iloc[0]==pytest.approx(.4)
    assert not (tmp_path/"prepared/SUCCESS.json").exists()


def test_crossfit_passes_declared_transforms_to_all_folds_without_optimizer(monkeypatch):
    import cc_hhgt.v32.multimodal_fusion as fusion
    frame=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":[f"L{i}" for i in range(20)],"pathway_id":"P1",
        "primary_probability":.5,"fusion_target":.5,"fusion_pair_fold":np.arange(20)%5,
        "signed_sc_probability":np.linspace(-.8,.8,20),"signed_sc_available":True})
    spec=ExpertSpec("signed_sc","rho","available",("discovery",),"donor",value_transform="signed_fisher")
    calls=[]
    def no_optimizer(train,validation,expert_ids,**kw):
        calls.append(kw["feature_transforms"])
        return ResidualFusionModel(kw["endpoint"],tuple(expert_ids),(0.,),(1.,),1,"0"*64,"1"*64,0,0,
                                   feature_transforms=kw["feature_transforms"])
    monkeypatch.setattr(fusion,"fit_residual_fusion_model",no_optimizer)
    result=fusion.train_pair_blocked_crossfit_fusion(frame,[spec],endpoint="discovery")
    assert len(calls)==6 and all(value==("signed_fisher",) for value in calls)
    assert len(result.oof_prediction)==20


def test_release_keeps_signed_native_measurement_names_without_optimizer(tmp_path,monkeypatch):
    import cc_hhgt.v32.multimodal_fusion as fusion
    frame=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":[f"L{i}" for i in range(20)],"pathway_id":"P1",
        "primary_probability":.5,"fusion_target":.5,"fusion_pair_fold":np.arange(20)%5,
        "signed_sc_probability":np.linspace(-.8,.8,20),"signed_sc_available":True})
    spec=ExpertSpec("signed_sc","rho","available",("discovery","confidence"),"donor",value_transform="signed_fisher")
    def no_optimizer(train,validation,expert_ids,**kw):
        return ResidualFusionModel(kw["endpoint"],tuple(expert_ids),(0.,),(1.,),1,"0"*64,"1"*64,0,0,
                                   feature_transforms=kw["feature_transforms"])
    monkeypatch.setattr(fusion,"fit_residual_fusion_model",no_optimizer)
    discovery=fusion.train_pair_blocked_crossfit_fusion(frame,[spec],endpoint="discovery")
    confidence=fusion.train_pair_blocked_crossfit_fusion(frame,[spec],endpoint="confidence")
    source=tmp_path/"fixture_source.txt";source.write_text("synthetic fixture; no fitted model")
    result=fusion.materialize_multimodal_fusion_release(tmp_path/"release",frame,discovery,confidence,
        source_declarations=[{"path":str(source),"sha256":artifact_sha256(source),
                              "source_role":"raw_source_data","generation":"raw_source"}],
        training_run_id="V32-FUSION-SYNTHETIC_FIXTURE")
    public_files=list((tmp_path/"release").glob("*.parquet"))
    public=[pd.read_parquet(p) for p in public_files if "signed_sc_native_value" in pd.read_parquet(p).columns]
    assert len(public)==1
    assert "signed_sc_native_probability" not in public[0]
    assert (public[0].signed_sc_native_value<0).any()
    assert public[0].signed_sc_value_transform.eq("signed_fisher").all()
