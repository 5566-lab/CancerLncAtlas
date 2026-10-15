"""Native Clinical/selected-score consumption on synthetic data only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from test_v32_current_state_context import current_sources
from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_clinical_sources import (INPUT_FORMAT, selected_sources, patient_risk_sources, entity_sources)
from cc_hhgt.v32.current_clinical_context import (public_entities, selected_clinical_tables, patient_risk_coverage,
    materialize_current_clinical_context, audit_current_clinical_context, CurrentClinicalContextQuery)
from cc_hhgt.v32.clinical_training import (CLINICAL_ENDPOINTS, CLINICAL_CHECKPOINT_FORMAT,
    build_clinical_head, reshape_endpoint_logits, survival_risk_probability)
from cc_hhgt.v32.clinical_entity_training import (CALCULATION_FORMAT, ANALYSIS_VERSION, HISTORICAL_STATE_IDS,
    SUBJECT_SPECS, analyse_entity_matrix, summarise_oof_associations, json_sha256)
from cc_hhgt.v32.input_lineage import _canonical_json_sha256


def record(path): return {"path":str(path),"sha256":sha256(path)}


def write(path,value): path.write_text(json.dumps(value),encoding="utf-8"); return path


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("No optimizer or Clinical neural-head training")
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr("cc_hhgt.v32.clinical_training.train_private_clinical_head",forbidden)


def clinical_fixture(root,current_sources):
    from test_v32_clinical_entity_training import _toy_inputs
    root.mkdir(); rp=root/"risk"; rp.mkdir(); ep=root/"entity"; ep.mkdir()
    core_path,selected_path,run=current_sources; core=json.loads(core_path.read_text())
    selected=json.loads(selected_path.read_text()); score=json.loads(Path(selected["selected_score_binding"]["path"]).read_text())
    primary=json.loads(Path(score["sources"]["reference_exact_producer"]["producer_binding"]["path"]).read_text())
    candidate=Path(primary["input_artifacts"][0]["path"])
    patients,fold_ids,endpoints,covariates=_toy_inputs()
    endpoints.cancer_id="BRCA"; covariates.cancer_id="BRCA"
    raw=pd.DataFrame({"cancer_id":"BRCA","sample_id":[f"A{i}" for i in range(50)],"patient_id":patients,"patient_fold_id":fold_ids})
    folds_path=root/"folds.tsv"; raw.to_csv(folds_path,sep="\t",index=False)
    workbook=write(root/"outcomes.source.json",{"synthetic_raw_outcome_fixture":True})
    expression=root/"expression.parquet"; pd.DataFrame({"patient_id":patients,"logcpm":np.arange(50)}).to_parquet(expression,index=False)
    input_records=[{**record(workbook),"artifact_kind":"training_label"},
        {**record(folds_path),"artifact_kind":"split_manifest"},{**record(core_path),"artifact_kind":"v32_core_embedding_manifest"}]
    audit={"policy_version":"synthetic-native-protocol","audit_mode":"READ_ONLY","input_count":3,"status":"PASS",
        "artifacts":[{**record(p),"status":"PASS"} for p in (workbook,folds_path,expression)]}
    audit["lineage_sha256"]=_canonical_json_sha256(audit); write(rp/"INPUT_LINEAGE_AUDIT.json",audit)
    rng=np.random.default_rng(381); features=raw[["cancer_id","patient_id","patient_fold_id"]].copy()
    for i in range(4): features[f"core_{i}"]=rng.normal(size=50).astype(np.float32)
    for i in range(2): features[f"domain_{i}"]=rng.normal(size=50).astype(np.float32)
    public=[]; tasks=[]
    for fold in range(5):
        fold_root=rp/f"patient_fold={fold}"; fold_root.mkdir()
        head,init=build_clinical_head(core_features=4,domain_features=2,n_bins=2,seed=17+fold)
        with torch.no_grad(): next(head.parameters()).add_(.01)
        head.eval(); test=features.loc[features.patient_fold_id.eq(fold)]
        with torch.inference_mode():
            logits=reshape_endpoint_logits(head(torch.tensor(test[[f"core_{i}" for i in range(4)]].to_numpy()),
                torch.tensor(test[[f"domain_{i}" for i in range(2)]].to_numpy())),2).numpy()
        rows=[]; edges={e:[10.+fold,20.+fold] for e in CLINICAL_ENDPOINTS if e!="DFS"}
        for i,endpoint in enumerate(CLINICAL_ENDPOINTS):
            probabilities=survival_risk_probability(logits[:,i])
            for j,patient in enumerate(test.patient_id):
                trained=endpoint in edges
                rows.append({"cancer_id":"BRCA","subject_type":"patient","subject_id":patient,"clinical_endpoint":endpoint,
                    "clinical_relevance_probability":probabilities[j] if trained else np.nan,"availability":trained,
                    "failure_reason":"" if trained else "NO_TRAINABLE_DISTINCT_ENDPOINT_SOURCE","patient_fold_id":fold,
                    "analysis_version":"CancerLncAtlas_V3.2_FULL_MULTITASK","training_run_id":"CLINICAL-SYNTHETIC","core_run_id":run,
                    "model_artifact_id":f"clinical-fold-{fold}","changes_primary_ranking":False})
        frame=pd.DataFrame(rows); prediction=fold_root/"clinical_patient_risk.parquet"; frame.to_parquet(prediction,index=False); public.append(frame)
        feature=fold_root/"patient_features.fold_local.parquet"; features.to_parquet(feature,index=False)
        parent=core["folds"][str(fold)]
        checkpoint={"checkpoint_format":CLINICAL_CHECKPOINT_FORMAT,"analysis_version":"CancerLncAtlas_V3.2_FULL_MULTITASK",
            "training_run_id":"CLINICAL-SYNTHETIC","core_run_id":run,"core_manifest_sha256":sha256(core_path),"module_id":"clinical",
            "patient_fold":fold,"seed":17+fold,"initialization":{**init.as_dict(),"best_validation_loss":.6},
            "source_checkpoint_sha256":None,"parent_v32_core_checkpoint_sha256":parent["checkpoint_sha256"],
            "parent_v32_core_parameter_sha256":parent["core_parameter_sha256"],"core_parameters_before_sha256":parent["core_parameter_sha256"],
            "core_parameters_after_sha256":parent["core_parameter_sha256"],"core_parameters_frozen":True,
            "old_checkpoint_loaded":False,"old_predictions_used_as_features":False,"endpoint_time_bins":edges,
            "core_columns":[f"core_{i}" for i in range(4)],"domain_columns":[f"domain_{i}" for i in range(2)],
            "model_state":head.state_dict(),"synthetic_serialization_fixture":True,"no_optimizer_actually_run":True}
        cp=fold_root/"best_model_state.pt"; torch.save(checkpoint,cp)
        tasks.append({"status":"SUCCESS","training_run_id":"CLINICAL-SYNTHETIC","core_run_id":run,"core_manifest_sha256":sha256(core_path),
            "input_lineage_sha256":audit["lineage_sha256"],"training_config_sha256":"f"*64,"public_prediction_sha256":sha256(prediction),
            "module_id":"clinical","patient_fold":fold,"seed":17+fold,"n_test":len(test),"n_train":30,"n_validation":10,
            "trained_endpoints":sorted(edges),"unavailable_endpoints":["DFS"],"checkpoint_path":str(cp),"checkpoint_sha256":sha256(cp),
            "parent_v32_core_checkpoint_sha256":parent["checkpoint_sha256"],"parent_v32_core_parameter_sha256":parent["core_parameter_sha256"],
            "private_head_trained_from_scratch":True,"source_checkpoint_sha256":None,"core_parameters_frozen":True,
            "core_parameters_unchanged":True,"feature_artifact":record(feature)})
    all_public=pd.concat(public,ignore_index=True); prediction=rp/"clinical_patient_risk.parquet"; all_public.to_parquet(prediction,index=False)
    risk_lineage={"module_id":"clinical","analysis_version":"CancerLncAtlas_V3.2_FULL_MULTITASK","training_run_id":"CLINICAL-SYNTHETIC",
        "core_run_id":run,"core_manifest_sha256":sha256(core_path),"training_status":"SUCCESS","folds":5,
        "private_head_trained_from_scratch":True,"core_parameters_frozen":True,"old_checkpoint_loaded":False,
        "old_predictions_used_as_features":False,"old_rankings_used_as_outputs":False,"input_artifacts":input_records,
        "input_manifest_sha256":audit["lineage_sha256"],"checkpoint_manifest_sha256":json_sha256(tasks),"task_lineage":tasks,
        "public_prediction_rows":len(all_public),"public_prediction_artifact":record(prediction),"labels_in_public_output":False}
    risk_path=write(rp/"MODULE_LINEAGE.json",risk_lineage)
    summaries=[]; parts=[]
    for kind in SUBJECT_SPECS:
        ids=["LNC:Z"] if kind=="lncRNA" else ["P1"] if kind=="exact_pathway" else list(HISTORICAL_STATE_IDS)
        fold_records=analyse_entity_matrix(cancer_id="BRCA",subject_type=kind,patient_ids=patients,entity_ids=ids,
            values=rng.normal(size=(50,len(ids))),fold_ids=fold_ids,endpoint_frame=endpoints,covariate_frame=covariates,
            min_train_patients=20,min_train_events=5,min_validation_patients=5,min_validation_events=2,
            min_test_patients=5,min_test_events=2,min_train_measurements=10)
        summary=summarise_oof_associations(fold_records,ids,cancer_id="BRCA",subject_type=kind,min_folds_available=3)
        fp=ep/(kind+"_folds.parquet"); fold_records.to_parquet(fp,index=False)
        sp=ep/(kind+"_summary.parquet"); summary.to_parquet(sp,index=False); summaries.append(summary)
        parts.append({"cancer_id":"BRCA","subject_type":kind,"status":"SUCCESS","fold_rows":len(fold_records),"summary_rows":len(summary),
            "fold_records_path":str(fp),"fold_records_sha256":sha256(fp),"summary_part_path":str(sp),"summary_part_sha256":sha256(sp)})
    native=pd.concat(summaries,ignore_index=True); summary_path=ep/"entity_clinical_associations.parquet"; native.to_parquet(summary_path,index=False)
    config={"format":CALCULATION_FORMAT,"subject_types":list(SUBJECT_SPECS),"cancers":["BRCA"],"max_entities":None,"minimums":{"folds_available":3}}
    calc={"format":CALCULATION_FORMAT,"analysis_version":ANALYSIS_VERSION,"training_run_id":"ENTITY-SYNTHETIC","folds":5,
        "calculation_config":config,"split_policy":{str(f):{"test":[f],"validation":[(f+1)%5],"train":[i for i in range(5) if i not in {f,(f+1)%5}]} for f in range(5)},
        "method":"TRAIN_LOCAL_COVARIATE_RESIDUALIZED_BRESLOW_COX_NULL_SCORE_ONE_STEP","partitions":parts}
    calculation=write(ep/"CALCULATION_MANIFEST.json",calc)
    artifacts=[{**record(p),"artifact_kind":kind} for kind,p in [("raw_data",workbook),("split_manifest",folds_path),("annotation",candidate),("standardized_input",expression)]]
    lineage={"format":CALCULATION_FORMAT,"analysis_version":ANALYSIS_VERSION,"training_run_id":"ENTITY-SYNTHETIC","training_status":"SUCCESS","folds":5,
        "full_entity_universe":True,"all_candidate_entities_enumerated":True,"subject_types":list(SUBJECT_SPECS),
        "calculation_manifest_sha256":sha256(calculation),"calculation_config":config,"config_sha256":json_sha256(config),
        "input_artifacts":artifacts,"input_manifest_sha256":json_sha256(artifacts),"summary_path":str(summary_path),"summary_sha256":sha256(summary_path),
        "summary_rows":len(native),"expected_summary_rows_from_candidate_universe":len(native),"fresh_statistical_calculation":True,
        "old_checkpoint_loaded":False,"old_predictions_used_as_features":False,"old_rankings_used_as_outputs":False,"changes_primary_ranking":False}
    entity_path=write(ep/"ENTITY_CLINICAL_LINEAGE.json",lineage)
    manifest={"format":INPUT_FORMAT,"core_run_id":run,"patient_risk_run_id":"CLINICAL-SYNTHETIC","entity_run_id":"ENTITY-SYNTHETIC",
        "core_manifest":record(core_path),"selected_analysis":record(selected_path),"patient_risk_lineage":record(risk_path),
        "patient_risk_predictions":record(prediction),"entity_lineage":record(entity_path),"entity_calculation_manifest":record(calculation),"fdr_threshold":.05}
    return write(root/"INPUT.json",manifest)


def test_actual_native_readers_cli_selected_consumers_queries_and_replay(tmp_path,current_sources):
    from scripts.materialize_v32_current_clinical_context import main
    path=clinical_fixture(tmp_path/"inputs",current_sources); output=tmp_path/"out"
    assert main(["--manifest",str(path),"--output",str(output)])==0
    bp=output/"CURRENT_CLINICAL_CONTEXT_BINDING.json"; binding=json.loads(bp.read_text())
    query=CurrentClinicalContextQuery(bp,expected_sha=sha256(bp),selected_analysis_sha=binding["selected_analysis"]["sha256"])
    rows=query.query()["rows"]; assert len(rows)==6; json.dumps(rows,allow_nan=False)
    dfs=next(r for r in rows if r["clinical_endpoint"]=="DFS")
    assert not dfs["availability"] and dfs["supported_risk_weight_fraction"] is None
    assert binding["patient_risk_frozen_replay_folds"]==list(range(5))
    assert not binding["joint_geneset_survival_effect_estimated"] and not binding["training_started"]
    assert not binding["raw_patient_feature_projection_replayed"]
    risk=query.query(table="patient_risk_fold_coverage")["rows"]
    assert len(risk)==30 and all("subject_id" not in row and "patient_id" not in row for row in risk)
    assert len({r["last_training_time_bin_days"] for r in risk if r["clinical_endpoint"]=="OS"})==5
    entities=query.query(table="native_entity_clinical",limit=100)["rows"]
    assert len(entities)==54 and all("clinical_relevance_probability" not in r for r in entities)
    assert len(query.query(table="selected_pathway_clinical")["rows"])==6
    assert not query.query(filters={"clinical_endpoint":"' OR 1=1 --"})["rows"]
    with pytest.raises(ValueError): query.query(filters={"private_time_days":"10"})
    with pytest.raises(ValueError): CurrentClinicalContextQuery(bp,expected_sha=sha256(bp),selected_analysis_sha="e"*64)
    public=Path(query.download("selected_geneset_clinical_support")["path"]); frame=pd.read_parquet(public)
    frame.loc[0,"supported_risk_weight_fraction"]=.98; frame.to_parquet(public,index=False)
    binding["outputs"]["selected_geneset_clinical_support"]["sha256"]=sha256(public); write(bp,binding)
    with pytest.raises(ValueError,match="source replay differs"): audit_current_clinical_context(bp,expected_sha=sha256(bp))


@pytest.mark.parametrize("fault",["wrong_core","checkpoint_run","nonboolean","patient_fold","private_label","changed_prediction","changed_feature"])
def test_patient_risk_sources_reject_changed_identity_or_predictions(tmp_path,current_sources,fault):
    path=clinical_fixture(tmp_path/"inputs",current_sources); manifest=json.loads(path.read_text())
    lp=Path(manifest["patient_risk_lineage"]["path"]); lineage=json.loads(lp.read_text()); row=lineage["task_lineage"][0]
    cp=Path(row["checkpoint_path"]); checkpoint=torch.load(cp,weights_only=True)
    if fault=="wrong_core": checkpoint["parent_v32_core_checkpoint_sha256"]="e"*64
    if fault=="checkpoint_run": checkpoint["training_run_id"]="OTHER"
    if fault in {"wrong_core","checkpoint_run"}: torch.save(checkpoint,cp); row["checkpoint_sha256"]=sha256(cp)
    pp=cp.parent/"clinical_patient_risk.parquet"; public=pd.read_parquet(pp)
    if fault=="nonboolean": public.availability=public.availability.astype(str)
    if fault=="patient_fold": public.loc[0,"subject_id"]="TCGA-AA-0001"
    if fault=="private_label": public["time_days"]=20.
    if fault=="changed_prediction": public.loc[0,"clinical_relevance_probability"]=.99
    public.to_parquet(pp,index=False); row["public_prediction_sha256"]=sha256(pp)
    if fault=="changed_feature":
        fp=Path(row["feature_artifact"]["path"]); features=pd.read_parquet(fp); features.loc[features.patient_fold_id.eq(0),"core_0"]+=20
        features.to_parquet(fp,index=False); row["feature_artifact"]=record(fp)
    lineage["checkpoint_manifest_sha256"]=json_sha256(lineage["task_lineage"]); write(lp,lineage)
    manifest["patient_risk_lineage"]=record(lp); write(path,manifest)
    with pytest.raises(ValueError): patient_risk_sources(selected_sources(path))


@pytest.mark.parametrize("fault",["missing_fold","invalid_se","wrong_roles","patient_counts","dfs","summary","threshold","candidate"])
def test_entity_source_recomputation_requires_full_native_scope(tmp_path,current_sources,fault):
    path=clinical_fixture(tmp_path/"inputs",current_sources); context=selected_sources(path); risk=patient_risk_sources(context)
    manifest=json.loads(path.read_text()); lp=Path(manifest["entity_lineage"]["path"]); lineage=json.loads(lp.read_text())
    cp=Path(manifest["entity_calculation_manifest"]["path"]); calc=json.loads(cp.read_text()); part=calc["partitions"][0]
    fp=Path(part["fold_records_path"]); folds=pd.read_parquet(fp)
    idx=folds.loc[folds.clinical_endpoint.eq("OS")].index[0]
    if fault=="missing_fold": folds=folds.drop(idx); part["fold_rows"]-=1
    if fault=="invalid_se": folds.loc[idx,"test_standard_error"]=-1
    if fault=="wrong_roles": folds.loc[idx,"validation_fold_id"]=folds.loc[idx,"patient_fold_id"]
    if fault=="patient_counts": folds.loc[idx,"n_test_patients"]=1000
    if fault=="dfs": folds.loc[folds.clinical_endpoint.eq("DFS"),"test_beta"]=0.
    if fault=="summary":
        sp=Path(part["summary_part_path"]); data=pd.read_parquet(sp); data.loc[data.availability,"fdr"]=.001
        data.to_parquet(sp,index=False); part["summary_part_sha256"]=sha256(sp)
    if fault=="threshold": lineage["calculation_config"]["minimums"]["folds_available"]=1
    if fault=="candidate": lineage["input_artifacts"][2]=lineage["input_artifacts"][2]|record(Path(lineage["input_artifacts"][0]["path"]))
    folds.to_parquet(fp,index=False); part["fold_records_sha256"]=sha256(fp)
    write(cp,calc); lineage["calculation_manifest_sha256"]=sha256(cp)
    lineage["input_manifest_sha256"]=json_sha256(lineage["input_artifacts"]); write(lp,lineage)
    manifest["entity_lineage"]=record(lp); manifest["entity_calculation_manifest"]=record(cp); write(path,manifest)
    with pytest.raises((ValueError,AssertionError)): entity_sources(selected_sources(path),risk)


def test_selected_weights_change_support_without_inventing_joint_hr_or_retesting_fdr():
    members=pd.DataFrame({"geneset_id":["G"]*3,"geneset_rank":[1,2,3],"cancer_id":["C"]*3,"lncrna_id":["L1","L2","MISSING"],
        "pathway_id":["P"]*3,"selected_score":[.9,.1,.5],"association_direction":["positive"]*3})
    coverage=pd.DataFrame({"cancer_id":["C","C"],"pathway_id":["P","NOSET"],"association_direction":["positive","negative"],
        "geneset_available":[True,False],"unavailable_reason":[None,"few_members"],"exported_members":[3,0]})
    entities=pd.DataFrame({"cancer_id":["C"]*3,"subject_type":["lncRNA","lncRNA","exact_pathway"],"subject_id":["L1","L2","P"],
        "clinical_endpoint":["OS"]*3,"availability":[True]*3,"failure_reason":[""]*3,"clinical_effect_direction":["risk","protective","risk"],
        "fdr":[.01,.02,.03],"hazard_ratio":[2.,.5,1.5]})
    first=selected_clinical_tables(members,coverage,entities)
    os=first["selected_geneset_clinical_support"].query("clinical_endpoint=='OS'").iloc[0]
    assert os.supported_risk_weight_fraction==pytest.approx(.9) and os.selected_weight_coverage==pytest.approx(1/1.5)
    assert len(first["selected_clinical_coverage"])==12
    assert "hazard_ratio" not in first["selected_geneset_clinical_support"]
    assert first["selected_member_clinical"].query("clinical_endpoint=='OS'").fdr.dropna().tolist()==[.01,.02]
    members.loc[:1,"selected_score"]=[.1,.9]
    second=selected_clinical_tables(members,coverage,entities)["selected_geneset_clinical_support"].query("clinical_endpoint=='OS'").iloc[0]
    assert second.supported_protective_weight_fraction==pytest.approx(.9)
    members.selected_score=0.
    zero=selected_clinical_tables(members,coverage,entities)["selected_geneset_clinical_support"]
    assert not zero.availability.any() and zero.supported_risk_weight_fraction.isna().all()
    empty=selected_clinical_tables(members.iloc[:0],coverage.assign(geneset_available=False,exported_members=0),entities)
    assert empty["selected_geneset_clinical_support"].empty and len(empty["selected_clinical_coverage"])==12


def test_older_native_outputs_keep_explicit_nonreplay_and_hash_bound_config(tmp_path,current_sources):
    path=clinical_fixture(tmp_path/"inputs",current_sources); manifest=json.loads(path.read_text())
    rp=Path(manifest["patient_risk_lineage"]["path"]); risk_lineage=json.loads(rp.read_text())
    for task in risk_lineage["task_lineage"]: task.pop("feature_artifact")
    risk_lineage.pop("public_prediction_artifact")
    risk_lineage["checkpoint_manifest_sha256"]=json_sha256(risk_lineage["task_lineage"]); write(rp,risk_lineage)
    lp=Path(manifest["entity_lineage"]["path"]); lineage=json.loads(lp.read_text())
    config=lineage.pop("calculation_config"); manifest["entity_config"]=record(write(path.parent/"CONFIG.json",config))
    cp=Path(manifest["entity_calculation_manifest"]["path"]); calculation=json.loads(cp.read_text()); calculation.pop("calculation_config")
    write(cp,calculation); lineage["calculation_manifest_sha256"]=sha256(cp); write(lp,lineage)
    manifest["patient_risk_lineage"]=record(rp); manifest["entity_lineage"]=record(lp); manifest["entity_calculation_manifest"]=record(cp); write(path,manifest)
    context=selected_sources(path); risk=patient_risk_sources(context)
    assert risk["replayed_folds"]==[]
    assert len(entity_sources(context,risk))==54


def test_patient_risk_coverage_keeps_unrepresented_patients_and_observed_zero():
    patients=pd.DataFrame({"cancer_id":["C"]*2,"patient_id":["A","B"],"patient_fold_id":[0,0]})
    public=pd.DataFrame({"cancer_id":["C"],"subject_id":["A"],"patient_fold_id":[0],"clinical_endpoint":["OS"],
        "clinical_relevance_probability":[0.],"availability":[True],"failure_reason":[""]})
    bins=pd.DataFrame({"patient_fold_id":[0]*6,"clinical_endpoint":list(CLINICAL_ENDPOINTS),"trained":[True]*5+[False],
        "time_bins_days_json":["[10,20]"]*6,"last_training_time_bin_days":[20.]*6})
    data=patient_risk_coverage(dict(patients=patients,public=public,bins=bins))
    row=data.loc[data.clinical_endpoint.eq("OS")].iloc[0]
    assert row.risk_median==0 and row.patient_coverage==.5 and row.available_patients==1
    assert json.loads(row.missing_reasons_json)=={"NO_NATIVE_RISK_FOR_AUTHORIZED_PATIENT":1}


def test_unavailable_native_risk_must_be_null_not_an_unparseable_string():
    from cc_hhgt.v32.current_clinical_sources import _validate_risk
    patients=pd.DataFrame({"cancer_id":["C"],"patient_id":["P"],"patient_fold_id":[0]})
    data=pd.DataFrame({"cancer_id":["C"]*6,"subject_type":["patient"]*6,"subject_id":["P"]*6,
        "clinical_endpoint":list(CLINICAL_ENDPOINTS),"clinical_relevance_probability":[.2]*5+["not a measurement"],
        "availability":[True]*5+[False],"failure_reason":[""]*5+["NO_DISTINCT_DFS_SOURCE"],"patient_fold_id":[0]*6,
        "analysis_version":["V3.2"]*6,"training_run_id":["R"]*6,"core_run_id":["C"]*6,
        "model_artifact_id":["clinical-fold-0"]*6,"changes_primary_ranking":[False]*6})
    with pytest.raises(ValueError,match="invalid probabilities"):
        _validate_risk(data,0,{"patient_risk_run_id":"R","core_run_id":"C"},patients,list(CLINICAL_ENDPOINTS[:-1]))


def test_materializer_enforces_149_without_writing(tmp_path,monkeypatch):
    monkeypatch.setattr("socket.gethostname",lambda:"local")
    with pytest.raises(RuntimeError,match="149"): materialize_current_clinical_context(tmp_path/"missing",tmp_path/"out")
    assert not (tmp_path/"out").exists()
