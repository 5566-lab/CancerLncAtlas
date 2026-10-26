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


def frozen_clinical_head(core,domain,endpoint_arrays,*,train_indices,validation_indices,n_bins,seed,**kwargs):
    head,initialization=build_clinical_head(core_features=core.shape[1],domain_features=domain.shape[1],n_bins=n_bins,seed=seed)
    with torch.no_grad(): next(head.parameters()).add_(.01)
    head.eval()
    return head,{**initialization.as_dict(),"best_validation_loss":.6,"synthetic_serialization_fixture":True,
                 "no_optimizer_actually_run":True},[{"epoch":0,"training_loss":.7,"validation_loss":.6}]


def clinical_fixture(root,current_sources):
    from test_v32_current_state_context import native_fixture
    from cc_hhgt.v32.patient_fold_authority import build_patient_fold_authority
    from scripts.run_v32_clinical_training import main as risk_main
    from scripts.run_v32_clinical_entity_training import main as entity_main
    root.mkdir(); core_path,selected_path,run=current_sources
    state_manifest=json.loads(native_fixture(root/"state_container",current_sources).read_text())
    state_lineage=Path(state_manifest["state_training_lineage"]["path"])
    state_inputs={r["role"]:Path(r["path"]) for r in json.loads(state_lineage.read_text())["input_artifacts"]}
    samples=pd.read_parquet(state_inputs["v32_patient_fold_manifest"])
    expression=pd.read_parquet(state_inputs["lncrna_expression"]).merge(samples[["sample_id","patient_id"]],on="sample_id",validate="many_to_one")
    rng=np.random.default_rng(381)
    patients=sorted(samples.patient_id.unique()); n=len(patients)
    x=expression.groupby("patient_id").logcpm.mean().reindex(patients).to_numpy()
    expr_root=root/"expression"; expr=(expr_root/"cancer_id=BRCA");expr.mkdir(parents=True)
    expression.to_parquet(expr/"part-0.parquet",index=False)
    act_root=root/"activity";act=(act_root/"cancer_id=BRCA");act.mkdir(parents=True)
    activity=samples[["cancer_id","sample_id","patient_id"]].copy()
    values=dict(zip(patients,.4*x+rng.normal(size=n)))
    activity["pathway_id"]="P1";activity["activity_score"]=activity.patient_id.map(values)
    activity.to_parquet(act/"part-0.parquet",index=False)
    build_patient_fold_authority(activity_root=act_root,output_root=root/"authority",expected_cancers=["BRCA"],expected_patients=n)
    folds_path=root/"authority/SAMPLE_PATIENT_FOLD_MAP.tsv";authority=root/"authority/PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    main=pd.DataFrame({"bcr_patient_barcode":patients,"type":"BRCA",
        "age_at_initial_pathologic_diagnosis":rng.integers(35,75,n),"gender":["MALE" if i%2 else "FEMALE" for i in range(n)],
        "ajcc_pathologic_tumor_stage":["Stage I" if i%3 else "Stage II" for i in range(n)]})
    events=(np.arange(n)%3!=0).astype(int); times=300.-25.*x+rng.uniform(0,45,n)
    for i,endpoint in enumerate(["OS","DSS","PFI","DFI"]):
        main[endpoint]=events;main[endpoint+".time"]=times+i*20.
    extra=main[["bcr_patient_barcode","type"]].copy();extra["PFS"]=events;extra["PFS.time"]=times+80.
    workbook=root/"outcomes.xlsx"
    with pd.ExcelWriter(workbook,engine="openpyxl") as writer:
        main.to_excel(writer,sheet_name="TCGA-CDR",index=False);extra.to_excel(writer,sheet_name="ExtraEndpoints",index=False)
    selected=json.loads(selected_path.read_text());score=json.loads(Path(selected["selected_score_binding"]["path"]).read_text())
    primary=json.loads(Path(score["sources"]["reference_exact_producer"]["producer_binding"]["path"]).read_text())
    candidate=Path(primary["input_artifacts"][0]["path"])
    risk_args=["--repo-root",str(Path(__file__).resolve().parents[1]),"--workbook",str(workbook),
        "--fold-manifest",str(folds_path),"--patient-fold-authority-receipt",str(authority),"--patient-authority-mode","current",
        "--expression-root",str(expr_root),"--activity-root",str(act_root),"--core-root",str(core_path.parent),
        "--output-root",str(root/"risk"),"--training-run-id","CLINICAL-SYNTHETIC","--expected-core-run-id",run,
        "--seed","17","--epochs","1","--patience","1","--n-bins","2","--min-patients","5","--min-events","2"]
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError("No fit")))
        patch.setattr("cc_hhgt.v32.clinical_training.train_private_clinical_head",frozen_clinical_head)
        assert risk_main(risk_args)==0
    entity_args=["--tcga-cdr",str(workbook),"--patient-folds",str(folds_path),"--patient-fold-authority-receipt",str(authority),
        "--patient-authority-mode","current","--candidates",str(candidate),"--lncrna-expression-root",str(expr_root),
        "--pathway-activity-root",str(act_root),"--state-measurements",str(state_inputs["tumor_state_long_measurement"]),
        "--v32-state-lineage",str(state_lineage),"--output-root",str(root/"entity"),"--training-run-id","ENTITY-SYNTHETIC",
        "--min-train-patients","12","--min-train-events","2","--min-validation-patients","4","--min-validation-events","1",
        "--min-test-patients","4","--min-test-events","1","--min-train-measurements","8"]
    assert entity_main(entity_args)==0
    write(root/"PRODUCER_COMMANDS.json",{"risk":risk_args,"entity":entity_args})
    manifest={"format":INPUT_FORMAT,"core_run_id":run,"patient_risk_run_id":"CLINICAL-SYNTHETIC","entity_run_id":"ENTITY-SYNTHETIC",
        "core_manifest":record(core_path),"selected_analysis":record(selected_path),"patient_risk_lineage":record(root/"risk/MODULE_LINEAGE.json"),
        "patient_risk_predictions":record(root/"risk/clinical_patient_risk.parquet"),"entity_lineage":record(root/"entity/ENTITY_CLINICAL_LINEAGE.json"),
        "entity_calculation_manifest":record(root/"entity/CALCULATION_MANIFEST.json"),"fdr_threshold":.05}
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
    assert binding["raw_patient_feature_projection_replayed"] and binding["entity_raw_patient_statistics_replayed"]
    risk=query.query(table="patient_risk_fold_coverage")["rows"]
    assert len(risk)==30 and all("subject_id" not in row and "patient_id" not in row for row in risk)
    assert len({r["last_training_time_bin_days"] for r in risk if r["clinical_endpoint"]=="OS"})>=2
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
        fp=Path(row["feature_artifact"]["path"]); features=pd.read_parquet(fp); features.loc[features.patient_fold_id.eq(0),checkpoint["core_columns"][0]]+=20
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
    if fault=="candidate":
        item=next(r for r in lineage["input_artifacts"] if r["artifact_kind"]=="annotation")
        item.update(record(Path(lineage["input_artifacts"][0]["path"])))
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


def test_missing_native_features_cannot_claim_raw_numeric_replay(tmp_path,current_sources):
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
    with pytest.raises(ValueError,match="requires actual patient features"):
        patient_risk_sources(selected_sources(path))


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
