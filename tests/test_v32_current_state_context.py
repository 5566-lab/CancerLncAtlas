"""Synthetic native-format State integration; no optimizers or real cohort data."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.contracts import canonical_json_sha256
from cc_hhgt.v32.current_state_context import (INPUT_FORMAT, StateSetConfig, state_sources,
    native_aggregate, selected_state_tables, native_state_sets, materialize_current_state_context,
    audit_current_state_context, CurrentStateContextQuery)
from cc_hhgt.v32.state_training import (StateTrainingConfig, STATE_TRAINING_FORMAT, STATE_CHECKPOINT_FORMAT,
    HISTORICAL_STATE_IDS, aggregate_oof_predictions, load_v32_core_embeddings)
from cc_hhgt.v32.integrated_model import build_private_auxiliary_head, module_state_sha256


def record(path): return {"path":str(path),"sha256":sha256(path)}


def write(path,value): path.write_text(json.dumps(value),encoding="utf-8"); return path


def no_training(*args,**kwargs): raise AssertionError("Training is forbidden in State context verification")


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,no_training)
    import cc_hhgt.v32.state_training as state
    monkeypatch.setattr(state,"train_state_models",no_training)
    monkeypatch.setattr(state,"train_fold_private_head",no_training)


@pytest.fixture(scope="module")
def current_sources(tmp_path_factory):
    from test_v32_current_core_prediction import current_fixture
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.selected_genesets import materialize_selected_analysis
    from scripts.export_v32_current_core_embeddings import main as export_main
    root = tmp_path_factory.mktemp("current_state_parent")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,no_training)
        path = current_fixture(root/"inputs"); manifest = json.loads(path.read_text())
        run = manifest["core_run_id"]
        result = materialize_current_predictions(path,root/"predictions",expected_run_id=run)
        primary = root/"predictions"/"CURRENT_PREDICTION_BINDING.json"
        manifest["format"] = "CURRENT_CORE_EXPORT_INPUT_V1"
        export_input = write(root/"EXPORT_INPUT.json",manifest)
        assert export_main(["--manifest",str(export_input),"--expected-core-run-id",run,"--output",str(root/"core")])==0
        spec = {"model_id":"current","run_id":run,"producer_kind":"current_core_release",
                "producer_binding":record(primary),"score_column":"association_membership_probability"}
        selected = {"format":"CURRENT_SELECTED_ANALYSIS_INPUT_V1","selected_model":spec,"reference_exact_model":spec,
            "selection":{"kind":"EXPLICIT_SOURCE_REGISTRATION","rationale":"Synthetic serialization fixture, no winner claim"},
            "geneset":{"min_members":1,"max_members":1,"score_min":.1},
            "subtype":{"score_column":"selected_score","min_members":1,"min_cancers":2,"bootstrap_replicates":2}}
        selected_path = write(root/"SELECTED_INPUT.json",selected)
        result = materialize_selected_analysis(selected_path,root/"selected")
    return root/"core"/"CORE_EMBEDDING_MANIFEST.json",Path(result["binding_path"]),run


# Preserve the actual non-optimizer driver for the serialization fixture. Its
# sole fitting callback is replaced locally; the global optimizer guard stays on.
from cc_hhgt.v32.state_training import train_state_models as native_state_driver


def frozen_state_fold(train,validation,core,*,fold,config):
    """Explicit completed-head fixture; no fitting or fabricated input counts."""
    import cc_hhgt.v32.state_training as native
    _,design=native.prepare_state_head_inputs(train,validation,core,fold=fold)
    head,init=native.build_state_private_head(fold,design["core_features"],design["domain_features"],config)
    with torch.no_grad(): next(head.parameters()).add_(.01)
    lineage={"patient_fold":fold,"seed":config.seed+fold,"initialization":init.as_dict(),
        "final_parameter_sha256":module_state_sha256(head),**design,
        "history":[{"epoch":0,"training_loss":.7,"validation_loss":.6}],"epochs_completed":1,
        "synthetic_serialization_fixture":True,"no_optimizer_actually_run":True}
    return head,lineage,design["max_train_pairs"]


def native_fixture(root,current_sources,*,input_style="files"):
    import tempfile
    import cc_hhgt.v32.state_training as native
    root.mkdir(); core_path,selected_path,run = current_sources
    # Measurement loaders prohibit historical-result path tokens, including
    # those in pytest case names. Keep raw fixtures in an explicitly raw folder.
    raw = Path(tempfile.mkdtemp(prefix="native_state_",dir=core_path.parent.parent))
    cfg = StateTrainingConfig(hidden_features=4,min_pairs=2,min_available_folds=3,max_epochs=1,device="cpu")
    patient=[]; expression=[]; state=[]
    for fold in range(5):
        for j,x in enumerate([-3.,-2.,-1.,0.,0.,1.,2.,3.]):
            identity=f"P{fold}_{j}"
            for aliquot in range(2 if (fold,j)==(0,0) else 1):
                sample=f"{identity}_A{aliquot}"
                delta=(-.5 if aliquot==0 else .5) if (fold,j)==(0,0) else 0.
                patient.append(("BRCA",sample,identity,fold))
                expression.append(("BRCA",sample,"LNC:Z",x+delta))
                for i,sid in enumerate(HISTORICAL_STATE_IDS):
                    if i==2: continue
                    value=x*x if i==1 else -x if i==3 else x
                    state.append((sample,"BRCA",identity,sid,value+delta))
    for name,frame in {
        "patients":pd.DataFrame(patient,columns=["cancer_id","sample_id","patient_id","patient_fold_id"]),
        "expression":pd.DataFrame(expression,columns=["cancer_id","sample_id","lncrna_id","logcpm"]),
        "state_measurement":pd.DataFrame(state,columns=["sample_id","cancer_id","patient_id","state_id","state_value"])
    }.items(): frame.to_parquet(raw/(name+".parquet"),index=False)

    expression_path=raw/"expression.parquet"; patient_path=raw/"patients.parquet"
    if input_style=="dataset":
        table=pd.read_parquet(expression_path); expression_path=raw/"expression_dataset"
        for part in range(2):
            folder=expression_path/"cancer_id=BRCA"; folder.mkdir(parents=True,exist_ok=True)
            table.iloc[part::2].to_parquet(folder/f"part{part}.parquet",index=False)
    if input_style=="csv":
        patient_path=raw/"patients.csv"; pd.read_parquet(raw/"patients.parquet").to_csv(patient_path,index=False)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"): patch.setattr(torch.optim,name,no_training)
        patch.setattr(native,"train_state_models",native_state_driver)
        patch.setattr(native,"train_fold_private_head",frozen_state_fold)
        result=native.run_state_training_from_paths(state_measurements_path=raw/"state_measurement.parquet",
            fold_manifest_path=patient_path,lncrna_expression_path=expression_path,
            core_embedding_manifest_path=core_path,output_root=root,repo_root=root,run_id="v32-state-synthetic",config=cfg)
    manifest={"format":INPUT_FORMAT,"core_run_id":run,"state_run_id":"v32-state-synthetic",
        "core_manifest":record(core_path),"selected_analysis":record(selected_path),
        "state_training_lineage":record(Path(result["lineage"])),
        "state_sets":{"minimum_set_members":1,"pan_cancer_min_cancers":1}}
    return write(root/"INPUT.json",manifest)


def test_existing_cli_current_native_route_profiles_neutral_missing_and_queries(tmp_path,current_sources):
    from scripts.materialize_v32_state_gene_sets import main
    manifest_path = native_fixture(tmp_path/"inputs",current_sources); output = tmp_path/"out"
    assert main(["--current-manifest",str(manifest_path),"--output",str(output)])==0
    binding_path = output/"CURRENT_STATE_CONTEXT_BINDING.json"; binding = json.loads(binding_path.read_text())
    query = CurrentStateContextQuery(binding_path,expected_sha=sha256(binding_path),selected_analysis_sha=binding["selected_analysis"]["sha256"])
    rows = query.query()["rows"]; json.dumps(rows,allow_nan=False)
    assert len(rows)==7
    rna = next(r for r in rows if r["state_id"]==HISTORICAL_STATE_IDS[0])
    oof=pd.read_parquet(tmp_path/"inputs/v32_state_oof_predictions.parquet")
    measured=oof.loc[oof.state_id.eq(HISTORICAL_STATE_IDS[0])]
    assert rna["mean_state_membership_probability"]==pytest.approx(measured.state_membership_probability.mean())
    assert rna["mean_held_out_state_effect"]==pytest.approx(1.)
    assert rna["selected_weight_coverage"]==1
    zero = next(r for r in rows if r["state_id"]==HISTORICAL_STATE_IDS[1])
    assert zero["descriptive_effect_direction"]=="neutral" and zero["mean_held_out_state_effect"]==0
    absent = next(r for r in rows if r["state_id"]==HISTORICAL_STATE_IDS[2])
    assert absent["mean_held_out_state_effect"] is None and not absent["availability"]
    assert binding["private_head_predictions_replayed"] and binding["raw_measurement_statistics_replayed"]
    assert binding["native_numeric_replay"]["folds"][0]["alignment_counts"]["analysis_patients"]==40
    assert not binding["training_started"] and not binding["context_changes_ranking"]
    gmt = Path(query.download("gmt")["path"]).read_text()
    assert HISTORICAL_STATE_IDS[1] not in gmt and "Z\n" in gmt
    assert not query.query(filters={"state_id":"' OR 1=1 --"})["rows"]
    with pytest.raises(ValueError): query.query(filters={"state_id;DROP TABLE oof":"x"})
    with pytest.raises(ValueError): CurrentStateContextQuery(binding_path,expected_sha=sha256(binding_path),selected_analysis_sha="e"*64)
    # Rehashing modified profiles cannot bypass source recomputation.
    path = Path(binding["outputs"]["selected_geneset_state_profiles"]["path"])
    frame = pd.read_parquet(path); frame.loc[0,"mean_state_membership_probability"] = .99; frame.to_parquet(path,index=False)
    binding["outputs"]["selected_geneset_state_profiles"]["sha256"] = sha256(path); write(binding_path,binding)
    with pytest.raises(ValueError,match="source replay differs"):
        audit_current_state_context(binding_path,expected_sha=sha256(binding_path))


@pytest.mark.parametrize("fault",["state_run","fold_parent","frozen","checkpoint_run","checkpoint_tensor","missing_input"])
def test_native_source_identity_and_actual_tensors_are_required(tmp_path,current_sources,fault):
    path = native_fixture(tmp_path/"inputs",current_sources); manifest = json.loads(path.read_text())
    lp = Path(manifest["state_training_lineage"]["path"]); lineage = json.loads(lp.read_text())
    if fault=="state_run": manifest["state_run_id"]="different"
    if fault=="fold_parent": lineage["fold_lineage"]["0"]["v32_core_checkpoint_sha256"]="e"*64
    if fault=="frozen": lineage["fold_lineage"]["0"]["core_parameters_after_sha256"]="e"*64
    if fault.startswith("checkpoint_"):
        cp = Path(lineage["new_private_head_checkpoints"]["0"]["path"]); checkpoint = torch.load(cp,weights_only=True)
        if fault=="checkpoint_run": checkpoint["run_id"]="different"
        else: next(iter(checkpoint["model_state"].values())).add_(1)
        torch.save(checkpoint,cp); lineage["new_private_head_checkpoints"]["0"]=record(cp)
    if fault=="missing_input": Path(lineage["input_artifacts"][-1]["path"]).unlink()
    lineage.pop("lineage_sha256"); lineage["lineage_sha256"]=canonical_json_sha256(lineage); write(lp,lineage)
    manifest["state_training_lineage"]=record(lp); write(path,manifest)
    with pytest.raises((ValueError,FileNotFoundError)): state_sources(path)


@pytest.mark.parametrize("fault",["partial_folds","string_bool","bad_effect","bad_direction","sentinel","aggregate"])
def test_native_five_fold_recomputation_and_null_semantics(tmp_path,current_sources,fault):
    path = native_fixture(tmp_path/"inputs",current_sources); context = state_sources(path)
    oof_path = context["native_paths"]["oof_predictions"]; data = pd.read_parquet(oof_path)
    if fault=="partial_folds": data = data.iloc[1:]; context["lineage"]["outputs"]["oof_predictions"]["rows"]-=1
    if fault=="string_bool": data.availability=data.availability.astype(str)
    if fault=="bad_effect": data.loc[0,"state_effect"]=2.
    if fault=="bad_direction": data.loc[data.state_id.eq(HISTORICAL_STATE_IDS[0]),"association_direction"]="negative"
    if fault=="sentinel": data.loc[~data.availability,"state_membership_probability"]=0.
    if fault=="aggregate":
        release_path = context["native_paths"]["lncrna_state_release"]; release = pd.read_parquet(release_path)
        release.loc[0,"state_membership_probability"]=.9; release.to_parquet(release_path,index=False)
    data.to_parquet(oof_path,index=False)
    with pytest.raises(ValueError): native_aggregate(context)


def test_same_run_id_cannot_substitute_a_different_selected_core_checkpoint(tmp_path,current_sources):
    import shutil
    path = native_fixture(tmp_path/"inputs",current_sources); manifest = json.loads(path.read_text())
    source = Path(manifest["core_manifest"]["path"])
    copied = tmp_path/"changed_core"; shutil.copytree(source.parent,copied)
    core = json.loads(source.read_text())
    first = core["folds"]["0"]; receipt = json.loads((copied/first["training_receipt"]["path"]).read_text())
    first["checkpoint_sha256"]="e"*64; receipt["best_model_state_sha256"]="e"*64
    first["training_receipt"]=record(write(copied/"changed_receipt.json",receipt))
    manifest["core_manifest"]=record(write(copied/"changed_core.json",core)); write(path,manifest)
    with pytest.raises(ValueError,match="different current core checkpoints"): state_sources(path)


def test_minimum_available_fold_count_preserves_distinct_missing_reasons(tmp_path,current_sources):
    path = native_fixture(tmp_path/"inputs",current_sources); context = state_sources(path)
    oof_path = context["native_paths"]["oof_predictions"]; data = pd.read_parquet(oof_path)
    mask = data.state_id.eq(HISTORICAL_STATE_IDS[0]) & data.patient_fold_id.lt(3)
    data.loc[mask,"availability"]=False
    data.loc[mask,["state_membership_probability","state_effect"]]=np.nan
    data.loc[mask,"association_direction"]=None
    data.loc[mask,"availability_reason"]=["insufficient_pairs","state_measurement_unavailable","insufficient_pairs"]
    data.to_parquet(oof_path,index=False)
    aggregate_oof_predictions(data,min_available_folds=3).to_parquet(context["native_paths"]["lncrna_state_release"],index=False)
    observed = native_aggregate(context); row = observed.loc[observed.state_id.eq(HISTORICAL_STATE_IDS[0])].iloc[0]
    assert row.folds_available==2 and not row.availability and pd.isna(row.state_membership_probability)
    assert row.availability_reason=="fewer_than_3_available_folds:insufficient_pairs|state_measurement_unavailable"


def test_selected_scores_change_profiles_and_missing_weight_is_not_a_zero_effect():
    members = pd.DataFrame({"geneset_id":["G"]*3,"geneset_rank":[1,2,3],"cancer_id":["C"]*3,
        "lncrna_id":["L1","L2","L3"],"pathway_id":["P"]*3,"selected_score":[.9,.1,.8],"association_direction":["positive"]*3})
    coverage = pd.DataFrame({"cancer_id":["C","C"],"pathway_id":["P","NOSET"],"association_direction":["positive","negative"],
        "geneset_available":[True,False],"unavailable_reason":[None,"few_members"],"exported_members":[3,0]})
    aggregate = pd.DataFrame({"cancer_id":["C"]*2,"lncrna_id":["L1","L2"],"state_id":[HISTORICAL_STATE_IDS[0]]*2,
        "state_membership_probability":[0.,1.],"state_effect":[-.5,.5],"association_direction":["negative","positive"],
        "availability":[True,True],"availability_reason":[None,None],"folds_available":[5,5]})
    tables = selected_state_tables(members,coverage,aggregate); profiles = tables["selected_geneset_state_profiles"]
    row = profiles.loc[profiles.state_id.eq(HISTORICAL_STATE_IDS[0])].iloc[0]
    assert row.mean_state_membership_probability==pytest.approx(.1) and row.mean_held_out_state_effect==pytest.approx(-.4)
    assert row.selected_weight_coverage==pytest.approx(1/1.8) and row.member_coverage==pytest.approx(2/3)
    assert len(tables["selected_state_coverage"])==14 and len(tables["selected_member_states"])==21
    missing = tables["selected_member_states"].query("lncrna_id=='L3'")
    assert missing.state_effect.isna().all() and missing.availability_reason.eq("NO_NATIVE_STATE_OOF_RESULT_FOR_SELECTED_MEMBER").all()
    members.loc[:1,"selected_score"]=[.1,.9]
    changed = selected_state_tables(members,coverage,aggregate)["selected_geneset_state_profiles"]
    assert changed.loc[changed.state_id.eq(HISTORICAL_STATE_IDS[0]),"mean_held_out_state_effect"].iloc[0]==pytest.approx(.4)
    members["selected_score"]=0.
    zero = selected_state_tables(members,coverage,aggregate)["selected_geneset_state_profiles"]
    assert zero.mean_held_out_state_effect.isna().all() and not zero.availability.any()


def test_empty_selected_sets_keep_seven_state_coverage_without_invented_profiles():
    members = pd.DataFrame(columns=["geneset_id","geneset_rank","cancer_id","lncrna_id","pathway_id","selected_score","association_direction"])
    coverage = pd.DataFrame({"cancer_id":["C"],"pathway_id":["P"],"association_direction":["positive"],
        "geneset_available":[False],"unavailable_reason":["few_members"],"exported_members":[0]})
    aggregate = pd.DataFrame(columns=["cancer_id","lncrna_id","state_id","association_direction","state_membership_probability",
        "state_effect","availability","availability_reason","folds_available"])
    tables = selected_state_tables(members,coverage,aggregate)
    assert tables["selected_geneset_state_profiles"].empty and tables["selected_member_states"].empty
    assert len(tables["selected_state_coverage"])==7 and not tables["selected_state_coverage"].availability.any()
    native = native_state_sets(aggregate,StateSetConfig())
    assert all(table.empty for table in native.values())


def test_current_cli_does_not_allow_fixture_authority_or_missing_inputs(tmp_path):
    from scripts.materialize_v32_state_gene_sets import parse_args
    with pytest.raises(SystemExit): parse_args(["--current-manifest","x","--fixture-authority","--output",str(tmp_path)])
    with pytest.raises(SystemExit): parse_args(["--current-manifest","x","--minimum-set-members","1","--output",str(tmp_path)])
    with pytest.raises(SystemExit): parse_args(["--output",str(tmp_path)])


def test_materializer_enforces_server149_before_writing(tmp_path,monkeypatch):
    monkeypatch.setattr("socket.gethostname",lambda:"local")
    with pytest.raises(RuntimeError,match="149"): materialize_current_state_context(tmp_path/"missing",tmp_path/"out")
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("fault", ["raw_state", "raw_expression", "patient_split", "alignment",
    "normalizer", "train_counts", "state_inventory", "head_identity", "frozen_head", "oof_probability", "oof_effect", "oof_pairs"])
def test_numeric_replay_rejects_repinned_sources_and_self_consistent_aggregate(tmp_path,current_sources,fault):
    path=native_fixture(tmp_path/"inputs",current_sources); manifest=json.loads(path.read_text())
    lp=Path(manifest["state_training_lineage"]["path"]); lineage=json.loads(lp.read_text())
    if fault in {"raw_state","raw_expression","patient_split"}:
        role={"raw_state":"tumor_state_long_measurement","raw_expression":"lncrna_expression",
            "patient_split":"v32_patient_fold_manifest"}[fault]
        source=next(r for r in lineage["input_artifacts"] if r["role"]==role)
        raw=Path(source["path"]); data=pd.read_parquet(raw)
        if fault=="raw_state": data.loc[0,"state_value"]+=20.
        elif fault=="raw_expression": data.loc[0,"logcpm"]+=20.
        else: data.loc[data.patient_id.eq("P0_0"),"patient_fold_id"]=1
        data.to_parquet(raw,index=False); source["sha256"]=sha256(raw)
    elif fault=="alignment": lineage["fold_lineage"]["0"]["alignment_counts"]["analysis_patients"]+=1
    elif fault=="normalizer": lineage["fold_lineage"]["0"]["max_train_pairs"]+=1
    elif fault=="train_counts": lineage["fold_lineage"]["0"]["positive_candidates"]+=1
    elif fault=="state_inventory": lineage["missing_state_ids"]=[]
    elif fault in {"head_identity","frozen_head"}:
        cp=Path(lineage["new_private_head_checkpoints"]["0"]["path"]); checkpoint=torch.load(cp,weights_only=True)
        if fault=="head_identity":
            checkpoint["initialization"]["module_id"]="state"
            lineage["fold_lineage"]["0"]["initialization"]=checkpoint["initialization"]
        else:
            # Repin every tensor checksum; only the frozen forward can detect
            # that this changed head no longer produces the recorded OOF table.
            from types import SimpleNamespace
            next(reversed(checkpoint["model_state"].values())).add_(3.)
            digest=module_state_sha256(SimpleNamespace(state_dict=lambda:checkpoint["model_state"]))
            checkpoint["final_parameter_sha256"]=digest; lineage["fold_lineage"]["0"]["final_parameter_sha256"]=digest
        torch.save(checkpoint,cp); lineage["new_private_head_checkpoints"]["0"].update(record(cp))
    else:
        oof=Path(lineage["outputs"]["oof_predictions"]["path"]); data=pd.read_parquet(oof)
        index=data.index[data.availability][0]
        if fault=="oof_probability": data.loc[index,"state_membership_probability"]+=.01
        elif fault=="oof_pairs": data.loc[index,"test_n_pairs"]+=1
        else:
            data.loc[index,"state_effect"]=-.2; data.loc[index,"association_direction"]="negative"
        data.to_parquet(oof,index=False); lineage["outputs"]["oof_predictions"].update(record(oof))
        aggregate=Path(lineage["outputs"]["lncrna_state_release"]["path"])
        aggregate_oof_predictions(data,min_available_folds=3).to_parquet(aggregate,index=False)
        lineage["outputs"]["lncrna_state_release"].update(record(aggregate))
    lineage.pop("lineage_sha256"); lineage["lineage_sha256"]=canonical_json_sha256(lineage)
    write(lp,lineage); manifest["state_training_lineage"]=record(lp); write(path,manifest)
    with pytest.raises(ValueError,match="State.*(replay|inventory|initialization)"):
        state_sources(path)


def test_real_serialization_patient_alignment_and_measured_effect_oracles(tmp_path,current_sources):
    path=native_fixture(tmp_path/"inputs",current_sources); context=state_sources(path)
    replay=context["replay"]
    assert replay["rows"]==35 and replay["same_implementation_replay"]
    assert not replay["training_started"] and not replay["independent_predictive_performance_verified"]
    for row in replay["folds"]:
        assert row["max_train_pairs"]==24 and row["trainable_candidates"]==6
        assert row["positive_candidates"]==5 and row["negative_candidates"]==1
        assert row["alignment_counts"]["expression_rows_collapsed_to_patient"]==1
        assert row["alignment_counts"]["state_rows_collapsed_to_patient"]==6
        native=context["lineage"]["fold_lineage"][str(row["fold"])]
        assert native["initialization"]["module_id"]==f"state_fold_{row['fold']}"
        assert native["no_optimizer_actually_run"]
    data=pd.read_parquet(context["native_paths"]["oof_predictions"])
    assert data.loc[data.availability,"test_n_pairs"].eq(8).all()
    for index,effect in [(0,1.),(1,0.),(3,-1.)]:
        assert data.loc[data.state_id.eq(HISTORICAL_STATE_IDS[index]),"state_effect"].eq(effect).all()
    assert not data.loc[data.state_id.eq(HISTORICAL_STATE_IDS[2]),"availability"].any()


@pytest.mark.parametrize("input_style", ["dataset","csv"])
def test_native_file_and_partition_formats_reach_actual_publication(tmp_path,current_sources,input_style):
    path=native_fixture(tmp_path/"inputs",current_sources,input_style=input_style)
    result=materialize_current_state_context(path,tmp_path/"release")
    binding=json.loads(Path(result["binding_path"]).read_text())
    assert binding["native_numeric_replay"]["rows"]==35
    assert all(Path(row["path"]).is_file() for row in binding["source_artifacts"])
    if input_style=="dataset":
        assert sum("expression_dataset" in r["path"] for r in binding["source_artifacts"])==2


@pytest.mark.parametrize("change", ["added","modified","deleted"])
def test_dataset_tree_membership_remains_pinned_during_audit(tmp_path,current_sources,change):
    path=native_fixture(tmp_path/"inputs",current_sources,input_style="dataset")
    result=materialize_current_state_context(path,tmp_path/"release")
    manifest=json.loads(path.read_text()); lineage=json.loads(Path(manifest["state_training_lineage"]["path"]).read_text())
    tree=Path(next(r for r in lineage["input_artifacts"] if r["role"]=="lncrna_expression")["path"])
    first=next(tree.rglob("*.parquet"))
    if change=="added": (tree/"unexpected.txt").write_text("new unrelated member",encoding="utf-8")
    elif change=="deleted": first.unlink()
    else:
        frame=pd.read_parquet(first); frame.loc[0,"logcpm"]+=2.; frame.to_parquet(first,index=False)
    with pytest.raises(ValueError,match="dataset SHA256"):
        audit_current_state_context(result["binding_path"],expected_sha=result["binding_sha256"])
