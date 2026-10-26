"""Actual Clinical producer/replay checks on synthetic patients, without fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from test_v32_current_clinical_context import (
    clinical_fixture, current_sources, forbid_training, record, write,
)
from cc_hhgt.v32.current_clinical_sources import selected_sources, patient_risk_sources, entity_sources
from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.input_lineage import artifact_sha256, _canonical_json_sha256
from cc_hhgt.v32.clinical_entity_training import input_path_sha256, json_sha256
from scripts import run_v32_clinical_training as producer


def read(path): return json.loads(Path(path).read_text())


def replace_arg(argv, key, value):
    argv=list(argv); argv[argv.index(key)+1]=str(value); return argv


def repin_risk(path, lineage):
    manifest=read(path); lp=Path(manifest["patient_risk_lineage"]["path"])
    audit_path=lp.parent/"INPUT_LINEAGE_AUDIT.json"; audit=read(audit_path)
    for item in [*lineage["source_contract"]["files"].values(), *lineage["input_artifacts"], *audit["artifacts"]]:
        item["sha256"]=artifact_sha256(item["path"])
    audit["lineage_sha256"]=_canonical_json_sha256({k:v for k,v in audit.items() if k not in {"lineage_sha256","patient_fold_authority"}})
    write(audit_path,audit); lineage["input_manifest_sha256"]=audit["lineage_sha256"]
    for task in lineage["task_lineage"]: task["input_lineage_sha256"]=audit["lineage_sha256"]
    lineage["checkpoint_manifest_sha256"]=json_sha256(lineage["task_lineage"])
    write(lp,lineage); manifest["patient_risk_lineage"]=record(lp); write(path,manifest)


def repin_entity(path, lineage, calculation):
    manifest=read(path); lp=Path(manifest["entity_lineage"]["path"]); cp=Path(manifest["entity_calculation_manifest"]["path"])
    for item in [*lineage["source_contract"]["files"].values(), *lineage["input_artifacts"]]:
        item["sha256"]=input_path_sha256(item["path"])
    calculation["source_contract"]=lineage["source_contract"]
    lineage["input_manifest_sha256"]=json_sha256(lineage["input_artifacts"])
    write(cp,calculation); lineage["calculation_manifest_sha256"]=sha256(cp); write(lp,lineage)
    manifest["entity_lineage"]=record(lp); manifest["entity_calculation_manifest"]=record(cp); write(path,manifest)


def test_actual_risk_prepare_only_collects_same_five_inputs_without_head_or_forward(tmp_path,current_sources,monkeypatch):
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    lineage=read(read(path)["patient_risk_lineage"]["path"])
    args=replace_arg(read(path.parent/"PRODUCER_COMMANDS.json")["risk"],"--output-root",tmp_path/"prepared")
    def forbidden(*a,**k): raise AssertionError("Preparation may not construct, fit or forward a model")
    monkeypatch.setattr("cc_hhgt.v32.clinical_training.build_clinical_head",forbidden)
    monkeypatch.setattr("cc_hhgt.v32.clinical_training.train_private_clinical_head",forbidden)
    monkeypatch.setattr(torch.nn.Module,"__call__",forbidden)
    assert producer.main([*args,"--prepare-only"])==0
    out=tmp_path/"prepared"; preflight=read(out/"HEAD_PREFLIGHT.json")
    assert preflight["target_host"]=="149" and preflight["status"]=="PASS_READY"
    assert not preflight["training_started"] and not preflight["model_forward_started"]
    assert preflight["source_contract"]==lineage["source_contract"]
    assert preflight["training_config"]==lineage["training_config"]
    assert len(preflight["folds"])==5
    for actual,done in zip(preflight["folds"],lineage["task_lineage"]):
        assert all(actual[k]==v for k,v in done["input_description"].items())
        pd.testing.assert_frame_equal(pd.read_parquet(actual["feature_artifact"]["path"]),pd.read_parquet(done["feature_artifact"]["path"]))
    for pattern in ("best_model_state.pt","SUCCESS.json","MODULE_LINEAGE.json","clinical_patient_risk.parquet"):
        assert not list(out.rglob(pattern))


@pytest.mark.parametrize("role",["expression_root","activity_root","workbook"])
def test_rehashed_raw_risk_source_must_reproduce_saved_features_and_targets(tmp_path,current_sources,role):
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    lineage=read(read(path)["patient_risk_lineage"]["path"])
    raw=Path(lineage["source_contract"]["files"][role]["path"])
    if role=="workbook":
        main=pd.read_excel(raw,sheet_name="TCGA-CDR");extra=pd.read_excel(raw,sheet_name="ExtraEndpoints")
        main.loc[0,"OS.time"]+=17.
        with pd.ExcelWriter(raw,engine="openpyxl") as writer:
            main.to_excel(writer,sheet_name="TCGA-CDR",index=False);extra.to_excel(writer,sheet_name="ExtraEndpoints",index=False)
    else:
        part=raw/"cancer_id=BRCA/part-0.parquet";frame=pd.read_parquet(part)
        column="logcpm" if role=="expression_root" else "activity_score"
        frame.loc[0,column]+=11.;frame.to_parquet(part,index=False)
    repin_risk(path,lineage)
    with pytest.raises(ValueError,match="Clinical (raw patient feature projection replay|replay endpoint bins, labels, splits)"):
        patient_risk_sources(selected_sources(path))


def test_changed_training_features_cannot_hide_behind_unchanged_test_predictions(tmp_path,current_sources):
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    lineage=read(read(path)["patient_risk_lineage"]["path"]); row=lineage["task_lineage"][0]
    fp=Path(row["feature_artifact"]["path"]);features=pd.read_parquet(fp)
    column=row["input_description"]["core_columns"][0]
    features.loc[~features.patient_fold_id.isin([0,1]),column]+=2.
    features.to_parquet(fp,index=False);row["feature_artifact"]=record(fp);repin_risk(path,lineage)
    with pytest.raises(ValueError,match="raw patient feature projection replay"):
        patient_risk_sources(selected_sources(path))


@pytest.mark.parametrize("field",["source_contract","training_config","input_description"])
def test_missing_original_input_contract_is_not_replaced_with_defaults(tmp_path,current_sources,field):
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    manifest=read(path);lp=Path(manifest["patient_risk_lineage"]["path"]);lineage=read(lp)
    if field=="input_description": lineage["task_lineage"][0].pop(field)
    else: lineage.pop(field)
    lineage["checkpoint_manifest_sha256"]=json_sha256(lineage["task_lineage"])
    write(lp,lineage);manifest["patient_risk_lineage"]=record(lp);write(path,manifest)
    with pytest.raises(ValueError,match="native source roles|complete training settings|endpoint bins, labels, splits"):
        patient_risk_sources(selected_sources(path))


def test_rehashed_state_measurements_must_reproduce_entity_statistics(tmp_path,current_sources):
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    risk=patient_risk_sources(selected_sources(path));manifest=read(path)
    lineage=read(manifest["entity_lineage"]["path"]);calc=read(manifest["entity_calculation_manifest"]["path"])
    raw=Path(lineage["source_contract"]["files"]["state_measurements"]["path"])
    frame=pd.read_parquet(raw);available=frame.state_value.notna();frame.loc[available,"state_value"]=-frame.loc[available,"state_value"]
    frame.to_parquet(raw,index=False);repin_entity(path,lineage,calc)
    with pytest.raises(ValueError,match="raw entity patient statistic replay|entity collected input description"):
        entity_sources(selected_sources(path),risk)


def test_entity_recalculated_summary_does_not_validate_forged_fold_statistics(tmp_path,current_sources):
    from cc_hhgt.v32.clinical_entity_training import summarise_oof_associations
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    risk=patient_risk_sources(selected_sources(path));manifest=read(path)
    lineage=read(manifest["entity_lineage"]["path"]);calc=read(manifest["entity_calculation_manifest"]["path"])
    part=calc["partitions"][0];fp=Path(part["fold_records_path"]);folds=pd.read_parquet(fp)
    assert folds.test_beta.notna().any()
    folds.loc[folds.test_beta.notna(),"test_beta"]+=.2
    folds.to_parquet(fp,index=False);part["fold_records_sha256"]=sha256(fp)
    summary=summarise_oof_associations(folds,sorted(folds.subject_id.unique()),cancer_id=part["cancer_id"],
        subject_type=part["subject_type"],min_folds_available=lineage["calculation_config"]["minimums"]["folds_available"])
    sp=Path(part["summary_part_path"]);summary.to_parquet(sp,index=False);part["summary_part_sha256"]=sha256(sp)
    repin_entity(path,lineage,calc)
    with pytest.raises(ValueError,match="raw entity patient statistic replay"):
        entity_sources(selected_sources(path),risk)


def test_native_entity_subset_does_not_require_unused_measurement_files(tmp_path,current_sources):
    from scripts.run_v32_clinical_entity_training import main
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    args=replace_arg(read(path.parent/"PRODUCER_COMMANDS.json")["entity"],"--output-root",tmp_path/"subset")
    for key in ("--pathway-activity-root","--state-measurements"): args=replace_arg(args,key,tmp_path/"unused-missing")
    assert main([*args,"--subject-types","lncRNA"])==0
    lineage=read(tmp_path/"subset/ENTITY_CLINICAL_LINEAGE.json")
    assert lineage["subject_types"]==["lncRNA"]
    assert set(lineage["source_contract"]["files"])=={"workbook","patient_folds","patient_authority_receipt","candidates","expression_root","state_lineage"}


def test_legacy_frozen_default_and_current_output_guards_remain(tmp_path,current_sources,monkeypatch):
    from scripts.run_v32_clinical_entity_training import _parser as entity_parser
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    commands=read(path.parent/"PRODUCER_COMMANDS.json");args=commands["risk"]
    assert producer._parser().parse_args([]).patient_authority_mode=="frozen"
    entity_args=commands["entity"].copy();pos=entity_args.index("--patient-authority-mode");del entity_args[pos:pos+2]
    assert entity_parser().parse_args(entity_args).patient_authority_mode=="frozen"
    with pytest.raises(FileExistsError): producer.main([*args,"--prepare-only"])
    with pytest.raises(ValueError,match="cannot resume"): producer.main([*args,"--prepare-only","--resume"])
    frozen=replace_arg(args,"--output-root",tmp_path/"frozen");pos=frozen.index("--patient-authority-mode");del frozen[pos:pos+2]
    with pytest.raises((ValueError,RuntimeError)): producer.main([*frozen,"--prepare-only"])
    assert not (tmp_path/"frozen").exists()
    monkeypatch.setattr("socket.gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"): producer.main([*args,"--prepare-only"])


def test_prepare_detects_raw_source_changed_during_collection(tmp_path,current_sources,monkeypatch):
    path=clinical_fixture(tmp_path/"inputs",current_sources)
    args=replace_arg(read(path.parent/"PRODUCER_COMMANDS.json")["risk"],"--output-root",tmp_path/"prepared")
    original=producer.collect_fold_inputs; altered=[]
    def collect(*a,**k):
        result=original(*a,**k)
        if not altered:
            source=Path(args[args.index("--expression-root")+1])/"cancer_id=BRCA/part-0.parquet"
            frame=pd.read_parquet(source);frame.loc[0,"logcpm"]+=.1;frame.to_parquet(source,index=False);altered.append(True)
        return result
    monkeypatch.setattr(producer,"collect_fold_inputs",collect)
    with pytest.raises(RuntimeError,match="source changed during execution"): producer.main([*args,"--prepare-only"])
    assert not (tmp_path/"prepared/HEAD_PREFLIGHT.json").exists()
