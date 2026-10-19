"""Real synthetic raw-count -> TMM -> GSVA replay, never model fitting."""
from pathlib import Path
import shutil
import socket
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import activity_source_replay as replay
from cc_hhgt.v32 import normalization_source_replay as tmm
from cc_hhgt.v32.current_model_use import Inventory,inventory_tables,read,write,record
from cc_hhgt.v32.current_auxiliary_model_use import input_source
from cc_hhgt.v32.fold_local_activity import convert,validate
from cc_hhgt.v32.model_use_completeness import is_complete
from test_v32_normalization_source_replay import expression
from test_v32_gdc_star_counts import GENES
from test_v32_fold_local_expression import run_r

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"scripts/convert_fold_local_activity.py"


@pytest.fixture(autouse=True)
def no_model_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**kw):raise AssertionError("No neural or L1 model fitting during ssGSEA replay")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def activity(expression):
    root=expression/"activity";root.mkdir()
    annotation=root/"annotation.tsv"
    pd.DataFrame({"ensembl_gene_id":GENES,"gene_symbol":[f"S{i}" for i in range(83)],
                  "gene_type":["lncRNA"]*3+["protein_coding"]*80,"pmid":"99887711"}).to_csv(annotation,sep="\t",index=False)
    sets={"P1":[f"S{i}" for i in range(3,25)],"P2":[f"S{i}" for i in range(30,55)],
          "SMALL":["S1","S4"],"LARGE":[f"S{i}" for i in range(501)]}
    gmt=root/"pathways.gmt"
    gmt.write_text("\n".join("\t".join([name,"synthetic",*genes]) for name,genes in sets.items())+"\n",encoding="utf-8")
    manifest=root/"gmt.json";write(manifest,{"format":"SSGSEA_GMT_INPUT_V1","collections":[
        {"collection":"HALLMARK","file":record(gmt),"study_metadata":{"pmid":"12344321"}}]})
    run_r(replay.PRODUCER,expression/"normal",annotation,manifest,root/"native")
    convert([root/"native"],root/"converted",runner=RUNNER)
    return root


def consumed(root,fold=0):
    path=root/f"converted/fold={fold}/cancer_id=BRCA/part-0.parquet"
    return {**record(path),"upstream_activity":record(path.parent/"ACTIVITY_SOURCES.json")}


def test_five_fold_gsva_replay_is_cached_and_does_not_waive_unresolved_resources(activity,monkeypatch):
    before={str(p):record(p)["sha256"] for p in activity.rglob("*") if p.is_file()}
    calls=[];original=subprocess.run
    def observed(argv,*a,**kw):
        if any(str(p) in argv for p in (replay.PRODUCER,tmm.PRODUCER,replay.COMPARATOR)):calls.append(argv)
        return original(argv,*a,**kw)
    monkeypatch.setattr(replay.subprocess,"run",observed)
    inv=Inventory([],activity)
    for fold in range(5):
        item=consumed(activity,fold)
        input_source(inv,item,activity,"core_input","activity",fold)
        frame=pd.read_parquet(item["path"])
        unavailable=frame.loc[frame.pathway_id.str.endswith((":SMALL",":LARGE"))]
        assert len(unavailable) and unavailable.activity_score.isna().all()
        assert not unavailable.analysis_eligible.any()
    tables=inventory_tables(inv,["core_input"])
    proof=tables["transformation_replays"]
    assert len(calls)==4  # Each native producer + comparator once for the five-fold release.
    assert len(proof)==10 and set(proof.format)=={replay.FORMAT,tmm.FORMAT}
    assert proof.outputs_compared.eq(25).all() and proof.folds_recomputed.eq(5).all()
    assert not proof.model_training_started.any()
    activity_proof=proof.loc[proof.format.eq(replay.FORMAT)]
    assert set(activity_proof.activity_sha256)=={record(activity/f"native/fold={f}/ACTIVITY_NATIVE_SOURCES.json")["sha256"] for f in range(5)}
    assert "SSGSEA_RAW_SCORE_TRANSFORMATION_NOT_REPLAYED" not in set(tables["gaps"].reason)
    assert any("RAW_UPSTREAM" in reason for reason in tables["gaps"].reason)
    assert not is_complete(tables["completeness"])  # Annotation/reference ancestry is still unresolved.
    for fold in range(5):
        rows=tables["usage"].loc[tables["usage"].source_file_sha256.eq(consumed(activity,fold)["sha256"])]
        assert len(rows)==1
        assert all(token in rows.iloc[0].study_tokens_json for token in ("TCGA-BRCA","99887711","12344321"))
    assert before=={str(p):record(p)["sha256"] for p in activity.rglob("*") if p.is_file()}
    # Cached success is never permission to ignore a later changed consumed file.
    target=activity/"native/fold=0/raw.RDS";original_bytes=target.read_bytes()
    try:
        target.write_bytes(original_bytes+b"changed-after-proof")
        with pytest.raises(ValueError,match="SHA256"):
            replay.replay(record(activity/"native/SUCCESS.json"),activity,
                          record(activity/"native/fold=0/ACTIVITY_NATIVE_SOURCES.json"),cache=inv.visited)
    finally:target.write_bytes(original_bytes)
    assert len(calls)==4


@pytest.mark.parametrize("fault",["self_consistent_scores","fitted_state","software_version"])
def test_repinned_native_outputs_cannot_hide_wrong_gsva_calculation(activity,tmp_path,fault):
    target=tmp_path/"native";shutil.copytree(activity/"native",target)
    if fault!="software_version":
        script=tmp_path/"tamper.R"
        script.write_text("p<-commandArgs(TRUE)[[1]];"+(
            "r<-readRDS(file.path(p,'raw.RDS'));a<-readRDS(file.path(p,'activity.RDS'));"
            "s<-readRDS(file.path(p,'state.RDS'));r<-r+37.5;a<-r/s$training_score_range;"
            "saveRDS(r,file.path(p,'raw.RDS'));saveRDS(a,file.path(p,'activity.RDS'))"
            if fault=="self_consistent_scores" else
            "s<-readRDS(file.path(p,'state.RDS'));s$alpha<-.5;saveRDS(s,file.path(p,'state.RDS'))"),encoding="utf-8")
        run_r(script,target/"fold=0")
    success=read(target/"SUCCESS.json")
    for fold in range(5):
        path=target/f"fold={fold}/ACTIVITY_NATIVE_SOURCES.json";value=read(path)
        value["outputs"]={name:record(target/f"fold={fold}"/Path(item["path"]).name) for name,item in value["outputs"].items()}
        if fault=="software_version" and fold==0:value["versions"]["GSVA"]="unverified-version"
        write(path,value);success["folds"][str(fold)]=record(path)
    write(target/"SUCCESS.json",success)
    # These altered files still pass conversion/coverage/scaling consistency.
    convert([target],tmp_path/"converted",runner=RUNNER)
    item=consumed(tmp_path);validate(item["upstream_activity"],tmp_path,Path(item["path"]),replay=True)
    inv=Inventory([],tmp_path)
    with pytest.raises(ValueError,match="ssGSEA numerical replay differs"):
        replay.inspect(inv,record(target/"SUCCESS.json"),record(target/"fold=0/ACTIVITY_NATIVE_SOURCES.json"),
                       read(target/"fold=0/ACTIVITY_NATIVE_SOURCES.json")["normalization"],tmp_path,"core_input","activity",0)
    assert not inv.transform_replays


def test_legacy_activity_keeps_missing_replay_gaps(activity,tmp_path):
    item=consumed(activity);value=read(item["upstream_activity"]["path"]);value.pop("activity_release")
    source=tmp_path/"legacy.json";write(source,value);item["upstream_activity"]=record(source)
    inv=Inventory([],tmp_path);input_source(inv,item,tmp_path,"core_input","activity",0)
    tables=inventory_tables(inv,["core_input"])
    assert {"SSGSEA_REPLAY_RELEASE_NOT_BOUND","SSGSEA_RAW_SCORE_TRANSFORMATION_NOT_REPLAYED"}<=set(tables["gaps"].reason)
    assert set(tables["transformation_replays"].format)=={tmm.FORMAT}
    assert not is_complete(tables["completeness"])


def test_incomplete_release_wrong_partition_and_wrong_host_stop_before_r(activity,tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError("No numerical replay without valid execution authority")
    monkeypatch.setattr(replay.subprocess,"run",forbidden)
    selected=record(activity/"native/fold=0/ACTIVITY_NATIVE_SOURCES.json")
    broken=tmp_path/"incomplete.json";value=read(activity/"native/SUCCESS.json");value["folds"].pop("4");write(broken,value)
    with pytest.raises(ValueError,match="completed five-fold"):replay.replay(record(broken),tmp_path,selected)
    value=read(activity/"native/SUCCESS.json");value["folds"]["0"],value["folds"]["1"]=value["folds"]["1"],value["folds"]["0"]
    write(broken,value)
    with pytest.raises(ValueError,match="release fold/cancer"):replay.replay(record(broken),tmp_path,selected)
    item=consumed(activity);converted=read(item["upstream_activity"]["path"]);converted["activity_release"]=record(broken)
    receipt=tmp_path/"wrong.json";write(receipt,converted)
    with pytest.raises(ValueError,match="differs from its completed release"):
        validate(record(receipt),tmp_path,item["path"])
    monkeypatch.setattr(socket,"gethostname",lambda:"wrong-host")
    with pytest.raises(RuntimeError,match="149"):replay.replay(record(activity/"native/SUCCESS.json"),activity,selected)


def test_all_unavailable_pathways_compare_null_scale_and_na_outputs(activity,tmp_path):
    gmt=tmp_path/"small.gmt";gmt.write_text("SMALL\tsynthetic\tS3\n",encoding="utf-8")
    manifest=tmp_path/"gmt.json";write(manifest,{"format":"SSGSEA_GMT_INPUT_V1","collections":[{"collection":"HALLMARK","file":record(gmt)}]})
    target=tmp_path/"native"
    run_r(replay.PRODUCER,activity.parent/"normal",activity/"annotation.tsv",manifest,target)
    convert([target],tmp_path/"converted",runner=RUNNER)
    source=record(target/"fold=0/ACTIVITY_NATIVE_SOURCES.json");info=read(source["path"])
    assert info["training_score_range"] is None
    replay.replay(record(target/"SUCCESS.json"),tmp_path,source)
    for fold in range(5):
        frame=pd.read_parquet(consumed(tmp_path,fold)["path"])
        assert frame.activity_score.isna().all() and frame.raw_score.isna().all()
        assert not frame.analysis_eligible.any()
