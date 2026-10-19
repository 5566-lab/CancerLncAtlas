"""Actual raw-count TMM replay and inherited provenance, without model fitting."""
from pathlib import Path
import shutil
import socket
import subprocess

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import gdc_star_counts as star
from cc_hhgt.v32 import normalization_source_replay as replay
from cc_hhgt.v32.current_model_use import Inventory,inventory_tables,read,write,record
from cc_hhgt.v32.current_auxiliary_model_use import input_source
from cc_hhgt.v32.fold_local_expression import convert
from cc_hhgt.v32.model_use_completeness import is_complete
from test_v32_gdc_star_counts import inputs
from test_v32_fold_local_expression import run_r

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_model_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**kw):raise AssertionError("Only normalization replay, never model fitting")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def expression(tmp_path_factory):
    assert socket.gethostname()=="149"
    root=tmp_path_factory.mktemp("normalization_replay")
    query,cache,_=inputs(root/"raw",n=20)
    counts=star.materialize(query,"BRCA",cache,root/"counts")
    patient=root/"patients.tsv"
    pd.DataFrame({"cancer_id":"BRCA","patient_id":[f"TCGA-ZZ-{i:04}" for i in range(20)],
                  "patient_fold_id":[i%5 for i in range(20)]}).to_csv(patient,sep="\t",index=False)
    run_r(replay.PRODUCER,counts,patient,"BRCA",root/"normal")
    convert([root/"normal"],root/"converted",runner=ROOT/"scripts/convert_fold_local_expression.py")
    return root


def consumed(root,fold=0,kind="lncrna"):
    path=root/"converted"/kind/f"fold={fold}/cancer_id=BRCA/part-0.parquet"
    return {**record(path),"upstream_expression":record(path.parent/"EXPRESSION_SOURCES.json")}


def test_all_five_folds_and_both_expression_views_inherit_replayed_raw_studies(expression,monkeypatch):
    before={str(p):record(p)["sha256"] for p in (expression/"normal").rglob("*") if p.is_file()}
    calls=[];original=subprocess.run
    def observed(argv,*a,**kw):
        if str(replay.PRODUCER) in argv or str(replay.COMPARATOR) in argv:calls.append(argv)
        return original(argv,*a,**kw)
    monkeypatch.setattr(replay.subprocess,"run",observed)
    inv=Inventory([],expression)
    for fold in range(5):
        for kind in ("lncrna","gene"):
            item=consumed(expression,fold,kind)
            input_source(inv,item,expression,"core_input","expression."+kind,fold)
    tables=inventory_tables(inv,["core_input"])
    assert is_complete(tables["completeness"])
    assert len(calls)==2  # One five-fold native producer and one numerical comparison.
    assert len(tables["transformation_replays"])==10
    assert tables["transformation_replays"].outputs_compared.eq(25).all()
    assert tables["transformation_replays"].folds_recomputed.eq(5).all()
    assert not tables["transformation_replays"].model_training_started.any()
    assert set(tables["gaps"].reason)=={"GDC_UPSTREAM_ALIGNMENT_AND_BIOSPECIMEN_VERACITY_UNVERIFIED"}
    for fold in range(5):
        for kind in ("lncrna","gene"):
            rows=tables["usage"].loc[tables["usage"].source_file_sha256.eq(consumed(expression,fold,kind)["sha256"])]
            assert len(rows)==1 and "TCGA-BRCA" in rows.iloc[0].study_tokens_json
            assert rows.iloc[0].assay=="VERIFIED_UPSTREAM_STUDY_METADATA"
    assert before=={str(p):record(p)["sha256"] for p in (expression/"normal").rglob("*") if p.is_file()}


def test_formal_preparation_retains_inherited_studies_without_false_expression_gaps(expression,tmp_path):
    from cc_hhgt.v32.preparation_sources import SourceCapture,attach_sources
    capture=SourceCapture()
    control=tmp_path/"control.json";write(control,{"synthetic_only":True})
    for role in ("authorization","patient_map","patient_receipt"):
        capture.read(control,read,role=role,kind="control")
    table=tmp_path/"other.parquet";pd.DataFrame({"value":[1]}).to_parquet(table,index=False)
    for role in ("detection","pathway_hierarchy","activity"):
        capture.read(table,pd.read_parquet,role=role)
    capture.read(Path(consumed(expression)["path"]),pd.read_parquet,role="expression",cancer_id="BRCA")
    payload={"patient_fold":0,"input_scope":{"association_mode":"legacy"}}
    attach_sources(payload,capture.receipt(fold=0,mode="legacy",cancers=["BRCA"]))
    inv=Inventory([],tmp_path);inv.preparation(payload,tmp_path,"core_input",0,False)
    tables=inventory_tables(inv,["core_input"])
    expression_gaps=tables["gaps"].loc[tables["gaps"].role.str.startswith("prepared.sources.expression")]
    assert set(expression_gaps.reason)=={"GDC_UPSTREAM_ALIGNMENT_AND_BIOSPECIMEN_VERACITY_UNVERIFIED"}
    rows=tables["usage"].loc[tables["usage"].use_view.eq("PREPARATION_SOURCE_METADATA")
                              & tables["usage"].source_file_sha256.eq(consumed(expression)["sha256"])]
    assert set(rows.phase)=={"FEATURE_CONSTRUCTION","TRAINING","MODEL_SELECTION"}
    assert rows.study_tokens_json.str.contains("TCGA-BRCA").all()
    assert not is_complete(tables["completeness"])  # Other deliberately untraced fixture tables remain blocking.


@pytest.mark.parametrize("fault",["gene_matrix","fitted_state","library_size","declared_scale"])
def test_rehashed_self_consistent_conversion_cannot_hide_wrong_tmm_values(expression,tmp_path,fault):
    normal=tmp_path/"normal";shutil.copytree(expression/"normal",normal)
    if fault in {"gene_matrix","fitted_state"}:
        script=tmp_path/"tamper.R"
        script.write_text("p<-commandArgs(TRUE)[[1]];x<-readRDS(p);"+
            ("x[1,1]<-x[1,1]+1;" if fault=="gene_matrix" else "x$train_factors[1]<-x$train_factors[1]*1.1;")+
            "saveRDS(x,p)",encoding="utf-8")
        run_r(script,normal/("fold=0/gene.RDS" if fault=="gene_matrix" else "fold=0/normalization_state.RDS"))
    elif fault=="library_size":
        target=normal/"fold=0/normalization_samples.tsv"
        frame=pd.read_csv(target,sep="\t");frame.loc[0,"library_size"]+=100
        frame.to_csv(target,sep="\t",index=False)
    success=read(normal/"SUCCESS.json")
    for fold in range(5):
        source=normal/f"fold={fold}/NORMALIZATION_SOURCES.json";value=read(source)
        value["outputs"]={key:record(normal/f"fold={fold}"/Path(item["path"]).name) for key,item in value["outputs"].items()}
        if fault=="declared_scale" and fold==0:value["mean_train_effective_library_size"]*=1.1
        write(source,value);success["folds"][str(fold)]=record(source)
    write(normal/"SUCCESS.json",success)
    # The conversion remains consistent with its repinned RDS and manifest;
    # only raw-count numerical replay can reject the forged normalization.
    convert([normal],tmp_path/"converted",runner=ROOT/"scripts/convert_fold_local_expression.py")
    inv=Inventory([],tmp_path)
    with pytest.raises(ValueError,match="TMM numerical replay differs"):
        input_source(inv,consumed(tmp_path,kind="gene"),tmp_path,"core_input","gene",0)
    assert not inv.transform_replays


def test_legacy_conversion_without_release_does_not_claim_replay(expression,tmp_path):
    item=consumed(expression);value=read(item["upstream_expression"]["path"]);value.pop("normalization_release")
    source=tmp_path/"legacy.json";write(source,value);item["upstream_expression"]=record(source)
    inv=Inventory([],tmp_path);input_source(inv,item,tmp_path,"core_input","expression",0)
    tables=inventory_tables(inv,["core_input"])
    assert "NORMALIZATION_REPLAY_RELEASE_NOT_BOUND" in set(tables["gaps"].reason)
    assert not is_complete(tables["completeness"]) and not len(tables["transformation_replays"])


def test_missing_release_fold_and_wrong_host_stop_before_replay(expression,tmp_path,monkeypatch):
    source=tmp_path/"incomplete.json";value=read(expression/"normal/SUCCESS.json");value["folds"].pop("4");write(source,value)
    norm=record(expression/"normal/fold=0/NORMALIZATION_SOURCES.json")
    def forbidden(*a,**kw):raise AssertionError("No R execution before valid replay authority")
    monkeypatch.setattr(replay.subprocess,"run",forbidden)
    with pytest.raises(ValueError,match="completed five-fold"):replay.replay(record(source),tmp_path,norm)
    monkeypatch.setattr(socket,"gethostname",lambda:"wrong-host")
    with pytest.raises(RuntimeError,match="149"):replay.replay(record(expression/"normal/SUCCESS.json"),expression,norm)
