"""Actual R distal-peak aggregation, native downstream statistics and replay."""
from pathlib import Path
import hashlib
import socket
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_distal_atac_workflow as workflow,native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.atac_training import RAW_COVERED_CANCERS
from test_v32_native_distal_workflow import source,completed
from test_v32_current_state_context import current_sources


def forbidden(*a,**k):raise AssertionError("No optimizer or redundant raw replay")


@pytest.fixture(scope="module",autouse=True)
def no_fit():
    assert socket.gethostname()=="149"
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        yield


@pytest.fixture(scope="module")
def raw_source(tmp_path_factory,source,completed):
    root=tmp_path_factory.mktemp("native_distal_atac");mroot,_,_,_=completed
    write(root/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"synthetic_raw_matrix":True})
    rows=[]
    for cindex,cancer in enumerate(RAW_COVERED_CANCERS):
        for index in range(60 if cancer=="BRCA" else 1):
            patient=f"TCGA-AA-{index:04d}" if cancer=="BRCA" else f"TCGA-{cindex:02d}-9999"
            rows.append({"bam_prefix":f"{cancer}-{index:03d}","aliquot_id":patient+"-01A","Case_ID":patient,
                "synthetic_value":.2 if cancer=="BRCA" and index==0 else .5+.02*np.sin(index)})
    for suffix,aliquot,value in (("060","01A",.6),("061","02A",.9)):
        rows.append({"bam_prefix":"BRCA-"+suffix,"aliquot_id":"TCGA-AA-0000-"+aliquot,"Case_ID":"TCGA-AA-0000","synthetic_value":value})
    pd.DataFrame(rows).to_csv(root/"mapping.tsv",sep="\t",index=False)
    generator=root/"generate.R";generator.write_text('''suppressPackageStartupMessages(library(data.table))
root <- commandArgs(TRUE)[[1]]; mapping <- fread(file.path(root,"mapping.tsv")); n <- 562709L
x <- data.frame(seqnames=rep("chr3",n), start=(0:(n-1))*20L+1L,end=(0:(n-1))*20L+10L,
                name=paste0("peak",seq_len(n)),score=0,strand="*",extra=0)
x$seqnames[1:2] <- c("chr1","chr2"); x$start[1:2] <- 201L; x$end[1:2] <- 210L; x$name[1:2] <- c("dz","dy")
for (j in seq_len(nrow(mapping))) {
  v <- numeric(n); v[1] <- mapping$synthetic_value[j]; v[2] <- .1
  x[[gsub("-","_",mapping$bam_prefix[j],fixed=TRUE)]] <- v
}
saveRDS(x,file.path(root,"matrix.rds"),compress=FALSE)
fwrite(x[,c("seqnames","start","end","name")],file.path(root,"peaks.tsv"),sep="\\t")
''',encoding="utf-8")
    subprocess.run([str(workflow.RSCRIPT),str(generator),str(root)],check=True,timeout=120)
    folds=pd.DataFrame({"cancer_id":"BRCA","patient_id":[f"TCGA-AA-{n:04d}" for n in range(60)],"patient_fold_id":np.arange(60)%5})
    folds=pd.concat([folds,pd.DataFrame({"cancer_id":"DLBC","patient_id":[f"TCGA-DD-{n:04d}" for n in range(5)],"patient_fold_id":range(5)})],ignore_index=True)
    folds.to_csv(root/"folds.tsv",sep="\t",index=False)
    pd.DataFrame({"cancer_id":["BRCA","BRCA","DLBC"],"lncrna_id":["LNC:Z","LNC:UNLINKED","LNC:Z"],"pathway_id":"P1"}).to_parquet(root/"candidates.parquet",index=False)
    paths={"matrix":root/"matrix.rds","mapping":root/"mapping.tsv","peak_set":root/"peaks.tsv",
        "topology_links":mroot/"topology/INDEPENDENT_DISTAL_LINKS.tsv.gz","patient_folds":root/"folds.tsv","candidate_universe":root/"candidates.parquet"}
    spec={"format":workflow.INPUT,**{k:record(p) for k,p in paths.items()}}
    path=root/"INPUT.json";write(path,spec);return path,spec


@pytest.fixture(scope="module")
def finished(raw_source,source):
    path,spec=raw_source;_,distal=source;root=path.parent/"chain"
    request=path.parent/"CHAIN_INPUT.json";write(request,{"format":chain.INPUT,"chain_id":"raw-atac-to-distal","stages":[
        {"id":"atac","operation":"distal_atac.run","inputs":spec},
        {"id":"distal","operation":"distal.run","inputs":{**distal,"distal_atac_root":{"stage_ref":"atac","pointer":"/feature_root"}}}]})
    chain.initialize(request,root);chain.advance(root);state=read(root/"CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    return root,state["results"]


def test_actual_r_aggregation_retains_patient_weight_and_raw_gaps(finished):
    _,results=finished;binding=read(results["atac"]["path"]);audit=read(binding["audit"]["path"])
    features=Path(binding["feature_root"]["path"]);frame=pd.read_csv(features/"BRCA.distal_accessibility.tsv.gz",sep="\t")
    assert frame.gene_id.tolist()==["Z"] and len(frame.columns)==61
    assert frame["TCGA-AA-0000"].iloc[0]==pytest.approx(.65)
    assert not (features/"DLBC.distal_accessibility.tsv.gz").exists()
    assert audit["matrix_rows"]==562709 and audit["scope_policy"]=="DECLARED_CANDIDATE_SCOPE"
    assert audit["required_lncrnas"]==2 and audit["distal_linked_lncrnas"]==1
    assert audit["raw_gap_cancers"]=="DLBC" or audit["raw_gap_cancers"]==["DLBC"]
    distal=read(results["distal"]["path"]);closure=read(distal["closure"]["path"])
    assert closure["candidate_available_rows"]==1 and not closure["raw_methylation_delete_authorized"]


def test_actual_raw_atac_reaches_public_source_inventory_and_coverage(finished,current_sources,tmp_path,raw_source):
    from cc_hhgt.v32.current_omics_context import materialize_current_omics_context,CurrentOmicsContextQuery
    _,results=finished;binding=read(results["distal"]["path"]);manifest=tmp_path/"INPUT.json"
    write(manifest,{"format":"CURRENT_OMICS_CONTEXT_INPUT_V1","kind":"distal","selected_analysis":record(current_sources[1]),"native_binding":binding["audit"]})
    result=materialize_current_omics_context(manifest,tmp_path/"public")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha=record(current_sources[1])["sha256"],kind="distal")
    coverage=pd.read_parquet(query.download("native_feature_coverage")["path"]).set_index("feature")
    assert coverage.loc["atac_distal_accessibility","available_patients"]==60
    member=pd.read_parquet(query.download("selected_member_distal")["path"]).iloc[0]
    assert member.distal_regulatory_mutation_pathway_available and member.mean_oof_slope<0
    assert raw_source[1]["matrix"]["path"] in {p["path"] for p in read(result["binding_path"])["source_artifacts"]}


def test_cached_replay_still_checks_raw_inputs(finished,raw_source,monkeypatch):
    _,results=finished;binding=read(results["atac"]["path"]);pin=binding["audit"]
    workflow.validate_features(pin,Path(pin["path"]).parent)
    monkeypatch.setattr(workflow,"invoke",forbidden)
    workflow.validate_features(pin,Path(pin["path"]).parent)
    mapping=Path(raw_source[1]["mapping"]["path"]);original=mapping.read_bytes()
    try:
        mapping.write_bytes(original+b"\n")
        with pytest.raises(ValueError):workflow.validate_features(pin,Path(pin["path"]).parent)
    finally:mapping.write_bytes(original)


def test_rehashed_accessibility_and_audit_cannot_replace_raw_reconstruction(finished):
    from cc_hhgt.v32.distal_topology import feature_topology_lineage
    _,results=finished;binding=read(results["atac"]["path"]);audit_path=Path(binding["audit"]["path"]);audit=read(audit_path)
    file=audit_path.parent/"BRCA.distal_accessibility.tsv.gz";original={p:p.read_bytes() for p in (audit_path,file)}
    try:
        frame=pd.read_csv(file,sep="\t");frame["TCGA-AA-0000"]+=.1;frame.to_csv(file,sep="\t",index=False,compression="gzip")
        audit["output_md5"][str(file)]=hashlib.md5(file.read_bytes()).hexdigest();write(audit_path,audit)
        with pytest.raises(ValueError,match="output differs from raw-matrix replay"):feature_topology_lineage(file,kind="atac")
    finally:
        for p,data in original.items():p.write_bytes(data)


def test_fractional_fold_rejected_before_output(raw_source,tmp_path):
    _,spec=raw_source;frame=pd.read_csv(spec["patient_folds"]["path"],sep="\t");frame["patient_fold_id"]=frame.patient_fold_id.astype(float)
    frame.loc[0,"patient_fold_id"]=.5;frame.to_csv(tmp_path/"folds.tsv",sep="\t",index=False)
    manifest=tmp_path/"INPUT.json";write(manifest,{**spec,"patient_folds":record(tmp_path/"folds.tsv")})
    with pytest.raises(ValueError,match="integers"):workflow.run(manifest,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_raw_mapping_must_still_cover_all_official_cancers(raw_source,tmp_path):
    _,spec=raw_source;frame=pd.read_csv(spec["mapping"]["path"],sep="\t")
    frame.loc[frame.bam_prefix.str.startswith("BRCA-")].to_csv(tmp_path/"mapping.tsv",sep="\t",index=False)
    manifest=tmp_path/"INPUT.json";write(manifest,{**spec,"mapping":record(tmp_path/"mapping.tsv")})
    with pytest.raises(subprocess.CalledProcessError):workflow.run(manifest,tmp_path/"out")
    assert "exact 23" in (tmp_path/"out/r.log").read_text()
    assert (tmp_path/"out/RUN_FAILED.json").exists() and not (tmp_path/"out/NATIVE_DISTAL_ATAC_BINDING.json").exists()


def test_same_peak_name_cannot_hide_topology_coordinate_drift(raw_source,tmp_path):
    _,spec=raw_source;original=Path(spec["topology_links"]["path"])
    links=pd.read_csv(original,sep="\t");links["peak_start0"]+=1
    path=tmp_path/"links.tsv.gz";links.to_csv(path,sep="\t",index=False,compression="gzip")
    audit=read(Path(str(original)+".audit.json"));audit["output_sha256"]=record(path)["sha256"]
    audit["output_md5"]=hashlib.md5(path.read_bytes()).hexdigest();write(Path(str(path)+".audit.json"),audit)
    manifest=tmp_path/"INPUT.json";write(manifest,{**spec,"topology_links":record(path)})
    with pytest.raises(subprocess.CalledProcessError):workflow.run(manifest,tmp_path/"out")
    assert "Topology peak coordinates differ" in (tmp_path/"out/r.log").read_text()


def test_plan_is_readonly_and_host_gate(raw_source,tmp_path,monkeypatch):
    from scripts.run_native_distal_atac_workflow import main
    path,_=raw_source;assert main(["--manifest",str(path),"--output",str(tmp_path/"out"),"--plan"])==0
    assert not (tmp_path/"out").exists()
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.run(path,tmp_path/"out")
