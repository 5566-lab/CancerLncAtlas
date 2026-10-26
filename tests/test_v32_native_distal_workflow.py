"""Actual patient assembly, nested statistics, declared closure and publication."""
from pathlib import Path
import hashlib
import socket

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cc_hhgt.v32 import native_distal_workflow as workflow,native_workflow_chain as chain
from cc_hhgt.v32 import native_methylation_workflow as methylation
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.input_lineage import artifact_sha256
from test_v32_native_methylation_workflow import completed,assemble
from test_v32_current_state_context import current_sources


def forbidden(*a,**k):raise AssertionError("No learned-model optimizer is allowed")


@pytest.fixture(scope="module",autouse=True)
def no_fit():
    assert socket.gethostname()=="149"
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        yield


def directory(path):return {"path":str(path),"sha256":artifact_sha256(path)}


@pytest.fixture(scope="module")
def source(tmp_path_factory,completed):
    root=tmp_path_factory.mktemp("native_distal");_,spec,_,lineage=completed
    raw=root/"raw";raw.mkdir();selected=[];i=np.arange(60)
    patients=[f"TCGA-AA-{n:04d}" for n in i];beta=.4+.1*np.sin(i*.8)
    mutation=(i%7).astype(float);cnv=.2*np.cos(i*.7);expression=1.8*mutation+.4*cnv+1.3*beta+.07*np.sin(i)
    for index,patient in enumerate(patients):
        folder=raw/str(index);folder.mkdir();path=folder/"beta.txt"
        path.write_text(f"cg1\t{beta[index]}\ncg2\tNA\ncg3\tNA\n")
        selected.append({"file_id":str(index),"file_name":path.name,"file_size":path.stat().st_size,
            "md5sum":hashlib.md5(path.read_bytes()).hexdigest(),"cancer_id":"BRCA","patient_id":patient,"sample_id":patient+"-01A"})
    pd.DataFrame(selected).to_csv(root/"selected.tsv",sep="\t",index=False)
    pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"patient_fold_id":i%5}).to_csv(root/"folds.tsv",sep="\t",index=False)
    m=root/"METHYLATION.json";write(m,{**spec,"selected_files":record(root/"selected.tsv"),
        "patient_folds":record(root/"folds.tsv"),"download_root":str(raw),"batch_size":30})
    binding=record(methylation.run(m,root/"methylation"))
    phenotype=pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"expression":expression,"mutation":mutation,"cnv":cnv})
    prepared=assemble(root/"inputs",binding,lineage,phenotype);audit=read(prepared)
    mutation_audit=root/"inputs/mutation/AUDIT.json";native=read(mutation_audit)
    native.update(status="PASS_DISTAL_MUTATION_FEATURES",locus_callability_claimed=False,missing_assay_assumed_zero=False);write(mutation_audit,native)
    write(root/"inputs/atac/SOURCE_UNAVAILABLE.json",{"synthetic":True,"reason":"No ATAC samples in this synthetic fixture"})
    activity=root/"activity/cancer_id=BRCA";activity.mkdir(parents=True)
    pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"pathway_id":"P1",
        "activity_score":-2*expression+.01*np.cos(i)}).to_parquet(activity/"part-0.parquet",index=False)
    candidates=root/"candidates.parquet"
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidates,index=False)
    spec={"format":workflow.INPUT,"methylation_workflow":binding,
        "formal_expression_root":directory(root/"inputs/expression"),"distal_atac_root":directory(root/"inputs/atac"),
        "pathway_activity_root":directory(root/"activity"),"candidate_universe":record(candidates),
        **{k:audit["sources"][k] for k in ("distal_mutation","local_cnv","purity_covariates")}}
    path=root/"INPUT.json";write(path,spec)
    return path,spec


@pytest.fixture(scope="module")
def finished(source):
    path,spec=source;manifest=path.parent/"CHAIN.json"
    write(manifest,{"format":chain.INPUT,"chain_id":"distal-analysis","stages":[{"id":"distal","operation":"distal.run","inputs":spec}]})
    root=path.parent/"chain";chain.initialize(manifest,root);chain.advance(root)
    state=read(root/"CHAIN.json");assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    return root,state["results"]["distal"]


def test_actual_declared_scope_closure_and_chain_resume(finished,monkeypatch):
    root,pin=finished;binding=read(pin["path"]);closure=read(binding["closure"]["path"])
    assert closure["candidate_scope_policy"]=="DECLARED_CANDIDATE_FILE" and closure["candidate_rows"]==1
    assert closure["candidate_available_rows"]==1 and not closure["full33"]
    assert not closure["raw_methylation_delete_authorized"]
    before={str(p):record(p)["sha256"] for p in (root/"stages").rglob("*") if p.is_file()}
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(root)
    assert before=={str(p):record(p)["sha256"] for p in (root/"stages").rglob("*") if p.is_file()}


def test_actual_workflow_result_reaches_publication_and_query(finished,current_sources,tmp_path):
    from cc_hhgt.v32.current_omics_context import materialize_current_omics_context,CurrentOmicsContextQuery
    _,pin=finished;binding=read(pin["path"]);path=tmp_path/"INPUT.json"
    write(path,{"format":"CURRENT_OMICS_CONTEXT_INPUT_V1","kind":"distal","selected_analysis":record(current_sources[1]),"native_binding":binding["audit"]})
    result=materialize_current_omics_context(path,tmp_path/"public")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],
        selected_analysis_sha=record(current_sources[1])["sha256"],kind="distal")
    row=pd.read_parquet(query.download("selected_member_distal")["path"]).iloc[0]
    assert row.distal_regulatory_mutation_pathway_available and row.mean_oof_slope<0 and row.oof_prediction_correlation>.9
    records=read(result["binding_path"])["source_artifacts"]
    assert binding["preflight"]["path"] in {r["path"] for r in records}
    assert not row.causal_effect_claimed and not row.joint_geneset_effect_estimated


def test_rehashed_non_methylation_covariate_is_rejected(finished):
    _,pin=finished;binding=read(pin["path"]);path=Path(binding["prepared"]["path"]);audit=read(path)
    file=Path(audit["outputs"][0]["path"]);original={p:p.read_bytes() for p in (path,file)}
    try:
        table=pq.ParquetFile(file).read();index=table.schema.get_field_index("local_cnv_log2")
        table=table.set_column(index,table.schema.field(index),pa.array([.99]*len(table),type=table.column(index).type));pq.write_table(table,file)
        audit["outputs"][0]={**audit["outputs"][0],**record(file),"bytes":file.stat().st_size};write(path,audit)
        with pytest.raises(ValueError,match="Distal prepared replay differs"):workflow.validate_prepared(record(path),path.parent)
    finally:
        for p,data in original.items():p.write_bytes(data)


def test_closure_rejects_equal_count_wrong_candidate_identity(finished,tmp_path):
    _,pin=finished;binding=read(pin["path"]);output=Path(pin["path"]).parent;value=read(binding["preflight"]["path"])
    audit_path=Path(binding["audit"]["path"]);audit=read(audit_path);success=audit_path.parent/"SUCCESS.json"
    row=next(r for r in audit["artifacts"] if "candidate_oof_pathway_head" in r["path"]);file=Path(row["path"])
    original={p:p.read_bytes() for p in (audit_path,success,file)}
    try:
        frame=pd.read_parquet(file);frame["pathway_id"]="WRONG";frame.to_parquet(file,index=False)
        row.update(record(file),bytes=file.stat().st_size);write(audit_path,audit)
        write(success,{"status":"PASS","audit_sha256":record(audit_path)["sha256"],"success_written_last":True})
        with pytest.raises(RuntimeError,match="Candidate identities differ"):workflow.closure(value,output,tmp_path/"bad")
    finally:
        for p,data in original.items():p.write_bytes(data)


def test_legacy_closure_still_requires_original_full_cohort(finished,tmp_path):
    from scripts.audit_v32_distal_regulatory_mutation_head import main
    _,pin=finished;binding=read(pin["path"]);output=Path(pin["path"]).parent;value=read(binding["preflight"]["path"])
    m=read(value["methylation_workflow"]["path"])
    with pytest.raises(RuntimeError,match="not full33"):
        main(["--head-root",str(output/"statistics"),"--head-input-root",str(output/"head_input"),
            "--methylation-root",str(Path(m["audit"]["path"]).parent),"--distal-mutation-root",str(Path(value["files"]["distal_mutation"]["path"]).parent),
            "--candidate-universe",value["files"]["candidate_universe"]["path"],"--output",str(tmp_path/"old")])


def test_no_available_effect_does_not_receive_workflow_completion(source,tmp_path):
    _,spec=source;expression=tmp_path/"constant/cancer_id=BRCA";expression.mkdir(parents=True)
    original=Path(spec["formal_expression_root"]["path"])/"cancer_id=BRCA/part-0.parquet"
    frame=pd.read_parquet(original);frame["logcpm"]=1.;frame.to_parquet(expression/"part-0.parquet",index=False)
    manifest=tmp_path/"INPUT.json";write(manifest,{**spec,"formal_expression_root":directory(expression.parent)})
    with pytest.raises(RuntimeError,match="No distal-regulatory mutation head row is available"):
        workflow.run(manifest,tmp_path/"out")
    assert (tmp_path/"out/statistics/SUCCESS.json").is_file()
    assert (tmp_path/"out/RUN_FAILED.json").is_file()
    assert not (tmp_path/"out/NATIVE_DISTAL_BINDING.json").exists()
    assert not (tmp_path/"out/closure/SUCCESS.json").exists()


def test_completed_statistics_cannot_be_relabelled_with_different_ridge_settings(finished):
    _,pin=finished;binding=read(pin["path"]);value=read(binding["preflight"]["path"])
    value["parameters"]["ridge_alpha_pathway"]+=1
    with pytest.raises(ValueError,match="settings differ from preparation"):
        workflow.check_results(value,Path(pin["path"]).parent)


@pytest.mark.parametrize("parameters",[{"ridge_alpha_expression":float("nan")},{"min_pathway_train_patients":4},{"unexpected":1}])
def test_invalid_parameters_fail_before_output(source,tmp_path,parameters):
    _,spec=source;path=tmp_path/"INPUT.json";write(path,{**spec,"parameters":parameters})
    with pytest.raises(ValueError):workflow.run(path,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_plan_is_readonly_and_host_gate(source,tmp_path,monkeypatch):
    from scripts.run_native_distal_workflow import main
    path,_=source;assert main(["--manifest",str(path),"--output",str(tmp_path/"out"),"--plan"])==0
    assert not (tmp_path/"out").exists()
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.run(path,tmp_path/"out")
