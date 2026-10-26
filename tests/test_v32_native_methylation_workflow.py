"""Real raw/batch aggregation and distal consumers on synthetic HM450 inputs."""
from pathlib import Path
import hashlib
import shutil
import socket

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cc_hhgt.v32 import native_methylation_workflow as workflow,native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.input_lineage import artifact_sha256
from test_v32_current_state_context import current_sources


def forbidden(*a,**k):raise AssertionError("No fitting or deletion is allowed")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp("native_methylation");raw=root/"raw";raw.mkdir();rows=[]
    patients=["TCGA-AA-0001","TCGA-AA-0002"]
    for index,(patient,sample,beta) in enumerate([(patients[0],"S1",.2),(patients[0],"S1",.6),(patients[0],"S2",.9),(patients[1],"S3",None)]):
        file_id="F"+str(index);folder=raw/file_id;folder.mkdir();path=folder/"beta.txt"
        path.write_text("cg1\t"+(str(beta) if beta is not None else "NA")+"\ncg2\tNA\ncg3\t"+("0" if beta is None else "NA")+"\n")
        rows.append({"file_id":file_id,"file_name":path.name,"file_size":path.stat().st_size,"md5sum":hashlib.md5(path.read_bytes()).hexdigest(),
            "cancer_id":"BRCA","patient_id":patient,"sample_id":sample})
    selected=root/"selected.tsv";pd.DataFrame(rows).to_csv(selected,sep="\t",index=False)
    folds=root/"folds.tsv";pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"patient_fold_id":[0,1]}).to_csv(folds,sep="\t",index=False)
    from scripts.build_v32_independent_distal_topology import main as build
    from cc_hhgt.v32.distal_topology import topology_lineage
    pd.DataFrame({"#chrom":["chr1","chr2"],"start":[100,100],"end":[110,110],"gene_id":["Z.1","Y.1"]}).to_csv(root/"promoters.tsv",sep="\t",index=False)
    pd.DataFrame({"seqnames":["chr1","chr2"],"start":[201,201],"end":[210,210],"name":["dz","dy"]}).to_csv(root/"peaks.tsv",sep="\t",index=False)
    pd.DataFrame({"gene_id":["LNC:Z","LNC:Y"]}).to_csv(root/"genes.tsv",sep="\t",index=False)
    assert build(["--promoters",str(root/"promoters.tsv"),"--peak-set",str(root/"peaks.tsv"),"--required-genes",str(root/"genes.tsv"),
        "--reference-build","GRCh38","--max-distance-bp","200","--output",str(root/"topology")])==0
    lineage=topology_lineage(root/"topology/INDEPENDENT_DISTAL_LINKS.tsv.gz")
    probe_root=root/"probe";probe_root.mkdir();probes=probe_root/"probes.parquet"
    pd.DataFrame({"probe_id":["cg1","cg2","cg3"],"lncrna_id":["LNC:Z","LNC:Z","LNC:Y"],
        "region_type":["promoter","distal","distal"]}).to_parquet(probes,index=False)
    write(probe_root/"AUDIT.json",{"output":record(probes),"topology_lineage":lineage,"synthetic_probe_coordinate_fixture":True})
    spec={"format":workflow.INPUT,"mode":"raw","selected_files":record(selected),"probe_map":record(probes),
        "patient_folds":record(folds),"download_root":str(raw),"workers":1,"batch_size":2}
    source=root/"CHAIN_INPUT.json";write(source,{"format":chain.INPUT,"chain_id":"hm450-inputs","stages":[
        {"id":"raw","operation":"methylation.run","inputs":spec},
        {"id":"recovered","operation":"methylation.run","inputs":{**spec,"mode":"recovery","batches":{"stage_ref":"raw","pointer":"/batch_root"}}}]})
    chain.initialize(source,root/"chain");chain.advance(root/"chain");state=read(root/"chain/CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    return root,spec,state,lineage


def assemble(root,binding,lineage,phenotype=None):
    from scripts.materialize_v32_distal_regulatory_head_input import main
    root.mkdir();values=read(binding["path"]);features=values["outputs"]["patient_features"]["path"]
    methylation=pd.read_parquet(features);base=methylation[["cancer_id","patient_id","lncrna_id","patient_fold_id"]]
    if phenotype is not None:base=base.merge(phenotype,on=["cancer_id","patient_id"],validate="many_to_one")
    expression=root/"expression/cancer_id=BRCA";expression.mkdir(parents=True)
    base.assign(logcpm=base["expression"] if phenotype is not None else 2.)[["cancer_id","patient_id","lncrna_id","logcpm"]].to_parquet(expression/"part-0.parquet",index=False)
    mutation=root/"mutation";mutation.mkdir();mut=mutation/"mutation.parquet"
    base.assign(distal_mutation_burden=base["mutation"] if phenotype is not None else 1.,distal_mutation_burden__available=True).to_parquet(mut,index=False)
    write(mutation/"AUDIT.json",{"topology_lineage":lineage,"outputs":{"mutation":record(mut)},"synthetic_mutation_fixture":True})
    atac=root/"atac";atac.mkdir()
    base.assign(cnv_value=base["cnv"] if phenotype is not None else .1,cnv_callable=True).to_parquet(root/"cnv.parquet",index=False)
    base[["cancer_id","patient_id"]].drop_duplicates().assign(purity=.7).to_parquet(root/"purity.parquet",index=False)
    assert main(["--formal-expression-root",str(expression.parent),"--distal-mutation",str(mut),"--distal-atac-root",str(atac),
        "--methylation",features,"--methylation-workflow",binding["path"],"--methylation-workflow-sha256",binding["sha256"],
        "--local-cnv",str(root/"cnv.parquet"),"--purity-covariates",str(root/"purity.parquet"),"--cancers","BRCA","--output",str(root/"assembled")])==0
    return root/"assembled/AUDIT.json"


def test_actual_raw_recovery_chain_preserves_weight_missingness_and_resume(completed,monkeypatch):
    root,_,state,_=completed;bindings={k:read(v["path"]) for k,v in state["results"].items()}
    frames={k:pd.read_parquet(v["outputs"]["patient_features"]["path"]) for k,v in bindings.items()}
    pd.testing.assert_frame_equal(frames["raw"],frames["recovered"])
    assert frames["raw"].loc[frames["raw"].lncrna_id.eq("LNC:Z"),"promoter_methylation_beta"].iloc[0]==pytest.approx(.65)
    assert frames["raw"].loc[frames["raw"].lncrna_id.eq("LNC:Y"),"distal_methylation_beta"].iloc[0]==0
    assert bindings["raw"]["raw_checksum_reverified"] and not bindings["recovered"]["raw_checksum_reverified"]
    assert read(bindings["recovered"]["audit"]["path"])["download_files_md5_verified"]==0
    before={str(p):record(p)["sha256"] for p in (root/"chain/stages").rglob("*") if p.is_file()}
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(root/"chain")
    assert before=={str(p):record(p)["sha256"] for p in (root/"chain/stages").rglob("*") if p.is_file()}


def test_native_methylation_reaches_actual_distal_publication_without_invented_effect(completed,current_sources,tmp_path):
    from scripts.run_v32_distal_regulatory_mutation_head import main
    from cc_hhgt.v32.current_omics_context import materialize_current_omics_context,CurrentOmicsContextQuery
    _,_,state,lineage=completed;prepared=assemble(tmp_path/"distal",state["results"]["recovered"],lineage)
    activity=tmp_path/"activity/cancer_id=BRCA";activity.mkdir(parents=True)
    frame=pd.read_parquet(Path(prepared).parent/"cancer_id=BRCA/part-0.parquet")
    frame[["cancer_id","patient_id"]].drop_duplicates().assign(pathway_id="P1",activity_score=1.).to_parquet(activity/"part-0.parquet",index=False)
    candidates=tmp_path/"candidates.parquet";pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidates,index=False)
    assert main(["--head-input-root",str(prepared.parent),"--pathway-activity-root",str(activity.parent),"--candidate-universe",str(candidates),
        "--output",str(tmp_path/"statistics")])==0
    source=tmp_path/"CONTEXT.json";write(source,{"format":"CURRENT_OMICS_CONTEXT_INPUT_V1","kind":"distal", "selected_analysis":record(current_sources[1]),
        "native_binding":record(tmp_path/"statistics/AUDIT.json")})
    result=materialize_current_omics_context(source,tmp_path/"context")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha=record(current_sources[1])["sha256"],kind="distal")
    table=pd.read_parquet(query.download("selected_member_distal")["path"])
    assert not table.distal_regulatory_mutation_pathway_available.any()
    receipt=read(result["binding_path"])
    assert state["results"]["recovered"]["path"] in {r["path"] for r in receipt["source_artifacts"]}


def test_rehashed_distal_methylation_projection_is_rejected(completed,tmp_path):
    _,_,state,lineage=completed;path=assemble(tmp_path/"distal",state["results"]["raw"],lineage);audit=read(path)
    output=Path(audit["outputs"][0]["path"]);frame=pd.read_parquet(output)
    frame.loc[frame.promoter_methylation_beta.notna(),"promoter_methylation_beta"]+=.1;frame.to_parquet(output,index=False)
    audit["outputs"][0]=record(output)
    with pytest.raises(ValueError,match="values differ"):workflow.validate_distal_input(audit,path.parent)


def test_legacy_batches_cannot_acquire_predictive_status_or_raw_deletion(completed,tmp_path):
    _,spec,state,_=completed;raw=read(state["results"]["raw"]["path"]);batches=tmp_path/"old/batches"
    shutil.copytree(raw["batch_root"]["path"],batches)
    source=tmp_path/"INPUT.json";write(source,{**spec,"mode":"recovery","download_root":str(tmp_path/"MISSING_RAW"),
        "batches":{"path":str(batches),"sha256":artifact_sha256(batches)}})
    binding=read(workflow.run(source,tmp_path/"out"));audit=read(binding["audit"]["path"])
    assert not audit["topology_lineage"]["predictive_evaluation_eligible"]
    assert audit["raw_files_read_this_run"]==0 and not audit["raw_download_delete_ready"]


def test_raw_corruption_blocks_raw_validation_but_does_not_fake_reverification_in_recovery(completed):
    root,_,state,_=completed;path=root/"raw/F0/beta.txt";original=path.read_bytes()
    try:
        path.write_bytes(original+b"\n")
        with pytest.raises(ValueError,match="checksum"):workflow.validate(state["results"]["raw"],root)
        _,recovered=workflow.validate(state["results"]["recovered"],root)
        assert recovered["raw_checksum_reverified"] is False
    finally:path.write_bytes(original)


def test_rehashed_patient_feature_and_audit_cannot_replace_raw_reconstruction(completed):
    root,_,state,_=completed;path=Path(state["results"]["raw"]["path"]);binding=read(path)
    native=Path(binding["audit"]["path"]);audit=read(native);feature=Path(binding["outputs"]["patient_features"]["path"])
    originals={p:p.read_bytes() for p in (path,native,feature)}
    try:
        table=pq.ParquetFile(feature).read();index=table.schema.get_field_index("promoter_methylation_beta")
        altered=[None if value is None else value+.1 for value in table.column(index).to_pylist()]
        table=table.set_column(index,table.schema.field(index),pa.array(altered,type=table.column(index).type));pq.write_table(table,feature)
        audit["outputs"]["patient_features"]={**audit["outputs"]["patient_features"],**record(feature),"bytes":feature.stat().st_size};write(native,audit)
        binding["outputs"]=audit["outputs"];binding["audit"]=record(native);write(path,binding)
        with pytest.raises(ValueError,match="Methylation replay differs"):workflow.validate(record(path),root)
    finally:
        for p,data in originals.items():p.write_bytes(data)


def test_raw_patient_covariates_reach_available_nested_statistics_and_public_query(completed,current_sources,tmp_path):
    from scripts.run_v32_distal_regulatory_mutation_head import main
    from cc_hhgt.v32.current_omics_context import materialize_current_omics_context,CurrentOmicsContextQuery
    _,spec,_,lineage=completed;raw=tmp_path/"raw";raw.mkdir();selected=[]
    i=np.arange(60);patients=[f"TCGA-AA-{n:04d}" for n in i]
    beta=.4+.1*np.sin(i*.8);mutation=(i%7).astype(float);cnv=.2*np.cos(i*.7)
    expression=1.8*mutation+.4*cnv+1.3*beta+.07*np.sin(i)
    for index,patient in enumerate(patients):
        folder=raw/str(index);folder.mkdir();path=folder/"beta.txt"
        path.write_text(f"cg1\t{beta[index]}\ncg2\tNA\ncg3\tNA\n")
        selected.append({"file_id":str(index),"file_name":path.name,"file_size":path.stat().st_size,
            "md5sum":hashlib.md5(path.read_bytes()).hexdigest(),"cancer_id":"BRCA","patient_id":patient,"sample_id":patient+"-01A"})
    pd.DataFrame(selected).to_csv(tmp_path/"selected.tsv",sep="\t",index=False)
    pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"patient_fold_id":i%5}).to_csv(tmp_path/"folds.tsv",sep="\t",index=False)
    source=tmp_path/"INPUT.json";write(source,{**spec,"selected_files":record(tmp_path/"selected.tsv"),
        "patient_folds":record(tmp_path/"folds.tsv"),"download_root":str(raw),"batch_size":30})
    binding=record(workflow.run(source,tmp_path/"methylation"))
    phenotype=pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"expression":expression,"mutation":mutation,"cnv":cnv})
    prepared=assemble(tmp_path/"distal",binding,lineage,phenotype)
    assembled=pd.read_parquet(prepared.parent/"cancer_id=BRCA/part-0.parquet").sort_values("patient_id")
    np.testing.assert_allclose(assembled.promoter_methylation_beta,beta,atol=1e-12,rtol=0)
    activity=tmp_path/"activity/cancer_id=BRCA";activity.mkdir(parents=True)
    pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"pathway_id":"P1",
        "activity_score":-2*expression+.01*np.cos(i)}).to_parquet(activity/"part-0.parquet",index=False)
    candidates=tmp_path/"candidates.parquet"
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidates,index=False)
    assert main(["--head-input-root",str(prepared.parent),"--pathway-activity-root",str(activity.parent),
        "--candidate-universe",str(candidates),"--output",str(tmp_path/"statistics")])==0
    context=tmp_path/"CONTEXT.json";write(context,{"format":"CURRENT_OMICS_CONTEXT_INPUT_V1","kind":"distal",
        "selected_analysis":record(current_sources[1]),"native_binding":record(tmp_path/"statistics/AUDIT.json")})
    result=materialize_current_omics_context(context,tmp_path/"context")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],
        selected_analysis_sha=record(current_sources[1])["sha256"],kind="distal")
    row=pd.read_parquet(query.download("selected_member_distal")["path"]).iloc[0]
    assert row.distal_regulatory_mutation_pathway_available and row.mean_oof_slope<0 and row.oof_prediction_correlation>.9
    coverage=pd.read_parquet(query.download("native_feature_coverage")["path"]).set_index("feature")
    assert coverage.loc["promoter_methylation_beta","available_patients"]==60
    assert coverage.loc["distal_methylation_beta","available_patients"]==0
    assert not row.causal_effect_claimed and not row.joint_geneset_effect_estimated


def test_plan_is_readonly_and_host_gate_precedes_output(completed,tmp_path,monkeypatch):
    from scripts.run_native_methylation_workflow import main
    _,spec,_,_=completed;source=tmp_path/"INPUT.json";write(source,spec)
    assert main(["--manifest",str(source),"--output",str(tmp_path/"out"),"--plan"])==0
    assert not (tmp_path/"out").exists()
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"):workflow.run(source,tmp_path/"out")
