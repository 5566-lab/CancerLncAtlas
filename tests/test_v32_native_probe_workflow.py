"""Official static reference, synthetic regions/beta values, actual consumers."""
from pathlib import Path
import hashlib
import os
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_probe_workflow as workflow,native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.distal_topology import methylation_probe_lineage,topology_lineage
from test_v32_native_methylation_workflow import assemble
from test_v32_current_state_context import current_sources

REFERENCE=Path(os.environ.get("CANCERLNCATLAS_HM450_REFERENCE",
    "/dell_2/DSC/CancerLncAtlas/runtime/references/gdc_methylation_hm450_gencode_v36_20260830_r1/HM450.hg38.manifest.gencode.v36.tsv.gz"))


def forbidden(*args,**kwargs):raise AssertionError("No learned-model optimizer")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp("native_probe")
    reference=pd.read_csv(REFERENCE,sep="\t",usecols=["CpG_chrm","CpG_beg","CpG_end","probeID"])
    local=reference.loc[reference.CpG_chrm.eq("chr1") & reference.CpG_beg.gt(2000)].sort_values("CpG_beg")
    first=local.iloc[0];second=local.loc[local.CpG_beg.gt(first.CpG_end+1000)].iloc[0]
    pd.DataFrame({"#chrom":["1","chrUn_synthetic"],"start":[int(first.CpG_beg),100],
        "end":[int(first.CpG_end),110],"gene_id":["LNC:Z.1","LNC:UNLINKED"]}).to_csv(root/"promoters.tsv",sep="\t",index=False)
    pd.DataFrame({"seqnames":["chr1"],"start":[int(second.CpG_beg)+1],"end":[int(second.CpG_end)],
        "name":["distal-z"]}).to_csv(root/"peaks.tsv",sep="\t",index=False)
    pd.DataFrame({"gene_id":["LNC:Z","LNC:UNLINKED"]}).to_csv(root/"genes.tsv",sep="\t",index=False)
    spec={"format":workflow.INPUT,"hm450_manifest":record(REFERENCE),"promoters":record(root/"promoters.tsv"),
          "required_genes":record(root/"genes.tsv"),"peak_set":record(root/"peaks.tsv"),
          "max_distance_bp":int(second.CpG_end-first.CpG_beg)+100}
    raw=root/"raw";raw.mkdir();rows=[];patients=["TCGA-AA-0001","TCGA-AA-0002"]
    for index,(patient,sample,promoter,distal) in enumerate([
            (patients[0],"S1",.2,.1),(patients[0],"S1",.6,.3),(patients[0],"S2",.9,.8),
            (patients[1],"S3",None,0)]):
        file_id="F"+str(index);folder=raw/file_id;folder.mkdir();path=folder/"beta.txt"
        path.write_text(str(first.probeID)+"\t"+(str(promoter) if promoter is not None else "NA")+"\n"+
                        str(second.probeID)+"\t"+str(distal)+"\n")
        rows.append({"file_id":file_id,"file_name":path.name,"file_size":path.stat().st_size,
                     "md5sum":hashlib.md5(path.read_bytes()).hexdigest(),"cancer_id":"BRCA","patient_id":patient,"sample_id":sample})
    pd.DataFrame(rows).to_csv(root/"selected.tsv",sep="\t",index=False)
    pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"patient_fold_id":[0,1]}).to_csv(root/"folds.tsv",sep="\t",index=False)
    methylation={"format":"NATIVE_METHYLATION_WORKFLOW_INPUT_V1","mode":"raw",
        "selected_files":record(root/"selected.tsv"),"patient_folds":record(root/"folds.tsv"),
        "probe_map":{"stage_ref":"probes","pointer":"/probe_map"},"download_root":str(raw),"workers":1,"batch_size":2}
    source=root/"CHAIN_INPUT.json";write(source,{"format":chain.INPUT,"chain_id":"official-hm450","stages":[
        {"id":"probes","operation":"probe.run","inputs":spec},
        {"id":"raw","operation":"methylation.run","inputs":methylation},
        {"id":"recovered","operation":"methylation.run","inputs":{**methylation,"mode":"recovery",
            "batches":{"stage_ref":"raw","pointer":"/batch_root"}}}]})
    chain.initialize(source,root/"chain");chain.advance(root/"chain");state=read(root/"chain/CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    return root,spec,state,reference,first,second


def test_reference_overlap_oracle_and_native_chain(completed,monkeypatch):
    root,spec,state,reference,first,second=completed
    _,binding=workflow.validate(state["results"]["probes"],root)
    actual=pd.read_parquet(binding["probe_map"]["path"])
    expected=[]
    for region,probe in (("promoter",first),("distal",second)):
        matches=reference.loc[reference.CpG_chrm.eq("chr1") & reference.CpG_beg.lt(probe.CpG_end) & reference.CpG_end.gt(probe.CpG_beg)]
        expected += [(str(p),"LNC:Z",region) for p in matches.probeID]
    assert set(map(tuple,actual.to_numpy()))==set(expected)
    audit=read(binding["audit"]["path"])
    assert audit["valid_hm450_probes"]==485547
    assert audit["gdc_hm450_manifest_md5"]=="e163fc110043abb5a7ef623816383bb9"
    assert audit["promoter_definition"]=="INPUT_BED0_PROMOTERS_SOURCE_BOUND"
    coverage=pd.read_csv(Path(binding["topology_links"]["path"]).parent/"GENE_COVERAGE.tsv",sep="\t").set_index("gene_id")
    assert coverage.loc["UNLINKED","linked_peaks"]==0 and coverage.loc["Z","linked_peaks"]==1
    values={key:read(state["results"][key]["path"]) for key in ("raw","recovered")}
    raw=pd.read_parquet(values["raw"]["outputs"]["patient_features"]["path"])
    pd.testing.assert_frame_equal(raw,pd.read_parquet(values["recovered"]["outputs"]["patient_features"]["path"]))
    patients=raw.set_index("patient_id")
    assert patients.loc["TCGA-AA-0001","promoter_methylation_beta"]==pytest.approx(.65)
    assert patients.loc["TCGA-AA-0001","distal_methylation_beta"]==pytest.approx(.5)
    assert patients.loc["TCGA-AA-0002","distal_methylation_beta"]==0
    assert pd.isna(patients.loc["TCGA-AA-0002","promoter_methylation_beta"])
    dependencies=read(values["raw"]["preflight"]["path"])["source_files"]
    assert str(REFERENCE) in {pin["path"] for pin in dependencies}
    assert not values["recovered"]["raw_checksum_reverified"]
    monkeypatch.setattr(chain,"dispatch",forbidden);chain.advance(root/"chain")


def test_official_reference_reaches_distal_public_query_without_fabricating_effect(completed,current_sources,tmp_path):
    from scripts.run_v32_distal_regulatory_mutation_head import main
    from cc_hhgt.v32.current_omics_context import materialize_current_omics_context,CurrentOmicsContextQuery
    root,_,state,_,_,_=completed;probe=read(state["results"]["probes"]["path"])
    lineage=topology_lineage(Path(probe["topology_links"]["path"]))
    prepared=assemble(tmp_path/"distal",state["results"]["recovered"],lineage)
    activity=tmp_path/"activity/cancer_id=BRCA";activity.mkdir(parents=True)
    frame=pd.read_parquet(prepared.parent/"cancer_id=BRCA/part-0.parquet")
    frame[["cancer_id","patient_id"]].drop_duplicates().assign(pathway_id="P1",activity_score=1.).to_parquet(activity/"part-0.parquet",index=False)
    candidates=tmp_path/"candidates.parquet"
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidates,index=False)
    assert main(["--head-input-root",str(prepared.parent),"--pathway-activity-root",str(activity.parent),
                 "--candidate-universe",str(candidates),"--output",str(tmp_path/"statistics")])==0
    source=tmp_path/"CONTEXT.json";write(source,{"format":"CURRENT_OMICS_CONTEXT_INPUT_V1","kind":"distal",
        "selected_analysis":record(current_sources[1]),"native_binding":record(tmp_path/"statistics/AUDIT.json")})
    result=materialize_current_omics_context(source,tmp_path/"context")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],
                                  selected_analysis_sha=record(current_sources[1])["sha256"],kind="distal")
    assert not pd.read_parquet(query.download("selected_member_distal")["path"]).distal_regulatory_mutation_pathway_available.any()
    assert str(REFERENCE) in {pin["path"] for pin in read(result["binding_path"])["source_artifacts"]}


def test_rehashed_probe_assignment_rejected_after_cache_hit(completed):
    root,_,state,_,_,_=completed;binding=read(state["results"]["probes"]["path"])
    path=Path(binding["probe_map"]["path"]);audit_path=path.parent/"AUDIT.json"
    workflow.validate_map(path);originals={p:p.read_bytes() for p in (path,audit_path)}
    try:
        frame=pd.read_parquet(path);frame.loc[0,"lncrna_id"]="LNC:FALSE";frame.to_parquet(path,index=False,compression="zstd")
        audit=read(audit_path);audit["output"]={**record(path),"bytes":path.stat().st_size};write(audit_path,audit)
        with pytest.raises(ValueError,match="Probe replay"):methylation_probe_lineage(path)
    finally:
        for p,data in originals.items():p.write_bytes(data)


@pytest.mark.parametrize("mode",["shifted_peak","fractional_promoter","negative_promoter"])
def test_invalid_or_inconsistent_coordinates_rejected(completed,tmp_path,mode):
    from scripts.materialize_v32_hm450_lncrna_probe_map import main
    _,spec,state,_,_,_=completed;binding=read(state["results"]["probes"]["path"])
    key="peak_set" if mode=="shifted_peak" else "promoters"
    frame=pd.read_csv(spec[key]["path"],sep="\t")
    frame["start"]=frame.start.astype(float)
    frame.loc[0,"start"]+=1 if mode=="shifted_peak" else .5
    if mode=="negative_promoter":frame.loc[0,"start"]=-1
    altered=tmp_path/"altered.tsv";frame.to_csv(altered,sep="\t",index=False)
    files={k:(str(altered) if k==key else spec[k]["path"]) for k in ("hm450_manifest","promoters","required_genes","peak_set")}
    args=[part for key,path in files.items() for part in ("--"+key.replace("_","-"),path)]
    with pytest.raises(ValueError,match="coordinates differ|finite integers|Invalid half-open"):
        main(args+["--topology-links",binding["topology_links"]["path"],"--output",str(tmp_path/"out")])


def test_source_change_rejected_with_warmed_replay_cache(completed):
    root,spec,state,_,_,_=completed;binding=read(state["results"]["probes"]["path"])
    workflow.validate_map(binding["probe_map"]["path"]);path=Path(spec["promoters"]["path"]);original=path.read_bytes()
    try:
        path.write_bytes(original+b"\n")
        with pytest.raises(ValueError):workflow.validate_map(binding["probe_map"]["path"])
    finally:path.write_bytes(original)


def test_plan_and_reference_guard(completed,tmp_path,monkeypatch):
    from scripts.run_native_probe_workflow import main
    _,spec,_,_,_,_=completed;manifest=tmp_path/"INPUT.json";write(manifest,spec)
    assert main(["--manifest",str(manifest),"--output",str(tmp_path/"out"),"--plan"])==0
    assert not (tmp_path/"out").exists()
    damaged=tmp_path/"wrong-reference.gz";damaged.write_bytes(b"not an official reference")
    write(manifest,{**spec,"hm450_manifest":record(damaged)})
    with pytest.raises(ValueError,match="MD5 mismatch"):workflow.plan(manifest,tmp_path/"out")
    assert not (tmp_path/"out").exists()
    write(manifest,spec);monkeypatch.setattr(socket,"gethostname",lambda:"local")
    with pytest.raises(RuntimeError,match="149"):workflow.plan(manifest,tmp_path/"out")
