"""Actual GDC preprocessing workflow feeds actual local-CNV workers/closeout."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_v32_native_expression_workflow import completed,no_fit
from test_v32_gdc_star_counts import GENES,LNCS
from cc_hhgt.v32 import native_cnv_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.local_cnv_confounding import DEFAULT_COVARIATES
from cc_hhgt.v32.formal_graph_authority import GRAPH_INPUT_RECEIPT_FORMAT,GRAPH_INPUT_RECEIPT_STATUS


@pytest.fixture(scope="module")
def cnv_input(completed,tmp_path_factory):
    expression,original=completed
    root=tmp_path_factory.mktemp("cnv_after_gdc")
    binding=read(expression/"NATIVE_EXPRESSION_BINDING.json")["outputs"]
    patients=pd.read_csv(binding["patient_map"]["path"],sep="\t")
    cov=patients[["cancer_id","patient_id"]].copy()
    for name in DEFAULT_COVARIATES:cov[name]=0.
    cov["purity"]=np.tile(np.linspace(.2,.8,20),2)
    covpath=root/"covariates.parquet";cov.to_parquet(covpath,index=False)
    config=read(expression/"SOURCE_CONFIG_FRAGMENT.json");config["sources"]["covariates"]=record(covpath)
    source_config=root/"sources.json";write(source_config,config)
    cancers=sorted(patients.cancer_id.unique());pathways=sorted(pd.read_parquet(binding["candidates"]["path"]).pathway_id.unique())
    for cancer in cancers:
        ids=patients.loc[patients.cancer_id.eq(cancer),"patient_id"].tolist()
        compact=root/"directional"/("cancer="+cancer);compact.mkdir(parents=True)
        for key,values in (("patient_ids",ids),("lncrna_ids",LNCS),("pathway_ids",pathways)):write(compact/(key+".json"),values)
        for kind,count in (("lncrna",len(LNCS)),("pathway",len(pathways))):
            values=np.zeros((len(ids),count),dtype=np.float32)
            for feature,array in {"signed_mean":values,"callable":np.ones_like(values,dtype=bool),
                                  "amplification":values>.3,"deletion":values<-.3}.items():np.save(compact/(kind+"_"+feature+".npy"),array)
        raw=root/(cancer+"_synthetic_cnv.tsv");pd.DataFrame({"patient_id":ids,"signed":0.}).to_csv(raw,sep="\t",index=False)
        write(compact/"SOURCE_FILES.json",{"records":[record(raw)]})
        write(compact/"SUCCESS.json",{"status":"SUCCESS","files":{p.name:record(p)["sha256"] for p in compact.iterdir()}})
    write(root/"directional/SUCCESS.json",{"status":"SUCCESS","cancers":{c:{"processed_segment_patients":20,"typed_unavailable_patients":0} for c in cancers}})
    static=root/"graph/static";static.mkdir(parents=True)
    for key in ("signed_membership","protein_gene_encoding"):
        pd.DataFrame({"gene_id":["GENE:"+GENES[3].split(".")[0]]}).to_parquet(static/(key+".parquet"),index=False)
    graph=root/"source_graph.json"
    write(graph,{"format":GRAPH_INPUT_RECEIPT_FORMAT,"status":GRAPH_INPUT_RECEIPT_STATUS,"cancer_scope":cancers,
        "synthetic_static_control_only":True,"counts":{},"gates":{},
        "artifacts":{k:record(static/(k+".parquet")) for k in ("signed_membership","protein_gene_encoding")},
        "fold_artifacts":{str(f):{"train_coexpression":{}} for f in range(5)}})
    manifest=root/"workflow.json"
    write(manifest,{"format":workflow.INPUT,"run_id":"SYNTHETIC_GDC_NATIVE_CNV", "candidates":binding["candidates"],
        "patient_map":binding["patient_map"],"source_config":record(source_config),
        "directional_success":record(root/"directional/SUCCESS.json"),"graph_root":str(static.parent),"graph_receipt":record(graph),
        "expression_workflow":record(expression/"NATIVE_EXPRESSION_BINDING.json")})
    return root,manifest,expression


@pytest.fixture(scope="module")
def cnv_completed(cnv_input):
    root,manifest,_=cnv_input
    path=workflow.run(manifest,root/"output")
    return path


def test_actual_gdc_workflow_consumption_and_raw_inventory(cnv_input,cnv_completed):
    root,manifest,expression=cnv_input
    path=cnv_completed
    _,binding=workflow.validate(record(path),root)
    # Twenty patients/cancer give twelve training patients per outer fold,
    # exactly the existing matched-patient threshold. Variable outcomes remain
    # estimable; the constant/undetected candidate remains unavailable.
    assert binding["decision"]=="LOCAL_CNV_SENSITIVITY_ONLY_REVIEW"
    rows=pd.read_parquet(root/"output/association/local_cnv_confounding_pair_level.parquet")
    assert len(rows)==30 and set(rows.lncrna_id)==set(LNCS)
    assert rows.loc[rows.lncrna_id.eq(LNCS[0]),"availability"].eq("available").all()
    assert rows.loc[rows.lncrna_id.eq(LNCS[1]),"availability"].eq("typed_unavailable").all()
    available=rows.loc[rows.association_available]
    assert available.n_A1_local.eq(12).all() and np.isfinite(available.rho_A1_local).all()
    from cc_hhgt.v32.current_model_use import Inventory
    from cc_hhgt.v32.local_cnv_sources import partition_binding,inspect
    inventory=Inventory([],root)
    for cancer in ("BRCA","LUAD"):
        output=root/f"output/association/pair_level/cancer={cancer}/fold=0/part.parquet"
        source=partition_binding(output,root/f"output/association/receipts/{cancer}.json")
        inspect(inventory,source,root,"core","association",0)
    expected={i["path"] for i in read(expression/"LAUNCH_PREFLIGHT.json")["sources"] if "/cache/" in i["path"]}
    assert len(expected)==40 and expected <= {i["path"] for i in inventory.dependencies}
    assert not any(i["reason"]=="EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED" for i in inventory.gaps)
    assert any("GDC_PROJECT:TCGA-BRCA" in i["study_tokens_json"] for i in inventory.usage)
    assert any("GDC_PROJECT:TCGA-LUAD" in i["study_tokens_json"] for i in inventory.usage)


def test_changed_original_gdc_file_invalidates_completed_cnv_workflow(cnv_input,cnv_completed):
    root,_,expression=cnv_input
    path=Path(next(i["path"] for i in read(expression/"LAUNCH_PREFLIGHT.json")["sources"] if "/cache/" in i["path"]))
    original=path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        with pytest.raises(ValueError):workflow.validate(record(cnv_completed),root)
    finally:path.write_bytes(original)


def test_expression_binding_does_not_allow_a_different_patient_map(cnv_input,tmp_path):
    _,manifest,_=cnv_input;spec=read(manifest)
    clone=tmp_path/"other_patients.tsv";clone.write_bytes(Path(spec["patient_map"]["path"]).read_bytes())
    spec["patient_map"]=record(clone);copied=tmp_path/"changed.json";write(copied,spec)
    with pytest.raises(ValueError,match="differ from the native expression workflow"):workflow.preflight(copied,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_native_chain_dispatches_actual_local_cnv_and_preserves_decision(cnv_input,tmp_path):
    from cc_hhgt.v32 import native_workflow_chain as chain
    _,source,_=cnv_input;manifest=tmp_path/"chain.json"
    write(manifest,{"format":chain.INPUT,"chain_id":"native-cnv-chain",
        "stages":[{"id":"association","operation":"cnv.run","inputs":read(source)}]})
    output=tmp_path/"chain";chain.initialize(manifest,output);chain.advance(output);state=read(output/"CHAIN.json")
    assert state["status"]=="DECLARED_STAGES_COMPLETE",state
    _,binding=workflow.validate(state["results"]["association"],output)
    assert binding["decision"]=="LOCAL_CNV_SENSITIVITY_ONLY_REVIEW" and not binding["corrected_inputs_available"]
