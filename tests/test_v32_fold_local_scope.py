"""Training-only scope and annotation-defined candidate production; no fits."""
from pathlib import Path
from types import SimpleNamespace
import copy
import socket
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from cc_hhgt.v32.fold_local_scope import (materialize,aggregate_scope,validate_scope,validate_candidates,
    scope_input,release,inspect_scope,INPUT)
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.current_model_use import Inventory
from test_v32_fold_local_expression import normalized,forbidden

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"scripts/materialize_fold_local_scope.py"
P1="MSIGDB:HALLMARK:P1"


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def scoped(normalized):
    root=normalized/"scopes";root.mkdir()
    annotation=root/"annotation.tsv"
    pd.DataFrame({"ensembl_gene_id":["Z","LOW","LEAK","G1"],"gene_type":["lncRNA"]*3+["protein_coding"],
                  "pmid":"12344321"}).to_csv(annotation,sep="\t",index=False)
    gmt=root/"hallmark.gmt";gmt.write_text("P1\tdescription\t"+"\t".join(f"G{i}" for i in range(1,11))+"\n")
    manifest=root/"gmt.json";write(manifest,{"format":"SSGSEA_GMT_INPUT_V1","collections":[
        {"collection":"HALLMARK","file":record(gmt),"study_metadata":{"pmid":"99881122"}}]})
    for name in ("normal","changed"):
        source=root/(name+".json");write(source,{"format":INPUT,"annotation":record(annotation),"gmt_manifest":record(manifest),
            "normalizations":{"BRCA":record(normalized/name/"SUCCESS.json")},"budget_per_cancer":3,"seed":20260726})
        result=subprocess.run([sys.executable,str(RUNNER),"--manifest",str(source),"--output",str(root/name)],
                              capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
    return root


def test_candidates_include_unexpressed_lncrnas_and_ignore_all_heldout_measurements(scoped):
    a=pd.read_parquet(scoped/"normal/candidates.parquet");b=pd.read_parquet(scoped/"changed/candidates.parquet")
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert set(a.lncrna_id)=={"LNC:LOW","LNC:LEAK","LNC:Z"} and a.pathway_id.eq(P1).all()
    validate_candidates(record(scoped/"normal/CANDIDATE_SOURCES.json"),scoped,replay=True)
    first=pd.read_parquet(scoped/"normal/fold=0/scope.parquet")
    changed=pd.read_parquet(scoped/"changed/fold=0/scope.parquet")
    pd.testing.assert_frame_equal(first,changed,check_exact=True)
    rates=first.set_index("lncrna_id")
    assert rates.loc["LNC:LOW","detection_rate"]==rates.loc["LNC:LEAK","detection_rate"]==0
    assert rates.loc["LNC:Z","detection_rate"]==1 and rates.training_tumor_n.eq(72).all()
    later=pd.read_parquet(scoped/"normal/fold=2/scope.parquet").set_index("lncrna_id")
    assert later.loc["LNC:LEAK","within_cancer_eligible"]


def test_shared_classification_and_denominator_use_declared_training_cancers():
    cancers=["BRCA","LUAD","LUSC"]
    rates=pd.DataFrame([(c,l,r,20) for c in cancers for l,r in [("LNC:A",.10),("LNC:B",.09)]],
                       columns=["cancer_id","lncrna_id","detection_rate","training_tumor_n"])
    rates.loc[(rates.cancer_id=="BRCA") & (rates.lncrna_id=="LNC:B"),"detection_rate"]=.8
    result=aggregate_scope(rates,fold=0,cancers=cancers,ids=["LNC:A","LNC:B"])
    assert result.loc[result.lncrna_id.eq("LNC:A"),"shared_or_local_scope"].eq("shared").all()
    assert result.loc[result.lncrna_id.eq("LNC:B") & result.cancer_id.eq("BRCA"),"shared_or_local_scope"].eq("cancer_local").all()
    assert result.assessed_cancers.eq(3).all()
    with pytest.raises(ValueError,match="every declared"):
        aggregate_scope(rates.iloc[:-1],fold=0,cancers=cancers,ids=["LNC:A","LNC:B"])


@pytest.mark.parametrize("fault",["scope_value","candidate_value","fold","patient_map"])
def test_wrong_scope_candidates_or_patient_authority_rejected(scoped,tmp_path,fault):
    if fault=="scope_value":
        path=tmp_path/"scope.parquet";pd.read_parquet(scoped/"normal/fold=0/scope.parquet").assign(detected_cancers=33).to_parquet(path,index=False)
        value=read(scoped/"normal/fold=0/SCOPE_SOURCES.json");value["output"]=record(path)
        source=tmp_path/"source.json";write(source,value)
        with pytest.raises(ValueError,match="training-tumor detection"):validate_scope(record(source),tmp_path,path,replay=True)
    elif fault=="candidate_value":
        path=tmp_path/"candidates.parquet";pd.read_parquet(scoped/"normal/candidates.parquet").iloc[1:].to_parquet(path,index=False)
        value=read(scoped/"normal/CANDIDATE_SOURCES.json");value["output"]=record(path)
        source=tmp_path/"source.json";write(source,value)
        with pytest.raises(ValueError,match="annotation-only sampling"):validate_candidates(record(source),tmp_path,path,replay=True)
    elif fault=="fold":
        source=record(scoped/"normal/fold=0/SCOPE_SOURCES.json")
        with pytest.raises(ValueError,match="different model fold"):
            inspect_scope(Inventory([],tmp_path),source,scoped,scoped/"normal/fold=0/scope.parquet","core","scope",1)
    else:
        other=tmp_path/"patient.tsv";other.write_bytes((scoped.parent/"patient_map.tsv").read_bytes())
        with pytest.raises(ValueError,match="patient map differs"):
            release(record(scoped/"normal/SCOPE_RELEASE.json"),scoped,candidates=scoped/"normal/candidates.parquet",patient_map=other)


def test_actual_scope_capture_batch_context_and_source_inventory(scoped,tmp_path):
    from test_v32_preparation_sources import adjusted_capture
    from scripts.prepare_v32_formal import make_batches
    from test_v32_current_binding_expansion import current_bundle
    from cc_hhgt.v32.preparation_sources import attach_sources
    capture,_,_=adjusted_capture(tmp_path/"prepared",0)
    capture.reads=[r for r in capture.reads if r["role"] not in {"detection","candidate_authority"}]
    candidates=capture.read(scoped/"normal/candidates.parquet",pd.read_parquet,role="candidate_authority")
    _,value=release(record(scoped/"normal/SCOPE_RELEASE.json"),scoped,
                    candidates=scoped/"normal/candidates.parquet",patient_map=scoped.parent/"patient_map.tsv")
    scope,source=scope_input(value,scoped,0,capture=capture)
    receipt=capture.receipt(fold=0,mode="local-cnv-adjusted",cancers=["BRCA"])
    payload={"patient_fold":0,"input_scope":{"association_mode":"local-cnv-adjusted"}};attach_sources(payload,receipt)
    inventory=Inventory([],tmp_path);inventory.preparation(payload,tmp_path,"core",0,False)
    assert any("12344321" in r["study_tokens_json"] for r in inventory.usage)
    assert any("99881122" in r["study_tokens_json"] for r in inventory.usage)
    assert any("77889900" in r["study_tokens_json"] for r in inventory.usage)
    assert not any("fold=1/" in r["path"] for r in inventory.dependencies)
    frame=candidates.merge(scope,on=["cancer_id","lncrna_id"]).loc[lambda f:f.lncrna_id.eq("LNC:Z")]
    frame=frame.assign(discovery_effect=.3,proxy_label=1,weak_positive=False,association_available=True)
    batch=make_batches(frame,np.zeros(1),current_bundle(0,pathway_id=P1),split="train")[0]
    np.testing.assert_allclose(batch["conservation_context"].numpy(),[[0,1,.3,1]],atol=1e-7)


def corrected_graph_fixture(scoped):
    """Actual five-fold E1 statistics/producer, with reduced synthetic controls."""
    from test_v32_local_cnv_source_ancestry import native_inputs
    import scripts.materialize_v32_local_cnv_corrected_graph_authority as corrected
    from cc_hhgt.v32.formal_graph_authority import GRAPH_INPUT_RECEIPT_FORMAT, GRAPH_INPUT_RECEIPT_STATUS
    root=scoped/"native_corrected_graph"
    result=root/"corrected/GRAPH_INPUT_AUTHORITY_RECEIPT.json"
    if result.is_file():return result
    root.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam","AdamW","SGD"):patch.setattr(torch.optim,name,forbidden)
        args,_,_,_=native_inputs(root/"native",patch,fold_expression_root=scoped.parent/"converted",
            patient_map_source=scoped.parent/"patient_map.tsv",candidate_ids=["LNC:LEAK","LNC:LOW","LNC:Z"],
            pathway_ids=[P1],candidate_source=scoped/"normal/candidates.parquet")
        patch.setattr(corrected,"CANCERS",("BRCA",))
        source=root/"source_graph.json"
        write(source,{"format":GRAPH_INPUT_RECEIPT_FORMAT,"status":GRAPH_INPUT_RECEIPT_STATUS,"counts":{},"gates":{},
            "fold_artifacts":{str(f):{"train_coexpression":{}} for f in range(5)}})
        settings=SimpleNamespace(output_root=root/"corrected",source_graph_receipt=source,
            source_graph_receipt_sha256=record(source)["sha256"],local_cnv_audit_root=args.output_root,
            local_cnv_audit_success_sha256=record(args.output_root/"SUCCESS.json")["sha256"],cancer="BRCA")
        assert corrected.initialize(settings)==0
        assert corrected.materialize_cancer(settings)==0
        assert corrected.finalize(settings)==0
    return result


def graph_fixture(root,scoped,fold):
    from cc_hhgt.v32.formal_graph_authority import FormalGraphInputAuthority,train_identity_hashes
    from test_v32_formal_prepare_graph_authority import _declaration
    root.mkdir()
    patients=pd.read_csv(scoped.parent/"patient_map.tsv",sep="\t")
    patients["sample_id"]=patients.patient_id+"-01A"
    patients["split"]=np.where(patients.patient_fold_id.eq(fold),"test",np.where(patients.patient_fold_id.eq((fold+1)%5),"validation","train"))
    identity=train_identity_hashes(patients,cancer_scope=["BRCA"])
    expression=root/f"expression{fold}.parquet"
    patients.loc[patients.split.eq("train"),["cancer_id","sample_id","patient_id"]].to_parquet(expression,index=False)
    corrected=corrected_graph_fixture(scoped)
    native_graph=read(corrected)
    coexpression=Path(native_graph["fold_artifacts"][str(fold)]["train_coexpression"]["path"])
    frames={
        "detection":pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:LOW"],"detection_rate":[1.]}),
        "signed_membership":pd.DataFrame({"gene_id":["G1"],"pathway_id":[P1],"weight":[1.]}),
        "pathway_hierarchy":pd.DataFrame({"pathway_id":[P1],"pathway_family_id":["F1"],"weight":[1.]}),
        "lnc_protein_binding":pd.DataFrame({"lncrna_id":["LNC:Z"],"protein_id":["PR1"],"weight":[.9],"cancer_id":[None],"is_context_specific":[False]}),
        "protein_gene_encoding":pd.DataFrame({"protein_id":["PR1"],"gene_id":["G1"],"weight":[1.]}),
        "ppi":pd.DataFrame({"protein_id_a":["PR1"],"protein_id_b":["PR2"],"weight":[.7]})}
    paths={}
    for key,frame in frames.items():paths[key]=root/(key+".parquet");frame.to_parquet(paths[key],index=False)
    receipt={"cancer_scope":["BRCA"],"local_cnv_adjustment":native_graph["local_cnv_adjustment"],
        "fold_artifacts":{str(fold):{"outer_fold":fold,
        "train_expression":_declaration(expression,source_role="outer_train_expression",outer_fold=fold,identity=identity),
        "train_coexpression":_declaration(coexpression,source_role="outer_train_coexpression",outer_fold=fold,identity=identity)}}}
    path=root/"receipt.json";write(path,receipt)
    return FormalGraphInputAuthority(receipt_path=path,receipt_sha256=record(path)["sha256"],receipt=receipt,
        static_paths=paths,static_sha256={k:record(p)["sha256"] for k,p in paths.items()},
        fold_expression_pattern=str(root/"expression{fold}.parquet"),
        fold_coexpression_pattern=str(corrected.parent/"folds/fold_{fold}/train_coexpression")),patients


def test_actual_graph_uses_training_detection_and_keeps_unexpressed_nodes(scoped,tmp_path,monkeypatch):
    import cc_hhgt.v32.formal_graph_authority as graph
    from cc_hhgt.v32.formal_graph import variant_edges
    inputs,patients=graph_fixture(tmp_path/"graph",scoped,0)
    _,source=scope_input(read(scoped/"normal/SCOPE_RELEASE.json"),scoped,0)
    original=graph._read_table
    def reader(path,**kwargs):
        assert Path(path)!=inputs.static_paths["detection"],"Cohort detection must not be read"
        return original(path,**kwargs)
    monkeypatch.setattr(graph,"_read_table",reader)
    candidates=pd.read_parquet(scoped/"normal/candidates.parquet")
    result=graph.build_bound_formal_graph(inputs,outer_fold=0,split_manifest=patients,
                                        candidate_pairs=candidates,detection_override=source)
    edges=result.authority.edges
    expression=edges.loc[edges.relation_type.eq("expressed_in")]
    assert set(expression.source_id)=={"LNC:Z"} and expression.weight.eq(1).all()
    assert expression.edge_role.eq("fold_train_expression_eligibility").all() and expression.source_split.eq("train").all()
    assert {"LNC:LOW","LNC:LEAK"}<=set(result.authority.nodes.canonical_id)
    assert result.binding["static_artifacts"]["detection"]["upstream_scope"]==source["upstream_scope"]
    assert result.binding["gates"]["cohort_detection_used"] is False


def test_wrong_host_rejected_before_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,"gethostname",lambda:"not-149")
    with pytest.raises(RuntimeError,match="host 149"):materialize("missing",tmp_path/"out",runner=RUNNER)
    assert not (tmp_path/"out").exists()


def test_actual_adjusted_prepare_writes_all_three_graph_arms_with_training_scope(scoped,tmp_path,monkeypatch):
    """Only frozen-cohort control checks and L1 fitting use synthetic substitutes.

    Native five-fold statistics, scope/candidate reads, graph construction,
    feature tensors and all three actual payload writers execute unchanged.
    """
    from test_v32_local_cnv_source_ancestry import native_inputs
    import scripts.prepare_v32_formal as prepare
    import cc_hhgt.v32.patient_fold_authority as patient
    import cc_hhgt.v32.formal_graph_authority as graph
    import cc_hhgt.v32.local_cnv_core_preparation as cnv
    candidates=scoped/"normal/candidates.parquet";patient_map=scoped.parent/"patient_map.tsv"
    args,vargs,_,_=native_inputs(tmp_path/"native",monkeypatch,
        fold_expression_root=scoped.parent/"converted",patient_map_source=patient_map,
        candidate_ids=["LNC:LEAK","LNC:LOW","LNC:Z"],pathway_ids=[P1],candidate_source=candidates)
    controls={}
    for key in ("patients","authorization"):
        controls[key]=tmp_path/(key+".json");write(controls[key],{"synthetic_control":True})
    for key in ("SUCCESS","INDEPENDENT_AUDIT"):
        write(vargs.output_root/(key+".json"),{"synthetic_control":True})
    monkeypatch.setattr(patient,"validate_frozen_v32_patient_fold_binding",lambda *a:{
        "manifest_sha256":record(patient_map)["sha256"],"receipt_sha256":record(controls["patients"])["sha256"]})
    monkeypatch.setattr(cnv,"validate_validation_a1_authority",lambda *a,**kw:{"root":str(vargs.output_root),
        "success_sha256":record(vargs.output_root/"SUCCESS.json")["sha256"],
        "independent_audit_sha256":record(vargs.output_root/"INDEPENDENT_AUDIT.json")["sha256"],
        "candidate_sha256":record(candidates)["sha256"]})
    inputs,_=graph_fixture(tmp_path/"graph",scoped,0)
    monkeypatch.setattr(graph,"load_formal_graph_input_authority",lambda **kw:inputs)
    monkeypatch.setattr(prepare,"load_scope",forbidden)
    fixed=SimpleNamespace(decision_function=lambda x:np.zeros(len(x)))
    monkeypatch.setattr(prepare,"_fast_l1",lambda frames,**kw:(fixed,{"synthetic_fixed_no_fit":True},frames["train"],frames["validation"]))
    output=tmp_path/"prepared"
    argv=["prepare","--hashes",str(controls["authorization"]),"--patient-folds",str(patient_map),
        "--patient-fold-authority-receipt",str(controls["patients"]),"--graph-authority-receipt",str(inputs.receipt_path),
        "--graph-authority-receipt-sha256",inputs.receipt_sha256,
        "--graph-fold-expression-pattern",inputs.fold_expression_pattern,"--graph-fold-coexpression-pattern",inputs.fold_coexpression_pattern,
        "--formal-candidates",str(candidates),"--formal-candidates-sha256",record(candidates)["sha256"],
        "--association-mode","local-cnv-adjusted","--local-cnv-audit-root",str(args.output_root),
        "--local-cnv-audit-success-sha256",record(args.output_root/"SUCCESS.json")["sha256"],
        "--local-cnv-validation-root",str(vargs.output_root),
        "--local-cnv-validation-success-sha256",record(vargs.output_root/"SUCCESS.json")["sha256"],
        "--fold-local-scope",str(scoped/"normal/SCOPE_RELEASE.json"),"--fold-local-scope-sha256",record(scoped/"normal/SCOPE_RELEASE.json")["sha256"],
        "--budget-per-cancer","3","--fold-count","1","--output",str(output)]
    for key,path in inputs.static_paths.items():
        role=key.replace("_","-");argv += ["--graph-"+role,str(path),"--graph-"+role+"-sha256",record(path)["sha256"]]
    monkeypatch.setattr(sys,"argv",argv);assert prepare.main()==0
    summary=read(output/"PREP_INPUT_SUMMARY.json")
    assert summary["candidate_definition"]=="annotation_and_gmt_only" and summary["detection_fit_split"]=="train"
    assert set(summary["scope_by_fold"])=={"0"} and "shared_lnc_n" not in summary
    assert not (output/"FORMAL_LNCRNA_SCOPE.parquet").exists()
    actual_scope=pd.read_parquet(output/"FORMAL_LNCRNA_SCOPE/fold=0/scope.parquet")
    assert actual_scope.set_index("lncrna_id").loc["LNC:LOW","detection_rate"]==0
    for arm in ("G0","G1","G2"):
        payload=torch.load(output/arm/"PATIENT_FOLD_0.pt",weights_only=False)
        batch=payload["train_batches"][0];maps=payload["bundle"].node_maps["lncRNA"]
        z=batch["candidate_batch"]["l"].eq(maps["LNC:Z"])
        low=batch["candidate_batch"]["l"].eq(maps["LNC:LOW"])
        assert batch["conservation_context"][z,1].eq(1).all() and batch["conservation_context"][z,3].eq(1).all()
        assert batch["conservation_context"][low,1].eq(0).all() and not batch["association_available"][low].any()
        assert payload["formal_graph_authority"]["gates"]["cohort_detection_used"] is False
        detection=[r for r in payload["preparation_sources"]["source_reads"] if r["role"]=="detection"]
        assert len(detection)==1 and detection[0]["fold"]==0


def test_scope_cannot_relabel_associations_from_other_count_sources(scoped,tmp_path,monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs
    from cc_hhgt.v32.preparation_sources import SourceCapture
    from cc_hhgt.v32.fold_local_scope import verify_association_normalizations
    from scripts.prepare_v32_formal import load_adjusted_associations
    args,vargs,_,_=native_inputs(tmp_path/"native",monkeypatch,
        fold_expression_root=scoped.parent/"converted",patient_map_source=scoped.parent/"patient_map.tsv",
        candidate_ids=["LNC:LEAK","LNC:LOW","LNC:Z"],pathway_ids=[P1],candidate_source=scoped/"normal/candidates.parquet")
    capture=SourceCapture();load_adjusted_associations(capture,args.output_root,vargs.output_root,fold=0,cancers=["BRCA"])
    verify_association_normalizations(capture,record(scoped/"normal/fold=0/SCOPE_SOURCES.json"),scoped,0)
    with pytest.raises(ValueError,match="normalization sources differ"):
        verify_association_normalizations(capture,record(scoped/"changed/fold=0/SCOPE_SOURCES.json"),scoped,0)


def test_training_expression_edges_cannot_be_used_in_another_fold():
    from cc_hhgt.v32.formal_graph import materialize_expressed_in
    from cc_hhgt.v32.safe_graph import build_safe_graph
    scope=pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"detection_rate":[.8],"fold_id":[0],"fit_split":["train"]})
    edges=materialize_expressed_in(scope,scope,outer_fold=0)
    build_safe_graph(edges,outer_fold=0)
    with pytest.raises(RuntimeError,match="Unsafe edges"):build_safe_graph(edges,outer_fold=1)
    with pytest.raises(ValueError,match="matching training"):materialize_expressed_in(scope.assign(fit_split="validation"),scope,outer_fold=0)
