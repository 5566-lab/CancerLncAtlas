"""Actual native association calculations -> preparation -> external isolation."""
from pathlib import Path
from types import SimpleNamespace
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_model_use import record,read,write,Inventory,materialize_current_model_use
from cc_hhgt.v32.local_cnv_sources import validate,partition_binding,inspect
from cc_hhgt.v32.local_cnv_confounding import DEFAULT_COVARIATES,COVARIATE_CONTRACT_VERSION
from cc_hhgt.v32.current_core_export import sha256


def forbidden(*a,**kw):raise AssertionError("No optimizer or LASSO fitting in local-CNV source checks")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.mark.parametrize("module,function",[
    ("run_v32_local_cnv_confounding_audit_r2","main"),
    ("materialize_v32_local_cnv_validation_associations","main"),
    ("audit_v32_local_cnv_validation_associations","main"),
    ("finalize_and_audit_v32_local_cnv_r3","main"),
    ("materialize_v32_local_cnv_corrected_graph_authority","initialize"),
    ("materialize_v32_local_cnv_corrected_graph_authority","materialize_cancer"),
    ("materialize_v32_local_cnv_corrected_graph_authority","finalize"),
])
def test_recovered_producers_reject_wrong_host_before_execution(monkeypatch,module,function):
    import importlib
    entry=getattr(importlib.import_module("scripts."+module),function)
    monkeypatch.setattr(socket,"gethostname",lambda:"not-149")
    with pytest.raises(RuntimeError,match="target host 149"):
        entry() if function=="main" else entry(None)


def native_inputs(root,monkeypatch,*,expression_source=None,fold_expression_root=None,patient_map_source=None,candidate_ids=None):
    import scripts.run_v32_local_cnv_confounding_audit_r2 as association
    import scripts.materialize_v32_local_cnv_validation_associations as validation
    root.mkdir(parents=True)
    # Only fixture size is reduced. Native matrix reads, five folds, all
    # covariates, statistical calculations and source writers are unchanged.
    candidate_ids = candidate_ids or ["LNC:Z"]
    monkeypatch.setattr(association,"CANDIDATES_PER_CANCER",len(candidate_ids))
    monkeypatch.setattr(validation,"CANDIDATES_PER_CANCER",len(candidate_ids))
    n=120;t=np.arange(n,dtype=float);patients=[f"PRIVATE_LOCAL_PATIENT_{i:03}" for i in range(n)]
    if expression_source is not None:
        patients=pd.read_parquet(expression_source).patient_id.drop_duplicates().tolist()
        assert len(patients)==n
    if patient_map_source is not None:
        patients=pd.read_csv(patient_map_source,sep="\t").patient_id.tolist()
        assert len(patients)==n
    values=np.sin(t*.63)+np.cos(t*.27)
    sources={}
    for name,entity,column,ids,signal,pmid in (
        ("lnc_expression_root","lncrna_id","logcpm",["LNC:Z"],values,"55667788"),
        ("pathway_activity_root","pathway_id","activity_score",["P1"],values+np.sin(t*.13)*.2,"66778899"),
        ("gene_expression_root","gene_id","logcpm",["G1"],values+np.cos(t*.1)*.1,"77665544")):
        if name=="lnc_expression_root" and expression_source is not None:
            sources[name]={"path":str(Path(expression_source).parent.parent)};continue
        if fold_expression_root is not None and name in {"lnc_expression_root","gene_expression_root"}:
            kind="lncrna" if name=="lnc_expression_root" else "gene"
            sources[name]={"path":str(Path(fold_expression_root)/kind),"layout":"fold_local"};continue
        directory=root/name;file=directory/"cancer_id=BRCA/part-0.parquet";file.parent.mkdir(parents=True)
        pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"sample_id":patients,
                      entity:ids[0],column:signal,"pmid":pmid}).to_parquet(file,index=False)
        sources[name]={"path":str(directory)}
    covariates=pd.DataFrame({"cancer_id":"BRCA","patient_id":patients})
    for column in DEFAULT_COVARIATES:covariates[column]=0.
    covariates["purity"]=np.linspace(.2,.8,n)
    covariates["pmid"]="88997766"
    cov=root/"covariates.parquet";covariates.to_parquet(cov,index=False);sources["covariates"]=record(cov)
    config=root/"source_config.json";write(config,{"sources":sources})
    candidates=root/"candidates.parquet"
    pd.DataFrame({"cancer_id":"BRCA","lncrna_id":candidate_ids,"pathway_id":"P1"}).to_parquet(candidates,index=False)
    folds=Path(patient_map_source) if patient_map_source is not None else root/"patient_map.tsv"
    if patient_map_source is None:
        pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"patient_fold_id":np.arange(n)%5}).to_csv(folds,sep="\t",index=False)
    partition=root/"directional/cancer=BRCA";partition.mkdir(parents=True)
    for name,ids in (("patient_ids",patients),("lncrna_ids",candidate_ids),("pathway_ids",["P1"])):write(partition/(name+".json"),ids)
    for entity in ("lncrna","pathway"):
        for feature in ("signed_mean","callable","amplification","deletion"):
            shape=(n,len(candidate_ids) if entity=="lncrna" else 1)
            array=np.ones(shape,dtype=bool) if feature=="callable" else np.zeros(shape,dtype=np.float32)
            np.save(partition/(entity+"_"+feature+".npy"),array)
    raw=root/"raw_segments.tsv"
    pd.DataFrame({"patient_id":[patients[0]],"pmid":["11224488"],"value":[0.]}).to_csv(raw,sep="\t",index=False)
    write(partition/"SOURCE_FILES.json",{"records":[record(raw)]})
    write(partition/"SUCCESS.json",{"status":"SUCCESS","files":{p.name:sha256(p) for p in partition.iterdir()}})
    write(root/"directional/SUCCESS.json",{"status":"SUCCESS"})
    graph=root/"graph/static";graph.mkdir(parents=True)
    for name in ("signed_membership","protein_gene_encoding"):
        pd.DataFrame({"gene_id":["GENE:G1" if fold_expression_root is not None else "G1"]}).to_parquet(graph/(name+".parquet"),index=False)
    args=SimpleNamespace(cancer="BRCA",source_config=config,covariates=cov,candidates=candidates,patient_map=folds,
                         graph_root=graph.parent,directional_root=partition.parent,output_root=root/"train")
    train=association.run_cancer(args)
    # Synthetic aggregate authorization; it does not claim the 33-cancer
    # finalizer ran. The two per-cancer transformations below actually run.
    success={"status":"SUCCESS","decision":"LOCAL_CNV_CORE_RETRAIN_REQUIRED",
        "association_C0":"shared_covariate_current_v32_formal_association",
        "covariate_contract_version":COVARIATE_CONTRACT_VERSION,
        "covariate_contract":{"version":COVARIATE_CONTRACT_VERSION},"covariates":{"declared":list(DEFAULT_COVARIATES)},
        "synthetic_fixture":True}
    write(args.output_root/"SUCCESS.json",success)
    write(args.output_root/"LOCAL_CNV_INDEPENDENT_AUDIT.json",{"status":"PASS","synthetic_fixture":True})
    write(args.output_root/"LOCAL_CNV_DECISION.json",{"status":"PASS","decision":success["decision"]})
    vargs=SimpleNamespace(**{**vars(args),"local_cnv_audit_root":args.output_root,
        "local_cnv_success_sha256":sha256(args.output_root/"SUCCESS.json"),"run_id":"SYNTHETIC_VALIDATION",
        "output_root":root/"validation"})
    valid=validation._run(vargs)
    return args,vargs,train,valid


def prepared_capture(root,args,vargs,fold):
    from test_v32_preparation_sources import adjusted_capture
    from scripts.prepare_v32_formal import load_adjusted_associations
    capture,_,_=adjusted_capture(root,fold)
    capture.reads=[r for r in capture.reads if not r["role"].endswith("_associations")]
    frames=load_adjusted_associations(capture,args.output_root,vargs.output_root,fold=fold,cancers=["BRCA"])
    receipt=capture.receipt(fold=fold,mode="local-cnv-adjusted",cancers=["BRCA"])
    return receipt,frames


def test_native_five_fold_statistics_and_actual_preparation_preserve_sources(tmp_path,monkeypatch):
    args,vargs,train,valid=native_inputs(tmp_path/"native",monkeypatch)
    assert len(train["pair_outputs"])==5 and len(valid["outputs"])==5
    for output in train["pair_outputs"]:
        frame=pd.read_parquet(output["path"])
        assert frame.availability.eq("available").all()
        np.testing.assert_allclose(frame.rho_A1_local,frame.rho_A0_matched)
        assert frame.residual_design_rank_A1_local.eq(frame.residual_design_rank_A0_matched).all()
    receipt,frames=prepared_capture(tmp_path/"prepared",args,vargs,0)
    native=pd.read_parquet(valid["outputs"][0]["path"])
    assert frames["validation"].correlation_df.tolist()==native.correlation_df_A1_validation.tolist()
    from cc_hhgt.v32.preparation_sources import attach_sources
    payload={"patient_fold":0,"input_scope":{"association_mode":"local-cnv-adjusted"}}
    attach_sources(payload,receipt)
    inventory=Inventory([],tmp_path);inventory.preparation(payload,tmp_path,"core",0,False)
    usage=pd.DataFrame(inventory.usage)
    assert usage.study_tokens_json.str.contains("55667788").any()
    assert not usage.study_tokens_json.str.contains("77665544").any()  # gene inputs belong to G0, not A1 tables
    assert usage.study_tokens_json.str.contains("11224488").any()
    assert usage.loc[usage.use_view.eq("AUXILIARY_SOURCE_METADATA"),"phase"].isin(["FEATURE_CONSTRUCTION","MODEL_SELECTION"]).all()
    assert usage.loc[usage.phase.eq("MODEL_SELECTION"),"study_tokens_json"].str.contains("55667788").any()
    mapping=inventory.patient_use.tables()["patient_mapping"]
    assert mapping.observed_identity_count.max()==240  # sample and patient aliases
    assert not any("PRIVATE_LOCAL_PATIENT" in str(r) for r in inventory.usage)


@pytest.mark.parametrize("fault",["candidates","compact","run_id"])
def test_native_reuse_rejects_different_requested_inputs(tmp_path,monkeypatch,fault):
    import scripts.run_v32_local_cnv_confounding_audit_r2 as association
    args,_,_,_=native_inputs(tmp_path/"native",monkeypatch)
    assert association.run_cancer(args)["reused_after_sha_verification"]
    if fault=="candidates":
        changed=tmp_path/"other_candidates.parquet";changed.write_bytes(Path(args.candidates).read_bytes())
        args.candidates=changed
    elif fault=="compact":args.directional_root=tmp_path/"different_compact"
    else:monkeypatch.setattr(association,"RUN_ID","DIFFERENT_RUN")
    with pytest.raises(ValueError,match="requested authority|run identity"):association.run_cancer(args)


def test_native_e1_graph_materializer_and_source_inspector_replay_actual_edges(tmp_path,monkeypatch):
    import scripts.materialize_v32_local_cnv_corrected_graph_authority as graph
    from cc_hhgt.v32.formal_graph_authority import GRAPH_INPUT_RECEIPT_FORMAT,GRAPH_INPUT_RECEIPT_STATUS
    from cc_hhgt.v32.current_core_model_use import graph_sources
    args,_,_,_=native_inputs(tmp_path/"native",monkeypatch)
    monkeypatch.setattr(graph,"CANCERS",("BRCA",))
    source=tmp_path/"source_graph.json"
    write(source,{"format":GRAPH_INPUT_RECEIPT_FORMAT,"status":GRAPH_INPUT_RECEIPT_STATUS,"counts":{},"gates":{},
        "fold_artifacts":{str(f):{"train_coexpression":{}} for f in range(5)}})
    settings=SimpleNamespace(output_root=tmp_path/"corrected_graph",source_graph_receipt=source,
        source_graph_receipt_sha256=sha256(source),local_cnv_audit_root=args.output_root,
        local_cnv_audit_success_sha256=sha256(args.output_root/"SUCCESS.json"),cancer="BRCA")
    assert graph.initialize(settings)==0 and graph.materialize_cancer(settings)==0 and graph.finalize(settings)==0
    receipt=settings.output_root/"GRAPH_INPUT_AUTHORITY_RECEIPT.json";value=read(receipt)
    bound={"receipt":record(receipt),"fold_artifacts":value["fold_artifacts"]["0"]}
    inventory=Inventory([],tmp_path);graph_sources(inventory,bound,tmp_path,"core",0)
    usage=pd.DataFrame(inventory.usage)
    assert usage.study_tokens_json.str.contains("77665544").any()  # actual G0 gene input
    assert not usage.study_tokens_json.str.contains("66778899").any()  # pathway activity did not construct G0
    # Repinning the graph output and aggregate does not bypass E1 replay.
    output=settings.output_root/"folds/fold_0/train_coexpression/part_BRCA.parquet"
    frame=pd.read_parquet(output);assert len(frame)>0
    frame["rho"]=-frame.rho;frame.to_parquet(output,index=False)
    per_cancer=settings.output_root/"receipts/BRCA.json";producer=read(per_cancer)
    producer["outputs"][0]["sha256"]=sha256(output);write(per_cancer,producer)
    value["local_cnv_adjustment"]["source_receipts"]=[record(per_cancer)];write(receipt,value)
    from cc_hhgt.v32.input_lineage import artifact_sha256
    bound["receipt"]=record(receipt);bound["fold_artifacts"]["train_coexpression"]["sha256"]=artifact_sha256(output.parent)
    with pytest.raises(ValueError,match="native E1 transformation"):
        graph_sources(Inventory([],tmp_path),bound,tmp_path,"core",0)


@pytest.mark.parametrize("fault",["raw_table","array","missing_array","output","wrong_partition"])
def test_native_transformation_drift_rejected(tmp_path,monkeypatch,fault):
    args,vargs,train,_=native_inputs(tmp_path/"native",monkeypatch)
    item=train["consumed_sources"];path,value=validate(item,tmp_path)
    if fault=="raw_table":
        source=next(r for r in value["records"] if r["role"]=="lncrna_expression")
        pd.read_parquet(source["path"]).assign(logcpm=99.).to_parquet(source["path"],index=False)
    elif fault=="array":
        source=next(r for r in value["records"] if Path(r["path"]).name=="lncrna_signed_mean.npy")
        np.save(source["path"],np.ones((120,1),np.float32))
    elif fault=="missing_array":
        value["records"]=[r for r in value["records"] if Path(r["path"]).name!="lncrna_signed_mean.npy"]
        write(path,value);item=record(path)
    elif fault=="output":
        source=value["outputs"][0]
        pd.read_parquet(source["path"]).assign(rho_A1_local=.7).to_parquet(source["path"],index=False)
    else:
        with pytest.raises(ValueError,match="does not bind"):
            partition_binding(vargs.output_root/"validation_associations/cancer=BRCA/fold=0/part.parquet",args.output_root/"receipts/BRCA.json")
        return
    with pytest.raises(ValueError,match="SHA256 mismatch|incomplete"):validate(item,tmp_path)


def test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs(tmp_path,monkeypatch,*,native_factory=native_inputs,external_pmid="55667788"):
    import test_v32_current_binding_expansion as fixtures
    from test_v32_current_core_prediction import current_fixture
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.preparation_sources import attach_sources
    from scripts.prepare_v32_formal import make_batches,attach_replication_labels
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    args,vargs,_,_=native_factory(tmp_path/"native",monkeypatch)
    original=fixtures.verified_payload
    def payload(fold=0):
        result=original(fold);receipt,frames=prepared_capture(tmp_path/f"preparation{fold}",args,vargs,fold)
        for frame in frames.values():
            frame["shared_or_local_scope"]="shared";frame["detection_rate"]=.8;frame["detected_cancers"]=5
        frames["validation"]=attach_replication_labels(frames["train"],frames["validation"])
        for split in ("train","validation"):
            result[split+"_batches"]=make_batches(frames[split],np.zeros(1),result["bundle"],split=split)
        result["input_scope"]["association_mode"]="local-cnv-adjusted"
        result["label_contract"].update(train_direction_label_source="train_local_cnv_adjusted_effect",
            validation_direction_label_source="validation_local_cnv_adjusted_replication_effect",
            association_target="A1_LOCAL",association_available_mask="association_available")
        attach_sources(result,receipt);return result
    monkeypatch.setattr(fixtures,"verified_payload",payload)
    manifest=current_fixture(tmp_path/"current")
    primary=materialize_current_predictions(manifest,tmp_path/"prediction",expected_run_id="CURRENT-V4-SYNTHETIC")
    spec={"model_id":"core","producer_kind":"current_core_release","producer_binding":record(primary),
          "run_id":"CURRENT-V4-SYNTHETIC","score_column":"association_membership_probability"}
    source=tmp_path/"model_use.json";write(source,{"format":"CURRENT_MODEL_USE_INPUT_V1","models":[spec]})
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter,"_build_model",forbidden)
    inventory=Path(materialize_current_model_use(source,tmp_path/"inventory")["binding_path"])
    crosswalk=tmp_path/"entities.parquet"
    pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","breast","BRCA"),("pathway","p","P1")],
        columns=["namespace","raw_id","canonical_id"]).to_parquet(crosswalk,index=False)
    observed=tmp_path/"external.parquet"
    pd.DataFrame({"record_id":["r1"],"lncrna":["new"],"cancer":["breast"],"pathway":["p"],"label":[1],
        "pmid":[external_pmid],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic"]}).to_parquet(observed,index=False)
    external=tmp_path/"external.json";write(external,{"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":[spec],
        "model_use_inventory":record(inventory),"identity_crosswalk":record(crosswalk),
        "external_sources":[{"source_id":"upstream_only","kind":"exact_assay","table":record(observed)}],
        "evaluation":{"primary_model":"core","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}})
    result=materialize_current_external(external,tmp_path/"evaluation")
    query=CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    row=query.query(table="isolation")["rows"][0]
    assert row["study_overlap"] and not row["lncrna_overlap"] and not row["provenance_held_out"]
    assert not any("path" in r for r in query.query(table="model_dependencies")["rows"])
