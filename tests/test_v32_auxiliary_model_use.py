"""Native auxiliary inputs reach actual fusion and external consumers; no fit."""
import json
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_model_use import INPUT_FORMAT,record,read,write,derive_manifest,materialize_current_model_use,load_current_model_use
from cc_hhgt.v32.current_core_export import sha256
from test_v32_native_auxiliary_fusion import fixture,repin_native


@pytest.fixture(autouse=True)
def forbid_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**kw):raise AssertionError("No optimization or model forwards during source inspection")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    import cc_hhgt.v32.multimodal_fusion as fusion
    monkeypatch.setattr(fusion,"fit_residual_fusion_model",forbidden)
    monkeypatch.setattr(fusion,"train_pair_blocked_crossfit_fusion",forbidden)
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter,"_build_model",forbidden)


def setup(root,kind):
    import cc_hhgt.v32.multimodal_fusion as fusion
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
    from scripts.run_v32_current_multimodal_fusion import main
    path,keys=fixture(root,kind)
    raw=root/"raw.parquet"
    pd.DataFrame({"pmid":["11112222"],"dataset_accession":["GSE_SYNTHETIC_AUX"],
        "patient_id":["PRIVATE_TEST_PATIENT"],"value":[1.]}).to_parquet(raw,index=False)
    if kind in {"genomic","atac"}:
        native=read(Path(read(path)["native_receipt"]["path"]));lp=Path(native["lineage_path"]);lineage=read(lp)
        if kind=="genomic":lineage["input_artifacts"].append({**record(raw),"source_role":"standardized_input"})
        else:
            ip=Path(lineage["input_lineage_audit_path"]);inputs=read(ip)
            inputs["materialization_raw_input_artifacts"]=[{**record(raw),"role":"raw_atac"}]
            write(ip,inputs);lineage["input_lineage_audit_sha256"]=sha256(ip)
        write(lp,lineage);repin_native(path)
    adapter=materialize_native_auxiliary(path,root/"adapter",expected_core_run_id="current-G2")
    primary=root/"primary.parquet"
    keys.assign(**{fusion.PRIMARY_COLUMN:.25,"analysis_version":fusion.ANALYSIS_VERSION}).to_parquet(primary,index=False)
    folds=[]
    for f in range(5):
        p=root/f"labels{f}.parquet";keys.assign(patient_fold_id=f,held_out_proxy_label=f%2).to_parquet(p,index=False)
        folds.append({"fold_id":f,"artifact":record(p)})
    parent=root/"primary.json";write(parent,{"status":"PASS","run_id":"current-G2","prediction":record(primary),
        "fold_prediction_sha256":[r["artifact"]["sha256"] for r in folds]})
    manifest=root/"fusion.json";experts=read(root/"adapter/FUSION_EXPERTS.json")
    write(manifest,{"format":"CURRENT_MULTIMODAL_FUSION_MANIFEST_V1","core_run_id":"current-G2",
        "primary_prediction":record(primary),"primary_binding_audit":record(parent),"primary_folds":folds,"experts":experts})
    assert main(["--manifest",str(manifest),"--expected-core-run-id","current-G2","--output",str(root/"prepared")])==0
    prepared_audit=read(root/"prepared/INPUT_AUDIT.json")
    assert set(prepared_audit["expert_bindings"])=={e["spec"]["expert_id"] for e in experts}
    assert read(root/"prepared/PREPARED.json")["training_started"] is False
    specs=[]
    for e in experts:
        definition=dict(e["spec"]);definition["endpoint_roles"]=tuple(definition["endpoint_roles"])
        specs.append(fusion.ExpertSpec(**definition))
    # Native runner refuses fitting unverified legacy labels. Here the actual
    # low-level serializer publishes explicitly synthetic fixed parameters;
    # no training-ready or independently validated release is claimed.
    frame,audit=fusion.build_pair_aggregated_fusion_frame_from_parquet(primary,
        [Path(r["artifact"]["path"]) for r in folds],{s:Path(e["prediction"]["path"]) for s,e in zip(specs,experts)})
    audit.update(core_run_id="current-G2",expert_bindings=prepared_audit["expert_bindings"],synthetic_only=True)
    ids=tuple(s.expert_id for s in specs);results=[]
    for endpoint in ("discovery","confidence"):
        # All experts have zero coefficients: ancestry must still be retained.
        model=fusion.ResidualFusionModel(endpoint,ids,(0.,)*len(ids),(0.,)*len(ids),1,"0"*64,"1"*64,0,0.)
        results.append(fusion.CrossfitFusionResult(endpoint,tuple(specs),fusion.apply_residual_fusion_model(model,frame),
            pd.DataFrame({"synthetic_only":[True]}),(model,)*5,model,"SYNTHETIC_NOT_ESTIMATED","SYNTHETIC_NOT_ESTIMATED"))
    sources={str(p):record(p) for p in [primary,*[Path(r["artifact"]["path"]) for r in folds],adapter,
        *[Path(e["prediction"]["path"]) for e in experts]]}
    output=fusion.materialize_multimodal_fusion_release(root/"release",frame,*results,
        source_declarations=[{**r,"source_role":"current_v32_oof_prediction","generation":"current_v32"} for r in sources.values()],
        training_run_id="V32-FUSION-AUXILIARY_SOURCES_TEST",input_audit=audit)
    spec={"model_id":"fusion","producer_kind":"fusion_release","producer_binding":record(Path(output["binding_path"])),
        "run_id":"V32-FUSION-AUXILIARY_SOURCES_TEST","score_column":"discovery_adjusted_probability"}
    artifacts=read(Path(output["binding_path"]))["artifacts"]
    assert all(artifacts[name]["public"] is False for name in ("source_inputs","module_lineage","fusion_input_audit"))
    model_input=root/"use.json";write(model_input,{"format":INPUT_FORMAT,"models":[spec]})
    return model_input,raw


@pytest.mark.parametrize("kind",["genomic","atac","cnv","cnv_event"])
def test_native_producers_and_zero_weight_experts_retain_dependencies(tmp_path,kind):
    path,raw=setup(tmp_path,kind)
    _,_,tables,_=derive_manifest(path);dependencies=tables["dependencies"];usage=tables["usage"]
    assert dependencies.role.str.contains(".checkpoint.",regex=False).any()
    assert set(dependencies.loc[dependencies.role.str.contains(".checkpoint.",regex=False),"patient_fold_id"])==set(range(5))
    assert usage.dependency_context.str.startswith("FUSION_EXPERT:").any()
    assert not usage.astype(str).apply(lambda c:c.str.contains("PRIVATE_TEST_PATIENT",regex=False)).any().any()
    assert "AUXILIARY_RAW_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED" not in set(tables["gaps"].reason)
    if kind in {"genomic","atac"}:
        assert usage.study_tokens_json.str.contains("11112222").any()
        assert (dependencies.sha256==sha256(raw)).any()
    else:assert "CNV_CONSUMED_PARTITION_ANCESTRY_NOT_RECONSTRUCTED" in set(tables["gaps"].reason)
    if kind!="atac":assert "AUXILIARY_CORE_NODE_INPUT" in set(usage.use_view)


def test_auxiliary_only_study_excludes_unseen_external_entity(tmp_path):
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    path,_=setup(tmp_path/"native","genomic");models=read(path)["models"]
    inventory=Path(materialize_current_model_use(path,tmp_path/"inventory")["binding_path"])
    binding,_=load_current_model_use(inventory,expected_sha=sha256(inventory))
    assert binding["model_use_inventory_completeness_verified"] is False
    files={}
    frames={"crosswalk":pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","breast","BRCA"),("pathway","p","PATHWAY:P1")],
        columns=["namespace","raw_id","canonical_id"]),
        "external":pd.DataFrame({"record_id":["synthetic"],"lncrna":["new"],"cancer":["breast"],"pathway":["p"],"label":[1],
            "pmid":["11112222"],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic assay"]})}
    for name,data in frames.items():
        f=tmp_path/(name+".parquet");data.to_parquet(f,index=False);files[name]=record(f)
    source=tmp_path/"external.json";write(source,{"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":models,
        "model_use_inventory":record(inventory),"identity_crosswalk":files["crosswalk"],
        "external_sources":[{"kind":"exact_assay","source_id":"auxiliary_only_study","table":files["external"]}],
        "evaluation":{"primary_model":"fusion","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}})
    result=materialize_current_external(source,tmp_path/"external")
    query=CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    rows=query.query(table="isolation")["rows"]
    assert rows and all(r["study_overlap"] and not r["lncrna_overlap"] and not r["provenance_held_out"] for r in rows)
    filtered=query.query(table="model_dependencies",filters={"dependency_context":"FUSION_EXPERT:mutation"})["rows"]
    assert filtered and all("path" not in r for r in filtered)


@pytest.mark.parametrize("fault",["source_change","snapshot_change","run_identity","unconsumed_binding"])
def test_native_source_and_declared_expert_corruption_rejected(tmp_path,fault):
    path,raw=setup(tmp_path,"atac")
    if fault=="source_change":
        pd.DataFrame({"pmid":["99999999"]}).to_parquet(raw,index=False)
        with pytest.raises(ValueError,match="SHA256 mismatch"):derive_manifest(path)
        return
    # Directly invoke the same inspector to isolate semantic validation even
    # when an attacker has repinned a new outer input-audit/file hash.
    from cc_hhgt.v32.current_auxiliary_model_use import inspect_fusion_expert
    from cc_hhgt.v32.current_model_use import Inventory
    binding_path=Path(read(path)["models"][0]["producer_binding"]["path"]);binding=read(binding_path)
    audit=read(Path(binding["artifacts"]["fusion_input_audit"]["path"]))
    frame=pd.read_parquet(binding["artifacts"]["fusion_input_private"]["path"])
    if fault=="snapshot_change":frame.loc[0,"atac_probability"]+=.2
    elif fault=="run_identity":audit["expert_bindings"]["atac"]["run_id"]="unrelated-run"
    else:binding["source_inputs"]=[]
    with pytest.raises(ValueError,match="differs|absent"):
        inspect_fusion_expert(Inventory([],tmp_path),binding_path,binding,audit,frame,"atac","fusion","discovery")
