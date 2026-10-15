"""Native genomic serializers and nested distal statistics, synthetic only."""
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_omics_context import (INPUT_FORMAT,KINDS,sources,materialize_current_omics_context,
    audit_current_omics_context,CurrentOmicsContextQuery)
from test_v32_current_state_context import current_sources


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value),encoding="utf-8");return path
def record(path):return {"path":str(path),"sha256":sha256(path)}
def forbidden(*a,**k):raise AssertionError("No neural optimization in omics context verification")


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


def genomic_fixture(root,current_sources):
    from test_v32_native_auxiliary_fusion import fixture
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
    root.mkdir();bindings={}
    for kind in KINDS:
        path,_=fixture(root/kind,kind=kind,current_sources=current_sources,unavailable=kind=="cnv")
        audit=materialize_native_auxiliary(path,root/(kind+"_native"),expected_core_run_id=current_sources[2])
        bindings[kind]={"status":"AVAILABLE","binding":record(audit)}
    return write(root/"INPUT.json",{"format":INPUT_FORMAT,"kind":"genomic",
        "selected_analysis":record(current_sources[1]),"native_bindings":bindings})


def distal_fixture(root,current_sources,*,missing=False):
    from scripts.run_v32_distal_regulatory_mutation_head import main
    from cc_hhgt.v32.distal_topology import POLICY
    root.mkdir();prepared=root/"prepared";activity=root/"activity"
    (prepared/"cancer_id=BRCA").mkdir(parents=True);(activity/"cancer_id=BRCA").mkdir(parents=True)
    i=np.arange(60);mutation=(i%7).astype(float);cnv=np.cos(i*.7)*.2
    expression=1.8*mutation+.4*cnv+.07*np.sin(i)
    patients=[f"TCGA-AA-{n:04}" for n in i]
    data=pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"lncrna_id":"LNC:Z","patient_fold_id":i%5,
        "lncrna_expression":expression,"distal_mutation_burden":mutation,"distal_mutation_burden__available":True,
        "atac_distal_accessibility":.2+.1*np.cos(i),"atac_distal_accessibility__available":True,
        "promoter_methylation_beta":.4+.04*np.sin(i*.8),"promoter_methylation_beta__available":True,
        "distal_methylation_beta":np.nan,"distal_methylation_beta__available":False,
        "local_cnv_log2":cnv,"local_cnv_log2__available":True,"tumor_purity":.65+.1*np.sin(i*.3),"tumor_purity__available":True})
    if missing:data["distal_mutation_burden"]=np.nan;data["distal_mutation_burden__available"]=False
    input_path=prepared/"cancer_id=BRCA/part-0.parquet";data.to_parquet(input_path,index=False)
    activity_path=activity/"cancer_id=BRCA/part-0.parquet"
    pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"pathway_id":"P1","activity_score":-2*expression+.01*np.cos(i)}).to_parquet(activity_path,index=False)
    candidate=root/"candidates.parquet"
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidate,index=False)
    topology=root/"synthetic_topology.json";write(topology,{"synthetic":True,"outcome_selected":False})
    write(prepared/"AUDIT.json",{"format":"CANCERLNCATLAS_V32_DISTAL_REGULATORY_HEAD_INPUT_V1","status":"PASS","target_host":"149",
        "topology_lineage":{"predictive_evaluation_eligible":True,"policy":POLICY,"coordinate_build":"GRCh38","links_sha256":sha256(topology)},
        "outputs":[record(input_path)],"synthetic_prepared_input_fixture":True})
    assert main(["--head-input-root",str(prepared),"--pathway-activity-root",str(activity),"--candidate-universe",str(candidate),
        "--output",str(root/"native"),"--min-expression-train-patients","15","--min-pathway-train-patients","15"])==0
    return write(root/"INPUT.json",{"format":INPUT_FORMAT,"kind":"distal","selected_analysis":record(current_sources[1]),
        "native_binding":record(root/"native/AUDIT.json")})


@pytest.fixture(scope="module")
def releases(tmp_path_factory,current_sources):
    from scripts.materialize_v32_current_omics_context import main
    root=tmp_path_factory.mktemp("current_omics");results={}
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
        for kind,fixture in (("genomic",genomic_fixture),("distal",distal_fixture)):
            path=fixture(root/(kind+"_inputs"),current_sources)
            assert main(["--manifest",str(path),"--output",str(root/kind)])==0
            bp=root/kind/"CURRENT_OMICS_CONTEXT_BINDING.json"
            query=CurrentOmicsContextQuery(bp,expected_sha=sha256(bp),selected_analysis_sha=sha256(current_sources[1]),kind=kind)
            results[kind]=(path,bp,query)
    return results


def test_distinct_native_genomic_estimands_and_folds_reach_selected_members(releases):
    _,bp,query=releases["genomic"]
    table=pd.read_parquet(query.download("selected_member_genomic")["path"])
    assert len(table)==5 and set(table.producer_kind)==set(KINDS)
    cnv=table.loc[table.modality.eq("cnv")].set_index("producer_kind")
    assert cnv.loc["genomic","native_estimand"]!=cnv.loc["cnv","native_estimand"]
    assert cnv.loc["cnv_event","expert_id"]=="cnv_event" and cnv.loc["cnv","expert_id"]=="cnv"
    assert not cnv.loc["cnv","available"] and pd.isna(cnv.loc["cnv","probability"])
    assert table.loc[table.available,"folds_available"].eq(4).all() and not table.used_in_selected_score.any()
    folds=pd.read_parquet(query.download("native_folds")["path"])
    assert len(folds)==25 and set(folds.patient_fold_id)==set(range(5))
    assert folds.loc[folds.available & folds.patient_fold_id.eq(0),"probability"].eq(0).all()
    assert "unused_observed_label" not in folds
    profile=pd.read_parquet(query.download("selected_geneset_genomic")["path"])
    assert profile.loc[profile.producer_kind.eq("cnv"),"mean_member_probability"].isna().all()
    assert profile.loc[profile.producer_kind.eq("cnv"),"member_coverage"].eq(0).all()
    assert audit_current_omics_context(bp,expected_sha=sha256(bp))["native_replayed_in_this_audit"]


def test_actual_nested_distal_statistics_and_missing_assay_coverage_reach_members(releases):
    _,bp,query=releases["distal"]
    member=pd.read_parquet(query.download("selected_member_distal")["path"]).iloc[0]
    assert member.distal_regulatory_mutation_pathway_available and member.mean_oof_slope<0
    assert member.oof_delta_mse>0 and member.oof_prediction_correlation>.9
    assert not member.joint_geneset_effect_estimated and not member.causal_effect_claimed
    coverage=pd.read_parquet(query.download("native_feature_coverage")["path"]).set_index("feature")
    assert coverage.loc["distal_methylation_beta","available_patients"]==0
    assert coverage.loc["promoter_methylation_beta","available_patients"]==60
    assert "patient_id" not in coverage.columns
    for name in query.paths:
        assert not {"patient_id","outer_fold_id"}&set(pq.ParquetFile(query.download(name)["path"]).schema_arrow.names)
    assert audit_current_omics_context(bp,expected_sha=sha256(bp))["native_replayed_in_this_audit"]


def test_unavailable_distal_signal_is_retained_without_fabricated_effects(tmp_path,current_sources):
    path=distal_fixture(tmp_path/"inputs",current_sources,missing=True)
    result=materialize_current_omics_context(path,tmp_path/"release")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha=sha256(current_sources[1]),kind="distal")
    table=pd.read_parquet(query.download("selected_member_distal")["path"])
    assert not table.distal_regulatory_mutation_pathway_available.any() and table.mean_oof_slope.isna().all()
    profile=pd.read_parquet(query.download("selected_geneset_distal_coverage")["path"])
    assert profile.member_coverage.eq(0).all() and not profile.summary_available.any()


def test_rehashed_distal_effect_is_rejected_by_actual_command_replay(tmp_path,current_sources):
    path=distal_fixture(tmp_path/"inputs",current_sources);manifest=read(path)
    native_path=Path(manifest["native_binding"]["path"]);native=read(native_path)
    item=next(r for r in native["artifacts"] if "candidate_oof_pathway_head" in r["path"])
    target=Path(item["path"]);table=pq.ParquetFile(target).read();index=table.schema.get_field_index("mean_oof_slope")
    table=table.set_column(index,table.schema.field(index),pa.array([99.],type=table.column(index).type));pq.write_table(table,target)
    item.update(record(target));write(native_path,native)
    write(native_path.parent/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(native_path)})
    manifest["native_binding"]=record(native_path);write(path,manifest)
    with pytest.raises(ValueError,match="replay differs"):materialize_current_omics_context(path,tmp_path/"bad")
    assert not (tmp_path/"bad/SUCCESS.json").exists()


def test_explicit_genomic_absence_and_selected_identity_are_enforced(releases,tmp_path,current_sources):
    path,bp,_=releases["genomic"];manifest=read(path)
    manifest["native_bindings"]["atac"]={"status":"NOT_PRODUCED","reason":"Synthetic missing source"}
    result=materialize_current_omics_context(write(tmp_path/"INPUT.json",manifest),tmp_path/"release")
    query=CurrentOmicsContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha=sha256(current_sources[1]),kind="genomic")
    catalog=pd.read_parquet(query.download("producer_catalog")["path"])
    assert catalog.set_index("producer_kind").loc["atac","status"]=="NOT_PRODUCED"
    assert "atac" not in set(pd.read_parquet(query.download("native_folds")["path"]).producer_kind)
    with pytest.raises(ValueError,match="different selected"):
        CurrentOmicsContextQuery(bp,expected_sha=sha256(bp),selected_analysis_sha="f"*64,kind="genomic")
    del manifest["native_bindings"]["atac"]
    with pytest.raises(ValueError,match="every native producer status"):sources(write(tmp_path/"missing.json",manifest))


def test_unbound_distal_input_cannot_be_presented_as_current(releases,tmp_path):
    path,_,_=releases["distal"];manifest=read(path);native=read(manifest["native_binding"]["path"])
    native.pop("implementation_sha256");bp=write(tmp_path/"AUDIT.json",native)
    write(tmp_path/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(bp)})
    manifest["native_binding"]=record(bp)
    with pytest.raises(ValueError,match="recorded implementation"):sources(write(tmp_path/"INPUT.json",manifest))
