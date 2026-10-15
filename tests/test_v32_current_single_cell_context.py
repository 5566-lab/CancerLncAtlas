"""Actual signed-SC producer and selected consumers, no real model training."""
import json
from pathlib import Path
import socket

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_single_cell_context import (INPUT_FORMAT,materialize_current_single_cell_context,
    audit_current_single_cell_context,CurrentSingleCellContextQuery,sources)
from test_v32_current_state_context import current_sources


def record(path):return {"path":str(path),"sha256":sha256(path)}
def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.write_text(json.dumps(value),encoding="utf-8");return path
def forbidden(*a,**k):raise AssertionError("No optimizer in single-cell context checks")


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


def native_fixture(root,current_sources,case="available"):
    from scripts.materialize_v32_signed_single_cell_fusion import main
    root.mkdir();selected=current_sources[1]
    candidates=pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]})
    candidates.to_parquet(root/"candidates.parquet",index=False)
    rows=[];donors=[]
    for fold in range(5):
        donors.append({"dataset_id":"S1","query_donor_id":f"d{fold}","canonical_donor_id":f"PERSON{fold}"})
        for cell,rho in (("tumour",.8),("immune",-.6),("neutral",0.)):
            rows.append({"dataset_id":"S1","cancer_id":"BRCA","cell_key":cell if case!="sparse" else cell+str(fold),
                "query_donor_id":f"d{fold}","lncrna_id":"LNC:OUTSIDE" if case=="missing" else "LNC:Z","pathway_id":"P1",
                "outer_fold_id":fold,"predicted_rho":rho,"arm":"with_lasso","rho":999})
    # Reused donor in another study remains one donor in the pooled contexts.
    if case=="available":
        donors.append({"dataset_id":"S2","query_donor_id":"alias0","canonical_donor_id":"PERSON0"})
        rows.append({**rows[0],"dataset_id":"S2","query_donor_id":"alias0"})
    pd.DataFrame(rows).to_parquet(root/"predictions.parquet",index=False)
    pd.DataFrame(donors).to_parquet(root/"crosswalk.parquet",index=False)
    validation=write(root/"validation.json",{"status":"PASS","synthetic_fixture":True,"optimizer_actually_run":False})
    pointer=write(root/"pointer.json",{"status":"PASS_ACTIVATED_SC_HHGT","estimand":"signed_within_donor_metacell_spearman_rho",
        "split_policy":"donor_blocked_outer_oof_with_nested_inner_validation","training_complete":True,"outer_folds":5,
        "primary_arm":"with_lasso","run_id":"SYNTHETIC_SIGNED_SC","oof_predictions":record(root/"predictions.parquet"),
        "independent_validation":record(validation),"synthetic_fixture":True,"optimizer_actually_run":False})
    assert main(["--current-pointer",str(pointer),"--candidates",str(root/"candidates.parquet"),
        "--donor-crosswalk",str(root/"crosswalk.parquet"),"--output",str(root/"native")])==0
    return write(root/"INPUT.json",{"format":INPUT_FORMAT,"selected_analysis":record(selected),"native_binding":record(root/"native/AUDIT.json")})


@pytest.fixture(scope="module")
def release(tmp_path_factory,current_sources):
    from scripts.materialize_v32_current_single_cell_context import main
    root=tmp_path_factory.mktemp("current_single_cell")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
        manifest=native_fixture(root/"inputs",current_sources)
        assert main(["--manifest",str(manifest),"--output",str(root/"release")])==0
    binding=root/"release/CURRENT_SINGLE_CELL_CONTEXT_BINDING.json"
    query=CurrentSingleCellContextQuery(binding,expected_sha=sha256(binding),selected_analysis_sha=sha256(current_sources[1]))
    return root,manifest,binding,query


def test_native_replay_preserves_opposite_contexts_zero_and_unique_donor_counts(release):
    _,_,binding,query=release
    context=pd.read_parquet(query.download("native_cell_context")["path"]).set_index("cell_key")
    assert context.signed_rho_mean.to_dict()==pytest.approx({"tumour":.8,"immune":-.6,"neutral":0})
    assert context.donors.eq(5).all() and context.donor_folds.eq(5).all()
    assert context.context_available.all() and context.donor_rho_sd.eq(0).all()
    dataset=pd.read_parquet(query.download("native_dataset_context")["path"])
    second=dataset.loc[dataset.dataset_id.eq("S2")].iloc[0]
    assert second.donors==1 and not second.context_available and pd.isna(second.donor_rho_sd)
    assert second.context_unavailable_reason=="INSUFFICIENT_INDEPENDENT_DONORS"
    pooled=pd.read_parquet(query.download("selected_member_support")["path"]).iloc[0]
    assert pooled.single_cell_positive_strength==pytest.approx(.8/3)
    assert pooled.single_cell_negative_strength==pytest.approx(.6/3)
    assert pooled.single_cell_donors==5 and not pooled.used_in_selected_score
    profile=pd.read_parquet(query.download("selected_geneset_context")["path"]).set_index("cell_key")
    assert profile.mean_member_signed_rho.to_dict()==pytest.approx({"tumour":.8,"immune":-.6,"neutral":0})
    assert profile.member_coverage.eq(1).all() and profile.selected_weight_coverage.eq(1).all()
    for name in query.paths:
        columns=pd.read_parquet(query.download(name)["path"]).columns
        assert not {"patient_id","query_donor_id","canonical_donor_id","rho"}&set(columns)
    result=audit_current_single_cell_context(binding,expected_sha=sha256(binding))
    assert result["native_aggregation_replayed_in_this_audit"] and not result["independent_biological_validation"]


@pytest.mark.parametrize("case",["missing","sparse"])
def test_missing_or_insufficient_contexts_keep_nulls_not_false_neutrality(tmp_path,current_sources,case):
    manifest=native_fixture(tmp_path/"inputs",current_sources,case=case)
    result=materialize_current_single_cell_context(manifest,tmp_path/"release")
    query=CurrentSingleCellContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha=sha256(current_sources[1]))
    profile=pd.read_parquet(query.download("selected_geneset_context")["path"])
    assert not profile.summary_available.any() and profile.mean_member_signed_rho.isna().all()
    assert profile.member_coverage.eq(0).all() and profile.selected_weight_coverage.eq(0).all()
    if case=="missing":
        assert len(profile)==1 and profile.cell_key.isna().all()
        assert profile.unavailable_reason.eq("NO_DONOR_OOF_ROWS").all()
        support=pd.read_parquet(query.download("selected_member_support")["path"])
        assert support.single_cell_positive_strength.isna().all() and not support.single_cell_available.any()
    else:
        assert len(profile)==15 and profile.unavailable_reason.eq("NO_SUFFICIENT_DONOR_MEMBER").all()


def test_rehashed_native_aggregate_cannot_override_actual_donor_rows(tmp_path,current_sources):
    manifest=native_fixture(tmp_path/"inputs",current_sources)
    spec=read(manifest); native_path=Path(spec["native_binding"]["path"]);native=read(native_path)
    target=Path(native["outputs"]["cell_context_summary"]["path"])
    table=pq.ParquetFile(target).read();index=table.schema.get_field_index("signed_rho_mean")
    table=table.set_column(index,table.schema.field(index),pa.array([.99]*len(table),type=table.column(index).type))
    pq.write_table(table,target)
    native["outputs"]["cell_context_summary"].update(record(target));write(native_path,native)
    write(native_path.parent/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(native_path)})
    spec["native_binding"]=record(native_path);write(manifest,spec)
    with pytest.raises(ValueError,match="replay differs"):
        materialize_current_single_cell_context(manifest,tmp_path/"bad")
    assert not (tmp_path/"bad/SUCCESS.json").exists()


def test_wrong_candidate_universe_and_selected_identity_fail(release,tmp_path,current_sources):
    _,_,binding,_=release
    with pytest.raises(ValueError,match="different selected"):
        CurrentSingleCellContextQuery(binding,expected_sha=sha256(binding),selected_analysis_sha="f"*64)
    manifest=native_fixture(tmp_path/"inputs",current_sources)
    spec=read(manifest);native_path=Path(spec["native_binding"]["path"]);native=read(native_path)
    candidates=Path(native["inputs"]["candidates"]["path"])
    pd.DataFrame({"cancer_id":["LUAD"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]}).to_parquet(candidates,index=False)
    native["inputs"]["candidates"]=record(candidates);write(native_path,native)
    write(native_path.parent/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(native_path)})
    spec["native_binding"]=record(native_path);write(manifest,spec)
    with pytest.raises(ValueError,match="candidate universe"):
        sources(manifest,replay=True)


def test_public_profile_replay_rejects_rehashed_sign_change(release,tmp_path):
    _,_,original,_=release;binding=read(original)
    source=Path(binding["outputs"]["selected_geneset_context"]["path"])
    table=pq.ParquetFile(source).read();index=table.schema.get_field_index("mean_member_signed_rho")
    table=table.set_column(index,table.schema.field(index),pa.array([-v for v in table.column(index).to_pylist()],type=table.column(index).type))
    changed=tmp_path/"profile.parquet";pq.write_table(table,changed)
    binding["outputs"]["selected_geneset_context"].update(record(changed))
    bad=write(tmp_path/"BINDING.json",binding)
    with pytest.raises(ValueError,match="replay differs"):
        audit_current_single_cell_context(bad,expected_sha=sha256(bad))
