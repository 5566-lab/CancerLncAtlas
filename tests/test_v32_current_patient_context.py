"""Native synthetic patient analyses and their current selected consumers."""
import json
from pathlib import Path
import socket

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_patient_context import (INPUT_FORMAT,materialize_current_patient_context,
    audit_current_patient_context,CurrentPatientContextQuery,sources)
from test_v32_current_state_context import current_sources


def record(path):return {"path":str(path),"sha256":sha256(path)}
def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.write_text(json.dumps(value),encoding="utf-8");return path
def forbidden(*a,**k):raise AssertionError("No neural optimizer in patient context checks")


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


def native_fixture(root,current_sources):
    from scripts.analyze_v32_patient_heterogeneity import main
    from test_v32_patient_heterogeneity import fixture
    root.mkdir();analyses=[]
    for name,mutation,sparse in (("declared",False,False),("mutation",True,True)):
        folder=root/name;folder.mkdir();data=fixture();data["lncrna_id"]="LNC:Z";data["group_id"]=name
        if sparse:
            data["group_value"]=0.;data.loc[0,"group_value"]=1.
        keys=["cancer_id","patient_id"]
        groups=data[keys+["lncrna_id","group_id","group_value","group_available","group_unavailable_reason"]]
        if mutation:
            groups=data[keys+["lncrna_id"]].assign(variants=data.group_value.astype(int),assayed=True,callable=True)
            groups.loc[1,"callable"]=False  # no report without callable locus stays unknown
        frames={"expression":data[keys+["lncrna_id","lncrna_expression"]],"activity":data[keys+["pathway_id","activity_score"]],
            "groups":groups,"covariates":data[keys+["lncrna_id","local_cnv_log2","local_cnv_log2__available"]],
            "folds":data[keys+["patient_fold_id"]],"candidates":pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]})}
        args=[]
        for label,frame in frames.items():
            path=folder/(label+".parquet");frame.to_parquet(path,index=False);args += ["--"+label,str(path)]
        args += ["--group-id",name,"--nuisance-columns","local_cnv_log2","--candidate-selection","exploratory_current_core",
            "--output",str(folder/"native")]
        if mutation:args += ["--groups-are-mutation-calls","--mutation-count-column","variants",
            "--mutation-assay-column","assayed","--locus-callable-column","callable"]
        assert main(args)==0
        analyses.append({"source_id":name,"native_binding":record(folder/"native/AUDIT.json")})
    return write(root/"INPUT.json",{"format":INPUT_FORMAT,"selected_analysis":record(current_sources[1]),"analyses":analyses})


@pytest.fixture(scope="module")
def release(tmp_path_factory,current_sources):
    from scripts.materialize_v32_current_patient_context import main
    root=tmp_path_factory.mktemp("current_patient")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
        manifest=native_fixture(root/"inputs",current_sources)
        assert main(["--manifest",str(manifest),"--output",str(root/"release")])==0
    binding=root/"release/CURRENT_PATIENT_CONTEXT_BINDING.json"
    query=CurrentPatientContextQuery(binding,expected_sha=sha256(binding),selected_analysis_sha=sha256(current_sources[1]))
    return root,manifest,binding,query


def test_native_patient_statistics_reach_members_with_original_uncertainty_and_fdr(release):
    _,_,binding,query=release
    table=pd.read_parquet(query.download("selected_member_interactions")["path"]).set_index("analysis_source_id")
    observed=table.loc["declared"]
    assert observed.group0_slope==pytest.approx(1,abs=.1) and observed.group1_slope==pytest.approx(-1.5,abs=.1)
    assert observed.slope_difference_ci_high<0 and observed.interaction_fdr<1e-12
    assert observed.candidate_selection=="exploratory_current_core" and not observed.joint_geneset_interaction_estimated
    missing=table.loc["mutation"]
    assert not missing.inference_available and pd.isna(missing.slope_difference) and pd.isna(missing.interaction_fdr)
    assert missing.group_unknown_patients==1 and missing.inference_reason=="INSUFFICIENT_GROUP_SUPPORT"
    native=pd.read_parquet(query.download("native_interactions")["path"]).set_index("analysis_source_id")
    pd.testing.assert_series_equal(table.interaction_fdr,native.interaction_fdr)
    metrics=pd.read_parquet(query.download("selected_member_oof_metrics")["path"])
    assert metrics.loc[metrics.analysis_source_id.eq("declared"),"delta_mse"].gt(0).all()
    coverage=pd.read_parquet(query.download("selected_geneset_coverage")["path"]).set_index("analysis_source_id")
    assert coverage.loc["declared","opposite_group_direction_members"]==1
    assert coverage.loc["mutation","inference_member_coverage"]==0
    for name in query.paths:
        assert "patient_id" not in pq.ParquetFile(query.download(name)["path"]).schema_arrow.names
    assert audit_current_patient_context(binding,expected_sha=sha256(binding))["native_replayed_in_this_audit"]


def test_rehashed_native_statistics_cannot_replace_source_replay(tmp_path,current_sources):
    manifest=native_fixture(tmp_path/"inputs",current_sources)
    spec=read(manifest);source=Path(spec["analyses"][0]["native_binding"]["path"]);native=read(source)
    target=Path(native["outputs"]["INTERACTION_SUMMARY.parquet"]["path"])
    table=pq.ParquetFile(target).read();index=table.schema.get_field_index("slope_difference")
    table=table.set_column(index,table.schema.field(index),pa.array([99.],type=table.column(index).type));pq.write_table(table,target)
    native["outputs"]["INTERACTION_SUMMARY.parquet"]=record(target);write(source,native)
    write(source.parent/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(source)})
    spec["analyses"][0]["native_binding"]=record(source);write(manifest,spec)
    with pytest.raises(ValueError,match="replay differs"):
        materialize_current_patient_context(manifest,tmp_path/"bad")
    assert not (tmp_path/"bad/SUCCESS.json").exists()


def test_missing_group_policy_is_not_guessed_and_source_changes_are_rejected(tmp_path,current_sources):
    manifest=native_fixture(tmp_path/"inputs",current_sources)
    spec=read(manifest);source=Path(spec["analyses"][0]["native_binding"]["path"]);native=read(source)
    policy=native.pop("group_source_policy");write(source,native)
    write(source.parent/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(source)})
    spec["analyses"][0]["native_binding"]=record(source);write(manifest,spec)
    with pytest.raises(ValueError,match="group-source policy"):sources(manifest,replay=True)
    native["group_source_policy"]=policy;write(source,native)
    write(source.parent/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(source)})
    spec["analyses"][0]["native_binding"]=record(source);write(manifest,spec)
    changed=Path(native["sources"]["activity"]["path"]);changed.write_bytes(changed.read_bytes()+b"changed")
    with pytest.raises(ValueError):sources(manifest,replay=True)


def test_selected_identity_and_analysis_inventory_are_enforced(release,tmp_path):
    _,manifest,binding,_=release
    with pytest.raises(ValueError,match="different selected"):
        CurrentPatientContextQuery(binding,expected_sha=sha256(binding),selected_analysis_sha="f"*64)
    spec=read(manifest);spec["analyses"].append(spec["analyses"][0])
    with pytest.raises(ValueError,match="unique stable"):
        sources(write(tmp_path/"duplicate.json",spec),replay=False)
