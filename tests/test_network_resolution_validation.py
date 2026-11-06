"""Numerical compatibility of the fine-resolution runner with the frozen solver."""
import importlib.util,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import sha

def test_resolution_runner_matches_frozen_solver_and_zero_power(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("network_fixture_source",root/"tests/test_network_ridge.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.test_end_to_end_nested_outer_and_final(tmp_path)
    from scripts import run_network_resolution_validation as fine
    from scripts import audit_network_final_models as export
    net=tmp_path/"fine_net/COAD"
    shutil.copytree(tmp_path/"net/COAD",net)
    policy={"resolution":4,"fitted_modes":list(fine.FINE_MODES),"synthetic_identical_partition_for_solver_equivalence":True}
    (net/"PROVENANCE.json").write_text(json.dumps(policy))
    def receipt(folder):
        (folder/"COMPLETE.json").write_text(json.dumps({"files":{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!="COMPLETE.json"}}))
    receipt(net)
    for phase in ("repeat0_outer0","final"):
        fine.run(tmp_path/"input",net.parent,tmp_path/"fine_out","COAD",phase)
        old=pd.read_parquet(tmp_path/"out/COAD"/phase/"AUDIT.parquet")
        new=pd.read_parquet(tmp_path/"fine_out/COAD"/phase/"AUDIT.parquet")
        assert len(new)==72 and new.status.eq("FITTED").all()
        old.variant=old.variant.str.replace("module_ridge_","module_resolution4_",regex=False)
        for field in ("parameter","lam","inner_mse","n_train","n_test"):
            a=new.set_index(["target_id","variant","arm"])[field].sort_index()
            b=old.set_index(["target_id","variant","arm"])[field].reindex(a.index)
            np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9)
        if phase!="final":
            oldp=pd.read_parquet(tmp_path/"out/COAD"/phase/"PREDICTIONS.parquet")
            newp=pd.read_parquet(tmp_path/"fine_out/COAD"/phase/"PREDICTIONS.parquet")
            oldp.variant=oldp.variant.str.replace("module_ridge_","module_resolution4_",regex=False)
            keys=["patient_id","target_id","variant","arm"]
            a=newp.set_index(keys)[["observed","predicted"]].sort_index()
            b=oldp.set_index(keys)[["observed","predicted"]].reindex(a.index)
            np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-8)
    monkeypatch.setattr(export,"VARIANTS",tuple(fine.variant_name(m) for m in fine.FINE_MODES))
    export.audit_cancer(tmp_path/"input",net.parent,tmp_path/"fine_out",tmp_path/"fine_export_audit","COAD")
    r=json.loads((tmp_path/"fine_export_audit/COAD/COMPLETE.json").read_text())
    assert r["status"]=="PASS" and r["models_checked"]==72
    monkeypatch.setattr(fine,"POWERS",(0.,))
    fine.run(tmp_path/"input",net.parent,tmp_path/"zero","COAD","repeat0_outer0")
    zero=pd.read_parquet(tmp_path/"zero/COAD/repeat0_outer0/PREDICTIONS.parquet")
    original=pd.read_parquet(tmp_path/"out/COAD/repeat0_outer0/PREDICTIONS.parquet")
    keys=["patient_id","target_id","arm"]
    baseline=original[original.variant.eq("ridge_equal")].set_index(keys).predicted
    for variant,part in zero.groupby("variant"):
        a=part.set_index(keys).predicted.sort_index()
        np.testing.assert_allclose(a,baseline.reindex(a.index),rtol=1e-8,atol=1e-8)
    policy["resolution"]=1
    (net/"PROVENANCE.json").write_text(json.dumps(policy));receipt(net)
    with pytest.raises(ValueError,match="RESOLUTION4_PARTITION_REQUIRED"):
        fine.run(tmp_path/"input",net.parent,tmp_path/"rejected","COAD","repeat0_outer0")
