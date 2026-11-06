"""Vectorized export must reproduce frozen predictions without refitting."""
import importlib.util,io,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,pytest
from cla_engine.common import sha
from cc_hhgt import network_ridge_predict as slow
from cc_hhgt import network_ridge_predict_fast as fast

def check_case(tmp_path,result):
    src=tmp_path/"input/COAD";net=tmp_path/"net/COAD";final=result/"COAD/final"
    blocks={b:pd.read_parquet(src/(b+".parquet")).to_numpy(float) for b in ("RNA","CNV","MUTATION","PROTEIN","PHOSPHO")}
    cov=pd.read_parquet(src/"COVARIATES.parquet")
    fmap=pd.read_parquet(net/"FEATURE_MAP.parquet");ft=pd.read_csv(src/"RNA_FEATURES.tsv",sep="\t")
    models=joblib.load(final/"MODELS.joblib")
    pre={p:joblib.load(final/("PREPROCESS_purity"+str(int(p))+".joblib")) for p in (False,True)}
    checked=0
    for key,model in models.items():
        if model["kind"]=="legacy":
            with pytest.raises(ValueError,match="USE_LEGACY_LAPLACIAN_KERNEL"):
                fast.coefficients(model,pre[model["purity"]],blocks,fmap,ft)
            continue
        frozen=pre[model["purity"]]
        a=slow.coefficients(model,frozen,blocks,fmap,ft);b=fast.coefficients(model,frozen,blocks,fmap,ft)
        assert set(a)==set(b)
        for block in a:
            assert a[block]["feature_ids"]==b[block]["feature_ids"]
            for field in ("selector","feature_positions"):
                np.testing.assert_array_equal(a[block][field],b[block][field])
            np.testing.assert_allclose(a[block]["standardized_coefficients"],b[block]["standardized_coefficients"],rtol=1e-13,atol=1e-13)
        expected,ok=slow.predict(model,frozen,blocks,blocks,cov,fmap,ft)
        compiled=fast.compile_model(model,frozen,blocks,fmap,ft)
        buffer=io.BytesIO();joblib.dump(compiled,buffer);buffer.seek(0);compiled=joblib.load(buffer)
        got,supported=fast.predict_compiled(compiled,blocks,cov)
        np.testing.assert_array_equal(ok,supported)
        np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
        order=np.array([13,2,8])
        query={block:raw[order] for block,raw in blocks.items()}
        shuffled,ok=fast.predict_compiled(compiled,query,cov.iloc[order])
        np.testing.assert_allclose(shuffled,expected[order],rtol=1e-12,atol=1e-12,equal_nan=True)
        bad=cov.iloc[order].copy();bad.iloc[0,bad.columns.get_loc("age_years")]=np.nan
        pred,supported=fast.predict_compiled(compiled,query,bad)
        assert not supported[0] and np.isnan(pred[0])
        if model["kind"]=="module" and model["mode"]=="union" and model["arm"]=="combined":
            wrong=fmap.copy();wrong.loc[wrong.block.eq("RNA")&wrong.position.eq(0),"module_union"]=99999
            with pytest.raises(ValueError,match="FROZEN_GROUP_MEMBERSHIP_DRIFT"):
                fast.coefficients(model,frozen,blocks,wrong,ft)
        checked+=1
    assert checked==136
    return pre

def test_vectorized_and_compiled_export_match_frozen_api_with_missing_assays(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("existing_network_fixture",root/"tests/test_network_ridge.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.test_end_to_end_nested_outer_and_final(tmp_path)
    check_case(tmp_path,tmp_path/"out")
    src=tmp_path/"input/COAD"
    for block in ("PROTEIN","PHOSPHO"):
        data=pd.read_parquet(src/(block+".parquet"));data.iloc[:7,:]=np.nan
        data.to_parquet(src/(block+".parquet"))
    data=pd.read_parquet(src/"CNV.parquet");data.iloc[:,2]=0.;data.to_parquet(src/"CNV.parquet")
    data=pd.read_parquet(src/"MUTATION.parquet");data.iloc[:,3]=np.nan;data.to_parquet(src/"MUTATION.parquet")
    data=pd.read_parquet(src/"RNA.parquet");data.iloc[:3,6]=np.nan;data.to_parquet(src/"RNA.parquet")
    (src/"COMPLETE.json").write_text(json.dumps(dict(files={f.name:sha(f) for f in src.iterdir() if f.name!="COMPLETE.json"})))
    from scripts.run_network_validation import run
    run(src.parent,tmp_path/"net",tmp_path/"missing_out","COAD","final")
    pre=check_case(tmp_path,tmp_path/"missing_out")
    assert any((p["preprocess"]["PROTEIN"]["index"]>=p["preprocess"]["PROTEIN"]["original_width"]).any() for p in pre.values())

    from scripts.audit_network_compiled_models import audit_compiled
    audit_compiled(tmp_path/"input",tmp_path/"net",tmp_path/"missing_out",tmp_path/"compiled_audit","COAD")
    receipt=json.loads((tmp_path/"compiled_audit/COAD/COMPLETE.json").read_text())
    assert receipt["status"]=="PASS" and receipt["models_checked"]==88
    assert receipt["compiled_export_calls"]==dict(compiled_models=88,prediction_calls=264,coefficient_calls=88)
