import numpy as np
import pandas as pd
import pytest
from cc_hhgt import m0_no_screen as ns

def test_failure_table_keeps_zero_signatures_and_unavailable_targets():
    audits={}
    for arm in ["lncrna","mrna","combined"]:
        rows=[]
        for tid in ["ok","empty","partial"]:
            for phase in [f"outer{k}" for k in range(5)]+["final"]:
                fail=tid=="empty" or (tid=="partial" and phase=="outer2")
                rows.append(dict(target_id=tid,phase=phase,
                    status=ns.FAIL_ZERO if fail else "FITTED",n_selected=0 if fail else 1))
        audits[arm]=pd.DataFrame(rows)
    s=ns.success_status(audits,["ok","empty","partial","absent"])
    assert s.loc[s.target_id.eq("ok"),"training_success"].all()
    assert not s.loc[s.target_id.isin(["empty","partial","absent"]),"training_success"].any()
    assert s.loc[s.target_id.eq("empty"),"zero_rna_outer_folds"].eq(5).all()
    assert s.loc[s.target_id.eq("partial"),"successful_outer_folds"].eq(4).all()
    assert not s.loc[s.target_id.eq("absent"),"attempted"].any()

def test_empty_rna_cannot_be_silently_counted_as_fitted():
    a=pd.DataFrame([dict(target_id="p",phase="outer0",status="FITTED",n_selected=0)])
    with pytest.raises(ValueError,match="Empty RNA signature"):
        ns.success_status({"lncrna":a},["p"])

@pytest.mark.parametrize("variable", [True,False])
def test_zero_rna_phase_does_not_export_success_predictions_or_models(tmp_path,variable):
    import sys,joblib
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_m0_three_arm import phase_fit
    rng=np.random.default_rng(2026);n=48
    c=pd.DataFrame(dict(cohort=["CPTAC"]*n,age_years=rng.uniform(30,80,n),
        sex=np.tile(["male","female"],24),stage=np.tile(["I","II","III"],16),
        purity=rng.uniform(.3,.9,n)))
    c.index=["P"+str(k) for k in range(n)]
    x=rng.normal(size=(n,8)) if variable else np.zeros((n,8))
    y=(x[:,0]+rng.normal(size=n))[:,None]
    out=tmp_path/"phase";out.mkdir()
    a=phase_fit(x,y,c,["PROTEIN_PATHWAY|P"],None,None,
        dict(train=list(range(36)),test=list(range(36,48))),None,np.array([1.]),
        np.array(["LNC:"+str(k) for k in range(8)]),"final","lncrna",out,1,True)
    assert a.status.iloc[0] in [ns.FAIL_ZERO,"FAILED_NO_VARIABLE_RNA_OUTER"]
    assert a.n_selected.iloc[0]==0
    assert pd.read_parquet(out/"PREDICTIONS.parquet").empty
    assert pd.read_parquet(out/"SELECTED_GENES.parquet").empty
    assert joblib.load(out/"MODELS.joblib")=={}

def test_training_quality_context_does_not_use_y_or_heldout_rows():
    rng=np.random.default_rng(405);n=48
    c=pd.DataFrame(dict(cohort=["CPTAC"]*n,age_years=rng.uniform(30,80,n),
        sex=np.tile(["male","female"],24),stage=np.tile(["I","II","III"],16),
        purity=rng.uniform(.3,.9,n)))
    x=rng.normal(size=(n,10));x[:,2]=1
    ns.G=dict(x=x,cohort=c.cohort.to_numpy(),covariates=c)
    ns._CONTEXTS.clear();a=ns.context(np.arange(30),True)
    ns.G["x"][30:]=100000
    ns._CONTEXTS.clear();b=ns.context(np.arange(30),True)
    np.testing.assert_array_equal(a["z"],b["z"])
    assert 2 not in a["feature_index"]
    assert len(a["feature_index"])==9
    ns.G={};ns._CONTEXTS.clear()

def test_all_failed_run_retains_failures_and_has_no_performance(tmp_path):
    import joblib
    from cc_hhgt.m0_three_arm import ARMS
    tid="PROTEIN_PATHWAY|P"
    for arm in ARMS:
        for phase in [f"outer{k}" for k in range(5)]+["final"]:
            p=tmp_path/phase/arm;p.mkdir(parents=True)
            pd.DataFrame([dict(target_id=tid,phase=phase,arm=arm,status=ns.FAIL_ZERO,n_selected=0)]).to_parquet(p/"AUDIT.parquet")
            pd.DataFrame(columns=["patient_id","cohort","target_id","observed","predicted","train_mean_baseline","covariate_only_baseline","phase"]).to_parquet(p/"PREDICTIONS.parquet")
            pd.DataFrame(columns=["target_id","feature_id","coefficient","screening_rho","screening_fdr","phase","arm"]).to_parquet(p/"SELECTED_GENES.parquet")
            joblib.dump({},p/"MODELS.joblib")
    meta=pd.DataFrame([dict(target_id=tid,layer="PROTEIN_PATHWAY",cohort="CPTAC")])
    features=pd.DataFrame([dict(feature_id="LNC:X",gene_id="X",feature_type="lncrna")])
    result=ns.summarize(tmp_path,meta,features)
    assert result["n_complete_fivefold_targets"]==0
    assert result["failed_by_arm"]==dict.fromkeys(ARMS,1)
    scores=pd.read_csv(tmp_path/"TARGET_COMPARISON.tsv",sep="\t")
    assert scores.r2_combined.isna().all()
    assert pd.read_csv(tmp_path/"FINAL_SIGNATURES.tsv.gz",sep="\t").empty
