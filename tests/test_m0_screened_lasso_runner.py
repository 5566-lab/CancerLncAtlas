import importlib,sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from cla_engine.common import sha
runner=importlib.import_module("run_m0_screened_lasso")

def test_demo_exports_replayable_fixed_signatures_and_complete_masked_oof(tmp_path):
    rng=np.random.default_rng(6);n=125;p=15
    x=rng.normal(size=(n,p));y=2*x[:,0]-x[:,3]+rng.normal(size=n)*.2
    yy=np.c_[y,1+x[:,2]+rng.normal(size=n)*.3,np.full(n,np.nan)]
    yy[::9,1]=np.nan;x[::11,3]=np.nan
    ids=["P"+str(i) for i in range(n)];features=["LNC:"+str(i) for i in range(p)]
    targets=["RNA_PATHWAY|A","STATE|B","STATE|unavailable"]
    src=tmp_path/"inputs"/"TCGA_DEMO";src.mkdir(parents=True)
    pd.DataFrame(x,index=ids,columns=features).to_parquet(src/"X.parquet")
    pd.DataFrame(yy,index=ids,columns=targets).to_parquet(src/"Y.parquet")
    pd.DataFrame({"patient_id":ids,"outer_fold":np.arange(n)%5}).to_csv(src/"PATIENT_FOLDS.tsv",sep="\t",index=False)
    pd.DataFrame({"lncrna_id":features,"gene_symbol":features}).to_csv(src/"FEATURES.tsv",sep="\t",index=False)
    pd.DataFrame({"target_id":targets,"native_target_id":targets,"layer":["RNA_PATHWAY","STATE","STATE"],
                  "family":["RNA_PATHWAY","STATE","STATE"],"method":["synthetic"]*3}).to_csv(src/"TARGETS.tsv",sep="\t",index=False)
    info={"dataset":"TCGA","cohort":"DEMO","cancer_id":"DEMO","n_patients":n,"n_features":p}
    (src/"INPUTS.json").write_text(json.dumps({"info":info,"sources":[]}))
    (src/"COMPLETE.json").write_text(json.dumps({"status":"COMPLETE","target_host":"149","info":info,
        "files":{f.name:sha(f) for f in src.iterdir() if f.is_file()}}))
    summary=runner.run_case(src.parent,"TCGA_DEMO",tmp_path/"out",.2,.05,1)
    out=tmp_path/"out"/"TCGA_DEMO"
    m=pd.read_csv(out/"METRICS.tsv",sep="\t").set_index("target_id")
    assert m.loc[targets[0],"r2"]>.8
    assert m.loc[targets[1],"status"]=="COMPLETE_OOF"
    assert m.loc[targets[2],"status"]=="UNAVAILABLE"
    co=pd.read_parquet(out/"FOLD_SELECTED_GENES.parquet")
    fm=pd.read_csv(out/"FOLD_MODELS.tsv",sep="\t")
    predictions=pd.read_parquet(out/"OOF_PREDICTIONS.parquet")
    for row in fm.itertuples():
        c=co.loc[co.target_id.eq(row.target_id)&co.outer_fold.eq(row.outer_fold)]
        test=np.arange(n)%5==row.outer_fold
        v=x[test][:,[features.index(i) for i in c.lncrna_id]]
        v=np.where(np.isfinite(v),v,c.imputation_median.to_numpy())
        replay=v@c.coefficient.to_numpy()+row.intercept
        obs=predictions.loc[np.array(ids)[test],row.target_id].to_numpy()
        ok=np.isfinite(obs)
        np.testing.assert_allclose(replay[ok],obs[ok],atol=1e-10)
    final=pd.read_parquet(out/"FINAL_GENE_SETS.parquet")
    assert set(final.lncrna_id)<=set(features)
    assert final.screening_fdr.le(.05).all()
    assert final.screening_rho.abs().ge(.2).all()
    assert summary["final_nonempty_signatures"]>=2
