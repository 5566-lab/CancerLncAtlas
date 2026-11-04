"""Verify screened Lasso demo receipts, thresholds and saved-model replay on 149."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from cla_engine.common import sha
from run_m0_linear import write_json
from cc_hhgt.m0_screened_lasso import screen,fit_fixed

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--results",required=True);ap.add_argument("--inputs",required=True)
    a=ap.parse_args()
    if socket.gethostname()!="149":raise RuntimeError("Run on 149")
    root=Path(a.results);rec=json.loads((root/"COMPLETE.json").read_text())
    write_json(root/"AUDIT_PREFLIGHT.json",dict(target_host="149",operation="LASSO_INPUT_OUTPUT_AND_MODEL_REPLAY"))
    checked=0;replays=[];case_stats=[]
    for case in rec["cases"]:
        d=root/case;src=Path(a.inputs)/case
        done=json.loads((d/"COMPLETE.json").read_text());inp=json.loads((src/"COMPLETE.json").read_text())
        assert done["experiment"]["input_complete_sha256"]==sha(src/"COMPLETE.json")
        for parent,receipt in [(d,done),(src,inp)]:
            for rel,digest in receipt["files"].items():
                assert sha(parent/rel)==digest,(case,rel);checked+=1
        x=pd.read_parquet(src/"X.parquet");y=pd.read_parquet(src/"Y.parquet")
        pred=pd.read_parquet(d/"OOF_PREDICTIONS.parquet")
        folds=pd.read_csv(src/"PATIENT_FOLDS.tsv",sep="\t").set_index("patient_id").outer_fold.reindex(x.index)
        coef=pd.read_parquet(d/"FOLD_SELECTED_GENES.parquet");models=pd.read_csv(d/"FOLD_MODELS.tsv",sep="\t")
        final=pd.read_parquet(d/"FINAL_GENE_SETS.parquet");fm=pd.read_csv(d/"FINAL_MODELS.tsv",sep="\t").set_index("target_id")
        if len(final):
            assert final.coefficient.ne(0).all()
            assert final.screening_rho.abs().ge(done["experiment"]["min_abs_rho"]-1e-7).all()
            assert final.screening_fdr.le(done["experiment"]["max_fdr"]+1e-7).all()
            assert final.screening_n.ge(10).all()
            assert not final.duplicated(["target_id","lncrna_id"]).any()
            assert set(final.lncrna_id)<=set(x.columns)
        actual=final.groupby("target_id").size().reindex(fm.index,fill_value=0)
        fitting=fm.status.eq("FITTED")
        assert (actual.loc[fitting].to_numpy()==fm.loc[fitting,"n_selected"].to_numpy()).all()
        # Check models across folds and layers, not only favorable predictions.
        meta=pd.read_csv(src/"TARGETS.tsv",sep="\t").set_index("target_id")
        pick=models.assign(layer=models.target_id.map(meta.layer)).groupby(["outer_fold","layer"],group_keys=False).head(2)
        for row in pick.itertuples():
            c=coef.loc[coef.target_id.eq(row.target_id)&coef.outer_fold.eq(row.outer_fold)]
            xx=x.loc[folds.eq(row.outer_fold),c.lncrna_id].to_numpy()
            xx=np.where(np.isfinite(xx),xx,c.imputation_median.to_numpy())
            replay=xx@c.coefficient.to_numpy()+row.intercept
            observed=pred.loc[folds.eq(row.outer_fold),row.target_id].to_numpy()
            ok=np.isfinite(observed)
            assert np.allclose(replay[ok],observed[ok],atol=1e-8,rtol=1e-8)
            replays.append(dict(case=case,target_id=row.target_id,fold=int(row.outer_fold),n=int(ok.sum()),
                                max_absolute_error=float(np.max(np.abs(replay[ok]-observed[ok]))) if ok.any() else 0.))
        # Independently recompute a fixed biological final signature per case.
        target="STATE|signifinder::Hypoxia_Buffa"
        if target in fm.index and fm.loc[target,"status"]=="FITTED":
            yy=y[target].to_numpy();xx=x.to_numpy()
            sc=screen(xx,yy,min_abs_rho=done["experiment"]["min_abs_rho"],max_fdr=done["experiment"]["max_fdr"])
            pp,st=fit_fixed(xx,yy,sc["passed"][:,0],xx[:5],float(fm.loc[target,"alpha_ratio"]))
            c=final.loc[final.target_id.eq(target)]
            query=x.iloc[:5][c.lncrna_id].to_numpy()
            query=np.where(np.isfinite(query),query,c.imputation_median.to_numpy())
            replay=query@c.coefficient.to_numpy()+float(fm.loc[target,"intercept"])
            assert np.allclose(pp,replay,atol=1e-7,rtol=1e-7)
            replays.append(dict(case=case,target_id=target,fold="FINAL_REFIT",n=5,max_absolute_error=float(np.max(np.abs(pp-replay)))))
        if not np.isfinite(y.to_numpy()).all():
            assert not np.isfinite(pred.to_numpy()[~np.isfinite(y.to_numpy())]).any()
        au=pd.read_parquet(d/"FOLD_AUDIT.parquet")
        fitted=au.loc[au.status.eq("FITTED")]
        assert fitted.path_max_dual_gap.le(1.01e-4).all()
        assert fitted.fit_dual_gap.le(1.01e-4).all()
        case_stats.append(dict(case=case,patients=len(x),features=x.shape[1],final_nonempty=int(actual.gt(0).sum()),
                          final_selected_rows=len(final),evaluated_predictions=int(np.isfinite(pred.to_numpy()).sum())))
        print("AUDIT_CASE_PASS",case,flush=True)
    pd.DataFrame(replays).to_csv(root/"MODEL_REPLAY_CHECKS.tsv",sep="\t",index=False)
    write_json(root/"AUDIT_COMPLETE.json",dict(status="PASS",target_host="149",verified_files=checked,
       case_stats=case_stats,replayed_models=len(replays),input_hashes_verified=True,
       outer_test_patients_excluded_from_selection_by_unit_tests=True,final_thresholds_verified=True,
       score_calibration="NOT_ESTABLISHED_BETWEEN_COHORTS",cohort_pooling=False))
if __name__=="__main__":main()
