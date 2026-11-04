#!/usr/bin/env python3
"""Verify M0 outputs, observed-label masks and replay representative real checkpoints."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from cla_engine.common import sha
from cc_hhgt.m0_linear import predict_checkpoint
from run_m0_linear import load_case,code_hash,write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",required=True);ap.add_argument("--config",required=True)
    a=ap.parse_args()
    if socket.gethostname()!="149":raise RuntimeError("Requires 149")
    root=Path(a.results);complete=json.loads((root/"COMPLETE.json").read_text())
    assert complete["code_hash"]==code_hash()
    cfg=json.loads(Path(a.config).read_text());sources={};checks=[];replayed=[]
    for p in sorted(root.glob("*/COMPLETE.json")):
        case=p.parent;rec=json.loads(p.read_text());info=rec["summary"]
        assert rec["target_host"]=="149" and rec["code_hash"]==complete["code_hash"]
        assert rec["config_sha256"]==sha(a.config)
        for rel,digest in rec["files"].items():
            assert sha(case/rel)==digest, str(case/rel)
        for f in json.loads((case/"INPUTS.json").read_text())["sources"]:
            if f["path"] in sources:assert sources[f["path"]]==f["sha256"]
            sources[f["path"]]=f["sha256"]
        y=pd.read_parquet(case/"OBSERVED_SCORES.parquet")
        pred=pd.read_parquet(case/"OOF_PREDICTIONS.parquet")
        base=pd.read_parquet(case/"OOF_TRAIN_MEAN_BASELINE.parquet")
        folds=pd.read_csv(case/"PATIENT_FOLDS.tsv",sep="\t").set_index("patient_id")
        assert y.index.is_unique and y.columns.is_unique and y.index.equals(pred.index)
        assert y.columns.equals(pred.columns) and y.index.equals(base.index)
        assert set(y.index)==set(folds.index) and set(folds.outer_fold)==set(range(5))
        assert not (np.isfinite(pred.to_numpy()) & ~np.isfinite(y.to_numpy())).any()
        assert np.array_equal(np.isfinite(pred.to_numpy()),np.isfinite(base.to_numpy()))
        met=pd.read_csv(case/"METRICS.tsv",sep="\t").set_index("target_id").reindex(y.columns)
        np.testing.assert_array_equal(np.isfinite(pred).sum(axis=0),met.n_oof)
        np.testing.assert_array_equal(np.isfinite(y).sum(axis=0),met.n_observed)
        assert int(np.isfinite(pred).sum().sum())==info["n_heldout_predictions"]
        audit=pd.read_parquet(case/"FOLD_AUDIT.parquet")
        fitted=audit.loc[audit.status.eq("FITTED")]
        assert (fitted.n_train>=10).all() and (fitted.n_validation>=3).all() and (fitted.n_test>=3).all()
        expected_prefix="TCGA-" if info["dataset"]=="TCGA" else "CPTAC:"+info["cohort"]+":"
        assert all(str(i).startswith(expected_prefix) for i in y.index)
        for fold in range(5):
            # Every checkpoint's patient partition is checked; arrays are read lazily.
            for model in (case/"models").glob(f"fold_{fold}_mask_*.npz"):
                with np.load(model,allow_pickle=False) as z:
                    fit=set(z["fit_patient_ids"]);test=set(z["test_patient_ids"])
                    assert not fit & test and fit <= set(folds.index[folds.outer_fold.ne(fold)])
                    assert test <= set(folds.index[folds.outer_fold.eq(fold)])
        if case.name in ("TCGA_ACC","TCGA_BRCA","CPTAC_BRCA","CPTAC_LUAD"):
            task={k:info[k] for k in ("dataset","cohort","cancer_id")}
            x,_,_,_,_,_=load_case(cfg,task,[])
            selected=sorted((case/"models").glob("fold_0_mask_*.npz"))
            # Include a large complete-label group and a second missing-label group.
            for model in selected[:2]:
                with np.load(model,allow_pickle=False) as z:
                    test=z["test_patient_ids"].tolist();targets=z["target_ids"].tolist()
                got=predict_checkpoint(model,x.reindex(test).to_numpy(float))
                np.testing.assert_allclose(got,pred.loc[test,targets].to_numpy(float),atol=1e-10,rtol=1e-9)
                replayed.append(dict(study=case.name,checkpoint=model.name,values=int(got.size)))
        checks.append(dict(study=case.name,files=len(rec["files"]),targets=len(y.columns),patients=len(y),
                           oof_values=int(np.isfinite(pred).sum().sum()),partition_checks="PASS"))
        print("AUDITED",case.name,flush=True)
    for path,digest in sources.items():assert sha(path)==digest,"INPUT_DRIFT:"+path
    assert len(checks)==complete["studies"]
    result=dict(status="PASS",target_host="149",studies=len(checks),unique_source_files=len(sources),
                output_files_verified=sum(c["files"] for c in checks),replayed=replayed,checks=checks,
                predictors="lncRNA_expression_only",outcome_imputation=False,
                upstream_fully_inductive=False)
    write_json(root/"M0_AUDIT.json",result)
    print("M0_AUDIT_PASS",flush=True)
if __name__=="__main__":main()
