"""Independent checks against raw assay inputs and ordinary least squares."""
import argparse,pathlib,socket,json,hashlib
import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def review(root,output):
    assert socket.gethostname()=="149"
    root=pathlib.Path(root);out=pathlib.Path(output);out.mkdir(exist_ok=False)
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="INDEPENDENT_COHORT_CV_RECONSTRUCTION",input=str(root),source_code_sha256=sha(__file__)),indent=2))
    m=json.loads((root/"COMPLETE.json").read_text())
    for rel,r in m["files"].items():assert sha(root/rel)==r["sha256"],rel
    B=pathlib.Path("/dell_2/DSC/CancerLncAtlas");P=B/"runtime/tcpa_cptac_calibration_20261106_r1"
    proof=pd.read_csv(P/"specimen_audit_r1/PATIENT_SPECIMEN_CONCORDANCE.tsv",sep="\t")
    candidates=pd.read_csv(P/"mapping_r1/RETROSPECTIVE_PATIENT_PAIR_CANDIDATES.tsv",sep="\t")
    candidates=candidates.loc[candidates.dataset.eq(m["dataset"]+"_protein")&candidates.cancer_id.eq(m["cancer_id"])&candidates.in_current_rppa_rna_scope]
    expected=set(candidates.patient_id)&set(proof.loc[proof.dataset.eq(m["dataset"]+"_protein")&proof.pdc_vial_concordant,"patient_id"])
    records=[];heldout=[];allrows=[];tol=1e-9
    for fold in range(5):
        Q=root/("fold"+str(fold));receipt=json.loads((Q/"COMPLETE.json").read_text())
        for p,h in receipt["source_sha256"].items():assert sha(p)==h,p
        for rel,r in receipt["files"].items():assert sha(Q/rel)==r["sha256"],rel
        split=pd.read_csv(Q/"PATIENT_SPLIT.tsv",sep="\t");ids=split.patient_id.tolist()
        assert len(set(ids))==len(ids) and set(ids)==expected
        assert split.heldout.eq(split.patient_fold_id.eq(fold)).all()
        train=~split.heldout.to_numpy(bool);test=~train;heldout.extend(split.loc[split.heldout,"patient_id"])
        sc=pd.read_csv(Q/"TRAIN_REFERENCE.tsv",sep="\t")
        rx=pd.read_parquet(B/"inputs/rppa500_20261104_r4/PROTEIN.parquet").set_index("patient_id").loc[ids,sc.rppa_feature_id];rx.columns=sc.gene_id
        source=[r for r in json.loads((P/"DOWNLOAD_COMPLETE.json").read_text())["files"] if r["cancer"]+"_"+r["center"]==m["dataset"] and r["layer"]=="protein"]
        assert len(source)==1
        ms=pd.read_csv(source[0]["path"],sep="\t",index_col=0)
        native=candidates.set_index("patient_id").loc[ids,"native_patient_id"]
        cx=ms.loc[sc.ms_gene_symbol,native].T;cx.index=ids;cx.columns=sc.gene_id
        err=dict(reference=0.,score=0.,prediction=0.,r2=0.,slope=0.)
        def check(a,b,kind):
            a=np.asarray(a,float);b=np.asarray(b,float)
            np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
            f=np.isfinite(a)&np.isfinite(b)
            if f.any():err[kind]=max(err[kind],float(np.max(np.abs(a[f]-b[f]))))
        valid=np.ones(len(sc),bool)
        frames=[]
        for frame,prefix in [(rx,"rppa"),(cx,"ms")]:
            mu=frame.iloc[train].mean();sd=frame.iloc[train].std(ddof=1)
            check(mu,sc[prefix+"_train_mean"],"reference");check(sd,sc[prefix+"_train_sd"],"reference")
            valid&=(frame.iloc[train].count().to_numpy()>=2)&np.isfinite(sd.to_numpy())&(sd.to_numpy()>0)
            frames.append(frame.subtract(mu).divide(sd))
        np.testing.assert_array_equal(valid,sc.train_usable.to_numpy(bool))
        rz,cz=[f.loc[:,sc.gene_id[valid]] for f in frames]
        read=lambda name:pd.read_parquet(Q/(name+".parquet")).set_index("patient_id").loc[ids]
        rout,cout,pout,rn,cn=[read(x) for x in ["RPPA_COMMON_PANEL_SCORE","MS_COMMON_PANEL_SCORE","RPPA_TO_MS_PREDICTED_SCORE","RPPA_OBSERVED_COUNTS","MS_OBSERVED_COUNTS"]]
        metrics=pd.read_csv(Q/"HELDOUT_PATHWAY_METRICS.tsv",sep="\t").set_index("pathway_id")
        bridge=pd.read_parquet(B/"inputs/cptac_20261104_r1/bundle_r1/PATHWAY_GENE_PROTEIN_BRIDGE.parquet")
        assert set(metrics.index)==set(bridge.pathway_id) and metrics.index.is_unique
        fitted=tested=0
        rows=[]
        for pathway,g in bridge.groupby("pathway_id"):
            hit=sorted(set(g.gene_id)&set(rz.columns));x=rz[hit].mean(axis=1).to_numpy();y=cz[hit].mean(axis=1).to_numpy()
            check(x,rout[pathway],"score");check(y,cout[pathway],"score")
            np.testing.assert_array_equal(rz[hit].count(axis=1),rn[pathway]);np.testing.assert_array_equal(cz[hit].count(axis=1),cn[pathway])
            row=metrics.loc[pathway];assert row.panel_genes==len(hit)
            keep=train&np.isfinite(x)&np.isfinite(y);assert row.train_pairs==int(keep.sum())
            r2=np.nan;ntest=0;coef=[np.nan,np.nan];prediction=np.full(len(x),np.nan)
            if keep.sum()>=3 and np.var(x[keep])>1e-15:
                coef=np.linalg.lstsq(np.column_stack([np.ones(keep.sum()),x[keep]]),y[keep],rcond=None)[0]
                prediction=coef[0]+coef[1]*x;fitted+=1
                k=test&np.isfinite(prediction)&np.isfinite(y);ntest=int(k.sum())
                if ntest>=3 and np.var(y[k])>0:
                    r2=1-((y[k]-prediction[k])**2).sum()/((y[k]-y[k].mean())**2).sum();tested+=1
            check(coef,[row.intercept,row.slope],"slope");check(prediction,pout[pathway],"prediction");check([r2],[row.heldout_r2],"r2")
            assert ntest==row.heldout_pairs
            rows.append(dict(pathway_id=pathway,r2=r2,panel_genes=len(hit),heldout_pairs=ntest))
        allrows.extend(rows)
        records.append(dict(fold=fold,patients=len(ids),heldout=int(test.sum()),fitted_pathways=fitted,heldout_r2_checked=tested,maximum_errors=err))
        print("FOLD_INDEPENDENT_PASS",m["dataset"],fold,tested,flush=True)
    assert len(heldout)==len(set(heldout))==len(expected)
    rows=pd.DataFrame(allrows);summary=pd.read_csv(root/"PATHWAY_CV_SUMMARY.tsv",sep="\t").set_index("pathway_id")
    for pathway,g in rows.groupby("pathway_id"):
        s=summary.loc[pathway];finite=g.r2.dropna()
        assert int(s.completed_folds)==len(finite) and int(s.positive_folds)==int(finite.gt(0).sum())
        assert s.min_panel_genes==g.panel_genes.min() and s.min_heldout_pairs==g.heldout_pairs.min()
        np.testing.assert_allclose([s.mean_fold_r2,s.median_fold_r2],[finite.mean(),finite.median()],atol=tol,rtol=0,equal_nan=True)
    rec=dict(target_host="149",status="PASS_INDEPENDENT_COHORT_CV",dataset=m["dataset"],cancer_id=m["cancer_id"],patients=len(expected),heldout_exactly_once=True,folds=records,summary_pathways=len(summary),production_validation=False,input_receipt_sha256=sha(root/"COMPLETE.json"),source_code_sha256=sha(__file__))
    (out/"COMPLETE.json").write_text(json.dumps(rec,indent=2));print("REVIEW_COMPLETE",m["dataset"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);a=p.parse_args();review(a.root,a.output)
