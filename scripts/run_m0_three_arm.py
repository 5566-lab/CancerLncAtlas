"""Compare lncRNA, mRNA and their union on matched native CPTAC assay outcomes."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, multiprocessing, socket, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from run_m0_linear import write_json
from cc_hhgt.m0_covariates import complete_rows
from cc_hhgt.m0_joint_lasso import cohort_screen
from cc_hhgt.m0_three_arm import ARMS, feature_axes, select_targets, restrict_split, matched_predictions, compare_target, regression_metrics
import run_m0_joint_lasso as fit_worker


def verify_receipt(directory, names):
    d=Path(directory); rec=json.loads((d/'COMPLETE.json').read_text())
    for name in names:
        if sha(d/name)!=rec['files'][name]:raise ValueError('Input changed: '+str(d/name))


def core_hash():
    paths=[Path(__file__), ROOT/'scripts/run_m0_joint_lasso.py',ROOT/'cc_hhgt/m0_three_arm.py',ROOT/'cc_hhgt/m0_joint_lasso.py',ROOT/'cc_hhgt/m0_covariates.py',ROOT/'cc_hhgt/m0_screened_lasso.py']
    return hashlib.sha256(''.join(sha(p) for p in paths).encode()).hexdigest()


def phase_fit(x, y, cov, ids, sf, si, split, inner, ratios, feature_ids, phase, arm, out, workers):
    ft=np.asarray(split['train'],int);te=np.asarray(split['test'],int)
    tr=np.asarray(inner['train'],int) if inner else np.empty(0,int)
    va=np.asarray(inner['validation'],int) if inner else np.empty(0,int)
    fit_worker.G=dict(x=x,cohort=cov.cohort.to_numpy(str),covariates=cov,target_ids=ids,yf=y,yi=y if inner else None,
                      ft=ft,tr=tr,va=va,te=te,sf=sf,si=si,final=inner is None,ratios=ratios)
    audits=[];models={};genes=[];predictions=[]
    t0=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for n,res in enumerate(pool.map(fit_worker.worker,range(len(ids)),chunksize=4),1):
            j=res['j'];tid=ids[j]
            row={k:v for k,v in res.items() if k not in ['j','model','rows','prediction','baseline','covariate_baseline']}
            row.update(target_id=tid,phase=phase,arm=arm)
            if res['status']=='FITTED':
                model=res['model'];models[tid]=model;idx=model['feature_index']
                row.update(n_selected=len(idx),dual_gap=model['dual_gap'],n_eligible=model['n_eligible'])
                if len(idx):
                    genes.append(pd.DataFrame(dict(target_id=tid,feature_id=feature_ids[idx],coefficient=model['coefficient'],screening_rho=sf['rho'][idx,j],screening_fdr=sf['q'][idx,j],phase=phase,arm=arm)))
                ix=res['rows']
                if len(ix):
                    predictions.append(pd.DataFrame(dict(patient_id=cov.index[ix],cohort=cov.cohort.iloc[ix].to_numpy(),target_id=tid,observed=y[ix,j],predicted=res['prediction'],train_mean_baseline=res['baseline'],covariate_only_baseline=res['covariate_baseline'],phase=phase)))
            audits.append(row)
            if n%100==0 or n==len(ids):print(phase,arm,n,'/',len(ids),'seconds',round(time.time()-t0),flush=True)
    fit_worker.G={}
    audit=pd.DataFrame(audits)
    audit.to_parquet(out/'AUDIT.parquet',index=False)
    joblib.dump(models,out/'MODELS.joblib',compress=3)
    (pd.concat(genes,ignore_index=True) if genes else pd.DataFrame(columns=['target_id','feature_id','coefficient','screening_rho','screening_fdr','phase','arm'])).to_parquet(out/'SELECTED_GENES.parquet',index=False)
    (pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=['patient_id','cohort','target_id','observed','predicted','train_mean_baseline','covariate_only_baseline','phase'])).to_parquet(out/'PREDICTIONS.parquet',index=False)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',target_host='149',phase=phase,arm=arm,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    return audit


def summarize(out, meta, features):
    combined={}
    for arm in ARMS:
        combined[arm]=pd.concat([pd.read_parquet(out/f'outer{k}'/arm/'PREDICTIONS.parquet') for k in range(5)],ignore_index=True)
    paired=matched_predictions(combined)
    if paired.duplicated(['patient_id','target_id']).any():raise ValueError('Repeated patient in OOF comparison')
    paired.to_parquet(out/'PAIRED_OOF_PREDICTIONS.parquet',index=False,compression='zstd')
    rows=[];fold_rows=[]
    for (tid,cohort),f in paired.groupby(['target_id','cohort'],sort=True):
        seed=int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)
        rows.append(dict(target_id=tid,cohort=cohort,**compare_target(f,seed=seed)))
        for phase,ff in f.groupby('phase'):
            for arm in ARMS:
                fold_rows.append(dict(target_id=tid,cohort=cohort,phase=phase,arm=arm,**regression_metrics(ff.observed,ff['prediction_'+arm])))
    if not rows:raise ValueError('No matched held-out target predictions')
    result=pd.DataFrame(rows).merge(meta.drop(columns='cohort'),on='target_id',validate='many_to_one')
    result['complete_fivefold']=result.evaluated_folds.eq(5)
    result.to_csv(out/'TARGET_COMPARISON.tsv',sep='\t',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'FOLD_COMPARISON.tsv',sep='\t',index=False)
    aggregate=[]
    for layer,f in result.loc[result.complete_fivefold].groupby('layer'):
        finite=f[['r2_'+a for a in ARMS]].notna().all(axis=1);f=f.loc[finite]
        for arm in ARMS:
            aggregate.append(dict(layer=layer,arm=arm,n_targets=len(f),median_oof_r2=f['r2_'+arm].median(),median_oof_spearman=f['spearman_'+arm].median(),positive_r2_targets=int(f['r2_'+arm].gt(0).sum()),better_than_covariates_targets=int((f['r2_'+arm]>f.r2_covariates+1e-9).sum()),uniquely_best_targets=int(f.best_arm.eq(arm).sum()),tied_targets=int(f.best_arm.eq('TIE').sum()),median_delta_r2_combined_vs_mrna=f.delta_r2_combined_vs_mrna.median(),combined_better_than_mrna_targets=int(f.delta_r2_combined_vs_mrna.gt(1e-9).sum())))
    pd.DataFrame(aggregate).to_csv(out/'LAYER_SUMMARY.tsv',sep='\t',index=False)
    final_genes=[];nuisance=[]
    for arm in ARMS:
        f=pd.read_parquet(out/'final'/arm/'SELECTED_GENES.parquet')
        folds=pd.concat([pd.read_parquet(out/f'outer{k}'/arm/'SELECTED_GENES.parquet') for k in range(5)],ignore_index=True)
        if len(folds):
            stable=folds.groupby(['target_id','feature_id']).phase.nunique().rename('selected_outer_folds').reset_index()
            f=f.merge(stable,on=['target_id','feature_id'],how='left',validate='one_to_one');f['selected_outer_folds']=f.selected_outer_folds.fillna(0).astype(int)
        else:f['selected_outer_folds']=0
        f['selection_frequency']=f.selected_outer_folds/5
        final_genes.append(f)
        for tid,m in joblib.load(out/'final'/arm/'MODELS.joblib').items():
            for i,col in enumerate(m['design_spec']['columns']):nuisance.append(dict(arm=arm,target_id=tid,covariate=col,coefficient=m['nuisance_coefficients'][i],penalized=False))
    pd.concat(final_genes,ignore_index=True).merge(features,on='feature_id',how='left',validate='many_to_one').to_csv(out/'FINAL_SIGNATURES.tsv.gz',sep='\t',index=False)
    pd.DataFrame(nuisance).to_csv(out/'FINAL_COVARIATE_COEFFICIENTS.tsv',sep='\t',index=False)
    return dict(n_paired_targets=len(result),n_complete_fivefold_targets=int(result.complete_fivefold.sum()),n_paired_predictions=len(paired),by_layer=result.groupby('layer').size().to_dict())


def run(aligned,splits,covariates,cancer,output,workers):
    if socket.gethostname()!='149':raise RuntimeError('Run on 149')
    if cancer not in ['BRCA','LUAD','COAD']:raise ValueError('Three-cancer demo only')
    src=Path(aligned)/cancer;sp=Path(splits)/cancer;cv=Path(covariates)/cancer;out=Path(output)/cancer;out.mkdir(parents=True,exist_ok=True)
    contract=json.loads((ROOT/'config/m0_three_arm_20261105.json').read_text())
    launch=dict(target_host='149',code_hash=core_hash(),config=contract,input_receipt_sha256=sha(src/'COMPLETE.json'),covariate_receipt_sha256=sha(cv/'COMPLETE.json'),splits_sha256=sha(sp/'SPLITS.json'),cancer=cancer)
    if (out/'LAUNCH_PREFLIGHT.json').exists() and json.loads((out/'LAUNCH_PREFLIGHT.json').read_text())!=launch:raise ValueError('Existing experiment differs')
    write_json(out/'LAUNCH_PREFLIGHT.json',launch)
    if (out/'COMPLETE.json').exists():
        done=json.loads((out/'COMPLETE.json').read_text())
        verify_receipt(out,list(done['files']));return done['summary']
    verify_receipt(src,['GENE_LOGTPM.parquet','GENE_ANNOTATION.tsv','PATIENTS.tsv','NATIVE_TARGETS.tsv','CPTAC_NATIVE_Y.parquet','X.parquet'])
    verify_receipt(cv,['COVARIATES.parquet'])
    patients=pd.read_csv(src/'PATIENTS.tsv',sep='\t').rename(columns={'index':'patient_id'}).set_index('patient_id')
    full_index=patients.index.to_list();keep=patients.index[patients.cohort.eq('CPTAC')]
    annotation=pd.read_csv(src/'GENE_ANNOTATION.tsv',sep='\t');features,axes=feature_axes(annotation)
    raw=pd.read_parquet(src/'GENE_LOGTPM.parquet',columns=features.gene_id.to_list()).reindex(keep)
    raw.columns=features.feature_id.to_numpy();x=raw.to_numpy(float)
    original_lnc=pd.read_parquet(src/'X.parquet')
    if original_lnc.index.to_list()!=full_index:raise ValueError('Split authority patient order differs from RNA input')
    original_lnc=original_lnc.reindex(keep)
    if not np.array_equal(raw.iloc[:,axes['lncrna']].to_numpy(),original_lnc.reindex(columns=raw.columns[axes['lncrna']]).to_numpy(),equal_nan=True):raise ValueError('Restored lncRNA axis differs from original demo')
    del original_lnc,raw
    cov=pd.read_parquet(cv/'COVARIATES.parquet').reindex(keep)
    if not cov.cohort.eq('CPTAC').all():raise ValueError('CPTAC covariate alignment failure')
    meta=select_targets(pd.read_csv(src/'NATIVE_TARGETS.tsv',sep='\t'))
    yf=pd.read_parquet(src/'CPTAC_NATIVE_Y.parquet',columns=meta.target_id.to_list()).reindex(keep)
    if np.isinf(yf.to_numpy()).any():raise ValueError('Infinite native assay score')
    good=complete_rows(cov,True)
    meta['n_observed_y']=np.isfinite(yf.to_numpy()).sum(axis=0)
    meta['n_observed_y_and_clinical_purity']=np.isfinite(yf.loc[good].to_numpy()).sum(axis=0)
    meta['availability']=np.where(meta.n_observed_y.eq(0),'NO_MEASURED_Y','OBSERVED_NATIVE_Y')
    meta.to_csv(out/'TARGETS.tsv',sep='\t',index=False)
    features.to_csv(out/'FEATURES.tsv',sep='\t',index=False)
    yf.to_parquet(out/'NATIVE_REFERENCE_Y.parquet',compression='zstd')
    cov.to_parquet(out/'COVARIATES.parquet',compression='zstd')
    spec=restrict_split(json.loads((sp/'SPLITS.json').read_text()),full_index,keep)
    write_json(out/'SPLITS.json',spec)
    print('INPUT_READY',cancer,'patients',len(keep),'clinical_complete',int(good.sum()),'features',{a:len(v) for a,v in axes.items()},flush=True)
    chosen={a:{} for a in ARMS};start=time.time()
    for phase,inner in [(f'outer{k}',f'inner{k}') for k in range(5)]+[('final',None)]:
        ft=np.asarray(spec[phase]['train'],int);tr=np.asarray(spec[inner]['train'],int) if inner else np.empty(0,int)
        valid=np.flatnonzero((np.isfinite(yf.iloc[ft].to_numpy())&good[ft,None]).sum(axis=0)>=12)
        if inner:valid=valid[(np.isfinite(yf.iloc[tr,valid].to_numpy())&good[tr,None]).sum(axis=0)>=12]
        ids=meta.target_id.to_numpy()[valid].tolist();y=yf.iloc[:,valid].to_numpy(float)
        if not len(ids):raise ValueError('No assay targets have enough observed training patients')
        for arm in ARMS:
            d=out/phase/arm;d.mkdir(parents=True,exist_ok=True)
            if (d/'COMPLETE.json').exists():
                rec=json.loads((d/'COMPLETE.json').read_text());verify_receipt(d,list(rec['files']));audit=pd.read_parquet(d/'AUDIT.parquet');print('RESUME',cancer,phase,arm,flush=True)
            else:
                xx=x[:,axes[arm]];feature_ids=features.feature_id.to_numpy()[axes[arm]]
                print('SCREEN',cancer,phase,arm,'targets',len(ids),flush=True)
                with threadpool_limits(limits=min(workers,8)):
                    sf=cohort_screen(xx[ft],y[ft],cov.cohort.iloc[ft],min_abs_rho=contract['min_abs_rho'],max_fdr=contract['max_fdr'],covariates=cov.iloc[ft],target_ids=ids)
                    si=cohort_screen(xx[tr],y[tr],cov.cohort.iloc[tr],min_abs_rho=contract['min_abs_rho'],max_fdr=contract['max_fdr'],covariates=cov.iloc[tr],target_ids=ids) if inner else None
                ratios=np.array([np.median(chosen[arm].get(t,[])) if len(chosen[arm].get(t,[]))>=3 else np.nan for t in ids]) if not inner else np.full(len(ids),np.nan)
                audit=phase_fit(xx,y,cov,ids,sf,si,spec[phase],spec[inner] if inner else None,ratios,feature_ids,phase,arm,d,workers)
                del sf,si,xx
            if inner:
                for row in audit.loc[audit.status.eq('FITTED')].itertuples():chosen[arm].setdefault(row.target_id,[]).append(float(row.ratio))
    summary=summarize(out,meta,features)
    summary.update(cancer_id=cancer,cohort='CPTAC',n_patients=len(keep),clinical_and_purity_complete=int(good.sum()),n_features={a:len(v) for a,v in axes.items()},seconds=round(time.time()-start),upstream_assay_preprocessing='NATIVE_RETAINED_NOT_FOLD_REFIT')
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',target_host='149',launch=launch,summary=summary,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    print('CASE_COMPLETE',json.dumps(summary),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['aligned','splits','covariates','cancer','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=24:raise ValueError('At most 24 CPU workers')
    run(a.aligned,a.splits,a.covariates,a.cancer,a.output,a.workers)
