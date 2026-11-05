"""Joint cohort-aware screened Lasso: five-fold and bidirectional transfer tests."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,socket,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from cc_hhgt.m0_joint_scores import JointScoreBuilder
from cc_hhgt.m0_joint_lasso import fit_x_context,select_ratio,fit_final,predict,cohort_screen
from cc_hhgt.m0_linear import metrics
from cc_hhgt.m0_covariates import complete_rows,uses_purity
from run_m0_linear import write_json
G={}

def worker(j):
    x=G['x'];cohort=G['cohort'];cov=G['covariates'];tid=G['target_ids'][j];use_purity=uses_purity(tid)
    eligible=complete_rows(cov,use_purity);y=G['yf'][:,j];ft=G['ft'];ft=ft[np.isfinite(y[ft]) & eligible[ft]]
    base=dict(j=j,n_fit=len(ft),purity_covariate_used=use_purity,adjustment='AGE_SEX_STAGE_COHORT'+('_FIXED_DNA_PURITY_COHORT_INTERACTION' if use_purity else '_PURITY_SELF_EXCLUDED'));mask=G['sf']['passed'][:,j]
    if len(ft)<12:return dict(**base,status='INSUFFICIENT_OBSERVED_TRAINING')
    if G['final']:
        ratio=G['ratios'][j]
        if not np.isfinite(ratio):return dict(**base,status='INSUFFICIENT_OUTER_VALIDATION')
        pa={}
    else:
        yi=G['yi'][:,j];tr=G['tr'];va=G['va'];tr=tr[np.isfinite(yi[tr]) & eligible[tr]];va=va[np.isfinite(yi[va]) & eligible[va]]
        if len(tr)<12 or len(va)<3:return dict(**base,status='INSUFFICIENT_OBSERVED_TUNING')
        context=fit_x_context(x[tr],cohort[tr],G['si']['passed'][:,j],cov.iloc[tr],use_purity)
        try:ratio,pa=select_ratio(context,yi[tr],x[va],yi[va],cohort[va],cov.iloc[va])
        except ValueError as ex:
            if str(ex)=='INSUFFICIENT_SUPPORTED_VALIDATION':return dict(**base,status=str(ex))
            raise
        del context
    context=fit_x_context(x[ft],cohort[ft],mask,cov.iloc[ft],use_purity);model=fit_final(context,y[ft],ratio);del context
    te=G['te'];n_observed_test=int(np.isfinite(y[te]).sum());te=te[np.isfinite(y[te]) & eligible[te]]
    pred=predict(model,x[te],cohort[te],cov.iloc[te]) if len(te) else np.empty(0)
    cb=predict(model,x[te],cohort[te],cov.iloc[te],covariate_only=True) if len(te) else np.empty(0)
    ok=np.isfinite(pred)&np.isfinite(cb);te=te[ok];pred=pred[ok];cb=cb[ok]
    baseline=np.array([model['outcome_mean'][model['groups'].index(c)] if c in model['groups'] else model['outcome_mean'].mean() for c in cohort[te]])
    return dict(**base,status='FITTED',model=model,rows=te,prediction=pred,baseline=baseline,covariate_baseline=cb,ratio=ratio,n_screened=int(mask.sum()),n_observed_test=n_observed_test,n_supported_test=len(te),**pa)

def core_hash():
    files=[Path(__file__),ROOT/'cc_hhgt/m0_joint_lasso.py',ROOT/'cc_hhgt/m0_joint_scores.py',ROOT/'cc_hhgt/m0_screened_lasso.py',ROOT/'cc_hhgt/m0_covariates.py']
    return hashlib.sha256(''.join(sha(p) for p in files).encode()).hexdigest()

def run_case(aligned,cache,splits,covariates,cancer,output,workers):
    global G
    if socket.gethostname()!='149':raise RuntimeError('Run on 149')
    src=Path(aligned)/cancer;ca=Path(cache)/cancer;sp=Path(splits)/cancer;cv=Path(covariates)/cancer;out=Path(output)/cancer;out.mkdir(parents=True,exist_ok=True)
    contract=json.loads((ROOT/'config/m0_joint_demo_20261105.json').read_text())
    launch=dict(target_host='149',cancer=cancer,code_hash=core_hash(),config=contract,input_complete_sha256=sha(src/'COMPLETE.json'),cache_complete_sha256=sha(ca/'COMPLETE.json'),apm_complete_sha256=sha(sp/'apm/COMPLETE.json'),covariates_complete_sha256=sha(cv/'COMPLETE.json'))
    if (out/'LAUNCH_PREFLIGHT.json').exists() and json.loads((out/'LAUNCH_PREFLIGHT.json').read_text())!=launch:raise ValueError('Different existing experiment')
    write_json(out/'LAUNCH_PREFLIGHT.json',launch)
    if (out/'COMPLETE.json').exists():return json.loads((out/'COMPLETE.json').read_text())['summary']
    xf=pd.read_parquet(src/'X.parquet');patients=pd.read_csv(src/'PATIENTS.tsv',sep='\t').rename(columns={'index':'patient_id'}).set_index('patient_id').reindex(xf.index)
    if patients.isna().any().any():raise ValueError('Patient authority mismatch')
    cov=pd.read_parquet(cv/'COVARIATES.parquet').reindex(xf.index)
    if not cov.cohort.equals(patients.cohort):raise ValueError('Covariate patient/cohort mismatch')
    cv_receipt=json.loads((cv/'COMPLETE.json').read_text())
    for name,digest in cv_receipt['files'].items():
        if sha(cv/name)!=digest:raise ValueError('Covariate input drift')
    x=xf.to_numpy(float);cohort=patients.cohort.to_numpy(str);spec=json.loads((sp/'SPLITS.json').read_text())
    scorer=JointScoreBuilder(src,ca,sp/'apm');t0=time.time();models_by_phase={};metrics_all=[];audits=[];fold_genes=[];chosen={}
    phases=[(f'outer{k}',f'inner{k}') for k in range(5)]+[(f'loco_{c}_outer',f'loco_{c}_inner') for c in ['TCGA','CPTAC']]+[('final',None)]
    for phase,inner in phases:
        d=out/phase;d.mkdir(exist_ok=True)
        if (d/'COMPLETE.json').exists():
            receipt=json.loads((d/'COMPLETE.json').read_text())
            for name,digest in receipt['files'].items():
                if sha(d/name)!=digest:raise ValueError('Checkpoint drift')
            print(cancer,'RESUME',phase,flush=True)
        else:
            ft=np.asarray(spec[phase]['train'],int);te=np.asarray(spec[phase]['test'],int)
            print(cancer,'SCORE',phase,'train',len(ft),flush=True)
            yf,state=scorer.fit_apply(ft,phase);target_ids=yf.columns.to_list();meta=scorer.target_metadata.copy()
            yf.to_parquet(d/'REFERENCE_Y.parquet',compression='zstd');joblib.dump(state,d/'SCORE_REFERENCE.joblib',compress=3)
            if inner:
                tr=np.asarray(spec[inner]['train'],int);va=np.asarray(spec[inner]['validation'],int)
                yi,inner_state=scorer.fit_apply(tr,inner);joblib.dump(inner_state,d/'INNER_SCORE_REFERENCE.joblib',compress=3)
            else:tr=va=np.empty(0,int);yi=None
            valid=np.flatnonzero(np.isfinite(yf.iloc[ft].to_numpy()).sum(axis=0)>=12)
            if inner:valid=valid[np.isfinite(yi.iloc[tr,valid].to_numpy()).sum(axis=0)>=12]
            # Restrict calculations to testable targets, retaining all original
            # targets in TARGETS/availability exports. BH still includes all X.
            yv=yf.iloc[:,valid].to_numpy(float)
            print(cancer,'SCREEN',phase,'targets',len(valid),flush=True)
            with threadpool_limits(limits=min(workers,8)):
                sf=cohort_screen(x[ft],yv[ft],cohort[ft],min_abs_rho=contract['min_abs_rho'],max_fdr=contract['max_fdr'],covariates=cov.iloc[ft],target_ids=[target_ids[j] for j in valid])
                si=cohort_screen(x[tr],yi.iloc[tr,valid].to_numpy(float),cohort[tr],min_abs_rho=contract['min_abs_rho'],max_fdr=contract['max_fdr'],covariates=cov.iloc[tr],target_ids=[target_ids[j] for j in valid]) if inner else None
            if inner:ratios=np.full(len(valid),np.nan)
            else:
                ratios=np.array([np.median(chosen.get(target_ids[j],[])) if len(chosen.get(target_ids[j],[]))>=3 else np.nan for j in valid])
            G=dict(x=x,cohort=cohort,covariates=cov,target_ids=[target_ids[j] for j in valid],yf=yv,yi=yi.iloc[:,valid].to_numpy(float) if inner else None,ft=ft,tr=tr,va=va,te=te,sf=sf,si=si,final=inner is None,ratios=ratios)
            phase_models={};phase_audit=[dict(target_id=target_ids[j],phase=phase,status='INSUFFICIENT_OBSERVED_TRAINING',n_fit=int(np.isfinite(yf.iloc[ft,j]).sum())) for j in sorted(set(range(len(target_ids)))-set(valid))];phase_metrics=[];genes=[];predictions=[]
            if phase=='final':
                gi,tj=np.nonzero(sf['passed']);pd.DataFrame(dict(target_id=np.asarray(target_ids)[valid][tj],lncrna_id=np.asarray(xf.columns)[gi],rho=sf['rho'][gi,tj],fdr=sf['q'][gi,tj],n=sf['n'][gi,tj],df=sf['df'][gi,tj])).to_parquet(d/'ALL_SCREEN_PASSED.parquet',index=False,compression='zstd')
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
                for count,res in enumerate(pool.map(worker,range(len(valid)),chunksize=4),1):
                    j=res['j'];tid=target_ids[valid[j]];row={k:v for k,v in res.items() if k not in ['j','model','rows','prediction','baseline','covariate_baseline']};row.update(target_id=tid,phase=phase)
                    if res['status']=='FITTED':
                        st=res['model'];phase_models[tid]=st;row.update(n_selected=len(st['feature_index']),dual_gap=st['dual_gap'],n_eligible=st['n_eligible'])
                        ii=st['feature_index'];gg=pd.DataFrame(dict(target_id=tid,lncrna_id=np.asarray(xf.columns)[ii],coefficient=st['coefficient'],screening_rho=sf['rho'][ii,j],screening_fdr=sf['q'][ii,j],screening_n=sf['n'][ii,j],phase=phase));genes.append(gg)
                        test=res['rows']
                        if len(test):
                            frame=pd.DataFrame(dict(patient_id=xf.index[test],cohort=cohort[test],target_id=tid,observed=yv[test,j],predicted=res['prediction'],train_mean_baseline=res['baseline'],covariate_only_baseline=res['covariate_baseline'],phase=phase));predictions.append(frame)
                            for cc,ff in frame.groupby('cohort'):
                                mm=metrics(ff.observed.to_numpy(),ff.predicted.to_numpy(),ff.train_mean_baseline.to_numpy());bm=metrics(ff.observed.to_numpy(),ff.covariate_only_baseline.to_numpy(),ff.train_mean_baseline.to_numpy());mm['covariate_only_r2']=bm['r2'];mm['delta_r2_over_covariates']=mm['r2']-bm['r2'];mm['covariate_only_mse']=float(np.mean((ff.observed-ff.covariate_only_baseline)**2));mm['incremental_r2_vs_covariates']=1-float(np.sum((ff.observed-ff.predicted)**2))/float(np.sum((ff.observed-ff.covariate_only_baseline)**2)) if np.sum((ff.observed-ff.covariate_only_baseline)**2)>0 else np.nan;mm.update(target_id=tid,cohort=cc,phase=phase,cancer_id=cancer);phase_metrics.append(mm)
                    phase_audit.append(row)
                    if count%50==0 or count==len(valid):print(cancer,phase,count,'/',len(valid),'seconds',round(time.time()-t0),flush=True)
            G={};joblib.dump(phase_models,d/'MODELS.joblib',compress=3)
            pd.DataFrame(phase_audit).to_parquet(d/'AUDIT.parquet',index=False)
            (pd.DataFrame(phase_metrics) if phase_metrics else pd.DataFrame(columns=['target_id','cohort','phase','cancer_id','r2','spearman','n_oof','covariate_only_r2','delta_r2_over_covariates','incremental_r2_vs_covariates'])).to_csv(d/'METRICS.tsv',sep='\t',index=False)
            (pd.concat(genes,ignore_index=True) if genes else pd.DataFrame(columns=['target_id','lncrna_id','coefficient','screening_rho','screening_fdr','screening_n','phase'])).to_parquet(d/'SELECTED_GENES.parquet',index=False,compression='zstd')
            if predictions:pd.concat(predictions,ignore_index=True).to_parquet(d/'PREDICTIONS.parquet',index=False,compression='zstd')
            meta.to_csv(out/'TARGETS.tsv',sep='\t',index=False)
            write_json(d/'COMPLETE.json',dict(status='COMPLETE',target_host='149',phase=phase,files={p.name:sha(p) for p in d.iterdir() if p.is_file()}))
            del sf,si,yf,yi,phase_models
        a=pd.read_parquet(d/'AUDIT.parquet');audits.append(a)
        if phase.startswith('outer'):
            for row in a.loc[a.status.eq('FITTED')].itertuples():chosen.setdefault(row.target_id,[]).append(float(row.ratio))
            fold_genes.append(pd.read_parquet(d/'SELECTED_GENES.parquet'))
        if phase!='final':metrics_all.append(pd.read_csv(d/'METRICS.tsv',sep='\t'))
    allm=pd.concat(metrics_all,ignore_index=True).merge(pd.read_csv(out/'TARGETS.tsv',sep='\t'),on='target_id',validate='many_to_one')
    allm.to_csv(out/'FOLD_METRICS.tsv',sep='\t',index=False)
    # Reference-fitted labels differ between folds. Report foldwise metrics,
    # never a pooled R2 across changing scoring references.
    oof=allm.loc[allm.phase.str.startswith('outer')]
    agg=oof.groupby(['target_id','cohort','layer','scope'],dropna=False).agg(mean_fold_r2=('r2','mean'),median_fold_r2=('r2','median'),mean_fold_spearman=('spearman','mean'),mean_covariate_only_r2=('covariate_only_r2','mean'),mean_delta_r2_over_covariates=('delta_r2_over_covariates','mean'),mean_incremental_r2_vs_covariates=('incremental_r2_vs_covariates','mean'),n_predictions=('n_oof','sum'),evaluated_folds=('r2','count')).reset_index()
    agg['cancer_id']=cancer;agg.to_csv(out/'COHORT_VALIDATION.tsv',sep='\t',index=False)
    allm.loc[allm.phase.str.startswith('loco')].to_csv(out/'CROSS_COHORT_VALIDATION.tsv',sep='\t',index=False)
    fin=pd.read_parquet(out/'final/SELECTED_GENES.parquet');fold=pd.concat(fold_genes,ignore_index=True)
    if len(fold):
        st=fold.assign(positive=fold.coefficient.gt(0)).groupby(['target_id','lncrna_id']).agg(selected_folds=('phase','nunique'),positive_folds=('positive','sum')).reset_index();st['selection_frequency']=st.selected_folds/5;st['direction_agreement']=np.maximum(st.positive_folds,st.selected_folds-st.positive_folds)/st.selected_folds
        fin=fin.merge(st,on=['target_id','lncrna_id'],how='left',validate='one_to_one');fin['selected_folds']=fin.selected_folds.fillna(0);fin['selection_frequency']=fin.selection_frequency.fillna(0)
    fin=fin.merge(pd.read_csv(src/'FEATURES.tsv',sep='\t'),on='lncrna_id',how='left',validate='many_to_one');fin.to_csv(out/'FINAL_GENE_SETS.tsv.gz',sep='\t',index=False);fin.to_parquet(out/'FINAL_GENE_SETS.parquet',index=False)
    final=joblib.load(out/'final/MODELS.joblib');rows=[];nuisance=[]
    for tid,model in final.items():
        for k,col in enumerate(model['design_spec']['columns']):nuisance.append(dict(target_id=tid,covariate=col,coefficient=model['nuisance_coefficients'][k],covariate_only_coefficient=model['covariate_only_coefficients'][k],penalized=False))
    pd.DataFrame(nuisance).to_csv(out/'FINAL_COVARIATE_COEFFICIENTS.tsv',sep='\t',index=False)
    write_json(out/'FINAL_COVARIATE_ENCODING.json',{tid:m['design_spec'] for tid,m in final.items()})
    for tid,model in final.items():
        for k,cc in enumerate(model['groups']):rows.append(dict(target_id=tid,cohort=cc,intercept=model['intercept'][k],n_selected=len(model['feature_index']),alpha=model['alpha'],alpha_ratio=model['alpha_ratio'],unknown_cohort_intercept=model['unknown_intercept']))
    pd.DataFrame(rows).to_csv(out/'FINAL_INTERCEPTS.tsv',sep='\t',index=False);pd.concat(audits).to_parquet(out/'ALL_AUDITS.parquet',index=False)
    cov.reset_index().to_csv(out/'COVARIATES.tsv',sep='\t',index=False)
    patients.reset_index().to_csv(out/'PATIENTS.tsv',sep='\t',index=False)
    summary=dict(cancer_id=cancer,n_patients=len(x),cohort_counts=patients.cohort.value_counts().to_dict(),n_input_lncrnas=x.shape[1],n_final_models=len(final),nonempty_signatures=sum(len(m['feature_index'])>0 for m in final.values()),selected_gene_rows=len(fin),heldout_predictions=int(oof.n_oof.sum()),seconds=round(time.time()-t0))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',target_host='149',launch=launch,summary=summary,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}));print('CASE_COMPLETE',cancer,json.dumps(summary),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['aligned','cache','splits','covariates','cancer','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    if not 1<=a.workers<=24:raise ValueError('At most 24 CPU workers')
    run_case(a.aligned,a.cache,a.splits,a.covariates,a.cancer,a.output,a.workers)
