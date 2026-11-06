"""Paired patient bootstrap; correlated targets averaged within score families."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_graph_ridge import dump
from cla_engine.common import sha
ARMS=('lncrna','mrna','combined')
METHODS=('ridge_equal','ridge_adaptive_omics','rna_laplacian_union','module_ridge_union')+tuple('module_ridge_union_random_'+str(s) for s in [101,202,303,404,505])
LAYERS=('PROTEIN_PATHWAY','PHOSPHO_PTMSEA')

def family_mean_bootstrap(delta,valid,families,counts):
    den=valid.astype(float)@counts.T;num=np.where(valid,delta,0)@counts.T;boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    point=np.nansum(np.where(valid,delta,np.nan),axis=1)/valid.sum(1)
    values=[];points=[]
    for f in sorted(set(families)):
        take=families==f;values.append(np.nanmean(boot[take],axis=0));points.append(np.mean(point[take]))
    return float(np.mean(points)),np.mean(values,axis=0)

def bootstrap(source,inputs,summary,replicates=2000):
    assert socket.gethostname()=='149';source=Path(source);inputs=Path(inputs);out=Path(summary)
    if not (out/'SUMMARY_COMPLETE.json').exists():raise ValueError('SUMMARY_NOT_VERIFIED')
    dump(out/'BOOTSTRAP_PREFLIGHT.json',dict(target_host='149',source=str(source),summary_receipt=sha(out/'SUMMARY_COMPLETE.json'),code=sha(Path(__file__)),replicates=replicates,unit='patient within cancer',paired_repeats=True,target_aggregation='equal families, equal targets within family, equal cancers',interval='conditional on existing fitted CV models; does not repeat model fitting',target_variance='Original common-support variance fixed; bootstrap reweights patient losses in Delta R2 units',null_baseline='mean squared prediction error across five independently fitted degree-preserving null graphs; not prediction averaging'))
    met=pd.read_parquet(out/'ALL_TARGET_METRICS.parquet');met=met[met.success&met.display_layer.isin(LAYERS)&met.variant.isin(METHODS)&met.arm.isin(ARMS)]
    rows=[];aggregate={};n_cancer={}
    for ci,c in enumerate(['BRCA','LUAD','COAD']):
        mm=met[met.cancer_id.eq(c)];num=mm.groupby('target_id').size();tids=sorted(num[num.eq(2*len(METHODS)*3)].index);meta=mm.drop_duplicates('target_id').set_index('target_id').loc[tids];pids=pd.read_parquet(inputs/c/'COVARIATES.parquet').index;ti={x:i for i,x in enumerate(tids)};pi={x:i for i,x in enumerate(pids)};mi={x:i for i,x in enumerate(METHODS)};ai={x:i for i,x in enumerate(ARMS)}
        predictions=np.full((2,len(METHODS),3,len(tids),len(pids)),np.nan)
        for repeat in range(2):
            for fold in range(5):
                p=pd.read_parquet(source/c/f'repeat{repeat}_outer{fold}'/'PREDICTIONS.parquet',columns=['patient_id','target_id','variant','arm','predicted'],filters=[('target_id','in',tids),('variant','in',list(METHODS)),('arm','in',list(ARMS))]);a=p.target_id.map(ti).to_numpy(int);b=p.patient_id.map(pi).to_numpy(int);m=p.variant.map(mi).to_numpy(int);arm=p.arm.map(ai).to_numpy(int);predictions[repeat,m,arm,a,b]=p.predicted.to_numpy()
        y=pd.read_parquet(inputs/c/'Y.parquet',columns=tids).reindex(pids).to_numpy(float).T
        valid=np.isfinite(predictions).all(axis=(0,1,2))&np.isfinite(y);nn=valid.sum(1);mu=np.nansum(np.where(valid,y,np.nan),axis=1)/np.maximum(nn,1);variance=np.nansum(np.where(valid,(y-mu[:,None])**2,np.nan),axis=1)/np.maximum(nn,1);eligible=(nn>=12)&(variance>1e-12)
        err=np.mean((predictions-y[None,None,None])**2,axis=0)/np.maximum(variance[None,None,:,None],1e-30)
        rng=np.random.default_rng(20261106+ci);counts=rng.multinomial(len(pids),np.repeat(1/len(pids),len(pids)),size=replicates)
        contrasts=[]
        for arm in ARMS:
            for baseline in ['ridge_adaptive_omics','ridge_equal','rna_laplacian_union']:
                contrasts.append((arm,'module_ridge_union',baseline,err[mi[baseline],ai[arm]]-err[mi['module_ridge_union'],ai[arm]]))
            nullerr=np.mean(err[[mi['module_ridge_union_random_'+str(s)] for s in [101,202,303,404,505]],ai[arm]],axis=0);contrasts.append((arm,'module_ridge_union','mean_five_null_models',nullerr-err[mi['module_ridge_union'],ai[arm]]))
        for method in ['module_ridge_union','ridge_adaptive_omics','ridge_equal']:
            contrasts.append(('combined_vs_mrna',method,method,err[mi[method],ai['mrna']]-err[mi[method],ai['combined']]))
            contrasts.append(('lncrna_vs_mrna',method,method,err[mi[method],ai['mrna']]-err[mi[method],ai['lncrna']]))
        for layer in LAYERS:
            take=(meta.display_layer.to_numpy()==layer)&eligible;families=meta.family.fillna('UNSPECIFIED').to_numpy()[take]
            if not take.any():continue
            for arm,method,baseline,delta in contrasts:
                point,boot=family_mean_bootstrap(delta[take],valid[take],families,counts);key=(layer,arm,method,baseline);aggregate.setdefault(key,[]).append((point,boot));n_cancer.setdefault(key,[]).append(c)
                rows.append(dict(cancer_id=c,layer=layer,arm=arm,method=method,baseline=baseline,targets=int(take.sum()),patients=len(pids),family_balanced_mean_delta_r2=point,ci_low=float(np.percentile(boot,2.5)),ci_high=float(np.percentile(boot,97.5)),replicates=replicates))
        print('PATIENT_BOOTSTRAP_CANCER',c,'targets',int(eligible.sum()),flush=True)
    for key,values in aggregate.items():
        point=float(np.mean([p for p,b in values]));boot=np.mean([b for p,b in values],axis=0);layer,arm,method,baseline=key;rows.append(dict(cancer_id='EQUAL_CANCER_MEAN',layer=layer,arm=arm,method=method,baseline=baseline,targets=None,patients=None,family_balanced_mean_delta_r2=point,ci_low=float(np.percentile(boot,2.5)),ci_high=float(np.percentile(boot,97.5)),replicates=replicates,cancers=';'.join(n_cancer[key])))
    pd.DataFrame(rows).to_csv(out/'PATIENT_BOOTSTRAP.tsv',sep='\t',index=False);dump(out/'BOOTSTRAP_COMPLETE.json',dict(target_host='149',status='COMPLETE',files={n:sha(out/n) for n in ['PATIENT_BOOTSTRAP.tsv','BOOTSTRAP_PREFLIGHT.json']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--inputs',required=True);p.add_argument('--summary',required=True);p.add_argument('--replicates',type=int,default=2000);a=p.parse_args();bootstrap(a.source,a.inputs,a.summary,a.replicates)
