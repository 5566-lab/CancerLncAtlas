"""Extend the existing three-arm CPTAC demo with graph-ridge multiomics."""
import argparse,json,socket,sys,time,collections
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from cc_hhgt.m0_covariates import complete_rows
from cc_hhgt.m0_three_arm import regression_metrics,compare_target
from cc_hhgt.graph_ridge import *
from prepare_graph_ridge import dump
from cc_hhgt.network_evidence import require_physical_network_receipt
VARIANTS=('rna_only','multiomics_ridge','multiomics_graph','omics_only')

def run(source,output,cancer):
    if socket.gethostname()!='149':raise RuntimeError('HOST_149_REQUIRED')
    src=Path(source)/cancer;out=Path(output)/cancer;out.mkdir(parents=True,exist_ok=True)
    require_physical_network_receipt(json.loads((src/'NETWORK_AUDIT.json').read_text()))
    launch=dict(target_host='149',cancer=cancer,input_receipt=sha(src/'COMPLETE.json'),code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'cc_hhgt/graph_ridge.py',ROOT/'cc_hhgt/m0_covariates.py',ROOT/'cc_hhgt/m0_three_arm.py',ROOT/'cc_hhgt/network_evidence.py']},etas=ETAS,lambdas=LAMBDAS.tolist(),split_policy='UNCHANGED_M0_FIVE_FOLDS',preprocessing='TRAIN_ONLY_X_MEDIAN_SCALE_QUALITY; all covariate-complete training patients including those without specific Y; no held-out fitting',variants=VARIANTS)
    if (out/'LAUNCH_PREFLIGHT.json').exists() and json.loads((out/'LAUNCH_PREFLIGHT.json').read_text())!=json.loads(json.dumps(launch)):raise ValueError('EXPERIMENT_DRIFT')
    dump(out/'LAUNCH_PREFLIGHT.json',launch)
    rec=json.loads((src/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():
        if sha(src/name)!=h:raise ValueError('INPUT_DRIFT: '+name)
    blocks={b:pd.read_parquet(src/(b+'.parquet')) for b in ['RNA']+list(BLOCKS)}
    pids=blocks['RNA'].index
    for b in blocks:
        if not blocks[b].index.equals(pids):raise ValueError('PATIENT_ORDER_MISMATCH')
        blocks[b]=blocks[b].to_numpy(float)
    features=pd.read_csv(src/'RNA_FEATURES.tsv',sep='\t');lap=sparse.load_npz(src/'RNA_LAPLACIAN.npz')
    cov=pd.read_parquet(src/'COVARIATES.parquet').reindex(pids);yf=pd.read_parquet(src/'Y.parquet').reindex(pids)
    meta=pd.read_csv(src/'TARGETS.tsv',sep='\t');meta['n_observed_y']=meta.target_id.map(yf.notna().sum());policies=[target_policy(row) for row in meta.to_dict('records')]
    meta['role']=[p['role'] for p in policies];meta['excluded_blocks']=[';'.join(p['excluded_blocks']) for p in policies];meta['allowed_blocks']=[';'.join(p['allowed_blocks']) for p in policies]
    meta.to_csv(out/'TARGETS.tsv',sep='\t',index=False);split=json.loads((src/'SPLITS.json').read_text());dump(out/'SPLITS.json',split)
    y=yf[meta.target_id].to_numpy(float);groups={}
    for j,p in enumerate(policies):
        if meta.iloc[j].n_observed_y<12:continue
        valid=np.isfinite(y[:,j])&complete_rows(cov,p['use_purity']);key=(p['use_purity'],p['allowed_blocks'],tuple(valid))
        groups.setdefault(key,[]).append(j)
    print('INPUT_READY',cancer,'targets',len(meta),'observed_groups',len(groups),'patients',len(pids),flush=True)
    chosen=collections.defaultdict(list)
    for phase in [f'outer{k}' for k in range(5)]+['final']:
        d=out/phase;d.mkdir(exist_ok=True)
        if (d/'COMPLETE.json').exists():
            for name,h in json.loads((d/'COMPLETE.json').read_text())['files'].items():
                if sha(d/name)!=h:raise ValueError('CHECKPOINT_DRIFT')
            audits=pd.read_parquet(d/'AUDIT.parquet')
            for row in audits.loc[audits.status.eq('FITTED')].to_dict('records'):chosen[(row['target_id'],row['arm'],row['variant'])].append((row['eta'],row['lam']))
            print('RESUME',cancer,phase,flush=True);continue
        final=phase=='final';outer=np.asarray(split[phase]['train'],int);test=np.asarray(split[phase]['test'],int)
        inner=None if final else split['inner'+phase[5:]];cache={};audit=[];predictions=[];models={}
        def context(which,purity):
            key=(which,purity)
            if key in cache:return cache[key]
            rows=outer if which=='outer' else np.asarray(inner['train'],int)
            cp=d/(which+'_purity'+str(int(purity))+'.joblib')
            if cp.exists():ctx=joblib.load(cp)
            else:
                def prog(a,e,st):print('KERNEL',cancer,phase,which,purity,a,e,st['iterations'],flush=True)
                ctx=make_context(blocks,features,lap,cov,rows,purity,prog);joblib.dump(ctx,cp,compress=3)
            cache[key]=ctx;return ctx
        for g,(key,js) in enumerate(groups.items(),1):
            purity,allowed,valid_tuple=key;valid=np.asarray(valid_tuple,bool);oc=context('outer',purity)
            ft=outer[valid[outer]];te=test[valid[test]&oc['supported'][test]]
            ic=None if final else context('inner',purity)
            tr=None if final else np.array(inner['train'],int);va=None if final else np.array(inner['validation'],int)
            if not final:tr=tr[valid[tr]];va=va[valid[va]&ic['supported'][va]]
            ok=(len(ft)>=12) and (final or (len(tr)>=12 and len(va)>=3 and len(te)>0))
            active=[]
            for j in js:
                status='INSUFFICIENT_SUPPORTED_OBSERVATIONS' if not ok else 'CONSTANT_TRAIN_Y' if np.std(y[ft,j])<1e-12 or (not final and np.std(y[tr,j])<1e-12) else None
                if status:
                    for variant in VARIANTS:
                        for arm in (['shared'] if variant=='omics_only' else ARMS):audit.append(dict(target_id=meta.iloc[j].target_id,arm=arm,variant=variant,phase=phase,status=status))
                else:active.append(j)
            js=active
            if not js:continue
            tids=meta.target_id.iloc[js].tolist();yy=y[np.ix_(ft,js)]
            for variant in VARIANTS:
                arms=['shared'] if variant=='omics_only' else ARMS
                for arm in arms:
                    options=ETAS if variant=='multiomics_graph' else (0.,)
                    if final:
                        eta_choice=[];lam_choice=[]
                        for tid in tids:
                            history=chosen[(tid,arm,variant)]
                            if len(history)<3:eta_choice.append(np.nan);lam_choice.append(np.nan);continue
                            freq=collections.Counter(e for e,l in history);eta_choice.append(sorted(freq,key=lambda e:(-freq[e],e))[0]);lm=float(np.median(np.log10([l for e,l in history])));lam_choice.append(float(LAMBDAS[np.argmin(abs(np.log10(LAMBDAS)-lm))]))
                        eta_choice=np.array(eta_choice);lam_choice=np.array(lam_choice)
                    else:
                        losses=[]
                        for eta in options:
                            k=combined_kernel(ic,arm,eta,allowed,variant);pp,_=ridge_path(k,ic['design'],tr,va,y[np.ix_(tr,js)])
                            losses.append(((pp-y[np.ix_(va,js)][None,:,:])**2).mean(axis=1))
                        loss=np.stack(losses);best=loss.reshape(-1,len(js)).argmin(axis=0);eta_choice=np.asarray(options)[best//len(LAMBDAS)];lam_choice=LAMBDAS[best%len(LAMBDAS)]
                    for eta in options:
                        take=np.flatnonzero(eta_choice==eta)
                        if not len(take):continue
                        k=combined_kernel(oc,arm,eta,allowed,variant);pp,st=ridge_path(k,oc['design'],ft,te,yy[:,take])
                        cb=oc['design'][te]@st['covbeta']
                        for jj,pos in enumerate(take):
                            tid=tids[pos];lam=float(lam_choice[pos]);li=int(np.argmin(abs(LAMBDAS-lam)))
                            audit.append(dict(target_id=tid,arm=arm,variant=variant,phase=phase,status='FITTED',eta=float(eta),lam=lam,n_train=len(ft),n_test=len(te),n_inner_train=0 if final else len(tr),n_inner_validation=0 if final else len(va),purity_covariate_used=purity,excluded_blocks=';'.join(policies[js[pos]]['excluded_blocks']),allowed_blocks=';'.join(allowed)))
                            if not final:
                                chosen[(tid,arm,variant)].append((float(eta),lam))
                                predictions.append(pd.DataFrame(dict(patient_id=pids[te],cohort='CPTAC',target_id=tid,phase=phase,variant=variant,arm=arm,observed=y[te,js[pos]],predicted=pp[li,:,jj],train_mean_baseline=float(yy[:,pos].mean()),covariate_only_baseline=cb[:,jj])))
                            else:
                                state={key:(val[:,jj:jj+1] if key in ['uy','covbeta'] else val) for key,val in st.items()}
                                model=extract_fit(state,lam,yy[:,pos:pos+1]);model.update(eta=float(eta),allowed_blocks=allowed,variant=variant,arm=arm,context_file='outer_purity'+str(int(purity))+'.joblib');models[(tid,arm,variant)]=model
                    for pos in np.flatnonzero(~np.isfinite(eta_choice)):audit.append(dict(target_id=tids[pos],arm=arm,variant=variant,phase=phase,status='INSUFFICIENT_TUNING_HISTORY'))
            print('GROUP_COMPLETE',cancer,phase,g,'/',len(groups),'targets',len(js),'audit',len(audit),flush=True)
        pd.DataFrame(audit).to_parquet(d/'AUDIT.parquet',index=False)
        (pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=['patient_id','cohort','target_id','phase','variant','arm','observed','predicted','train_mean_baseline','covariate_only_baseline'])).to_parquet(d/'PREDICTIONS.parquet',index=False)
        if final:joblib.dump(models,d/'MODELS.joblib',compress=3)
        dump(d/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
        print('PHASE_COMPLETE',cancer,phase,flush=True)
    summarize(out,meta)

def summarize(out,meta):
    from cc_hhgt.m0_three_arm import matched_predictions
    phases=[f'outer{k}' for k in range(5)]+['final']
    aud=pd.concat([pd.read_parquet(out/p/'AUDIT.parquet') for p in phases],ignore_index=True)
    pred=pd.concat([pd.read_parquet(out/p/'PREDICTIONS.parquet') for p in phases[:-1]],ignore_index=True)
    status=[];metrics=[];comparisons=[]
    for variant in VARIANTS:
        arms=['shared'] if variant=='omics_only' else ARMS;successful={}
        for arm in arms:
            a=aud[aud.arm.eq(arm)&aud.variant.eq(variant)];f=pred[pred.arm.eq(arm)&pred.variant.eq(variant)]
            phases_ok=a[a.status.eq('FITTED')].groupby('target_id').phase.nunique();evaluated=f.groupby('target_id').phase.nunique()
            successful[arm]=set(phases_ok[phases_ok.eq(6)].index)&set(evaluated[evaluated.eq(5)].index)
            for tid in meta.target_id:
                aa=a[a.target_id.eq(tid)];status.append(dict(target_id=tid,arm=arm,variant=variant,training_success=tid in successful[arm],attempted=bool(len(aa)),fitted_phases=int(aa.status.eq('FITTED').sum())))
            for tid,ff in f.groupby('target_id'):
                if tid in successful[arm]:metrics.append(dict(target_id=tid,arm=arm,variant=variant,**regression_metrics(ff.observed,ff.predicted)))
        if variant=='omics_only':continue
        common=set.intersection(*(successful[a] for a in ARMS));paired=matched_predictions({a:pred[pred.arm.eq(a)&pred.variant.eq(variant)&pred.target_id.isin(common)].drop(columns=['arm','variant']) for a in ARMS})
        if paired.duplicated(['target_id','patient_id']).any():raise ValueError('OOF_PATIENT_REPEATED')
        for tid,ff in paired.groupby('target_id'):comparisons.append(dict(target_id=tid,variant=variant,**compare_target(ff,bootstrap=1000)))
    st=pd.DataFrame(status).merge(meta,on='target_id',validate='many_to_one');st.to_csv(out/'TARGET_TRAINING_STATUS.tsv',sep='\t',index=False)
    pd.DataFrame(metrics).merge(meta,on='target_id',validate='many_to_one').to_csv(out/'ARM_TARGET_METRICS.tsv',sep='\t',index=False)
    pd.DataFrame(comparisons).merge(meta,on='target_id',validate='many_to_one').to_csv(out/'TARGET_COMPARISON.tsv',sep='\t',index=False)
    pred.to_parquet(out/'OOF_PREDICTIONS.parquet',index=False);aud.to_parquet(out/'AUDIT.parquet',index=False)
    dump(out/'COMPLETE.json',dict(target_host='149',status='COMPLETE',comparisons=len(comparisons),files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'COMPLETE.json'}))
    print('CANCER_COMPLETE',out.name,len(comparisons),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cancer',choices=['BRCA','LUAD','COAD'],required=True);a=p.parse_args();run(a.source,a.output,a.cancer)
