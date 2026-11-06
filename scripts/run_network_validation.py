"""Repeated nested-CV network validation; one resumable phase per CPU job."""
from __future__ import annotations
import argparse,collections,json,socket,sys,time,gc
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump
from cc_hhgt.network_ridge import *
from cc_hhgt.graph_ridge import target_policy,extract_fit
from cc_hhgt.m0_covariates import complete_rows
from cc_hhgt.network_evidence import require_physical_network_receipt

def split_plan(original,n):
    phases={}
    for repeat in range(2):
        pairs=[(np.array(original['outer'+str(f)]['train']),np.array(original['outer'+str(f)]['test'])) for f in range(5)] if repeat==0 else list(KFold(5,shuffle=True,random_state=20261106).split(np.arange(n)))
        for f,(tr,te) in enumerate(pairs):phases['repeat'+str(repeat)+'_outer'+str(f)]=dict(train=tr.tolist(),test=te.tolist(),repeat=repeat,fold=f)
    phases['final']=dict(train=list(range(n)),test=[],repeat=-1,fold=-1)
    for name,p in phases.items():
        outer=np.asarray(p['train'],int);p['inner']=[]
        for ir in range(2):
            seed=20261106+1000*(p['repeat']+1)+10*(p['fold']+1)+ir
            for tr,va in KFold(3,shuffle=True,random_state=seed).split(outer):p['inner'].append(dict(train=outer[tr].tolist(),validation=outer[va].tolist(),seed=seed))
    return phases

def check_receipt(path):
    for name,h in json.loads((path/'COMPLETE.json').read_text())['files'].items():
        if sha(path/name)!=h:raise ValueError('INPUT_CHECKSUM_DRIFT: '+str(path/name))

def reweight(weights1,mass,power):
    w=mass[:,None]*(weights1/mass[:,None])**power;return w/w.sum(axis=0,keepdims=True)

def run(inputs,networks,output,cancer,phase):
    if socket.gethostname()!='149':raise RuntimeError('HOST_149_REQUIRED')
    src=Path(inputs)/cancer;net=Path(networks)/cancer;out=Path(output)/cancer;d=out/phase;d.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),ROOT/'cc_hhgt/network_ridge.py',ROOT/'cc_hhgt/graph_ridge.py',ROOT/'cc_hhgt/m0_covariates.py',ROOT/'cc_hhgt/network_evidence.py']
    preflight=dict(target_host='149',cancer=cancer,phase=phase,input_receipt=sha(src/'COMPLETE.json'),network_receipt=sha(net/'COMPLETE.json'),plan=sha(ROOT/'config/network_validation_20261106.json'),code={str(p.relative_to(ROOT)):sha(p) for p in code},preprocessing='Train-only quality, medians, scale, assay clipping; clinical complete cases; same X preprocessing pool for all targets of a purity policy',version=1)
    if (d/'LAUNCH_PREFLIGHT.json').exists() and json.loads((d/'LAUNCH_PREFLIGHT.json').read_text())!=preflight:raise ValueError('RUN_CONFIGURATION_DRIFT')
    dump(d/'LAUNCH_PREFLIGHT.json',preflight)
    if (d/'COMPLETE.json').exists():check_receipt(d);print('RESUME_PHASE_COMPLETE',cancer,phase,flush=True);return
    check_receipt(src);check_receipt(net)
    require_physical_network_receipt(json.loads((src/'NETWORK_AUDIT.json').read_text()))
    require_physical_network_receipt(json.loads((net/'CORRECTED_RNA_NETWORK_AUDIT.json').read_text()))
    blocks={b:pd.read_parquet(src/(b+'.parquet')) for b in ('RNA',)+BLOCKS};pids=blocks['RNA'].index
    for b in blocks:
        if not blocks[b].index.equals(pids):raise ValueError('PATIENT_AXIS_DRIFT')
        blocks[b]=blocks[b].to_numpy(float)
    features=pd.read_csv(src/'RNA_FEATURES.tsv',sep='\t');fmap=pd.read_parquet(net/'FEATURE_MAP.parquet');cov=pd.read_parquet(src/'COVARIATES.parquet').reindex(pids);yf=pd.read_parquet(src/'Y.parquet').reindex(pids);meta=pd.read_csv(src/'TARGETS.tsv',sep='\t');policies=[target_policy(r) for r in meta.to_dict('records')]
    plan=split_plan(json.loads((src/'SPLITS.json').read_text()),len(pids));dump(d/'SPLIT.json',plan[phase]);sp=plan[phase];outer=np.asarray(sp['train'],int);test=np.asarray(sp['test'],int);final=phase=='final';y=yf[meta.target_id].to_numpy(float)
    groups={}
    for j,p in enumerate(policies):
        valid=np.isfinite(y[:,j])&complete_rows(cov,p['use_purity']);groups.setdefault((p['use_purity'],p['allowed_blocks'],tuple(valid)),[]).append(j)
    print('INPUT_READY',cancer,phase,'patients',len(pids),'targets',len(meta),'groups',len(groups),flush=True)
    contexts={};models={};allpred=[];allaudit=[]
    def get_contexts(purity):
        if purity in contexts:return contexts[purity]
        result=[]
        for name,rows in [('outer',outer)]+[('inner'+str(i),np.asarray(s['train'],int)) for i,s in enumerate(sp['inner'])]:
            t=time.time();ctx=make_context(blocks,features,fmap,cov,rows,purity);result.append(ctx);print('CONTEXT',cancer,phase,purity,name,'seconds',round(time.time()-t,2),flush=True)
        laps={mode:sparse.load_npz(net/('LEGACY_'+mode+'.npz')) for mode in MODES}
        add_legacy_kernels(result,laps,lambda mode,eta:print('LEGACY_KERNEL',cancer,phase,purity,mode,eta,flush=True))
        contexts[purity]=result
        if final:
            joblib.dump({k:v for k,v in result[0].items() if k in ['train','design','design_spec','supported','preprocess','scales']},d/('PREPROCESS_purity'+str(int(purity))+'.joblib'),compress=3)
        return result
    specs=[('rna_only','base',None),('omics_only','base',None),('ridge_equal','base',None),('ridge_adaptive_omics','module','omics')]
    for mode in MODES:
        suffix=mode if not mode.startswith('random_') else 'union_'+mode
        specs.extend([('rna_laplacian_'+suffix,'legacy',mode),('module_ridge_'+suffix,'module',mode)])
    for gi,(key,jlist) in enumerate(groups.items()):
        purity,allowed,vv=key;valid=np.asarray(vv,bool);gcpath=d/('group_'+str(gi)+'.joblib')
        if gcpath.exists():
            saved=joblib.load(gcpath);allpred.extend(saved['predictions']);allaudit.extend(saved['audit']);models.update(saved['models']);print('RESUME_GROUP',gi,flush=True);continue
        contexts_now=get_contexts(purity);oc=contexts_now[0];ics=contexts_now[1:];ft=outer[valid[outer]];te=test[valid[test]&oc['supported'][test]]
        pairs=[]
        for s,ic in zip(sp['inner'],ics):
            tr=np.asarray(s['train'],int);va=np.asarray(s['validation'],int);pairs.append((tr[valid[tr]],va[valid[va]&ic['supported'][va]]))
        enough=len(ft)>=12 and (final or len(te)>0) and all(len(tr)>=12 and len(va)>=3 for tr,va in pairs)
        js=[];audit=[];predictions=[];gmodels={}
        for j in jlist:
            reason=None
            if not enough:reason='INSUFFICIENT_SUPPORTED_NESTED_CV_OBSERVATIONS'
            elif np.std(y[ft,j])<1e-12 or any(np.std(y[tr,j])<1e-12 for tr,va in pairs):reason='CONSTANT_TRAINING_TARGET'
            if reason:
                for variant,kind,mode in specs:
                    for arm in (['shared'] if variant=='omics_only' else ARMS):audit.append(dict(target_id=meta.iloc[j].target_id,variant=variant,arm=arm,status=reason,phase=phase,n_train=len(ft),n_test=len(te)))
            else:js.append(j)
        if js:
            yy=y[np.ix_(ft,js)];tids=meta.target_id.iloc[js].tolist();base_losses={};validation_count=sum(len(va) for tr,va in pairs)
            for variant,kind,mode in specs:
                for arm in (['shared'] if variant=='omics_only' else ARMS):
                    started=time.time();options=(0.,) if kind=='base' else ETAS if kind=='legacy' else POWERS
                    losses=np.zeros((len(options),len(LAMBDAS),len(js)))
                    if kind=='base':
                        for ic,(tr,va) in zip(ics,pairs):
                            k=base_kernel(ic,arm,allowed,variant);pp,_=ridge_path(k,ic['design'],tr,va,y[np.ix_(tr,js)],LAMBDAS);losses[0]+=np.sum((pp-y[np.ix_(va,js)][None])**2,axis=1)
                        if variant=='ridge_equal':base_losses[arm]=losses[0].copy()
                    else:
                        losses[0]=base_losses[arm]
                        for ic,(tr,va) in zip(ics,pairs):
                            target=y[np.ix_(tr,js)];truth=y[np.ix_(va,js)]
                            if kind=='module':kg,mass,labels=group_kernels(ic,arm,allowed,mode);w1=alignment_weights(kg,mass,ic['design'],tr,target,1.)
                            for oi,param in enumerate(options[1:],1):
                                if kind=='legacy':pp,_=ridge_path(legacy_kernel(ic,mode,arm,allowed,param),ic['design'],tr,va,target,LAMBDAS)
                                else:pp,_=weighted_path(kg,reweight(w1,mass,param),ic['design'],tr,va,target,LAMBDAS)
                                losses[oi]+=np.sum((pp-truth[None])**2,axis=1)
                    losses/=validation_count;best=losses.reshape(-1,len(js)).argmin(axis=0);choice=np.asarray(options)[best//len(LAMBDAS)];lchoice=LAMBDAS[best%len(LAMBDAS)]
                    kg=mass=labels=w1=None
                    if kind=='module':kg,mass,labels=group_kernels(oc,arm,allowed,mode);w1=alignment_weights(kg,mass,oc['design'],ft,yy,1.)
                    for param in options:
                        take=np.flatnonzero(choice==param)
                        if not len(take):continue
                        if kind=='base':k=base_kernel(oc,arm,allowed,variant)
                        elif kind=='legacy':k=legacy_kernel(oc,mode,arm,allowed,param)
                        elif param==0:k=base_kernel(oc,arm,allowed)
                        if kind=='module' and param!=0:
                            weights=reweight(w1[:,take],mass,param);pp,states=weighted_path(kg,weights,oc['design'],ft,te,yy[:,take],LAMBDAS,return_fits=final)
                        else:
                            pp,state=ridge_path(k,oc['design'],ft,te,yy[:,take],LAMBDAS)
                        for jj,pos in enumerate(take):
                            li=int(np.argmin(abs(LAMBDAS-lchoice[pos])));tid=tids[pos];record=dict(target_id=tid,variant=variant,arm=arm,status='FITTED',phase=phase,parameter=float(param),lam=float(lchoice[pos]),inner_mse=float(losses.reshape(-1,len(js))[best[pos],pos]),n_train=len(ft),n_test=len(te),n_inner_train_min=min(len(tr) for tr,va in pairs),n_inner_validation_min=min(len(va) for tr,va in pairs),inner_validations=len(pairs),purity_covariate_used=purity,allowed_blocks=';'.join(allowed),excluded_blocks=';'.join(policies[js[pos]]['excluded_blocks']),role=policies[js[pos]]['role']);audit.append(record)
                            if final:
                                if kind=='module' and param!=0:fit=extract_weighted_fit(states[jj],lchoice[pos],weights[:,jj])
                                else:
                                    ss={key:(val[:,jj:jj+1] if key in ['uy','covbeta'] else val) for key,val in state.items()};fit=extract_fit(ss,lchoice[pos],yy[:,pos:pos+1]);fit['dual']=fit['dual'].ravel();fit['nuisance']=fit['nuisance'].ravel()
                                    if kind=='module':fit['group_weights']=mass
                                fit.update(variant=variant,kind=kind,mode=mode,arm=arm,parameter=float(param),allowed_blocks=allowed,purity=purity,group_labels=labels if kind=='module' else None);gmodels[(tid,variant,arm)]=fit
                            else:predictions.append(pd.DataFrame(dict(patient_id=pids[te],target_id=tid,variant=variant,arm=arm,phase=phase,repeat=sp['repeat'],fold=sp['fold'],observed=y[te,js[pos]],predicted=pp[li,:,jj],train_mean_baseline=float(yy[:,pos].mean()))))
                    print('MODEL_COMPLETE',cancer,phase,gi,variant,arm,'targets',len(js),'seconds',round(time.time()-started,2),flush=True)
        saved=dict(audit=audit,predictions=predictions,models=gmodels);temp=gcpath.with_suffix('.tmp');joblib.dump(saved,temp,compress=3);temp.replace(gcpath);allaudit.extend(audit);allpred.extend(predictions);models.update(gmodels);print('GROUP_COMPLETE',cancer,phase,gi+1,'of',len(groups),flush=True)
    ad=pd.DataFrame(allaudit);ad.to_parquet(d/'AUDIT.parquet',index=False)
    pd.concat(allpred,ignore_index=True).to_parquet(d/'PREDICTIONS.parquet',index=False) if allpred else pd.DataFrame().to_parquet(d/'PREDICTIONS.parquet')
    if final:joblib.dump(models,d/'MODELS.joblib',compress=3)
    meta.to_csv(d/'TARGETS.tsv',sep='\t',index=False)
    dump(d/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'}));print('PHASE_COMPLETE',cancer,phase,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--networks',required=True);p.add_argument('--output',required=True);p.add_argument('--cancer',required=True,choices=['BRCA','LUAD','COAD']);p.add_argument('--phase',required=True);a=p.parse_args();run(a.inputs,a.networks,a.output,a.cancer,a.phase)
