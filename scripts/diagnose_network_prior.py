"""Known-truth sensitivity diagnostic; synthetic results are not patient evidence."""
import argparse,json,os,socket,time
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from cc_hhgt.graph_ridge import fit_transform,ridge_path
from cc_hhgt.network_ridge import center_trace,alignment_weights,weighted_path,LAMBDAS,POWERS
from cla_engine.common import sha

def dump(path,value):path.write_text(json.dumps(value,indent=2))
def kernels(raw,train,assignment):
    x,spec=fit_transform(raw,train);assignment=assignment[spec['index']]
    rawk=np.stack([x[:,assignment==g]@x[:,assignment==g].T for g in sorted(set(assignment))])
    mass=np.array([center_trace(k,train) for k in rawk]);total=mass.sum();mass/=total
    kg=rawk/(total*mass[:,None,None]);base=np.einsum('g,gij->ij',mass,kg)
    return kg,mass,base

def predictions(ctx,z,tr,query,y,power):
    kg,mass,base=ctx
    if power==0:return ridge_path(base,z,tr,query,y,LAMBDAS)[0]
    w=alignment_weights(kg,mass,z,tr,y,power)
    return weighted_path(kg,w,z,tr,query,y,LAMBDAS)[0]

def run(output):
    assert socket.gethostname()=='149'
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'PLAN.json').exists():raise ValueError('USE_NEW_OUTPUT')
    plan=dict(target_host='149',status='FIXED_BEFORE_SIMULATION',seeds=list(range(202611060,202611070)),training_n=100,independent_test_n=200,features=400,modules=20,module_size=20,signal_sd=3.,noise_sd=1.,scenarios=['one_module_positive','one_module_opposite_signs','signal_dispersed','no_feature_signal'],inner_repeats=2,inner_folds=3,lambda_grid=LAMBDAS.tolist(),power_grid=list(POWERS),models=['equal_ridge','true_module','shuffled_module','oracle_support'],oracle='Known nonzero features; diagnostic comparator only',code_sha256=sha(Path(__file__)),real_patient_data_used=False,patient_experiment_changed=False,numerical_zero_tolerance=1e-10)
    dump(out/'PLAN.json',plan);results=[]
    for seed in plan['seeds']:
        started=time.time();rng=np.random.default_rng(seed);n=300;p=400;assignment=np.repeat(np.arange(20),20);shuffled=rng.permutation(assignment);x=rng.normal(size=(n,p));z=np.column_stack([np.ones(n),rng.normal(size=n)]);beta=np.zeros((p,4));beta[:20,0]=3/np.sqrt(20);beta[:20,1]=np.tile([1.,-1.],10)*3/np.sqrt(20);beta[:,2]=rng.normal(size=p);beta[:,2]*=3/np.linalg.norm(beta[:,2]);y=x@beta+z@np.array([[.5]*4,[1.]*4])+rng.normal(size=(n,4));tr=np.arange(100);te=np.arange(100,300);splits=[]
        for rep in range(2):splits.extend(KFold(3,shuffle=True,random_state=seed+rep).split(tr))
        assignments={'equal_ridge':np.zeros(p,int),'true_module':assignment,'shuffled_module':shuffled}
        for method,groups in assignments.items():
            powers=(0.,) if method=='equal_ridge' else POWERS;loss=np.zeros((len(powers),len(LAMBDAS),4));den=0
            for a,b in splits:
                ctx=kernels(x,a,groups);den+=len(b)
                for k,power in enumerate(powers):loss[k]+=((predictions(ctx,z,a,b,y[a],power)-y[b][None])**2).sum(axis=1)
            best=(loss/den).reshape(-1,4).argmin(axis=0);ctx=kernels(x,tr,groups);pred=np.empty((len(te),4))
            for oi,power in enumerate(powers):
                take=np.flatnonzero(best//len(LAMBDAS)==oi)
                if not len(take):continue
                pp=predictions(ctx,z,tr,te,y[tr][:,take],power)
                for j,t in enumerate(take):pred[:,t]=pp[best[t]%len(LAMBDAS),:,j]
            for j,scenario in enumerate(plan['scenarios']):
                mse=float(np.mean((pred[:,j]-y[te,j])**2));results.append(dict(seed=seed,scenario=scenario,method=method,test_r2=1-mse/np.var(y[te,j]),test_mse=mse,power=float(powers[best[j]//len(LAMBDAS)]),lam=float(LAMBDAS[best[j]%len(LAMBDAS)])))
        for j,scenario in enumerate(plan['scenarios']):
            support=np.flatnonzero(beta[:,j]);truth=y[te,j]
            if not len(support):pred=z[te]@np.linalg.lstsq(z[tr],y[tr,j],rcond=None)[0];lam=np.nan
            else:
                xx=x[:,support];groups=np.zeros(len(support),int);loss=np.zeros(len(LAMBDAS));den=0
                for a,b in splits:
                    ctx=kernels(xx,a,groups);pp=predictions(ctx,z,a,b,y[a,j:j+1],0)[:,:,0];loss+=((pp-y[b,j][None])**2).sum(axis=1);den+=len(b)
                li=int(np.argmin(loss/den));ctx=kernels(xx,tr,groups);pred=predictions(ctx,z,tr,te,y[tr,j:j+1],0)[li,:,0];lam=float(LAMBDAS[li])
            mse=float(np.mean((pred-truth)**2));results.append(dict(seed=seed,scenario=scenario,method='oracle_support',test_r2=1-mse/np.var(truth),test_mse=mse,power=np.nan,lam=lam))
        pd.DataFrame(results).to_csv(out/'PER_SEED.tsv',sep='\t',index=False);print('SIMULATION_SEED_COMPLETE',seed,'seconds',round(time.time()-started,2),flush=True)
    frame=pd.DataFrame(results);summary=frame.groupby(['scenario','method']).agg(mean_test_r2=('test_r2','mean'),sd_test_r2=('test_r2','std'),minimum_test_r2=('test_r2','min'),maximum_test_r2=('test_r2','max'),mean_test_mse=('test_mse','mean'),seeds=('seed','nunique')).reset_index();summary.to_csv(out/'SUMMARY.tsv',sep='\t',index=False)
    wide=frame.pivot(index=['seed','scenario'],columns='method',values='test_r2');contrast=[]
    for scenario,f in wide.groupby(level='scenario'):
        for base in ['equal_ridge','shuffled_module','oracle_support']:
            delta=f.true_module-f[base];contrast.append(dict(scenario=scenario,baseline=base,mean_delta_r2=float(delta.mean()),median_delta_r2=float(delta.median()),improved_seeds=int((delta>1e-10).sum()),tied_seeds=int((delta.abs()<=1e-10).sum()),seeds=len(delta)))
    pd.DataFrame(contrast).to_csv(out/'PAIRED_CONTRASTS.tsv',sep='\t',index=False);dump(out/'COMPLETE.json',dict(target_host='149',status='COMPLETE',interpretation='Known-truth simulated sensitivity only; does not establish patient benefit, topology validity, or biological causality',files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print(summary.to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
