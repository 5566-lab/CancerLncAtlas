"""Audit fitted patient identity, splits, exclusions and paired comparisons."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cc_hhgt.graph_ridge import target_policy
from cla_engine.common import sha
from prepare_graph_ridge import dump

def audit(source,inputs,output):
    assert socket.gethostname()=='149';source=Path(source);inputs=Path(inputs);results=[]
    for c in ['BRCA','LUAD','COAD']:
        d=source/c;s=inputs/c;spec=json.loads((d/'SPLITS.json').read_text());y=pd.read_parquet(s/'Y.parquet');meta=pd.read_csv(s/'TARGETS.tsv',sep='\t').set_index('target_id');pids=y.index.to_numpy();pi={p:i for i,p in enumerate(pids)};yi={t:i for i,t in enumerate(y.columns)};yv=y.to_numpy(float)
        for phase in [f'outer{k}' for k in range(5)]+['final']:
            p=d/phase;rec=json.loads((p/'COMPLETE.json').read_text())
            for name,h in rec['files'].items():
                if sha(p/name)!=h:raise ValueError('PHASE_HASH_DRIFT')
            a=pd.read_parquet(p/'AUDIT.parquet');f=pd.read_parquet(p/'PREDICTIONS.parquet')
            if a.duplicated(['target_id','arm','variant']).any():raise ValueError('DUPLICATE_AUDIT')
            ok=a[a.status.eq('FITTED')]
            for tid,g in ok.groupby('target_id'):
                pol=target_policy(dict(target_id=tid,**meta.loc[tid].to_dict()))
                if not g.allowed_blocks.fillna('').eq(';'.join(pol['allowed_blocks'])).all():raise ValueError('TARGET_LAYER_INPUT_LEAKAGE')
                if not g.purity_covariate_used.eq(pol['use_purity']).all():raise ValueError('PURITY_SELF_LEAKAGE')
            maxres=0
            for cp in p.glob('*purity*.joblib'):
                ctx=joblib.load(cp);which='inner'+phase[5:] if cp.name.startswith('inner') else phase
                if not set(ctx['train_preprocessing'])<=set(spec[which]['train']):raise ValueError('PREPROCESSING_TEST_LEAKAGE')
                maxres=max(maxres,max(z['max_relative_residual'] for z in ctx['stats'].values()))
            if maxres>5e-9:raise ValueError('SOLVER_RESIDUAL')
            if len(f):
                if f.duplicated(['target_id','patient_id','arm','variant']).any():raise ValueError('DUPLICATE_PREDICTION')
                rows=f.patient_id.map(pi).to_numpy(int);cols=f.target_id.map(yi).to_numpy(int)
                if not set(rows)<=set(spec[phase]['test']) or set(rows)&set(spec[phase]['train']):raise ValueError('PATIENT_SPLIT_LEAKAGE')
                np.testing.assert_allclose(f.observed.to_numpy(),yv[rows,cols],rtol=0,atol=0)
                if not np.isfinite(f[['observed','predicted','covariate_only_baseline']].to_numpy()).all():raise ValueError('NONFINITE_PREDICTION')
                grouped=f.groupby(['target_id','patient_id'])
                if not grouped.size().eq(10).all():raise ValueError('UNPAIRED_ARMS_OR_ABLATIONS')
                for col in ['observed','train_mean_baseline','covariate_only_baseline']:
                    if (grouped[col].max()-grouped[col].min()).abs().max()>1e-9:raise ValueError('OUTCOME_OR_BASELINE_MISMATCH')
            results.append(dict(cancer_id=c,phase=phase,status='PASS',fitted_models=len(ok),predictions=len(f),max_graph_residual=maxres))
        print('AUDITED',c,flush=True)
    dump(Path(output),dict(target_host='149',status='PASS',checks=results,paired_comparators_per_patient_target=10,checks_description=['all phase checksums','outcomes exactly equal original prepared native Y','no held-out patient in fit or preprocessing','identical patients/outcomes/covariate baseline across three arms and ablations','target-assay exclusion and purity self-exclusion','finite predictions','graph-system residual tolerance']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.source,a.inputs,a.output)
