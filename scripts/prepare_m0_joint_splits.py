"""Patient-only split definitions and frozen-APM launch on server 149."""
import argparse,json,os,socket,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from run_m0_linear import write_json

def split_definitions(patients):
    folds=patients.outer_fold.to_numpy(int);cohort=patients.cohort.to_numpy(str);n=len(patients);spec={}
    for k in range(5):
        tr=np.flatnonzero((folds!=k)&(folds!=(k+1)%5));va=np.flatnonzero(folds==(k+1)%5);te=np.flatnonzero(folds==k);ft=np.flatnonzero(folds!=k)
        spec[f'inner{k}']=dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist(),role='INNER_TUNING')
        spec[f'outer{k}']=dict(train=ft.tolist(),test=te.tolist(),role='COHORT_STRATIFIED_OUTER_TEST')
    for c in sorted(set(cohort)):
        spec[f'loco_{c}_inner']=dict(train=np.flatnonzero((cohort==c)&(folds!=0)).tolist(),validation=np.flatnonzero((cohort==c)&(folds==0)).tolist(),test=np.flatnonzero(cohort!=c).tolist(),role='LEAVE_COHORT_OUT_TUNING')
        spec[f'loco_{c}_outer']=dict(train=np.flatnonzero(cohort==c).tolist(),test=np.flatnonzero(cohort!=c).tolist(),role='LEAVE_COHORT_OUT_TEST')
    spec['final']=dict(train=list(range(n)),test=[],role='FINAL_AFTER_VALIDATION')
    return spec

def main(a):
    if socket.gethostname()!='149':raise RuntimeError('Run on 149')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/cptac_149_sources_20261104.json').read_text());defs=json.loads(Path(a.definitions).read_text())
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    processes=[]
    for c in ['BRCA','LUAD','COAD']:
        d=out/c
        if d.exists() and any(d.iterdir()):raise ValueError('Use a new output')
        d.mkdir();write_json(d/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='JOINT_M0_SPLITS_AND_APM',cancer=c))
        src=Path(a.aligned)/c;pt=pd.read_csv(src/'PATIENTS.tsv',sep='\t').rename(columns={'index':'patient_id'});sp=split_definitions(pt)
        write_json(d/'SPLITS.json',sp);write_json(d/'APM_GENES.json',[x['SYMBOL'] for x in defs['APM_Wang']])
        e=pd.read_parquet(src/'SYMBOL_LOGTPM.parquet').reindex(pt.patient_id);e.index.name='patient_id'
        e.reset_index().to_csv(d/'APM_EXPRESSION.tsv.gz',sep='\t',index=False,float_format='%.17g')
        write_json(d/'APM_INPUT.json',dict(target_host='149',expression_sha256=sha(d/'APM_EXPRESSION.tsv.gz'),input_complete_sha256=sha(src/'COMPLETE.json'),script_sha256=sha(ROOT/'scripts/materialize_m0_joint_apm.R')))
        cmd=['timeout','12h','/usr/bin/Rscript',str(ROOT/'scripts/materialize_m0_joint_apm.R'),str(d),':'.join(cfg['R_libraries']),'4']
        with open(d/'apm.log','w') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(dict(cancer=c,pid=p.pid,command=cmd));print('APM_STARTED',c,p.pid,flush=True)
        write_json(out/'PROCESSES.json',dict(target_host='149',processes=processes))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['aligned','output','definitions']:p.add_argument('--'+n,required=True)
 main(p.parse_args())
