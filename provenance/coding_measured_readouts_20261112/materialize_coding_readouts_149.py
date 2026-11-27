import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import concurrent.futures
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
REPO=ROOT/'source_integration/repo'
BASE=WORK/'measured_y_definition_r1'
assert socket.gethostname().split('.')[0]=='149'
sys.path.insert(0,str(REPO))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

launch={'target_host':'149','time_utc':datetime.now(timezone.utc).isoformat(),
        'operation':'coding_only_patient_readouts_with_frozen_reference',
        'model_training':False,'lncrna_association_computation':False,
        'sealed_outer_test_opened':False,'script_sha256':sha(__file__)}
save(BASE/'CODING_READOUT_REFERENCE_PREFLIGHT.json',launch)

import numpy as np
import pandas as pd
from cla_engine.coding_readout_reference import build_coding_readouts,METHOD_VARIANT

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray,pd.Index)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def verify(folder,rec):
    for name,digest in rec['files'].items():
        if sha(folder/name)!=digest:raise RuntimeError('artifact drift: '+str(folder/name))

TESTS=['tests/test_coding_readout_reference.py','tests/test_protein_coding_scoring.py']
r=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
save(BASE/'CODING_REFERENCE_TESTS.json',dict(target_host='149',exit_code=r.returncode,tests=TESTS,output=r.stdout))
print(r.stdout,flush=True)
if r.returncode:raise SystemExit(r.returncode)

RSCRIPT=REPO/'scripts/materialize_m0_joint_apm.R'
RLIBS=':'.join(['/dsk2/user/dengsc/cla_cptac_integration_20261104_r1/R_library','/dsk2/user/dengsc/cla_state_expansion_20261101_r1/R_library','/public8/DSC/CancerLncAtlas/r_libs','/dsk2/user/dengsc/R/x86_64-pc-linux-gnu-library/4.5'])
MODEL=ROOT/'progeny_standard_r1/OFFICIAL_TOP100_MODEL.tsv'

def case(name):
    aligned=BASE/'protein_coding_inputs'/name
    cache=BASE/'protein_coding_cache'/name
    apmroot=BASE/'protein_coding_apm'/name
    output=BASE/'protein_coding_readouts'/name/'validation'
    src=json.loads((aligned/'COMPLETE.json').read_text())
    verify(aligned,src)
    cached=json.loads((cache/'COMPLETE.json').read_text())
    verify(cache,cached)
    if cached['input_complete_sha256']!=sha(aligned/'COMPLETE.json'):raise RuntimeError('input drift')
    split=json.loads((aligned/'PATIENT_PARTITIONS.json').read_text())
    fit=split['source_patients'];query=split['label_patients']
    assert split['outer_fold']==0 and split['source_folds']==[2,3,4] and split['label_folds']==[1]
    assert not set(fit)&set(query)
    if (output/'COMPLETE.json').exists():
        done=json.loads((output/'COMPLETE.json').read_text());verify(output,done)
        assert done['input_complete_sha256']==sha(aligned/'COMPLETE.json')
        assert done['coding_reference_code_sha256']==sha(REPO/'cla_engine/coding_readout_reference.py')
        return done
    output.mkdir(parents=True,exist_ok=True)
    save(output/'LAUNCH_PREFLIGHT.json',dict(launch,cohort_case=name,reference_patients=len(fit),query_patients=len(query)))
    apmroot.mkdir(parents=True,exist_ok=True)
    if not (apmroot/'APM_RUN_COMPLETE.json').exists():
        save(apmroot/'LAUNCH_PREFLIGHT.json',dict(launch,cohort_case=name,operation='APM_GSVA_coding_background_reference'))
        e=pd.read_parquet(aligned/'SYMBOL_LOGTPM.parquet')
        assert set(e.index)==set(fit+query) and e.index.is_unique
        positions=e.index.get_indexer(fit)
        assert (positions>=0).all()
        e.rename_axis('patient_id').reset_index().to_csv(apmroot/'APM_EXPRESSION.tsv.gz',sep='\t',index=False,na_rep='NA',compression='gzip')
        defs=json.loads((cache/'DEFINITIONS.json').read_text())
        save(apmroot/'APM_GENES.json',[r['SYMBOL'] for r in defs['APM_Wang']])
        save(apmroot/'SPLITS.json',{'validation':{'train':positions.tolist(),'label':e.index.get_indexer(query).tolist()}})
        save(apmroot/'APM_INPUT.json',{'target_host':'149','expression_sha256':sha(apmroot/'APM_EXPRESSION.tsv.gz'),
             'input_complete_sha256':sha(aligned/'COMPLETE.json'),'script_sha256':sha(RSCRIPT),'method_variant':METHOD_VARIANT})
        print('APM_START '+name,flush=True)
        with (apmroot/'APM_RUN.log').open('w') as log:
            ret=subprocess.run(['/usr/bin/Rscript',str(RSCRIPT),str(apmroot),RLIBS,'1'],stdout=log,stderr=subprocess.STDOUT)
        if ret.returncode:raise RuntimeError(name+' APM failed: '+(apmroot/'APM_RUN.log').read_text()[-5000:])
        files={str(p.relative_to(apmroot)):sha(p) for p in sorted(apmroot.rglob('*')) if p.is_file()}
        save(apmroot/'APM_RUN_COMPLETE.json',{'target_host':'149','aligned_receipt_sha256':sha(aligned/'COMPLETE.json'),'files':files})
        print('APM_COMPLETE '+name,flush=True)
    print('READOUT_REFERENCE_START '+name,flush=True)
    scores,methods,reference=build_coding_readouts(aligned,cache,apmroot,MODEL,fit,context='validation')
    assert set(scores.index)==set(fit+query) and scores.index.is_unique and scores.columns.is_unique
    assert reference['train_patient_ids']==fit
    methods['source_score_id']=methods.target_id
    methods['readout_id']=methods.target_id+'@@'+METHOD_VARIANT
    scores.columns=methods.set_index('source_score_id').loc[scores.columns,'readout_id'].tolist()
    methods['patients_in_scope']=len(scores)
    methods['patients_measured']=[int(scores[k].notna().sum()) for k in methods.readout_id]
    methods['source_published_method_equivalence']=False
    scores.to_parquet(output/'VALUES.parquet',compression='zstd')
    scores.notna().to_parquet(output/'OBSERVED.parquet',compression='zstd')
    methods.drop(columns=['target_id']).to_csv(output/'READOUTS.tsv',sep='\t',index=False)
    save(output/'REFERENCE.json',clean(reference))
    save(output/'PATIENT_PARTITIONS.json',{'outer_fold':0,'reference_folds':[2,3,4],'query_folds':[1],
         'reference_patients':fit,'query_patients':query,'target_host':'149'})
    ann=pd.read_csv(aligned/'SCORING_GENE_ANNOTATION.tsv',sep='\t')
    assert ann.gene_biotype.eq('protein_coding').all()
    members=pd.read_csv(cache/'PROGRAM_NETWORK.tsv',sep='\t')
    members['retained_in_coding_background']=members.gene.isin(set(ann.reference_symbol.dropna()))
    members.to_csv(output/'PROGRAM_MEMBER_AUDIT.tsv',sep='\t',index=False)
    check=pd.read_parquet(output/'VALUES.parquet')
    mask=pd.read_parquet(output/'OBSERVED.parquet')
    pd.testing.assert_frame_equal(check,scores)
    pd.testing.assert_frame_equal(mask,scores.notna())
    done={'target_host':'149','cohort_case':name,'context':'validation','method_variant':METHOD_VARIANT,
       'patients':len(scores),'reference_patients':len(fit),'query_patients':len(query),
       'readouts':len(scores.columns),'readouts_with_measurement':int(scores.notna().any().sum()),
       'input_complete_sha256':sha(aligned/'COMPLETE.json'),'cache_complete_sha256':sha(cache/'COMPLETE.json'),
       'coding_reference_code_sha256':sha(REPO/'cla_engine/coding_readout_reference.py'),
       'builder_code_sha256':sha(REPO/'cc_hhgt/m0_joint_scores.py'),
       'model_training':False,'lncrna_association_computation':False,
       'files':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='COMPLETE.json'}}
    save(output/'COMPLETE.json',done)
    print('READOUT_REFERENCE_COMPLETE '+name+' '+str(done['readouts']),flush=True)
    return done

cases=[cohort+'_'+cancer for cancer in ['BRCA','LUAD','COAD'] for cohort in ['TCGA','CPTAC']]
done={};errors={}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    jobs={pool.submit(case,name):name for name in cases}
    for future in concurrent.futures.as_completed(jobs):
        name=jobs[future]
        try:done[name]=future.result()
        except Exception as exc:
            errors[name]=str(exc)
            print('CASE_ERROR '+name+' '+str(exc),flush=True)
        save(BASE/'CODING_READOUT_REFERENCE_PROGRESS.json',{'target_host':'149','completed':done,'errors':errors})
print(json.dumps({'completed':list(done),'errors':errors}),flush=True)
if errors:raise SystemExit(1)
