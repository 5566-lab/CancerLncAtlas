import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
REPO=ROOT/'source_integration/repo'
BASE=ROOT/'kegg_cancer_redefinition_20261111_r1/measured_y_definition_r1'
assert socket.gethostname().split('.')[0]=='149'
sys.path.insert(0,str(REPO))

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify(folder):
    r=json.loads((folder/'COMPLETE.json').read_text())
    assert r['target_host']=='149'
    for k,v in r['files'].items():
        if sha(folder/k)!=v:raise ValueError('artifact drift '+str(folder/k))
    return r

preflight={'target_host':'149','time_utc':datetime.now(timezone.utc).isoformat(),
    'operation':'replace_RNA_measurement_versions_keep_biological_axis',
    'model_training':False,'lncrna_association_computation':False,'script_sha256':sha(__file__)}
save(BASE/'CODING_R3_INTEGRATION_PREFLIGHT.json',preflight)
tests=['tests/test_coding_readout_replacement.py','tests/test_coding_readout_reference.py','tests/test_measured_readouts.py','tests/test_coding_lm22.py','tests/test_pathway_conditions.py']
r=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(r.stdout,flush=True)
save(BASE/'CODING_R3_INTEGRATION_TESTS.json',dict(preflight,tests=tests,exit_code=r.returncode,output=r.stdout))
if r.returncode:raise SystemExit(r.returncode)

import numpy as np
import pandas as pd
from cla_engine.coding_readout_replacement import replace_coding_readouts
from cla_engine.measured_readouts import measurement_mask,summarize_target_coverage

summary=[]
for cancer in ('BRCA','LUAD','COAD'):
    old=BASE/'measurements'/cancer/'validation'
    out=BASE/'measurements_coding_r3'/cancer/'validation'
    if out.exists():raise RuntimeError('do not overwrite measurement version '+str(out))
    original=verify(old)
    out.mkdir(parents=True)
    save(out/'LAUNCH_PREFLIGHT.json',dict(preflight,cancer=cancer))
    md=pd.read_csv(old/'READOUTS.tsv',sep='\t',keep_default_na=False)
    bindings=pd.read_csv(old/'READOUT_BINDINGS.tsv',sep='\t',keep_default_na=False)
    targets=pd.read_csv(old/'TARGETS.tsv',sep='\t',keep_default_na=False)
    refs=json.loads((old/'INPUT_REFERENCES.json').read_text())
    policy=pd.read_csv(refs['structural']['TARGET_POLICY.tsv']['path'],sep='\t',keep_default_na=False).set_index('canonical_id')
    all_meta=[];maps=[];masks={};partitions={};inputs={};coverage=[];unchanged_streams=[]
    for cohort in ('TCGA','CPTAC'):
        src=BASE/'protein_coding_readouts'/(cohort+'_'+cancer)/'validation'
        coding_receipt=verify(src)
        cv=pd.read_parquet(src/'VALUES.parquet')
        cm=pd.read_csv(src/'READOUTS.tsv',sep='\t',keep_default_na=False)
        lm=BASE/'lm22_coding_r1'/(cohort+'_'+cancer)
        lm_receipt=verify(lm)
        assert lm_receipt['patients_computed']==lm_receipt['patients']
        assert lm_receipt['cell_readouts']==22 and lm_receipt['coding_reference_genes']==517
        lv=pd.read_parquet(lm/'VALUES.parquet')
        lm_md=pd.read_csv(lm/'READOUTS.tsv',sep='\t',keep_default_na=False)
        if set(lv.index)!=set(cv.index):raise ValueError('LM22 patient identity mismatch')
        cv=pd.concat([cv,lv.reindex(cv.index)],axis=1)
        cm=pd.concat([cm,lm_md],ignore_index=True)
        partitions[cohort]=json.loads((src/'PATIENT_PARTITIONS.json').read_text())
        masks[cohort]={}
        inputs[cohort]={'path':str(src),'receipt_sha256':sha(src/'COMPLETE.json'),'lm22_path':str(lm),'lm22_receipt_sha256':sha(lm/'COMPLETE.json')}
        for layer,metadata in md[md.cohort.eq(cohort)].groupby('layer',sort=False):
            values=pd.read_parquet(old/cohort/layer/'VALUES.parquet')
            new,nmd,mapping=replace_coding_readouts(values,metadata,cv,cm)
            previous=pd.read_parquet(BASE/'measurements_coding_r2'/cancer/'validation'/cohort/layer/'VALUES.parquet')
            retained=[column for column in previous if column in new]
            pd.testing.assert_frame_equal(new[retained],previous[retained])
            mask=measurement_mask(new)
            folder=out/cohort/layer;folder.mkdir(parents=True)
            new.to_parquet(folder/'VALUES.parquet',compression='zstd')
            mask.to_parquet(folder/'OBSERVED.parquet',compression='zstd')
            pd.testing.assert_frame_equal(pd.read_parquet(folder/'VALUES.parquet'),new)
            pd.testing.assert_frame_equal(pd.read_parquet(folder/'OBSERVED.parquet'),mask)
            # The exact original non-RNA values and missingness must survive.
            nr=metadata.loc[~metadata.measurement_layer.str.startswith('RNA'),'target_id'].tolist()
            pd.testing.assert_frame_equal(new[nr],values[nr])
            if nr:unchanged_streams.append({'cohort':cohort,'layer':layer,'non_RNA_columns_equal':len(nr)})
            nmd['display_category']=nmd.canonical_id.map(policy.roles)
            nmd['analysis_role']=nmd.canonical_id.map(policy.analysis_role)
            if nmd[['display_category','analysis_role']].isna().any().any():raise ValueError('unmapped identity policy')
            mapping['cohort']=cohort;mapping['layer']=layer
            all_meta.append(nmd);maps.append(mapping);masks[cohort][layer]=mask
    nmd=pd.concat(all_meta,ignore_index=True)
    mapping=pd.concat(maps,ignore_index=True)
    lookup=mapping.set_index(['cohort','layer','original_target_id'])
    new_bind=[]
    for row in bindings.to_dict('records'):
        mapped=lookup.loc[(row['cohort'],row['layer'],row['target_id'])]
        row['original_target_id']=row['target_id']
        row['original_column_index']=row['column_index']
        row['target_id']=mapped.target_id
        row['column_index']=int(mapped.column_index)
        new_bind.append(row)
    bindings=pd.DataFrame(new_bind)
    for cohort in masks:
        coverage.extend(summarize_target_coverage(bindings.to_dict('records'),masks[cohort],targets.to_dict('records'),cohort,'validation'))
    for filename,frame in [('TARGETS.tsv',targets),('READOUTS.tsv',nmd),('READOUT_BINDINGS.tsv',bindings),
                           ('READOUT_VERSION_MAP.tsv',mapping),('TARGET_MEASUREMENT_COVERAGE.tsv',pd.DataFrame(coverage))]:
        frame.to_csv(out/filename,sep='\t',index=False)
    review=nmd[nmd.measurement_definition.eq('PRESERVED_EXTERNAL_RNA_SOURCE')]
    assert int(nmd.coding_expression_basis_verified.sum())==4168
    assert len(review)==4
    review.to_csv(out/'EXTERNAL_RNA_DEFINITIONS.tsv',sep='\t',index=False)
    save(out/'PATIENT_PARTITIONS.json',partitions)
    save(out/'INPUT_REFERENCES.json',{'measured_source_directory':str(old),'measured_source_receipt_sha256':sha(old/'COMPLETE.json'),
                                    'coding_readouts':inputs,'structural':refs['structural']})
    assert set(targets.canonical_id)==set(pd.read_csv(old/'TARGETS.tsv',sep='\t').canonical_id)
    assert targets.analysis_role.eq('primary_pathway').sum()==477
    assert len(nmd)==len(md) and len(bindings)==len(pd.read_csv(old/'READOUT_BINDINGS.tsv',sep='\t'))
    result={'format':'CLA_PATIENT_MEASURED_READOUTS_CODING_V3','target_host':'149','cancer':cancer,'context':'validation',
       'biological_targets':len(targets),'primary_pathways':477,'state_identities':int(targets.analysis_role.eq('state_auxiliary').sum()),
       'readout_columns':len(nmd),'coding_readouts_replaced':int(nmd.coding_expression_basis_verified.sum()),
       'preserved_external_RNA_readouts':len(review),'structural_bindings':len(bindings),
       'patient_counts':{c:len(next(iter(m.values()))) for c,m in masks.items()},
       'exact_non_RNA_preservation':unchanged_streams,'scalar_fusion_performed':False,'lncrna_association_computation':False,
       'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    save(out/'COMPLETE.json',result)
    summary.append({k:v for k,v in result.items() if k!='files'})
    print('INTEGRATED '+cancer+' '+str(result['coding_readouts_replaced'])+' corrected readouts; biological axis 477+197',flush=True)
save(BASE/'CODING_R3_MEASUREMENT_INTEGRATION.json',{'target_host':'149','cases':summary})
