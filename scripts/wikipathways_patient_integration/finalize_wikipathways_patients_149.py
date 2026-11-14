from pathlib import Path
import socket,sys,json,hashlib,subprocess
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
W=R/'kegg_cancer_redefinition_20261111_r1';REPO=R/'source_integration/repo'
O=W/'wiki_patient_integration_20261114_r1/patient_sources'
OLD=W/'fixed_y_remake_20261114_r1/Y_bundle'
sys.path.insert(0,str(REPO))
import pandas as pd
from cla_engine.measured_y_registry import verify_matrix
from cla_engine.wikipathways_observations import existing_measurement_ids,verify_reused_observations
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def wr(df,p):df.to_csv(p,sep='\t',index=False)
def sj(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def hashes(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f!=p/'COMPLETE.json'}
sj(O/'FINALIZATION_PREFLIGHT.json',dict(target_host='149',actual_host=socket.gethostname(),operation='Reuse pinned molecular observation IDs and derive actual finite-score coverage',training=False,code_sha256=sha(Path(__file__))))
root=json.loads((O/'COMPLETE.json').read_text())
assert root['target_host']=='149' and root['patients']==1910
for n,h in root['files'].items():assert sha(O/n)==h,n
test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_wikipathways_observations.py','tests/test_measured_members.py'],cwd=REPO,capture_output=True,text=True)
assert test.returncode==0,test.stdout+test.stderr
sj(O/'TESTS.json',dict(returncode=test.returncode,output=test.stdout+test.stderr))
summaries=[];audit=[]
for case in root['cohorts']:
    name=case['case'];folder=O/'cohorts'/name;previous=OLD/'cohorts'/name
    rec=json.loads((folder/'COMPLETE.json').read_text());oldrec=json.loads((previous/'COMPLETE.json').read_text())
    assert sha(previous/'OBSERVATIONS.tsv.gz')==oldrec['files']['OBSERVATIONS.tsv.gz']
    meta=rd(folder/'OBSERVATIONS.tsv.gz');oldmeta=rd(previous/'OBSERVATIONS.tsv.gz')
    remap=existing_measurement_ids(meta,oldmeta)
    values=pd.read_parquet(folder/'Y_VALUES.parquet');mask=pd.read_parquet(folder/'Y_OBSERVED.parquet')
    values.columns=[remap[x] for x in values.columns];mask.columns=values.columns
    meta['observation_id']=meta.observation_id.map(remap)
    links=rd(folder/'SOURCE_OBSERVATIONS.tsv.gz');links['observation_id']=links.observation_id.map(remap)
    try:contexts=rd(folder/'SOURCE_CONTEXT_OBSERVATIONS.tsv.gz')
    except pd.errors.EmptyDataError:contexts=pd.DataFrame()
    if 'observation_id' in contexts:contexts['observation_id']=contexts.observation_id.map(remap)
    shared=sorted(set(meta.observation_id)&set(oldmeta.observation_id))
    assert shared,'No reused molecular observations in '+name
    assert sha(previous/'Y_VALUES.parquet')==oldrec['files']['Y_VALUES.parquet']
    oldvalues=pd.read_parquet(previous/'Y_VALUES.parquet',columns=shared).reindex(values.index)
    n=verify_reused_observations(values,meta,oldvalues,oldmeta)
    verify_matrix(values,mask,list(values.index))
    assert meta.observation_id.is_unique and set(links.observation_id)<=set(meta.observation_id)
    values.to_parquet(folder/'Y_VALUES.parquet',compression='zstd');mask.to_parquet(folder/'Y_OBSERVED.parquet',compression='zstd')
    pd.testing.assert_frame_equal(pd.read_parquet(folder/'Y_VALUES.parquet'),values)
    wr(meta,folder/'OBSERVATIONS.tsv.gz');wr(links,folder/'SOURCE_OBSERVATIONS.tsv.gz');wr(contexts,folder/'SOURCE_CONTEXT_OBSERVATIONS.tsv.gz')
    rec['summary']['reused_existing_observation_ids']=n;rec['summary']['new_observation_ids']=len(meta)-n
    rec['files']=hashes(folder);sj(folder/'COMPLETE.json',rec);summaries.append(rec['summary'])
    audit.append(dict(case=name,reused_ids=n,remapped_ids=sum(k!=v for k,v in remap.items()),exact_raw_values_verified=True,old_receipt_sha256=sha(previous/'COMPLETE.json')))
    print('PINNED_IDS_VERIFIED',name,n,flush=True)
coverage=rd(O/'SOURCE_LAYER_COVERAGE.tsv');targets=rd(O/'SOURCE_TARGETS.tsv')
targets=targets.drop(columns=['has_computed_patient_scores','score_layers'],errors='ignore')
targets['score_dimensions_registered']=2;targets['registered_score_layers']='RNA|PROTEIN'
valid=coverage[coverage.patients_with_score.gt(0)].groupby('source_target_id').layer.agg(lambda x:'|'.join(sorted(set(x))))
targets['measured_score_layers']=targets.source_target_id.map(valid).fillna('');targets['has_finite_patient_score']=targets.measured_score_layers.ne('')
wr(targets,O/'SOURCE_TARGETS.tsv');wr(pd.DataFrame(summaries),O/'COHORT_SUMMARY.tsv')
score_coverage=pd.concat([rd(O/'cohorts'/r['case']/'SCORE_COVERAGE.tsv') for r in summaries],ignore_index=True)
failed=score_coverage[score_coverage.observed_score_patients.eq(0)].copy()
wr(failed,O/'SOURCE_SCORE_FAILURES.tsv')
reason_counts=failed.groupby(['case','layer','reason'],dropna=False).size().rename('source_count').reset_index()
wr(reason_counts,O/'SOURCE_SCORE_FAILURE_REASON_COUNTS.tsv')
print('SOURCE_SCORE_FAILURE_REASONS',reason_counts.to_json(orient='records'),flush=True)
sj(O/'PINNED_OBSERVATION_ID_AUDIT.json',dict(target_host='149',cases=audit))
sj(O/'PROGRESS.json',dict(target_host='149',completed={x['case']:x for x in summaries}))
root['cohorts']=summaries;root['sources_with_finite_scores']=int(targets.has_finite_patient_score.sum());root['files']=hashes(O)
sj(O/'COMPLETE.json',root)
print('FINAL_WP_COLLECTION',json.dumps(dict(sources=193,patients=1910,sources_with_finite_scores=int(targets.has_finite_patient_score.sum()),receipt_sha256=sha(O/'COMPLETE.json'))),flush=True)
