from pathlib import Path
import socket,json,sys,time,hashlib
import pandas as pd
import pyarrow.parquet as pq
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;receipt=R/'project_log/PROJECT_DATA_RELEASE.json';deadline=time.monotonic()+10800
while not receipt.exists():
 if time.monotonic()>deadline:raise TimeoutError('Project release receipt')
 time.sleep(10)
release=json.loads(receipt.read_text());D=Path(release['directory']).resolve();rec=json.loads((D/'COMPLETE.json').read_text());assert rec['status']=='COMPLETE' and rec['target_host']=='149'
assert hashlib.sha256((D/'COMPLETE.json').read_bytes()).hexdigest()==release['release_receipt_sha256']
for rel,record in rec['files'].items():
 p=D/rel;assert p.is_file() and p.resolve().is_relative_to(D) and p.stat().st_size==record['bytes'],rel
index=pd.read_csv(D/'PATIENT_SCORE_INDEX.tsv',sep='\t').fillna('');assert len(index)==710 and not index.duplicated(['cohort','context','layer']).any();assert index.cohort.nunique()==43
assert len(index.loc[index.cohort.str.startswith('TCGA:')])==660 and len(index.loc[index.cohort.str.startswith('CPTAC:')])==50
master=pd.read_csv(D/'catalog/TARGET_MASTER.tsv',sep='\t');ids=set(master.canonical_id);assert len(master)==len(ids)==3456
matrices={};axes={}
for row in index.itertuples():
 for name,digest in [(row.matrix_file,row.matrix_sha256),(row.readout_axis_file,row.readout_axis_sha256)]:assert rec['files'][name]['sha256']==digest
 if row.readout_axis_file not in axes:
  axis=pd.read_csv(D/row.readout_axis_file,sep='\t');assert axis.score_column.is_unique and set(axis.canonical_id)<=ids;axes[row.readout_axis_file]=axis
 if row.matrix_file not in matrices:matrices[row.matrix_file]=pq.read_schema(D/row.matrix_file).names
 cols=matrices[row.matrix_file];axis=axes[row.readout_axis_file]
 if row.orientation=='patient_by_readout':assert set(axis.score_column)<=set(cols)
 else:assert row.orientation=='readout_by_patient' and 'target_id' in cols
 if row.patient_filter:assert row.patient_filter.startswith('source_cohort=') and 'source_cohort' in cols
sys.path.insert(0,str(R/'source_integration/repo'));from cla_engine.functional_y import load_functional_context
loaded=load_functional_context(D/'Y','BRCA','sealed_test');counts=loaded.counts();assert counts['canonical_targets']==3456 and counts['lncrnas']==16889
proof={'target_host':'149','status':'PASS','release_directory':release['directory'],'physical_directory':str(D),'release_receipt_sha256':release['release_receipt_sha256'],'files_checked':len(rec['files']),'score_index_entries':len(index),'cohorts':43,'distinct_score_matrices':len(matrices),'distinct_readout_axes':len(axes),'actual_release_context':counts,'checks':['All manifest files exist with recorded size within release directory','All score-index SHA256 references bind to release manifest','All readout identities resolve and patient matrices contain the declared axes','Public reader loads BRCA sealed_test from the published Y directory']}
(R/'project_log/PROJECT_DATA_RELEASE_VALIDATION.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
