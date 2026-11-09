from pathlib import Path
import json,hashlib,socket,time
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'supervision_summary_r2';Y=R/'functional_relation_y_catalog_r6';S=R/'supervision_summary_r1';V=R/'functional_y_validation_r2';deadline=time.monotonic()+8*3600
O.mkdir(exist_ok=False);(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'FINAL_NUMERICALLY_DEFINED_SUPERVISION_COUNTS','relation_y':str(Y),'unchanged_patient_score_coverage':str(S)},indent=2))
while not all((p/'COMPLETE.json').exists() for p in [Y,S,V]):
 if time.monotonic()>deadline:raise TimeoutError('Final summary source deadline')
 time.sleep(10)
for p in [Y,S,V]:assert json.loads((p/'COMPLETE.json').read_text())['status'] in ['PASS','COMPLETE'],str(p)
for name in ['CURRENT_SCORE_COVERAGE.tsv.gz','CANONICAL_SCORE_COVERAGE.tsv.gz','PATIENT_SCORE_COVERAGE.tsv.gz','SCORE_LAYER_SUMMARY.tsv','SCORE_SOURCES.json']:(O/name).symlink_to((S/name).resolve())
rows=[]
for p in sorted(Y.glob('*/*/COMPLETE.json')):
 d=json.loads(p.read_text());b=p.parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert sha(b/'TARGET_COUNTS.tsv')==d['files']['TARGET_COUNTS.tsv'];t=pd.read_csv(b/'TARGET_COUNTS.tsv',sep='\t');split=json.loads((b/'SPLIT.json').read_text());assert not set(split['source_patients'])&set(split['label_patients']);axes=json.loads((Y.parent/'numerical_relation_evidence_r1'/d['cancer_id']/'AXES.json').read_text());fields=['observable_relations','strong_positive','weak_positive','unlabeled_observed','direction_discordant']
 for k in fields:assert int(t[k].sum())==d[k]
 row={k:d[k] for k in ['cancer_id','context','target_identities']+fields};row.update(patients_in_scope=len(axes['patient_ids']),source_patients=len(split['source_patients']),label_patients=len(split['label_patients']),canonical_targets_estimable=int(t.observable_relations.gt(0).sum()),canonical_targets_supported=int((t.strong_positive+t.weak_positive).gt(0).sum()),canonical_targets_multilayer_strict=int(t.strong_positive.gt(0).sum()))
 for role in ['Pathway','State']:
  keep=t.roles.str.split('|').map(lambda x:role in x);row[role.lower()+'_targets_estimable']=int((keep&t.observable_relations.gt(0)).sum())
 rows.append(row)
a=pd.DataFrame(rows);assert len(a)==165 and a.cancer_id.nunique()==33;a.to_csv(O/'Y_SUPERVISION_BY_CANCER_CONTEXT.tsv',sep='\t',index=False);a.loc[a.context.eq('sealed_test')].to_csv(O/'Y_SUPERVISION_BY_CANCER.tsv',sep='\t',index=False)
q={'target_host':'149','status':'COMPLETE','cancers':33,'contexts':165,'canonical_registry_sha256':hashlib.sha256((R/'canonical_targets_r6/COMPLETE.json').read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()}};(O/'COMPLETE.json').write_text(json.dumps(q,indent=2));print('FINAL_SUPERVISION_SUMMARY_COMPLETE',flush=True)
