from pathlib import Path
import socket,json,time,sys,hashlib,traceback
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'supervision_summary_r1';C=R/'canonical_targets_r6';Y=R/'functional_relation_y_catalog_r5'
sys.path.insert(0,'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/repo')
from cla_engine.common import sha
M=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t');M=M.loc[M.has_scored_readout.astype(str).str.lower().eq('true')].copy();IDS=set(M.canonical_id)
B=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t');binding=dict(zip(B.source_target_id,B.canonical_id));red=pd.read_csv(C/'ID_REDIRECTS.tsv',sep='\t');redirect=dict(zip(red.previous_internal_id,red.canonical_id))
def canon(s):
 seen=set()
 while s in redirect:
  assert s not in seen;seen.add(s);s=redirect[s]
 assert s in IDS,s
 return s
def js(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
AUTH=Path('/dell_2/DSC/CancerLncAtlas/inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv');A=pd.read_csv(AUTH,sep='\t');CANCERS=sorted(A.cancer_id.unique());CONTEXTS=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']
P=pd.read_csv(R/'rppa_functional_evidence_demo_r1/PROTEIN_TARGETS.tsv',sep='\t');F=pd.read_csv(R/'rppa_functional_evidence_demo_r1/PHOSPHO_TARGETS.tsv',sep='\t')
rows=[];summaries=[];canrows=[];patientrows=[];sourcefiles=[];contexts=[];cancerrows=[]
def add_scores(matrix,cohort,cancer,scope,source_ids,cids,layers,method,reference,source_file):
 assert matrix.index.is_unique and matrix.columns.is_unique
 assert len(matrix.columns)==len(source_ids)==len(cids)==len(layers)
 matrix=matrix.reindex(scope);v=matrix.to_numpy(float);assert not np.isinf(v).any()
 observed=np.isfinite(v);counts=observed.sum(0);variable=np.array([np.unique(v[observed[:,j],j]).size>1 for j in range(v.shape[1])]);mapped=[canon(c) for c in cids]
 for j in range(v.shape[1]):
  rows.append(dict(cohort=cohort,cancer_id=cancer,canonical_id=mapped[j],source_readout_id=source_ids[j],measurement_layer=layers[j],method=method,reference=reference,patients_in_scope=len(scope),patients_with_score=int(counts[j]),coverage_fraction=float(counts[j]/len(scope)) if scope else 0.,variable=bool(variable[j])))
 for layer in sorted(set(layers)):
  jj=[j for j,l in enumerate(layers) if l==layer];df=pd.DataFrame(observed[:,jj],index=scope,columns=[mapped[j] for j in jj]);df=df.T.groupby(level=0).any().T
  for cid,n in df.sum().items():canrows.append(dict(cohort=cohort,cancer_id=cancer,canonical_id=cid,measurement_layer=layer,patients_in_scope=len(scope),patients_with_score=int(n),coverage_fraction=float(n/len(scope)),readout_group=method))
  for pat,n in df.sum(1).items():patientrows.append(dict(cohort=cohort,cancer_id=cancer,patient_id=pat,measurement_layer=layer,readout_group=method,observed_canonical_targets=int(n)))
  summaries.append(dict(cohort=cohort,cancer_id=cancer,measurement_layer=layer,readout_group=method,patients_in_scope=len(scope),patients_with_any_score=int(df.any(axis=1).sum()),source_readouts=len(jj),observed_readouts=int((counts[jj]>0).sum()),canonical_targets_observed=int(df.any(axis=0).sum()),canonical_patient_values=int(df.to_numpy().sum())))
 sourcefiles.append({'path':str(source_file),'sha256':sha(source_file)})
def tcga(cancer):
 scope=A.loc[A.cancer_id.eq(cancer),'patient_id'].tolist()
 for context in CONTEXTS:
  b=Y/cancer/context;d=json.loads((b/'COMPLETE.json').read_text());assert d['target_host']=='149';assert sha(b/'TARGET_COUNTS.tsv')==d['files']['TARGET_COUNTS.tsv'];t=pd.read_csv(b/'TARGET_COUNTS.tsv',sep='\t');assert set(t.canonical_id)==IDS
  cols=['observable_relations','strong_positive','weak_positive','unlabeled_observed','direction_discordant']
  for col in cols:assert int(t[col].sum())==d[col]
  split=json.loads((b/'SPLIT.json').read_text());assert not set(split['source_patients'])&set(split['label_patients'])
  row={k:d[k] for k in ['cancer_id','context','target_identities']+cols};row.update(patients_in_scope=len(scope),source_patients=len(split['source_patients']),label_patients=len(split['label_patients']),canonical_targets_estimable=int(t.observable_relations.gt(0).sum()),canonical_targets_supported=int((t.strong_positive+t.weak_positive).gt(0).sum()),canonical_targets_multilayer_strict=int(t.strong_positive.gt(0).sum()))
  for role in ['Pathway','State']:
   m=t.roles.str.split('|').map(lambda x:role in x);row[role.lower()+'_targets_estimable']=int((m&t.observable_relations.gt(0)).sum())
  contexts.append(row)
  if context=='sealed_test':cancerrows.append(row.copy())
 b=R/'common_rna_scores_r1/cache'/('TCGA_'+cancer);p=b/'PATHWAY_RAW.parquet';s=pd.read_parquet(p);sid=[x.split('|',1)[1] for x in s.columns];add_scores(s,'TCGA',cancer,scope,sid,[binding[x] for x in sid],['RNA_EXPRESSION']*len(sid),'Common-background ssGSEA raw ES','Fixed common gene background; sample-wise ranks',p)
 b=R/'state_relation_evidence_scope_adjusted_r2'/cancer/'sealed_test';p=b/'PATIENT_S.parquet';s=pd.read_parquet(p);md=pd.read_csv(b/'TARGETS.tsv',sep='\t').set_index('target_id').loc[s.columns]
 add_scores(s,'TCGA',cancer,scope,md.source_target_id.tolist(),md.canonical_id.tolist(),md.measurement_layer.tolist(),'Signature-specific State score','Training source-patient parameters; native units for published phenotypes',p)
 b=R/('rppa_functional_evidence_demo_r1' if cancer in ['BRCA','LUAD','COAD'] else 'rppa_functional_evidence_catalog_r1')/cancer/'sealed_test'
 for layer,md,method in [('PROTEIN',P,'Mean train-standardized antibody gene-group abundance'),('PHOSPHO',F,'Signed mean train-standardized protein-adjusted phosphosite residual')]:
  p=b/layer/'PATIENT_S.parquet';s=pd.read_parquet(p);assert list(s.columns)==md.canonical_id.tolist();sid=md.source_readout_id.tolist() if layer=='PHOSPHO' else md.canonical_id.tolist();add_scores(s,'TCPA_RPPA500',cancer,scope,sid,md.canonical_id.tolist(),['PROTEIN_ABUNDANCE' if layer=='PROTEIN' else 'PHOSPHORYLATION']*len(sid),method,'TCGA source-patient reference',p)
 print('TCGA_SUMMARY_COMPLETE',cancer,flush=True)
def cptac():
 B0=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/bundle_r1');clinical=pd.read_parquet(B0/'CLINICAL_ALIGNED.parquet');idx=pd.read_csv(B0/'SCORE_INDEX.tsv',sep='\t');state=pd.read_parquet(B0/'STATE_PATIENT.parquet').set_index('patient_id');reg=pd.read_parquet(C/'SCORE_REGISTRY.parquet')
 for cohort,cl in clinical.groupby('source_cohort'):
  cancer=cl.cancer_id.unique();assert len(cancer)==1;cancer=cancer[0];scope=cl.patient_id.tolist();p=R/'common_rna_scores_r1/cache'/('CPTAC_'+cohort)/'PATHWAY_RAW.parquet';s=pd.read_parquet(p);sid=[x.split('|',1)[1] for x in s.columns];add_scores(s,'CPTAC:'+cohort,cancer,scope,sid,[binding[x] for x in sid],['RNA_EXPRESSION']*len(sid),'Common-background ssGSEA raw ES','Fixed common gene background; sample-wise ranks',p)
  for rec in idx.loc[idx.source_cohort.eq(cohort)&~idx.layer.eq('RNA_PATHWAY')].itertuples():
   p=B0/rec.path;assert sha(p)==rec.sha256;s=pd.read_parquet(p).set_index('target_id').T;s.index.name='patient_id';sid=[x.split('|',1)[-1] for x in s.columns];cids=[binding[x] for x in sid];layer={'PROTEIN_PATHWAY':'PROTEIN_ABUNDANCE','PHOSPHO_PTMSEA':'PHOSPHORYLATION','METHYLATION_PATHWAY':'DNA_METHYLATION'}[rec.layer];add_scores(s,'CPTAC:'+cohort,cancer,scope,sid,cids,[layer]*len(sid),rec.method,'Source cohort assay and scoring reference',p)
  rr=reg.loc[reg.cohort.eq('CPTAC:'+cohort)&reg.target_class.eq('state')&reg.score_source_file.str.endswith('STATE_PATIENT.parquet')].drop_duplicates('canonical_source_column')
  if len(rr):
   cols=rr.canonical_source_column.tolist();assert all(x in state.columns for x in cols);add_scores(state.loc[:,cols],'CPTAC:'+cohort,cancer,scope,rr.source_target_id.tolist(),rr.canonical_id.tolist(),rr.measurement_layer.tolist(),'Source-specific State score','Published or native source-cohort scoring definition',B0/'STATE_PATIENT.parquet')
  print('CPTAC_SUMMARY_COMPLETE',cohort,flush=True)
if __name__=='__main__':
 O.mkdir(exist_ok=False);js(O/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'SCORE_COVERAGE_AND_RELATION_COUNTS','canonical_registry':str(C),'relation_y':str(Y),'cancer_context':'sealed_test; all five contexts exported separately'})
 pending=set(CANCERS);deadline=time.monotonic()+7200
 while pending:
  for c in sorted(pending):
   if all((Y/c/k/'COMPLETE.json').exists() for k in CONTEXTS):tcga(c);pending.remove(c)
  js(O/'PROGRESS.json',{'target_host':'149','completed_tcga':33-len(pending),'pending':sorted(pending)})
  if pending:
   if time.monotonic()>deadline:raise TimeoutError('Y materialization deadline')
   time.sleep(10)
 cptac()
 tables={'CURRENT_SCORE_COVERAGE.tsv.gz':pd.DataFrame(rows),'CANONICAL_SCORE_COVERAGE.tsv.gz':pd.DataFrame(canrows),'PATIENT_SCORE_COVERAGE.tsv.gz':pd.DataFrame(patientrows),'SCORE_LAYER_SUMMARY.tsv':pd.DataFrame(summaries),'Y_SUPERVISION_BY_CANCER_CONTEXT.tsv':pd.DataFrame(contexts),'Y_SUPERVISION_BY_CANCER.tsv':pd.DataFrame(cancerrows)}
 for name,df in tables.items():df.to_csv(O/name,sep='\t',index=False)
 assert len(cancerrows)==33 and len(contexts)==165
 js(O/'SCORE_SOURCES.json',sourcefiles);js(O/'COMPLETE.json',{'target_host':'149','status':'COMPLETE','rows':{k:len(v) for k,v in tables.items()},'canonical_registry_sha256':sha(C/'COMPLETE.json'),'files':{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='COMPLETE.json'}});print('SUMMARY_COMPLETE',{k:len(v) for k,v in tables.items()},flush=True)
