"""Publish validated Y, source scores and fixed identities into the project data tree."""
from pathlib import Path
import json,socket,time,hashlib,shutil,subprocess,datetime
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;ROOT=Path('/dell_2/DSC/CancerLncAtlas');DEST=ROOT/'processed/functional_y_20261109_r1';STAGE=DEST.with_name(DEST.name+'.staging');Y=R/'functional_relation_y_catalog_r6';N=R/'numerical_relation_evidence_r1';C=R/'canonical_targets_r6';Q=R/'supervision_summary_r2';D=R/'reader_dictionary_r4';deadline=time.monotonic()+8*3600
required=[Y/'COMPLETE.json',Q/'COMPLETE.json',R/'functional_y_validation_r2/COMPLETE.json',R/'project_log/DELIVERY_CONTENT_AUDIT.json']
while not all(p.exists() for p in required):
 if time.monotonic()>deadline:raise TimeoutError('Final data release input deadline')
 time.sleep(10)
for p in required:assert json.loads(p.read_text())['status'] in ['COMPLETE','PASS'],str(p)
assert not DEST.exists() and not STAGE.exists();assert shutil.disk_usage(ROOT).free>15*2**30
STAGE.mkdir();(STAGE/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'PUBLISH_VALIDATED_FUNCTIONAL_Y_DATA','destination':str(DEST),'source_workdir':str(R)},indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def copy(source,relative,expected=None):
 source=Path(source);dest=STAGE/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 source_sha=sha(source) if expected is None else expected
 shutil.copyfile(source,dest);assert sha(dest)==source_sha,(str(source),str(dest));return source_sha
objects={};reference_rows=[]
def obj(source,category,expected=None):
 source=Path(source).resolve();key=(str(source),category)
 if key in objects:return objects[key]
 digest=sha(source) if expected is None else expected;suffix=''.join(source.suffixes);rel=Path('scores')/category/(digest+suffix)
 if not (STAGE/rel).exists():copy(source,rel,digest)
 value={'path':rel.as_posix(),'sha256':digest};objects[key]=value;return value
redirect_table=pd.read_csv(C/'ID_REDIRECTS.tsv',sep='\t');oldcol=next(k for k in ['previous_internal_id','old_canonical_id','from_canonical_id','previous_canonical_id'] if k in redirect_table);newcol=next(k for k in ['canonical_id','new_canonical_id','to_canonical_id'] if k in redirect_table);redirect=dict(zip(redirect_table[oldcol],redirect_table[newcol]));bind=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t').set_index('source_target_id').canonical_id.to_dict();master=pd.read_csv(D/'TARGET_MASTER.tsv',sep='\t');ids=set(master.canonical_id)
def canonical(cid):
 seen=set()
 while cid in redirect:
  assert cid not in seen;seen.add(cid);cid=redirect[cid]
 assert cid in ids,cid;return cid
def axis(frame):
 assert frame.score_column.is_unique and frame.source_definition_id.notna().all()
 frame=frame.copy();frame['canonical_id']=frame.canonical_id.map(canonical)
 data=frame.to_csv(sep='\t',index=False).encode();digest=hashlib.sha256(data).hexdigest();rel=Path('scores/axes')/(digest+'.tsv');dest=STAGE/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.write_bytes(data)
 assert sha(dest)==digest;return {'path':rel.as_posix(),'sha256':digest}
def ref(source,cohort,context,layer):
 if not source.exists():return
 item=obj(source,'references');reference_rows.append(dict(cohort=cohort,context=context,layer=layer,source_file=str(source),reference_file=item['path'],sha256=item['sha256']))
# Y is compact and contains its own context receipts, axes and split authority.
shutil.copytree(Y,STAGE/'Y')
for source in Y.rglob('*'):
 if source.is_file():assert sha(source)==sha(STAGE/'Y'/source.relative_to(Y))
score_rows=[]
for context in sorted(N.glob('*/*/COMPLETE.json')):
 cancer=context.parent.parent.name;kind=context.parent.name
 for layer in ['RNA','STATE','PROTEIN','PHOSPHO']:
  base=context.parent/layer;receipt=json.loads((base/'COMPLETE.json').read_text());source=(base/'PATIENT_S.parquet').resolve();matrix=obj(source,'matrices',receipt['files']['PATIENT_S.parquet']);md=pd.read_csv(base/'TARGETS.tsv',sep='\t');a=axis(pd.DataFrame({'score_column':md.target_id,'source_definition_id':md.source_target_id,'canonical_id':md.canonical_id,'measurement_layer':md.measurement_layer}));score_rows.append(dict(cohort='TCGA:'+cancer,cancer_id=cancer,context=kind,layer=layer,orientation='patient_by_readout',patient_filter='',matrix_file=matrix['path'],matrix_sha256=matrix['sha256'],readout_axis_file=a['path'],readout_axis_sha256=a['sha256'],source_file=str(source)))
  ref(base/'ADJUSTMENT.json','TCGA:'+cancer,kind,layer)
  if layer=='STATE':ref(source.parent/'SCORE_REFERENCE.json','TCGA:'+cancer,kind,layer)
  elif layer in ['PROTEIN','PHOSPHO']:ref(source.parent.parent/'SCORING_REFERENCE.json','TCGA:'+cancer,kind,layer)
  else:ref(source.parent/'DEFINITIONS.json','TCGA:'+cancer,kind,layer)
 print('RELEASE_TCGA_CONTEXT',cancer,kind,flush=True)
assert len(score_rows)==660
# Current CPTAC RNA uses the same common background; other native readouts keep their own axes.
P=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/bundle_r1');cohorts=pd.read_csv(R/'cptac_method_index_r1/CPTAC_COVARIATE_COVERAGE.tsv',sep='\t');native_sources={str(Path(x['path']).resolve()):x['sha256'] for x in json.loads((Q/'SCORE_SOURCES.json').read_text())}
rmd=pd.read_csv(N/'BRCA/sealed_test/RNA/TARGETS.tsv',sep='\t');rna_axis=axis(pd.DataFrame({'score_column':rmd.target_id,'source_definition_id':rmd.source_target_id,'canonical_id':rmd.canonical_id,'measurement_layer':rmd.measurement_layer}))
state_reg=pd.read_csv(R/'catalog_inventory/STATE_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t');state_axis=axis(pd.DataFrame({'score_column':state_reg.source_column,'source_definition_id':state_reg.source_target_id,'canonical_id':state_reg.source_target_id.map(bind),'measurement_layer':[json.loads(v).get('measurement','') for v in state_reg.original_definition]}));ptm=pd.read_csv(C/'PTM_READOUT_BINDINGS.tsv',sep='\t').set_index('source_readout_id').canonical_id.to_dict()
for row in cohorts.itertuples():
 cohort=row.cohort.split(':',1)[1]
 streams=[('RNA',R/'common_rna_scores_r1/cache'/('CPTAC_'+cohort)/'PATHWAY_RAW.parquet','patient_by_readout'),('PROTEIN',P/'scores/PROTEIN_PATHWAY'/(cohort+'.parquet'),'readout_by_patient'),('PHOSPHO',P/'scores/PHOSPHO_PTMSEA'/(cohort+'.parquet'),'readout_by_patient'),('METHYLATION',P/'scores/METHYLATION_PATHWAY'/(cohort+'.parquet'),'readout_by_patient'),('STATE',P/'STATE_PATIENT.parquet','patient_by_readout')]
 for layer,path,orientation in streams:
  source=path.resolve();assert str(source) in native_sources;matrix=obj(source,'matrices',native_sources[str(source)])
  if layer=='RNA':a=rna_axis
  elif layer=='STATE':a=state_axis
  else:
   names=pd.read_parquet(path,columns=['target_id']).target_id;mapping=ptm if layer=='PHOSPHO' else bind;a=axis(pd.DataFrame({'score_column':names,'source_definition_id':names,'canonical_id':names.map(mapping),'measurement_layer':{'PROTEIN':'PROTEIN_ABUNDANCE','PHOSPHO':'PHOSPHORYLATION','METHYLATION':'DNA_METHYLATION'}[layer]}))
  score_rows.append(dict(cohort=row.cohort,cancer_id=row.cancer_id,context='native_readout',layer=layer,orientation=orientation,patient_filter=('source_cohort='+cohort if layer=='STATE' else ''),matrix_file=matrix['path'],matrix_sha256=matrix['sha256'],readout_axis_file=a['path'],readout_axis_sha256=a['sha256'],source_file=str(source)))
 print('RELEASE_CPTAC_COHORT',cohort,flush=True)
assert len(score_rows)==710
known_cohorts={x.split(':',1)[1] for x in cohorts.cohort}
for record in json.loads((P/'BOUND_SOURCES.json').read_text()):
 if record['role'] in ['RNA_METHOD_RECEIPT','PROTEIN_METHOD','PTM_METHOD']:
  source=Path(record['path']);assert sha(source)==record['sha256'];parts=[x for x in source.parts if x in known_cohorts];native_cohort=('CPTAC:'+parts[-1]) if parts else ('CPTAC:'+source.stem if source.stem in known_cohorts else 'CPTAC')
  ref(source,native_cohort,'native_readout',record['role'])
for source in [P/'METHOD_POLICY.json',P.parent/'native_states_r1/COMPLETE.json',P.parent/'extra_states_r1/COMPLETE.json']:ref(source,'CPTAC','native_readout','STATE')
pd.DataFrame(score_rows).to_csv(STAGE/'PATIENT_SCORE_INDEX.tsv',sep='\t',index=False);pd.DataFrame(reference_rows).drop_duplicates().to_csv(STAGE/'SCORING_REFERENCE_INDEX.tsv',sep='\t',index=False)
# The release carries the reader's current table definitions and the source composition.
for name in ['TARGET_MASTER.tsv','STRUCTURAL_CONTEXTS.tsv','STATE_SCORE_DEFINITIONS.tsv','OBSERVED_SCORE_DEFINITIONS.tsv','SCORE_COVERAGE.tsv.gz']:copy(D/name,Path('catalog')/name)
for name in ['SOURCE_ID_BINDINGS.tsv','SOURCE_ALIASES.tsv','ID_REDIRECTS.tsv','FUNCTIONAL_SCOPE_RELATIONS.tsv','STATE_SCOPE_EQUIVALENCES.tsv','STATE_COMPONENT_RELATIONS.tsv','PTM_READOUT_BINDINGS.tsv','PHOSPHO_READOUT_DEFINITIONS.tsv','PHOSPHO_READOUT_COMPOSITION.tsv.gz','KINASE_MOLECULAR_COMPONENTS.tsv','KINASE_CONSTRUCT_RELATIONS.tsv','SOURCE_KINASE_PROGRAMS.tsv','PATHWAY_GENE_COMPOSITION.tsv.gz','PATHWAY_REACTION_COMPOSITION.tsv.gz','STATE_MOLECULAR_COMPOSITION.tsv.gz']:copy(C/name,Path('catalog')/name)
for directory in ['target_composition_index_r1','cptac_method_index_r1','reaction_scope_relations_r1']:
 for path in (R/directory).iterdir():
  if path.is_file() and ('.tsv' in path.name):copy(path,Path('mapping')/path.name)
for path in Q.iterdir():
 if path.is_file() and '.tsv' in path.name:copy(path,Path('summary')/path.name)
for name in ['REACTOME_SYNONYMS.tsv','REACTOME_PARENT_CHILD.tsv','REACTOME_HIERARCHY_SIBLINGS.tsv.gz','REACTION_ORDER_BRANCHES.tsv.gz']:copy(R/'catalog_inventory'/name,Path('mapping')/name)
for name in ['BIOPAX_NODES.tsv.gz','REACTION_ENTITY_ROLES.tsv.gz','COMPLEX_AND_ENTITY_SET_MEMBERS.tsv.gz','PROTEIN_STATES.tsv.gz','PROTEIN_MODIFICATIONS.tsv.gz']:copy(R/'structure_reactome98'/name,Path('mechanism')/name)
for name in ['COMMON_GENE_ANNOTATION.tsv','PROTEIN_CODING_SYMBOLS.tsv']:copy(R/'common_rna_scores_r1'/name,Path('scoring_background')/name)
for name in ['ADJUSTMENT_SUPPORT_BY_CANCER.tsv','SOURCE_COVARIATE_COVERAGE.tsv']:copy(R/'adjustment_coverage_r1'/name,Path('summary')/name)
copy(R/'project_log/DATA_RELEASE_AXIS_PREFLIGHT.json','validation/DATA_RELEASE_AXIS_PREFLIGHT.json')
for name in ['VALIDATION.json','COMPLETE.json']:copy(R/'target_composition_index_r1'/name,Path('validation/composition')/name)
copy(R/'functional_y_validation_r2/COMPLETE.json','validation/RELATION_Y_VALIDATION.json');copy(R/'project_log/DELIVERY_CONTENT_AUDIT.json','validation/DELIVERY_CONTENT_AUDIT.json')
for source in (R/'functional_y_validation_r2').glob('*__*.json'):copy(source,Path('validation/contexts')/source.name)
readme="""Functional pathway / State data release — 2026-11-09

Main unit: cancer × lncRNA × canonical Pathway/State identity.
Patient measurements are S. Relationship support is Y.

Y/: 33 TCGA cancers, 5 separate patient contexts, 16,889 lncRNAs and
3,456 fixed target identities. Read with cla_engine.functional_y.
Membership 0 is unlabelled; 1 is one-family or additional support;
2 is strict support from at least two assay families.

PATIENT_SCORE_INDEX.tsv: 660 TCGA context/layer entries and 50 CPTAC
cohort/layer entries. Paths are relative to this directory. Matrices retain
source readout axes; the paired axis table supplies the surviving canonical ID.
For readout_by_patient matrices, target_id is the readout column. For the
shared CPTAC State matrix, apply the listed source_cohort patient filter.

SCORING_REFERENCE_INDEX.tsv: target adjustment and source-fitted score parameters.
CPTAC native methods and references are in mapping/CPTAC_STATE_METHODS.tsv.
Raw association archives and original input expression/CNV files remain in the
source project data store; their hashes and locations are retained in Y receipts.

catalog/: fixed identities, readouts and molecular composition.
mapping/: synonyms, parent-child, shared reaction core, extension, branch and
State-component relations, with source identifiers.
mechanism/: Reactome reaction roles, complexes and protein states.
summary/: measured score coverage and cancer-specific supervision counts.
validation/: independent relationship and table validation records.
"""
(STAGE/'README.txt').write_text(readme)
manifest={p.relative_to(STAGE).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in STAGE.rglob('*') if p.is_file()};record={'target_host':'149','status':'COMPLETE','format':'CLA_FUNCTIONAL_Y_RELEASE_V1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'canonical_targets':3456,'lncrnas':16889,'tcga_cancers':33,'relation_contexts':165,'score_index_entries':len(score_rows),'cptac_cohorts':10,'files':manifest,'source_canonical_sha256':sha(C/'COMPLETE.json'),'source_y_complete_sha256':sha(Y/'COMPLETE.json')};(STAGE/'COMPLETE.json').write_text(json.dumps(record,indent=2));STAGE.rename(DEST)
summary={'target_host':'149','status':'COMPLETE','directory':str(DEST),'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'release_receipt_sha256':sha(DEST/'COMPLETE.json'),'canonical_targets':3456,'relation_contexts':165,'score_index_entries':710,'created_utc':record['created_utc']};(R/'project_log/PROJECT_DATA_RELEASE.json').write_text(json.dumps(summary,indent=2));(ROOT/'logs/functional_y_20261109_r1.json').write_text(json.dumps(summary,indent=2));print('PROJECT_DATA_RELEASE_COMPLETE',json.dumps(summary))
