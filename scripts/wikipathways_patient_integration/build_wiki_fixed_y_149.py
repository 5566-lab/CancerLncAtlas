from pathlib import Path
import os,socket,json,sys,hashlib,shutil,datetime
os.environ['OPENBLAS_NUM_THREADS']='2'
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'wiki_patient_integration_20261114_r1'
REPO=R/'source_integration/repo';sys.path.insert(0,str(REPO))
import pandas as pd
from cla_engine.wiki_patient_y import merge_primary_observations,load_wikipathways_source
from cla_engine.fixed_y_bundle import load_fixed_target,supervision_mask
from cla_engine.measured_y_registry import verify_matrix

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def wr(x,p):x.to_csv(p,sep='\t',index=False)
def hashes(p):return {x.name:sha(x) for x in p.iterdir() if x.is_file() and x.name!='COMPLETE.json'}
def tree_hashes(p):return {str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file() and x!=p/'COMPLETE.json'}
def verify(p):
 r=json.loads((p/'COMPLETE.json').read_text())
 for name,h in r.get('files',{}).items():
  if isinstance(h,dict):h=h['sha256']
  assert sha(p/name)==h,(p,name)
 return r

sj(O/'INTEGRATION_PREFLIGHT.json',dict(target_host='149',operation='Integrate WikiPathways measured source Y; preserve primary identity and existing measurements',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=False,lncrna_association=False))
OLD=W/'fixed_y_remake_20261114_r1/Y_bundle';SRC=O/'patient_sources';A=O/'source_audit';B=O/'Y_bundle'
NOLD=R/'canonical_targets_r11_state_groups_20261114_r1';N=R/'canonical_targets_r12_wikipathways_patients_20261114_r1'
assert not B.exists() and not N.exists(),'Output exists: inspect before resuming'
assert (SRC/'COMPLETE.json').exists(),'Patient collection incomplete'
verify(OLD);verify(SRC);verify(NOLD)
routes=rd(A/'WP_SOURCE_ROUTES.tsv');exact=routes[routes.route_type.eq('equivalent_definition')].copy();assert len(exact)==3
dispositions=rd(A/'WP_SOURCE_DISPOSITIONS.tsv');assert len(dispositions)==193
source_targets=rd(SRC/'SOURCE_TARGETS.tsv');assert len(source_targets)==193
shutil.copytree(OLD,B);(B/'COMPLETE.json').unlink();(B/'sources').mkdir(exist_ok=True);shutil.copytree(SRC,B/'sources/WikiPathways')
for name in ['WP_SOURCE_ROUTES.tsv','WP_SOURCE_DISPOSITIONS.tsv','WP_EQUIVALENT_ROUTES.tsv']:
 shutil.copy2(A/name,B/name)
shutil.copy2(REPO/'cla_engine/wiki_patient_y.py',B/'wiki_reader.py')
targets=rd(B/'TARGETS.tsv');assert len(targets)==524
stats=[];coverage=[];layer_coverage=[];pathway_layers=[];score_defs=[];source_binding=[];source_summaries=[];cohort_summaries=[];pool_rows=[]
for folder in sorted((B/'cohorts').iterdir()):
 case=folder.name;old=OLD/'cohorts'/case;src=SRC/'cohorts'/case
 verify(old);srec=verify(src)
 ov=pd.read_parquet(old/'Y_VALUES.parquet');om=pd.read_parquet(old/'Y_OBSERVED.parquet');oo=rd(old/'OBSERVATIONS.tsv.gz');ol=rd(old/'TARGET_OBSERVATIONS.tsv.gz')
 sv=pd.read_parquet(src/'Y_VALUES.parquet');sm=pd.read_parquet(src/'Y_OBSERVED.parquet');so=rd(src/'OBSERVATIONS.tsv.gz');sl=rd(src/'SOURCE_OBSERVATIONS.tsv.gz')
 assert list(sv.index)==list(ov.index)
 values,mask,obs,links,provenance=merge_primary_observations(ov,om,oo,ol,sv,sm,so,sl,exact,targets)
 obs['patients_measured']=mask.sum().reindex(obs.observation_id).to_numpy()
 obs['bound_target_count']=obs.observation_id.map(links.groupby('observation_id').canonical_id.nunique()).fillna(0).astype(int)
 values.to_parquet(folder/'Y_VALUES.parquet',compression='zstd');mask.to_parquet(folder/'Y_OBSERVED.parquet',compression='zstd')
 wr(obs,folder/'OBSERVATIONS.tsv.gz');wr(links,folder/'TARGET_OBSERVATIONS.tsv.gz');wr(provenance,folder/'WP_EQUIVALENT_OBSERVATION_PROVENANCE.tsv.gz')
 # A source-to-backbone context link connects only an already-admitted member
 # observation to the backbone. The whole source score remains source-specific.
 member_obs=set(so.loc[so.kind.eq('member_measurement'),'observation_id'])
 direct=sl.copy()
 if 'is_source_observation' in direct:direct=direct[direct.is_source_observation.astype(str).str.lower().isin(['true','1'])]
 if 'evidence_role' in direct:direct=direct[~direct.evidence_role.eq('ambiguous_group_context')]
 shared=direct[direct.observation_id.isin(member_obs)].merge(routes[['source_target_id','canonical_id','route_type']],on='source_target_id')
 shared=shared.merge(ol.loc[ol.evidence_role.eq('member_measurement'),['canonical_id','observation_id']].drop_duplicates(),on=['canonical_id','observation_id'])
 shared=shared.drop_duplicates(['source_target_id','canonical_id','observation_id'])
 wr(shared,folder/'WP_SHARED_MEMBER_PROVENANCE.tsv.gz')
 applicable=pd.read_parquet(folder/'TARGET_APPLICABILITY.parquet');eligible=obs.set_index('observation_id').supervision_eligible
 cr=[]
 for row in targets.to_dict('records'):
  cid=row['canonical_id'];ids=set(links.loc[links.canonical_id.eq(cid),'observation_id']);cols=[c for c in values if c in ids]
  mm=mask[cols];qualified=supervision_mask(mm,eligible.loc[cols],applicable[cid]);meta=obs.set_index('observation_id').loc[cols]
  cr.append(dict(case=case,cancer=case.split('_')[1],cohort=case.split('_')[0],canonical_id=cid,name=row['display_name'],analysis_role=row['analysis_role'],defined_dimensions=len(cols),measured_dimensions=int(mm.any().sum()),eligible_dimensions=int(qualified.any().sum()),patients_measured=int(mm.any(axis=1).sum()),patients_supervision=int(qualified.any(axis=1).sum()),patients_total=len(values),condition_unknown=int(applicable[cid].isna().sum())))
 coverage.extend(cr)
 for (kind,layer),g in obs.groupby(['kind','layer']):
  m=mask[g.observation_id];layer_coverage.append(dict(case=case,kind=kind,layer=layer,defined_dimensions=len(g),measured_dimensions=int(m.any().sum()),patients_measured=int(m.any(axis=1).sum()),patients_total=len(m),measured_fraction=float(m.to_numpy().mean())))
 for layer,g in obs.groupby('layer'):
  lids=set(g.observation_id);npath=0
  for cid in targets.loc[targets.analysis_role.eq('primary_pathway'),'canonical_id']:
   ids=list(lids&set(links.loc[links.canonical_id.eq(cid),'observation_id']))
   npath+=int(mask[ids].any().any())
  pathway_layers.append(dict(case=case,layer=layer,pathways_with_measurement=npath))
 readouts=so[so.kind.eq('source_readout')].copy();readouts['case']=case;score_defs.append(readouts)
 exact_readout_ids=set(provenance.observation_id)&set(readouts.observation_id)
 ss=dict(case=case,patients_total=len(values),source_definitions=193,source_observation_dimensions=len(so),source_measured_dimensions=int(sm.any().sum()),source_patients_measured=int(sm.any(axis=1).sum()),source_score_dimensions=len(readouts),source_score_dimensions_measured=int(sm[readouts.observation_id].any().sum()),primary_dimensions_before=len(ov.columns),primary_dimensions_after=len(values.columns),new_primary_dimensions=len(values.columns)-len(ov.columns),primary_wp_source_score_dimensions=len(exact_readout_ids),primary_wp_measured_source_score_dimensions=int(mask[list(exact_readout_ids)].any().sum()),source_member_dimensions_already_in_primary=len(set(ov)&member_obs),shared_member_route_count=len(shared),primary_pathways=476,state_top_level=48,primary_pathways_with_supervision=sum(x['analysis_role']=='primary_pathway' and x['patients_supervision']>0 for x in cr))
 stats.append(ss);assert ss['primary_pathways_with_supervision']==335
 cs=rd(OLD/'COHORT_SUMMARY.tsv').set_index('case').loc[case].to_dict();cs.update(case=case,defined_dimensions=len(obs),measured_dimensions=int(mask.any().sum()));cohort_summaries.append(cs)
 refs=json.loads((folder/'SOURCE_RECEIPTS.json').read_text());legacy=rd(Path(refs['source_registry'])/'OBSERVATIONS.tsv.gz')
 legacy=legacy[legacy.cohort.eq(case.split('_')[0])];pool_ids=set(legacy.observation_id)|set(so.observation_id)
 assert set(obs.observation_id)<=pool_ids
 pool_rows.append(dict(case=case,source_pool=len(pool_ids),bound_Y_dimensions=len(obs),unbound_or_context_dimensions=len(pool_ids-set(obs.observation_id)),wiki_source_definitions=193))
 refs['wikipathways']={'source_cohort':str(src),'source_receipt_sha256':sha(src/'COMPLETE.json'),'route_file':str(A/'WP_SOURCE_ROUTES.tsv'),'route_sha256':sha(A/'WP_SOURCE_ROUTES.tsv')};sj(folder/'SOURCE_RECEIPTS.json',refs)
 for wp,g in sl.groupby('source_target_id'):
  ids=sorted(set(g.observation_id));meta=so.set_index('observation_id').loc[ids]
  for (kind,layer),group in meta.groupby(['kind','layer']):
   m=sm[group.index];source_summaries.append(dict(case=case,source_target_id=wp,kind=kind,layer=layer,defined_dimensions=len(group),measured_dimensions=int(m.any().sum()),patients_measured=int(m.any(axis=1).sum()),patients_total=len(sm)))
 pd.testing.assert_frame_equal(values[ov.columns],ov);pd.testing.assert_frame_equal(mask[om.columns],om)
 receipt=json.loads((old/'COMPLETE.json').read_text());receipt.update(format='CLA_FIXED_Y_COHORT_WITH_WP_V1',defined_dimensions=len(obs),measured_dimensions=int(mask.any().sum()),wiki_additions=ss,files=hashes(folder));sj(folder/'COMPLETE.json',receipt)
 # Test the actual reader and unique-ID view, not only a construction function.
 for cid in exact.canonical_id:
  rv,rm,ra,rmeta=load_fixed_target(B,case,cid,for_supervision=True)
  assert rv.columns.is_unique and rv.index.equals(ov.index)
  assert not (rm & rv.isna()).any().any()
 rv,rm,rmeta=load_wikipathways_source(B,case,'WP531',layer='RNA',kind='source_readout')
 assert rv.index.equals(ov.index)
 print('INTEGRATED',json.dumps(ss),flush=True)

wr(pd.DataFrame(stats),B/'WP_INTEGRATION_SUMMARY.tsv');wr(pd.DataFrame(source_summaries),B/'WP_SOURCE_COVERAGE.tsv')
wr(pd.DataFrame(coverage),B/'TARGET_COVERAGE.tsv');wr(pd.DataFrame(layer_coverage),B/'LAYER_COVERAGE.tsv')
wr(pd.DataFrame(cohort_summaries),B/'COHORT_SUMMARY.tsv');wr(pd.DataFrame(pathway_layers),B/'PATHWAY_LAYER_COVERAGE.tsv')
wr(pd.DataFrame(pool_rows),B/'POOL_ROUTING.tsv')
all_scores=pd.concat(score_defs,ignore_index=True);wr(all_scores,B/'WP_SCORE_DEFINITIONS.tsv')
shutil.copytree(NOLD,N)
sj(N/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='Add WikiPathways source observation definitions and repaired composition routes',original_catalog=str(NOLD)))
mapping=rd(N/'DEFINITION_TO_IDENTITY.tsv');scored=set(all_scores.loc[all_scores.patients_measured.gt(0),'source_target_id'])
wp=mapping.database_family.eq('WIKIPATHWAYS');mapping.loc[wp,'has_scored_readout']=mapping.loc[wp,'node_id'].map(lambda x:x.rsplit(':',1)[-1] in scored)
wr(mapping,N/'DEFINITION_TO_IDENTITY.tsv')
assert (A/'BACKBONE_ALIGNMENT_WITH_WP.tsv').exists(),'Require coherent joint-BH table'
shutil.copy2(A/'BACKBONE_ALIGNMENT_WITH_WP.tsv',N/'BACKBONE_ALIGNMENT.tsv')
for name in ['ALIGNMENT_SUMMARY.json','BACKBONE_ALIGNMENT_TOP3.tsv']:
 shutil.copy2(A/name,N/name)
for name in ['WP_SOURCE_ROUTES.tsv','WP_SOURCE_DISPOSITIONS.tsv','WP_EQUIVALENT_ROUTES.tsv']:
 shutil.copy2(A/name,N/name)
for name in ['SOURCE_TARGETS.tsv','SOURCE_MEMBERS.tsv.gz']:
 if (SRC/name).exists():shutil.copy2(SRC/name,N/('WP_'+name))
shutil.copy2(B/'WP_SCORE_DEFINITIONS.tsv',N/'WP_SCORE_DEFINITIONS.tsv');shutil.copy2(B/'WP_SOURCE_COVERAGE.tsv',N/'WP_SOURCE_COVERAGE.tsv')
bindings=rd(N/'SOURCE_ID_BINDINGS.tsv')
rows=[dict(source_target_id=x.source_target_id,source_column='',canonical_id=x.canonical_id,source_name=x.source_name,source_database='WIKIPATHWAYS',relation='same_functional_identity',identity_evidence='existing_reviewed_functional_scope') for x in exact.itertuples()]
bindings=pd.concat([bindings,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(['source_target_id','source_column','canonical_id']);wr(bindings,N/'SOURCE_ID_BINDINGS.tsv')
score_registry=rd(N/'OBSERVED_SCORE_DEFINITIONS.tsv');exact_ids=exact.set_index('source_target_id').canonical_id.to_dict()
wp_identity={x.node_id.rsplit(':',1)[-1]:x.integrated_canonical_id for x in mapping[mapping.database_family.eq('WIKIPATHWAYS')].itertuples()}
rows=[]
for x in all_scores.drop_duplicates('observation_id').to_dict('records'):
 rows.append(dict(readout_id=x['observation_id'],canonical_id=wp_identity[x['source_target_id']],source_target_id=x['source_target_id'],platform=x['cohort'],measurement_layer=x['layer'],method=x['method'],score_unit=x['units'],source_node_id='REF:WikiPathways:'+x['source_target_id']))
wr(pd.concat([score_registry,pd.DataFrame(rows)],ignore_index=True),N/'OBSERVED_SCORE_DEFINITIONS.tsv')
nr=json.loads((NOLD/'COMPLETE.json').read_text());nr.update(format='CLA_CANCER_CATALOG_WP_PATIENTS_V1',source_catalog_complete_sha256=sha(NOLD/'COMPLETE.json'),wikipathways_source_definitions=193,wikipathways_equivalent_definitions=3,original_primary_members_unchanged=True,files=tree_hashes(N));sj(N/'COMPLETE.json',nr)
definition=json.loads((B/'DEFINITION.json').read_text());definition.update(catalog=str(N),catalog_sha256=sha(N/'COMPLETE.json'),wikipathways_source_collection='sources/WikiPathways',source_target_accessor='wiki_reader.load_wikipathways_source',source_context_policy='Only reviewed equivalent definitions contribute whole-source observations to primary Y; non-equivalent definitions have source-level observations and shared-member/context routes.',wiki_integration=stats,cohorts=cohort_summaries)
sj(B/'DEFINITION.json',definition)
(B/'使用说明.txt').write_text((OLD/'使用说明.txt').read_text()+'''\nWikiPathways患者观测补充\n收集193条来源通路定义；各自名称、成员和患者观测在sources/WikiPathways中。\n3条已核实与主干描述同一过程：WP438非同源末端连接、WP531错配修复、WP561血红素生物合成。\n这些定义中的可用成员测量和来源评分进入对应主干Y；成员测量沿用唯一观测ID，不因来源重复计数。\n其他来源整体评分保留来源级身份；通过组成关联连接共享成员，不能作为同一个精确主干的整体分数。\nWP_SOURCE_DISPOSITIONS.tsv列出全部193条的保留关系与原因；零交集来源不进入癌症主干。\nRNA评分使用蛋白编码背景的原始ssGSEA；蛋白评分保留各测量平台方法，磷酸化等保留成员实测值。\n未达到评分适用条件的来源保留缺失及原因，不能把保留定义数当作有评分数。\n''',encoding='utf8')
root_receipt=json.loads((OLD/'COMPLETE.json').read_text());root_receipt.update(format='CLA_FIXED_MEASURED_Y_WITH_WP_V1',catalog=str(N),wiki_source_receipt_sha256=sha(SRC/'COMPLETE.json'),cohort_receipts={p.parent.name:sha(p) for p in (B/'cohorts').glob('*/COMPLETE.json')},files=tree_hashes(B));sj(B/'COMPLETE.json',root_receipt)
sj(O/'INTEGRATION_COMPLETE.json',dict(target_host='149',Y_bundle=str(B),catalog=str(N),source_collection=str(SRC),existing_patient_values_unchanged=True,outer_test_opened=False,training=False,lncrna_association=False,primary_pathways=476,state_top_level=48,patients_total=sum(x['patients_total'] for x in stats),cohorts=stats,files={str(B/'COMPLETE.json'):sha(B/'COMPLETE.json'),str(N/'COMPLETE.json'):sha(N/'COMPLETE.json')}))
print('FIXED_Y_WIKI_COMPLETE',B,flush=True)
