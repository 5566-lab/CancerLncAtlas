from pathlib import Path
import socket,json,sys,shutil,hashlib,subprocess
import pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';B=W/'measured_y_definition_r1';O=W/'exact_member_merge_20261114_r1';REPO=R/'source_integration/repo'
C=R/'canonical_targets_r9_cancer_evidence_20261111_r6';N=R/'canonical_targets_r10_exact_members_20261114_r1';G=B/'observation_registry_r2';NG=O/'observation_registry_r3'
sys.path.insert(0,str(REPO))
from cla_engine.exact_member_groups import exact_member_groups,merge_redirects,grouped_applicability
from cla_engine.measured_y_registry import load_target
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def wr(d,p):d.to_csv(p,sep='\t',index=False)
O.mkdir(exist_ok=True)
sj(O/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='materialize exact-member merged dictionary and measured Y',training=False,lncrna_association=False,outer_test_opened=False))
t=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_exact_member_groups.py','tests/test_measured_y_registry.py','tests/test_measured_members.py','tests/test_pathway_conditions.py'],cwd=REPO,capture_output=True,text=True)
sj(O/'TESTS.json',dict(returncode=t.returncode,output=t.stdout+t.stderr));print(t.stdout,flush=True);assert t.returncode==0,t.stderr
receipt=json.loads((C/'COMPLETE.json').read_text())
for name,h in receipt['files'].items():assert sha(C/name)==h,name
cfg=json.loads((C/'CATALOG_DEFINITION.json').read_text());back=rd(Path(cfg['backbone_path'])/'KEGG_CANCER_BACKBONE.tsv');pc=rd(C/'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz')
sets={s:set(d.uniprot_id) for s,d in pc.groupby('source_definition')};groups=exact_member_groups(back.to_dict('records'),sets);redirect=merge_redirects(groups);multi=[g for g in groups if g['merged']]
assert len(groups)==476 and len(multi)==1
group=multi[0];KEEP=group['canonical_id'];IDS=[r['canonical_id'] for r in group['members']];DROP=next(i for i in IDS if i!=KEEP)
assert set(IDS)=={'CLA-P-08A1A41F05A3E9ED','CLA-P-549730F42F0B148D'}
NAME='Hedgehog signaling, shared SMO/PTCH1 downstream module';CN='Hedgehog 信号：SMO/PTCH1 共同下游模块'
source_names={r['canonical_id']:r['network_name'] for r in back.to_dict('records')}
sj(O/'GROUP_DEFINITION.json',dict(target_host='149',canonical_id=KEEP,display_name=NAME,display_name_zh=CN,member_set_sha256=group['member_set_sha256'],proteins=group['protein_members'],source_mechanisms=group['members'],rule='nonempty original primary-source protein sets exactly identical (J=1)',source_events_remain_distinct_annotations=True,group_requires_specific_upstream_event=False))
assert not N.exists(),'Use a new catalog directory'
shutil.copytree(C,N)
rowdrop={'TARGET_MASTER.tsv','IDENTITY_MASTER.tsv','TARGET_POLICY.tsv','PRIMARY_PATHWAY_TARGETS.tsv','MECHANISM_TARGETS.tsv'}
special_original={'DEFINITION_TO_IDENTITY.tsv','FUNCTIONAL_SCOPE_DECISIONS.tsv','CURATED_EXTERNAL_RELATIONS.tsv'}
for p in sorted(N.iterdir()):
    if not (p.name.endswith('.tsv') or p.name.endswith('.tsv.gz')):continue
    d=rd(p)
    if p.name in rowdrop and 'canonical_id' in d:d=d[d.canonical_id.ne(DROP)].copy()
    for col in d.columns:
        active=('canonical_id' in col and not col.startswith(('previous','original','source_definition')))
        if p.name in special_original:active=col=='integrated_canonical_id'
        if active:d[col]=d[col].map(lambda v:redirect.get(v,v))
    for col in ['child_id','parent_id','identity_a','identity_b']:
        if col in d and p.name not in special_original:d[col]=d[col].map(lambda v:redirect.get(v,v))
    if p.name in rowdrop and 'canonical_id' in d:
        ix=d.canonical_id.eq(KEEP)
        for col in ['display_name']:
            if col in d:d.loc[ix,col]=NAME
        if 'entity_type' in d:d.loc[ix,'entity_type']='pathway_observation_group'
        if 'biological_type' in d:d.loc[ix,'biological_type']='pathway_observation_group'
        if 'merged_definition_count' in d:d.loc[ix,'merged_definition_count']=2
        if 'definition_count' in d:d.loc[ix,'definition_count']=2
        if 'existing_target_count' in d:d.loc[ix,'existing_target_count']=2
        if 'source_definition_ids' in d:d.loc[ix,'source_definition_ids']='|'.join(IDS)
        d['observation_group_basis']=d.get('observation_group_basis','')
        d.loc[ix,'observation_group_basis']='identical_original_protein_members_J1'
    if p.name.startswith('BACKBONE_ALIGNMENT'):
        d.loc[d.target_canonical_id.eq(KEEP),'target_name']=NAME
    wr(d,p)
score=pd.read_parquet(N/'SCORE_REGISTRY.parquet')
for col in score:
    if 'canonical_id' in col and not col.startswith(('previous','original')):score[col]=score[col].map(lambda v:redirect.get(v,v))
score.to_parquet(N/'SCORE_REGISTRY.parquet',index=False)
members=[]
for g in groups:
    for r in g['members']:
        members.append(dict(source_canonical_id=r['canonical_id'],canonical_id=g['canonical_id'],source_network_id=r['network_id'],source_name=r['network_name'],display_name=NAME if g['merged'] else r['network_name'],protein_count=len(g['protein_members']),member_set_sha256=g['member_set_sha256'],merged=g['merged'],representative=r['canonical_id']==g['canonical_id'],source_network_type=r['network_type']))
wr(pd.DataFrame(members),N/'EXACT_MEMBER_GROUP_MEMBERS.tsv')
wr(back[back.canonical_id.isin(IDS)],N/'GROUP_SOURCE_MECHANISMS.tsv')
red=rd(N/'PROTEIN_SCOPE_ID_REDIRECTS.tsv');red=pd.concat([red,pd.DataFrame([dict(previous_canonical_id=DROP,canonical_id=KEEP,relation='same_observation_group',basis='user_authorized_identical_protein_members_J1')])],ignore_index=True);wr(red,N/'PROTEIN_SCOPE_ID_REDIRECTS.tsv')
red=rd(N/'ID_REDIRECTS.tsv');red=pd.concat([red,pd.DataFrame([dict(previous_internal_id=DROP,canonical_id=KEEP,relation='same_observation_group',evidence='user_authorized_identical_protein_members_J1')])],ignore_index=True);wr(red,N/'ID_REDIRECTS.tsv')
groupback=back[back.canonical_id.ne(DROP)].copy();groupback.loc[groupback.canonical_id.eq(KEEP),'display_name']=NAME;groupback.loc[groupback.canonical_id.eq(KEEP),'network_name']=NAME
wr(groupback,N/'GROUPED_BACKBONE.tsv')
cfg.update(format='CLA_CANCER_RELATION_CATALOG_EXACT_MEMBER_GROUPS_V1',primary_pathway_count=476,source_catalog=str(C),source_primary_pathway_count=477,exact_member_groups='EXACT_MEMBER_GROUP_MEMBERS.tsv',grouped_backbone='GROUPED_BACKBONE.tsv',source_backbone_path=cfg.pop('backbone_path'),group_definition=str(O/'GROUP_DEFINITION.json'),target_policy_sha256=sha(N/'TARGET_POLICY.tsv'))
sj(N/'CATALOG_DEFINITION.json',cfg)
prep=json.loads((N/'PREPARE_POLICY_BINDING.json').read_text());prep['primary_pathway_count']=476;prep['exact_member_groups']=str(N/'EXACT_MEMBER_GROUP_MEMBERS.tsv')
binding=prep['relation_target_policy'];binding.update(path=str(N/'TARGET_POLICY.tsv'),sha256=sha(N/'TARGET_POLICY.tsv'),source_identity_map=str(N/'SOURCE_IDENTITY_MAP.tsv'),source_identity_map_sha256=sha(N/'SOURCE_IDENTITY_MAP.tsv'))
binding['source_backbone_path']=binding.pop('backbone_path');binding['source_backbone_manifest_sha256']=binding.pop('backbone_manifest_sha256');binding['grouped_backbone']={'path':str(N/'GROUPED_BACKBONE.tsv'),'sha256':sha(N/'GROUPED_BACKBONE.tsv')}
for name,meta in binding['context_graph'].items():meta.update(path=str(N/name),sha256=sha(N/name))
prep.update(format='CLA_MEASURED_Y_EXACT_GROUP_BINDING_V1',observation_registry=str(NG),training_authorized=False);sj(N/'PREPARE_POLICY_BINDING.json',prep)
sj(N/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='exact-member grouping of frozen primary pathways',source_catalog=str(C),source_primary_count=477,grouped_primary_count=476))
pol=rd(N/'TARGET_POLICY.tsv');assert pol.canonical_id.is_unique and int(pol.analysis_role.eq('primary_pathway').sum())==476
receipt.update(format=cfg['format'],primary_pathways=476,analysis_roles=pol.analysis_role.value_counts().to_dict(),source_catalog=str(C),source_catalog_complete_sha256=sha(C/'COMPLETE.json'),exact_member_group_count=1,source_mechanisms_preserved=477,display_roles=rd(N/'TARGET_MASTER.tsv').roles.value_counts().to_dict())
receipt['source_backbone_official_cancer_supported']=receipt['backbone_official_cancer_supported']
receipt['backbone_official_cancer_supported']=int((pol.analysis_role.eq('primary_pathway') & pol.official_cancer_supported.astype(str).str.lower().eq('true')).sum())
sj(N/'COMPLETE.json',dict(receipt,files={p.name:sha(p) for p in N.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
print('CATALOG_COMPLETE',476,flush=True)

assert not NG.exists(),'Use a new measured registry directory'
NG.mkdir();checks=[];case_receipts=[]
source_collection=json.loads((G/'COMPLETE.json').read_text())
for ca in ['BRCA','LUAD','COAD']:
    assert sha(G/ca/'COMPLETE.json')==source_collection['receipt_hashes'][ca]
    old=G/ca;new=NG/ca;new.mkdir();rec=json.loads((old/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():assert sha(old/name)==h
    for name in ['PATIENT_PARTITIONS.json','INPUT_REFERENCES.json']:shutil.copy2(old/name,new/name)
    targets=rd(old/'TARGETS.tsv');targets=targets[targets.canonical_id.ne(DROP)].copy();targets.loc[targets.canonical_id.eq(KEEP),'condition_required']=False
    targets['applicability_basis']='original_source_definition';targets.loc[targets.canonical_id.eq(KEEP),'applicability_basis']='shared_member_module_no_specific_upstream_event_required';wr(targets,new/'TARGETS.tsv')
    obs=rd(old/'OBSERVATIONS.tsv.gz');li=rd(old/'PATHWAY_OBSERVATIONS.tsv.gz');li['source_canonical_id']=li.canonical_id;li.canonical_id=li.canonical_id.map(lambda v:redirect.get(v,v))
    merged=li.canonical_id.eq(KEEP);li.loc[merged,'condition_required']=False;li.loc[merged,'scope_relation']='exact_member_observation_group'
    sources=li.groupby(['canonical_id','observation_id','evidence_role']).source_canonical_id.agg(lambda v:'|'.join(sorted(set(v))))
    li=li.drop_duplicates(['canonical_id','observation_id','evidence_role']).drop(columns='source_canonical_id').merge(sources.rename('source_canonical_ids').reset_index(),on=['canonical_id','observation_id','evidence_role'],validate='one_to_one')
    obs['bound_target_count']=obs.observation_id.map(li.groupby('observation_id').canonical_id.nunique()).fillna(0).astype(int)
    wr(obs,new/'OBSERVATIONS.tsv.gz');wr(li,new/'PATHWAY_OBSERVATIONS.tsv.gz')
    methods=rd(old/'SOURCE_READOUT_METHODS.tsv.gz');methods['source_canonical_id']=methods.canonical_id;methods.canonical_id=methods.canonical_id.map(lambda v:redirect.get(v,v));wr(methods,new/'SOURCE_READOUT_METHODS.tsv.gz')
    matrices=json.loads((old/'MATRICES.json').read_text());matrices['source_event_conditions']=dict(matrices['conditions'])
    for co,meta in list(matrices['conditions'].items()):
        assert sha(meta['path'])==meta['sha256'];original=pd.read_parquet(meta['path']);ap=grouped_applicability(original,groups)
        assert original[IDS].isna().all().all();assert ap[KEEP].eq(1).all();assert DROP not in ap
        dest=new/(co+'_GROUP_APPLICABILITY.parquet');ap.to_parquet(dest);matrices['conditions'][co]={'path':str(dest),'sha256':sha(dest)}
    sj(new/'MATRICES.json',matrices)
    refs=json.loads((new/'INPUT_REFERENCES.json').read_text());refs.update(source_registry={'path':str(old/'COMPLETE.json'),'sha256':sha(old/'COMPLETE.json')},group_catalog={'path':str(N/'COMPLETE.json'),'sha256':sha(N/'COMPLETE.json')},group_definition={'path':str(O/'GROUP_DEFINITION.json'),'sha256':sha(O/'GROUP_DEFINITION.json')});sj(new/'INPUT_REFERENCES.json',refs)
    coverage=rd(old/'COVERAGE.tsv');oldcov=coverage[coverage.canonical_id.isin(IDS)].copy();coverage=coverage[~coverage.canonical_id.isin(IDS)].copy();wr(coverage,new/'COVERAGE.tsv')
    def write_receipt():
        rec.update(format='CLA_PATHWAY_PATIENT_OBSERVATION_REGISTRY_V3_EXACT_MEMBERS',primary_pathways=476,targets=len(targets),observations=len(obs),target_bindings=len(li),catalog=str(N),source_registry=str(old),source_conditions_preserved=True,source_patient_measurement_values_changed=False,files={p.name:sha(p) for p in new.iterdir() if p.is_file() and p.name!='COMPLETE.json'})
        sj(new/'COMPLETE.json',rec)
    write_receipt()
    newcov=[]
    for co in ['TCGA','CPTAC']:
        v,mask,app,metadata=load_target(new,KEEP,co,include_external_reference=True)
        av,am,aa,amd=load_target(old,KEEP,co,include_external_reference=True);bv,bm,ba,bmd=load_target(old,DROP,co,include_external_reference=True)
        union=set(av.columns)|set(bv.columns);assert set(v.columns)==union
        for source_v,source_m in [(av,am),(bv,bm)]:
            pd.testing.assert_frame_equal(v[source_v.columns],source_v)
            pd.testing.assert_frame_equal(mask[source_m.columns],source_m)
        assert app.eq(1).all() and aa.isna().all() and ba.isna().all()
        for layer in sorted(set(oldcov.layer)|set(metadata.layer)):
            ids=metadata.loc[metadata.layer.eq(layer),'observation_id'].tolist();mm=mask[ids];num=int(mm.any(axis=1).sum())
            newcov.append(dict(cancer=ca,cohort=co,canonical_id=KEEP,layer=layer,defined_observations=len(ids),observations_with_measurement=int(mm.any().sum()),patients_measured=num,patients_total=len(v),measured_fraction=num/len(v)))
        checks.append(dict(cancer=ca,cohort=co,patients=len(v),source_observation_counts=[len(av.columns),len(bv.columns)],merged_unique_observations=len(v.columns),shared_member_values_unchanged=True,source_event_conditions_unknown=True,merged_module_requires_specific_event=False,patient_partition_sha256=sha(new/'PATIENT_PARTITIONS.json')))
        print('REGISTRY_VERIFIED',ca,co,len(v),len(v.columns),flush=True)
    coverage=pd.concat([coverage,pd.DataFrame(newcov)],ignore_index=True);wr(coverage,new/'COVERAGE.tsv');write_receipt();case_receipts.append(dict(rec))
sj(NG/'COMPLETE.json',dict(target_host='149',format='CLA_PATHWAY_PATIENT_OBSERVATION_COLLECTION_V3_EXACT_MEMBERS',primary_pathways=476,state_identities=197,targets=673,source_registry=str(G),catalog=str(N),cases=case_receipts,receipt_hashes={ca:sha(NG/ca/'COMPLETE.json') for ca in ['BRCA','LUAD','COAD']},training=False,lncrna_association=False))
sj(O/'VALIDATION.json',dict(target_host='149',primary_before=477,primary_after=476,state_identities=197,total_targets=673,merged_groups=1,removed_active_identity=DROP,representative_identity=KEEP,display_name=NAME,source_matrix_blocks_changed=False,source_event_conditions_changed=False,checks=checks,tests=sha(O/'TESTS.json')))
definition=json.loads((B/'final_definition_review_r1/FIXED_Y_DEFINITION.json').read_text());definition.update(definition_version='KEGG_CANCER_MEASURED_Y_20261114_EXACT_MEMBERS_R3',registry=str(NG),registry_receipt_sha256=sha(NG/'COMPLETE.json'),catalog=str(N),primary_pathways=476,group_definition=str(O/'GROUP_DEFINITION.json'),validation=str(O/'VALIDATION.json'),condition_policy='Shared exact-member modules require no specific upstream event; original event applicability remains separately preserved')
sj(O/'FIXED_Y_DEFINITION.json',definition)
print('EXACT_MEMBER_MERGE_VERIFIED',flush=True)
