from pathlib import Path
import json, hashlib, socket, shutil, sys, subprocess
import pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';REPO=R/'source_integration/repo'
C=R/'canonical_targets_r10_exact_members_20261114_r1';N=R/'canonical_targets_r11_state_groups_20261114_r1'
O=W/'state_groups_20261114_r1';G=W/'exact_member_merge_20261114_r1/observation_registry_r3';NG=O/'observation_registry_r4'
sys.path.insert(0,str(REPO))
from cla_engine.state_observation_groups import GROUPS,group_id,state_membership,group_observation_links
from cla_engine.measured_y_registry import load_target,normalize_links,verify_matrix
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def wr(d,p):d.to_csv(p,sep='\t',index=False)
def filehashes(d):return {p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
O.mkdir(exist_ok=True)
sj(O/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='Group cell composition and SBS into two top-level State collections',training=False,lncrna_association=False))
# Extend the measured-readout relation vocabulary for collection membership.
module=REPO/'cla_engine/measured_y_registry.py';text=module.read_text()
if "'state_component_readout'" not in text:
    assert "roles = {'same_state_readout'," in text
    module.write_text(text.replace("roles = {'same_state_readout',","roles = {'same_state_readout', 'state_component_readout',"))
tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_state_observation_groups.py','tests/test_exact_member_groups.py','tests/test_measured_y_registry.py','tests/test_measured_members.py','tests/test_pathway_conditions.py'],cwd=REPO,capture_output=True,text=True)
sj(O/'TESTS.json',dict(returncode=tests.returncode,output=tests.stdout+tests.stderr));print(tests.stdout,flush=True);assert tests.returncode==0
source_receipt=json.loads((C/'COMPLETE.json').read_text())
for name,h in source_receipt['files'].items():assert sha(C/name)==h,name
master=rd(C/'TARGET_MASTER.tsv');pol=rd(C/'TARGET_POLICY.tsv');defs=rd(C/'STATE_SCORE_DEFINITIONS.tsv')
states=master[master.canonical_id.isin(pol.loc[pol.analysis_role.eq('state_auxiliary'),'canonical_id'])].copy()
assert len(states)==197
members=state_membership(states,defs);counts=members.state_family.value_counts().to_dict();assert counts=={'cell_composition':91,'sbs_spectrum':60}
parent_map=members.set_index('component_id').parent_id.to_dict();children=set(parent_map);parents=set(parent_map.values())
assert not N.exists() and not NG.exists(),'Use new output directories'
shutil.copytree(C,N)
newrows=[]
for family,(name,zh) in GROUPS.items():
    row={c:'' for c in master.columns};row.update(canonical_id=group_id(family),display_name=name,entity_type='state_observation_group',authority_id='CLA:STATE_GROUP:'+family,source_database='CancerLncAtlas',membership_gene_count=0,roles='State',definition_version='20261114_r1',has_scored_readout=False,qualifier='Collection of distinct measured components',merged_definition_count=0,merged_source_databases='',source_primary_gene_count=0,integrated_protein_union_count=0,shared_protein_core_count=0,observation_group_basis='biological_family_component_collection')
    newrows.append(row)
master=pd.concat([master,pd.DataFrame(newrows)],ignore_index=True)
master['parent_state_id']=master.canonical_id.map(parent_map).fillna('')
master['state_hierarchy_level']='';master.loc[master.canonical_id.isin(states.canonical_id),'state_hierarchy_level']='top_level'
master.loc[master.canonical_id.isin(children),'state_hierarchy_level']='component';master.loc[master.canonical_id.isin(parents),'state_hierarchy_level']='top_level'
master['display_name_zh']=master.canonical_id.map({group_id(f):GROUPS[f][1] for f in GROUPS}).fillna('')
wr(master,N/'TARGET_MASTER.tsv')
newpol=[]
for family in GROUPS:
    row={c:'' for c in pol.columns};row.update(canonical_id=group_id(family),roles='State',biological_type='state_observation_group',analysis_role='state_auxiliary',loss_group='State',loss_weight=0.5,active_in_analysis=True,cancer_scope_status='existing_state_collection',backbone_shared_protein_count=0,observation_group_basis='biological_family_component_collection');newpol.append(row)
pol.loc[pol.canonical_id.isin(children),'analysis_role']='context_only';pol.loc[pol.canonical_id.isin(children),'loss_group']='none';pol.loc[pol.canonical_id.isin(children),'loss_weight']=0.0
pol=pd.concat([pol,pd.DataFrame(newpol)],ignore_index=True);pol['parent_state_id']=pol.canonical_id.map(parent_map).fillna('');pol['state_hierarchy_level']=pol.canonical_id.map(master.set_index('canonical_id').state_hierarchy_level).fillna('')
assert len(pol[pol.analysis_role.eq('state_auxiliary')])==48
wr(pol,N/'TARGET_POLICY.tsv');wr(members,N/'STATE_GROUP_MEMBERSHIP.tsv')
top=master[master.canonical_id.isin(pol.loc[pol.analysis_role.eq('state_auxiliary'),'canonical_id'])].copy()
top['detail_component_count']=top.canonical_id.map(members.groupby('parent_id').size()).fillna(0).astype(int)
wr(top,N/'STATE_TOP_LEVEL_TARGETS.tsv');wr(master[master.canonical_id.isin(children)],N/'STATE_DETAIL_TARGETS.tsv')
wr(pd.concat([rd(C/'IDENTITY_MASTER.tsv'),pd.DataFrame(newrows)],ignore_index=True).fillna(''),N/'IDENTITY_MASTER.tsv')
sj(N/'STATE_GROUP_DEFINITION.json',dict(primary_pathways=476,state_top_level_count=48,state_source_identity_count=197,state_detail_component_count=151,total_top_level_targets=524,groups=[dict(parent_id=group_id(f),name=GROUPS[f][0],name_zh=GROUPS[f][1],components=counts[f]) for f in GROUPS],relation='component_of',numeric_aggregation='none',automatic_independent_losses=False))
cfg=json.loads((C/'CATALOG_DEFINITION.json').read_text());cfg.update(format='CLA_CANCER_CATALOG_STATE_COLLECTIONS_V1',source_catalog=str(C),state_top_level_count=48,state_source_identity_count=197,state_group_membership='STATE_GROUP_MEMBERSHIP.tsv',state_top_level_targets='STATE_TOP_LEVEL_TARGETS.tsv',target_policy_sha256=sha(N/'TARGET_POLICY.tsv'));sj(N/'CATALOG_DEFINITION.json',cfg)
prep=json.loads((C/'PREPARE_POLICY_BINDING.json').read_text());prep=json.loads(json.dumps(prep).replace(str(C),str(N)));prep.update(observation_registry=str(NG),state_top_level_count=48,state_group_membership=str(N/'STATE_GROUP_MEMBERSHIP.tsv'))
prep['relation_target_policy']['sha256']=sha(N/'TARGET_POLICY.tsv');sj(N/'PREPARE_POLICY_BINDING.json',prep)
sj(N/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='State collection grouping',source_catalog=str(C),source_state_identities=197,state_top_level_count=48))
receipt=dict(source_receipt);receipt.update(format=cfg['format'],source_catalog=str(C),source_catalog_complete_sha256=sha(C/'COMPLETE.json'),analysis_roles=pol.analysis_role.value_counts().to_dict(),display_roles=master.roles.value_counts().to_dict(),state_identities=48,state_source_identity_count=197,state_detail_component_count=151,total_targets=524,files=filehashes(N));sj(N/'COMPLETE.json',receipt)
print('CATALOG_476_PATHWAY_48_STATE',flush=True)
NG.mkdir();checks=[];case_receipts=[]
for ca in ['BRCA','LUAD','COAD']:
    old=G/ca;new=NG/ca;new.mkdir();rec=json.loads((old/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():assert sha(old/name)==h,(ca,name)
    original_targets=rd(old/'TARGETS.tsv');targets=original_targets[~original_targets.canonical_id.isin(children)].copy()
    targetrows=[]
    for pid in sorted(parents):
        row={c:'' for c in targets.columns};row.update(canonical_id=pid,analysis_role='state_auxiliary',display_category='State',condition_required=False,applicability_basis='state_component_collection');targetrows.append(row)
    targets=pd.concat([targets,pd.DataFrame(targetrows)],ignore_index=True);assert len(targets)==524 and targets.canonical_id.is_unique
    wr(targets,new/'TARGETS.tsv');wr(original_targets[original_targets.canonical_id.isin(children)],new/'STATE_CHILD_TARGETS.tsv');wr(members,new/'STATE_GROUP_MEMBERSHIP.tsv')
    obs=rd(old/'OBSERVATIONS.tsv.gz');origlinks=rd(old/'PATHWAY_OBSERVATIONS.tsv.gz');links=group_observation_links(origlinks,members)
    # Module vocabulary was updated before tests; reload for this materializer process.
    import importlib,cla_engine.measured_y_registry as measured_registry
    importlib.reload(measured_registry)
    links=measured_registry.normalize_links(links,obs,targets)
    untouched=origlinks[~origlinks.canonical_id.isin(children)].copy()
    pd.testing.assert_frame_equal(links.loc[~links.canonical_id.isin(parents),list(origlinks)].sort_values(['canonical_id','observation_id','evidence_role']).reset_index(drop=True),untouched.sort_values(['canonical_id','observation_id','evidence_role']).reset_index(drop=True))
    wr(origlinks[origlinks.canonical_id.isin(children)],new/'STATE_CHILD_OBSERVATIONS.tsv.gz')
    obs['bound_target_count']=obs.observation_id.map(links.groupby('observation_id').canonical_id.nunique()).fillna(0).astype(int)
    wr(obs,new/'OBSERVATIONS.tsv.gz');wr(links,new/'PATHWAY_OBSERVATIONS.tsv.gz')
    for name in ['PATIENT_PARTITIONS.json','MATRICES.json','SOURCE_READOUT_METHODS.tsv.gz']:
        shutil.copy2(old/name,new/name)
    refs=json.loads((old/'INPUT_REFERENCES.json').read_text());refs.update(source_registry={'path':str(old/'COMPLETE.json'),'sha256':sha(old/'COMPLETE.json')},group_catalog={'path':str(N/'COMPLETE.json'),'sha256':sha(N/'COMPLETE.json')},state_group_membership={'path':str(N/'STATE_GROUP_MEMBERSHIP.tsv'),'sha256':sha(N/'STATE_GROUP_MEMBERSHIP.tsv')});sj(new/'INPUT_REFERENCES.json',refs)
    coverage=rd(old/'COVERAGE.tsv');wr(coverage[coverage.canonical_id.isin(children)],new/'STATE_CHILD_COVERAGE.tsv');coverage=coverage[~coverage.canonical_id.isin(children)].copy();wr(coverage,new/'COVERAGE.tsv')
    def save_receipt():
        rec.update(format='CLA_PATHWAY_PATIENT_OBSERVATION_REGISTRY_V4_STATE_GROUPS',primary_pathways=476,state_identities=48,state_source_identity_count=197,state_detail_component_count=151,targets=524,observations=len(obs),target_bindings=len(links),catalog=str(N),source_registry=str(old),source_patient_measurement_values_changed=False,files=filehashes(new));sj(new/'COMPLETE.json',rec)
    save_receipt();added=[];matrices=json.loads((new/'MATRICES.json').read_text())
    assert sha(new/'MATRICES.json')==sha(old/'MATRICES.json') and sha(new/'PATIENT_PARTITIONS.json')==sha(old/'PATIENT_PARTITIONS.json')
    for co in ['TCGA','CPTAC']:
        for family in GROUPS:
            pid=group_id(family);ids=set(members.loc[members.parent_id.eq(pid),'component_id']);vv,mm,aa,meta=load_target(new,pid,co,include_external_reference=True)
            expected_obs=set(origlinks.loc[origlinks.canonical_id.isin(ids)&origlinks.is_target_observation,'observation_id']) & set(obs.loc[obs.cohort.eq(co),'observation_id'])
            assert set(vv)==expected_obs and aa.eq(1).all()
            assert not links.loc[links.canonical_id.eq(pid),'independent_loss_enabled'].any()
            # Directly compare every delivered grouped dimension with its original matrix column.
            for matrix_id,meta_block in meta.groupby('matrix_id',sort=False):
                b=matrices['blocks'][matrix_id];cols=meta_block.feature_id.tolist()
                raw=pd.read_parquet(b['values']['path'],columns=cols).reindex(vv.index);mask=pd.read_parquet(b['observed']['path'],columns=cols).reindex(vv.index)
                raw.columns=mask.columns=meta_block.observation_id.tolist()
                pd.testing.assert_frame_equal(vv[raw.columns],raw);pd.testing.assert_frame_equal(mm[mask.columns],mask)
            for layer in sorted(set(meta.layer)):
                columns=meta.loc[meta.layer.eq(layer),'observation_id'].tolist();mask=mm[columns];count=int(mask.any(axis=1).sum());added.append(dict(cancer=ca,cohort=co,canonical_id=pid,layer=layer,defined_observations=len(columns),observations_with_measurement=int(mask.any().sum()),patients_measured=count,patients_total=len(vv),measured_fraction=count/len(vv)))
            checks.append(dict(cancer=ca,cohort=co,parent_id=pid,family=family,components=len(ids),observations=len(vv.columns),patients=len(vv),patients_with_measurement=int(mm.any(axis=1).sum()),values_and_masks_unchanged=True))
            print('GROUP_VERIFIED',ca,co,family,len(vv.columns),flush=True)
    coverage=pd.concat([coverage,pd.DataFrame(added)],ignore_index=True);wr(coverage,new/'COVERAGE.tsv');save_receipt();case_receipts.append(dict(rec))
sj(NG/'COMPLETE.json',dict(target_host='149',format='CLA_STATE_GROUPED_OBSERVATION_COLLECTION_V4',primary_pathways=476,state_identities=48,state_source_identity_count=197,state_detail_component_count=151,targets=524,catalog=str(N),source_registry=str(G),cases=case_receipts,receipt_hashes={ca:sha(NG/ca/'COMPLETE.json') for ca in ['BRCA','LUAD','COAD']},training=False,lncrna_association=False))
definition=json.loads((W/'exact_member_merge_20261114_r1/FIXED_Y_DEFINITION.json').read_text());definition.update(definition_version='KEGG_CANCER_MEASURED_Y_STATE_GROUPS_20261114_R4',registry=str(NG),registry_receipt_sha256=sha(NG/'COMPLETE.json'),catalog=str(N),state_identities=48,state_source_identity_count=197,state_detail_component_count=151,total_top_level_targets=524,state_group_definition=str(N/'STATE_GROUP_DEFINITION.json'),validation=str(O/'VALIDATION.json'));sj(O/'FIXED_Y_DEFINITION.json',definition)
sj(O/'VALIDATION.json',dict(target_host='149',primary_pathways=476,state_before=197,state_after=48,top_level_targets=524,component_counts=counts,retained_other_states=46,checks=checks,source_numeric_blocks_changed=False,source_patient_partitions_changed=False,automatic_independent_losses=False,training=False,lncrna_association=False))
print('STATE_GROUPS_COMPLETE',flush=True)
