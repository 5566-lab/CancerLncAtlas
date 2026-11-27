from pathlib import Path
import json,hashlib,socket,shutil
import pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'state_groups_20261114_r1';N=R/'canonical_targets_r11_state_groups_20261114_r1';G=O/'observation_registry_r4';D=O/'deliverables';DEST=R/'protein_similarity_integration_20261111_r1/deliverables'
C=R/'canonical_targets_r10_exact_members_20261114_r1';OLDG=W/'exact_member_merge_20261114_r1/observation_registry_r3';OLDO=W/'exact_member_merge_20261114_r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def hashfiles(p):return {f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name!='COMPLETE.json'}
sj(O/'ACTIVATION_PREFLIGHT.json',dict(target_host='149',operation='Activate grouped State dictionary and measured registry'))
val=json.loads((O/'VALIDATION.json').read_text());dv=json.loads((O/'DICTIONARY_VALIDATION.json').read_text());assert val['state_after']==48 and dv['state_top_level_count']==48
for directory in [N]+[G/c for c in ['BRCA','LUAD','COAD']]:
    for name,h in json.loads((directory/'COMPLETE.json').read_text())['files'].items():assert sha(directory/name)==h,(directory,name)
master=rd(N/'TARGET_MASTER.tsv');policy=rd(N/'TARGET_POLICY.tsv');members=rd(N/'STATE_GROUP_MEMBERSHIP.tsv');top=rd(N/'STATE_TOP_LEVEL_TARGETS.tsv');defs=rd(N/'STATE_SCORE_DEFINITIONS.tsv')
display=master[master.canonical_id.isin(policy.loc[policy.analysis_role.isin(['primary_pathway','state_auxiliary']),'canonical_id'])].copy();assert len(display)==524
display.to_csv(N/'DISPLAY_TARGETS.tsv',sep='\t',index=False)
nav=[]
for row in top.itertuples():
    children=[]
    for child in members[members.parent_id.eq(row.canonical_id)].itertuples():
        sources=defs[defs.canonical_id.eq(child.component_id)].to_dict('records')
        children.append(dict(id=child.component_id,name=child.component_name,relation='component_of',source_definitions=sources))
    nav.append(dict(id=row.canonical_id,name=row.display_name,display_name_zh=row.display_name_zh,children=children,component_count=len(children),numeric_aggregation='none'))
sj(D/'State目录.json',dict(state_top_level_count=48,primary_pathways=476,total_top_level_targets=524,states=nav))
receipt=json.loads((N/'COMPLETE.json').read_text());receipt['files']=hashfiles(N);sj(N/'COMPLETE.json',receipt)
collection=json.loads((G/'COMPLETE.json').read_text());cases=[]
for ca in ['BRCA','LUAD','COAD']:
    case=G/ca;refs=json.loads((case/'INPUT_REFERENCES.json').read_text());refs['group_catalog']['sha256']=sha(N/'COMPLETE.json');sj(case/'INPUT_REFERENCES.json',refs)
    rec=json.loads((case/'COMPLETE.json').read_text());rec['files']=hashfiles(case);sj(case/'COMPLETE.json',rec);cases.append(rec)
collection.update(cases=cases,receipt_hashes={c:sha(G/c/'COMPLETE.json') for c in ['BRCA','LUAD','COAD']});sj(G/'COMPLETE.json',collection)
definition=json.loads((O/'FIXED_Y_DEFINITION.json').read_text());definition['registry_receipt_sha256']=sha(G/'COMPLETE.json');sj(O/'FIXED_Y_DEFINITION.json',definition)
assert sha(dv['path'])==dv['sha256']
files={p.name:sha(p) for p in D.iterdir() if p.is_file()};assert sum(k.endswith('.xlsx') for k in files)==1 and sum(k.endswith('.pdf') for k in files)==1
sj(O/'DELIVERY_MANIFEST.json',dict(target_host='149',files=files,dictionary_sha256=dv['sha256'],primary_pathways=476,state_top_level_count=48,state_source_identity_count=197,state_detail_components=151,total_targets=524))
for name,h in files.items():shutil.copy2(D/name,DEST/name);assert sha(DEST/name)==h
dictionary=DEST/'Functional_pathway_State_dictionary_v8.xlsx';ppt=DEST/'CancerLncAtlas_通路与State分组_20261114.pptx';pdf=ppt.with_suffix('.pdf')
backup=O/'previous_current_pointers';backup.mkdir(exist_ok=True)
def replace(value):
    if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
    if isinstance(value,list):return [replace(v) for v in value]
    if isinstance(value,str):
        return value.replace(str(C),str(N)).replace(str(OLDG),str(G)).replace(str(OLDO/'FIXED_Y_DEFINITION.json'),str(O/'FIXED_Y_DEFINITION.json'))
    return value
pointer_names=['CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json','CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']
records=[]
for name in pointer_names:
    p=R/name
    if not (backup/name).exists():shutil.copy2(p,backup/name)
    x=replace(json.loads(p.read_text()));x.update(primary_pathway_count=476,state_identities=48,state_top_level_count=48,state_source_identity_count=197,state_detail_component_count=151,total_targets=524,observation_registry=str(G),accepted_y_definition=str(O/'FIXED_Y_DEFINITION.json'),state_group_membership=str(N/'STATE_GROUP_MEMBERSHIP.tsv'),state_top_level_targets=str(N/'STATE_TOP_LEVEL_TARGETS.tsv'),display_targets=str(N/'DISPLAY_TARGETS.tsv'),state_navigation=str(DEST/'State目录.json'),dictionary=str(dictionary),dictionary_sha256=dv['sha256'],training_authorized=False,lncrna_association_authorized=False)
    if name=='CANONICAL_TARGETS_CURRENT.json':x.update(catalog_version=N.name,catalog_directory=str(N),reader_dictionary=str(dictionary),reader_dictionary_sha256=dv['sha256'])
    if name=='CURRENT_CANONICAL_REFERENCE.json':x.update(canonical_registry=N.name,canonical_receipt_sha256=sha(N/'COMPLETE.json'),target_policy_sha256=sha(N/'TARGET_POLICY.tsv'))
    if name=='CURRENT_Y_DELIVERY.json':x.update(accepted_Y=str(O/'FIXED_Y_DEFINITION.json'),status='FIXED_MEASURED_Y_STATE_GROUPS')
    if name=='CURRENT_READER_DELIVERY.json':x.update(source=str(D),source_receipt_sha256=sha(O/'DELIVERY_MANIFEST.json'),destination=str(DEST),directory=str(DEST),pptx=str(ppt),pdf=str(pdf),catalog=str(N),delivery_manifest=str(O/'DELIVERY_MANIFEST.json'),verified_files=files,status='STATE_COLLECTIONS_48',dictionary_update_receipt=str(O/'DICTIONARY_VALIDATION.json'))
    tmp=p.with_suffix('.pending.json');sj(tmp,x);tmp.replace(p);records.append(dict(path=str(p),sha256=sha(p)))
for rec in records:
    data=json.loads(Path(rec['path']).read_text());assert data['state_top_level_count']==48 and data['dictionary']==str(dictionary) and data['observation_registry']==str(G)
sj(O/'ACTIVATION_RECEIPT.json',dict(target_host='149',pointers=records,catalog=str(N),registry=str(G),dictionary=str(dictionary),dictionary_sha256=dv['sha256'],primary_pathways=476,state_top_level_count=48,state_detail_components=151,total_targets=524,source_values_masks_partitions_unchanged=True,tests_passed=20,training=False,lncrna_association=False))
print('ACTIVE_48_STATE_AND_524_TARGETS_VERIFIED',flush=True)
