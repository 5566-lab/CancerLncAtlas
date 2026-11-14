from pathlib import Path
import json,hashlib,socket,shutil
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'exact_member_merge_20261114_r1';N=R/'canonical_targets_r10_exact_members_20261114_r1';G=O/'observation_registry_r3';D=O/'deliverables';DEST=R/'protein_similarity_integration_20261111_r1/deliverables'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
val=json.loads((O/'VALIDATION.json').read_text());dv=json.loads((O/'DICTIONARY_VALIDATION.json').read_text());definition=json.loads((O/'FIXED_Y_DEFINITION.json').read_text());assert val['primary_after']==476 and dv['primary_pathways']==476
assert definition['registry_receipt_sha256']==sha(G/'COMPLETE.json')
for folder in [N]+[G/c for c in ['BRCA','LUAD','COAD']]:
    for name,h in json.loads((folder/'COMPLETE.json').read_text())['files'].items():assert sha(folder/name)==h,(folder,name)
assert sha(dv['path'])==dv['sha256']
files={p.name:sha(p) for p in D.iterdir() if p.is_file()};assert any(k.endswith('.pdf') for k in files)
sj(O/'DELIVERY_MANIFEST.json',dict(target_host='149',files=files,dictionary_sha256=dv['sha256'],primary_pathways=476,state_identities=197,total_targets=673))
for name,h in files.items():shutil.copy2(D/name,DEST/name);assert sha(DEST/name)==h
dictionary=DEST/'Functional_pathway_State_dictionary_v7.xlsx';ppt=DEST/'CancerLncAtlas_通路字典与J1合并_20261114.pptx';pdf=ppt.with_suffix('.pdf')
backup=O/'previous_current_pointers';backup.mkdir(exist_ok=True)
replacements={str(R/'canonical_targets_r9_cancer_evidence_20261111_r6'):str(N),str(W/'measured_y_definition_r1/observation_registry_r2'):str(G),str(W/'measured_y_definition_r1/final_definition_review_r1/FIXED_Y_DEFINITION.json'):str(O/'FIXED_Y_DEFINITION.json')}
def replace(v):
    if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
    if isinstance(v,list):return [replace(x) for x in v]
    if isinstance(v,str):
        for old,new in replacements.items():v=v.replace(old,new)
    return v
pointers=[]
for name in ['CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json','CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:
    p=R/name
    if not (backup/name).exists():shutil.copy2(p,backup/name)
    x=replace(json.loads(p.read_text()));x.update(observation_registry=str(G),accepted_y_definition=str(O/'FIXED_Y_DEFINITION.json'),Y_representation='PATHWAY_GROUPED_PATIENT_OBSERVATIONS',Y_definition_complete=True,primary_pathway_count=476,state_identities=197,total_targets=673,exact_member_groups=str(N/'EXACT_MEMBER_GROUP_MEMBERS.tsv'),group_definition=str(O/'GROUP_DEFINITION.json'),training_authorized=False,lncrna_association_authorized=False)
    if name=='CANONICAL_TARGETS_CURRENT.json':
        x.update(catalog_version='r10_exact_member_groups_20261114_r1',catalog_directory=str(N),reader_dictionary=str(dictionary),reader_dictionary_sha256=dv['sha256'],grouped_backbone=str(N/'GROUPED_BACKBONE.tsv'),source_backbone=x.pop('backbone',None))
    if name=='CURRENT_CANONICAL_REFERENCE.json':x.update(canonical_registry=N.name,canonical_receipt_sha256=sha(N/'COMPLETE.json'),target_policy_sha256=sha(N/'TARGET_POLICY.tsv'))
    if name=='CURRENT_Y_DELIVERY.json':x.update(accepted_Y=str(O/'FIXED_Y_DEFINITION.json'),status='FIXED_MEASURED_Y_EXACT_MEMBER_GROUPS')
    if name=='CURRENT_READER_DELIVERY.json':
        x.update(source=str(D),source_receipt_sha256=sha(O/'DELIVERY_MANIFEST.json'),destination=str(DEST),directory=str(DEST),pptx=str(ppt),pdf=str(pdf),dictionary=str(dictionary),dictionary_sha256=dv['sha256'],catalog=str(N),delivery_manifest=str(O/'DELIVERY_MANIFEST.json'),verified_files={k:v for k,v in files.items() if k.endswith(('.xlsx','.pptx','.pdf'))},status='EXACT_MEMBER_GROUPED_MEASURED_Y',dictionary_update_receipt=str(O/'DICTIONARY_VALIDATION.json'))
    temp=p.with_suffix('.pending.json');sj(temp,x);temp.replace(p);pointers.append(dict(path=str(p),sha256=sha(p)))
for rec in pointers:
    x=json.loads(Path(rec['path']).read_text());assert x['primary_pathway_count']==476 and x['observation_registry']==str(G)
sj(O/'ACTIVATION_RECEIPT.json',dict(target_host='149',pointers=pointers,dictionary=str(dictionary),dictionary_sha256=dv['sha256'],catalog=str(N),registry=str(G),definition=str(O/'FIXED_Y_DEFINITION.json'),primary_pathways=476,state_identities=197,total_targets=673,tests=16,source_matrices_unchanged=True,source_events_preserved=True,training=False,lncrna_association=False))
print('ACTIVE_MERGED_Y_AND_DICTIONARY_VERIFIED',flush=True)
