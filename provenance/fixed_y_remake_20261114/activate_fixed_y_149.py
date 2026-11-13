from pathlib import Path
import json,hashlib,socket,shutil
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'fixed_y_remake_20261114_r1';B=O/'Y_bundle';D=O/'deliverables';DEST=R/'protein_similarity_integration_20261111_r1/deliverables';N=R/'canonical_targets_r11_state_groups_20261114_r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sj(O/'ACTIVATION_PREFLIGHT.json',dict(target_host='149',operation='Activate portable fixed Y and rebuilt scientific presentation'))
pkg=json.loads((O/'PACKAGE_VALIDATION.json').read_text());v=json.loads((O/'VALIDATION.json').read_text());ppt=json.loads((O/'PPT_MANIFEST.json').read_text());render=json.loads((O/'RENDER_RECEIPT.json').read_text())
assert sha(B/'COMPLETE.json')==pkg['bundle_receipt_sha256'];assert sha(pkg['archive'])==pkg['archive_sha256'];assert ppt['slides']==render['rendered_pages']==17
for name,h in json.loads((B/'COMPLETE.json').read_text())['files'].items():assert sha(B/name)==h,name
final=json.loads((W/'state_groups_20261114_r1/FIXED_Y_DEFINITION.json').read_text());final.update(definition_version='CLA_FIXED_MEASURED_Y_524_MATERIALIZED_20261114_V1',fixed_y_bundle=str(B),fixed_y_bundle_receipt_sha256=sha(B/'COMPLETE.json'),fixed_y_definition=str(B/'DEFINITION.json'),fixed_y_target_axis=str(B/'TARGETS.tsv'),fixed_y_representation='Per-cohort patient-by-unique-observation matrices with explicit target membership and masks',materialized=True,patients=1910,validation=str(O/'VALIDATION.json'),source_observation_registry=final['registry']);sj(O/'FINAL_Y_DEFINITION.json',final)
files={p.name:sha(p) for p in D.iterdir() if p.is_file()};assert len(files)==5
manifest=dict(target_host='149',files=files,primary_pathways=476,state_top_level_count=48,total_targets=524,patients=1910,slides=17,y_archive_sha256=pkg['archive_sha256'],coverage_workbook='Y定义与实测覆盖_20261114.xlsx',dictionary_sha256=json.loads((R/'CANONICAL_TARGETS_CURRENT.json').read_text())['reader_dictionary_sha256'])
sj(O/'DELIVERY_MANIFEST.json',manifest)
for name,h in files.items():shutil.copy2(D/name,DEST/name);assert sha(DEST/name)==h
backup=O/'previous_current_pointers';backup.mkdir(exist_ok=True);records=[]
for name in ['CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json','CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:
    p=R/name
    if not (backup/name).exists():shutil.copy2(p,backup/name)
    x=json.loads(p.read_text());x.update(accepted_y_definition=str(O/'FINAL_Y_DEFINITION.json'),fixed_y_bundle=str(B),fixed_y_bundle_receipt_sha256=sha(B/'COMPLETE.json'),fixed_y_target_axis=str(B/'TARGETS.tsv'),primary_pathway_count=476,state_top_level_count=48,total_targets=524,Y_definition_complete=True,Y_materialized=True,training_authorized=False,lncrna_association_authorized=False)
    if name=='CURRENT_Y_DELIVERY.json':x.update(accepted_Y=str(O/'FINAL_Y_DEFINITION.json'),status='MATERIALIZED_FIXED_MEASURED_Y_524',materialization_validation=str(O/'VALIDATION.json'),archive=str(DEST/Path(pkg['archive']).name),archive_sha256=pkg['archive_sha256'],coverage_workbook=str(DEST/'Y定义与实测覆盖_20261114.xlsx'))
    if name=='CURRENT_READER_DELIVERY.json':x.update(source=str(D),source_receipt_sha256=sha(O/'DELIVERY_MANIFEST.json'),destination=str(DEST),directory=str(DEST),pptx=str(DEST/Path(ppt['path']).name),pdf=str(DEST/Path(ppt['path']).with_suffix('.pdf').name),delivery_manifest=str(O/'DELIVERY_MANIFEST.json'),verified_files=files,status='REBUILT_Y_AND_PRESENTATION',slides=17)
    tmp=p.with_suffix('.pending.json');sj(tmp,x);tmp.replace(p);records.append(dict(path=str(p),sha256=sha(p)))
sj(O/'ACTIVATION_RECEIPT.json',dict(target_host='149',pointers=records,primary_pathways=476,state_top_level_count=48,total_targets=524,patients=1910,slides=17,fixed_y_bundle=str(B),archive_sha256=pkg['archive_sha256'],validation=str(O/'VALIDATION.json'),training=False,lncrna_association=False))
print('FIXED_Y_AND_NEW_PRESENTATION_ACTIVATED',flush=True)
