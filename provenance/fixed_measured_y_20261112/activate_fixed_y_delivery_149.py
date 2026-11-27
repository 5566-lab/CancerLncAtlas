from pathlib import Path
import socket, sys, json, shutil
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
REPO=ROOT/'source_integration/repo';BASE=WORK/'measured_y_definition_r1';REVIEW=BASE/'final_definition_review_r1'
REPORT=WORK/'reader_delivery_fixed_y_20261112';DEST=ROOT/'protein_similarity_integration_20261111_r1/deliverables'
sys.path.insert(0,str(REPO))
from cla_engine.common import sha
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
save(REVIEW/'ACTIVATION_PREFLIGHT.json',dict(target_host='149',operation='align active Y pointers and reader delivery',training=False,lncrna_association=False))
manifest=json.loads((REPORT/'DELIVERY_MANIFEST.json').read_text());definition=REVIEW/'FIXED_Y_DEFINITION.json';registry=BASE/'observation_registry_r2'
assert json.loads(definition.read_text())['registry_receipt_sha256']==sha(registry/'COMPLETE.json')
for record in manifest['files']:
    source=Path(record['path']);assert sha(source)==record['sha256']
    target=DEST/record['name'];assert target.resolve().is_relative_to(DEST.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=target.resolve():shutil.copyfile(source,target)
    assert sha(target)==record['sha256']
obsolete=['CancerLncAtlas_KEGG癌症主干与关系Y_v9.pptx','CancerLncAtlas_KEGG癌症主干与关系Y_v9.pdf']
for stem in ['01_癌症证据与主干范围','02_机制类型与精确颗粒度','03_跨库整合规则','04_错配修复实际跨库归并','05_相似组成与不同机制','06_分类与模型任务','07_关系Y与患者分离','08_监督覆盖与线性对照']:
    obsolete.extend('figures/'+stem+'.'+ext for ext in ['png','pdf'])
removed=[]
for name in obsolete:
    p=DEST/name;assert p.resolve().is_relative_to(DEST.resolve())
    if p.is_file():p.unlink();removed.append(name)
records=[]
for name in ['CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json','CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:
    p=ROOT/name;x=json.loads(p.read_text());x.update(observation_registry=str(registry),Y_representation='PATHWAY_GROUPED_PATIENT_OBSERVATIONS',accepted_y_definition=str(definition),Y_definition_complete=True)
    if name=='CURRENT_Y_DELIVERY.json':x.update(accepted_Y=str(definition),status='FIXED_MEASURED_Y_DEFINITION',open_items=[])
    if name=='CURRENT_READER_DELIVERY.json':
        ppt=next(DEST.glob('*多维Y*.pptx'));pdf=next(DEST.glob('*多维Y*.pdf'))
        x.update(source=str(REPORT),source_receipt_sha256=sha(REPORT/'DELIVERY_MANIFEST.json'),destination=str(DEST),directory=str(DEST),pptx=str(ppt),pdf=str(pdf),retired_obsolete_files=removed,code_changed=True)
        x.pop('older_presentation_Y_interpretation_superseded',None);x.pop('model_code_changed',None)
    save(p,x);records.append(dict(path=str(p),sha256=sha(p)))
for record in records:
    x=json.loads(Path(record['path']).read_text());assert x['Y_definition_complete'] and x['accepted_y_definition']==str(definition) and x['observation_registry']==str(registry)
save(REVIEW/'ACTIVE_DELIVERY_VERIFIED.json',dict(target_host='149',definition=str(definition),definition_sha256=sha(definition),pointers=records,reader_files=len(manifest['files']),retired_obsolete_files=removed,training=False,lncrna_association=False))
print('ACTIVE_Y_AND_READER_POINTERS_VERIFIED')
