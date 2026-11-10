from pathlib import Path
import json,hashlib,shutil,socket,time
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';O=R/'shared_relation_demo_r1';P=Path('/dsk2/user/dengsc/CancerLncAtlas/processed/functional_y_20261110_r2');assert socket.gethostname()=='149';assert json.loads((F/'VALIDATION_COMPLETE.json').read_text())['status']=='PASS';P.mkdir(exist_ok=False)
(P/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'versioned functional Y publication','source':str(F)},indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={};start=time.monotonic()
def copy(src,rel,expected=None):
 dst=P/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);h=sha(dst);assert h==(sha(src) if expected is None else expected),str(src);files[str(rel)]=h
copy(F/'Y_DEFINITION.json',Path('Y/Y_DEFINITION.json'));copy(F/'VALIDATION_COMPLETE.json',Path('VALIDATION_COMPLETE.json'))
for c in ['BRCA','LUAD','COAD']:
 for ctx in ['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']:
  base=F/c/ctx;rec=json.loads((base/'COMPLETE.json').read_text());copy(base/'COMPLETE.json',Path('Y')/c/ctx/'COMPLETE.json')
  for name,h in rec['files'].items():copy(base/name,Path('Y')/c/ctx/name,h)
  for src in (base/'null_references').iterdir():
   if src.is_file():copy(src,Path('Y')/c/ctx/'null_references'/src.name)
  print('PUBLISHED_Y',c,ctx,flush=True)
# The patient-score matrices retain their platform and fold-specific references.
index=[]
for c in ['BRCA','LUAD','COAD']:
 for ctx in ['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']:
  for ds,layers in [('TCGA',['RNA','STATE','PROTEIN','PHOSPHO']),('CPTAC',['RNA','STATE','PROTEIN','PHOSPHO','METHYLATION'])]:
   srcds='CPTAC_DEDUPLICATED' if ds=='CPTAC' and c=='BRCA' and ctx=='sealed_test' else ds
   for layer in layers:
    base=O/srcds/c/ctx/layer;rec=json.loads((base/'COMPLETE.json').read_text());rel=Path('S')/ds/c/ctx/layer
    for name in ['PATIENT_S.parquet','TARGETS.tsv','ADJUSTMENT.json']:copy(base/name,rel/name,rec['files'][name])
    index.append({'dataset':ds,'cancer_id':c,'context':ctx,'layer':layer,'score_matrix':str(rel/'PATIENT_S.parquet'),'readout_axis':str(rel/'TARGETS.tsv'),'adjustment':str(rel/'ADJUSTMENT.json'),'source_association_directory':str(base),'source_complete_sha256':sha(base/'COMPLETE.json')})
import csv
with (P/'PATIENT_SCORE_INDEX.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(index[0]),delimiter='\t');w.writeheader();w.writerows(index)
files['PATIENT_SCORE_INDEX.tsv']=sha(P/'PATIENT_SCORE_INDEX.tsv')
(P/'DATA_COMPLETE.json').write_text(json.dumps({'target_host':'149','status':'COMPLETE','contexts':15,'patient_score_entries':len(index),'source_y_definition_sha256':sha(F/'Y_DEFINITION.json'),'source_validation_sha256':sha(F/'VALIDATION_COMPLETE.json'),'seconds':time.monotonic()-start,'files':files},indent=2));print('DATA_PUBLICATION_COMPLETE',len(files),flush=True)
