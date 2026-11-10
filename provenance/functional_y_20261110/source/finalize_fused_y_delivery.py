from pathlib import Path
import json,hashlib,shutil,socket,datetime
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';O=F/'reports';D=F/'dictionary_data';P=Path('/dsk2/user/dengsc/CancerLncAtlas/processed/functional_y_20261110_r2');assert socket.gethostname()=='149'
(P/'FINALIZATION_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'Bind final reader deliverables to completed numerical Y release'}))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
dc=json.loads((P/'DATA_COMPLETE.json').read_text());assert dc['status']=='COMPLETE' and dc['contexts']==15 and dc['patient_score_entries']==135
assert all((P/n).is_file() and (P/n).stat().st_size>0 for n in dc['files'])
dictrec=json.loads((O/'DICTIONARY_V5_RECEIPT.json').read_text());dv=json.loads((O/'DICTIONARY_VALIDATION.json').read_text());pr=json.loads((O/'PRESENTATION_V6_RECEIPT.json').read_text());assert dv['status']=='PASS' and dictrec['sha256']==dv['dictionary_sha256']==pr['dictionary_sha256']==sha(Path(dictrec['file']))
assert sha(Path(pr['pptx']))==pr['pptx_sha256'] and sha(Path(pr['pdf']))==pr['pdf_sha256'] and sha(F/'VALIDATION_COMPLETE.json')==pr['validation_sha256']==dc['source_validation_sha256']
files={}
def copy(src,rel):
 dst=P/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists() or sha(dst)!=sha(src):shutil.copy2(src,dst)
 assert sha(dst)==sha(src);files[str(rel)]=sha(dst)
for src in D.iterdir():
 if src.is_file() and src.name not in ['LAUNCH_PREFLIGHT.json']:copy(src,Path('dictionary_data')/src.name)
for sub in ['canonical_targets_r6','target_composition_index_r1','reader_dictionary_r4','catalog_inventory','reaction_scope_relations_r1','semantic_relations_r1','structure_reactome98']:
 for src in (R/sub).iterdir():
  if src.is_file() and (src.name.endswith('.tsv') or src.name.endswith('.tsv.gz')):copy(src,Path('definitions')/sub/src.name)
assets=[]
for src in O.iterdir():
 if src.is_file() and src.suffix.lower() in ['.png','.pdf','.pptx','.xlsx']:
  copy(src,Path('reports')/src.name);assets.append({'filename':src.name,'bytes':src.stat().st_size,'sha256':sha(src)})
for name in ['DICTIONARY_V5_RECEIPT.json','DICTIONARY_VALIDATION.json','PRESENTATION_V6_RECEIPT.json','SCIENTIFIC_FIGURES.json','FIGURE_RECEIPT.json']:copy(O/name,Path('report_receipts')/name)
server_reader=Path('/dsk2/user/dengsc/CancerLncAtlas/results/model/图片/Functional_Y_20261109_r1/联合关系Y_20261110');server_reader.mkdir(parents=True,exist_ok=True)
for a in assets:
 src=O/a['filename'];dst=server_reader/a['filename'];shutil.copy2(src,dst);assert sha(dst)==a['sha256']
manifest={'target_host':'149','source_reports':str(O),'server_reader_directory':str(server_reader),'assets':assets,'figure_pairs':sum(x['filename'].endswith('.png') for x in assets),'presentation_slides':pr['slides'],'dictionary_sheets':len(dictrec['sheets'])}
(O/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));copy(O/'DELIVERY_MANIFEST.json',Path('report_receipts/DELIVERY_MANIFEST.json'))
complete={'target_host':'149','status':'COMPLETE','directory':str(P),'data_complete_sha256':sha(P/'DATA_COMPLETE.json'),'contexts':15,'patient_score_entries':135,'canonical_identities':3456,'lncrna_axis':16889,'reader_artifacts':len(assets),'reader_assets':manifest,'additional_files':files,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'COMPLETE.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2));(R/'project_log/JOINT_Y_RELEASE_FINAL.json').write_text(json.dumps({'target_host':'149','status':'COMPLETE','directory':str(P),'complete_sha256':sha(P/'COMPLETE.json'),'report_manifest_sha256':sha(O/'DELIVERY_MANIFEST.json'),'contexts':15,'patient_score_entries':135,'reader_assets':len(assets),'figure_pairs':manifest['figure_pairs'],'slides':pr['slides'],'dictionary_sheets':len(dictrec['sheets'])},indent=2));print('RELEASE_FINALIZED',len(files),len(assets),pr['slides'],flush=True)
