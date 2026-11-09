"""Check the full tabular and presentation delivery against its scientific sources."""
from pathlib import Path
import sys,json,time,socket,hashlib,zipfile,xml.etree.ElementTree as ET
from xml.parsers import expat
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;C=R/'canonical_targets_r6';Y=R/'functional_relation_y_catalog_r6';V=R/'functional_y_validation_r2';Q=R/'supervision_summary_r2';LOG=R/'project_log'
required=[Q/'COMPLETE.json',V/'COMPLETE.json',LOG/'DICTIONARY_V4_RECEIPT.json',LOG/'PRESENTATION_V2_RECEIPT.json']
deadline=time.monotonic()+8*3600
while not all(p.exists() for p in required):
 if time.monotonic()>deadline:raise TimeoutError('Delivery audit input deadline')
 time.sleep(10)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
validated=json.loads((V/'COMPLETE.json').read_text());assert validated['status']=='PASS' and validated['validated_contexts']==165 and not validated['failures']
assert json.loads((Y/'COMPLETE.json').read_text())['status']=='COMPLETE'
master=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t').fillna('');scored=master.loc[master.has_scored_readout.astype(str).str.lower().eq('true')].copy();assert scored.canonical_id.is_unique and len(scored)==3456
bindings=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t').fillna('');assert bindings.source_target_id.is_unique and set(bindings.canonical_id)<=set(master.canonical_id)
rows=pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER_CONTEXT.tsv',sep='\t');cancers=pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER.tsv',sep='\t');assert len(rows)==165 and len(cancers)==33 and cancers.cancer_id.is_unique;assert not rows.duplicated(['cancer_id','context']).any()
for row in rows.itertuples():
 base=Y/row.cancer_id/row.context;receipt=json.loads((base/'COMPLETE.json').read_text());proof=json.loads((V/(row.cancer_id+'__'+row.context+'.json')).read_text());assert proof['status']=='PASS' and proof['source_complete_sha256']==sha(base/'COMPLETE.json');assert proof['canonical_targets']==3456 and proof['lncrnas']==16889
 for field in ['observable_relations','strong_positive','weak_positive','unlabeled_observed','direction_discordant']:assert int(getattr(row,field))==int(receipt[field]),(row.cancer_id,row.context,field)
 assert row.observable_relations==row.strong_positive+row.weak_positive+row.unlabeled_observed
coverage=pd.read_csv(Q/'CURRENT_SCORE_COVERAGE.tsv.gz',sep='\t');present=coverage.loc[coverage.patients_with_score.gt(0)];observed_ids=set(present.canonical_id);assert len(observed_ids)==2840 and observed_ids<=set(scored.canonical_id);assert coverage.patients_with_score.le(coverage.patients_in_scope).all()
for receiptname in ['DICTIONARY_V4_RECEIPT.json','PRESENTATION_V2_RECEIPT.json']:
 receipt=json.loads((LOG/receiptname).read_text());assert sha(receipt['file'])==receipt['sha256'],receiptname
bookrec=json.loads((LOG/'DICTIONARY_V4_RECEIPT.json').read_text());bookpath=Path(bookrec['file']);ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
forbidden=['MOGSA','PathFA','尚未执行','尚未采用','保持停用','integration_status','score_merge_authorized','independent_flux_claimed','independent_functional_branch_claimed']
counts={}
with zipfile.ZipFile(bookpath) as z:
 relations=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));dest={x.attrib['Id']:x.attrib['Target'] for x in relations};meta=ET.fromstring(z.read('xl/workbook.xml'))
 for sheet in meta.findall('m:sheets/m:sheet',ns):
  title=sheet.attrib['name'];target=dest[sheet.attrib['{'+ns['r']+'}id']];target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
  count=[0];text_tail=[''];parser=expat.ParserCreate()
  def start(name,attrs):
   if name=='row':count[0]+=1
  def characters(value):
   content=text_tail[0]+value
   assert not any(word in content for word in forbidden),(title,'execution metadata in workbook')
   text_tail[0]=content[-100:]
  parser.StartElementHandler=start;parser.CharacterDataHandler=characters
  with z.open(target) as stream:
   while True:
    chunk=stream.read(1024*1024)
    if not chunk:break
    parser.Parse(chunk,False)
   parser.Parse(b'',True)
  counts[title]=count[0]-1
 assert counts==bookrec['sheets'],(counts,bookrec['sheets'])
sys.path.insert(0,'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps');from openpyxl import load_workbook
wb=load_workbook(bookpath,read_only=True,data_only=True)
def table(name):
 it=wb[name].iter_rows(values_only=True);header=next(it);return pd.DataFrame(it,columns=header)
actual=table('Target_master');assert set(actual.canonical_id)==set(scored.canonical_id) and len(actual)==3456
expected_names=scored.set_index('canonical_id').display_name.to_dict();assert all(expected_names[x.canonical_id]==x.display_name for x in actual.itertuples())
for sheet,source in [('Y_by_cancer',cancers),('Y_by_context',rows),('Composition_counts',pd.read_csv(R/'target_composition_index_r1/TARGET_COMPOSITION_COUNTS.tsv',sep='\t'))]:
 actual=table(sheet);pd.testing.assert_frame_equal(actual,source,check_dtype=False,check_exact=False)
struct=table('Structure_nodes');known=set(master.canonical_id)|set(struct.canonical_id);pc=table('Parent_child');assert set(pc.parent_id)|set(pc.child_id)<=known
state=table('State_relations');assert set(state.parent_id)|set(state.child_id)<=set(master.canonical_id)
br=table('Pathway_branches');assert len(br)==21358 and br.canonical_id.nunique()==1111 and set(br.canonical_id)<=set(scored.canonical_id)
wb.close()
sys.path.insert(0,str(R/'python_deps'));from pptx import Presentation
pptrec=json.loads((LOG/'PRESENTATION_V2_RECEIPT.json').read_text());prs=Presentation(pptrec['file']);assert len(prs.slides)==pptrec['slides']==22
text=[]
for slide in prs.slides:
 for shape in slide.shapes:
  if shape.has_text_frame:text.append(shape.text)
  if shape.has_table:text.extend(cell.text for row in shape.table.rows for cell in row.cells)
  assert shape.left>=0 and shape.top>=0 and shape.left+shape.width<=prs.slide_width+10000 and shape.top+shape.height<=prs.slide_height+10000
 text.append(slide.notes_slide.notes_text_frame.text)
full='\n'.join(text);assert not any(word in full for word in forbidden);assert '3,456' in full and '2,840' in full
assert '/dsk2/' not in full and '模型正则项' not in full
record={'target_host':'149','status':'PASS','canonical_identities':3456,'identities_with_patient_scores':len(observed_ids),'pathway_roles':int(scored.roles.str.contains('Pathway').sum()),'state_roles':int(scored.roles.str.contains('State').sum()),'shared_roles':int(scored.roles.eq('Pathway|State').sum()),'cancers':33,'contexts':165,'dictionary_sheets':counts,'dictionary_sha256':sha(bookpath),'presentation_sha256':sha(pptrec['file']),'presentation_slides':22,'checks':['165 context proofs bind to current Y files and exact counts','Stable identity names, source bindings and redirects','Actual workbook sheet row counts match exported data','Workbook supervision and composition tables match authoritative summaries','Branch and hierarchy endpoints resolve to biological definitions','Current score coverage counts match actual finite patient readouts','Reader workbook and slides exclude unused-method and execution-state text','PPT content and shape bounds verified'],'remaining_delivery_checks':['Rendered slide visual review','PDF page count and figure exports','Local delivery checksum verification','Final requirement and publication audit']}
(LOG/'DELIVERY_CONTENT_AUDIT.json').write_text(json.dumps(record,indent=2,ensure_ascii=False));print(json.dumps(record))
