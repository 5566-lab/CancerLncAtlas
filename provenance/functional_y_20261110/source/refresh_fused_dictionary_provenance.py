from pathlib import Path
import sys,json,socket,hashlib,zipfile,shutil,subprocess
from xml.etree import ElementTree as ET
R=Path(__file__).resolve().parent;assert socket.gethostname()=='149'
F=R/'functional_relation_y_fused_r1';D=F/'dictionary_data';O=F/'reports'
py='/dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python'
subprocess.run([py,str(R/'export_fused_y_dictionary.py'),'--metadata-only'],check=True)
import pandas as pd
frame=pd.read_csv(D/'SCORE_DEFINITIONS.tsv',sep='\t').fillna('')
for c in ['definition_database','source_resource','score_matrix']:assert frame[c].astype(str).str.len().gt(0).all()
receipt=json.loads((O/'DICTIONARY_V5_RECEIPT.json').read_text());path=Path(receipt['file']);oldsha=hashlib.sha256(path.read_bytes()).hexdigest();assert oldsha==receipt['sha256']
ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main';tag=lambda s:'{'+ns+'}'+s
ET.register_namespace('',ns)
def colname(n):
 out=''
 while n:n,k=divmod(n-1,26);out=chr(65+k)+out
 return out
with zipfile.ZipFile(path) as zin:
 workbook=ET.fromstring(zin.read('xl/workbook.xml'));rels=ET.fromstring(zin.read('xl/_rels/workbook.xml.rels'));rid=next(x.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'] for x in workbook.find(tag('sheets')) if x.attrib['name']=='Score_definitions');target=next(x.attrib['Target'] for x in rels if x.attrib['Id']==rid);name=target.lstrip('/') if target.startswith('/') else 'xl/'+target
 root=ET.fromstring(zin.read(name));data=root.find(tag('sheetData'));headerstyle=data[0][0].attrib.get('s');bodystyle=data[1][0].attrib.get('s');data.clear()
 def row(i,values,style):
  el=ET.SubElement(data,tag('row'),{'r':str(i)})
  for j,v in enumerate(values,1):
   a={'r':colname(j)+str(i),'t':'inlineStr'}
   if style is not None:a['s']=style
   cell=ET.SubElement(el,tag('c'),a);inline=ET.SubElement(cell,tag('is'));ET.SubElement(inline,tag('t')).text=str(v)
 row(1,list(frame),headerstyle)
 for i,values in enumerate(frame.itertuples(index=False,name=None),2):row(i,values,bodystyle)
 end=colname(len(frame.columns))+str(len(frame)+1)
 dim=root.find(tag('dimension'))
 if dim is not None:dim.set('ref','A1:'+end)
 filt=root.find(tag('autoFilter'))
 if filt is not None:filt.set('ref','A1:'+end)
 cols=root.find(tag('cols'))
 if cols is not None:
  cols.clear()
  for j,c in enumerate(frame.columns,1):ET.SubElement(cols,tag('col'),{'width':'48' if any(x in c.lower() for x in ['method','reference','source','database','matrix']) else '25','customWidth':'1','min':str(j),'max':str(j)})
 updated=ET.tostring(root,encoding='utf-8',xml_declaration=True);temp=path.with_name(path.stem+'.updating.xlsx')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
  for item in zin.infolist():zout.writestr(item,updated if item.filename==name else zin.read(item.filename))
archive=R/'project_log/reader_history'/('dictionary_'+oldsha+'.xlsx');archive.parent.mkdir(exist_ok=True)
if not archive.exists():shutil.copy2(path,archive)
temp.replace(path);receipt['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(O/'DICTIONARY_V5_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print('READOUT_PROVENANCE_COMPLETE',len(frame),receipt['sha256'],flush=True)
subprocess.run([py,str(R/'validate_fused_y_dictionary.py')],check=True)
subprocess.run([py,str(R/'build_fused_y_presentation.py')],check=True)
