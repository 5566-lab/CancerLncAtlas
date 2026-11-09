from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import socket,urllib.request,urllib.parse,json,hashlib,time,re,threading,zipfile,io
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'sources/crossdatabase';O.mkdir(exist_ok=True);(O/'msigdb').mkdir(exist_ok=True)
(R/'CROSSDATABASE_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','action':'FETCH_EXACT_DEFINITION_METADATA_AND_CURATED_MAPPING','requests_per_second_max':2},indent=2))
lock=threading.Lock();last=[0.0]
def get(url):
 with lock:
  dt=.5-(time.monotonic()-last[0])
  if dt>0:time.sleep(dt)
  last[0]=time.monotonic()
 req=urllib.request.Request(url,headers={'User-Agent':'CancerLncAtlas reference integration/1.0'})
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=35) as h:return h.read(),dict(h.headers)
P=pd.read_csv(R/'catalog_inventory/PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').fillna('');cov=pd.read_csv(R/'catalog_inventory/PATHWAY_MECHANISM_COVERAGE.tsv',sep='\t').fillna('')
names=P.loc[P.source_database.ne('REACTOME')|P.source_target_id.isin(set(cov.loc[cov.structure_status.eq('UNRESOLVED_REACTOME_ID'),'source_target_id'])),'source_name'].tolist()
def one(name):
 url='https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp?'+urllib.parse.urlencode({'geneSetName':name,'fileType':'json'});p=O/'msigdb'/(name+'.json')
 try:
  if not p.exists():data,headers=get(url);v=json.loads(data)[name];p.write_bytes(data)
  data=p.read_bytes();v=json.loads(data)[name]
  return {'source_name':name,'exact_source':v.get('exactSource',''),'systematic_name':v.get('systematicName',''),'collection':v.get('collection',''),'members':v.get('geneSymbols',[]),'external_details':v.get('externalDetailsURL',[]),'url':url,'sha256':hashlib.sha256(data).hexdigest(),'path':str(p),'status':'VERIFIED_SOURCE_METADATA'}
 except Exception as e:
  # HTML retains archived aliases for several renamed gene sets.
  url2='https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/'+name+'.html'
  try:
   data,headers=get(url2);text=data.decode('utf-8');p=O/'msigdb'/(name+'.html');p.write_bytes(data)
   from html.parser import HTMLParser
   class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,s):self.parts.append(s.strip())
   parser=Text();parser.feed(text);plain=' '.join(parser.parts);m=re.search(r'Exact source\s+(R-HSA-\d+|N\d+)',plain)
   if not m:raise ValueError('Exact source field absent')
   return {'source_name':name,'exact_source':m.group(1),'url':url2,'sha256':hashlib.sha256(data).hexdigest(),'path':str(p),'status':'VERIFIED_EXACT_SOURCE_HTML','json_error':repr(e)}
  except Exception as e2:return {'source_name':name,'status':'SOURCE_UNRESOLVED','json_error':repr(e),'html_error':repr(e2)}
rows=[]
with ThreadPoolExecutor(max_workers=3) as ex:
 fs={ex.submit(one,n):n for n in names}
 for f in as_completed(fs):
  rows.append(f.result())
  if len(rows)%50==0:
   (R/'CROSSDATABASE_FETCH_PROGRESS.json').write_text(json.dumps({'target_host':'149','completed':len(rows),'total':len(names),'failures':sum(z['status']=='SOURCE_UNRESOLVED' for z in rows)},indent=2));print('METADATA',len(rows),len(names),flush=True)
(R/'CROSSDATABASE_DEFINITION_SOURCES.json').write_text(json.dumps({'target_host':'149','rows':rows,'created_utc':datetime.now(timezone.utc).isoformat()},indent=2))
url='https://codeload.github.com/ComPath/compath-resources/zip/refs/heads/master'
try:
 data,headers=get(url);z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None;p=O/'compath_resources_master.zip';p.write_bytes(data)
 entries=[]
 for item in z.infolist():
  if '/mappings/' in item.filename and not item.is_dir():
   q=O/('compath_'+Path(item.filename).name);q.write_bytes(z.read(item));entries.append({'name':item.filename,'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
 (R/'COMPATH_SOURCE_RECEIPT.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(data).hexdigest(),'headers':headers,'files':entries},indent=2));print('COMPATH_FILES',len(entries),flush=True)
except Exception as e:print('COMPATH_FETCH_FAILED',repr(e),flush=True)
print('DEFINITION_FETCH_COMPLETE',len(rows),'UNRESOLVED',sum(z['status']=='SOURCE_UNRESOLVED' for z in rows),flush=True)
