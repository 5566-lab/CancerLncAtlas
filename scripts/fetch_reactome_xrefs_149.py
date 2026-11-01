import concurrent.futures,hashlib,json,os,re,socket,time,urllib.request,urllib.parse
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
import argparse
parser=argparse.ArgumentParser(description='Verify MSigDB Reactome stable IDs directly from public gene-set metadata on 149')
parser.add_argument('--gmt',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args()
w=Path(args.output).resolve();d=w/'method_sources/msigdb_reactome';d.mkdir(parents=True,exist_ok=True)
gmt=Path(args.gmt)
sets={p[0]:set(p[2:]) for line in gmt.read_text().splitlines() if (p:=line.split('\t'))}
def one(name):
 p=d/(name+'.json');url='https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp?'+urllib.parse.urlencode({'geneSetName':name,'fileType':'json'})
 try:
  if not p.exists():
   request=urllib.request.Request(url,headers={'User-Agent':'CancerLncAtlas research reference verification'})
   with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=45) as h:data=h.read()
   v=json.loads(data)[name]
   assert re.fullmatch(r'R-HSA-\d+',v.get('exactSource',''))
   tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(p)
  v=json.loads(p.read_text())[name]
  return {'pathway_id':'MSIGDB:REACTOME:'+name,'reactome_id':v['exactSource'],'source_url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'original_gmt_members_unchanged':True,'metadata_members_match_gmt':set(v['geneSymbols'])==sets[name],'status':'VERIFIED_DIRECT_XREF'}
 except Exception as e:return {'pathway_id':'MSIGDB:REACTOME:'+name,'status':'UNAVAILABLE','error':str(e)}
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for row in pool.map(one,sets):
  rows.append(row)
  if len(rows)%100==0:
   (w/'XREF_PROGRESS.json').write_text(json.dumps({'target_host':'149','pid':os.getpid(),'completed':len(rows),'total':len(sets),'failures':sum(r['status']!='VERIFIED_DIRECT_XREF' for r in rows)}));print('XREF',len(rows),flush=True)
(w/'REACTOME_XREFS.json').write_text(json.dumps({'target_host':'149','gmt_sha256':hashlib.sha256(gmt.read_bytes()).hexdigest(),'rows':rows},indent=2))
print('XREF_COMPLETE',len(rows),'FAILED',sum(r['status']!='VERIFIED_DIRECT_XREF' for r in rows),flush=True)
