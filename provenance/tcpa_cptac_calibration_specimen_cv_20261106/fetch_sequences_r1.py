import pathlib,socket,json,urllib.request,urllib.parse,hashlib,concurrent.futures
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/tcpa_cptac_calibration_20261106_r1/sequence_sources_r1");P.mkdir(exist_ok=False)
(P/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"FETCH_EXACT_REFSEQ_AND_HUMAN_UNIPROT_REFERENCES"}))
tasks=[]
for a in ["Q13541","Q13315","P15056","O15530"]:
 tasks.append((a+".json","https://rest.uniprot.org/uniprotkb/"+a+".json"))
ids=["NP_004086.1","NP_000042.3","NP_004324.2","NP_001248745.1"]
tasks.append(("refseq.fasta","https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"+urllib.parse.urlencode(dict(db="protein",id=",".join(ids),rettype="fasta",retmode="text"))))
for n in [9452,2873,3062]:
 tasks.append(("CST_"+str(n)+".pdf","https://media.cellsignal.com/pdf/"+str(n)+".pdf"))
def fetch(t):
 name,url=t
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=25) as r:data=r.read()
  if name.endswith(".json"):assert "sequence" in json.loads(data)
  elif name.endswith(".fasta"):assert data.startswith(b">")
  else:assert data.startswith(b"%PDF")
  (P/name).write_bytes(data);rec=dict(name=name,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status="DOWNLOADED")
 except Exception as e:rec=dict(name=name,url=url,status="FAILED",error=str(e))
 print(json.dumps(rec),flush=True);return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,tasks))
(P/"FETCH_RECEIPT.json").write_text(json.dumps({"target_host":"149","records":records},indent=2))
