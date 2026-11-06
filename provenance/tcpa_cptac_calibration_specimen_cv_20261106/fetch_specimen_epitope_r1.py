import pathlib,socket,urllib.request,urllib.parse,json,time,hashlib,concurrent.futures
assert socket.gethostname()=="149"
P=pathlib.Path("/dell_2/DSC/CancerLncAtlas/runtime/tcpa_cptac_calibration_20261106_r1")
O=P/"specimen_epitope_sources_r1";O.mkdir(exist_ok=False)
(O/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"FETCH_VERSION_BOUND_PDC_AND_CST_EVIDENCE","time":time.time()}))
catalog=json.loads((P/"pdc_sources_r1/study_catalog.json").read_text())["data"]["studyCatalog"]
ids=["PDC000173","PDC000174","PDC000111","PDC000113","PDC000114","PDC000115"]
tasks=[]
for pid in ids:
 entry=next(r for r in catalog if r["pdc_study_id"]==pid);v=next(x for x in entry["versions"] if x["is_latest_version"]=="yes")
 q='{biospecimenPerStudy(study_id:"'+v["study_id"]+'"){aliquot_id sample_id case_id aliquot_submitter_id sample_submitter_id case_submitter_id aliquot_status case_status sample_status project_name sample_type disease_type primary_site pool taxon}}'
 tasks.append({"name":pid+"_v"+v["study_version"]+"_biospecimen.json","url":"https://pdc.cancer.gov/graphql?"+urllib.parse.urlencode({"query":q}),"study":dict(pdc_study_id=pid,**v)})
for code in [9456,9455,13050,2696,3891,3377,9205,3061,2101,2105,3617,2900]:
 tasks.append({"name":"CST_"+str(code)+".pdf","url":"https://media.cellsignal.com/pdf/"+str(code)+".pdf","catalog_number":str(code)})
def fetch(t):
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(urllib.request.Request(t["url"],headers={"User-Agent":"Mozilla/5.0"}),timeout=25) as r:data=r.read()
  if t["name"].endswith(".json"):
   v=json.loads(data);assert not v.get("errors"),v.get("errors")
  else:assert data.startswith(b"%PDF"),data[:80]
  (O/t["name"]).write_bytes(data)
  rec=dict(t,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status="DOWNLOADED")
 except Exception as e:rec=dict(t,status="FAILED",error=type(e).__name__+": "+str(e))
 print(json.dumps(rec),flush=True);return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,tasks))
(O/"FETCH_RECEIPT.json").write_text(json.dumps({"target_host":"149","records":records},indent=2))
print("FETCH_FINISHED",flush=True)
