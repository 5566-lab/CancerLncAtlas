import concurrent.futures,json,os,socket,subprocess,time
from pathlib import Path
import hashlib,shutil
assert socket.gethostname().split('.')[0]=='149'
import argparse
parser=argparse.ArgumentParser(description='Score verified raw pathway assays on 149; descriptive outputs are forbidden as prepared labels.')
parser.add_argument('--raw-inputs',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--r-libraries',nargs='+',required=True);parser.add_argument('--workers',type=int,default=3)
args=parser.parse_args()
if not 1<=args.workers<=6:parser.error('workers must be between 1 and 6')
w=Path(__file__).resolve().parents[1]
raw=Path(args.raw_inputs).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise RuntimeError('Use a new output directory')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
snapshot=out/'immutable_scripts';snapshot.mkdir()
for filename in ('pathway_fusion_core.R','pathway_fusion_run.R'):
 source=w/'scripts'/filename;target=snapshot/filename
 expected=sha(source);shutil.copyfile(source,target)
 if sha(target)!=expected:raise RuntimeError('Scoring source changed during snapshot')
 target.chmod(0o444)
core=snapshot/'pathway_fusion_core.R';runner=snapshot/'pathway_fusion_run.R'
meta=json.loads((raw/'COMPLETE.json').read_text());assert meta['target_host']=='149' and meta['status']=='COMPLETE'
assert meta.get('format')=='CLA_FIVE_ASSAY_RAW_INPUTS_V1' and set(meta.get('assays',[]))=={'RNA','CNV','METHYLATION','PROTEIN','PHOSPHORYLATION'},'FIVE_ASSAY_INPUT_REQUIRED'
(out/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','scope':'DESCRIPTIVE_COHORT_ONLY','cancer_type':meta['cancer_type'],'input_complete_sha256':sha(raw/'COMPLETE.json')}))
def one(part):
 cancer=part['cancer_id'];dest=out/cancer;dest.mkdir(exist_ok=True)
 try:
  assert sha(part['path'])==part['sha256']
  m=json.loads(Path(part['path']).read_text());assert m['catalog_sha256']==sha(raw/'CATALOG.json')
  for b in m['blocks'].values():assert sha(raw/cancer/b['file'])==b['sha256']
  request={'target_host':'149','scope':'DESCRIPTIVE_COHORT_ONLY','cancer_id':cancer,'input_dir':str(raw/cancer),'catalog':str(raw/'CATALOG.json'),
   'fit_patients':m['patient_ids'],'output_dir':str(dest),'nf':10,
   'r_libraries':args.r_libraries,
   'core_script':str(core),'core_sha256':sha(core),
   'runner_sha256':sha(runner),'input_sha256':part['sha256']}
  req=dest/'REQUEST.json';req.write_text(json.dumps(request,indent=2))
  env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
  with (dest/'run.log').open('w') as log:
   result=subprocess.run(['Rscript',str(runner),str(req)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
  if result.returncode:raise RuntimeError('R_EXIT_'+str(result.returncode))
  receipt=json.loads((dest/'SCORES_RECEIPT.json').read_text())
  record={k:receipt.get(k) for k in ('cancer_id','status','mogsa_status','mogsa_available_patients','final_measured_pathways','parents_supported','parents_requested','parent_status_counts','mogsa_explained_variance')}
  record['output_dir']=str(dest)
  record['files']={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
  (dest/'COMPLETE.json').write_text(json.dumps(record,indent=2))
  print('COMPLETE',cancer,record['mogsa_available_patients'],record['parents_supported'],flush=True);return record
 except Exception as e:
  print('FAILED',cancer,str(e),flush=True);return {'cancer_id':cancer,'status':'FAILED','error':str(e)}
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
 for record in pool.map(one,meta['partitions']):
  rows.append(record);(out/'PROGRESS.json').write_text(json.dumps({'target_host':'149','done':len(rows),'total':len(meta['partitions']),'failures':sum(r['status']!='COMPLETE' for r in rows)}))
(out/'COMPLETE.json').write_text(json.dumps({'target_host':'149','status':'COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'FAILED','cancer_type':meta['cancer_type'],'partitions':rows},indent=2))

if any(r['status']!='COMPLETE' for r in rows):raise SystemExit(1)
