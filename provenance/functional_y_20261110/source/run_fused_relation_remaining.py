from pathlib import Path
import socket,sys,json,subprocess,time,os
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';assert socket.gethostname()=='149'
pre=json.loads((F/'REMAINING_PREFLIGHT.json').read_text());started=time.monotonic();records=[]
def js(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def work(job):
 c,context=job;log=F/'logs'/(c+'__'+context+'.log');t=time.monotonic()
 with log.open('w') as out:
  p=subprocess.Popen([sys.executable,'-u',str(R/'materialize_fused_relation_y.py'),'--cancer',c,'--context',context],stdout=out,stderr=subprocess.STDOUT,cwd=R)
  js(F/'logs'/(c+'__'+context+'.PROCESS.json'),{'target_host':'149','pid':p.pid,'cancer_id':c,'context':context})
  try:rc=p.wait(timeout=7200)
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);rc=124
 return {'cancer_id':c,'context':context,'returncode':rc,'seconds':time.monotonic()-t,'complete':(F/c/context/'COMPLETE.json').exists()}
with ThreadPoolExecutor(max_workers=pre['workers']) as pool:
 for future in as_completed([pool.submit(work,j) for j in pre['jobs']]):
  d=future.result();records.append(d);print('FUSED_JOB',json.dumps(d),flush=True);js(F/'REMAINING_PROGRESS.json',{'target_host':'149','completed':len(records),'total':len(pre['jobs']),'records':records})
js(F/'REMAINING_COMPLETE.json',{'target_host':'149','status':'COMPLETE' if all(x['returncode']==0 and x['complete'] for x in records) else 'FAILED','seconds':time.monotonic()-started,'records':records})
