from pathlib import Path
import subprocess,sys,json,time,socket
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';assert socket.gethostname()=='149';records=[]
jobs=[(c,x) for x in ['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test'] for c in ['BRCA','LUAD','COAD']]
def js(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def run(job):
 c,x=job;deadline=time.monotonic()+7200
 while not (F/c/x/'COMPLETE.json').exists():
  if time.monotonic()>deadline:raise TimeoutError(job)
  time.sleep(10)
 dest=F/'validation'/(c+'__'+x+'.json')
 if dest.exists():return json.loads(dest.read_text())
 cmd=[sys.executable,str(R/'validate_fused_y_context.py'),'--cancer',c,'--context',x]
 if x=='train_label_fold2':cmd.append('--independent')
 with (F/'logs'/('VALIDATE_'+c+'__'+x+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  js(F/'logs'/('VALIDATE_'+c+'__'+x+'.PROCESS.json'),{'target_host':'149','pid':proc.pid})
  rc=proc.wait(timeout=3600)
 if rc:raise RuntimeError((job,rc))
 return json.loads(dest.read_text())
with ThreadPoolExecutor(max_workers=3) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):
  d=future.result();records.append(d);print('VALIDATED',d['cancer_id'],d['context'],flush=True);js(F/'VALIDATION_PROGRESS.json',{'target_host':'149','complete':len(records),'total':15,'records':records})
js(F/'VALIDATION_COMPLETE.json',{'target_host':'149','status':'PASS','contexts':15,'records':records})
