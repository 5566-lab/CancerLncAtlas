"""Attempt-scoped status and progress checks; never restore ephemeral worker state."""
from __future__ import annotations
import os,shutil,time,uuid,tarfile,re,math
from pathlib import Path
from .common import atomic_json,read_json,fail,status

_CONTEXT={}

def begin_attempt(out):
    out=Path(out);attempt=os.environ.get('CLA_REMOTE_ATTEMPT_ID') or uuid.uuid4().hex
    quarantine=out/'quarantine'/('before_'+attempt)
    for name in ('STATUS.json','HEARTBEAT.json','predictions','COMPLETION.json'):
        p=out/name
        if p.exists():quarantine.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(quarantine/name))
    _CONTEXT[str(out.resolve())]={'attempt_id':attempt,'attempt_start_unix':time.time()}
    atomic_json(out/'ACTIVE_ATTEMPT.json',_CONTEXT[str(out.resolve())])
    worker_status(out,'STARTING',phase='initializing',next_step=0)
    return attempt

def worker_status(out,code,message='',**extra):
    context=_CONTEXT.get(str(Path(out).resolve()),{})
    if context.get('device_proof',{}).get('device_type')=='cuda':
        import torch
        ix=context['device_proof']['device_index']
        extra['device_memory']={'allocated_bytes':int(torch.cuda.memory_allocated(ix)),
            'reserved_bytes':int(torch.cuda.memory_reserved(ix)),
            'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(ix))}
    r=status(out,code,message,**context,**extra)
    atomic_json(Path(out)/'HEARTBEAT.json',{'attempt_id':context.get('attempt_id'),
        'updated_unix':time.time(),'pid':os.getpid(),'phase':extra.get('phase',code),'next_step':extra.get('next_step')})
    return r

def check_remote_status(st,attempt_id,alive,now,started,startup_timeout=120,progress_timeout=1800):
    """Pure function used by SSH supervisor and fault-injection tests."""
    if st and st.get('attempt_id')!=attempt_id:
        if not alive:return 'WORKER_EXITED_WITHOUT_TERMINAL'
        if now-started>startup_timeout:return 'STALE_ATTEMPT_STATUS'
        return 'WAIT_FOR_CURRENT_ATTEMPT'
    if st and st.get('state') in {'GO_TO_FORMAL_EVALUATION','NO_GO_WITHIN_DIAGNOSTIC_BUDGET','TRAINED_FORMAL_CANDIDATE','ERROR','PAUSED_BUDGET_OR_SIGNAL'}:
        return 'TERMINAL'
    if not alive:return 'WORKER_EXITED_WITHOUT_TERMINAL'
    if not st:
        return 'STARTUP_TIMEOUT' if now-started>startup_timeout else 'WAIT_FOR_CURRENT_ATTEMPT'
    if now-st.get('updated_unix',0)>progress_timeout:return 'WORKER_PROGRESS_STALLED'
    return 'RUNNING'

def validate_checkpoint_pointer(record):
    if not isinstance(record, dict):fail('CHECKPOINT_POINTER_INVALID','Expected a bound checkpoint record')
    raw=record.get('file');value=record.get('sha256')
    if not isinstance(raw,str) or not re.fullmatch(r'checkpoints/[A-Za-z0-9_.-]+\.pt',raw):
        fail('CHECKPOINT_POINTER_INVALID','Checkpoint must be a simple relative checkpoints/*.pt path')
    if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value):
        fail('CHECKPOINT_POINTER_INVALID','A full SHA-256 is required')
    for key in ('next_step','created_unix'):
        if key in record and (not isinstance(record[key],(int,float)) or not math.isfinite(record[key]) or record[key]<0):
            fail('CHECKPOINT_POINTER_INVALID','Invalid checkpoint progress metadata',field=key)
    if 'next_step' in record and int(record['next_step'])!=record['next_step']:
        fail('CHECKPOINT_POINTER_INVALID','next_step must be an integer')
    return record

def make_resume_tar(source,target):
    """Whitelist only immutable checkpoints + pointers. Never STATUS, lease or predictions."""
    root=Path(source)
    if (root/"LATEST.json").exists():
        from .common import sha
        latest=validate_checkpoint_pointer(read_json(root/"LATEST.json"))
        cp=root/latest["file"]
        if cp.is_symlink() or not cp.is_file() or sha(cp)!=latest["sha256"]:
            fail("RESUME_CHECKPOINT_HASH_MISMATCH","Local persisted checkpoint is missing or changed")
    with tarfile.open(target,'w') as tar:
        for name in ('LATEST.json',):
            if (root/name).is_file():tar.add(root/name,arcname=name)
        for p in sorted((root/'checkpoints').glob('*')):
            if p.is_file() and (p.suffix=='.pt' or p.name.endswith('.pt.ready.json')):
                tar.add(p,arcname=p.relative_to(root).as_posix())


class OptimizerProgressWatch:
    """Attempt-scoped monotonic optimization progress, independent of heartbeats."""
    def __init__(self,attempt_id,started_unix,*,first_step_timeout=1800,progress_timeout=1800,state=None):
        self.attempt_id=attempt_id;self.started=started_unix
        self.first_timeout=first_step_timeout;self.progress_timeout=progress_timeout
        self.last_step=None;self.last_advance=started_unix;self.optimization_observed=False
        if state:
            if state.get('attempt_id')!=attempt_id:fail('MONITOR_ATTEMPT_MISMATCH','Cannot reuse another attempt monitor')
            self.last_step=state.get('last_step');self.last_advance=state.get('last_advance',started_unix)
            self.optimization_observed=bool(state.get('optimization_observed'))
    def observe(self,st,now):
        if not st or st.get('attempt_id')!=self.attempt_id:return 'WAIT_FOR_CURRENT_ATTEMPT'
        step=st.get('next_step')
        if not isinstance(step,int) or isinstance(step,bool) or step<0:return 'WORKER_PROGRESS_INVALID'
        if self.last_step is not None and step<self.last_step:return 'WORKER_OPTIMIZER_STEP_REGRESSED'
        if self.last_step is None or step>self.last_step:
            self.last_advance=now
            if step>0 and st.get('state')=='TRAINING' and st.get('optimizer_steps_this_attempt',0)>0:self.optimization_observed=True
            self.last_step=step
        if not self.optimization_observed:
            return 'WORKER_NO_OPTIMIZER_PROGRESS' if now-self.started>self.first_timeout else 'INITIALIZING_NO_OPTIMIZER_PROGRESS_YET'
        if now-self.last_advance>self.progress_timeout:return 'WORKER_OPTIMIZER_PROGRESS_STALLED'
        return 'OPTIMIZER_PROGRESS_OBSERVED'
    def snapshot(self):
        return {'attempt_id':self.attempt_id,'last_step':self.last_step,'last_advance':self.last_advance,
                'optimization_observed':self.optimization_observed,'worker_started_unix':self.started}


def register_worker_device(out,device):
    import torch
    dev=torch.device(device)
    proof={'pid':os.getpid(),'device_type':dev.type,'device_requested':str(dev),
        'torch_version':str(torch.__version__),'cuda_runtime':torch.version.cuda,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_utilization_claimed':False}
    if dev.type=='cuda':
        if not torch.cuda.is_available():fail('GPU_NOT_AVAILABLE','Declared CUDA worker has no CUDA device')
        ix=dev.index if dev.index is not None else torch.cuda.current_device()
        p=torch.cuda.get_device_properties(ix)
        proof.update(device_index=ix,device_name=p.name,total_memory_bytes=int(p.total_memory))
    _CONTEXT.setdefault(str(Path(out).resolve()),{})['device_proof']=proof
    atomic_json(Path(out)/'DEVICE_PROOF.json',proof)
    return proof
