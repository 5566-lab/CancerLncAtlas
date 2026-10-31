"""Integrity, atomic commits and machine-readable failure states. No cloud actions."""
from __future__ import annotations
import contextlib, hashlib, json, os, socket, time, uuid
from pathlib import Path
import numpy as np

class Stop(RuntimeError):
    def __init__(self, code: str, message: str, **detail):
        super().__init__(message); self.code=code; self.detail=detail

def fail(code, message, **detail): raise Stop(code, message, **detail)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return jsonable(x.tolist())
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('w',encoding='utf-8') as f:
        f.write(json.dumps(jsonable(value),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fsync_dir(path.parent)

def fsync_dir(path):
    if os.name!='nt':
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('wb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fsync_dir(path.parent)

def commit_manifest(root, metadata, *, name='MANIFEST.json'):
    root=Path(root)
    files={str(p.relative_to(root)).replace(os.sep,'/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in [name,'STATUS.json','.lock'] and '.tmp.' not in p.name and p.name != name+'.attestation.json'}
    result={'format':'CLA_FULL_PYG_R62_COMMITTED_V1','metadata':metadata,'files':files}
    atomic_json(root/name,result);return result

def verify_manifest(root, *, expected=None, name='MANIFEST.json'):
    root=Path(root);manifest=root/name
    if expected and sha(manifest)!=expected:fail('INPUT_DRIFT','Manifest SHA mismatch',path=str(manifest))
    v=read_json(manifest)
    if v.get('format')!='CLA_FULL_PYG_R62_COMMITTED_V1':fail('WRONG_GENERATION','Only committed R6.2 payloads are accepted; do not relabel older prepared/checkpoint/results')
    for rel,record in v['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()):fail('PATH_TRAVERSAL',rel)
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:fail('INPUT_DRIFT','Payload file mismatch',path=rel)
    actual={str(p.relative_to(root)).replace(os.sep,'/') for p in root.rglob('*') if p.is_file() and p.name not in [name,'STATUS.json','.lock'] and '.tmp.' not in p.name and p.name != name+'.attestation.json'}
    if actual!=set(v['files']):fail('UNLISTED_INPUT_FILE','Unexpected files in sealed package',extra=sorted(actual-set(v['files'])))
    return v

def status(root, code, message='', **extra):
    value={'format':'CLA_FULL_PYG_R62_STATUS_V1','state':code,'message':message,'updated_unix':time.time(),'pid':os.getpid(),'host':socket.gethostname(),'automatic_deploy':False,'attempt_id':os.environ.get('CLA_ATTEMPT_ID'),**extra}
    atomic_json(Path(root)/'STATUS.json',value)
    print(json.dumps(jsonable(value),ensure_ascii=False),flush=True);return value

@contextlib.contextmanager
def lock(root):
    """OS advisory lock; released by kernel on SIGKILL. Never delete someone else's lock."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    f=(root/'.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:fail('DUPLICATE_WORKER','Directory already locked')
        else:
            import fcntl
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:fail('DUPLICATE_WORKER','Directory already locked')
        yield
    finally:
        f.close()

def config(path):
    c=read_json(path)
    if c.get('format')!='CLA_FULL_PYG_R62_CONFIG_V1':fail('CONFIG_GENERATION_MISMATCH','R6.2 config required; do not rename older configurations')
    if c['model']['task_weights']!={'pathway':1.0,'state':0.5}:fail('TASK_WEIGHT_DRIFT','Frozen loss: pathway=1, state=0.5')
    if c.get('subtype_feedback') is not False:fail('SUBTYPE_FEEDBACK_FORBIDDEN','Subtype is downstream only')
    if c['graph'].get('formal_variants') != ['G0','G1','G2']:fail('FORMAL_GRID_DRIFT','All three variants are required in the formal plan')
    if int(c['model']['hidden']) <= 0 or int(c['model']['heads']) <= 0 or int(c['model']['hidden']) % int(c['model']['heads']):fail('MODEL_CONFIG','Invalid hidden/head dimensions')
    if int(c['model']['layers']) < 2:fail('MODEL_CONFIG','At least two original HGT layers are required')
    if c['model'].get('backend') != 'original_pyg_cc_hhgt':fail('BACKBONE_SUBSTITUTION_FORBIDDEN','Only the original PyG backbone is accepted')
    stats=c.get('statistics',{})
    if stats.get('cnv_transform','raw_signed') not in {'raw_signed','rank'}:fail('CNV_TRANSFORM_CONTRACT','Explicit raw_signed or rank required')
    ids=[t['id'] for t in c['targets']];cols=[t['column'] for t in c['targets']]
    if len(set(ids))!=len(ids) or len(set(cols))!=len(cols):fail('TARGET_CATALOG_CONFLICT','Duplicate target ID or source column')
    if any(t.get('role') not in {'primary','auxiliary'} for t in c['targets']):fail('TARGET_ROLE_INVALID','Targets must have explicit roles')
    from .execution_contract import validate_config_contract
    validate_config_contract(c)
    return c

def code_digest():
    root=Path(__file__).resolve().parents[1]
    files=[p for folder in ('cla_engine','cc_hhgt','scripts','config') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in {'.py','.r','.sh','.yaml','.yml','.json'}]
    files += [root/'cla.py']
    return digest({p.relative_to(root).as_posix():sha(p) for p in sorted(files)})
