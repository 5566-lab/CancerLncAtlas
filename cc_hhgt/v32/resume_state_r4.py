"""Atomic, hash-bound private-module checkpoints. Local trusted state only."""
from __future__ import annotations
import hashlib,json,os,time,uuid
from pathlib import Path
import numpy as np

class TrainingPaused(RuntimeError):
    """Recoverable pause; never means the module is trained or admitted."""

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial.'+uuid.uuid4().hex)
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save(root,state):
    import torch
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/f"state_{int(state['steps']):010d}_{uuid.uuid4().hex[:10]}.pt";tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    record={'file':path.name,'sha256':sha(path),'signature':state['signature'],'steps':state['steps'],'created_unix':time.time()}
    atomic_json(root/(path.name+'.ready.json'),record);atomic_json(root/'LATEST.json',record)
    # Keep two COMPLETE generations; no partial file is a recovery candidate.
    records=sorted(root.glob('state_*.pt.ready.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for rp in records[2:]:
        try:r=json.loads(rp.read_text());(root/r['file']).unlink(missing_ok=True);rp.unlink()
        except (OSError,ValueError,KeyError):pass
    return record

def load(root,signature):
    import torch
    root=Path(root)
    for p in sorted(root.glob('state_*.pt.ready.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        try:r=json.loads(p.read_text())
        except (OSError,ValueError):continue
        if r.get('signature')!=signature:raise RuntimeError('PRIVATE_RESUME_INPUT_OR_CONFIG_DRIFT')
        f=(root/r['file']).resolve()
        if not f.is_relative_to(root.resolve()):raise RuntimeError('PRIVATE_CHECKPOINT_PATH_ESCAPE')
        if not f.is_file() or sha(f)!=r['sha256']:continue
        v=torch.load(f,map_location='cpu',weights_only=False)
        if v.get('signature')!=signature:raise RuntimeError('PRIVATE_CHECKPOINT_INNER_SIGNATURE_DRIFT')
        return v
    if (root/'LATEST.json').exists():raise RuntimeError('NO_VALID_PRIVATE_CHECKPOINT')
    return None
