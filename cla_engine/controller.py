"""Explicit controller-role enrollment; never rely on a hostname literally '149'.

This is an operational guard against running a paid workflow on the wrong host,
not security against the account owner replacing code and its own private key.
Only a hashed machine identity is stored. No enrollment/key is uploaded to GPU.
"""
from __future__ import annotations
import hashlib,hmac,os,platform,socket,sys,time
from pathlib import Path
from .common import atomic_json,read_json,digest,fail,lock,sha

FORMAT='CLA_R62_CONTROLLER_ENROLLMENT_V1'


def machine_fingerprint():
    if platform.system()!='Linux':fail('CONTROLLER_OS_MISMATCH','149 controller requires Linux; Windows may export measurements only')
    identity=Path('/etc/machine-id')
    if not identity.is_file() or not identity.read_text().strip():
        fail('CONTROLLER_MACHINE_ID_UNAVAILABLE','Cannot bind controller identity; no hostname-only fallback')
    return hashlib.sha256(('CLA_CONTROLLER_V1|'+identity.read_text().strip()).encode()).hexdigest()


def _check_runtime(cfg):
    contract=cfg.get('controller_contract',{})
    if contract.get('role')!='149':fail('CONTROLLER_ROLE_CONTRACT_MISSING','Expected controller role 149')
    actual=str(Path(sys.executable).resolve())
    expected=contract.get('python_executable')
    if expected and actual!=str(Path(expected).resolve()):
        fail('CONTROLLER_PYTHON_MISMATCH','Use the declared existing 149 interpreter',expected=expected,actual=actual)
    return {'role':'149','machine_fingerprint':machine_fingerprint(),
            'python_executable':actual,'work_root':str(Path(cfg['work_root']).resolve()),
            'imports_root':str(Path(cfg['imports_root']).resolve())}


def enroll_controller(cfg,role='149'):
    if role!='149' or cfg.get('synthetic_fixture'):fail('CONTROLLER_ENROLLMENT_INVALID','Enroll the real 149 controller only')
    from .lineage import authority_key
    identity=_check_runtime(cfg);root=Path(cfg['work_root']);root.mkdir(parents=True,exist_ok=True)
    directory=root/'.controller_enrollment_r61'
    with lock(directory):
        path=directory/'ENROLLMENT.json'
        if path.exists():return assert_controller(cfg)
        key=authority_key(cfg,True)
        body={'format':FORMAT,'identity':identity,'enrolled_unix':time.time(),'hostname_display_only':socket.gethostname()}
        body['hmac_sha256']=hmac.new(key,digest(body).encode(),hashlib.sha256).hexdigest()
        atomic_json(path,body)
        try:path.chmod(0o600)
        except OSError:pass
    return {'state':'CONTROLLER_149_ENROLLED_NOT_TRAINED','identity':identity,'path':str(path)}


def assert_controller(cfg):
    if cfg.get('synthetic_fixture') is True:
        return {'state':'SYNTHETIC_CONTROLLER_TEST_ONLY','production_authorized':False}
    from .lineage import authority_key
    identity=_check_runtime(cfg);path=Path(cfg['work_root'])/'.controller_enrollment_r61'/'ENROLLMENT.json'
    if not path.is_file():
        fail('CONTROLLER_NOT_ENROLLED','Run: bash run_149.sh controller-init --role 149; this does not start or rent a GPU')
    record=read_json(path);mac=record.pop('hmac_sha256','');key=authority_key(cfg)
    if not hmac.compare_digest(mac,hmac.new(key,digest(record).encode(),hashlib.sha256).hexdigest()):
        fail('CONTROLLER_ENROLLMENT_CHANGED','Controller enrollment signature mismatch')
    if record.get('format')!=FORMAT or record.get('identity')!=identity:
        fail('CONTROLLER_HOST_OR_ROOT_MISMATCH','This is not the enrolled controller/runtime/persistent root')
    return {'state':'CONTROLLER_149_VERIFIED','identity':identity,'enrollment_sha256':sha(path)}
