"""One-time host149 recovery: reuse two completed checks and validate folds 2-4."""
import concurrent.futures
import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import socket
import traceback

ROOT = Path('/dell_2/DSC/CancerLncAtlas/inputs/gene_node_omics_20261029_r1')
OLD = ROOT/'node_folds_r1'
NEW = ROOT/'node_folds_r2'

def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value,indent=2)+'\n')
def record(path):
    path=Path(path).resolve(); h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest()}
def pin(item):
    assert record(item['path'])['sha256']==item['sha256'], item['path']
    return Path(item['path'])

def validate_fold(fold):
    assert socket.gethostname()=='149'
    import torch
    def forbidden(*args,**kwargs):raise RuntimeError('No optimizer permitted in preparation')
    for name in ['Adam','AdamW','SGD']:setattr(torch.optim,name,forbidden)
    from cc_hhgt.v32.native_graph_input import load_training_graph_inputs
    from cc_hhgt.v32.training import _validate_prepared
    from cc_hhgt.v32.gene_patient_omics import CHANNELS
    manifest=read(NEW/'INPUT_MANIFEST.pending.json')
    row=next(r for r in manifest['fold_inputs'] if r['fold']==fold)
    cfg=read(NEW/'MODEL_CONFIG_PATCH.json')
    print(f'FOLD_{fold}_VERIFY_PARENT',flush=True)
    parent=pin(row)
    print(f'FOLD_{fold}_LOAD_AND_REPLAY_NODE_INPUTS',flush=True)
    payload=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    loaded,overlay=load_training_graph_inputs(torch,payload=payload,config=cfg,
        input_manifest_path=NEW/'INPUT_MANIFEST.pending.json',fold=fold,root=parent.parent,
        prepared_sha256=row['sha256'])
    print(f'FOLD_{fold}_VALIDATE_PREPARED',flush=True)
    _validate_prepared(loaded,fold=fold,artifact_hashes=payload['artifact_hashes'],authorized_graph_overlay=overlay)
    block=loaded['bundle'].node_omics['gene']
    check={'fold':fold,'prepared_validator':'PASS','gene_nodes':len(block),'node_input_dim':loaded['bundle'].node_input_dim,
        'observed_gene_nodes_by_channel':{c:int(block[:,i*2+1].sum()) for i,c in enumerate(CHANNELS)},
        'input_authority_hashes':loaded['input_authority_hashes'],'training_started':False}
    write(NEW/f'FOLD_{fold}_PREPARED_VALIDATION.json',check)
    print(f'FOLD_{fold}_PREPARED_PASS',flush=True)
    return fold

def main():
    assert socket.gethostname()=='149'
    assert not (OLD/'COMPLETE.json').exists()
    manifest=read(OLD/'INPUT_MANIFEST.pending.json')
    product=read(pin(manifest['gene_product']))
    assert product['status']=='PASS' and product['raw_file_count']==18600 and len(product['partitions'])==33
    assert {r['fold'] for r in manifest['fold_inputs']}==set(range(5)) and len(manifest['fold_inputs'])==5
    for item in product['implementation'].values():pin(item)
    expected=read(ROOT/'VERIFIED_LOCAL_PARENTS.json')
    assert expected['status']=='PASS'
    parents={r['original']['fold']:r['local'] for r in expected['parents']}
    prior=[]
    for row in manifest['fold_inputs']:
        fold=row['fold']
        assert row['sha256']==parents[fold]['sha256'] and Path(row['path']).resolve()==Path(parents[fold]['path']).resolve()
        assert Path(row['path']).stat().st_size==row['bytes']
        node=read(pin(row['node_omics']))
        assert node['outer_fold']==fold and node['channels']==['cnv_signed','methylation_promoter_beta']
        if fold in (0,1):
            check=read(OLD/f'FOLD_{fold}_PREPARED_VALIDATION.json')
            assert check['fold']==fold and check['prepared_validator']=='PASS' and check['node_input_dim']==6
            assert check['training_started'] is False
            assert check['input_authority_hashes']['node_omics_manifest_sha256']==row['node_omics']['sha256']
            assert check['input_authority_hashes']['node_omics_table_sha256']==node['table_sha256']
            prior.append(record(OLD/f'FOLD_{fold}_PREPARED_VALIDATION.json'))
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>260*1024**3, 'Insufficient available RAM for two prepared validation workers'
    NEW.mkdir(exist_ok=False)
    write(NEW/'LAUNCH_PREFLIGHT.json',{'target_host':'149','training_started':False,'paid_compute_started':False,
        'operation':'RESUME_REMAINING_FORMAL_VALIDATIONS','source_pending_manifest':record(OLD/'INPUT_MANIFEST.pending.json'),
        'reuse_validated_folds':[0,1],'prior_validation_receipts':prior,'fresh_validation_folds':[2,3,4],
        'max_workers':2,'timeout_seconds':7200,'memory_available_bytes':available,
        'recovery_script':record(__file__),'gene_product':manifest['gene_product'],
        'interruption':'Prior 10800-second controller expired while starting fold 2 after folds 0 and 1 passed.'})
    for item in prior:shutil.copyfile(item['path'],NEW/Path(item['path']).name)
    shutil.copyfile(OLD/'MODEL_CONFIG_PATCH.json',NEW/'MODEL_CONFIG_PATCH.json')
    write(NEW/'INPUT_MANIFEST.pending.json',manifest)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for fold in pool.map(validate_fold,[2,3,4]):print('COLLECTED_FOLD',fold,flush=True)
    checks=[read(NEW/f'FOLD_{fold}_PREPARED_VALIDATION.json') for fold in range(5)]
    assert all(c['prepared_validator']=='PASS' and c['node_input_dim']==6 for c in checks)
    assert record(OLD/'INPUT_MANIFEST.pending.json')['sha256']==read(NEW/'LAUNCH_PREFLIGHT.json')['source_pending_manifest']['sha256']
    for item in prior:pin(item)
    for item in product['implementation'].values():pin(item)
    write(NEW/'INPUT_MANIFEST.json',manifest)
    write(NEW/'COMPLETE.json',{'target_host':'149','status':'PASS','training_started':False,
        'input_manifest':record(NEW/'INPUT_MANIFEST.json'),'model_config_patch':record(NEW/'MODEL_CONFIG_PATCH.json'),
        'folds':checks,'learned_embeddings_generated':False,'recovery_preflight':record(NEW/'LAUNCH_PREFLIGHT.json'),
        'reused_previously_validated_folds':[0,1],'freshly_validated_folds':[2,3,4]})
    print('ALL_FIVE_FORMAL_FOLDS_PASS',flush=True)

if __name__=='__main__':
    def stopped(signum,frame):raise RuntimeError('Recovery deadline or termination signal '+str(signum))
    signal.signal(signal.SIGTERM,stopped)
    try:main()
    except BaseException as error:
        write(ROOT/'FORMAL_ATTACHMENT_r3_FAILED.json',{'target_host':'149','training_started':False,
            'error':str(error),'traceback':traceback.format_exc(),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
