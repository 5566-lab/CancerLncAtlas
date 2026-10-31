"""Controller-only outer-test opening after a verified immutable model freeze.

All test readers use the same registry. Freeze authenticates the completed model,
selection rules, candidates and cohort hashes BEFORE any held-out effects are read.
Filesystem administrators can still read raw source data; this is a workflow gate.
"""
from __future__ import annotations
from pathlib import Path
import time
from .common import atomic_json,read_json,digest,sha,fail,lock,code_digest
from .lineage import validate_prepared,verify_completion,sign_file,verify_signed_file
from .controller import assert_controller


def _freeze_identity(prepared,training):
    root=Path(prepared).resolve();run=Path(training).resolve()
    m,spec,cfg,lineage=validate_prepared(root);assert_controller(cfg)
    if not (run/'TRAINING_RECEIPT.json').is_file():
        fail('MODEL_NOT_FROZEN','A completed verified training run is required before any outer-test audit')
    receipt,_=verify_completion(root,run)
    return cfg,{
        'format':'CLA_R62_FROZEN_MODEL_V1','prepared_manifest_sha256':sha(root/'MANIFEST.json'),
        'model_signature':receipt['signature'],'training_receipt_sha256':sha(run/'TRAINING_RECEIPT.json'),
        'checkpoint_sha256':receipt['last_checkpoint']['sha256'],'selected_alpha':receipt['alpha'],
        'selected_step':receipt['best_step'],'training_state':receipt['state'],
        'fold':spec['fold'],'profile':m['metadata']['profile'],'graph_variant':spec['graph_variant'],
        'candidate_catalog_sha256':sha(root/'CANDIDATES.json'),
        'patient_assignments_sha256':sha(root/'PATIENT_ASSIGNMENTS.json'),
        'sealed_binding_sha256':sha(root/'SEALED_BINDING.json'),
        'task_contract_sha256':sha(root/'TASK_CONTRACT.json'),
        'selection_rules_sha256':digest({'training':cfg['training'],'gates':cfg['gates'],'metrics_contract':cfg['metrics_contract']}),
        'lineage':lineage,'code_sha256':code_digest(),
    }


def _registry_path(cfg,identity):
    return Path(cfg['work_root'])/'outer_access_registry_r61'/identity['prepared_manifest_sha256']/'MODEL_FREEZE.json'


def freeze_model(prepared,training):
    cfg,identity=_freeze_identity(prepared,training);path=_registry_path(cfg,identity)
    with lock(path.parent):
        if path.exists():
            verify_signed_file(path,cfg,'MODEL_FREEZE')
            record=read_json(path)
            if record.get('binding')!=identity:
                fail('FROZEN_MODEL_REPLACEMENT_FORBIDDEN','This prepared test set is already bound to another selected model; do not select on test')
        else:
            record={'binding':identity,'frozen_unix':time.time(),'outer_test_opened':False}
            atomic_json(path,record);sign_file(path,cfg,'MODEL_FREEZE')
    return {'state':'MODEL_FROZEN_OUTER_NOT_READ','freeze_path':str(path),'freeze_sha256':sha(path),'binding':identity}


def require_frozen_model(prepared,training):
    if training is None:fail('MODEL_FREEZE_REQUIRED','outer_test requires --run and a prior freeze-model command')
    cfg,identity=_freeze_identity(prepared,training);path=_registry_path(cfg,identity)
    if not path.is_file():fail('MODEL_FREEZE_REQUIRED','First run freeze-model for this completed model; source-only audits do not require it')
    verify_signed_file(path,cfg,'MODEL_FREEZE')
    if read_json(path).get('binding')!=identity:fail('FROZEN_MODEL_BINDING_CHANGED','Model/selection/cohort/candidate source no longer matches the freeze')
    return {'freeze_path':str(path),'freeze_sha256':sha(path),'binding':identity}


def record_outer_open(prepared,training,*,purpose,output):
    proof=require_frozen_model(prepared,training)
    _,_,cfg,_=validate_prepared(prepared)
    registry=Path(proof['freeze_path']).parent
    with lock(registry/'access_lock'):
        path=registry/'ACCESS_LOG.json'
        if path.exists():
            verify_signed_file(path,cfg,'OUTER_ACCESS_LOG');log=read_json(path)
        else:log={'format':'CLA_R62_OUTER_ACCESS_LOG_V1','freeze_sha256':proof['freeze_sha256'],'reads':[]}
        if log['freeze_sha256']!=proof['freeze_sha256']:fail('OUTER_ACCESS_FREEZE_CHANGED','Test access log freeze differs')
        log['reads'].append({'purpose':purpose,'output':str(Path(output).resolve()),'authorized_unix':time.time()})
        atomic_json(path,log);sign_file(path,cfg,'OUTER_ACCESS_LOG')
    return proof
