"""Cross-file identity bindings and controller-side attestations (never a cloud key).

The local HMAC protects against accidental/manual JSON edits by a supervisor.
It is not protection from an administrator who can replace code AND the private
controller key. No key or test label is sent to a GPU capsule.
"""
from __future__ import annotations
import hashlib,hmac,os,secrets
from pathlib import Path
import numpy as np
from .common import atomic_json,read_json,sha,digest,fail,code_digest,verify_manifest

FIELDS=('code_sha256','source_lineage_sha256','task_contract_sha256','target_catalog_sha256',
        'model_contract_sha256','candidate_catalog_sha256','config_sha256','graph_variant',
        'adjustment_contract_sha256','baseline_contract_sha256','import_authority_sha256')

def candidate_hash(lnc,target,lncs,targets):
    a=np.asarray(lnc);b=np.asarray(target)
    if a.shape!=b.shape or a.ndim!=1:fail('CANDIDATE_SHAPE_INVALID','Candidate keys must be equal one-dimensional arrays')
    if not np.issubdtype(a.dtype,np.integer) or not np.issubdtype(b.dtype,np.integer):fail('CANDIDATE_ID_INVALID','Candidate indices must be integers')
    if np.any(a<0)|np.any(a>=len(lncs))|np.any(b<0)|np.any(b>=len(targets)):fail('CANDIDATE_ID_RANGE','Invalid candidate index')
    return digest([[str(lncs[i]),str(targets[j])] for i,j in zip(a,b)])

def prepare_lineage(root):
    root=Path(root);spec=read_json(root/'SPEC.json');cfg=read_json(root/'CONFIG.json')
    inputs=read_json(root/'INPUT_HASHES.json')
    candidates=read_json(root/'CANDIDATES.json')
    result={'format':'CLA_R62_LINEAGE_V1','code_sha256':code_digest(),
            'source_lineage_sha256':digest(inputs),'task_contract_sha256':sha(root/'TASK_CONTRACT.json'),
            'target_catalog_sha256':sha(root/'STATE_TARGET_ROLES.json'),
            'model_contract_sha256':digest({'model':cfg['model'],'training':cfg['training'],'architecture':spec['architecture']}),
            'candidate_catalog_sha256':digest(candidates),'config_sha256':digest(cfg),
            'graph_variant':spec['graph_variant'],'fold':spec['fold'],
            'adjustment_contract_sha256':sha(root/'ADJUSTMENT_CONTRACT.json'),
            'baseline_contract_sha256':sha(root/'BASELINE_CONTRACT.json'),
            'import_authority_sha256':sha(root/'IMPORT_AUTHORITY_BINDING.json'),
            'patient_authority_sha256':inputs['patient_map']['sha256'],
            'cancer_scope_sha256':sha(root/'CANCER_SCOPE.json'),
            'cohorts_sha256':sha(root/'COHORTS.json'),'patient_assignments_sha256':sha(root/'PATIENT_ASSIGNMENTS.json')}
    if cfg.get('relation_target_policy') is not None:
        result['relation_target_policy_sha256']=sha(root/'RELATION_TARGET_POLICY.json')
    return result

def validate_prepared(root):
    from .pathway_fusion import reject_retired_prepared
    reject_retired_prepared(root)
    root=Path(root);m=verify_manifest(root);spec=read_json(root/'SPEC.json');cfg=read_json(root/'CONFIG.json')
    if m['metadata'].get('fold')!=spec['fold'] or m['metadata'].get('config_sha256')!=digest(cfg):fail('PREPARED_CONTRACT_MISMATCH','Config/fold and manifest differ')
    actual=prepare_lineage(root);stored=read_json(root/'LINEAGE.json')
    if actual!=stored:fail('PREPARED_LINEAGE_MISMATCH','Sources, code, candidates or task contract differ')
    from .cancer_scope import resolve_cancer_scope
    cancer_scope=read_json(root/'CANCER_SCOPE.json')
    expected_scope=resolve_cancer_scope(cfg,cancer_scope['available_cancers'],m['metadata']['profile'])
    if cancer_scope!=expected_scope or spec['cancers']!=expected_scope['selected_cancers'] or spec['nodes']['cancer']!=spec['cancers']:
        fail('CANCER_SCOPE_MISMATCH','Prepared cancer scope, configuration and node catalog differ')
    from .prepared_target_policy import validate_prepared_policy
    policy_validation=validate_prepared_policy(root,cfg,spec,m['metadata']['profile'])
    from .execution_contract import verify_capsule_contract
    verify_capsule_contract(root,cfg,spec)
    if cfg['model']['task_weights']!={'pathway':1.,'state':.5}:fail('TASK_WEIGHT_DRIFT','Expected Pathway=1, State=.5')
    if read_json(root/'STATE_TARGET_ROLES.json')!=cfg['targets']:fail('TARGET_ROLE_DRIFT','Actual target catalog differs')
    ids=[t['id'] for t in cfg['targets']];cols=[t['column'] for t in cfg['targets']]
    if len(ids)!=len(set(ids)) or len(cols)!=len(set(cols)):fail('TARGET_ROLE_DRIFT','Repeated State ID/source column')
    expected_states=([t['id'] for t in policy_validation['state_targets']] if policy_validation is not None
                     else [t['id'] for t in cfg['targets'] if t['role']=='primary'])
    if spec['states']!=expected_states:fail('TARGET_ROLE_DRIFT','Primary State universe differs')
    assignments=read_json(root/'PATIENT_ASSIGNMENTS.json');cohorts=read_json(root/'COHORTS.json');catalog=read_json(root/'CANDIDATES.json')
    if set(assignments)!=set(spec['cancers']):fail('PATIENT_SCOPE_MISMATCH','Cancer catalog differs')
    if spec['validation_fold']!=(spec['fold']+1)%5 or set(spec['train_folds'])!=set(range(5))-{spec['fold'],spec['validation_fold']}:
        fail('FOLD_ROLE_MISMATCH','Feature/validation/test folds must follow the frozen scheme')
    for c,rows in assignments.items():
        patients=[r[0] for r in rows]
        if patients!=sorted(set(patients)) or any(r[1] not in range(5) for r in rows):fail('PATIENT_ASSIGNMENT_INVALID',c)
        want={'all_patients':digest(patients),'folds':{str(f):digest([p for p,k in rows if k==f]) for f in range(5)}}
        if cohorts[c]!=want:fail('PATIENT_COHORT_HASH_MISMATCH',c)
    views=read_json(root/'VIEWS.json')
    if len({v['id'] for v in views})!=len(views):fail('DUPLICATE_TRAIN_VIEW','Repeated view identity')
    expected={f'{c}__{v}' for c in spec['cancers'] for v in ['validation']+[f'train{j}' for j in spec['train_folds']]}
    if {v['id'] for v in views}!=expected:fail('VIEW_GRID_MISMATCH','Missing or additional source/label views')
    for view in views:
        folder=root/'views'/view['id'];pr=read_json(folder/'PROVENANCE.json');c=pr['cancer']
        ci=view.get('cancer_index')
        if (not isinstance(ci,int) or isinstance(ci,bool) or ci<0 or ci>=len(spec['cancers'])
                or spec['cancers'][ci]!=c or not view['id'].startswith(c+'__')):
            fail('VIEW_CANCER_ROUTING_MISMATCH','view, provenance and cancer-index must agree',view=view)
        expected_split='validation' if view['id']==c+'__validation' else 'train'
        if view.get('split')!=expected_split:fail('VIEW_SPLIT_MISMATCH',view['id'])
        source=pr['feature_folds'];label=pr['label_folds']
        if set(source)&set(label) or spec['fold'] in source+label:fail('PATIENT_FOLD_LEAKAGE',view['id'])
        expect_label=[spec['validation_fold']] if view['split']=='validation' else [int(view['id'].rsplit('train',1)[1])]
        expect_source=spec['train_folds'] if view['split']=='validation' else [j for j in spec['train_folds'] if j not in expect_label]
        if source!=expect_source or label!=expect_label:fail('VIEW_FOLD_ROLE_DRIFT',view['id'])
        if pr['feature_patient_hash']!=digest([p for p,k in assignments[c] if k in source]) or pr['label_patient_hash']!=digest([p for p,k in assignments[c] if k in label]):fail('PATIENT_COHORT_HASH_MISMATCH',view['id'])
        from .execution_contract import verify_graph_view_audit
        verify_graph_view_audit(root,spec,view,cfg)
        for task in ('pathway','state'):
            targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
            with np.load(folder/f'{task}.npz',allow_pickle=False) as data:
                if candidate_hash(data['lnc'],data['target'],spec['nodes']['lncRNA'],targets)!=catalog[c][task]['ordered_keys_sha256']:fail('CANDIDATE_VIEW_MISMATCH',view['id']+':'+task)
                from .readiness import validate_task_arrays
                # Preserve zero-usable views as an audited availability mask;
                # aggregate task checks remain strict in prepare().
                validate_task_arrays(data,view['id'],task,spec['numeric_features'],allow_empty=True,
                                     objective=spec.get('relation_objective','binary'))
    return m,spec,cfg,stored

def verify_sealed(root,sealed):
    root=Path(root);sealed=Path(sealed);m,spec,cfg,lin=validate_prepared(root);sm=verify_manifest(sealed)
    expected=read_json(root/'SEALED_BINDING.json')
    meta=sm['metadata']
    for k,v in {'prepared_manifest_sha256':sha(root/'MANIFEST.json'),'fold':spec['fold'],
                'profile':m['metadata']['profile'],'binding_sha256':sha(root/'SEALED_BINDING.json')}.items():
        if meta.get(k)!=v:fail('SEALED_PREPARED_BINDING_MISMATCH',k)
    if set(sm['files'])!=set(expected['files']):fail('SEALED_FILE_SET_MISMATCH','Unexpected/missing label files')
    for rel,item in expected['files'].items():
        if sha(sealed/rel)!=item['sha256']:fail('SEALED_LABEL_HASH_MISMATCH',rel)
        with np.load(sealed/rel,allow_pickle=False) as f:
            if not {'lnc','target','label','available','outer_patient_hash'}<=set(f.files):fail('SEALED_KEYS_MISSING',rel)
            task=item['task'];targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
            ch=candidate_hash(f['lnc'],f['target'],spec['nodes']['lncRNA'],targets)
            if ch!=item['candidate_sha256'] or str(f['outer_patient_hash'].item())!=item['outer_patient_hash']:fail('SEALED_IDENTITY_MISMATCH',rel)
    return expected

def expected_predictions(spec):
    return {f'predictions/{c}__{t}.npz' for c in spec['cancers'] for t in ('pathway','state')}

def completion_state(cfg,profile,alpha,metrics):
    if profile=='formal':return 'TRAINED_FORMAL_CANDIDATE'
    return 'GO_TO_FORMAL_EVALUATION' if alpha>0 and min(metrics[t]['delta'] for t in ('pathway','state'))>=cfg['gates']['diagnostic_min_delta'] else 'NO_GO_WITHIN_DIAGNOSTIC_BUDGET'

def verify_completion(prepared,run):
    """Validate receipt against a SHA-checked project checkpoint's inner completion.
    Loads trusted project torch checkpoints only; never arbitrary third-party pt files.
    """
    import torch
    from .train import model_signature,load_npz,aggregate_metrics
    root=Path(prepared);run=Path(run);m,spec,cfg,lin=validate_prepared(root)
    r=read_json(run/'TRAINING_RECEIPT.json')
    if r.get('state') not in {'GO_TO_FORMAL_EVALUATION','NO_GO_WITHIN_DIAGNOSTIC_BUDGET','TRAINED_FORMAL_CANDIDATE'}:fail('TRAINING_NOT_COMPLETE','Invalid training receipt state')
    sig=model_signature(root,cfg,spec)
    if r.get('signature')!=sig or r.get('prepared_manifest_sha256')!=sha(root/'MANIFEST.json'):fail('RESULT_BINDING_MISMATCH','Wrong prepared/code/model')
    if r.get('lineage')!=lin:fail('RESULT_LINEAGE_MISMATCH','Training lineage differs')
    if r.get('task_weights')!={'pathway':1.,'state':.5}:fail('TASK_WEIGHT_DRIFT','Training loss contract differs')
    if set(r.get('prediction_files',{}))!=expected_predictions(spec):fail('PREDICTION_FILE_SET_MISMATCH','Missing/extra prediction files')
    actual={'predictions/'+p.name for p in (run/'predictions').iterdir() if p.is_file()}
    if actual!=expected_predictions(spec):fail('STALE_PREDICTION_FILE','Unexpected prediction file in current run')
    ck=r.get('last_checkpoint',{});cp=(run/ck.get('file','')).resolve()
    if not cp.is_relative_to(run.resolve()) or not cp.is_file() or sha(cp)!=ck.get('sha256') or ck.get('signature')!=sig:fail('COMPLETION_CHECKPOINT_MISMATCH','Last checkpoint missing or changed')
    inner=torch.load(cp,map_location='cpu',weights_only=False)
    if inner.get('format')!='CLA_FULL_PYG_R62_FULL_TRAINING_STATE_V1':
        fail('COMPLETION_CHECKPOINT_FORMAT','Only the current full training state can commit results')
    step=inner.get('next_step');best_step=r.get('best_step');alpha=r.get('alpha')
    if not isinstance(step,int) or step<=0 or not isinstance(best_step,int) or not 0<=best_step<=step:
        fail('COMPLETION_STEP_INVALID','Completed optimizer progress and selected checkpoint are inconsistent')
    if not isinstance(alpha,(int,float)) or not np.isfinite(alpha) or alpha not in cfg['training']['alpha_grid']:
        fail('COMPLETION_ALPHA_INVALID','Selected alpha is outside the predeclared validation grid')
    payload={k:v for k,v in r.items() if k!='last_checkpoint'}
    if inner.get('signature')!=sig or inner.get('completion')!=payload:fail('COMPLETION_RECEIPT_TAMPERED','Receipt must equal committed checkpoint payload')
    if inner['best']['alpha']!=r['alpha'] or inner['best']['step']!=r['best_step']:fail('COMPLETION_BEST_MISMATCH','alpha/best-step differs from optimizer checkpoint')
    if any(r.get('gradient_proof',{}).get(k,0)<=0 for k in ['pathway_max_shared_grad','state_max_shared_grad']):fail('SHARED_GRADIENT_PROOF_MISSING','Both tasks must update the core')
    pred={t:[] for t in ('pathway','state')}
    for cancer in spec['cancers']:
        for task in pred:
            rel=f'predictions/{cancer}__{task}.npz';p=run/rel
            if sha(p)!=r['prediction_files'][rel]:fail('RESULT_HASH_MISMATCH',rel)
            a=load_npz(p);src=load_npz(root/'views'/f'{cancer}__validation'/f'{task}.npz')
            for k in ['lnc','target']:
                if not np.array_equal(a[k],src[k]):fail('PREDICTION_IDENTITY_MISMATCH',rel)
            if not np.array_equal(a['baseline'],src['base']):fail('PREDICTION_BASELINE_MISMATCH',rel)
            n=len(src['lnc'])
            if any(a[k].shape!=(n,) or not np.isfinite(a[k]).all() for k in ['score','baseline','direction_logit']):fail('PREDICTION_NONFINITE_OR_SHAPE',rel)
            pred[task].append((cancer,src,a['score'],a['direction_logit']))
    metrics=aggregate_metrics(pred,spec)
    # Recompute admission from the saved validation predictions; receipt state is not an authority.
    if digest(metrics)!=digest(r['validation']):fail('VALIDATION_RECEIPT_MISMATCH','Saved validation metrics do not match predictions')
    if r['state']!=completion_state(cfg,m['metadata']['profile'],r['alpha'],metrics):fail('ADMISSION_STATE_MISMATCH','State contradicts independently recomputed validation metrics')
    return r,lin

def authority_key(cfg,create=False):
    d=Path(cfg['work_root'])/'.controller_authority_r61';d.mkdir(parents=True,exist_ok=True)
    try:os.chmod(d,0o700)
    except OSError:pass
    p=d/'attestation.key'
    if not p.exists():
        if not create:fail('CONTROLLER_ATTESTATION_MISSING','Controller key is missing; do not fabricate authorization')
        try:
            fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as f:f.write(secrets.token_bytes(32));f.flush();os.fsync(f.fileno())
        except FileExistsError:pass
    key=p.read_bytes()
    if len(key)!=32:fail('CONTROLLER_KEY_INVALID','Expected 32-byte local key')
    return key

def sign_file(path,cfg,role):
    from .controller import assert_controller
    assert_controller(cfg)
    key=authority_key(cfg,True);path=Path(path)
    body={'format':'CLA_R62_CONTROLLER_ATTESTATION_V1','role':role,'file_sha256':sha(path),
          'code_sha256':code_digest(),'config_sha256':digest(cfg)}
    mac=hmac.new(key,digest(body).encode(),hashlib.sha256).hexdigest()
    atomic_json(path.with_name(path.name+'.attestation.json'),{**body,'hmac_sha256':mac})

def verify_signed_file(path,cfg,role):
    from .controller import assert_controller
    assert_controller(cfg)
    path=Path(path);p=path.with_name(path.name+'.attestation.json')
    if not p.exists():fail('CONTROLLER_ATTESTATION_MISSING',str(path))
    r=read_json(p);mac=r.pop('hmac_sha256','');key=authority_key(cfg)
    if not hmac.compare_digest(mac,hmac.new(key,digest(r).encode(),hashlib.sha256).hexdigest()):fail('CONTROLLER_ATTESTATION_INVALID','File authorization was edited')
    if r!={'format':'CLA_R62_CONTROLLER_ATTESTATION_V1','role':role,'file_sha256':sha(path),'code_sha256':code_digest(),'config_sha256':digest(cfg)}:fail('CONTROLLER_ATTESTATION_STALE','File/code/config changed')
