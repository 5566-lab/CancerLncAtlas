"""Matched numeric Lasso components for both exact Pathway and State relations."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .common import atomic_json,atomic_npz,sha
from .exact_baseline import ENCODING
from .m0_relation import fit_strength_cancer
from .relation_components import CONTEXTS,load_relation_components
from .relation_features import verified_file
from .target_policy import load_target_policy


def resolve_state_families(rows,targets):
    """Phenotype families come from State definitions, not PROGENy routing."""
    family_sets={}
    for row in rows:
        # A PROGENy readout can support an existing State identity (e.g.
        # hypoxia), but its method class is not a second phenotype family.
        if row['state_id'].startswith('PROGENY:'):
            continue
        family_sets.setdefault(row['canonical_id'],set()).add(row['state_family'])
    if any(len(family_sets.get(t,set()))!=1 or '' in family_sets[t] for t in targets):
        raise ValueError('RELATION_BASELINE_STATE_FAMILY_AMBIGUOUS_OR_MISSING')
    return {t:next(iter(family_sets[t])) for t in targets}


def target_balanced_weights(target,available,targets,families=None):
    """Equal total weight per observed target; State also balances families."""
    target=np.asarray(target);available=np.asarray(available,bool)
    out=np.zeros(len(target),np.float32)
    counts=np.bincount(target[available],minlength=len(targets))
    active=np.flatnonzero(counts)
    if families is not None and set(targets)-set(families):
        raise ValueError('RELATION_BASELINE_STATE_FAMILY_MISSING')
    family_counts={}
    if families is not None:
        for j in active:
            family=families[targets[j]]
            family_counts[family]=family_counts.get(family,0)+1
    for j in active:
        denominator=counts[j]*(family_counts[families[targets[j]]] if families is not None else 1)
        out[available & (target==j)]=1./denominator
    return out


def task_block(component,task,targets,families=None):
    axes=component['axes'];arr=component['arrays'];lookup={t:i for i,t in enumerate(targets)}
    take=np.asarray(axes['loss_groups'])[arr['target']]==task
    selected=np.asarray(axes['canonical_ids'])[arr['target'][take]]
    target=np.asarray([lookup[t] for t in selected],np.int64)
    result={k:v[take] for k,v in arr.items()};result['target']=target
    result['weight']=target_balanced_weights(target,result['available'],targets,families)
    result['direction_label']=np.zeros(len(target),np.float32)
    result['direction_available']=np.zeros(len(target),bool)
    result['identities']=[(component['receipt']['cancer'],axes['lncrna_ids'][i],t)
                          for i,t in zip(result['lnc'],selected)]
    return result


def predict_numeric_head(head,numeric):
    x=(np.asarray(numeric,float)-head['numeric_mean'])/head['numeric_scale']
    return (x @ np.asarray(head['coef']).reshape(-1)+float(head['intercept'][0])).astype(np.float32)


def materialize_relation_baseline(component_paths,policy_binding,state_binding,out,*,cfg=None,progress=None):
    from .local_execution import require_host_149
    require_host_149()
    if set(component_paths)!=set(CONTEXTS):
        raise ValueError('RELATION_BASELINE_CONTEXT_GRID')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('RELATION_BASELINE_OUTPUT_NOT_EMPTY')
    policy_path=verified_file(policy_binding);state_path=verified_file(state_binding)
    policies=load_target_policy(policy_path)
    with state_path.open(encoding='utf-8-sig',newline='') as f:state_rows=list(csv.DictReader(f,delimiter='\t'))
    components={name:load_relation_components(path) for name,path in component_paths.items()}
    first=components[CONTEXTS[0]];cancer=first['receipt']['cancer'];axes=first['axes']
    if any(c['receipt']['cancer']!=cancer or c['axes']!=axes or c['feature_names']!=first['feature_names'] for c in components.values()):
        raise ValueError('RELATION_BASELINE_AXIS_MISMATCH')
    for name,c in components.items():
        if c['receipt']['context']!=name:raise ValueError('RELATION_BASELINE_CONTEXT_BINDING')
    targets={task:[t for t,g in zip(axes['canonical_ids'],axes['loss_groups']) if g==task] for task in ('pathway','state')}
    for task,ids in targets.items():
        if any(t not in policies or not policies[t].supervised or policies[t].loss_group!=task for t in ids):
            raise ValueError('RELATION_BASELINE_POLICY_ROLE')
    families=resolve_state_families(state_rows,targets['state'])
    if cfg is None:
        cfg={'seed':20261111,'baseline':{'C':[.01,.1,1.,10.,100.,1000.], 'max_iter':5000,
            'identity_encoding':ENCODING,'identity_mode':'numeric_only','batch_rows':32768,
            'max_identity_features':5000000,'max_working_gib':32.,'max_vocabulary_gib':2.,'max_available_memory_fraction':.65}}
    out.mkdir(parents=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','cancer':cancer,'policy':policy_binding,
        'state_definitions':state_binding,'components':{k:{'path':str(p),'sha256':sha(Path(p)/'COMPLETE.json')} for k,p in component_paths.items()},
        'config':cfg,'outer_fold':0,'validation_fold':1,'train_label_folds':[2,3,4],'sealed_test_opened':False})
    atomic_json(out/'AXES.json',{'targets':targets,'state_families':families,'feature_names':first['feature_names']})
    results={}
    for task in ('pathway','state'):
        if progress:progress({'stage':'fit_m0','cancer':cancer,'task':task})
        fam=families if task=='state' else None
        blocks={k:task_block(c,task,targets[task],fam) for k,c in components.items()}
        result=fit_strength_cancer([blocks[k] for k in CONTEXTS if k!='validation'],[blocks['validation']],
                                  targets[task],cfg,identity_mode='numeric_only',families=fam)
        head=result['heads']['strength']
        atomic_npz(out/(task+'_MODEL.npz'),**{k:head[k] for k in ('coef','intercept','numeric_mean','numeric_scale')})
        atomic_json(out/(task+'_FIT.json'),{k:v for k,v in head.items() if not isinstance(v,np.ndarray)})
        atomic_json(out/(task+'_VALIDATION.json'),result['exact_target_metrics'])
        for name,block in blocks.items():
            arrays={k:v for k,v in block.items() if isinstance(v,np.ndarray)}
            arrays['base']=predict_numeric_head(head,block['numeric'])
            atomic_npz(out/name/(task+'.npz'),**arrays)
        results[task]={'train_observed':head['train_rows'],'validation_observed':int(blocks['validation']['available'].sum()),
            'validation_macro_mse':head['validation_macro_mse'],'selection':head['selection'],
            'selected_C':head['selected_C'],'selected_alpha':head['selected_alpha'],
            'nonzero_numeric_coefficients':int(np.count_nonzero(head['coef']))}
        if progress:progress({'stage':'fit_m0_complete','cancer':cancer,'task':task,**results[task]})
    receipt={'format':'CLA_RELATION_BASELINE_COMPONENT_V1','target_host':'149','cancer':cancer,
        'tasks':results,'prepared_capsule':False,'hhgt_training_started':False,
        'code':{name:sha(Path(__file__).parent/name) for name in ('relation_baseline.py','m0_relation.py','exact_baseline.py')},
        'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    atomic_json(out/'COMPLETE.json',receipt)
    return receipt


def load_relation_baseline(path):
    root=Path(path)
    def read(name):return json.loads((root/name).read_text())
    receipt=read('COMPLETE.json');preflight=read('LAUNCH_PREFLIGHT.json');axes=read('AXES.json')
    if (receipt.get('format')!='CLA_RELATION_BASELINE_COMPONENT_V1' or receipt.get('target_host')!='149'
            or receipt.get('prepared_capsule') is not False or preflight.get('sealed_test_opened') is not False):
        raise ValueError('RELATION_BASELINE_FORMAT')
    for name,expected in receipt['files'].items():
        resolved=(root/name).resolve()
        if not resolved.is_relative_to(root.resolve()) or sha(resolved)!=expected:
            raise ValueError('RELATION_BASELINE_FILE_DRIFT')
    if set(preflight['components'])!=set(CONTEXTS):raise ValueError('RELATION_BASELINE_CONTEXT_GRID')
    result={name:{} for name in CONTEXTS}
    components={}
    for name,item in preflight['components'].items():
        if sha(Path(item['path'])/'COMPLETE.json')!=item['sha256']:
            raise ValueError('RELATION_BASELINE_SOURCE_DRIFT')
        c=load_relation_components(item['path']);components[name]=c
        if c['receipt']['cancer']!=receipt['cancer'] or c['receipt']['context']!=name:
            raise ValueError('RELATION_BASELINE_SOURCE_IDENTITY')
    policy=load_target_policy(verified_file(preflight['policy']))
    with verified_file(preflight['state_definitions']).open(encoding='utf-8-sig',newline='') as f:
        families=resolve_state_families(list(csv.DictReader(f,delimiter='\t')),axes['targets']['state'])
    if families!=axes['state_families']:raise ValueError('RELATION_BASELINE_STATE_FAMILY_DRIFT')
    for task,targets in axes['targets'].items():
        if task not in ('pathway','state') or any(not policy[t].supervised or policy[t].loss_group!=task for t in targets):
            raise ValueError('RELATION_BASELINE_TASK_ROLE_DRIFT')
        with np.load(root/(task+'_MODEL.npz'),allow_pickle=False) as f:head=dict(f)
        for name,c in components.items():
            canonical=[t for t,g in zip(c['axes']['canonical_ids'],c['axes']['loss_groups']) if g==task]
            if targets!=canonical or axes['feature_names']!=c['feature_names']:
                raise ValueError('RELATION_BASELINE_TARGET_AXIS_DRIFT')
            expected=task_block(c,task,targets,families if task=='state' else None)
            with np.load(root/name/(task+'.npz'),allow_pickle=False) as f:arrays=dict(f)
            for key,value in expected.items():
                if isinstance(value,np.ndarray) and not np.array_equal(value,arrays.get(key)):
                    raise ValueError('RELATION_BASELINE_TASK_ARRAY_DRIFT: '+key)
            if not np.allclose(arrays['base'],predict_numeric_head(head,arrays['numeric']),rtol=1e-6,atol=1e-7):
                raise ValueError('RELATION_BASELINE_PREDICTION_DRIFT')
            result[name][task]=arrays
    return {'receipt':receipt,'preflight':preflight,'axes':axes,'arrays':result,'components':components}
