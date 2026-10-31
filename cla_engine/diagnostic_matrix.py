"""Fixed-budget G0/G1/G2 joint-training diagnostic matrix.

This module is deliberately separate from formal admission. It never reads outer-test
labels and never changes the exact LASSO. Every graph variant uses the same joint
pathway + 0.5 * state objective, initialization, batches, and optimizer schedule.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from .common import atomic_json,atomic_npz,code_digest,commit_manifest,digest,fail,read_json,sha,status,verify_manifest
from .lineage import validate_prepared
from .model import JointHGT,loss_one,to_graph
from .runtime_guard import begin_attempt,register_worker_device,worker_status
from .train import aggregate_metrics,evaluate,load_checkpoint,load_npz,rng_state,save_checkpoint,seed_all,tensor_batch


FORMAT='CLA_R62_FIXED_216_STEP_G012_JOINT_DIAGNOSTIC_V1'
ARMS={
    'G0':{'variant':'G0','joint':True,'label':'G0_PATHWAY_PLUS_0.5_STATE'},
    'G1':{'variant':'G1','joint':True,'label':'G1_PATHWAY_PLUS_0.5_STATE'},
    'G2':{'variant':'G2','joint':True,'label':'G2_PATHWAY_PLUS_0.5_STATE'},
}
AUDIT_STEPS=(0,35,71,72,107,143,179,215)
FIXED_EPOCHS=(8,24)
STOP_REQUESTED=False


def _stop_handler(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def _ensure_149():
    if socket.gethostname()!='149':
        fail('WRONG_HOST','Diagnostic planning and reporting are authorized only on host 149',host=socket.gethostname())


def _array_digest(value):
    value=np.asarray(value)
    h=hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _state_dict_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        array=value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _schedule_digest(steps):
    h=hashlib.sha256()
    for row in steps:
        h.update(str(row['step']).encode())
        h.update(row['view'].encode())
        for task in ('pathway','state'):
            ids=np.asarray(row[task],dtype=np.int64)
            h.update(task.encode())
            h.update(str(len(ids)).encode())
            h.update(ids.tobytes())
    return h.hexdigest()


def build_sampling_schedule(prepared):
    """Reproduce the original NumPy draw order and persist both task batches."""
    prepared=Path(prepared)
    _,spec,cfg,_=validate_prepared(prepared)
    views=read_json(prepared/'VIEWS.json')
    trainviews=[view for view in views if view['split']=='train']
    steps_epoch=cfg['training']['steps_per_epoch'] or len(trainviews)
    total=steps_epoch*cfg['training']['epochs']
    if len(trainviews)!=9 or steps_epoch!=9 or total!=216:
        fail('DIAGNOSTIC_BUDGET_DRIFT','The controlled matrix requires nine train views and 24 x 9 = 216 steps',train_views=len(trainviews),steps_per_epoch=steps_epoch,total_steps=total)
    rng=np.random.RandomState(cfg['seed'])
    steps=[]
    rng_after={}
    for step in range(total):
        view=trainviews[step%len(trainviews)]
        root=prepared/'views'/view['id']
        row={'step':step,'view':view['id']}
        for task in ('pathway','state'):
            arr=load_npz(root/f'{task}.npz')
            valid=np.flatnonzero(arr['available'])
            if not len(valid):
                fail('TASK_MISSING_IN_VIEW','Both candidate schedules remain explicit in every arm',view=view['id'],task=task)
            row[task]=rng.choice(valid,size=min(cfg['training']['batch_size'],len(valid)),replace=False).astype(np.int64)
        steps.append(row)
        if step+1 in (72,216):
            rng_after[step+1]=copy.deepcopy(rng.get_state())
    return steps,rng_after,{'steps_per_epoch':steps_epoch,'total_steps':total,'train_views':[view['id'] for view in trainviews],'sha256':_schedule_digest(steps)}


def _write_schedule(out,steps,summary):
    arrays={}
    rows=[]
    for row in steps:
        item={'step':row['step'],'view':row['view']}
        for task,prefix in (('pathway','p'),('state','s')):
            key=f'{prefix}{row["step"]:03d}'
            values=np.asarray(row[task],dtype=np.int64)
            arrays[key]=values
            item[task]={'array_key':key,'rows':len(values),'sha256':_array_digest(values)}
        rows.append(item)
    atomic_npz(Path(out)/'SAMPLING_SCHEDULE.npz',**arrays)
    atomic_json(Path(out)/'SAMPLING_SCHEDULE.json',{**summary,'steps':rows,'candidate_indices_are_positions_within_prepared_task_arrays':True,'both_task_schedules_are_consumed_in_every_graph_variant':True})


def _compare_prepared(g0,g1,g2):
    roots={'G0':Path(g0),'G1':Path(g1),'G2':Path(g2)}
    validated={variant:validate_prepared(root) for variant,root in roots.items()}
    for variant,(manifest,spec,_,_) in validated.items():
        if manifest['metadata']['profile']!='diagnostic' or spec['fold']!=0:
            fail('DIAGNOSTIC_SCOPE_DRIFT','G0/G1/G2 comparison is restricted to diagnostic fold 0',variant=variant)
        if spec['graph_variant']!=variant:
            fail('GRAPH_VARIANT_MISMATCH','Prepared input is bound to the wrong graph variant',expected=variant,actual=spec['graph_variant'])
    _,reference_spec,reference_cfg,_=validated['G2']
    reference_views=read_json(roots['G2']/'VIEWS.json')
    for variant,(_,spec,cfg,_) in validated.items():
        if cfg.get('genomics',{}).get('enabled') and cfg['genomics'].get('context_graph_variant')!='G2':
            fail('CROSS_VARIANT_NUMERIC_BACKGROUND_NOT_FIXED','Topology comparison requires explicit G2 genomic background for every graph variant',variant=variant)
        for key in ('nodes','states','cancers','numeric_features','feature_dims','architecture'):
            if spec[key]!=reference_spec[key]:
                fail('CROSS_VARIANT_MODEL_INPUT_DRIFT','Common model input contract differs',variant=variant,field=key)
        if cfg['seed']!=reference_cfg['seed'] or cfg['model']!=reference_cfg['model'] or cfg['training']!=reference_cfg['training']:
            fail('CROSS_VARIANT_TRAINING_DRIFT','Seed/model/training must be identical',variant=variant)
        if read_json(roots[variant]/'VIEWS.json')!=reference_views:
            fail('CROSS_VARIANT_VIEW_DRIFT','View identities/order/splits differ',variant=variant)
    compared_arrays=0
    common_graph_arrays=0
    variant_graph_arrays=[]
    for view in reference_views:
        rel=Path('views')/view['id']
        for task in ('pathway','state'):
            arrays={variant:load_npz(root/rel/f'{task}.npz') for variant,root in roots.items()}
            if any(set(value)!=set(arrays['G2']) for value in arrays.values()):
                fail('CROSS_VARIANT_TASK_ARRAY_KEYS_DRIFT','Task array schemas differ',view=view['id'],task=task)
            for key in sorted(arrays['G2']):
                if any(not np.array_equal(arrays['G2'][key],arrays[variant][key],equal_nan=True) for variant in ('G0','G1')):
                    fail('CROSS_VARIANT_TASK_ARRAY_DRIFT','Candidates, labels, numeric inputs or exact LASSO logits differ',view=view['id'],task=task,array=key)
                compared_arrays+=1
        graphs={variant:load_npz(root/rel/'graph.npz') for variant,root in roots.items()}
        for left,right,allowed in (('G0','G1',('reported_binding',)),('G1','G2',('reported_PPI',)),('G0','G2',('reported_binding','reported_PPI'))):
            a=graphs[left];b=graphs[right]
            for key in sorted(set(a)&set(b)):
                if any(token in key for token in allowed):
                    variant_graph_arrays.append(f'{left}_vs_{right}:{view["id"]}:{key}')
                    continue
                if not np.array_equal(a[key],b[key],equal_nan=True):
                    fail('CROSS_VARIANT_COMMON_GRAPH_DRIFT','A non-variant graph input differs',pair=f'{left}_vs_{right}',view=view['id'],array=key)
                common_graph_arrays+=1
            differing=set(a)^set(b)
            if any(not any(token in key for token in allowed) for key in differing):
                fail('CROSS_VARIANT_GRAPH_SCHEMA_DRIFT','Only declared graph relations may differ',pair=f'{left}_vs_{right}',view=view['id'],differing=sorted(differing))
            variant_graph_arrays.extend(f'{left}_vs_{right}:{view["id"]}:{key}' for key in sorted(differing))
    return {
        'prepared_manifest_sha256':{variant:sha(root/'MANIFEST.json') for variant,root in roots.items()},
        'task_arrays_equal_including_candidates_labels_masks_weights_numeric_and_exact_lasso':True,
        'task_arrays_compared':compared_arrays,
        'common_non_variant_graph_arrays_equal':True,
        'common_graph_arrays_compared':common_graph_arrays,
        'variant_specific_graph_arrays':sorted(set(variant_graph_arrays)),
        'declared_graph_differences':{'G1_minus_G0':'reported_binding','G2_minus_G1':'reported_PPI'},
        'node_universe_equal':True,
    },reference_spec,reference_cfg


def create_plan(g0_prepared,g1_prepared,g2_prepared,out,source_step72_run=None):
    _ensure_149()
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        fail('OUTPUT_EXISTS','Diagnostic plan directory is immutable and must be new',path=str(out))
    out.mkdir(parents=True,exist_ok=True)
    parity,spec,cfg=_compare_prepared(g0_prepared,g1_prepared,g2_prepared)
    schedules={variant:build_sampling_schedule(path) for variant,path in {'G0':g0_prepared,'G1':g1_prepared,'G2':g2_prepared}.items()}
    if len({item[2]['sha256'] for item in schedules.values()})!=1:
        fail('CROSS_VARIANT_SCHEDULE_DRIFT','Identical candidate positions did not produce an identical schedule')
    schedule2,rng_after,summary2=schedules['G2']
    _write_schedule(out,schedule2,summary2)
    seed_all(cfg['seed'])
    model=JointHGT(spec,cfg['model'])
    initial_sha=_state_dict_digest(model.state_dict())
    del model
    resume_audit={'requested':bool(source_step72_run),'decision':'FRESH_START_REQUIRED_FOR_MATCHED_MATRIX'}
    if source_step72_run:
        run=Path(source_step72_run)
        receipt=read_json(run/'TRAINING_RECEIPT.json')
        record=receipt['last_checkpoint'];checkpoint=(run/record['file']).resolve()
        if not checkpoint.is_relative_to(run.resolve()) or not checkpoint.is_file() or sha(checkpoint)!=record['sha256']:
            fail('SOURCE_CHECKPOINT_INVALID','Step-72 source checkpoint is missing or changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected_completion={key:value for key,value in receipt.items() if key!='last_checkpoint'}
        if state.get('format')!='CLA_FULL_PYG_R62_FULL_TRAINING_STATE_V1' or state.get('next_step')!=72 or state.get('completion')!=expected_completion:
            fail('SOURCE_CHECKPOINT_NOT_COMPLETE_STEP72','Source must be the SHA-bound full optimizer state at step 72')
        old_rng=state['rng']['numpy'];want_rng=rng_after[72]
        rng_equal=(old_rng[0]==want_rng[0] and np.array_equal(old_rng[1],want_rng[1]) and old_rng[2:]==want_rng[2:])
        resume_audit={
            'requested':True,
            'source_run':str(run.resolve()),
            'source_receipt_sha256':sha(run/'TRAINING_RECEIPT.json'),
            'source_checkpoint_sha256':sha(checkpoint),
            'full_model_optimizer_scheduler_scaler_and_rng_present':all(key in state for key in ('model','optimizer','scheduler','scaler','rng')),
            'next_step':state['next_step'],
            'scheduler_last_epoch':state['scheduler'].get('last_epoch'),
            'scheduler_T_max':state['scheduler'].get('T_max'),
            'reconstructed_numpy_schedule_state_matches':rng_equal,
            'decision':'FRESH_START_REQUIRED_FOR_MATCHED_MATRIX',
            'decision_reasons':['HISTORICAL_EPOCHS_1_TO_7_LACK_FULL_ALPHA_GRID','HISTORICAL_RUN_LACKS_EXPLICIT_SAVED_BATCH_INDEX_SCHEDULE','THREE_VARIANTS_REQUIRE_ONE_CANONICAL_INSTRUMENTED_START'],
            'old_receipt_or_checkpoint_modified':False,
        }
        if not (rng_equal and state['scheduler'].get('last_epoch')==72 and state['scheduler'].get('T_max')==216):
            fail('SOURCE_STEP72_TRAJECTORY_DRIFT','The optional source is not an exact 72/216 continuation state')
    plan={
        'format':FORMAT,
        'host':'149',
        'created_unix':time.time(),
        'prepared':{variant:str(Path(path).resolve()) for variant,path in {'G0':g0_prepared,'G1':g1_prepared,'G2':g2_prepared}.items()},
        'prepared_manifest_sha256':parity['prepared_manifest_sha256'],
        'sampling_schedule_sha256':summary2['sha256'],
        'sampling_schedule_file_sha256':sha(out/'SAMPLING_SCHEDULE.npz'),
        'initial_model_state_sha256':initial_sha,
        'seed':cfg['seed'],
        'steps_per_epoch':9,
        'epochs':24,
        'total_steps':216,
        'alpha_grid':cfg['training']['alpha_grid'],
        'fixed_comparison_epochs':list(FIXED_EPOCHS),
        'gradient_audit_steps':list(AUDIT_STEPS),
        'arms':ARMS,
        'pathway_and_state_schedules_identical_across_all_graph_variants':True,
        'joint_objective_identical_across_all_graph_variants':True,
        'cosine_scheduler_T_max_unchanged':216,
        'performance_early_stopping_disabled':True,
        'safety_stops_retained':['DEADLINE','SIGNAL','NONFINITE_LOSS','NONFINITE_GRADIENT','OOM_PROCESS_FAILURE','SUPERVISOR_LEASE'],
        'outer_test_read':False,
        'formal_admission':False,
        'source_step72_reuse_audit':resume_audit,
        'code_sha256':code_digest(),
    }
    atomic_json(out/'CROSS_VARIANT_INPUT_AUDIT.json',parity)
    atomic_json(out/'DIAGNOSTIC_PLAN.json',plan)
    from .lineage import sign_file
    sign_file(out/'DIAGNOSTIC_PLAN.json',cfg,'DIAGNOSTIC_G012_JOINT_PLAN')
    commit_manifest(out,{'role':'FIXED_G012_JOINT_DIAGNOSTIC_PLAN','outer_test_read':False,'formal_admission':False})
    status(out,'DIAGNOSTIC_PLAN_READY_NOT_TRAINED',schedule_sha256=summary2['sha256'],initial_model_state_sha256=initial_sha)
    return plan


def _read_schedule(plan_root):
    plan_root=Path(plan_root)
    metadata=read_json(plan_root/'SAMPLING_SCHEDULE.json')
    with np.load(plan_root/'SAMPLING_SCHEDULE.npz',allow_pickle=False) as handle:
        steps=[]
        for row in metadata['steps']:
            item={'step':row['step'],'view':row['view']}
            for task in ('pathway','state'):
                values=handle[row[task]['array_key']].copy()
                if _array_digest(values)!=row[task]['sha256']:
                    fail('SAMPLING_SCHEDULE_ARRAY_DRIFT','Saved candidate positions changed',step=row['step'],task=task)
                item[task]=values
            steps.append(item)
    if _schedule_digest(steps)!=metadata['sha256']:
        fail('SAMPLING_SCHEDULE_DRIFT','Schedule digest changed')
    return steps,metadata


def _shared_named_parameters(model):
    prefixes=('joint.pathway_model.encoder.','context_projection.','sc_projection.','mutation_projection.')
    exact={'modal_gate','global_modal_gate'}
    selected=[(name,parameter) for name,parameter in model.named_parameters() if parameter.requires_grad and (name in exact or name.startswith(prefixes))]
    if not selected:
        fail('SHARED_PARAMETER_SET_EMPTY','Gradient conflict audit requires shared trainable parameters')
    return selected


def _gradient_geometry(model,pathway_loss,state_loss,step):
    named=_shared_named_parameters(model)
    parameters=[parameter for _,parameter in named]
    pathway=torch.autograd.grad(pathway_loss,parameters,retain_graph=True,allow_unused=True)
    state=torch.autograd.grad(.5*state_loss,parameters,retain_graph=True,allow_unused=True)
    device=pathway_loss.device
    dot=torch.zeros((),device=device,dtype=torch.float64)
    p2=torch.zeros_like(dot);s2=torch.zeros_like(dot)
    active_p=active_s=0
    for gp,gs in zip(pathway,state):
        if gp is not None:
            p2+=gp.detach().double().square().sum();active_p+=gp.numel()
        if gs is not None:
            s2+=gs.detach().double().square().sum();active_s+=gs.numel()
        if gp is not None and gs is not None:
            dot+=(gp.detach().double()*gs.detach().double()).sum()
    pnorm=float(p2.sqrt().cpu());snorm=float(s2.sqrt().cpu());dot_value=float(dot.cpu())
    cosine=dot_value/max(pnorm*snorm,1e-30)
    return {
        'step':int(step),
        'shared_parameter_tensors':len(named),
        'shared_parameter_names_sha256':digest([name for name,_ in named]),
        'pathway_weighted_gradient_norm':pnorm,
        'state_0.5_weighted_gradient_norm':snorm,
        'state_to_pathway_weighted_norm_ratio':snorm/max(pnorm,1e-30),
        'weighted_gradient_dot_product':dot_value,
        'weighted_gradient_cosine':cosine,
        'gradient_conflict':dot_value<0,
        'pathway_active_gradient_elements':active_p,
        'state_active_gradient_elements':active_s,
        'direction_loss_is_included_in_each_task_loss':True,
    }


def _epoch_evaluation(model,prepared,views,spec,cfg,device,batch_size,epoch,out,arm):
    _,raw=evaluate(model,prepared,views,spec,device,batch_size,alpha=1.)
    alpha_rows=[]
    for alpha in cfg['training']['alpha_grid']:
        mixed={task:[(cancer,arr,arr['base']+alpha*(score-arr['base']),direction) for cancer,arr,score,direction in items]
               for task,items in raw.items()}
        metrics=aggregate_metrics(mixed,spec)
        alpha_rows.append({'alpha':float(alpha),'metrics':metrics})
    selected=max(alpha_rows,key=lambda row:((row['metrics']['selection_min_task_delta'],row['metrics']['selection_weighted_ap']),-row['alpha']))
    residual={}
    for task,items in raw.items():
        values=np.concatenate([score-arr['base'] for _,arr,score,_ in items])
        residual[task]={'mean_abs':float(np.mean(np.abs(values))),'max_abs':float(np.max(np.abs(values))),'rms':float(np.sqrt(np.mean(values.astype(np.float64)**2)))}
        if epoch in FIXED_EPOCHS:
            for cancer,arr,score,direction in items:
                atomic_npz(Path(out)/'fixed_predictions'/f'epoch_{epoch:02d}'/f'{cancer}__{task}.npz',lnc=arr['lnc'],target=arr['target'],alpha1_score=score,baseline=arr['base'],direction_logit=direction,available=arr['available'])
    return {
        'epoch':int(epoch),
        'step':int(epoch*9),
        'alpha_grid':alpha_rows,
        'historical_rule_selection':{'alpha':selected['alpha'],'metrics':selected['metrics']},
        'residual':residual,
        'torch_rng_sha256':_array_digest(torch.get_rng_state().cpu().numpy()),
        'cuda_rng_sha256':[_array_digest(value.cpu().numpy()) for value in (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])],
    }


def arm_signature(prepared,plan_root,arm,cfg,spec):
    return digest({'format':FORMAT,'arm':arm,'prepared_manifest_sha256':sha(Path(prepared)/'MANIFEST.json'),'plan_manifest_sha256':sha(Path(plan_root)/'MANIFEST.json'),'model':cfg['model'],'training':cfg['training'],'architecture':spec['architecture'],'code_sha256':code_digest()})


def run_arm(prepared,plan_root,arm,out,device='cuda',deadline=None,lease_required=False):
    global STOP_REQUESTED
    STOP_REQUESTED=False
    if arm not in ARMS:
        fail('DIAGNOSTIC_ARM_INVALID','Expected one of G0/G1/G2',arm=arm)
    prepared=Path(prepared);plan_root=Path(plan_root);out=Path(out)
    verify_manifest(plan_root)
    plan=read_json(plan_root/'DIAGNOSTIC_PLAN.json')
    manifest,spec,cfg,lineage=validate_prepared(prepared)
    contract=ARMS[arm]
    if plan.get('format')!=FORMAT or plan['arms'][arm]!=contract or spec['graph_variant']!=contract['variant']:
        fail('DIAGNOSTIC_PLAN_BINDING_MISMATCH','Arm, graph variant and signed plan differ')
    if sha(prepared/'MANIFEST.json')!=plan['prepared_manifest_sha256'][contract['variant']]:
        fail('DIAGNOSTIC_PREPARED_DRIFT','Prepared manifest differs from the 149 plan')
    if manifest['metadata']['profile']!='diagnostic' or plan['total_steps']!=216 or plan['steps_per_epoch']!=9:
        fail('DIAGNOSTIC_SCOPE_DRIFT','Only the fixed three-cancer 216-step matrix is allowed')
    if device.startswith('cuda') and not torch.cuda.is_available():
        fail('CUDA_UNAVAILABLE','Diagnostic arms do not silently fall back to CPU')
    if deadline is None or deadline<=time.time():
        fail('BUDGET_NOT_BOUND','Diagnostic worker requires an active absolute deadline')
    steps,schedule_meta=_read_schedule(plan_root)
    if schedule_meta['sha256']!=plan['sampling_schedule_sha256']:
        fail('SAMPLING_SCHEDULE_PLAN_DRIFT','Schedule is not the one committed by 149')
    out.mkdir(parents=True,exist_ok=True)
    signature=arm_signature(prepared,plan_root,arm,cfg,spec)
    attempt_id=begin_attempt(out)
    seed_all(cfg['seed']);torch.set_num_threads(cfg['training'].get('cpu_threads',2));register_worker_device(out,device)
    model=JointHGT(spec,cfg['model']).to(device)
    if _state_dict_digest(model.state_dict())!=plan['initial_model_state_sha256']:
        fail('INITIAL_MODEL_STATE_DRIFT','All three graph variants require byte-identical initialized tensors')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['model']['learning_rate'],weight_decay=cfg['model']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=216)
    amp_enabled=device.startswith('cuda') and cfg['training']['amp']
    amp_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled and amp_dtype==torch.float16)
    environment={'torch':str(torch.__version__),'numpy':np.__version__,'sklearn':sklearn.__version__,'python':'.'.join(map(str,sys.version_info[:2])),'device_type':device.split(':')[0],'torch_geometric':__import__('torch_geometric').__version__}
    current,record=load_checkpoint(out,signature)
    if current:
        if current.get('environment')!=environment:
            fail('RESUME_ENVIRONMENT_DRIFT','Diagnostic continuation requires the same backend and software versions',expected=current.get('environment'),actual=environment)
        model.load_state_dict(current['model']);optimizer.load_state_dict(current['optimizer']);scheduler.load_state_dict(current['scheduler']);scaler.load_state_dict(current['scaler'])
        random.setstate(current['rng']['python']);np.random.set_state(current['rng']['numpy']);torch.set_rng_state(current['rng']['torch'])
        if torch.cuda.is_available() and current['rng'].get('cuda') is not None:torch.cuda.set_rng_state_all(current['rng']['cuda'])
        start=current['next_step'];history=current['history'];gradient_audit=current['gradient_audit'];old_best=current['old_best']
    else:
        start=0;history=[];gradient_audit=[];old_best=None
    views=read_json(prepared/'VIEWS.json');trainviews={view['id']:view for view in views if view['split']=='train'}
    batch_size=cfg['training']['batch_size']
    atomic_json(out/'RUN_CONTRACT.json',{'format':FORMAT,'arm':arm,'arm_contract':contract,'signature':signature,'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'plan_manifest_sha256':sha(plan_root/'MANIFEST.json'),'schedule_sha256':schedule_meta['sha256'],'initial_model_state_sha256':plan['initial_model_state_sha256'],'start_step':start,'total_steps':216,'epochs':24,'steps_per_epoch':9,'alpha_grid':cfg['training']['alpha_grid'],'performance_early_stopping_disabled':True,'outer_test_read':False,'formal_admission':False,'deadline_unix':deadline})
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,_stop_handler)
    cache={}
    def checkpoint(next_step):
        return save_checkpoint(out,{'format':'CLA_R62_DIAGNOSTIC_ARM_STATE_V1','signature':signature,'next_step':next_step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'history':history,'gradient_audit':gradient_audit,'old_best':old_best,'arm':arm,'schedule_sha256':schedule_meta['sha256'],'environment':environment})
    if record is None:
        record=checkpoint(0)
    for step in range(start,216):
        lease_expired=False
        if lease_required:
            lease=out/'SUPERVISOR_LEASE.json'
            lease_expired=(not lease.exists()) or read_json(lease).get('expires_unix',0)<=time.time() or read_json(lease).get('attempt_id')!=attempt_id
        if STOP_REQUESTED or lease_expired or time.time()>=deadline:
            record=checkpoint(step)
            return worker_status(out,'PAUSED_DIAGNOSTIC_BUDGET_OR_SIGNAL',arm=arm,next_step=step,checkpoint=record)
        scheduled=steps[step];view=trainviews[scheduled['view']];root=prepared/'views'/view['id']
        if cache.get('id')!=view['id']:
            cache={'id':view['id'],'graph':load_npz(root/'graph.npz'),'arrays':{task:load_npz(root/f'{task}.npz') for task in ('pathway','state')}}
        batches={task:tensor_batch(arr,scheduled[task],device) for task,arr in cache['arrays'].items()}
        model.train();optimizer.zero_grad(set_to_none=True);graph=to_graph(cache['graph'],device)
        with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',enabled=amp_enabled,dtype=amp_dtype):
            encoded=model.encode(graph,view['cancer_index'])
            outputs={task:model.forward_task(task,encoded,batches[task]) for task in ('pathway','state')}
            losses={task:loss_one(outputs[task],batches[task],cfg['model']['direction_weight']) for task in ('pathway','state')}
            if any(value is None for value in losses.values()):
                fail('TASK_NO_LOSS','Both task batches must remain evaluable in every graph variant')
            objective=losses['pathway']+.5*losses['state']+model.regularizer()
        if not torch.isfinite(objective):
            fail('NONFINITE_LOSS','No retry or imputed success')
        if step in AUDIT_STEPS:
            audit=_gradient_geometry(model,losses['pathway'],losses['state'],step)
            audit['arm']=arm;audit['training_objective_includes_state']=True;gradient_audit.append(audit)
            atomic_json(out/'GRADIENT_GEOMETRY.json',gradient_audit)
        scaler.scale(objective).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        from cc_hhgt.v32.optimizer_guard_v2 import optimizer_step_with_global_delta_guard
        class _ScaledOptimizerStep:
            def step(self):scaler.step(optimizer)
        guard=optimizer_step_with_global_delta_guard(model,_ScaledOptimizerStep(),torch=torch,objective_value=float(objective.detach().cpu()))
        scaler.update();scheduler.step();step_done=step+1
        atomic_json(out/'LAST_OPTIMIZER_GUARD.json',guard)
        worker_status(out,'DIAGNOSTIC_ARM_TRAINING',arm=arm,next_step=step_done,optimizer_steps_this_attempt=step_done-start,total_steps=216,view=view['id'],pathway_loss=float(losses['pathway'].detach().cpu()),state_loss=float(losses['state'].detach().cpu()),state_loss_in_objective=True)
        del encoded,graph,outputs,losses,objective
        if step_done%9==0:
            epoch=step_done//9
            worker_status(out,'DIAGNOSTIC_ARM_VALIDATING',arm=arm,epoch=epoch,next_step=step_done)
            row=_epoch_evaluation(model,prepared,views,spec,cfg,device,batch_size,epoch,out,arm)
            history.append(row)
            selected=row['historical_rule_selection'];rank=(selected['metrics']['selection_min_task_delta'],selected['metrics']['selection_weighted_ap'])
            if old_best is None or rank[0]>old_best['rank'][0]+1e-8 or (abs(rank[0]-old_best['rank'][0])<=1e-8 and rank[1]>old_best['rank'][1]+1e-8):
                old_best={'rank':rank,'step':step_done,'epoch':epoch,'alpha':selected['alpha'],'metrics':selected['metrics']}
            atomic_json(out/'EPOCH_HISTORY.json',history)
            record=checkpoint(step_done)
    if len(history)!=24 or history[-1]['step']!=216:
        fail('DIAGNOSTIC_ARM_INCOMPLETE','Fixed training budget was not completed')
    receipt={'format':FORMAT,'state':'DIAGNOSTIC_ARM_COMPLETE_NOT_FORMAL','arm':arm,'arm_contract':contract,'signature':signature,'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'plan_manifest_sha256':sha(plan_root/'MANIFEST.json'),'lineage':lineage,'initial_model_state_sha256':plan['initial_model_state_sha256'],'sampling_schedule_sha256':schedule_meta['sha256'],'optimizer_steps_total':216,'epochs_completed':24,'performance_early_stopping_used':False,'alpha_grid_fully_recorded_each_epoch':True,'fixed_comparison_epochs':list(FIXED_EPOCHS),'historical_selection_rule_result':old_best,'gradient_geometry':gradient_audit,'epoch_history_sha256':sha(out/'EPOCH_HISTORY.json'),'outer_test_read':False,'formal_admission':False,'old_no_go_modified':False,'attempt_id':attempt_id,'environment':environment,'last_checkpoint':record}
    atomic_json(out/'DIAGNOSTIC_ARM_RECEIPT.json',receipt)
    commit_manifest(out,{'role':'DEVELOPMENT_DIAGNOSTIC_ARM','arm':arm,'outer_test_read':False,'formal_admission':False},name='RESULT_MANIFEST.json')
    return worker_status(out,'DIAGNOSTIC_ARM_COMPLETE_NOT_FORMAL',arm=arm,next_step=216,receipt_sha256=sha(out/'DIAGNOSTIC_ARM_RECEIPT.json'))


def summarize(plan_root,runs,out):
    _ensure_149()
    plan_root=Path(plan_root);verify_manifest(plan_root);plan=read_json(plan_root/'DIAGNOSTIC_PLAN.json')
    if set(runs)!=set(ARMS):
        fail('DIAGNOSTIC_ARM_SET_INCOMPLETE','Exactly G0/G1/G2 are required')
    receipts={}
    histories={}
    for arm,path in runs.items():
        root=Path(path);verify_manifest(root,name='RESULT_MANIFEST.json')
        receipt=read_json(root/'DIAGNOSTIC_ARM_RECEIPT.json')
        if receipt.get('state')!='DIAGNOSTIC_ARM_COMPLETE_NOT_FORMAL' or receipt.get('arm')!=arm or receipt.get('plan_manifest_sha256')!=sha(plan_root/'MANIFEST.json'):
            fail('DIAGNOSTIC_ARM_RECEIPT_INVALID','Arm result is incomplete or belongs to another plan',arm=arm)
        receipts[arm]=receipt;histories[arm]=read_json(root/'EPOCH_HISTORY.json')
    rows=[]
    for epoch in FIXED_EPOCHS:
        for alpha in plan['alpha_grid']:
            values={}
            for arm in ARMS:
                epoch_row=next(row for row in histories[arm] if row['epoch']==epoch)
                values[arm]=next(row['metrics'] for row in epoch_row['alpha_grid'] if row['alpha']==alpha)
            rows.append({'epoch':epoch,'alpha':alpha,
                'pathway_macro_ap':{arm:values[arm]['pathway']['macro_ap'] for arm in ARMS},
                'pathway_delta':{arm:values[arm]['pathway']['delta'] for arm in ARMS},
                'G1_minus_G0_pathway_ap':values['G1']['pathway']['macro_ap']-values['G0']['pathway']['macro_ap'],
                'G2_minus_G1_pathway_ap':values['G2']['pathway']['macro_ap']-values['G1']['pathway']['macro_ap'],
                'G2_minus_G0_pathway_ap':values['G2']['pathway']['macro_ap']-values['G0']['pathway']['macro_ap'],
                'state_macro_ap':{arm:values[arm]['state']['macro_ap'] for arm in ARMS},
                'weighted_ap':{arm:values[arm]['selection_weighted_ap'] for arm in ARMS}})
    gradient_summary={arm:{'negative_dot_steps':sum(row['gradient_conflict'] for row in receipts[arm]['gradient_geometry']),'audited_steps':len(receipts[arm]['gradient_geometry']),'rows':receipts[arm]['gradient_geometry']} for arm in ARMS}
    report={'format':FORMAT,'state':'DIAGNOSTIC_MATRIX_COMPLETE_NOT_FORMAL','plan_manifest_sha256':sha(plan_root/'MANIFEST.json'),'fixed_epoch_alpha_comparisons':rows,'historical_selection_rule_results':{arm:receipts[arm]['historical_selection_rule_result'] for arm in ARMS},'gradient_geometry':gradient_summary,'interpretation_contract':{'training_sufficiency':'COMPARE_EPOCH24_VS_EPOCH8_WITHIN_EACH_VARIANT_AT_MATCHED_ALPHA','reported_binding_contribution':'USE_G1_MINUS_G0_AT_MATCHED_EPOCH_ALPHA','ppi_contribution':'USE_G2_MINUS_G1_AT_MATCHED_EPOCH_ALPHA','joint_training_objective':'PATHWAY_PLUS_0.5_STATE_FOR_ALL_VARIANTS','scope':'THREE_CANCERS_FOLD0_DEVELOPMENT_ONLY','formal_conclusion':False},'outer_test_read':False,'old_no_go_modified':False}
    out=Path(out)
    if out.exists() and any(out.iterdir()):fail('OUTPUT_EXISTS','Summary directory must be new',path=str(out))
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'DIAGNOSTIC_MATRIX_REPORT.json',report);commit_manifest(out,{'role':'G012_JOINT_DIAGNOSTIC_SUMMARY','outer_test_read':False,'formal_admission':False});status(out,'DIAGNOSTIC_MATRIX_COMPLETE_NOT_FORMAL',report_sha256=sha(out/'DIAGNOSTIC_MATRIX_REPORT.json'))
    return report
