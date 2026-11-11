"""Resumable bounded worker. Both task losses update one encoder; no sealed test access."""
from __future__ import annotations
import copy,json,math,os,random,signal,time,uuid,sys,sklearn
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .common import read_json,atomic_json,atomic_npz,verify_manifest,sha,digest,fail,status,lock,fsync_dir,code_digest
from .model import JointHGT,to_graph,loss_one
from .lineage import validate_prepared,completion_state,verify_completion,expected_predictions
from .runtime_guard import begin_attempt,worker_status,register_worker_device

STOP_REQUESTED=False

def stop_handler(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and x.get('cuda') is not None:torch.cuda.set_rng_state_all(x['cuda'])

def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def tensor_batch(arr,idx,device):
    return {k:torch.as_tensor(v[idx],device=device,dtype=torch.long if k in ['lnc','target'] else (torch.bool if v.dtype==bool else torch.float32)) for k,v in arr.items() if k in ['lnc','target','numeric','base','label','available','weight','direction_label','direction_available']}

def model_signature(prepared,cfg,spec):
    return digest({'prepared_manifest_sha256':sha(Path(prepared)/'MANIFEST.json'),'model':cfg['model'],'runtime':cfg['training'],'backbone':'ORIGINAL_PYG_CC_HHGT','seed':cfg['seed'],'architecture':spec['architecture'],'code_sha256':code_digest()})

def save_checkpoint(out,state):
    """Immutable generation + SHA ready marker, then atomic pointer. Keep two local generations."""
    out=Path(out);d=out/'checkpoints';d.mkdir(parents=True,exist_ok=True)
    name=f"step_{state['next_step']:09d}_{uuid.uuid4().hex[:10]}.pt";tmp=d/(name+'.partial');path=d/name
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fsync_dir(d);record={'file':f'checkpoints/{name}','sha256':sha(path),'bytes':path.stat().st_size,'next_step':state['next_step'],'signature':state['signature'],'created_unix':time.time()}
    atomic_json(d/(name+'.ready.json'),record);atomic_json(out/'LATEST.json',record)
    # Controller acknowledgement determines deletability; never discard unmirrored generations.
    ack=out/'PERSISTED_ACK.json'
    if ack.exists():
        acknowledged=read_json(ack).get('next_step',-1)
        ready=sorted(d.glob('*.ready.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for rp in ready[2:]:
            rr=read_json(rp)
            if rr['next_step']<=acknowledged:
                (out/rr['file']).unlink(missing_ok=True);rp.unlink()
    return record

def load_checkpoint(out,signature):
    candidates=[];latest=Path(out)/'LATEST.json'
    if latest.exists():
        try:candidates.append(read_json(latest))
        except (ValueError,OSError):pass
    for p in sorted((Path(out)/'checkpoints').glob('*.ready.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        try:candidates.append(read_json(p))
        except (ValueError,OSError):continue
    tried=[]
    for record in candidates:
        p=(Path(out)/record['file']).resolve()
        if not p.is_relative_to(Path(out).resolve()):fail('PATH_TRAVERSAL','Checkpoint path outside run')
        if record.get('signature')!=signature:fail('CHECKPOINT_CONTRACT_DRIFT','Checkpoint belongs to different inputs/config')
        if not p.exists() or sha(p)!=record['sha256']:tried.append(record['file']);continue
        # Trusted local project checkpoint only, verified SHA immediately before unpickling.
        value=torch.load(p,map_location='cpu',weights_only=False)
        if value.get('signature')!=signature:fail('CHECKPOINT_CONTRACT_DRIFT','Checkpoint inner signature mismatch')
        return value,record
    if candidates:fail('NO_VALID_CHECKPOINT','All committed checkpoints failed SHA; partial files ignored',tried=tried)
    return None,None

def aggregate_metrics(predictions,spec):
    if spec.get('relation_objective')=='observed_fused_rms_effect':
        from .relation_metrics import paired_relation_strength_metrics
        metrics={}
        for task in ('pathway','state'):
            row=[];targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
            for cancer,arr,score,direction in predictions[task]:
                result=paired_relation_strength_metrics(arr['label'],score,arr['base'],arr['available'],arr['target'],
                    task=task,targets=targets,families=spec.get('state_families'))
                result['cancer_id']=cancer;row.append(result)
            good=[item for item in row if item['available']]
            if not good:fail('VALIDATION_NOT_ESTIMABLE','No evaluable cancer',task=task)
            metrics[task]={'macro_mse':float(np.mean([item['mse'] for item in good])),
                'baseline_macro_mse':float(np.mean([item['baseline_mse'] for item in good])),
                'macro_mae':float(np.mean([item['mae'] for item in good])),
                'baseline_macro_mae':float(np.mean([item['baseline_mae'] for item in good])),
                'cancers':row,'metrics_contract':spec.get('metrics_contract')}
            metrics[task]['delta']=metrics[task]['baseline_macro_mse']-metrics[task]['macro_mse']
        metrics['selection_min_task_delta']=min(metrics[t]['delta'] for t in ('pathway','state'))
        metrics['selection_weighted_score']=-(metrics['pathway']['macro_mse']+.5*metrics['state']['macro_mse'])/1.5
        return metrics
    from .metrics import paired_task_metrics
    metrics={}
    for task in ('pathway','state'):
        row=[];targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
        for cancer,arr,score,direction in predictions[task]:
            result=paired_task_metrics(arr['label'],score,arr['base'],arr['available'],arr['target'],
                task=task,targets=targets,families=spec.get('state_families'),k=int(spec.get('precision_k',100)))
            result['cancer_id']=cancer
            if result['available']:result['lasso_auprc']=result['baseline_auprc']
            row.append(result)
        good=[r for r in row if r['available']]
        if not good:fail('VALIDATION_NOT_ESTIMABLE','No evaluable cancer',task=task)
        metrics[task]={'macro_ap':float(np.mean([r['auprc'] for r in good])),
            'baseline_macro_ap':float(np.mean([r['lasso_auprc'] for r in good])),
            'macro_precision_at_k':float(np.mean([r['precision_at_k'] for r in good])),
            'baseline_macro_precision_at_k':float(np.mean([r['baseline_precision_at_k'] for r in good])),
            'cancers':row,'metrics_contract':spec.get('metrics_contract')}
        metrics[task]['delta']=metrics[task]['macro_ap']-metrics[task]['baseline_macro_ap']
    metrics['selection_min_task_delta']=min(metrics[t]['delta'] for t in ('pathway','state'))
    metrics['selection_weighted_ap']=(metrics['pathway']['macro_ap']+.5*metrics['state']['macro_ap'])/1.5
    metrics['selection_weighted_score']=metrics['selection_weighted_ap']
    return metrics

def should_stop_early(profile,cfg,step,steps_epoch,stale):
    """Keep diagnostic admission runs on their complete declared step budget."""
    if profile=='diagnostic' and bool(cfg['training'].get('diagnostic_full_epochs',False)):
        return False
    return step>=cfg['training']['min_epochs']*steps_epoch and stale>=cfg['training']['patience']

@torch.no_grad()
def evaluate(model,prepared,views,spec,device,batch_size,alpha=1.,disabled=frozenset()):
    model.eval();pred={t:[] for t in ['pathway','state']}
    for view in views:
        if view['split']!='validation':continue
        root=Path(prepared)/'views'/view['id'];graph=to_graph(load_npz(root/'graph.npz'),device);enc=model.encode(graph,view['cancer_index'],disabled)
        for t in pred:
            arr=load_npz(root/f'{t}.npz');score=[];directions=[]
            for start in range(0,len(arr['lnc']),batch_size):
                batch=tensor_batch(arr,slice(start,start+batch_size),device);o=model.forward_task(t,enc,batch,alpha);score.append(o['logit'].float().cpu().numpy());directions.append(o['direction'].float().cpu().numpy())
            pred[t].append((spec['cancers'][view['cancer_index']],arr,np.concatenate(score),np.concatenate(directions)))
        del enc,graph
    return aggregate_metrics(pred,spec),pred

def train(prepared,out,*,device='cuda',deadline=None,resume=True,interrupt_after=None,unsafe_local_debug=False,
          lease_required=False,formal_plan=None,formal_job_id=None,formal_config=None):
    global STOP_REQUESTED
    STOP_REQUESTED=False;prepared=Path(prepared);out=Path(out)
    from .local_execution import require_host_149
    require_host_149()
    manifest,spec,cfg,lineage=validate_prepared(prepared);profile=manifest['metadata']['profile']
    formal_authorization=None
    if profile=='formal':
        if not formal_plan or not formal_job_id:
            fail('FORMAL_PLAN_REQUIRED','Formal worker requires a signed plan and exact job ID')
        if formal_config is None:
            fail('FORMAL_CONFIG_REQUIRED','Formal worker requires the signed plan base configuration')
        from .local_execution import validate_formal_job
        formal_authorization=validate_formal_job(prepared,formal_plan,formal_job_id,formal_config)
    views=read_json(prepared/'VIEWS.json')
    if cfg['model']['task_weights']!={'pathway':1.,'state':.5}:fail('TASK_WEIGHT_DRIFT','State is 0.5; Pathway is 1')
    if unsafe_local_debug and not cfg.get('synthetic_fixture',False):fail('REAL_DATA_DEBUG_BYPASS_FORBIDDEN','Budget-free debug applies only to synthetic fixtures')
    if device.startswith('cuda') and not torch.cuda.is_available():fail('CUDA_UNAVAILABLE','Do not fall back silently to CPU paid training')
    if deadline is None and not unsafe_local_debug:fail('BUDGET_NOT_BOUND','Worker requires controller absolute deadline')
    if not unsafe_local_debug and deadline<=time.time():fail('BUDGET_EXHAUSTED','Absolute deadline has passed; restart cannot reset it')
    signature=model_signature(prepared,cfg,spec);out.mkdir(parents=True,exist_ok=True)
    with lock(out):
        if (out/'TRAINING_RECEIPT.json').exists():
            done,_=verify_completion(prepared,out)
            return status(out,done['state'],message='Verified completed run; no optimizer steps repeated',signature=signature)
        attempt_id=begin_attempt(out)
        seed_all(cfg['seed']);torch.set_num_threads(cfg['training'].get('cpu_threads',2));register_worker_device(out,device)
        model=JointHGT(spec,cfg['model']).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['model']['learning_rate'],weight_decay=cfg['model']['weight_decay'])
        trainviews=[];deferred=[]
        for v in views:
            if v['split']!='train':continue
            counts={t:int(load_npz(prepared/'views'/v['id']/f'{t}.npz')['available'].sum()) for t in ['pathway','state']}
            if all(counts.values()):trainviews.append(v)
            else:deferred.append({'view':v['id'],'counts':counts,'reason':'INSUFFICIENT_PAIRED_SUPERVISION_NOT_NEGATIVE'})
        atomic_json(out/'DEFERRED_TRAIN_VIEWS.json',deferred)
        if not trainviews:fail('TASK_NOT_ESTIMABLE','No context has both tasks')
        batchsize=cfg['training']['batch_size'];steps_epoch=cfg['training']['steps_per_epoch'] or len(trainviews);total=steps_epoch*cfg['training']['epochs']
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(total,1));amp_enabled=device.startswith('cuda') and cfg['training']['amp'];amp_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled and amp_dtype==torch.float16)
        current,record=load_checkpoint(out,signature) if resume else (None,None)
        if not resume and (out/'LATEST.json').exists():fail('OVERWRITE_FORBIDDEN','Existing run requires resume')
        environment={'torch':str(torch.__version__),'numpy':np.__version__,'sklearn':sklearn.__version__,'python':'.'.join(map(str,sys.version_info[:2])),'device_type':device.split(':')[0],'torch_geometric':__import__('torch_geometric').__version__}
        if current:
            if current.get('environment')!=environment:fail('RESUME_ENVIRONMENT_DRIFT','Use the same software versions/backend for exact continuation',expected=current.get('environment'),actual=environment)
            model.load_state_dict(current['model']);optimizer.load_state_dict(current['optimizer']);scheduler.load_state_dict(current['scheduler']);scaler.load_state_dict(current['scaler']);restore_rng(current['rng']);start=current['next_step'];best=current['best'];history=current['history'];stale=current['stale'];gradproof=current['gradient_proof']
        else:
            start=0;history=[];stale=0;gradproof={'pathway_max_shared_grad':0.,'state_max_shared_grad':0.}
            metrics,_=evaluate(model,prepared,views,spec,device,batchsize,alpha=0.)
            best={'rank':(0.,metrics['selection_weighted_score']),'alpha':0.,'step':0,'metrics':metrics,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        diagnostic_full_epochs=profile=='diagnostic' and bool(cfg['training'].get('diagnostic_full_epochs',False))
        atomic_json(out/'RUN_CONTRACT.json',{'signature':signature,'prepared_sha256':sha(prepared/'MANIFEST.json'),'prepared_path':str(prepared.resolve()),'task_weights':cfg['model']['task_weights'],'deadline_unix':deadline,'optimizer':'AdamW','resume_from_step':start,'configured_total_steps':total,'steps_per_epoch':steps_epoch,'diagnostic_full_epochs':diagnostic_full_epochs,'no_test_labels':True,'gpu_bitwise_reproducibility_claimed':False,'formal_authorization':formal_authorization})
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop_handler)
        last_save=time.time();cache={};finished=False;budget_stop=False;stop_reason=None
        def checkpoint(next_step,completion=None):
            return save_checkpoint(out,{'format':'CLA_FULL_PYG_R62_FULL_TRAINING_STATE_V1','signature':signature,'next_step':next_step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'best':best,'history':history,'stale':stale,'gradient_proof':gradproof,'torch_version':str(torch.__version__),'environment':environment,'completion':completion})
        if current is None:record=checkpoint(0)
        if start>=total:finished=True;stop_reason='FULL_EPOCH_BUDGET'
        elif should_stop_early(profile,cfg,start,steps_epoch,stale):finished=True;stop_reason='EARLY_STOPPING'
        for step in range(start,total if not finished else start):
            lease_expired=False
            if lease_required and profile!='formal':
                lp=out/'SUPERVISOR_LEASE.json'
                lease_expired=(not lp.exists()) or read_json(lp).get('expires_unix',0)<=time.time() or read_json(lp).get('attempt_id')!=attempt_id
            if STOP_REQUESTED or lease_expired or (deadline is not None and time.time()>=deadline):budget_stop=True;record=checkpoint(step);break
            vi=step%len(trainviews);view=trainviews[vi];root=prepared/'views'/view['id']
            if not cache or cache.get('id')!=view['id']:
                cache={'id':view['id'],'graph':load_npz(root/'graph.npz'),'arrays':{t:load_npz(root/f'{t}.npz') for t in ['pathway','state']}}
            batches={}
            for t,arr in cache['arrays'].items():
                valid=np.flatnonzero(arr['available'])
                if not len(valid):fail('TASK_MISSING_IN_VIEW','Do not train a Pathway-only replacement',view=view['id'],task=t)
                ids=np.random.choice(valid,size=min(batchsize,len(valid)),replace=False);batches[t]=tensor_batch(arr,ids,device)
            model.train();optimizer.zero_grad(set_to_none=True);graph=to_graph(cache['graph'],device)
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',enabled=amp_enabled,dtype=amp_dtype):
                enc=model.encode(graph,view['cancer_index']);outputs={t:model.forward_task(t,enc,batches[t]) for t in batches};losses={t:loss_one(outputs[t],batches[t],cfg['model']['direction_weight'],spec.get('relation_objective','binary')) for t in batches}
                if any(v is None for v in losses.values()):fail('TASK_NO_LOSS','Both tasks are mandatory')
                loss=losses['pathway']+.5*losses['state']+model.regularizer()
            if not torch.isfinite(loss):fail('NONFINITE_LOSS','No retry or imputed success')
            # Real shared-HGT gradient proof, not merely a changed private head.
            if step<3:
                parameters=list(model.layers[0].parameters())
                for t in losses:
                    gg=torch.autograd.grad(losses[t],parameters,retain_graph=True,allow_unused=True)
                    norm=sum(float(g.detach().float().square().sum().cpu()) for g in gg if g is not None)**.5
                    gradproof[t+'_max_shared_grad']=max(gradproof[t+'_max_shared_grad'],norm)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            from cc_hhgt.v32.optimizer_guard_v2 import optimizer_step_with_global_delta_guard
            class _ScaledOptimizerStep:
                def step(self):scaler.step(optimizer)
            guard=optimizer_step_with_global_delta_guard(model,_ScaledOptimizerStep(),torch=torch,objective_value=float(loss.detach().cpu()))
            scaler.update();scheduler.step()
            atomic_json(out/'LAST_OPTIMIZER_GUARD.json',guard)
            stepdone=step+1
            worker_status(out,'TRAINING',next_step=stepdone,optimizer_steps_this_attempt=stepdone-start,total_steps=total,view=view['id'],pathway_loss=float(losses['pathway'].detach().cpu()),state_loss=float(losses['state'].detach().cpu()),gradient_proof=gradproof)
            del enc,graph,outputs,losses,loss
            if stepdone%steps_epoch==0:
                worker_status(out,'VALIDATING',phase='checkpoint_selection',next_step=stepdone)
                options=[]
                _,once=evaluate(model,prepared,views,spec,device,batchsize,alpha=1.)
                for alpha in cfg['training']['alpha_grid']:
                    mixed={t:[(c,a,a['base']+alpha*(s-a['base']),d) for c,a,s,d in items] for t,items in once.items()}
                    metrics=aggregate_metrics(mixed,spec);rank=(metrics['selection_min_task_delta'],metrics['selection_weighted_score']);options.append((rank,-alpha,alpha,metrics))
                rank,_,alpha,metrics=max(options,key=lambda v:(v[0],v[1]));history.append({'step':stepdone,'alpha':alpha,'metrics':metrics})
                if rank[0]>best['rank'][0]+1e-8 or (abs(rank[0]-best['rank'][0])<=1e-8 and rank[1]>best['rank'][1]+1e-8):
                    best={'rank':rank,'step':stepdone,'alpha':alpha,'metrics':metrics,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}};stale=0
                else:stale+=1
                atomic_json(out/'VALIDATION_HISTORY.json',history)
            if stepdone%cfg['training']['checkpoint_steps']==0 or time.time()-last_save>=cfg['training']['checkpoint_seconds'] or stepdone%steps_epoch==0:
                record=checkpoint(stepdone);last_save=time.time()
            if interrupt_after is not None and stepdone>=interrupt_after:
                record=checkpoint(stepdone);worker_status(out,'PAUSED_TEST_INTERRUPT',next_step=stepdone);return {'state':'PAUSED_TEST_INTERRUPT','next_step':stepdone}
            if stepdone>=total:
                record=checkpoint(stepdone);finished=True;stop_reason='FULL_EPOCH_BUDGET';break
            if should_stop_early(profile,cfg,stepdone,steps_epoch,stale):
                record=checkpoint(stepdone);finished=True;stop_reason='EARLY_STOPPING';break
        if budget_stop:
            return worker_status(out,'PAUSED_BUDGET_OR_SIGNAL',next_step=record['next_step'],checkpoint=record,requires_explicit_new_budget=True)
        if not finished:fail('INTERNAL_TRAIN_STATE','Exited without a terminal or pause condition')
        if min(gradproof.values())<=0:fail('STATE_OR_PATHWAY_NO_SHARED_GRADIENT','Both losses must demonstrably reach HGT',proof=gradproof)
        worker_status(out,'VALIDATING',phase='final_predictions',next_step=record['next_step'])
        last_optimizer_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best['model']);metrics,pred=evaluate(model,prepared,views,spec,device,batchsize,alpha=best['alpha'])
        for task,items in pred.items():
            for cancer,arr,score,direction in items:
                atomic_npz(out/'predictions'/f'{cancer}__{task}.npz',lnc=arr['lnc'],target=arr['target'],score=score,baseline=arr['base'],direction_logit=direction,discovery_available=arr['discovery_available'])
        statuscode=completion_state(cfg,manifest['metadata']['profile'],best['alpha'],metrics,
                                    spec.get('relation_objective','binary'))
        result={'state':statuscode,'signature':signature,'config_sha256':digest(cfg),'source_profile':profile,'synthetic_fixture':bool(cfg.get('synthetic_fixture',False)),'best_step':best['step'],'alpha':best['alpha'],'validation':metrics,'gradient_proof':gradproof,'task_weights':cfg['model']['task_weights'],'training_progress':{'optimizer_steps_total':int(record['next_step']),'configured_total_steps':int(total),'steps_per_epoch':int(steps_epoch),'completed_epochs':float(record['next_step']/steps_epoch),'stop_reason':stop_reason,'diagnostic_full_epochs':diagnostic_full_epochs},'prediction_files':{rel:sha(out/rel) for rel in sorted(expected_predictions(spec))},'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'lineage':lineage,'attempt_id':attempt_id,'formal_authorization':formal_authorization,'formal_test_evaluated':False,'not_deployed':True}
        # The completion payload is inside a SHA-committed full optimizer checkpoint.
        # A mutable receipt cannot switch alpha/status independently of this checkpoint.
        model.load_state_dict(last_optimizer_model)
        result['last_checkpoint']=checkpoint(record['next_step'],completion=copy.deepcopy(result))
        atomic_json(out/'TRAINING_RECEIPT.json',result)
        verify_completion(prepared,out)
        return worker_status(out,statuscode,**{k:v for k,v in result.items() if k not in ('state','attempt_id')})
