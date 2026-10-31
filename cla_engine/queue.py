"""Fixed G0/G1/G2 x five-fold comparison on host 149.

G2 is the declared auxiliary parent. Never select the best graph on outer-test.
"""
from pathlib import Path
import copy,time
from .common import read_json,atomic_json,sha,digest,fail,status,verify_manifest,lock
from .evaluate import evaluate_fold,gate
from .lineage import verify_completion,validate_prepared,sign_file,verify_signed_file
from .local_execution import require_host_149,validate_formal_job,validate_formal_plan
from .train import train

# Compatibility probe for tests and old importers.  The local queue never calls it.
supervise=None

def variant_config(cfg,variant):
    if variant not in ('G0','G1','G2'):fail('GRAPH_VARIANT_INVALID',str(variant))
    result=copy.deepcopy(cfg);result['graph']['variant']=variant
    return result

def job_grid(cfg,root):
    root=Path(root);variants=cfg['graph'].get('formal_variants',['G0','G1','G2'])
    if variants!=['G0','G1','G2']:fail('FORMAL_GRID_DRIFT','Formal comparison requires G0/G1/G2, not a silently reduced grid')
    return [{'job_id':f'{v}_PF{f:02d}','variant':v,'fold':f,'prepared':str((root/v/f'fold_{f}'/'prepared').resolve())} for v in variants for f in range(5)]

def check_admission(cfg,admission):
    admission=Path(admission)
    if not (admission/'TRAINING_RECEIPT.json').is_file() or not (admission/'RUN_CONTRACT.json').is_file():fail('DIAGNOSTIC_BINDING_MISMATCH','Completion receipt and run contract are both required')
    proof=read_json(admission/'TRAINING_RECEIPT.json')
    contract=read_json(admission/'RUN_CONTRACT.json')
    source=Path(contract.get('controller_prepared_path',contract.get('prepared_path','')))
    if not source.is_dir():fail('DIAGNOSTIC_PREPARED_BINDING_MISSING','Need the actual diagnostic prepared path, not a standalone receipt')
    checked,_=verify_completion(source,admission)
    if checked!=proof:fail('DIAGNOSTIC_RECEIPT_MISMATCH','Completion differs')
    if proof['state']!='GO_TO_FORMAL_EVALUATION' or proof['alpha']<=0:fail('DIAGNOSTIC_NOT_ADMITTED','Do not expand a no-gain diagnostic')
    if proof.get('config_sha256')!=digest(cfg) or proof.get('source_profile')!='diagnostic' or proof.get('synthetic_fixture',False):fail('DIAGNOSTIC_BINDING_MISMATCH','Admission must come from this real diagnostic configuration')
    record=proof.get('last_checkpoint',{});checkpoint=(admission/record.get('file','')).resolve()
    if not checkpoint.is_relative_to(admission.resolve()) or not checkpoint.is_file() or sha(checkpoint)!=record.get('sha256') or record.get('signature')!=proof['signature']:fail('DIAGNOSTIC_CHECKPOINT_MISMATCH','Diagnostic checkpoint missing or changed')
    if any(proof.get('gradient_proof',{}).get(k,0)<=0 for k in ['pathway_max_shared_grad','state_max_shared_grad']):fail('DIAGNOSTIC_GRADIENT_PROOF_MISSING','Both tasks must update the core')
    for rel,h in proof.get('prediction_files',{}).items():
        p=(admission/rel).resolve()
        if not p.is_relative_to(admission.resolve()) or not p.is_file() or sha(p)!=h:fail('DIAGNOSTIC_PREDICTION_MISMATCH','Diagnostic prediction changed')
    return proof

def create_plan(cfg,root,admission):
    root=Path(root);check_admission(cfg,admission)
    from .prepare import prepare
    jobs=job_grid(cfg,root)
    for item in jobs:
        vc=variant_config(cfg,item['variant']);prepare(vc,item['prepared'],item['fold'],'formal')
        item.update(sha256=sha(Path(item['prepared'])/'MANIFEST.json'),config_sha256=digest(vc))
    plan={'format':'CLA_FULL_PYG_R62_15_JOB_PLAN_V1','config_sha256':digest(cfg),
          'diagnostic_receipt_sha256':sha(Path(admission)/'TRAINING_RECEIPT.json'),'diagnostic_run':str(Path(admission).resolve()),'jobs':jobs,
          'all_five_folds_required':True,'all_three_graph_variants_required':True,
          'auxiliary_parent_variant':'G2','outer_test_variant_selection_allowed':False,'budget_authorized':False}
    atomic_json(root/'JOBS.json',plan);sign_file(root/'JOBS.json',cfg,'FORMAL_JOB_PLAN');status(root,'FORMAL_QUEUE_PREPARED_NOT_STARTED',plan=str(root/'JOBS.json'),jobs=len(jobs));return plan

def run_queue(package,plan_path,cfg,out,alias=None,hours=None,remote_python=None,resume=False,*,device='cpu'):
    """Run the signed formal grid locally under one persistent absolute deadline.

    ``alias`` and ``remote_python`` remain only to reject old callers explicitly;
    no SSH transport or off-host worker is reachable from this queue.
    """
    require_host_149()
    plan_path,plan=validate_formal_plan(plan_path,cfg)
    if alias is not None or remote_python is not None:
        fail('REMOTE_QUEUE_DISABLED','Formal queue executes locally on host 149; SSH arguments are forbidden')
    if hours is None or not 0<float(hours)<=96:fail('BUDGET_INVALID','Explicit budget must be >0 and <=96 hours')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with lock(out/'queue_control'):
        budget=out/'QUEUE_BUDGET.json'
        if budget.exists():
            b=read_json(budget)
            if not resume:fail('QUEUE_EXISTS','Use --resume; it does not reset budget')
            if b['plan_sha256']!=sha(plan_path):fail('QUEUE_INPUT_DRIFT','Plan changed')
            if b.get('config_sha256')!=digest(cfg) or b.get('device')!=device:fail('QUEUE_INPUT_DRIFT','Configuration or local device changed')
        else:
            b={'deadline':time.time()+float(hours)*3600,'plan_sha256':sha(plan_path),'config_sha256':digest(cfg),
               'authorized_hours':float(hours),'device':device,'target_host':'149','off_host_execution':False};atomic_json(budget,b)
        if b['deadline']<=time.time():
            return status(out,'QUEUE_BUDGET_EXHAUSTED',deadline=b['deadline'],budget_reset=False,off_host_execution=False)
        evals={v:[] for v in ['G0','G1','G2']}
        for j in plan['jobs']:
            validate_formal_job(j['prepared'],plan_path,j['job_id'],cfg)
            folder=out/j['variant']/f"fold_{j['fold']}";run=folder/'run';remaining=b['deadline']-time.time()
            if remaining<=0:return status(out,'QUEUE_BUDGET_EXHAUSTED',deadline=b['deadline'],budget_reset=False,off_host_execution=False)
            if not (run/'TRAINING_RECEIPT.json').exists():
                status(out,'QUEUE_RUNNING_LOCAL_149',job_id=j['job_id'],deadline=b['deadline'],device=device,off_host_execution=False)
                r=train(j['prepared'],run,device=device,deadline=b['deadline'],resume=True,
                    formal_plan=plan_path,formal_job_id=j['job_id'],formal_config=cfg)
                if not (run/'TRAINING_RECEIPT.json').exists():
                    return status(out,'QUEUE_PAUSED_LOCAL_149',job_id=j['job_id'],child_state=r['state'],
                        deadline=b['deadline'],budget_reset=False,off_host_execution=False)
            else:
                verify_completion(j['prepared'],run)
            ev=folder/'evaluation';evaluate_fold(j['prepared'],run,ev);evals[j['variant']].append(str(ev))
        results={}
        for variant,paths in evals.items():results[variant]=gate(paths,variant_config(cfg,variant),out/'gates'/variant)
        atomic_json(out/'GRAPH_COMPARISON.json',{'gates':results,'primary_variant':'G2','no_post_test_variant_selection':True})
        return status(out,'QUEUE_COMPLETE_LOCAL_149',prediction_gates={k:v['state'] for k,v in results.items()},
            deadline=b['deadline'],budget_reset=False,off_host_execution=False,production_release_allowed=False)
