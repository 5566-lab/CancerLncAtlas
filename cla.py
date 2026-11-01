#!/usr/bin/env python3
"""Executable entry point. No inferred training permission and no automatic model simplification."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def result_exit_code(result):
    """Machine result codes never hide a failed or paused child behind exit zero."""
    state=result.get('state','')
    child=result.get('worker_terminal_state','')
    if state in ('ERROR','MODULE_FAILED','WORKER_EXITED_WITHOUT_TERMINAL','STALE_ATTEMPT_STATUS','STARTUP_TIMEOUT','WORKER_PROGRESS_STALLED','WORKER_NO_OPTIMIZER_PROGRESS','WORKER_OPTIMIZER_STALLED','WORKER_STEP_REGRESSION','WORKER_PROGRESS_INVALID','RECOVERY_INCOMPLETE') or state.startswith('WORKER_') or child=='ERROR':return 70
    if 'NO_GO' in child or 'GATE_FAIL' in state:return 3
    if any(v in state for v in ('PAUSED','BUDGET_EXHAUSTED','BUDGET_DEADLINE','UNREACHABLE','RECOVERY_WINDOW_EXPIRED','RECOVERY_PARTIAL')) or 'PAUSED' in child:return 75
    if state in {'QUEUE_COMPLETE_RELEASE_INSTANCE','QUEUE_COMPLETE_LOCAL_149'} and any(v!='PREDICTION_GATE_PASS_NOT_DEPLOYED' for v in result.get('prediction_gates',{}).values()):return 3
    if 'BLOCK' in state:return 2
    return 0

def parser():
    p=argparse.ArgumentParser(description='CancerLncAtlas V3.2 executable CPU/GPU workflow')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('package-check')
    a=sub.add_parser('export-local');a.add_argument('--output',default='local_export');a.add_argument('--root')
    a=sub.add_parser('import-local');a.add_argument('archive');a.add_argument('--config',default=str(ROOT/'config/149.json'))
    for command in ['preflight','prepare']:
        a=sub.add_parser(command);a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--output')
        if command=='prepare':
            a.add_argument('--fold',type=int,choices=range(5),default=0);a.add_argument('--profile',choices=['diagnostic','formal'],default='diagnostic');a.add_argument('--allow-formal',action='store_true');a.add_argument('--variant',choices=['G0','G1','G2'],default=None)
    a=sub.add_parser('worker');a.add_argument('--prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu');a.add_argument('--deadline',required=True,type=float);a.add_argument('--plan');a.add_argument('--job-id');a.add_argument('--config',default=str(ROOT/'config/149.json'))
    a=sub.add_parser('supervise');a.add_argument('--prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--gpu',required=True,help='An existing SSH-config alias, not a password');a.add_argument('--hours',required=True,type=float);a.add_argument('--remote-python');a.add_argument('--remote-root',default='/root/cla_full_pyg_r6_2');a.add_argument('--resume',action='store_true')
    a=sub.add_parser('baseline-compare');a.add_argument('--prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--hash-dimensions',type=int,choices=[4096],default=4096)
    a=sub.add_parser('modality-report');a.add_argument('--prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--refit-lasso',action='store_true');a.add_argument('--run')
    a=sub.add_parser('sc-celltype-table');a.add_argument('--binding',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('sc-celltype-fusion-prepare');a.add_argument('--prepared',required=True);a.add_argument('--run',required=True);a.add_argument('--sc-table',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('sc-celltype-fusion-train');a.add_argument('--plan',required=True);a.add_argument('--output',required=True);a.add_argument('--allow-training',action='store_true');a.add_argument('--deadline',type=float,required=True)
    a=sub.add_parser('diagnostic-plan');a.add_argument('--g0-prepared',required=True);a.add_argument('--g1-prepared',required=True);a.add_argument('--g2-prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--source-step72-run')
    a=sub.add_parser('diagnostic-arm');a.add_argument('--prepared',required=True);a.add_argument('--plan',required=True);a.add_argument('--arm',required=True,choices=['G0','G1','G2']);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu');a.add_argument('--deadline',required=True,type=float);a.add_argument('--lease-required',action='store_true')
    a=sub.add_parser('diagnostic-report');a.add_argument('--plan',required=True);a.add_argument('--run-g0',required=True);a.add_argument('--run-g1',required=True);a.add_argument('--run-g2',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('explain');a.add_argument('--prepared',required=True);a.add_argument('--run',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu')
    a=sub.add_parser('evaluate');a.add_argument('--prepared',required=True);a.add_argument('--run',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('gate');a.add_argument('--evaluations',nargs=5,required=True);a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--output',required=True)
    a=sub.add_parser('sets');a.add_argument('--evaluations',nargs=5,required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('evidence-audit');a.add_argument('--source',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('prepare-formal');a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--output',required=True);a.add_argument('--diagnostic-run',required=True)
    a=sub.add_parser('run-queue');a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--plan',required=True);a.add_argument('--output',required=True);a.add_argument('--hours',required=True,type=float);a.add_argument('--device',default='cpu');a.add_argument('--resume',action='store_true')
    a=sub.add_parser('backend-check');a.add_argument('--output',required=True);a.add_argument('--device',default='cpu')
    a=sub.add_parser('core-export');a.add_argument('--plan',required=True);a.add_argument('--queue-output',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu')
    a=sub.add_parser('module-plan');a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--core-root',required=True);a.add_argument('--output',required=True);a.add_argument('--bindings')
    a=sub.add_parser('run-modules');a.add_argument('--plan',required=True);a.add_argument('--output',required=True);a.add_argument('--modules',nargs='+',default=['all']);a.add_argument('--hours',required=True,type=float);a.add_argument('--resume',action='store_true')
    a=sub.add_parser('controller-init');a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--role',choices=['149'],required=True)
    a=sub.add_parser('freeze-model');a.add_argument('--prepared',required=True);a.add_argument('--run',required=True)
    a=sub.add_parser('recover-only');a.add_argument('--output',required=True);a.add_argument('--gpu',required=True);a.add_argument('--seconds',type=int,default=300)
    a=sub.add_parser('cnv-sensitivity');a.add_argument('--prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--scope',choices=['source','outer_test'],default='source');a.add_argument('--run',help='Completed frozen run, required only for outer_test')
    for command in ('singlecell-interpret','knowledge-paths'):
        a=sub.add_parser(command);a.add_argument('--queries',required=True);a.add_argument('--config',default=str(ROOT/'config/149.json'));a.add_argument('--output',required=True)
        if command=='singlecell-interpret':a.add_argument('--activity');a.add_argument('--donor-pairs');a.add_argument('--sc-association')
        else:a.add_argument('--prepared')
    a=sub.add_parser('status');a.add_argument('directory')
    a=sub.add_parser('selftest-data');a.add_argument('--workdir');a.add_argument('--parquet',action='store_true')
    a=sub.add_parser('selftest');a.add_argument('--workdir');a.add_argument('--parquet',action='store_true');a.add_argument('--device',default='cpu')
    for command in ('preflight','prepare','prepare-formal','run-queue','worker','gate'):
        a=sub.choices[command]
        a.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default=None,
                       help='all: every cancer in patient authority; list: only cancer_list')
        a.add_argument('--cancer-list','--cancer_list',nargs='+',default=None,metavar='CANCER',
                       help='Space-separated cancer IDs, e.g. BRCA LUAD; requires cancer_type=list')
    sub.choices['preflight'].add_argument('--profile',choices=['diagnostic','formal'],default='diagnostic')
    return p

def main():
    args=parser().parse_args()
    from cla_engine.common import config,read_json,atomic_json,status,Stop,lock,fail
    def selected_config():
        return config(args.config,cancer_type=getattr(args,'cancer_type',None),
                      cancer_list=getattr(args,'cancer_list',None))
    try:
        from cla_engine.location import enter_command
        enter_command(args,ROOT)
        if args.command=='controller-init':
            from cla_engine.controller import enroll_controller
            print(json.dumps(enroll_controller(selected_config(),args.role),ensure_ascii=False,indent=2));return 0
        if args.command=='freeze-model':
            from cla_engine.test_access import freeze_model
            print(json.dumps(freeze_model(args.prepared,args.run),ensure_ascii=False,indent=2));return 0
        if args.command=='package-check':
            from cla_engine.package import verify
            verify(ROOT);return 0
        if args.command=='export-local':
            from cla_engine.inputs import export_local
            print(export_local(args.output,args.root));return 0
        if args.command=='import-local':
            from cla_engine.inputs import import_local
            print(json.dumps(import_local(args.archive,selected_config()),ensure_ascii=False));return 0
        if args.command in ['preflight','prepare']:
            cfg=selected_config()
            if args.command=='prepare' and args.profile=='formal':
                fail('FORMAL_PREPARE_REQUIRES_SIGNED_ADMISSION','Use prepare-formal with a verified diagnostic run; direct formal preparation is disabled')
            if getattr(args,'variant',None):
                from cla_engine.queue import variant_config
                cfg=variant_config(cfg,args.variant)
            out=Path(args.output) if args.output else Path(cfg['work_root'])/('input_check' if args.command=='preflight' else f"{args.profile}/{cfg['graph']['variant']}/fold_{args.fold}/prepared")
            from cla_engine.inputs import preflight
            with lock(out.parent/(out.name+'_cpu_lock')):
                preflight(cfg,out.parent/(out.name+'_preflight'),profile=args.profile)
                if args.command=='prepare':
                    from cla_engine.prepare import prepare
                    prepare(cfg,out,args.fold,args.profile)
            return 0
        if args.command=='worker':
            if (args.cancer_type is not None or args.cancer_list is not None) and not (args.plan or args.job_id):
                fail('CANCER_SCOPE_ALREADY_PREPARED','Set cancer_type/cancer_list during prepare; a diagnostic worker uses the sealed prepared scope')
            from cla_engine.train import train
            r=train(args.prepared,args.output,device=args.device,deadline=args.deadline,
                formal_plan=args.plan,formal_job_id=args.job_id,
                formal_config=selected_config() if args.plan or args.job_id else None)
            return 75 if r['state']=='PAUSED_BUDGET_OR_SIGNAL' else 0
        if args.command=='supervise':
            from cla_engine.transport import supervise
            r=supervise(ROOT,args.prepared,args.output,args.gpu,args.hours,args.remote_python,args.remote_root,args.resume)
            return result_exit_code(r)
        if args.command=='baseline-compare':
            from cla_engine.baseline_compare import compare
            compare(args.prepared,args.output,args.hash_dimensions);return 0
        if args.command=='modality-report':
            from cla_engine.modality_report import report
            report(args.prepared,args.output,args.refit_lasso,args.run);return 0
        if args.command=='sc-celltype-table':
            from cla_engine.celltype_fusion import export_table
            export_table(args.binding,args.output);return 0
        if args.command=='sc-celltype-fusion-prepare':
            from cla_engine.celltype_fusion import prepare_fusion
            prepare_fusion(args.prepared,args.run,args.sc_table,args.output);return 0
        if args.command=='sc-celltype-fusion-train':
            from cla_engine.celltype_fusion import fit_fusion
            fit_fusion(args.plan,args.output,allow_training=args.allow_training,deadline=args.deadline);return 0
        if args.command=='diagnostic-plan':
            from cla_engine.diagnostic_matrix import create_plan
            create_plan(args.g0_prepared,args.g1_prepared,args.g2_prepared,args.output,args.source_step72_run);return 0
        if args.command=='diagnostic-arm':
            from cla_engine.diagnostic_matrix import run_arm
            r=run_arm(args.prepared,args.plan,args.arm,args.output,args.device,args.deadline,args.lease_required)
            return 75 if r['state']=='PAUSED_DIAGNOSTIC_BUDGET_OR_SIGNAL' else 0
        if args.command=='diagnostic-report':
            from cla_engine.diagnostic_matrix import summarize
            summarize(args.plan,{'G0':args.run_g0,'G1':args.run_g1,'G2':args.run_g2},args.output);return 0
        if args.command=='explain':
            from cla_engine.explain import contributions
            contributions(args.prepared,args.run,args.output,args.device);return 0
        if args.command=='evaluate':
            from cla_engine.evaluate import evaluate_fold
            evaluate_fold(args.prepared,args.run,args.output);return 0
        if args.command=='gate':
            from cla_engine.evaluate import gate
            r=gate(args.evaluations,selected_config(),args.output);return 0 if r['state']=='PREDICTION_GATE_PASS_NOT_DEPLOYED' else 3
        if args.command=='sets':
            from cla_engine.postprocess import make_sets
            make_sets(args.evaluations,args.output);return 0
        if args.command=='evidence-audit':
            from cla_engine.evidence import audit
            audit(args.source,args.output);return 0
        if args.command=='prepare-formal':
            from cla_engine.queue import create_plan
            from cla_engine.inputs import preflight
            cfg=selected_config();preflight(cfg,Path(args.output)/'preflight',profile='formal');create_plan(cfg,args.output,args.diagnostic_run);return 0
        if args.command=='run-queue':
            from cla_engine.queue import run_queue
            result=run_queue(ROOT,args.plan,selected_config(),args.output,hours=args.hours,
                resume=args.resume,device=args.device);return result_exit_code(result)
        if args.command=='backend-check':
            from cla_engine.backend_check import run
            run(args.output,args.device);return 0
        if args.command=='core-export':
            from cla_engine.core_export import export_core
            export_core(args.plan,args.queue_output,args.output,device=args.device);return 0
        if args.command=='module-plan':
            from cla_engine.modules import build_plan
            result=build_plan(selected_config(),args.core_root,args.output,args.bindings)
            return 2 if any(j['missing_roles'] for j in result['jobs'].values()) else 0
        if args.command=='run-modules':
            from cla_engine.modules import run_modules
            r=run_modules(args.plan,args.output,modules=args.modules,hours=args.hours,resume=args.resume)
            return result_exit_code(r)
        if args.command=='recover-only':
            from cla_engine.transport import recover_only
            return result_exit_code(recover_only(args.output,args.gpu,args.seconds))
        if args.command=='cnv-sensitivity':
            from cla_engine.cnv_sensitivity import run_audit
            run_audit(args.prepared,args.output,scope=args.scope,training=args.run);return 0
        if args.command=='singlecell-interpret':
            from cla_engine.singlecell_interpretation import interpret
            interpret(args.queries,selected_config(),args.output,activity_path=args.activity,donor_path=args.donor_pairs,association_path=args.sc_association);return 0
        if args.command=='knowledge-paths':
            from cla_engine.knowledge_paths import knowledge_paths
            knowledge_paths(args.queries,selected_config(),args.output,prepared=args.prepared);return 0
        if args.command=='status':
            p=Path(args.directory)/'STATUS.json';print(json.dumps(read_json(p),ensure_ascii=False,indent=2));return 0
        if args.command=='selftest-data':
            from cla_engine.fixture import data_selftest
            data_selftest(args.workdir,args.parquet);return 0
        if args.command=='selftest':
            from cla_engine.fixture import selftest
            selftest(args.workdir,args.parquet,args.device);return 0
    except Stop as e:
        out=Path(getattr(args,'output',None) or ROOT/'runtime_errors')
        if e.code=='DUPLICATE_WORKER':
            atomic_json(out/f'REJECTED_WORKER_{os.getpid()}.json',{'state':'DUPLICATE_WORKER','message':str(e)});print(str(e),file=sys.stderr);return 73
        status(out,'ERROR',error_code=e.code,message=str(e),details=e.detail,next_action='STOP_FIX_CAUSE_NO_AUTOMATIC_RETRY_OR_CONTRACT_CHANGE')
        return 2
    except Exception as e:
        out=Path(getattr(args,'output',None) or ROOT/'runtime_errors');out.mkdir(parents=True,exist_ok=True)
        (out/'TRACEBACK.txt').write_text(traceback.format_exc(),encoding='utf-8')
        code='DEPENDENCY_MISSING' if isinstance(e,ModuleNotFoundError) else ('CUDA_OOM' if 'out of memory' in str(e).lower() else 'UNEXPECTED_ERROR')
        status(out,'ERROR',error_code=code,message=str(e),next_action='STOP_NO_AUTOMATIC_BATCH_OR_LABEL_CHANGE',traceback=str(out/'TRACEBACK.txt'))
        return 70

if __name__=='__main__':sys.exit(main())
