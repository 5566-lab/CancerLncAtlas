"""Original full-r1 modules, deterministic argument binding and bounded CPU execution.

Missing inputs are BLOCKED_INPUT, never substituted by old probabilities. The
registry executes the existing scientific runners; it is not an eight-head mock.
"""
from __future__ import annotations
import ast,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from .common import read_json,atomic_json,sha,digest,fail,status,lock,code_digest

PACKAGE=Path(__file__).resolve().parents[1]
SCRIPTS=PACKAGE/'scripts'

# flags map to data roles. Defaults are explicit in this registry, not inferred by an agent.
RECIPES={
 'single_cell':{'script':'run_v32_single_cell_training.py','paths':{'--candidates':'candidates','--dataset-manifest':'sc_dataset','--sc-association':'sc_association','--lnc-celltype':'sc_celltype','--activity':'sc_activity','--core-embedding-manifest':'core_manifest'},'values':{'--training-run-id':'V32_FULL_PYG_R62_SC','--epochs':40},'resume':False},
 'mutation_cnv':{'script':'run_v32_genomic_training.py','paths':{'--candidates':'candidates','--patient-folds':'patient_map','--patient-fold-authority-receipt':'patient_receipt','--pathway-gene-membership':'membership','--sample-gene-mutation':'gene_mutation','--sample-lncrna-mutation':'lnc_mutation','--mc3':'mc3','--core-embedding-manifest':'core_manifest','--cnv-streaming-success':'cnv_streaming_receipt'},'values':{'--training-run-id':'V32_FULL_PYG_R62_GENOMIC','--epochs':40},'resume':False},
 'directional_cnv':{'script':'run_v32_directional_cnv_head_oof.py','paths':{'--candidates':'candidates','--patient-folds':'patient_map','--directional-root':'cnv_root','--association-root':'directional_associations','--core-embedding-manifest':'core_manifest'},'values':{'--training-run-id':'V32_FULL_PYG_R62_DIRECTIONAL_CNV'},'resume':False},
 'drug':{'script':'run_v32_drug_training.py','paths':{'--exact-candidates':'candidates','--drug-candidates':'drug_candidates','--raw-drug-response':'drug_response','--raw-lncrna-expression':'cell_line_expression','--cell-line-map':'cell_line_map','--drug-gene-target':'drug_targets','--core-embedding-manifest':'core_manifest'},'hash_roles':{'--expected-staging-manifest-sha256':'drug_staging_manifest','--expected-exact-release-prediction-sha256':'drug_exact_prediction_proof','--expected-exact-release-lineage-sha256':'drug_exact_lineage_proof'},'values':{'--training-run-id':'V32_FULL_PYG_R62_DRUG'},'resume':False},
 'clinical':{'script':'run_v32_clinical_training.py','paths':{'--workbook':'clinical_workbook','--fold-manifest':'patient_map','--patient-fold-authority-receipt':'patient_receipt','--expression-root':'expression_root','--activity-root':'activity_root','--core-root':'core_root'},'values':{'--epochs':100},'resume':False},
 'atac':{'script':'run_v32_atac_training.py','paths':{'--candidates':'candidates','--patient-folds':'patient_map','--patient-fold-authority-receipt':'patient_receipt','--pathway-membership':'membership','--materialization-success':'atac_materialization'},'values':{'--training-run-id':'V32_FULL_PYG_R62_ATAC'},'resume':False},
 'epigenetic':{'script':'run_v32_epigenetic_expression_head.py','paths':{'--regulatory-features':'regulatory_features'},'values':{'--run-id':'V32_FULL_PYG_R62_EPIGENETIC'},'resume':False},
 'evidence_stage':{'script':'materialize_v32_evidence_streaming_stage.py','paths':{'--interaction-relation':'interaction_relation','--empty-evidence-event':'empty_evidence','--exact-pathway-members':'membership','--formal-candidates':'candidates','--conversion-manifest':'conversion_manifest','--patient-fold-authority':'patient_authority','--patient-fold-receipt':'patient_receipt','--r4-core-authority':'core_manifest'},'optional_paths':{'--direct-evidence-events':'direct_evidence'},'values':{'--threads':2},'resume':False},
 'evidence_train':{'script':'run_v32_evidence_streaming_training.py','paths':{'--stage-root':'@evidence_stage','--core-embedding-root':'core_root'},'values':{'--allow-training':True,'--device':'cpu','--checkpoint-steps':20},'deps':['evidence_stage'],'resume':True},
 'evidence_predict':{'script':'run_v32_evidence_streaming_predict.py','paths':{'--stage-root':'@evidence_stage','--training-root':'@evidence_train','--core-embedding-root':'core_root','--candidates':'candidates'},'values':{'--device':'cpu'},'deps':['evidence_stage','evidence_train'],'resume':False},
 'interaction':{'script':'materialize_v32_interaction_release.py','paths':{'--physical-facts':'physical_facts','--membership':'membership','--candidates':'candidates','--evidence-binding':'new_evidence_binding','--evidence-semantic-wrapper':'new_evidence_wrapper'},'values':{},'resume':False,'output_flag':'--output'},
}

def parser_flags(script):
    """Extract actual argparse flags, including literal loops, without importing runners."""
    found=set()
    def value(n,env):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return env.get(n.id)
        if isinstance(n,(ast.List,ast.Tuple)):return [value(v,env) for v in n.elts]
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):
            a,b=value(n.left,env),value(n.right,env)
            return a+b if isinstance(a,str) and isinstance(b,str) else None
        if isinstance(n,ast.JoinedStr):
            chunks=[]
            for v in n.values:
                t=value(v.value,env) if isinstance(v,ast.FormattedValue) else value(v,env)
                if t is None:return None
                chunks.append(str(t))
            return ''.join(chunks)
        return None
    def visit(n,env):
        if isinstance(n,ast.For) and isinstance(n.target,ast.Name):
            vals=value(n.iter,env)
            if isinstance(vals,(tuple,list)):
                for v in vals:
                    child=dict(env);child[n.target.id]=v
                    for body in n.body:visit(body,child)
                for body in n.orelse:visit(body,env)
                return
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
            for arg in n.args:
                text=value(arg,env)
                if isinstance(text,str) and text.startswith('--'):found.add(text)
        for child in ast.iter_child_nodes(n):visit(child,env)
    visit(ast.parse(script.read_text(encoding='utf-8')), {})
    return found

def path_record(path):
    p=Path(path)
    if p.is_file():return {'path':str(p.resolve()),'kind':'file','sha256':sha(p),'bytes':p.stat().st_size}
    if p.is_dir():
        # Declared source directory only. Never scan disks, home, or arbitrary parents.
        files=sorted(q for q in p.rglob('*') if q.is_file() and not q.is_symlink())
        if len(files)>10000:fail('MODULE_DIRECTORY_TOO_LARGE','Bind a narrower input directory',path=str(p),files=len(files))
        entries={str(q.relative_to(p)):sha(q) for q in files}
        return {'path':str(p.resolve()),'kind':'directory','sha256':digest(entries),'files':entries}
    return None

def known_candidates(cfg,core):
    i=Path(cfg['imports_root']);core=Path(core);src=cfg['sources']
    er=Path('/dell_2/DSC/CancerLncAtlas/results/v32_evidence_interaction_parquet_20260829_r3')
    pa=Path(src['patient_map']['path']).parent
    result={
      'core_manifest':[core/'CORE_EMBEDDING_MANIFEST.json'],'core_root':[core],'candidates':[core/'CANDIDATE_UNIVERSE.parquet'],
      'patient_map':[src['patient_map']['path']],'patient_authority':[pa/'PATIENT_FOLD_AUTHORITY.tsv'],'patient_receipt':[pa/'PATIENT_FOLD_AUTHORITY_RECEIPT.json'],
      'membership':[src['exact_membership']['path']],'expression_root':[src['lnc_expression_root']['path']],'activity_root':[src['pathway_activity_root']['path']],
      'cnv_root':[src['cnv_root']['path']],
      'sc_dataset':[i/'singlecell/dataset_manifest_33c.parquet'],'sc_celltype':[i/'singlecell/lnc_celltype'],
      'sc_association':[i/'singlecell/association',i/'singlecell/sc_association',i/'singlecell/lnc_exact_pathway'],'sc_activity':[i/'singlecell/activity'],
      'gene_mutation':[i/'genomic/sample_gene_mutation_mc3_assay_callable.parquet'],'lnc_mutation':[i/'genomic/sample_lncrna_mutation_mc3_assay_callable.parquet'],
      'mc3':[i/'modules/genomic/mc3.v0.2.8.PUBLIC.maf.gz',Path('/public8/DSC/CancerLncAtlas/input/mc3.v0.2.8.PUBLIC.maf.gz')],
      'cnv_streaming_receipt':[i/'modules/genomic/STREAMING_SEGMENT_CNV_SUCCESS.json'],
      'directional_associations':[Path(cfg['work_root'])/'directional_association_authority'],
      'drug_candidates':[i/'modules/drug/drug_candidates.parquet',i/'modules/drug/factored_candidates'],
      'drug_response':[i/'modules/drug/drug_response.parquet'],
      'cell_line_expression':[i/'modules/drug/cell_line_lncrna_expression.parquet'],
      'drug_targets':[i/'modules/drug/drug_targets.parquet'],
      'cell_line_map':[i/'modules/drug/cell_line_map.parquet'],
      'drug_staging_manifest':[i/'modules/drug/factored_candidates/STAGING_MANIFEST.json'],
      'drug_exact_prediction_proof':[Path(cfg['work_root'])/'module_bindings/exact_release_predictions.parquet'],
      'drug_exact_lineage_proof':[Path(cfg['work_root'])/'module_bindings/exact_release_lineage.json'],
      'clinical_workbook':[i/'modules/clinical/TCGA-CDR-SupplementalTableS1.xlsx',Path('/public8/DSC/CancerLncAtlas/input/TCGA-CDR-SupplementalTableS1.xlsx')],
      'atac_materialization':[i/'modules/atac/MATERIALIZATION_SUCCESS.json'],
      'regulatory_features':[i/'modules/epigenetic/regulatory_features.parquet'],
      'interaction_relation':[er/'interaction_relation_recovered.parquet'],
      'empty_evidence':[er/'evidence_event_empty_by_contract.parquet'],
      'conversion_manifest':[er/'CONVERSION_MANIFEST.json'],
      'direct_evidence':[i/'modules/evidence/direct_evidence_events.parquet'],
      'physical_facts':[er/'physical_interaction_facts.parquet'],
      'new_evidence_binding':[Path(cfg['work_root'])/'module_bindings/EVIDENCE_OUTPUT_BINDING.json'],
      'new_evidence_wrapper':[Path(cfg['work_root'])/'module_bindings/EVIDENCE_SEMANTIC_WRAPPER.json']}
    return {k:[str(p) for p in v] for k,v in result.items()}

def build_plan(cfg,core_root,out,bindings=None):
    from cc_hhgt.v32.core_export_contract_r4 import validate_export
    core_root=Path(core_root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest,core_sha,_=validate_export(core_root/'CORE_EMBEDDING_MANIFEST.json')
    candidates=known_candidates(cfg,core_root);overrides=read_json(bindings) if bindings else {}
    if not isinstance(overrides,dict):fail('MODULE_BINDINGS_SCHEMA','Bindings are role -> actual absolute path only')
    unknown=set(overrides)-set(candidates)
    if unknown:fail('UNKNOWN_MODULE_ROLE','Unexpected binding keys',roles=sorted(unknown))
    resolved={};missing={};path_cache={}
    for role,paths in candidates.items():
        paths=[overrides[role]] if role in overrides else paths
        existing=[str(Path(p).resolve()) for p in paths if Path(p).exists()]
        if len(set(existing))>1:
            # No "latest file wins" rule. Exact path required when multiple candidates exist.
            missing[role]={'reason':'AMBIGUOUS_PATHS','checked':existing};continue
        if not existing:missing[role]={'reason':'PATH_NOT_FOUND','checked':paths};continue
        p=existing[0]
        if p not in path_cache:path_cache[p]=path_record(p)
        resolved[role]=path_cache[p]
    jobs={}
    for name,r in RECIPES.items():
        flags=parser_flags(SCRIPTS/r['script']);requested=set(r['paths'])|set(r.get('optional_paths',{}))|set(r['values'])|set(r.get('hash_roles',{}))|{r.get('output_flag','--output-root')}
        if requested-flags:fail('MODULE_CLI_DRIFT','Registry flags do not exist in original runner',module=name,flags=sorted(requested-flags))
        needed=sorted({role for role in list(r['paths'].values())+list(r.get('hash_roles',{}).values()) if not role.startswith('@') and role not in resolved})
        jobs[name]={'script':r['script'],'script_sha256':sha(SCRIPTS/r['script']),'recipe':r,
                    'status':'BLOCKED_INPUT' if needed else ('WAITING_DEPENDENCY' if r.get('deps') else 'READY'),
                    'missing_roles':needed,'checkpoint_resume_supported':bool(r['resume'])}
    plan={'format':'R62_ORIGINAL_MULTIMODAL_RUNNERS_V1','configuration_sha256':digest(cfg),'code_sha256':code_digest(),
          'core_manifest_sha256':core_sha,'core_graph_sha256':manifest['graph_authority_receipt']['sha256'],
          'bindings':resolved,'missing':missing,'jobs':jobs,'state_module':'ALREADY_JOINT_IN_CORE_NOT_RETRAINED_AS_PRIVATE',
          'evidence_changes_primary_ranking':False,'all_modules_trained':False,'old_prediction_inputs_allowed':False}
    atomic_json(out/'MODULE_PLAN.json',plan);atomic_json(out/'NEEDS_INPUT_BINDINGS.json',missing)
    status(out,'MODULE_PLAN_READY' if all(j['status']!='BLOCKED_INPUT' for j in jobs.values()) else 'MODULE_PLAN_HAS_BLOCKERS',
           ready=[n for n,j in jobs.items() if j['status']=='READY'],blocked=[n for n,j in jobs.items() if j['status']=='BLOCKED_INPUT'])
    return plan

def _verify_sources(plan,roles):
    for role in roles:
        if role.startswith('@'):continue
        record=plan['bindings'][role];observed=path_record(record['path'])
        if observed is None or observed['sha256']!=record['sha256']:fail('MODULE_INPUT_DRIFT','Source changed after planning',role=role,path=record['path'])

def run_modules(plan_path,out,*,modules=None,hours,python=None,resume=False):
    plan_path=Path(plan_path).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);plan=read_json(plan_path)
    if plan.get('format')!='R62_ORIGINAL_MULTIMODAL_RUNNERS_V1' or plan['code_sha256']!=code_digest():fail('MODULE_PLAN_CODE_DRIFT','Rebuild a plan against current source; do not rewrite its SHA')
    wanted=list(plan['jobs']) if not modules or modules==['all'] else list(modules)
    if set(wanted)-set(plan['jobs']):fail('UNKNOWN_MODULE','Unknown requested module')
    expanded=[]
    def add(n):
        for d in plan['jobs'][n]['recipe'].get('deps',[]):add(d)
        if n not in expanded:expanded.append(n)
    for n in wanted:add(n)
    blockers={n:plan['jobs'][n]['missing_roles'] for n in expanded if plan['jobs'][n]['missing_roles']}
    if blockers:fail('MODULE_INPUT_BLOCKED','Bind the listed real paths; no old probabilities or fake labels substituted',modules=blockers)
    # Check code/output roles before spending any task budget.
    code_root=PACKAGE.resolve()
    if not (code_root/'cc_hhgt'/'v32').is_dir():fail('MODULE_CODE_ROOT_MISSING',str(code_root))
    if out==code_root or out.is_relative_to(code_root):fail('MODULE_OUTPUT_INSIDE_SOURCE','Keep outputs separate from source')
    for name in expanded:
        script=SCRIPTS/plan['jobs'][name]['script']
        if not script.is_file() or sha(script)!=plan['jobs'][name]['script_sha256']:
            fail('MODULE_SCRIPT_DRIFT',str(script))
    with lock(out/'controller'):
        bp=out/'BUDGET.json'
        if bp.exists():
            b=read_json(bp)
            if not resume or b['plan_sha256']!=sha(plan_path):fail('MODULE_RESUME_BINDING','Resume requires the same plan and does not reset deadline')
        else:
            if not 0<hours<=96:fail('BUDGET_INVALID','Explicit CPU job wall time is required')
            b={'deadline':time.time()+hours*3600,'plan_sha256':sha(plan_path),'authorized_hours':hours};atomic_json(bp,b)
        completed={}
        for name in expanded:
            job=plan['jobs'][name];r=job['recipe'];record_path=out/name/'MODULE_RECEIPT.json'
            if record_path.exists():
                rec=read_json(record_path)
                if rec.get('plan_sha256')!=sha(plan_path):fail('MODULE_RESULT_BINDING_DRIFT',name)
                current=path_record(rec['output_root'])
                if current is None or current['sha256']!=rec['output_sha256']:fail('MODULE_RESULT_HASH_DRIFT',name)
                completed[name]=rec;continue
            if time.time()>=b['deadline']:return status(out,'MODULE_BUDGET_EXHAUSTED',completed=list(completed),training_complete=False)
            _verify_sources(plan,[v for v in list(r['paths'].values())+list(r.get('hash_roles',{}).values()) if not v.startswith('@')])
            jobdir=out/name;jobdir.mkdir(parents=True,exist_ok=True);active=jobdir/'ACTIVE_ATTEMPT.json'
            previous=read_json(active) if active.exists() else None
            continue_attempt=bool(previous and r['resume'] and resume)
            if continue_attempt:target=Path(previous['output_root'])
            else:
                # Prior partial attempts remain quarantined by their immutable attempt name.
                target=jobdir/f'attempt_{uuid.uuid4().hex[:10]}'/'output'
                target.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(active,{'output_root':str(target),'plan_sha256':sha(plan_path)})
            command=[python or sys.executable,str(SCRIPTS/r['script'])]
            for flag,role in {**r['paths'],**r.get('optional_paths',{})}.items():
                if role.startswith('@'):
                    dependency=role[1:]
                    if dependency not in completed:fail('MODULE_DEPENDENCY_MISSING',dependency)
                    value=completed[dependency]['output_root']
                elif role in plan['bindings']:value=plan['bindings'][role]['path']
                else:continue
                command.extend([flag,value])
            for flag,role in r.get('hash_roles',{}).items():
                command.extend([flag,plan['bindings'][role]['sha256']])
            for flag,value in r['values'].items():
                if value is True:command.append(flag)
                elif value is not False:command.extend([flag,str(value)])
            command.extend([r.get('output_flag','--output-root'),str(target)])
            flags=parser_flags(SCRIPTS/r['script'])
            if '--repo-root' in flags:
                code_root=PACKAGE.resolve()
                if not (code_root/'cc_hhgt'/'v32').is_dir():fail('MODULE_CODE_ROOT_MISSING',str(code_root))
                command.extend(['--repo-root',str(code_root)])
            if name=='evidence_stage':
                roles={'interaction_relation':'interaction_relation','empty_evidence_event':'empty_evidence','exact_pathway_members':'membership','formal_candidates':'candidates','conversion_manifest':'conversion_manifest','patient_fold_authority':'patient_authority','patient_fold_receipt':'patient_receipt'}
                pins={k:plan['bindings'][v]['sha256'] for k,v in roles.items()}
                if 'direct_evidence' in plan['bindings']:pins['direct_evidence_event']=plan['bindings']['direct_evidence']['sha256']
                pinpath=target.parent/'EXPECTED_HASHES.json';atomic_json(pinpath,pins);command.extend(['--expected-hashes-json',str(pinpath)])
            if name in ('evidence_train','evidence_predict'):
                command.extend(['--expected-core-manifest-sha256',plan['core_manifest_sha256'],'--expected-graph-authority-receipt-sha256',plan['core_graph_sha256']])
            if name=='evidence_train':
                command.extend(['--deadline',str(b['deadline'])])
                if continue_attempt:command.append('--resume')
            if name=='interaction':command.extend(['--evidence-semantic-wrapper-sha256',plan['bindings']['new_evidence_wrapper']['sha256']])
            env=os.environ.copy();env['PYTHONPATH']=str(PACKAGE)+os.pathsep+env.get('PYTHONPATH','');env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
            atomic_json(target.parent/'COMMAND.json',{'argv':command,'deadline':b['deadline'],'old_checkpoints_allowed':False})
            status(out,'MODULE_RUNNING',module=name,output_root=str(target),deadline=b['deadline'])
            logpath=target.parent/'RUN.log'
            if time.time() >= b['deadline']:return status(out,'MODULE_BUDGET_EXHAUSTED',completed=list(completed),training_complete=False)
            with logpath.open('a',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=(os.name!='nt'))
                try:code=proc.wait(timeout=max(.1,b['deadline']-time.time()))
                except subprocess.TimeoutExpired:
                    if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                    else:proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                        else:proc.kill()
                        proc.wait()
                    return status(out,'MODULE_PAUSED_BUDGET',module=name,log=str(logpath),resume_supported=r['resume'],training_complete=False)
            if code==75:return status(out,'MODULE_PAUSED',module=name,log=str(logpath),training_complete=False)
            if code!=0:return status(out,'MODULE_FAILED',module=name,exit_code=code,log=str(logpath),next_action='STOP_FIX_REPORTED_CAUSE_NO_AUTOMATIC_RETRY')
            if not target.is_dir() or not any(target.rglob('*')):fail('MODULE_EMPTY_SUCCESS','Process exited zero without artifacts',module=name)
            # A zero process exit is execution completion, NOT a scientific PASS.
            results=path_record(target)
            rec={'module':name,'plan_sha256':sha(plan_path),'script_sha256':sha(SCRIPTS/r['script']),
                 'output_root':str(target),'output_sha256':results['sha256'],'state':'EXECUTED_SCIENTIFIC_GATE_NOT_INFERRED',
                 'process_exit':code,'production_release_allowed':False}
            atomic_json(record_path,rec);completed[name]=rec
        return status(out,'REQUESTED_MODULES_EXECUTED_NOT_DEPLOYED',completed=list(completed),all_modules_trained=False)
