"""R6 reassessment counterexamples tested against the repaired reachable paths.

All patients/checkpoints here are synthetic. Protocol checkpoint fixtures test
bindings only, never claim that a real PyG model or a cloud instance was trained.
"""
from __future__ import annotations
import copy,json,sys,time,subprocess,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import Stop,atomic_json,read_json,sha,digest
from cla_engine.statistics import design
from cla_engine.fixture import make_fixture
from cla_engine.prepare import prepare
from cla_engine.lineage import validate_prepared,verify_completion
from cla_engine.test_access import freeze_model,require_frozen_model
from cla_engine.cnv_sensitivity import run_audit
from cla_engine.controller import enroll_controller,assert_controller
from cla_engine.inputs import verify_import_authority
from cla_engine.metrics import paired_task_metrics
from cla_engine.runtime_guard import OptimizerProgressWatch,register_worker_device,begin_attempt,worker_status
from cc_hhgt.v32.adjustment_policy import COLUMNS,STATE_IDS,resolve_policy,CONTRACT_ID
from cc_hhgt.v32.covariate_contract import CovariateEncoder

ROOT=Path(__file__).resolve().parents[1]


def cc():
    rng=np.random.default_rng(790);n=80
    f=pd.DataFrame({'patient_id':[f'P{i:03}' for i in range(n)],'age_years':rng.normal(50,10,n),
        'sex':['F','M']*(n//2),'stage':['Stage IIA','Stage IVB']*(n//2),
        'purity':rng.uniform(.3,.9,n),'purity_source':'ABSOLUTE','leukocyte_fraction':rng.uniform(.1,.4,n)})
    for col in COLUMNS:
        if col.startswith('cell_fraction_'):f[col]=rng.uniform(.01,.1,n)
    return f


def fit_frame(f,target='RNAss'):
    ids=f.patient_id.tolist();_,policy=resolve_policy(target)
    return design(f,ids,ids[:60],family='pathway' if target=='__pathway__' else 'state',target_id=target,
        admission_cohorts={'source':ids[:60],'label':ids[60:]})


@pytest.mark.parametrize('column',['age_years','sex','stage','purity','leukocyte_fraction','cell_fraction_Macrophages.M0','cell_fraction_Macrophages.M1','cell_fraction_Macrophages.M2'])
def test_required_missing_column_is_not_an_adjusted_model(column):
    with pytest.raises(Stop) as exc:fit_frame(cc().drop(columns=column))
    assert exc.value.code=='REQUIRED_COVARIATES_UNAVAILABLE'


def test_required_low_coverage_not_imputed_as_success():
    f=cc();f.loc[:30,'age_years']=np.nan
    with pytest.raises(Stop) as exc:fit_frame(f)
    assert exc.value.code=='REQUIRED_COVARIATES_UNAVAILABLE'



@pytest.mark.parametrize('missing_count',[31,60])
def test_sparse_purity_is_explicitly_omitted_without_imputation(missing_count):
    f=cc();f.loc[:missing_count-1,'purity']=np.nan
    Z,r=fit_frame(f)
    assert r['admission_passed'] and np.isfinite(Z).all()
    assert r['dropped']['purity']=='OPTIONAL_SOURCE_UNAVAILABLE_EXPLICIT_NO_IMPUTATION'
    assert 'purity' not in r['required'] and 'purity' not in r['encoded']
    assert not any(c.startswith('purity') for c in r['column_names'])
    assert r['fit_counts']['purity']['observed']==60-missing_count
    # A held-out measurement cannot change an omitted source into a feature.
    changed=f.copy();changed.loc[60:,'purity']=.123
    other,_=fit_frame(changed)
    np.testing.assert_array_equal(Z,other)


def test_constant_complete_covariate_is_redundant_not_missing():
    f=cc();f['purity']=.7;f['sex']='F'
    Z,r=fit_frame(f)
    assert np.isfinite(Z).all() and r['admission_passed']
    assert set(r['redundant'])=={'purity','sex'}
    assert r['fit_counts']['purity']['observed']==60


def test_training_constant_new_variation_cannot_fake_adjustment():
    f=cc();f['purity']=.7;f.loc[60:,'purity']=.2
    with pytest.raises(Stop) as exc:fit_frame(f)
    assert exc.value.code=='COVARIATE_COHORT_NOT_ADMISSIBLE'


def test_required_high_cardinality_does_not_silently_drop():
    f=cc();f['sex']=[f'category{i}' for i in range(len(f))]
    with pytest.raises(Stop) as exc:fit_frame(f)
    assert exc.value.code=='REQUIRED_COVARIATES_UNAVAILABLE'


def test_optional_absent_is_explicitly_optional():
    z,r=fit_frame(cc())
    assert r['dropped']['technical_batch']=='COLUMN_ABSENT'
    assert 'technical_batch' not in r['required'] and r['admission_passed']


@pytest.mark.parametrize('target',list(STATE_IDS)+['__pathway__'])
def test_both_covariate_entrypoints_identical_for_each_target(target):
    f=cc();ids=f.patient_id.tolist();target,p=resolve_policy(target)
    a,r=fit_frame(f,target)
    b,old=CovariateEncoder.fit(f.iloc[:60],columns=COLUMNS,target_id=target,required=p['required']).transform(f)
    np.testing.assert_allclose(a,b,equal_nan=True)
    assert r['excluded']==old['excluded'] and r['column_names']==old['column_names']


def test_unrelated_components_are_not_silently_removed_from_immune_target():
    f=cc().drop(columns=['leukocyte_fraction','cell_fraction_Macrophages.M0','cell_fraction_Macrophages.M1','cell_fraction_Macrophages.M2'])
    with pytest.raises(Stop) as exc:fit_frame(f,'ImmuneScore')
    assert exc.value.code=='REQUIRED_COVARIATES_UNAVAILABLE'


def test_required_missing_rows_never_enter_rank_statistics():
    from cla_engine.statistics import pairs
    f=cc();f.loc[65,'purity']=np.nan
    z,r=fit_frame(f);assert np.isnan(z[65]).all()
    x=np.arange(80,dtype=float)[:,None];m=np.ones_like(x,dtype=bool)
    st=pairs(x,x,np.zeros_like(x),m,z,[0],[0],range(60,80),12)
    assert st['n'][0]==19


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp('r61_real_cpu');cfg=make_fixture(root/'sources')
    p=prepare(cfg,root/'prepared',0,'diagnostic')
    return cfg,p,root


def _completion(prepared_path,run):
    from test_joint_runtime_r5_input_and_bindings import completion_fixture
    return completion_fixture(prepared_path,run)


def test_actual_prepare_missing_all_covariates_fails_before_gpu(tmp_path):
    cfg=make_fixture(tmp_path/'src');p=Path(cfg['sources']['covariates']['path']);f=pd.read_csv(p,sep='\t')
    f[['cancer_id','patient_id','sample_id']].to_csv(p,sep='\t',index=False);cfg['sources']['covariates']['sha256']=sha(p)
    with pytest.raises(Stop) as exc:prepare(cfg,tmp_path/'prepared',0,'diagnostic')
    assert exc.value.code=='REQUIRED_COVARIATES_UNAVAILABLE'
    assert not (tmp_path/'prepared/MANIFEST.json').exists()


def test_actual_prepare_unattested_mutation_fails(tmp_path):
    cfg=make_fixture(tmp_path/'src');(Path(cfg['imports_root'])/'LOCAL_IMPORT_RECEIPT.json').unlink()
    with pytest.raises(Stop) as exc:prepare(cfg,tmp_path/'prepared',0,'diagnostic')
    assert exc.value.code=='IMPORT_PROVENANCE_REQUIRED'


def test_import_manifest_rewrite_not_provenance(prepared,tmp_path):
    cfg,_,_=prepared;original=Path(cfg['imports_root'])/'LOCAL_INPUT_MANIFEST.json'
    # Copy the complete sources so no shared fixture is mutated.
    import shutil
    root=tmp_path/'imports';shutil.copytree(cfg['imports_root'],root);c=copy.deepcopy(cfg);c['imports_root']=str(root)
    m=read_json(root/'LOCAL_INPUT_MANIFEST.json');rel=next(iter(m['files']));m['files'][rel]['original_path']='invented_original'
    atomic_json(root/'LOCAL_INPUT_MANIFEST.json',m)
    with pytest.raises(Stop):verify_import_authority(c)


@pytest.mark.parametrize('variant',['G0','G1','G2'])
def test_import_authority_shared_across_variants(prepared,variant):
    from cla_engine.queue import variant_config
    cfg,_,_=prepared
    assert verify_import_authority(variant_config(cfg,variant))['bound_files']


def test_outer_audit_without_run_cannot_open_raw_scores(prepared,tmp_path,monkeypatch):
    cfg,p,_=prepared
    monkeypatch.setattr('cla_engine.cnv_sensitivity.table',lambda *a,**kw:pytest.fail('must not read raw target before freeze'))
    with pytest.raises(Stop) as exc:run_audit(p,tmp_path/'outer',scope='outer_test')
    assert exc.value.code=='MODEL_FREEZE_REQUIRED' and not (tmp_path/'outer').exists()


def test_completed_but_unfrozen_run_is_still_blocked(prepared,tmp_path):
    # A new prepared directory is required to avoid sharing the freeze registry.
    cfg=make_fixture(tmp_path/'src');p=prepare(cfg,tmp_path/'p',0,'diagnostic');run=tmp_path/'run';_completion(p,run)
    with pytest.raises(Stop) as exc:run_audit(p,tmp_path/'audit',scope='outer_test',training=run)
    assert exc.value.code=='MODEL_FREEZE_REQUIRED'


def test_frozen_cpu_protocol_opens_bound_outer_and_records_access(tmp_path):
    cfg=make_fixture(tmp_path/'src');p=prepare(cfg,tmp_path/'p',0,'diagnostic');run=tmp_path/'run';_completion(p,run)
    first=freeze_model(p,run);second=freeze_model(p,run)
    assert first['freeze_sha256']==second['freeze_sha256']
    r=run_audit(p,tmp_path/'audit',scope='outer_test',training=run)
    assert r['model_freeze']['freeze_sha256']==first['freeze_sha256']
    assert (Path(first['freeze_path']).parent/'ACCESS_LOG.json').is_file()
    assert r['primary_training_target_changed'] is False


def test_modified_receipt_cannot_unlock_outer(prepared,tmp_path):
    _,p,_=prepared;r=_completion(p,tmp_path/'run');r['alpha']=1.;atomic_json(tmp_path/'run/TRAINING_RECEIPT.json',r)
    with pytest.raises(Stop):freeze_model(p,tmp_path/'run')


def test_source_audit_remains_training_only_without_checkpoint(prepared,tmp_path):
    _,p,_=prepared;r=run_audit(p,tmp_path/'source_audit',scope='source')
    assert r['scope']=='source' and r['model_freeze'] is None
    assert r['primary_training_target_changed'] is False


@pytest.mark.parametrize('script',['prepare_v32_joint_state.py','run_v32_joint_training.py','run_v32_state_training.py','run_native_state_workflow.py'])
def test_removed_state_cli_cannot_train_a_private_replacement(script,tmp_path):
    path=ROOT/'scripts'/script
    assert not path.exists(), 'A retired State execution path was restored'
    r=subprocess.run([sys.executable,str(path),'--output',str(tmp_path/'forbidden')],capture_output=True,text=True)
    assert r.returncode!=0
    assert not (tmp_path/'forbidden').exists()




def test_state_metrics_invariant_to_between_target_score_shift():
    y=np.tile([1.,0.,1.,0.],3);t=np.repeat(np.arange(3),4);s=np.tile([.9,.6,.4,.1],3)
    fam={'A':'f1','B':'f1','C':'f2'}
    a=paired_task_metrics(y,s,s,np.ones(12,bool),t,task='state',targets=['A','B','C'],families=fam,k=1)
    moved=s.copy();moved[t==1]+=100
    b=paired_task_metrics(y,moved,s,np.ones(12,bool),t,task='state',targets=['A','B','C'],families=fam,k=1)
    assert a['auprc']==b['auprc'] and a['precision_at_k']==b['precision_at_k']
    assert a['pooled_diagnostic']['auprc']!=b['pooled_diagnostic']['auprc']
    assert a['unit']=='TARGET_MACRO_WITHIN_FAMILY_THEN_FAMILY_MACRO_WITHIN_CANCER'


def test_state_lasso_and_model_select_same_aggregation():
    from cla_engine.prepare import macro_ap
    y=np.tile([1.,0.,1.,0.],3);t=np.repeat(np.arange(3),4);s=np.tile([.9,.6,.4,.1],3);ok=np.ones(12,bool)
    family={'A':'f1','B':'f1','C':'f2'};names=np.array(['A','B','C'])[t]
    a=paired_task_metrics(y,s,s,ok,t,task='state',targets=['A','B','C'],families=family,k=1)
    b=macro_ap(y,s,np.repeat('C',12),ok,np.array([family[n] for n in names]),names)
    assert b==a['auprc']


def test_controller_requires_enrollment_and_checks_machine(tmp_path,monkeypatch):
    cfg={'work_root':str(tmp_path/'work'),'imports_root':str(tmp_path/'imports'),'controller_contract':{'role':'149','python_executable':sys.executable}}
    monkeypatch.setattr('cla_engine.controller.machine_fingerprint',lambda:'a'*64)
    with pytest.raises(Stop) as exc:assert_controller(cfg)
    assert exc.value.code=='CONTROLLER_NOT_ENROLLED'
    enroll_controller(cfg);assert assert_controller(cfg)['state']=='CONTROLLER_149_VERIFIED'
    monkeypatch.setattr('cla_engine.controller.machine_fingerprint',lambda:'b'*64)
    with pytest.raises(Stop) as exc:assert_controller(cfg)
    assert exc.value.code=='CONTROLLER_HOST_OR_ROOT_MISMATCH'


def test_live_bootstrap_does_not_count_as_optimizer_progress():
    watch=OptimizerProgressWatch('a',100,first_step_timeout=10,progress_timeout=20)
    st={'attempt_id':'a','state':'STARTING','next_step':0,'updated_unix':109}
    assert watch.observe(st,109)=='INITIALIZING_NO_OPTIMIZER_PROGRESS_YET'
    st['updated_unix']=111
    assert watch.observe(st,111)=='WORKER_NO_OPTIMIZER_PROGRESS'


def test_monotonic_progress_and_stall_independent_of_heartbeat():
    watch=OptimizerProgressWatch('a',100,first_step_timeout=10,progress_timeout=20)
    st={'attempt_id':'a','state':'TRAINING','next_step':2,'optimizer_steps_this_attempt':1}
    assert watch.observe(st,105)=='OPTIMIZER_PROGRESS_OBSERVED'
    assert watch.observe(st,126)=='WORKER_OPTIMIZER_PROGRESS_STALLED'
    st['next_step']=1
    assert watch.observe(st,127)=='WORKER_OPTIMIZER_STEP_REGRESSED'


def test_device_proof_records_actual_cpu_not_claimed_gpu(tmp_path):
    begin_attempt(tmp_path);proof=register_worker_device(tmp_path,'cpu')
    assert proof['device_type']=='cpu' and proof['gpu_utilization_claimed'] is False
    r=worker_status(tmp_path,'TRAINING',next_step=1,optimizer_steps_this_attempt=1)
    assert r['device_proof']==proof


def test_full_supervise_launch_clock_excludes_transfer_but_budget_includes_it(prepared,tmp_path,monkeypatch):
    """Exercise original supervisor orchestration with a fake SSH endpoint and clock.
    It does not connect, run a GPU or claim a real network/Spot test.
    """
    from cla_engine import transport
    cfg,p,_=prepared;clock={'now':1000.};observed={};out=tmp_path/'supervisor'
    monkeypatch.setattr(transport.time,'time',lambda:clock['now'])
    monkeypatch.setattr(transport.time,'sleep',lambda n:None)
    def archive(source,dest):Path(dest).write_bytes(b'code-or-inputs')
    monkeypatch.setattr(transport,'make_code_tar',archive);monkeypatch.setattr(transport,'make_tar',archive)
    class Remote:
        def __init__(self,alias):self.deadline=None;self.last={}
        def run(self,command,timeout=120,check=True,**kw):
            if command.startswith('mkdir -p ') and 'tar -xf' not in command:self.root=command.split('mkdir -p ',1)[1]
            if command.startswith('sha256sum '):
                name=command.split()[-1];return SimpleNamespace(returncode=0,stdout=self.last.get(name,hashlib.sha256(b'code-or-inputs').hexdigest())+'  x\n',stderr='')
            if 'backend-check' in command:clock['now']+=70
            if command.startswith('nohup '):observed['spawn']=clock['now'];clock['now']+=1;return SimpleNamespace(returncode=0,stdout='321\n',stderr='')
            if command.startswith('ps -p '):
                rr=read_json(out/'REMOTE.json')['root'];return SimpleNamespace(returncode=0,stdout='S '+rr+'/code/cla.py worker',stderr='')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        def push(self,local,remote):
            if str(local).endswith('.tar'):clock['now']+=90
            self.last[remote]=sha(local)
        def pull(self,remote,local,**kw):atomic_json(local,{'state':'BACKEND_CHECK_TEST_DOUBLE_ONLY'})
        def json(self,path,**kw):return None
    monkeypatch.setattr(transport,'SSH',Remote)
    original=transport.check_remote_status
    def stop_after_observation(st,attempt,alive,now,started,**kwargs):
        observed['decision']=original(st,attempt,alive,now,started,**kwargs)
        observed['started']=started
        return 'WORKER_EXITED_WITHOUT_TERMINAL'  # stop the simulated loop after one observation
    monkeypatch.setattr(transport,'check_remote_status',stop_after_observation)
    transport.supervise(ROOT,p,out,'fixture-gpu',1,remote_python='/fixture/python')
    budget=read_json(out/'BUDGET.json');remote=read_json(out/'REMOTE.json')
    assert observed['started']==observed['spawn'] and observed['started']>=1250
    assert observed['decision']=='WAIT_FOR_CURRENT_ATTEMPT'
    assert budget['start']==1000 and budget['deadline']==4600
    assert remote['worker_launch_unix']>remote['transfer_started_unix']

@pytest.mark.parametrize('state',['WORKER_NO_OPTIMIZER_PROGRESS','WORKER_OPTIMIZER_STEP_REGRESSED','WORKER_OPTIMIZER_PROGRESS_STALLED','WORKER_PROGRESS_INVALID'])
def test_supervisor_optimizer_failures_have_nonzero_cli_exit(state):
    from cla import result_exit_code
    assert result_exit_code({'state':state})==70

@pytest.mark.parametrize('generation',['CLA_FULL_PYG_R6_LOCAL_MEASUREMENTS_V1','CLA_FULL_PYG_R61_LOCAL_MEASUREMENTS_V1','CLA_FULL_PYG_R62_LOCAL_MEASUREMENTS_V1'])
def test_real_import_workflow_accepts_verified_raw_archive_not_fake_receipt(tmp_path,generation):
    """Actual ZIP copy/hash/HMAC pipeline. Bytes are not actual Parquet and are not parsed."""
    import zipfile,sys
    from cla_engine.common import read_json,atomic_json,sha
    from cla_engine.controller import enroll_controller
    from cla_engine.inputs import import_local,verify_import_authority
    cfg=read_json(ROOT/'config/149.json')
    cfg['work_root']=str(tmp_path/'w');cfg['imports_root']=str(tmp_path/'i')
    cfg['controller_contract']['python_executable']=sys.executable
    enroll_controller(cfg,'149')
    source=tmp_path/'bytes';source.write_bytes(b'raw-measurement-archive-byte-test-not-a-parquet')
    from cla_engine.inputs import _required_imports
    roles=list(dict.fromkeys(_required_imports(cfg)))
    manifest={'format':generation,'files':{name:{'sha256':sha(source),'bytes':source.stat().st_size,'original_path':r'D:\\actual_user_root\\'+Path(name).name} for name in roles}}
    archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('LOCAL_INPUT_MANIFEST.json',json.dumps(manifest))
        for name in roles:z.write(source,name)
    got=import_local(archive,cfg)
    assert got['state']=='LOCAL_INPUTS_IMPORTED'
    result=verify_import_authority(cfg)
    assert result['archive_sha256']==sha(archive)
    assert result['biological_callability_certified'] is False
    (Path(cfg['imports_root'])/roles[0]).write_bytes(b'changed')
    with pytest.raises(Stop) as exc:
        verify_import_authority(cfg)
    assert exc.value.code=='IMPORT_MEASUREMENT_CHANGED'
