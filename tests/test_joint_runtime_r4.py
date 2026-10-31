import ast,copy,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from cla_engine.common import read_json,atomic_json,digest,verify_manifest,Stop
from cla_engine.queue import job_grid,variant_config
from cla_engine.modules import RECIPES,SCRIPTS,parser_flags
from cc_hhgt.v32.joint_model import TASK_WEIGHTS,weighted_task_selection
from cc_hhgt.v32.optimizer_guard_v2 import optimizer_step_with_global_delta_guard
from cc_hhgt.v32.evidence_training import BagExample
from cc_hhgt.v32.evidence_fit_streaming_v2 import fit_private_eventset_streaming,supervision_profile
from cc_hhgt.v32.resume_state_r4 import TrainingPaused
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def examples(n=8):
    rng=np.random.default_rng(123)
    return [BagExample(('C',f'L{i}','P'),rng.normal(size=(2,16)).astype('float32'),rng.normal(size=6).astype('float32'),float(i%2),[0,2][i%2],i%5,2) for i in range(n)]

def factory(ex):return lambda:iter([ex[:4],ex[4:]])

def test_evidence_real_transformer_batch_resume_exact(tmp_path):
    ex=examples();f=factory(ex);args=dict(event_feature_dim=16,core_feature_dim=6,seed=19,epochs=3,patience=4,batch_size=2,hidden_dim=8,dropout=.15)
    full=fit_private_eventset_streaming(f,f,**args)
    with pytest.raises(TrainingPaused,match='TEST_INTERRUPT'):
        fit_private_eventset_streaming(f,f,**args,checkpoint_root=tmp_path/'cp',input_binding={'raw_sha256':'a'*64},checkpoint_steps=1,interrupt_after=3)
    recovered=fit_private_eventset_streaming(f,f,**args,checkpoint_root=tmp_path/'cp',input_binding={'raw_sha256':'a'*64},checkpoint_steps=1,resume=True)
    assert full.optimizer_steps==recovered.optimizer_steps
    assert full.history==recovered.history
    assert full.final_parameter_sha256==recovered.final_parameter_sha256
    assert all(torch.equal(v,recovered.model.state_dict()[k]) for k,v in full.model.state_dict().items())

def test_evidence_resume_changed_binding_is_rejected(tmp_path):
    ex=examples();args=dict(event_feature_dim=16,core_feature_dim=6,seed=19,epochs=2,patience=3,batch_size=2,hidden_dim=8,dropout=0.,checkpoint_root=tmp_path,input_binding={'hash':'old'})
    with pytest.raises(TrainingPaused):fit_private_eventset_streaming(factory(ex),factory(ex),**args,interrupt_after=1)
    args['input_binding']={'hash':'new'}
    with pytest.raises(RuntimeError,match='DRIFT'):fit_private_eventset_streaming(factory(ex),factory(ex),**args,resume=True)

def test_evidence_one_class_is_not_trainable():
    ex=examples();ex=[BagExample(e.key,e.event_features,e.core_features,1.,-1,e.leakage_fold,e.event_count) for e in ex]
    with pytest.raises(ValueError,match='identifiable'):fit_private_eventset_streaming(factory(ex),factory(ex),event_feature_dim=16,core_feature_dim=6,seed=1,epochs=1,hidden_dim=8)

def test_evidence_evaluation_not_a_fitter_argument():
    import inspect
    assert 'evaluation_factory' not in inspect.signature(
        fit_private_eventset_streaming
    ).parameters
    tree = ast.parse(
        (ROOT/'scripts/run_v32_evidence_streaming_training.py').read_text()
    )
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'fit_private_eventset_streaming'
    ]
    assert len(calls) == 1
    call = calls[0]
    assert [
        arg.args[0].value
        for arg in call.args[:2]
        if isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Name)
        and arg.func.id == 'factory'
        and len(arg.args) == 1
        and isinstance(arg.args[0], ast.Constant)
    ] == ['train', 'validation']
    assert not any(
        isinstance(node, ast.Constant) and node.value == 'evaluation'
        for node in ast.walk(call)
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords['checkpoint_steps'], ast.Attribute)
    assert keywords['checkpoint_steps'].attr == 'checkpoint_steps'

def test_graph_grid_exactly_15(tmp_path):
    cfg=read_json(ROOT/'config/149.json');jobs=job_grid(cfg,tmp_path)
    assert len(jobs)==15 and len({j['prepared'] for j in jobs})==15
    assert {(j['variant'],j['fold']) for j in jobs}=={(v,f) for v in ('G0','G1','G2') for f in range(5)}
    original=copy.deepcopy(cfg);vc=variant_config(cfg,'G0');assert cfg==original and vc['graph']['variant']=='G0'

def test_task_weights_every_active_config():
    import yaml
    cfg=read_json(ROOT/'config/149.json');full=yaml.safe_load((ROOT/'config/model_v3_2_full_multitask.yaml').read_text())
    assert cfg['model']['task_weights']==TASK_WEIGHTS=={'pathway':1.,'state':.5}
    assert full['primary_task_weights']==TASK_WEIGHTS
    assert full['shared_core']['state_gradients_into_core'] is True
    assert full['modules']['state']['initialization_policy'].startswith('JOINT_')
    assert weighted_task_selection(.2,.8)==pytest.approx(.4)

def test_original_guard_all_parameters_not_single_probe():
    class M(torch.nn.Module):
        def __init__(self):super().__init__();self.small=torch.nn.Parameter(torch.tensor([1e15]));self.vector=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    m=M();opt=torch.optim.SGD(m.parameters(),lr=.01);(m.small.sum()*1e-12+m.vector.sum()).backward()
    proof=optimizer_step_with_global_delta_guard(m,opt,torch=torch,objective_value=3.)
    assert proof['parameter_delta_positive'] and proof['update_probe_parameter']=='vector'
    assert proof['trainable_parameter_tensors']==2

def test_original_guard_rejects_no_update():
    m=torch.nn.Linear(2,1);m(torch.ones(2)).sum().backward();opt=torch.optim.SGD(m.parameters(),lr=0.)
    with pytest.raises(RuntimeError,match='ZERO_PARAMETER_DELTA'):optimizer_step_with_global_delta_guard(m,opt,torch=torch,objective_value=1.)

def test_original_guard_source_is_unchanged():
    # The guard is shared with the active public encoder; a fixed source hash
    # records this invariant without retaining historical Python packages.
    expected='1b32927b591bdc4aee81574b6fcb86863f78d28ba8eb22d9cb7f1a21dc1a54f1'
    assert hashlib.sha256((ROOT/'cc_hhgt/v32/optimizer_guard_v2.py').read_bytes()).hexdigest()==expected



def test_current_encoder_has_no_vendored_source_tree():
    assert not (ROOT/'code/cc_hhgt').exists()
    from cc_hhgt import gnn
    assert Path(gnn.__file__).resolve()==(ROOT/'cc_hhgt/gnn.py').resolve()
    # Numerical forward/gradient equivalence is checked by the executable
    # backend regression, including zero context and both joint task losses.



def test_no_native_model_fallback():
    s=(ROOT/'cla_engine/model.py').read_text();tree=ast.parse(s)
    assert not any(isinstance(n,ast.ClassDef) and n.name=='HGTLayer' for n in ast.walk(tree))
    assert 'build_model(' in s and 'PYG_REQUIRED_NO_FALLBACK' in s
    assert 'r4_context_embedding' in s

def test_old_prepared_format_is_rejected(tmp_path):
    atomic_json(tmp_path/'MANIFEST.json',{'format':'CLA_R3_COMMITTED_V1','files':{},'metadata':{}})
    with pytest.raises(Stop) as caught:verify_manifest(tmp_path)
    assert caught.value.code == 'WRONG_GENERATION'

@pytest.mark.parametrize('name',sorted(RECIPES))
def test_all_module_registry_flags_are_real(name):
    r=RECIPES[name];flags=parser_flags(SCRIPTS/r['script'])
    expected=set(r['paths'])|set(r.get('optional_paths',{}))|set(r['values'])|{r.get('output_flag','--output-root')}
    assert expected<=flags

def test_evidence_not_replaced_by_audit_only():
    assert RECIPES['evidence_train']['script']=='run_v32_evidence_streaming_training.py'
    assert RECIPES['evidence_predict']['script']=='run_v32_evidence_streaming_predict.py'
    assert RECIPES['evidence_train']['resume'] is True
    assert (ROOT/'cc_hhgt/v32/evidence_training.py').is_file()

@pytest.mark.skipif(importlib.util.find_spec('torch_geometric') is None,reason='Real PyG not installed here; backend test must pass on target')
def test_real_pyg_equivalence_and_joint_gradient(tmp_path):
    from cla_engine.backend_check import run
    result=run(tmp_path,'cpu');assert result['cancer_is_graph_node'];assert result['global_representable_delta_guard']['parameter_delta_positive']

@pytest.mark.parametrize('name',sorted(RECIPES))
def test_registered_runner_required_literal_flags_are_bound(name):
    r=RECIPES[name];tree=ast.parse((SCRIPTS/r['script']).read_text())
    required=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            if any(k.arg=='required' and isinstance(k.value,ast.Constant) and k.value.value is True for k in node.keywords):
                required|={a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--')}
    provided=set(r['paths'])|set(r['values'])|set(r.get('hash_roles',{}))|{r.get('output_flag','--output-root')}
    if name=='evidence_stage':provided|={'--expected-hashes-json'}
    if name in ['evidence_train','evidence_predict']:provided|={'--expected-core-manifest-sha256','--expected-graph-authority-receipt-sha256','--deadline'}
    if name=='interaction':provided|={'--evidence-semantic-wrapper-sha256'}
    assert required<=provided

@pytest.mark.parametrize('result,code',[
    ({'state':'MODULE_FAILED'},70),({'state':'MODULE_BUDGET_EXHAUSTED'},75),
    ({'state':'INSTANCE_RELEASE_REQUIRED','worker_terminal_state':'ERROR'},70),
    ({'state':'INSTANCE_RELEASE_REQUIRED','worker_terminal_state':'NO_GO_WITHIN_DIAGNOSTIC_BUDGET'},3),
    ({'state':'INSTANCE_RELEASE_REQUIRED','worker_terminal_state':'GO_TO_FORMAL_EVALUATION'},0),
    ({'state':'QUEUE_COMPLETE_RELEASE_INSTANCE','prediction_gates':{'G2':'FORMAL_GATE_FAIL'}},3),
])
def test_child_failure_is_not_successful_exit(result,code):
    from cla import result_exit_code
    assert result_exit_code(result)==code

def test_dynamic_argparse_flags_are_detected_without_import(tmp_path):
    p=tmp_path/'script.py';p.write_text("for name in ('stage-root','output-root'):\n p.add_argument('--'+name, required=True)\n")
    assert parser_flags(p)=={'--stage-root','--output-root'}

def test_config_rejects_native_backend(tmp_path):
    from cla_engine.common import config
    cfg=read_json(ROOT/'config/149.json');cfg['model']['backend']='native_hgt'
    path=tmp_path/'config.json';atomic_json(path,cfg)
    with pytest.raises(Stop) as caught:config(path)
    assert caught.value.code=='BACKBONE_SUBSTITUTION_FORBIDDEN'

def test_core_export_generation_cannot_be_relabelled(tmp_path):
    from cc_hhgt.v32.core_export_contract_r4 import validate_export
    p=tmp_path/'CORE_EMBEDDING_MANIFEST.json';atomic_json(p,{'export_format':'CLA_R3'})
    with pytest.raises(RuntimeError,match='R62_CORE_CONTRACT'):validate_export(p)
