"""Actual prepare/wait/collect/SC-input chain with frozen test-only LASSO."""
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from cc_hhgt.v32 import native_single_cell_episodes_workflow as workflow
from cc_hhgt.v32 import native_single_cell_workflow as training
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, write, record
from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
from scripts import train_single_cell_hhgt_149 as trainer
from test_v32_native_single_cell_episode_sources import nested_inputs, frozen_lasso
from test_v32_native_single_cell_graph_workflow import completed as graph_completed
from test_v32_canonical_metacell_consumer import no_fit
from test_v32_native_single_cell_mapping_workflow import forbidden


def ref(name, pointer=''):
    return {'stage_ref':name,'pointer':pointer}


@pytest.fixture(scope='module')
def completed(nested_inputs):
    root, donor, _, graph_pin, _ = nested_inputs
    spec={'format':workflow.INPUT,'run_id':'native-episode-chain','graph_workflow':graph_pin,
          'lasso_config':{'alphas':[.2,.05],'stability_threshold':.6}}
    manifest=root/'EPISODE_WORKFLOW.json';write(manifest,spec)
    train={'format':training.INPUT,'run_id':'native-episode-training',
           **{key:ref('collected','/files/'+key) for key in training.FILES},
           'episode_workflow':ref('collected')}
    write(root/'EPISODE_CHAIN.json',{'format':chain.INPUT,'chain_id':'native-episodes','stages':[
        {'id':'prepared','operation':'single_cell_episodes.prepare','inputs':spec},
        {'id':'collected','operation':'single_cell_episodes.collect','inputs':{
            'preparation':ref('prepared'),'completed':{'external_ref':'episodes'}}},
        {'id':'training-inputs','operation':'single_cell.prepare','inputs':train}]})
    output=root/'episode-chain';chain.initialize(root/'EPISODE_CHAIN.json',output)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native,'fit_context_lasso',forbidden)
        chain.advance(output)
    state=read(output/'CHAIN.json');assert state['status']=='WAITING_EXTERNAL',state
    assert set(state['stages'])=={'prepared'}
    prepared=Path(state['results']['prepared']['path']);value=read(read(prepared)['plan']['path'])
    assert not (prepared.parent/'episodes/episodes').exists()
    assert not (prepared.parent/'episodes/static').exists()
    commands=read(prepared.parent/'EXECUTION_COMMANDS.json')
    assert commands['assemble']['fits_lasso'] and not commands['finalize']['fits_lasso']
    for command in commands.values():
        assert not command['automatically_executed']
        argv=[str(time.time()+600) if arg=='<UNIX_DEADLINE>' else arg for arg in command['argv'][3:]]
        native.settings_from_args(native.parser().parse_args(argv))
    settings=workflow.settings(value);settings.deadline=time.time()+600
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native,'fit_context_lasso',frozen_lasso)
        native.run_cancer(settings)
    settings.cancer_id=None;settings.finalize=True;native.finalize(settings)
    complete=record(settings.output_root/'SC_HHGT_EPISODE_MANIFEST.json')
    before={str(p):(p.stat().st_mtime_ns,record(p)['sha256']) for p in prepared.parent.rglob('*') if p.is_file()}
    registry=root/'EPISODE_EXTERNAL.json';write(registry,{'format':chain.EXTERNAL,'inputs':{'episodes':complete}})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native,'fit_context_lasso',forbidden)
        patch.setattr(trainer,'_initialize_model',forbidden)
        chain.advance(output,registry)
    state=read(output/'CHAIN.json');assert state['status']=='DECLARED_STAGES_COMPLETE',state
    assert before=={str(p):(p.stat().st_mtime_ns,record(p)['sha256']) for p in prepared.parent.rglob('*') if p.is_file()}
    assert not (prepared.parent/'input_inspection_only').exists()
    return root,manifest,output,prepared,value,complete,state


def test_collected_native_episodes_reach_actual_sc_prepare_and_resume(completed,monkeypatch):
    _,_,output,prepared,_,complete,state=completed
    binding=read(state['results']['collected']['path'])
    assert binding['preparation']==record(prepared) and binding['completed']==complete
    assert binding['both_arm_input_graphs_loaded'] and not binding['fitted_lasso_numerics_verified']
    training_binding=read(state['results']['training-inputs']['path'])
    plan=read(training_binding['plan']['path'])
    assert plan['episode_workflow']==state['results']['collected']
    assert plan['source_boundary']=='COLLECTED_CANONICAL_EPISODES_WITH_FROZEN_LASSO'
    assert {p['path'] for p in binding['source_files']} <= {p['path'] for p in plan['sources']}
    ready=read(training_binding['input_preflight']['path'])
    assert len(ready['models'])==10 and not ready['model_constructed'] and not ready['training_started']
    monkeypatch.setattr(chain,'dispatch',forbidden)
    chain.advance(output)
    assert read(output/'CHAIN.json')['stages']==state['stages']


@pytest.mark.parametrize('fault',['counts','folds'])
def test_repinned_completion_manifest_is_rederived_from_actual_assets(completed,fault):
    _,_,_,prepared,value,complete,_=completed
    path=Path(complete['path']);old=path.read_bytes()
    try:
        manifest=read(path)
        if fault=='counts':manifest['role_row_counts']['fit']+=1
        else:manifest['folds']=manifest['folds'][:-1]
        write(path,manifest)
        with pytest.raises(ValueError,match='read-only finalization'):
            workflow.completion(prepared,value,record(path))
    finally:path.write_bytes(old)


def test_same_completion_bytes_outside_preparation_are_rejected(completed,tmp_path):
    _,_,_,prepared,value,complete,_=completed
    copy=tmp_path/'SC_HHGT_EPISODE_MANIFEST.json';copy.write_bytes(Path(complete['path']).read_bytes())
    with pytest.raises(ValueError,match='outside the prepared output'):
        workflow.completion(prepared,value,record(copy))


def test_repinned_execution_commands_cannot_replace_prepared_policy(completed):
    _,_,_,prepared,_,_,_=completed
    command=prepared.parent/'EXECUTION_COMMANDS.json';old={p:p.read_bytes() for p in (prepared,command)}
    try:
        value=read(command);value['assemble']['automatically_executed']=True;write(command,value)
        bound=read(prepared);bound['execution_commands']=record(command);write(prepared,bound)
        with pytest.raises(ValueError,match='commands differ'):workflow.validate_preparation(record(prepared),prepared.parent)
    finally:
        for p,data in old.items():p.write_bytes(data)


def test_read_only_verification_never_creates_missing_preflight(completed,tmp_path):
    _,_,_,_,value,_,_=completed
    args=workflow.settings(value);args.output_root=tmp_path/'missing'
    with pytest.raises(RuntimeError,match='READ_ONLY_EPISODE_PREFLIGHT_MISSING'):
        native._initialize_preflight(args,read_only=True)
    assert not args.output_root.exists()


def test_dispatch_rejects_fitting_command_without_prepare_only(completed,tmp_path):
    _,_,_,_,value,_,_=completed
    with pytest.raises(ValueError,match='only prepare-only'):
        workflow.execute_prepare(value['execution_commands']['assemble']['argv'],tmp_path/'log')
    assert not (tmp_path/'log').exists()


@pytest.mark.parametrize('config',[{'min_donors':True},{'alphas':[float('nan')]},
                                 {'tolerance':float('inf')},{'unrecognized':1}])
def test_unvalidated_configuration_fails_before_output(completed,tmp_path,config):
    _,manifest,_,_,_,_,_=completed
    spec=read(manifest);spec['lasso_config']=config;path=tmp_path/'INPUT.json';write(path,spec)
    with pytest.raises(ValueError):workflow.plan(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
