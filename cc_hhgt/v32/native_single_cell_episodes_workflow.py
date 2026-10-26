"""Prepare canonical episode assembly and collect separately completed LASSO assets."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import math
import re
import sys

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from . import native_single_cell_episode_sources as sources
from . import single_cell_numeric_replay as numeric

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'scripts.materialize_single_cell_hhgt_training_episodes_149'
INPUT = 'NATIVE_SINGLE_CELL_EPISODES_INPUT_V1'
PREPARED = 'NATIVE_SINGLE_CELL_EPISODES_PREPARED_V1'
FORMAT = 'NATIVE_SINGLE_CELL_EPISODES_RESULTS_V1'
CODE = (*numeric.CODE, 'cc_hhgt/v32/native_single_cell_episodes_workflow.py',
        'scripts/run_native_single_cell_episodes_workflow.py',
        'scripts/materialize_single_cell_hhgt_training_episodes_149.py',
        'cc_hhgt/v32/native_single_cell_episode_sources.py',
        'cc_hhgt/v32/single_cell_hhgt_episodes.py', 'cc_hhgt/v32/single_cell_batched_lasso.py')
INTEGERS = {'max_genes', 'min_donors', 'inner_folds', 'bootstraps', 'response_batch_size', 'max_iter'}


def arguments(spec, graph_path, output):
    """Build only arguments accepted by the original parser; never shell text."""
    overrides = spec.get('lasso_config', {})
    if not isinstance(overrides, dict) or set(overrides)-INTEGERS-{'alphas', 'tolerance', 'stability_threshold'}:
        raise ValueError('Unknown episode LASSO configuration')
    extras = []
    for key, value in sorted(overrides.items()):
        if key == 'alphas':
            if (not isinstance(value, list) or not value
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in value)):
                raise ValueError('Episode LASSO alphas must be finite positive numbers')
            value = ','.join(map(str, value))
        elif key in INTEGERS:
            if type(value) is not int or value <= 0: raise ValueError('Episode LASSO dimensions must be positive integers')
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Episode LASSO scalars must be finite')
        extras += ['--lasso-'+key.replace('_','-'), str(value)]
    graph_value = read(read(graph_path)['preflight']['path'])
    donor = Path(graph_value['donor_preparation']['path']).parent
    return ['--run-id', spec['run_id'], '--source-result-root', str(donor/'metacells'),
        '--graph-asset-root', str(graph_path.parent), '--nested-asset-root', str(donor/'nested'),
        '--native-graph-binding', str(graph_path), '--output-root', str(output/'episodes'),
        '--cancer-id', graph_value['cancer_id'], '--deadline', '1.0', *extras]


def settings(value, *, finalize=False):
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    argv = list(value['prepare_command'][3:]); argv.remove('--prepare-only')
    if finalize:
        index = argv.index('--cancer-id'); del argv[index:index+2]; argv.append('--finalize')
    return native.settings_from_args(native.parser().parse_args(argv))


def plan(manifest, output, *, new=True):
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    required = {'format', 'run_id', 'graph_workflow'}
    if (not required <= set(spec) or set(spec)-required-{'lasso_config'} or spec['format'] != INPUT
            or not isinstance(spec['run_id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', spec['run_id'])):
        raise ValueError('Invalid native episode workflow manifest')
    if new and output.exists(): raise FileExistsError('Use a new native episode preparation directory')
    if output == manifest or output in manifest.parents: raise ValueError('Episode output contains its input manifest')
    graph_path = verified_path(spec['graph_workflow'], manifest.parent)
    argv = arguments(spec, graph_path, output)
    current = native.settings_from_args(native.parser().parse_args(argv))
    bindings = sources.preflight_sources(current)
    for source in (graph_path.parent, Path(bindings['native_donor_preparation']['path']).parent):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Episode preparation overlaps its source directory')
    fit = [sys.executable, '-m', MODULE, *argv]
    fit[fit.index('--deadline')+1] = '<UNIX_DEADLINE>'
    finish = list(fit); index = finish.index('--cancer-id'); del finish[index:index+2]; finish.append('--finalize')
    commands = {name:{'argv':command, 'automatically_executed':False, 'requires_separate_explicit_execution':True,
                      'fits_lasso':name=='assemble', 'fits_neural_model':False}
                for name,command in (('assemble',fit),('finalize',finish))}
    return {'target_host':'149', 'training_started':False, 'input':record(manifest), 'run_id':spec['run_id'],
        'graph_workflow':record(graph_path), 'source_bindings':bindings,
        'prepare_command':[sys.executable,'-m',MODULE,*argv,'--prepare-only'], 'execution_commands':commands,
        'lasso':native._jsonable(native.lasso_policy_receipt(current.lasso_policy)),
        'lasso_stability_threshold':current.lasso_stability_threshold,
        'implementations':[record(ROOT/name) for name in CODE], 'source_ancestry_complete':False}


def execute_prepare(command, log):
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    if command[:3] != [sys.executable,'-m',MODULE] or command.count('--prepare-only') != 1:
        raise ValueError('Episode workflow may dispatch only prepare-only')
    with Path(log).open('w',encoding='utf-8') as handle, redirect_stdout(handle), redirect_stderr(handle):
        if native.main(command[3:]) != 0: raise RuntimeError('Native episode preparation failed')


def prepare(manifest, output):
    value = plan(manifest,output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output/'LAUNCH_PREFLIGHT.json', value)
    try:
        execute_prepare(value['prepare_command'], output/'prepare.log')
        write(output/'EXECUTION_COMMANDS.json', value['execution_commands'])
        path = output/'PREPARED.json'
        write(path, {'format':PREPARED,'status':'PASS_READY','target_host':'149','training_started':False,
            'lasso_fitted':False,'plan':record(output/'LAUNCH_PREFLIGHT.json'),
            'episode_preflight':record(output/'episodes/LAUNCH_PREFLIGHT.json'),
            'execution_commands':record(output/'EXECUTION_COMMANDS.json')})
        validate_preparation(record(path),output); return path
    except Exception as error:
        write(output/'RUN_FAILED.json',{'target_host':'149','training_started':False,'error':str(error)}); raise


def validate_preparation(item, root):
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    require_host(); path = verified_path(item,root); bound = read(path)
    if (bound.get('format') != PREPARED or bound.get('status') != 'PASS_READY' or bound.get('target_host') != '149'
            or bound.get('training_started') is not False or bound.get('lasso_fitted') is not False):
        raise ValueError('Invalid native episode preparation binding')
    launch = verified_path(bound['plan'],path.parent); value = read(launch)
    if launch != path.parent/'LAUNCH_PREFLIGHT.json' or value != plan(verified_path(value['input'],path.parent),path.parent,new=False):
        raise ValueError('Native episode preparation sources or configuration differ')
    preflight = verified_path(bound['episode_preflight'],path.parent)
    command = verified_path(bound['execution_commands'],path.parent)
    if preflight != path.parent/'episodes/LAUNCH_PREFLIGHT.json' or command != path.parent/'EXECUTION_COMMANDS.json':
        raise ValueError('Native episode preparation locations differ')
    if read(command) != value['execution_commands']: raise ValueError('Native episode execution commands differ')
    actual = native._initialize_preflight(settings(value),read_only=True)
    if actual['bindings'] != value['source_bindings']: raise ValueError('Native episode input snapshot differs')
    return path,bound,value


def completion(prepared, value, item):
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as native
    path = verified_path(item,prepared.parent)
    if path != prepared.parent/'episodes/SC_HHGT_EPISODE_MANIFEST.json':
        raise ValueError('Completed episodes are outside the prepared output')
    expected = native.finalize(settings(value,finalize=True),read_only=True)
    observed = read(path)
    if {k:v for k,v in observed.items() if k!='finished_unix'} != {k:v for k,v in expected.items() if k!='finished_unix'}:
        raise ValueError('Completed episode manifest differs from original read-only finalization')
    return path


def input_collection(value):
    from scripts import train_single_cell_hhgt_149 as native
    current = settings(value)
    config = native.TrainingSettings(episode_root=current.output_root,graph_assets_root=current.graph_asset_root,
                                    output_root=current.output_root.parent/'input_inspection_only',prepare_only=True)
    descriptors,lineage = native.load_episode_manifest(config)
    return numeric.collect_input_preflight(config,descriptors,lineage)


def result_value(prepared, preparation, value, completed, audit):
    donor = Path(value['source_bindings']['native_donor_preparation']['path'])
    donor_value = read(read(donor)['preflight']['path'])
    graph = Path(value['graph_workflow']['path']).parent
    files = {'episode_manifest':record(completed), 'graph_asset_manifest':record(graph/'GRAPH_ASSET_MANIFEST.json'),
        'candidates':record(verified_path(donor_value['candidates'],donor.parent)),
        'donor_crosswalk':record(donor.parent/'CANONICAL_DONOR_CROSSWALK.parquet')}
    pins = [record(prepared),record(completed),record(audit),*files.values(),*value['implementations'],value['input'],
        *[preparation[k] for k in ('plan','episode_preflight','execution_commands')],
        *value['source_bindings']['nested_sources'],
        *[value['source_bindings'][k] for k in ('native_graph_binding','native_donor_preparation','nested_preflight','source_validator')],
        *read(audit)['source_files']]
    inventory = {str(verified_path(p,prepared.parent)):record(verified_path(p,prepared.parent)) for p in pins}
    return {'format':FORMAT,'status':'PASS','target_host':'149','training_started':False,'lasso_fitted':False,
        'preparation':record(prepared),'completed':record(completed),'input_preflight':record(audit),'files':files,
        'both_arm_input_graphs_loaded':True,'model_constructed':False,'source_ancestry_complete':False,
        'fitted_lasso_numerics_verified':False,'source_files':sorted(inventory.values(),key=lambda p:p['path'])}


def collect(preparation, completed, output):
    require_host(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a new native episode collection directory')
    prepared,bound,value = validate_preparation(preparation,ROOT)
    for source in (prepared.parent, Path(value['graph_workflow']['path']).parent,
                   Path(value['source_bindings']['native_donor_preparation']['path']).parent):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Episode collection overlaps an immutable source directory')
    path = completion(prepared,value,completed); inspection = input_collection(value)
    output.mkdir(parents=True)
    write(output/'EPISODE_INPUT_PREFLIGHT.json',inspection)
    result = output/'NATIVE_SINGLE_CELL_EPISODES_BINDING.json'
    write(result,result_value(prepared,bound,value,path,output/'EPISODE_INPUT_PREFLIGHT.json'))
    return result


def validate(item, root):
    require_host(); path = verified_path(item,root); bound = read(path)
    prepared,preparation,value = validate_preparation(bound['preparation'],path.parent)
    completed = completion(prepared,value,bound['completed'])
    audit = verified_path(bound['input_preflight'],path.parent)
    if audit != path.parent/'EPISODE_INPUT_PREFLIGHT.json' or read(audit) != input_collection(value):
        raise ValueError('Native episode input inspection differs')
    if bound != result_value(prepared,preparation,value,completed,audit):
        raise ValueError('Native episode collection binding differs')
    return path,bound
