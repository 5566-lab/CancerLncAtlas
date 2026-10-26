"""Reaggregate complete canonical metacells into original fold graph assets."""
from pathlib import Path
import math
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from . import native_single_cell_donors_workflow as donors
from . import native_single_cell_static_workflow as static

ROOT = Path(__file__).resolve().parents[2]
INPUT = 'NATIVE_SINGLE_CELL_GRAPH_INPUT_V1'
PREFLIGHT = 'NATIVE_SINGLE_CELL_GRAPH_PREFLIGHT_V1'
FORMAT = 'NATIVE_SINGLE_CELL_GRAPH_RESULTS_V1'
CODE = ('cc_hhgt/v32/native_single_cell_graph_workflow.py', 'scripts/run_native_single_cell_graph_workflow.py',
        'scripts/materialize_single_cell_hhgt_assets_149.py', 'cc_hhgt/v32/single_cell_graph_assets.py',
        'scripts/materialize_single_cell_hhgt_training_episodes_149.py')
_CACHE = {}


def plan(manifest, output, *, new=True):
    from scripts import materialize_single_cell_metacell_149 as base
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if set(spec) != {'format', 'donor_preparation', 'metacell_success', 'static_workflow'} or spec['format'] != INPUT:
        raise ValueError('Invalid native SC fold graph manifest')
    if new and output.exists(): raise FileExistsError('Use a new SC fold graph output')
    donor_path, donor_binding = donors.validate(spec['donor_preparation'], manifest.parent)
    donor_value = read(donor_binding['preflight']['path'])
    static_path, static_binding = static.validate(spec['static_workflow'], manifest.parent)
    static_value = read(static_binding['preflight']['path'])
    if donor_value['graph_preflight'] != static_binding['preflight'] or donor_value['mapping'] != static_value['mapping']:
        raise ValueError('SC donor/static preparations have different sources')
    cancer = donor_value['cancer_id']; success = verified_path(spec['metacell_success'], manifest.parent)
    if success != donor_path.parent / 'metacells/cancers' / cancer / 'SUCCESS.json':
        raise ValueError('SC metacells belong to another preparation')
    for source in (donor_path.parent, static_path.parent):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('SC graph output overlaps a source directory')
    if output == manifest or output in manifest.parents: raise ValueError('SC graph output contains its manifest')
    settings, _ = donors.build_metacell_settings(donor_value, donor_path.parent)
    completed = base.validate_completed_cancer(settings)
    sources = [record(success), *[record(p['path']) for p in completed['files'].values()]]
    folds = sorted(map(int, completed['fold_receipt_sha256']))
    for fold in folds:
        path = success.parent / f'outer_fold={fold}/SUCCESS.json'; receipt = read(path)
        sources += [record(path), *[record(p['path']) for p in receipt['files'].values()]]
    return {'format': PREFLIGHT, 'target_host':'149', 'training_started':False, 'input':record(manifest),
            'donor_preparation':record(donor_path), 'static_workflow':record(static_path),
            'metacell_success':record(success), 'cancer_id':cancer, 'folds':folds,
            'mapping':static_value['mapping'], 'mapped_success':static_value['mapped_success'],
            'pathway_family':static_value['pathway_family'], 'external_evidence':static_value['external_evidence'],
            'source_files':sources, 'implementations':[record(ROOT/name) for name in CODE],
            'source_ancestry_complete':False, 'geometry_recomputed':False}


def settings(value, output, fold, deadline):
    from scripts import materialize_single_cell_hhgt_assets_149 as native
    return native.parser().parse_args(['--cancer-id', value['cancer_id'], '--outer-fold', str(fold),
        '--source-result-root', str(Path(value['metacell_success']['path']).parents[2]), '--output-root', str(output),
        '--scratch-root', str(Path(value['mapped_success']['path']).parents[1]),
        '--pathway-family', value['pathway_family']['path'], '--external-evidence', value['external_evidence']['path'],
        '--deadline', str(deadline)])


def context(arguments, value):
    require_host(); output = arguments.output_root.resolve()
    if value != plan(verified_path(value['input'], output), output, new=False):
        raise ValueError('SC graph sources or implementation differ')
    if (not math.isfinite(arguments.deadline) or arguments.deadline <= 0
            or int(arguments.outer_fold) not in value['folds']):
        raise ValueError('SC graph deadline or fold is invalid')
    expected = settings(value, output, arguments.outer_fold, arguments.deadline)
    expected.outer_fold = arguments.outer_fold
    if vars(expected) != vars(arguments): raise ValueError('SC graph execution configuration differs')
    donor_value = read(read(value['donor_preparation']['path'])['preflight']['path'])
    return donors.derived(donor_value)[0]


def invoke(value, output):
    import time
    from scripts import materialize_single_cell_hhgt_assets_149 as native
    output.mkdir(parents=True, exist_ok=True); write(output / 'LAUNCH_PREFLIGHT.json', value)
    for fold in value['folds']:
        native.materialize_fold(settings(value, output, fold, time.time()+3600))


def fold_roots(value, output):
    return [output / 'cancers' / value['cancer_id'] / f'outer_fold={fold}' for fold in value['folds']]


def inventory(value, output):
    _, pins = static.inventories(value, output)
    for root in fold_roots(value, output):
        receipt = read(root / 'SUCCESS.json'); pins.append(record(root / 'SUCCESS.json'))
        if receipt.get('native_graph_preparation') != {**record(output/'LAUNCH_PREFLIGHT.json'), 'bytes':(output/'LAUNCH_PREFLIGHT.json').stat().st_size}:
            raise ValueError('SC fold graph preparation differs')
        for name, pin in receipt['files'].items():
            path = verified_path(pin, root)
            if path != root/name or pin.get('bytes') != path.stat().st_size: raise ValueError('SC fold graph file identity differs')
            pins.append(record(path))
    return pins


def replay(value, output):
    static.replay(value, output)
    with tempfile.TemporaryDirectory(prefix='native_sc_graph_replay_') as temporary:
        other = Path(temporary); invoke(value, other)
        for left_root, right_root in zip(fold_roots(value, output), fold_roots(value, other)):
            left, right = read(left_root/'SUCCESS.json'), read(right_root/'SUCCESS.json')
            omit = {'files', 'finished_unix', 'native_graph_preparation', 'cancer_static_success_path', 'cancer_static_success_sha256'}
            if {k:v for k,v in left.items() if k not in omit} != {k:v for k,v in right.items() if k not in omit}:
                raise ValueError('SC fold graph metadata differs from frozen-source reconstruction')
            target = output/'cancers'/value['cancer_id']/'STATIC/SUCCESS.json'
            if left['cancer_static_success_path'] != str(target) or left['cancer_static_success_sha256'] != record(target)['sha256']:
                raise ValueError('SC fold static source differs')
            if set(left['files']) != set(right['files']): raise ValueError('SC fold graph inventory differs')
            for name in right['files']:
                a,b = left_root/name, right_root/name
                if name.endswith('.npz'):
                    x,y = sparse.load_npz(a),sparse.load_npz(b)
                    if x.shape != y.shape or x.dtype != y.dtype or (x!=y).nnz: raise ValueError('SC fold graph matrix reconstruction differs')
                elif name.endswith('.npy'):
                    x,y = np.load(a,allow_pickle=False),np.load(b,allow_pickle=False)
                    if x.dtype != y.dtype or not np.array_equal(x,y): raise ValueError('SC fold graph activity reconstruction differs')
                elif name.endswith('.parquet'):
                    try: pd.testing.assert_frame_equal(pd.read_parquet(a),pd.read_parquet(b),check_exact=True)
                    except AssertionError as error: raise ValueError('SC fold graph table reconstruction differs') from error
                elif name == 'STATIC_BINDING.json':
                    from scripts.materialize_single_cell_hhgt_assets_149 import file_record
                    if read(a) != {'format':read(b)['format'], 'cancer_static_success':file_record(target)}:
                        raise ValueError('SC fold static binding differs')
                elif read(a) != read(b): raise ValueError('SC fold graph target binding reconstruction differs')


def envelopes(value, output):
    from scripts.materialize_single_cell_hhgt_assets_149 import file_record, ASSET_FORMAT
    cancer_root = output/'cancers'/value['cancer_id']
    cancer = {'format':'CANCERLNCATLAS_V32_SC_HHGT_CANCER_SUCCESS_V1', 'state':'CANCER_GRAPH_ASSETS_COMPLETE',
        'target_host':'149', 'cancer_id':value['cancer_id'], 'source_cancer_success':file_record(Path(value['metacell_success']['path'])),
        'cancer_static_success':file_record(cancer_root/'STATIC/SUCCESS.json'), 'outer_fold_ids':value['folds'],
        'outer_fold_successes':[file_record(p/'SUCCESS.json') for p in fold_roots(value,output)],
        'launch_preflight':file_record(output/'LAUNCH_PREFLIGHT.json'), 'formal_pointer_update_authorized':False,
        'native_graph_preparation':file_record(output/'LAUNCH_PREFLIGHT.json')}
    manifest = {'format':ASSET_FORMAT, 'state':'GRAPH_ASSETS_COMPLETE', 'target_host':'149',
        'cancers':[value['cancer_id']], 'cancer_count':1, 'cancer_static_receipts':[cancer['cancer_static_success']],
        'fold_receipts':cancer['outer_fold_successes'], 'fold_count':len(value['folds']),
        'preflight':cancer['launch_preflight'], 'native_graph_preparation':cancer['native_graph_preparation']}
    return cancer,manifest


def validate_outputs(output):
    require_host(); output=Path(output).resolve(); value=read(output/'LAUNCH_PREFLIGHT.json')
    if value != plan(verified_path(value['input'],output),output,new=False): raise ValueError('SC graph input/code drift')
    pins = inventory(value,output)
    cancer,manifest = envelopes(value,output)
    if read(output/'cancers'/value['cancer_id']/'SUCCESS.json') != cancer or read(output/'GRAPH_ASSET_MANIFEST.json') != manifest:
        raise ValueError('SC graph completion envelope differs')
    signature={'value':value,'outputs':pins,'cancer':cancer,'manifest':manifest}
    if _CACHE.get(str(output)) != signature:
        replay(value,output); _CACHE[str(output)]=signature
    return value


def run(manifest, output):
    value=plan(manifest,output); output=Path(output).resolve()
    try:
        invoke(value,output); inventory(value,output)
        cancer,graph=envelopes(value,output)
        write(output/'cancers'/value['cancer_id']/'SUCCESS.json',cancer); write(output/'GRAPH_ASSET_MANIFEST.json',graph)
        path=output/'NATIVE_SINGLE_CELL_GRAPH_BINDING.json'
        write(path,{'format':FORMAT,'status':'PASS','target_host':'149','training_started':False,
            'preflight':record(output/'LAUNCH_PREFLIGHT.json'),'graph_manifest':record(output/'GRAPH_ASSET_MANIFEST.json'),
            'geometry_recomputed':False,'lasso_fitted':False,'source_ancestry_complete':False,'frozen_source_reconstruction_verified':True})
        validate(record(path),output); return path
    except Exception as error:
        if output.exists(): write(output/'RUN_FAILED.json',{'target_host':'149','training_started':False,'error':str(error)})
        raise


def validate(item, root):
    require_host(); path=verified_path(item,root); bound=read(path)
    if (bound.get('format') != FORMAT or bound.get('status') != 'PASS' or bound.get('target_host') != '149'
            or bound.get('frozen_source_reconstruction_verified') is not True
            or any(bound.get(k) is not False for k in ('training_started','geometry_recomputed','lasso_fitted','source_ancestry_complete'))):
        raise ValueError('Invalid native SC fold graph binding')
    if verified_path(bound['preflight'],path.parent) != path.parent/'LAUNCH_PREFLIGHT.json' or verified_path(bound['graph_manifest'],path.parent) != path.parent/'GRAPH_ASSET_MANIFEST.json':
        raise ValueError('SC graph binding location differs')
    validate_outputs(path.parent); return path,bound
