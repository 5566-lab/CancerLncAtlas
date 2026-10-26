"""Build original static SC graph assets from explicit mapped/raw references."""
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from . import native_single_cell_mapping_workflow as mapping

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_SINGLE_CELL_STATIC_INPUT_V1"
FORMAT = "NATIVE_SINGLE_CELL_STATIC_RESULTS_V1"
CODE = ("cc_hhgt/v32/native_single_cell_static_workflow.py", "scripts/run_native_single_cell_static_workflow.py",
        "scripts/materialize_single_cell_hhgt_assets_149.py", "cc_hhgt/v32/single_cell_graph_assets.py")
_CACHE = {}


def plan(manifest, output, *, new=True):
    from scripts import materialize_single_cell_hhgt_assets_149 as native
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if set(spec) != {"format", "mapping", "pathway_family", "external_evidence"} or spec['format'] != INPUT:
        raise ValueError("Invalid native single-cell static manifest")
    if new and output.exists(): raise FileExistsError("Use a new SC static output")
    path, bound = mapping.validate(spec['mapping'], manifest.parent)
    sources = {key: verified_path(spec[key], manifest.parent) for key in ('pathway_family', 'external_evidence')}
    if (output == path.parent or output in path.parent.parents or path.parent in output.parents
            or any(output == p or output in p.parents for p in (manifest, *sources.values()))):
        raise ValueError("SC static output overlaps its sources")
    schema = pq.ParquetFile(sources['external_evidence']).schema_arrow
    if not set(native.EXTERNAL_COLUMNS) <= set(schema.names): raise ValueError("Missing SC external evidence columns")
    if any(not pa.types.is_boolean(schema.field(key).type) for key in ('is_experimental', 'is_predicted')):
        raise ValueError("SC external evidence flags must be typed booleans")
    mapped = read(bound['mapped_success']['path'])
    ids = read(Path(bound['mapped_success']['path']).parent / 'IDS.json')
    native.normalize_pathway_families(pd.read_parquet(sources['pathway_family']), ids['pathway_ids'])
    return {'input': record(manifest), 'mapping': record(path), 'mapped_success': bound['mapped_success'],
            'cancer_id': mapped['cancer_id'], **{key: record(p) for key, p in sources.items()},
            'implementations': [record(ROOT / name) for name in CODE], 'target_host': '149',
            'training_started': False, 'source_ancestry_complete': False}


def invoke(value, output):
    from scripts import materialize_single_cell_hhgt_assets_149 as native
    mapped = Path(value['mapped_success']['path']).parent; ids = read(mapped / 'IDS.json')
    settings = SimpleNamespace(cancer_id=value['cancer_id'], output_root=Path(output),
        pathway_family=Path(value['pathway_family']['path']), external_evidence=Path(value['external_evidence']['path']))
    return native._ensure_cancer_static(settings, mapped_success_path=mapped / 'MAPPED_SUCCESS.json',
        gene_ids=ids['protein_ids'], lnc_ids=ids['lncrna_ids'], pathway_ids=ids['pathway_ids'],
        pathway_by_gene=sparse.load_npz(mapped / 'EXACT_SIGNATURE_MATRIX.npz'),
        signature_counts=np.load(mapped / 'SIGNATURE_GENE_COUNTS.npy', allow_pickle=False),
        missing_counts=np.load(mapped / 'MISSING_GENE_COUNTS.npy', allow_pickle=False))


def inventories(value, output):
    roots = [output / 'STATIC_GLOBAL', output / 'cancers' / value['cancer_id'] / 'STATIC']
    pins = []
    for root in roots:
        receipt = read(root / 'SUCCESS.json'); pins.append(record(root / 'SUCCESS.json'))
        for name, pin in receipt['files'].items():
            path = verified_path(pin, root)
            if path != root / name or pin.get('bytes') != path.stat().st_size:
                raise ValueError('SC static inventory identity differs')
            pins.append(record(path))
    static = roots[1]; receipt = read(static / 'SUCCESS.json')
    if receipt['global_static_success_sha256'] != record(roots[0] / 'SUCCESS.json')['sha256']:
        raise ValueError('SC global static source differs')
    expected = {'global_static_success': 'SUCCESS.json', 'primary_relation_schema': 'EXTERNAL_PRIMARY_RELATION_SCHEMA.parquet',
                'sensitivity_relation_schema': 'EXTERNAL_TYPED_RELATION_SCHEMA.parquet'}
    binding = read(static / 'GLOBAL_STATIC_BINDING.json')
    if set(binding) != {'format', *expected}:
        raise ValueError('SC static global binding inventory differs')
    for key, name in expected.items():
        target = verified_path(binding[key], static)
        if target != roots[0] / name or binding[key].get('bytes') != target.stat().st_size:
            raise ValueError('SC static global binding differs')
    return roots, pins


def replay(value, output):
    actual_roots, _ = inventories(value, output)
    with tempfile.TemporaryDirectory(prefix='native_sc_static_replay_') as temporary:
        other = Path(temporary); invoke(value, other)
        expected_roots, _ = inventories(value, other)
        for left_root, right_root in zip(actual_roots, expected_roots):
            left, right = read(left_root / 'SUCCESS.json'), read(right_root / 'SUCCESS.json')
            omit = {'files', 'finished_unix', 'global_static_success_sha256', 'native_static_preparation'}
            if {k:v for k,v in left.items() if k not in omit} != {k:v for k,v in right.items() if k not in omit}:
                raise ValueError('SC static receipt differs from source reconstruction')
            if set(left['files']) != set(right['files']): raise ValueError('SC static file set differs')
            for name in right['files']:
                a, b = left_root / name, right_root / name
                if name.endswith('.parquet'):
                    try: pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)
                    except AssertionError as error: raise ValueError('SC static table differs from source reconstruction: '+name) from error
                elif name.endswith('.npz'):
                    x, y = sparse.load_npz(a), sparse.load_npz(b)
                    if x.shape != y.shape or x.dtype != y.dtype or (x != y).nnz:
                        raise ValueError('SC static matrix differs from source reconstruction: '+name)
                elif name == 'GLOBAL_STATIC_BINDING.json':
                    if read(a)['format'] != read(b)['format']: raise ValueError('SC static binding format differs')
                elif read(a) != read(b): raise ValueError('SC static source binding differs: '+name)


def validate_static(path):
    require_host(); path = Path(path).resolve(); receipt = read(path)
    preflight = verified_path(receipt['native_static_preparation'], path.parent); output = preflight.parent; value = read(preflight)
    if preflight != output / 'LAUNCH_PREFLIGHT.json' or path != output / 'cancers' / value['cancer_id'] / 'STATIC/SUCCESS.json':
        raise ValueError('SC static preparation location differs')
    if value != plan(verified_path(value['input'], output), output, new=False):
        raise ValueError('SC static source or implementation differs')
    _, pins = inventories(value, output)
    signature = {'value': value, 'preflight': record(preflight), 'outputs': pins}
    if _CACHE.get(str(output)) != signature:
        replay(value, output); _CACHE[str(output)] = signature
    return value


def run(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output / 'LAUNCH_PREFLIGHT.json', value)
    try:
        static, receipt = invoke(value, output); replay(value, output)
        if value != plan(manifest, output, new=False): raise ValueError('SC static inputs changed during execution')
        receipt['native_static_preparation'] = record(output / 'LAUNCH_PREFLIGHT.json'); write(static / 'SUCCESS.json', receipt)
        binding = output / 'NATIVE_SINGLE_CELL_STATIC_BINDING.json'
        write(binding, {'format': FORMAT, 'status': 'PASS', 'target_host': '149', 'training_started': False,
                        'source_ancestry_complete': False, 'preflight': record(output / 'LAUNCH_PREFLIGHT.json'),
                        'static_success': record(static / 'SUCCESS.json'), 'source_reconstruction_verified': True})
        validate(record(binding), output); return binding
    except Exception as error:
        write(output / 'RUN_FAILED.json', {'target_host': '149', 'training_started': False, 'error': str(error)}); raise


def validate(item, root):
    require_host(); path = verified_path(item, root); bound = read(path)
    if (bound.get('format') != FORMAT or bound.get('status') != 'PASS' or bound.get('target_host') != '149'
            or bound.get('source_reconstruction_verified') is not True
            or any(bound.get(k) is not False for k in ('training_started', 'source_ancestry_complete'))):
        raise ValueError('Incomplete native SC static workflow')
    preflight = verified_path(bound['preflight'], path.parent); value = read(preflight)
    success = verified_path(bound['static_success'], path.parent)
    if (preflight != path.parent / 'LAUNCH_PREFLIGHT.json'
            or success != path.parent / 'cancers' / value['cancer_id'] / 'STATIC/SUCCESS.json'
            or read(success)['native_static_preparation'] != bound['preflight']):
        raise ValueError('SC static binding differs')
    validate_static(success); return path, bound
