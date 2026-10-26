from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from cc_hhgt.v32 import native_single_cell_donors_workflow as workflow
from cc_hhgt.v32 import native_single_cell_mapping_workflow as mapping
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, record
from cc_hhgt.v32.current_analysis_workflow import write
from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
from test_v32_native_single_cell_mapping_workflow import source, forbidden


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    from cc_hhgt.v32.single_cell_metacells import LocalSpace
    from cc_hhgt.v32 import single_cell_hhgt_episodes
    assert socket.gethostname() == '149'
    monkeypatch.setattr(LocalSpace, 'fit', forbidden)
    monkeypatch.setattr(single_cell_hhgt_episodes, 'fit_context_lasso', forbidden)
    for name in ('Adam', 'AdamW', 'SGD'):
        monkeypatch.setattr(torch.optim, name, forbidden)


def inputs(root, donors=9):
    manifest, _, _, _ = source(root / 'raw')
    mapping_path = mapping.run(manifest, root / 'mapping')
    mapped = Path(read(mapping_path)['mapped_success']['path']).parent
    ids = read(mapped / 'IDS.json')
    crosswalk = root / 'donors.parquet'
    pd.DataFrame({'dataset_id': ['SYNTHETIC'] * 10, 'query_donor_id': [f'D{i:02d}' for i in range(10)],
                  'canonical_donor_id': [f'P{i % donors:02d}' for i in range(10)]}).to_parquet(crosswalk, index=False)
    graph = root / 'graphs'; static = graph / 'cancers/BRCA/STATIC'; static.mkdir(parents=True)
    nodes = pd.concat([pd.DataFrame({'node_type': key, 'canonical_id': values, 'global_index': range(len(values))})
                       for key, values in [('gene', ids['protein_ids']), ('lncRNA', ids['lncrna_ids']),
                                           ('pathway', ids['pathway_ids']), ('pathway_family', ['F1'])]], ignore_index=True)
    nodes.to_parquet(static / 'NODE_IDENTIFIERS.parquet', index=False)
    sparse.save_npz(static / 'GENE_PATHWAY_MEMBERSHIP.npz', sparse.load_npz(mapped / 'EXACT_SIGNATURE_MATRIX.npz'))
    sparse.save_npz(static / 'PATHWAY_FAMILY_MEMBERSHIP.npz', sparse.csr_matrix(np.ones((2135, 1))))
    pd.DataFrame({'lncrna_id': [], 'gene_id': []}).to_parquet(static / 'EXTERNAL_PHYSICAL_BINDING.parquet', index=False)
    write(static / 'SUCCESS.json', {'target_host': '149', 'state': 'CANCER_STATIC_GRAPH_ASSETS_READY', 'cancer_id': 'BRCA',
                                    'files': {p.name: native.file_record(p) for p in sorted(static.iterdir())}})
    write(graph / 'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'synthetic': True})
    candidates = root / 'candidates.parquet'
    pd.DataFrame({'cancer_id': ['BRCA'], 'lncrna_id': [ids['lncrna_ids'][0]], 'pathway_id': [ids['pathway_ids'][0]]}).to_parquet(candidates, index=False)
    spec = {'format': workflow.INPUT, 'mapping': record(mapping_path), 'donor_crosswalk': record(crosswalk),
            'graph_preflight': record(graph / 'LAUNCH_PREFLIGHT.json'), 'candidates': record(candidates)}
    path = root / 'INPUT.json'; write(path, spec)
    return path, spec


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp('native_sc_donors')
    manifest, spec = inputs(root)
    write(root / 'CHAIN_INPUT.json', {'format': chain.INPUT, 'chain_id': 'sc-donors', 'stages': [
        {'id': 'donors', 'operation': 'single_cell_donors.run', 'inputs': spec}]})
    chain.initialize(root / 'CHAIN_INPUT.json', root / 'chain'); chain.advance(root / 'chain')
    state = read(root / 'chain/CHAIN.json'); assert state['status'] == 'DECLARED_STAGES_COMPLETE', state
    pin = state['results']['donors']; binding = Path(pin['path']); value = read(read(binding)['preflight']['path'])
    return root, manifest, spec, pin, binding, value


def test_canonical_donors_reach_original_nested_consumer(completed, monkeypatch):
    root, _, _, pin, binding, value = completed
    workflow.validate(pin, root)
    settings, _ = workflow.build_settings(value, binding.parent, time.time() + 300)
    launch = native.validate_preflight(settings)
    metadata, folds, source_receipt = workflow.consumer(settings, launch)
    assert len(metadata) == 120 and metadata.donor_id.nunique() == 9
    assert metadata.loc[metadata.native_donor_id.isin(['D00', 'D09']), 'donor_id'].eq('P00').all()
    observed, _ = native._validate_source_cancer(settings, 'BRCA')
    pd.testing.assert_frame_equal(observed, folds)
    assert source_receipt['source_receipt_path'] == str(binding)
    crosswalk = pd.read_parquet(binding.parent / 'CANONICAL_DONOR_CROSSWALK.parquet')
    assert len(crosswalk) == 9 and crosswalk.query_donor_id.eq(crosswalk.canonical_donor_id).all()
    for row in read(binding.parent / 'NESTED_PLANS.json'):
        assert not set(row['source_donors']) & set(row['excluded_donors'])
        assert not set(row['outer_train_donors']) & set(row['outer_query_donors'])
    assert not (binding.parent / 'nested/cancers').exists()
    assert '<UNIX_DEADLINE>' in read(binding.parent / 'EXECUTION_COMMAND.json')['argv']
    monkeypatch.setattr(chain, 'dispatch', forbidden); chain.advance(root / 'chain')


def test_actual_nested_runner_receives_canonical_rows_before_geometry(completed, monkeypatch):
    _, _, _, _, binding, value = completed
    settings, _ = workflow.build_settings(value, binding.parent, time.time() + 300)
    def before_fit(coding_log, coding, lnc, library, metadata, plan, **kwargs):
        assert len(metadata) == coding.shape[0] == 120
        assert metadata.donor_id.nunique() == 9
        assert set(plan.source_donors) <= set(metadata.donor_id)
        assert not set(plan.source_donors) & set(plan.excluded_donors)
        raise RuntimeError('VERIFIED_BEFORE_GEOMETRY_NO_FITTING')
    monkeypatch.setattr(native, 'fit_nested_metacells', before_fit)
    with pytest.raises(RuntimeError, match='VERIFIED_BEFORE_GEOMETRY_NO_FITTING'):
        native.run_cancer(settings)


def test_three_donors_publish_unavailable_without_fitting(tmp_path):
    manifest, _ = inputs(tmp_path, donors=3)
    binding = workflow.run(manifest, tmp_path / 'prepared'); value = read(read(binding)['preflight']['path'])
    settings, _ = workflow.build_settings(value, binding.parent, time.time() + 300)
    receipts = native.run_cancer(settings)
    assert len(receipts) == 6
    assert all(r['state'] == 'NESTED_EPISODE_UNAVAILABLE' for r in receipts)
    result = read(settings.output_root / 'cancers/BRCA/SUCCESS.json')
    assert result['source_kind'] == 'NATIVE_CANONICAL_DONOR_PREPARATION'
    assert result['source_cancer_success_sha256'] == record(binding)['sha256']


@pytest.mark.parametrize('field', ['metadata', 'folds', 'plans', 'command', 'crosswalk'])
def test_rehashed_prepared_outputs_cannot_replace_reconstruction(completed, field):
    root, _, _, _, binding, _ = completed; bound = read(binding)
    name = {'metadata': 'CANONICAL_CELL_METADATA.parquet', 'folds': 'DONOR_FOLDS.parquet',
            'plans': 'NESTED_PLANS.json', 'command': 'EXECUTION_COMMAND.json',
            'crosswalk': 'CANONICAL_DONOR_CROSSWALK.parquet'}[field]
    target = binding.parent / name; launch = binding.parent / 'nested/LAUNCH_PREFLIGHT.json'
    original = {p: p.read_bytes() for p in (binding, target, launch)}
    try:
        if field in ('metadata', 'folds', 'crosswalk'):
            frame = pd.read_parquet(target)
            frame.loc[0, 'canonical_donor_id' if field == 'crosswalk' else 'donor_id'] = 'ALTERED'
            frame.to_parquet(target, index=False)
        else:
            data = read(target)
            if field == 'plans': data[0]['source_donors'] = []
            else: data['automatically_executed'] = True
            write(target, data)
        bound['files'][name] = record(target); write(binding, bound)
        preflight = read(launch); preflight['native_donor_preparation'] = record(binding); write(launch, preflight)
        with pytest.raises(ValueError, match='differs|differ'):
            workflow.validate(record(binding), root)
    finally:
        for p, data in original.items(): p.write_bytes(data)


@pytest.mark.parametrize('setting', ['seed', 'delete_verified_scratch', 'pca_components', 'scratch_root', 'deadline'])
def test_native_execution_rejects_unprepared_configuration(completed, setting):
    _, _, _, _, binding, value = completed
    current, _ = workflow.build_settings(value, binding.parent, time.time() + 300)
    changes = {'seed': 22, 'delete_verified_scratch': True, 'pca_components': 99,
               'scratch_root': binding.parent, 'deadline': float('inf')}
    setattr(current, setting, changes[setting])
    with pytest.raises(ValueError, match='configuration|deadline'):
        native.validate_preflight(current)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'whitespace', 'two_donors'])
def test_crosswalk_rejects_incomplete_or_ambiguous_identity(completed, tmp_path, fault):
    _, _, spec, _, _, _ = completed
    spec = dict(spec); frame = pd.read_parquet(spec['donor_crosswalk']['path'])
    if fault == 'missing': frame = frame.iloc[1:]
    elif fault == 'duplicate': frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif fault == 'whitespace': frame.loc[0, 'canonical_donor_id'] = ' P00'
    else: frame['canonical_donor_id'] = [f'P{i % 2}' for i in range(10)]
    path = tmp_path / 'donors.parquet'; frame.to_parquet(path, index=False)
    spec['donor_crosswalk'] = record(path); manifest = tmp_path / 'INPUT.json'; write(manifest, spec)
    with pytest.raises(ValueError, match='crosswalk|canonical donors'):
        workflow.run(manifest, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_plan_has_no_outputs_and_enforces_host(completed, tmp_path, monkeypatch):
    _, manifest, _, _, _, _ = completed
    workflow.plan(manifest, tmp_path / 'out'); assert not (tmp_path / 'out').exists()
    monkeypatch.setattr(socket, 'gethostname', lambda: 'local')
    with pytest.raises(RuntimeError, match='149'):
        workflow.plan(manifest, tmp_path / 'out')
