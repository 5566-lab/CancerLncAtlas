from pathlib import Path
from types import SimpleNamespace
import socket

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from cc_hhgt.v32 import native_single_cell_static_workflow as workflow
from cc_hhgt.v32 import native_single_cell_mapping_workflow as mapping
from cc_hhgt.v32 import native_single_cell_donors_workflow as donors
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.single_cell_graph_assets import UNASSIGNED_PATHWAY_FAMILY_ID
from scripts import materialize_single_cell_nested_hhgt_episodes_149 as nested
from test_v32_native_single_cell_mapping_workflow import source, forbidden


@pytest.fixture(scope='module', autouse=True)
def no_fit():
    from cc_hhgt.v32.single_cell_metacells import LocalSpace
    from cc_hhgt.v32 import single_cell_hhgt_episodes
    assert socket.gethostname() == '149'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LocalSpace, 'fit', forbidden)
        patch.setattr(single_cell_hhgt_episodes, 'fit_context_lasso', forbidden)
        for name in ('Adam', 'AdamW', 'SGD'): patch.setattr(torch.optim, name, forbidden)
        yield


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp('native_sc_static')
    manifest, _, _, _ = source(root / 'raw')
    mapped = mapping.run(manifest, root / 'mapping')
    mapped_root = Path(read(mapped)['mapped_success']['path']).parent; ids = read(mapped_root / 'IDS.json')
    family = root / 'family.parquet'
    pd.DataFrame({'pathway_id': ids['pathway_ids'][:-1], 'pathway_family_id': 'F1',
                  'membership_weight': 1., 'is_representative': True}).to_parquet(family, index=False)
    external = root / 'external.parquet'
    lnc, gene = ids['lncrna_ids'][0], ids['protein_ids'][0]
    rows = []
    for i, (experimental, predicted, status, kind) in enumerate([
        (True, False, 'lncrna_and_partner_mapped', 'physical_binding'),
        (True, False, 'lncrna_and_partner_mapped', 'physical_binding'),
        (True, True, 'lncrna_and_partner_mapped', 'physical_binding'),
        (False, False, 'lncrna_and_partner_mapped', 'physical_binding'),
        (True, False, 'unmapped', 'physical_binding'),
        (True, False, 'lncrna_and_partner_mapped', 'regulation')]):
        rows.append(dict(lncrna_id='LNC:'+lnc, partner_id='GENE:'+gene, partner_type='gene',
            relation_type='binding' if kind=='physical_binding' else 'activation', direction='direct',
            experiment_family=kind, is_experimental=experimental, is_predicted=predicted,
            pmid=str(i+1), source_database='SYNTHETIC', mapping_status=status))
    pd.DataFrame(rows).to_parquet(external, index=False)
    spec = {'format': workflow.INPUT, 'mapping': record(mapped), 'pathway_family': record(family), 'external_evidence': record(external)}
    path = root / 'INPUT.json'; write(path, spec)
    write(root / 'CHAIN_INPUT.json', {'format': chain.INPUT, 'chain_id': 'sc-static', 'stages': [
        {'id': 'static', 'operation': 'single_cell_static.run', 'inputs': spec}]})
    chain.initialize(root / 'CHAIN_INPUT.json', root / 'chain'); chain.advance(root / 'chain')
    state = read(root / 'chain/CHAIN.json'); assert state['status']=='DECLARED_STAGES_COMPLETE', state
    pin = state['results']['static']; binding = Path(pin['path']); out = binding.parent
    static = Path(read(binding)['static_success']['path']).parent
    return root, path, spec, pin, out, static, ids


def test_actual_static_producer_semantics_and_cached_consumers(completed, monkeypatch):
    root, _, _, pin, out, static, ids = completed
    workflow.validate(pin, root)
    matrix = sparse.load_npz(static / 'GENE_PATHWAY_MEMBERSHIP.npz')
    assert matrix.shape == (5, 2135) and matrix.nnz == 4270
    np.testing.assert_array_equal(matrix.toarray()[:2], np.ones((2, 2135)))
    edges = pd.read_parquet(static / 'EXTERNAL_PHYSICAL_BINDING.parquet')
    assert len(edges) == 1 and edges.evidence_record_count.iloc[0] == 2
    assert edges.distinct_pmid_count.iloc[0] == 2 and edges.edge_weight.iloc[0] == 1.
    assert edges.source_id.iloc[0] == ids['lncrna_ids'][0]
    assert edges.target_id.iloc[0] == ids['protein_ids'][0]
    hierarchy = pd.read_parquet(static / 'PATHWAY_FAMILY_HIERARCHY.parquet')
    assert hierarchy.loc[hierarchy.pathway_id.eq(ids['pathway_ids'][-1]), 'pathway_family_id'].iloc[0] == UNASSIGNED_PATHWAY_FAMILY_ID
    assert read(out / 'STATIC_GLOBAL/SUCCESS.json')['scan']['eligible_experimental_rows'] == 3
    nested._validate_static(SimpleNamespace(graph_asset_root=out), 'BRCA', ids)
    workflow.invoke(read(out / 'LAUNCH_PREFLIGHT.json'), out)
    monkeypatch.setattr(chain, 'dispatch', forbidden); chain.advance(root / 'chain')


def test_real_static_output_enters_canonical_donor_preparation(completed, tmp_path):
    _, _, spec, _, out, _, _ = completed
    crosswalk = tmp_path / 'donors.parquet'
    pd.DataFrame({'dataset_id': 'SYNTHETIC', 'query_donor_id': [f'D{i:02}' for i in range(10)],
                  'canonical_donor_id': [f'P{i:02}' for i in range(10)]}).to_parquet(crosswalk, index=False)
    candidates = tmp_path / 'candidates.parquet'
    pd.DataFrame({'cancer_id':['BRCA'], 'lncrna_id':['ENSG000006'], 'pathway_id':['P0000']}).to_parquet(candidates, index=False)
    manifest = tmp_path / 'INPUT.json'
    write(manifest, {'format': donors.INPUT, 'mapping': spec['mapping'], 'donor_crosswalk': record(crosswalk),
                     'graph_preflight': record(out / 'LAUNCH_PREFLIGHT.json'), 'candidates': record(candidates)})
    binding = donors.run(manifest, tmp_path / 'prepared')
    donors.validate(record(binding), tmp_path)
    assert len(pd.read_parquet(binding.parent / 'DONOR_FOLDS.parquet')) == 10


@pytest.mark.parametrize('fault', ['matrix', 'edge', 'node_order'])
def test_rehashed_static_corruption_rejected_by_original_consumer(completed, fault):
    _, _, _, _, out, static, ids = completed
    name = {'matrix':'GENE_PATHWAY_MEMBERSHIP.npz', 'edge':'EXTERNAL_PHYSICAL_BINDING.parquet',
            'node_order':'NODE_IDENTIFIERS.parquet'}[fault]
    target = static / name; receipt_path = static / 'SUCCESS.json'
    original = {p:p.read_bytes() for p in (target, receipt_path)}
    try:
        if fault == 'matrix':
            matrix = sparse.load_npz(target); matrix.data[0] = 0; sparse.save_npz(target, matrix)
        else:
            frame = pd.read_parquet(target)
            if fault == 'edge': frame.loc[0, 'edge_weight'] = 9.
            else: frame = frame.iloc[::-1].reset_index(drop=True)
            frame.to_parquet(target, index=False)
        receipt = read(receipt_path); receipt['files'][name] = {**record(target), 'bytes':target.stat().st_size}; write(receipt_path, receipt)
        with pytest.raises(ValueError, match='source reconstruction'):
            nested._validate_static(SimpleNamespace(graph_asset_root=out), 'BRCA', ids)
    finally:
        for p,data in original.items(): p.write_bytes(data)


def test_changed_external_source_invalidates_warm_cache(completed):
    root, _, spec, pin, _, _, _ = completed
    workflow.validate(pin, root); path = Path(spec['external_evidence']['path']); old = path.read_bytes()
    try:
        path.write_bytes(old+b' ')
        with pytest.raises(ValueError, match='SHA256|sha256|mismatch|drift'): workflow.validate(pin, root)
    finally: path.write_bytes(old)


def test_string_boolean_evidence_fails_before_output(completed, tmp_path):
    _, _, spec, _, _, _, _ = completed; spec = dict(spec)
    frame = pd.read_parquet(spec['external_evidence']['path']); frame['is_experimental'] = frame.is_experimental.astype(str)
    path = tmp_path / 'bad.parquet'; frame.to_parquet(path, index=False); spec['external_evidence'] = record(path)
    manifest = tmp_path / 'INPUT.json'; write(manifest, spec)
    with pytest.raises(ValueError, match='typed booleans'): workflow.run(manifest, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_plan_no_outputs_and_host_gate(completed, tmp_path, monkeypatch):
    _, manifest, _, _, _, _, _ = completed
    workflow.plan(manifest, tmp_path / 'out'); assert not (tmp_path / 'out').exists()
    monkeypatch.setattr(socket, 'gethostname', lambda:'local')
    with pytest.raises(RuntimeError, match='149'): workflow.plan(manifest, tmp_path / 'out')
