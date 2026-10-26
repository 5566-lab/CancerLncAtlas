"""Canonical outer-query materialization with explicitly frozen test geometry."""
from pathlib import Path
from types import SimpleNamespace
import socket
import time

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from cc_hhgt.v32 import native_single_cell_donors_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read, write, record
from scripts import materialize_single_cell_metacell_149 as native
import test_v32_native_single_cell_donors_workflow as donor_fixtures
from test_v32_native_single_cell_mapping_workflow import source as raw_source, forbidden


def expanded_source(root):
    manifest, spec, values, policy = raw_source(root)
    status_path = Path(spec['run_status']['path']); status = read(status_path); row = status['per_cancer']['BRCA']
    metadata = pd.read_parquet(row['metadata_path']).sort_values('cell_id').reset_index(drop=True)
    frame = pd.concat([metadata.iloc[:3], *[metadata.iloc[3:]] * 10], ignore_index=True)
    frame['cell_id'] = [f'C{i:05d}' if i >= 3 else f'C{i:04d}' for i in range(len(frame))]
    matrix = sparse.csc_matrix(np.concatenate([values[:, :3], np.tile(values[:, 3:], (1, 10))], axis=1))
    with h5py.File(row['h5_path'], 'r+') as handle:
        group = handle['matrix']
        for name, data in [('data', matrix.data), ('indices', matrix.indices), ('indptr', matrix.indptr),
                           ('shape', matrix.shape), ('barcodes', np.asarray(frame.cell_id, dtype='S'))]:
            del group[name]; group.create_dataset(name, data=data)
    frame.sample(frac=1, random_state=22).to_parquet(row['metadata_path'], index=False)
    for name in ('h5', 'metadata'):
        row[name + '_sha256'] = record(row[name + '_path'])['sha256']
    write(status_path, status); spec['run_status'] = record(status_path); write(manifest, spec)
    return manifest, spec, values, policy


def frozen_fit(matrix, donors, allowed, **kwargs):
    assert set(allowed) <= set(donors)
    return SimpleNamespace(transform=lambda values: values[:, :2].toarray())


def frozen_groups(coordinates, metadata, *, allowed_training_donors, **kwargs):
    assert set(allowed_training_donors) <= set(metadata.donor_id)
    assignment = np.full(len(metadata), -1, dtype=int); group = 0
    for _, indices in metadata.groupby(['donor_id', 'cell_type', 'state_id'], sort=True).indices.items():
        for start in range(0, len(indices) - 2, 3):
            assignment[indices[start:start + 3]] = group; group += 1
    return assignment, {'synthetic_frozen_geometry': True, 'source_donors': sorted(allowed_training_donors)}


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
    root = tmp_path_factory.mktemp('canonical_metacell')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(donor_fixtures, 'source', expanded_source)
        manifest, _ = donor_fixtures.inputs(root)
    binding = workflow.run(manifest, root / 'prepared'); value = read(read(binding)['preflight']['path'])
    settings, _ = workflow.build_metacell_settings(value, binding.parent, time.time() + 600)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, 'LocalSpace', SimpleNamespace(fit=frozen_fit))
        patch.setattr(native, 'local_disjoint_groups', frozen_groups)
        native.run_cancer(settings)
    return root, binding, value, settings


def test_actual_outer_queries_share_nested_canonical_partitions(completed, monkeypatch):
    _, binding, _, settings = completed
    root = settings.output_root / 'cancers/BRCA'; success = read(root / 'SUCCESS.json')
    assert success['donors'] == 9 and success['eligible_cells'] == 1200
    folds = pd.read_parquet(root / 'DONOR_FOLDS.parquet')
    pd.testing.assert_frame_equal(folds, pd.read_parquet(binding.parent / 'DONOR_FOLDS.parquet'))
    plans = read(binding.parent / 'NESTED_PLANS.json'); all_query = []
    for fold in range(5):
        folder = root / f'outer_fold={fold}'; receipt = read(folder / 'SUCCESS.json')
        assert receipt['native_donor_preparation'] == record(binding)
        rows = pd.read_parquet(folder / 'QUERY_ROWS.parquet')
        assert not rows.empty and np.isfinite(rows.rho).all()
        assert rows.query_donor_id.str.startswith('P').all()
        assert not set(receipt['source_donors']) & set(rows.query_donor_id)
        assert all(set(p['outer_query_donors']) == set(receipt['query_donors'])
                   for p in plans if p['outer_fold_id'] == fold)
        all_query.extend(rows.query_donor_id)
    assert set(all_query) == set(folds.donor_id)
    monkeypatch.setattr(native, 'LocalSpace', SimpleNamespace(fit=forbidden))
    monkeypatch.setattr(native, 'local_disjoint_groups', forbidden)
    native.run_cancer(settings)


@pytest.mark.parametrize('fault', ['query_donor', 'query_cancer', 'membership', 'preparation'])
def test_rehashed_outer_fold_cannot_change_canonical_identity(completed, fault):
    _, _, _, settings = completed; root = settings.output_root / 'cancers/BRCA'
    cancer_path = root / 'SUCCESS.json'; folder = root / 'outer_fold=0'; receipt_path = folder / 'SUCCESS.json'
    filename = 'QUERY_ROWS.parquet' if fault in ('query_donor', 'query_cancer') else 'METACELL_MEMBERSHIP.parquet'
    target = folder / filename; original = {p: p.read_bytes() for p in (target, receipt_path, cancer_path)}
    try:
        receipt = read(receipt_path)
        if fault == 'preparation': receipt['native_donor_preparation']['sha256'] = '0' * 64
        else:
            frame = pd.read_parquet(target)
            column = {'query_donor': 'query_donor_id', 'query_cancer': 'cancer_id', 'membership': 'donor_id'}[fault]
            frame.loc[0, column] = receipt['source_donors'][0] if fault == 'query_donor' else 'ALTERED'
            frame.to_parquet(target, index=False); receipt['files'][filename] = native.file_record(target)
        write(receipt_path, receipt); cancer = read(cancer_path)
        cancer['fold_receipt_sha256']['0'] = record(receipt_path)['sha256']; write(cancer_path, cancer)
        with pytest.raises(RuntimeError, match='canonical fold'):
            native.run_cancer(settings)
    finally:
        for p, data in original.items(): p.write_bytes(data)


@pytest.mark.parametrize('field', ['seed', 'delete_verified_scratch', 'max_train_donors_per_pair', 'deadline'])
def test_outer_consumer_rejects_unprepared_settings(completed, field):
    _, binding, value, _ = completed
    settings, _ = workflow.build_metacell_settings(value, binding.parent, time.time() + 300)
    setattr(settings, field, {'seed': 7, 'delete_verified_scratch': True,
                             'max_train_donors_per_pair': 2, 'deadline': float('inf')}[field])
    with pytest.raises(ValueError, match='configuration|deadline'):
        native.run_cancer(settings)
