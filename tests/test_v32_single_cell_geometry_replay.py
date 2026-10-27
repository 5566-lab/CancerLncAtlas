"""Persist/apply explicitly frozen PCA parameters; never fit PCA/LASSO/models."""
from pathlib import Path
from types import SimpleNamespace
import socket
import time

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
import torch

from cc_hhgt.v32 import single_cell_geometry_replay as replay
from cc_hhgt.v32.single_cell_metacells import LocalSpace, prepare_local_space, local_disjoint_groups
from cc_hhgt.v32.formal_expression_sources import read, write, record


def forbidden(*a, **k): raise AssertionError('Geometry verification cannot fit parameters')


@pytest.fixture(scope='module', autouse=True)
def no_fit():
    from cc_hhgt.v32 import single_cell_hhgt_episodes
    assert socket.gethostname() == '149'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PCA, 'fit', forbidden); patch.setattr(PCA, 'fit_transform', forbidden)
        patch.setattr(LocalSpace, 'fit', forbidden)
        patch.setattr(single_cell_hhgt_episodes, 'fit_context_lasso', forbidden)
        for name in ('Adam', 'AdamW', 'SGD'): patch.setattr(torch.optim, name, forbidden)
        yield


def frozen_space(matrix, donors, allowed, *, components=20, **policy):
    """Identity basis is a supplied fixture, never an estimated principal basis."""
    values = prepare_local_space(matrix, donors, allowed, **policy)
    count = min(components, len(values['genes']), len(values['fit_matrix'])-1)
    pca = PCA(n_components=count, svd_solver='randomized', random_state=policy.get('seed', 17))
    pca.components_ = np.eye(len(values['genes']))[:count]
    pca.explained_variance_ = np.ones(count)
    pca.mean_ = values['fit_matrix'].mean(axis=0); pca.n_components_ = count
    pca.n_features_in_ = len(values['genes']); pca._fit_svd_solver = 'randomized'
    return LocalSpace(values['genes'], values['mean'], values['scale'], pca,
                      values['center_z'], values['training_donors'])


def inputs():
    x = np.arange(40, dtype=float)
    matrix = sparse.csr_matrix(np.stack([x, np.sin(x), x*x/100, np.ones(40)], axis=1))
    meta = pd.DataFrame({'cell_id':[f'C{i}' for i in range(40)], 'donor_id':np.repeat(list('abcde'), 8),
                         'cell_type':'T', 'state_id':'S'})
    fit = dict(max_genes=3, max_cells_per_donor=4, seed=23)
    group = dict(target_size=3, min_size=2, radius_quantile=.9, seed=24)
    return matrix, meta, fit, group


def produce(root):
    matrix, meta, fit, group = inputs()
    space = frozen_space(matrix, meta.donor_id, ['a', 'b', 'c'], components=2, **fit)
    coords = space.transform(matrix)
    assignments, audit = local_disjoint_groups(coords, meta, allowed_training_donors=['a', 'b', 'c'], **group)
    replay.save(root, replay.capture(space, coords), audit)
    return matrix, meta, fit, group, assignments, audit


def verify(root, values):
    matrix, meta, fit, group, assignments, audit = values
    return replay.verify(root, matrix, meta, ['a', 'b', 'c'], fit, group, 2, assignments, audit)


def test_original_preparation_manual_oracle_and_query_exclusion():
    matrix, meta, fit, _ = inputs()
    actual = prepare_local_space(matrix, meta.donor_id, ['a', 'b', 'c'], **fit)
    dense = matrix.toarray(); mean = dense[:24].mean(axis=0)
    variance = (dense[:24]**2).mean(axis=0)-mean**2
    genes = sorted([i for i, v in enumerate(variance) if v > 1e-8], key=lambda i:-variance[i])[:3]
    assert actual['genes'].tolist() == genes
    np.testing.assert_allclose(actual['mean'], mean[genes], rtol=1e-14)
    np.testing.assert_allclose(actual['scale'], np.sqrt(variance[genes]), rtol=1e-14)
    rng = np.random.default_rng(23)
    balanced = np.concatenate([np.sort(rng.choice(np.arange(i*8, (i+1)*8), 4, replace=False)) for i in range(3)])
    assert actual['balanced_rows'].tolist() == balanced.tolist()
    z = np.clip((dense[balanced][:, genes]-mean[genes])/np.sqrt(variance[genes]), -10, 10)
    np.testing.assert_allclose(actual['center_z'], z.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(actual['fit_matrix'], z-z.mean(axis=0), atol=1e-14)
    dense[24:] = 1e9
    changed = prepare_local_space(sparse.csr_matrix(dense), meta.donor_id, ['a', 'b', 'c'], **fit)
    for key in ('genes', 'mean', 'scale', 'center_z', 'fit_matrix', 'balanced_rows'):
        np.testing.assert_array_equal(changed[key], actual[key])


def test_actual_frozen_transform_and_original_grouping_round_trip(tmp_path):
    values = produce(tmp_path)
    assert verify(tmp_path, values)
    matrix, meta, fit, group, assignments, audit = values
    space = frozen_space(matrix, meta.donor_id, ['a', 'b', 'c'], components=2, **fit)
    z = np.clip((matrix[:, space.genes].toarray()-space.mean)/space.scale, -10, 10)-space.center_z
    expected = (z-space.pca.mean_)@space.pca.components_.T
    np.testing.assert_allclose(expected, space.transform(matrix), atol=1e-14)
    assert audit['training_donors'] == ['a', 'b', 'c'] and len(set(assignments)-{-1}) > 0
    assert replay.capture(SimpleNamespace(transform=lambda x:x), expected) is None


def test_unequal_donor_cell_counts_keep_equal_total_donor_weight():
    matrix, meta, fit, _ = inputs()
    selected = np.r_[0:8, 8:13, 16:19, 24:40]
    matrix = matrix[selected]; meta = meta.iloc[selected].reset_index(drop=True)
    values = prepare_local_space(matrix, meta.donor_id, ['a', 'b', 'c'], **fit)
    dense = matrix.toarray()
    per_donor = [dense[meta.donor_id.eq(donor)] for donor in 'abc']
    mean = np.mean([x.mean(axis=0) for x in per_donor], axis=0)
    variance = np.mean([(x*x).mean(axis=0) for x in per_donor], axis=0)-mean**2
    genes = values['genes']
    np.testing.assert_allclose(values['mean'], mean[genes], rtol=1e-14)
    np.testing.assert_allclose(values['scale'], np.sqrt(variance[genes]), rtol=1e-14)
    assert not np.allclose(values['mean'], dense[:16, genes].mean(axis=0))


@pytest.mark.parametrize('fault', ['genes','mean','scale','center_z','pca_mean','pca_components',
    'training_donors','coordinates','membership','radius','marker','file','source','policy','variance'])
def test_rehashed_projection_or_membership_changes_are_rejected(tmp_path, fault):
    values = list(produce(tmp_path)); matrix, meta, fit, group, assignments, audit = values
    path = tmp_path/replay.FILE
    with np.load(path, allow_pickle=False) as archive: arrays = {key:archive[key] for key in archive.files}
    if fault in arrays:
        if fault == 'training_donors': arrays[fault][0] = 'e'
        else: arrays[fault].flat[0] += 1
        np.savez_compressed(path, **arrays)
    elif fault == 'variance':
        arrays['pca_explained_variance'][0] = -1; np.savez_compressed(path, **arrays)
    elif fault == 'membership': assignments[0] = 999
    elif fault == 'radius': audit['radius'] += 1
    elif fault == 'marker': audit.pop('local_space_format')
    elif fault == 'file': path.unlink()
    elif fault == 'source':
        changed = matrix.toarray(); changed[0, 0] += 100; values[0] = sparse.csr_matrix(changed)
    else: fit['max_cells_per_donor'] = 3
    with pytest.raises(ValueError, match='Geometry'): verify(tmp_path, values)


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    import test_v32_native_single_cell_graph_workflow as graphs
    import test_v32_native_single_cell_episode_sources as episodes
    from cc_hhgt.v32 import native_single_cell_donors_workflow as donors, single_cell_nested_episodes as nested_geometry
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graphs, 'frozen_fit', frozen_space)
        patch.setattr(graphs, 'frozen_groups', local_disjoint_groups)
        graph = graphs.completed.__wrapped__(tmp_path_factory)
    root, _, spec, graph_pin, graph_out = graph
    donor = Path(spec['donor_preparation']['path']); value = read(read(donor)['preflight']['path'])
    settings, _ = donors.build_settings(value, donor.parent, time.time()+1200)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nested_geometry, 'LocalSpace', SimpleNamespace(fit=frozen_space))
        receipts = native.run_cancer(settings)
    assert any(r['state'] == 'NESTED_EPISODE_ASSETS_READY' for r in receipts)
    result = episodes.completed.__wrapped__((root, donor, value, graph_pin, graph_out))
    return result


def test_actual_outer_nested_episodes_and_model_inspection(completed, monkeypatch):
    from test_v32_single_cell_model_use import checkpoint_fixture, inspect as inspect_model
    from scripts import train_single_cell_hhgt_149 as trainer
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as assembler
    root, settings, donor, value, manifest = completed
    assert manifest['outer_folds'] == 5
    binding = checkpoint_fixture(root/'inspection', settings.output_root, settings.graph_asset_root,
                                 candidates=Path(value['candidates']['path']))
    for name in ('_initialize_model', '_evaluate_episode', 'fit_fold_model', 'run_training'):
        monkeypatch.setattr(trainer, name, forbidden)
    monkeypatch.setattr(assembler, 'fit_context_lasso', forbidden)
    inventory = inspect_model(binding, root)
    proven = [r for r in inventory.transform_replays if r['format'] == replay.FORMAT]
    assert len(proven) > 5 and all(r['model_training_started'] is False for r in proven)
    assert all(r['outputs_compared'] == 3 for r in proven)
    assert 'SC_FITTED_GEOMETRY_NOT_INDEPENDENTLY_VERIFIED' in {r['reason'] for r in inventory.gaps}
    assert any(Path(r['path']).name == replay.FILE for r in inventory.dependencies)
    for folder in (donor.parent/'metacells/cancers/BRCA').glob('outer_fold=*'):
        assert (folder/replay.FILE).is_file()
        assert read(folder/'GEOMETRY_AUDIT.json')['local_space_format'] == replay.FORMAT


@pytest.mark.parametrize('scope', ['outer', 'nested'])
def test_actual_original_cache_rejects_rehashed_coordinates(completed, scope):
    from cc_hhgt.v32 import native_single_cell_donors_workflow as donors
    from scripts import materialize_single_cell_metacell_149 as outer_native
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as nested_native
    _, _, donor, value, _ = completed
    if scope == 'outer':
        settings, _ = donors.build_metacell_settings(value, donor.parent, time.time()+600)
        cancer = settings.output_root/'cancers/BRCA/SUCCESS.json'
        success = cancer.parent/'outer_fold=0/SUCCESS.json'
        parents = [cancer]
        native = outer_native
    else:
        settings, _ = donors.build_settings(value, donor.parent, time.time()+600)
        cancer = settings.output_root/'cancers/BRCA/SUCCESS.json'
        outer = Path(read(cancer)['outer_fold_receipts'][0]['path'])
        success = next(Path(p['path']) for p in read(outer)['episode_receipts']
                       if (Path(p['path']).parent/replay.FILE).is_file())
        parents = [outer, cancer]
        native = nested_native
    archive = success.parent/replay.FILE
    old = {p:p.read_bytes() for p in [archive, success, *parents]}
    try:
        with np.load(archive, allow_pickle=False) as data: arrays = {k:data[k] for k in data.files}
        arrays['coordinates'].flat[0] += .25; np.savez_compressed(archive, **arrays)
        current = read(success); current['files'][archive.name] = native.file_record(archive); write(success, current)
        if scope == 'outer':
            current = read(cancer); current['fold_receipt_sha256']['0'] = record(success)['sha256']; write(cancer, current)
        else:
            current = read(outer); current['episode_receipts'] = [native.file_record(Path(p['path'])) for p in current['episode_receipts']]; write(outer, current)
            current = read(cancer); current['outer_fold_receipts'] = [native.file_record(Path(p['path'])) for p in current['outer_fold_receipts']]; write(cancer, current)
        with pytest.raises(ValueError, match='Geometry reconstruction differs: coordinates'):
            native.run_cancer(settings)
    finally:
        for path, content in old.items(): path.write_bytes(content)
