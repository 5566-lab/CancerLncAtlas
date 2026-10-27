"""Replay native coordinates and local grouping from persisted PCA parameters.

This verifies preprocessing and application of the supplied projection. It does
not refit PCA or assert that supplied components came from the declared fit.
Legacy results without parameters retain that explicit ancestry gap.
"""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .cnv_input_sources import require_host
from .single_cell_metacells import LocalSpace, prepare_local_space, local_disjoint_groups

FORMAT = 'NATIVE_SC_FROZEN_GEOMETRY_REPLAY_V1'
FILE = 'LOCAL_SPACE.npz'
FIELDS = {'genes', 'mean', 'scale', 'center_z', 'pca_mean', 'pca_components', 'pca_explained_variance',
          'training_donors', 'coordinates'}
CODE = 'cc_hhgt/v32/single_cell_geometry_replay.py'


def capture(space, coordinates):
    """Capture the original LocalSpace only; external substitutes remain unknown."""
    if not isinstance(space, LocalSpace): return None
    if getattr(space.pca, 'whiten', None) is not False:
        raise ValueError('Geometry replay supports the original unwhitened PCA only')
    return {**{name: np.asarray(getattr(space, name)) for name in ('genes', 'mean', 'scale', 'center_z')},
        'pca_mean': np.asarray(space.pca.mean_), 'pca_components': np.asarray(space.pca.components_),
        'pca_explained_variance': np.asarray(space.pca.explained_variance_),
        'training_donors': np.asarray(space.training_donors, dtype=str), 'coordinates': np.asarray(coordinates)}


def save(folder, arrays, geometry):
    if arrays is None: return
    require_host()
    if set(arrays) != FIELDS: raise ValueError('Geometry parameter schema differs')
    path = Path(folder)/FILE
    with path.open('xb') as handle: np.savez_compressed(handle, **arrays)
    geometry['local_space_format'] = FORMAT


def extra_files(folder, geometry):
    path = Path(folder)/FILE
    marker = geometry.get('local_space_format')
    if marker is None and not path.exists(): return set()
    if marker != FORMAT or not path.is_file():
        raise ValueError('Geometry projection marker or parameters are missing')
    return {FILE}


def policies(settings, *, seed, radius_seed):
    fit = dict(max_genes=settings.pca_genes, max_cells_per_donor=settings.pca_cells_per_donor, seed=seed)
    grouping = dict(target_size=settings.target_size, min_size=settings.min_metacell_size,
                    radius_quantile=settings.radius_quantile, seed=radius_seed)
    return fit, grouping


def _same(observed, expected, name, *, exact=False):
    if observed.shape != expected.shape:
        raise ValueError('Geometry reconstruction shape differs: '+name)
    if exact:
        same = np.array_equal(observed, expected)
    else:
        same = np.isfinite(observed).all() and np.allclose(observed, expected, rtol=1e-12, atol=1e-12)
    if not same: raise ValueError('Geometry reconstruction differs: '+name)


def verify(folder, coding_log, metadata, source_donors, fit_policy, grouping_policy,
           components, assignments, geometry):
    """No-fit replay using exact donor rows and the original grouping function."""
    require_host()
    if not extra_files(folder, geometry): return False
    with np.load(Path(folder)/FILE, allow_pickle=False) as archive:
        if set(archive.files) != FIELDS: raise ValueError('Geometry parameter schema differs')
        arrays = {name: archive[name] for name in archive.files}
    expected = prepare_local_space(coding_log, metadata.donor_id, source_donors, **fit_policy)
    if not np.issubdtype(arrays['genes'].dtype, np.integer):
        raise ValueError('Geometry gene indices must remain integers')
    for name in ('genes', 'mean', 'scale', 'center_z'):
        _same(arrays[name], expected[name], name, exact=name == 'genes')
    _same(arrays['training_donors'], np.asarray(expected['training_donors'], dtype=str), 'training_donors', exact=True)
    projection = arrays['pca_components']
    count = min(components, len(expected['genes']), len(expected['fit_matrix'])-1)
    if (projection.shape != (count, len(expected['genes'])) or not np.isfinite(projection).all()
            or not np.allclose(projection@projection.T, np.eye(count), rtol=1e-7, atol=1e-7)):
        raise ValueError('Geometry PCA components are not a valid native projection')
    _same(arrays['pca_mean'], expected['fit_matrix'].mean(axis=0), 'pca_mean')
    variance = arrays['pca_explained_variance']
    if variance.shape != (count,) or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError('Geometry PCA explained variance is invalid')
    # Construct a frozen estimator and invoke its actual transform, never fit.
    pca = PCA(n_components=count, svd_solver='randomized', random_state=fit_policy['seed'])
    pca.components_ = projection; pca.mean_ = arrays['pca_mean']
    pca.explained_variance_ = variance
    pca.n_features_in_ = len(expected['genes']); pca.n_components_ = count
    pca._fit_svd_solver = 'randomized'
    space = LocalSpace(arrays['genes'], arrays['mean'], arrays['scale'], pca,
                       arrays['center_z'], expected['training_donors'])
    coordinates = space.transform(coding_log)
    _same(arrays['coordinates'], coordinates, 'coordinates')
    observed, audit = local_disjoint_groups(coordinates, metadata[['donor_id', 'cell_type', 'state_id']],
        allowed_training_donors=source_donors, **grouping_policy)
    _same(np.asarray(assignments), observed, 'membership', exact=True)
    if any(geometry.get(key) != value for key, value in audit.items()):
        raise ValueError('Geometry original radius/grouping audit differs')
    return True
