"""Training-donor-fitted local, disjoint metacells; no lncRNA target in grouping."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def donor_weights(donors):
    donors = np.asarray(donors, str)
    _, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


@dataclass
class LocalSpace:
    genes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca: object
    center_z: np.ndarray
    training_donors: tuple

    @classmethod
    def fit(cls, coding_log, donors, allowed_donors, max_genes=1500, components=20,
            max_cells_per_donor=2000, seed=17):
        donors = np.asarray(donors, str)
        allowed = set(map(str, allowed_donors))
        ix = np.flatnonzero(np.isin(donors, list(allowed)))
        if not len(ix) or not allowed.issubset(set(donors)):
            raise ValueError('TRAIN_DONORS_MISSING')
        x = coding_log[ix]
        w = donor_weights(donors[ix]); w /= w.sum()
        if sparse.issparse(x):
            mean = np.asarray(x.T @ w).ravel()
            variance = np.asarray(x.power(2).T @ w).ravel() - mean**2
        else:
            mean = w @ x; variance = w @ (x*x) - mean**2
        genes = np.flatnonzero(variance > 1e-8)
        genes = genes[np.argsort(-variance[genes], kind='stable')[:max_genes]]
        if len(genes) < 2 or len(ix) < 3:
            raise ValueError('INSUFFICIENT_CODING_SPACE')
        scale = np.sqrt(variance[genes])
        rng = np.random.default_rng(seed)
        balanced = []
        local_donors = donors[ix]
        for donor in sorted(allowed):
            rows = np.flatnonzero(local_donors == donor)
            if len(rows) > max_cells_per_donor:
                rows = np.sort(rng.choice(rows, max_cells_per_donor, replace=False))
            balanced.extend(rows.tolist())
        balanced = np.asarray(balanced, dtype=np.int64)
        if len(balanced) < 3:
            raise ValueError('INSUFFICIENT_BALANCED_CODING_SPACE')
        sample = x[balanced][:, genes]
        sample = sample.toarray() if sparse.issparse(sample) else np.asarray(sample)
        z = np.clip((sample - mean[genes])/scale, -10, 10)
        # Every donor contributes at most the same registered number of cells.
        sample_donors = local_donors[balanced]
        sample_w = donor_weights(sample_donors); sample_w /= sample_w.sum()
        center_z = sample_w @ z
        pca = PCA(n_components=min(components, len(genes), len(balanced)-1), svd_solver='randomized', random_state=seed)
        pca.fit(z-center_z)
        return cls(genes, mean[genes], scale, pca, center_z, tuple(sorted(allowed)))

    def transform(self, coding_log):
        x = coding_log[:, self.genes]
        x = x.toarray() if sparse.issparse(x) else np.asarray(x)
        return self.pca.transform(np.clip((x-self.mean)/self.scale, -10, 10)-self.center_z)


def local_disjoint_groups(z, metadata, *, target_size=5, min_size=3,
                          radius_quantile=.9, allowed_training_donors=(),
                          radius_sample_per_context=2000, candidate_multiplier=4,
                          seed=1729):
    """Fit locality radius on training contexts, apply inside each context.

    No reuse of a cell across groups. Optional state_id prohibits state mixing.
    Small/isolated cells stay unassigned rather than being merged across gaps.
    Radius is an unsupervised, prespecified geometry threshold, not tuned to rho.
    """
    if not (2 <= min_size <= target_size) or not 0 < radius_quantile <= 1:
        raise ValueError('INVALID_GROUPING_POLICY')
    meta = metadata.reset_index(drop=True)
    required = ['donor_id', 'cell_type']
    if meta[required].isna().any().any() or len(meta) != len(z):
        raise ValueError('CONTEXT_ALIGNMENT')
    keys = required + (['state_id'] if 'state_id' in meta else [])
    contexts = list(meta.groupby(keys, sort=True, observed=True).indices.items())
    distances = []
    rng = np.random.default_rng(seed)
    allowed = set(map(str, allowed_training_donors))
    for key, ix in contexts:
        if str(key[0]) in allowed and len(ix) >= target_size:
            radius_ix = np.asarray(ix, dtype=np.int64)
            if len(radius_ix) > radius_sample_per_context:
                radius_ix = np.sort(rng.choice(
                    radius_ix, radius_sample_per_context, replace=False
                ))
            d, _ = NearestNeighbors(n_neighbors=target_size).fit(z[ix]).kneighbors(z[radius_ix])
            distances.extend(d[:, -1].tolist())
    if not distances:
        raise ValueError('NO_TRAIN_CONTEXT_FOR_RADIUS')
    radius = float(np.quantile(distances, radius_quantile))
    assignments = np.full(len(z), -1, dtype=np.int64)
    reports = []; group_number = 0
    for key, ix in contexts:
        local = z[ix]; unused = np.ones(len(ix), bool)
        neighbor_count = min(len(ix), max(target_size, target_size * candidate_multiplier))
        distances, neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(local).kneighbors(local)
        # Sparse neighborhoods first preserve minor states when they support a group.
        local_counts = (distances <= max(radius, 1e-12)).sum(axis=1)
        order = sorted(range(len(ix)), key=lambda i: (int(local_counts[i]), i))
        diameters = []; sizes = []
        for seed in order:
            if not unused[seed]:
                continue
            within = distances[seed] <= max(radius, 1e-12)
            candidates = neighbors[seed][within]
            candidates = candidates[unused[candidates]]
            if seed not in candidates:
                candidates = np.r_[seed, candidates]
            chosen = candidates[:target_size]
            if len(chosen) < min_size:
                continue
            unused[chosen] = False
            assignments[ix[chosen]] = group_number; group_number += 1
            sizes.append(len(chosen))
            diameters.append(float(np.max(np.linalg.norm(local[chosen]-local[seed], axis=1))))
        reports.append({'donor_id':str(key[0]), 'cell_type':str(key[1]),
                        'state_id':str(key[2]) if len(key)>2 else None,
                        'n_cells':len(ix), 'n_metacells':len(sizes),
                        'unassigned_cells':int(unused.sum()), 'sizes':sizes,
                        'max_seed_radius':max(diameters, default=0.)})
    return assignments, {'training_donors':sorted(allowed), 'radius':radius,
                         'target_size':target_size, 'min_size':min_size,
                         'radius_quantile':radius_quantile,
                         'radius_sample_per_context':radius_sample_per_context,
                         'candidate_multiplier':candidate_multiplier,
                         'contexts':reports,
                         'overlapping_cells':0, 'lncrna_targets_used':False}


def aggregate_measurements(expression, assignments, *, scale, total_library=None):
    """Return metacell log-expression with an explicit scale contract.

    Raw counts: sum then library-normalize/log1p using ALL original genes.
    Unknown log-normalized source: arithmetic mean on its provided log scale;
    this is a separately named alternative, never expm1 or a count sum.
    """
    labels = np.asarray(assignments, int)
    keep = labels >= 0
    groups = np.unique(labels[keep])
    if not len(groups):
        raise ValueError('NO_METACELLS')
    index = np.searchsorted(groups, labels[keep])
    pooling = sparse.csr_matrix((np.ones(keep.sum()), (index, np.flatnonzero(keep))),
                                shape=(len(groups), len(labels)))
    values = sparse.csr_matrix(pooling @ expression, dtype=float)
    if scale == 'raw_counts':
        if total_library is None:
            raise ValueError('ALL_GENE_LIBRARY_REQUIRED')
        denominator = np.asarray(pooling @ np.asarray(total_library)).ravel()
        if (denominator <= 0).any():
            raise ValueError('EMPTY_LIBRARY')
        values = sparse.diags(10000. / denominator) @ values
        values.data = np.log1p(values.data)
        policy = 'RAW_SUM_ALL_GENE_LIBRARY_1E4_LOG1P'
    elif scale == 'log_normalized':
        values = sparse.diags(1. / np.asarray(pooling.sum(axis=1)).ravel()) @ values
        policy = 'PROVIDED_LOG_SCALE_ARITHMETIC_MEAN_NO_INVERSION'
    else:
        raise ValueError('UNKNOWN_MEASUREMENT_SCALE')
    return values.tocsr(), groups, policy
