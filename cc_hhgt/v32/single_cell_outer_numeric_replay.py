"""Reconstruct original outer train/query statistics with supplied grouping.

No geometry or regression is fitted. Frozen membership is an input, not proof
of the fitted PCA/radius procedure that originally chose that membership.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .single_cell_graph_assets import align_frozen_membership, validate_frozen_metacell_metadata

FORMAT = 'NATIVE_SC_OUTER_FROZEN_STATISTICS_REPLAY_V1'
ROOT = Path(__file__).resolve().parents[2]
CODE = ('cc_hhgt/v32/single_cell_outer_numeric_replay.py',
        'scripts/materialize_single_cell_metacell_149.py',
        'cc_hhgt/v32/single_cell_metacells.py', 'cc_hhgt/v32/single_cell_metacell_screen.py',
        'cc_hhgt/v32/single_cell_cell_level.py', 'cc_hhgt/v32/single_cell_graph_assets.py')
FILES = {'METACELL_MEMBERSHIP.parquet', 'METACELL_METADATA.parquet',
         'TRAIN_ROWS.parquet', 'QUERY_ROWS.parquet', 'UNIVERSE.parquet',
         'GEOMETRY_AUDIT.json', 'CONTEXT_COVERAGE.json'}
_CACHE = {}


def _table(path, expected):
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(path), expected, check_exact=True)
    except AssertionError as error:
        raise ValueError('Outer reconstruction differs: '+path.name) from error


def verify_fold(settings, canonical, fold, folder, receipt, mapped):
    """Rebuild all numeric targets, candidate selection and source features."""
    from scripts import materialize_single_cell_metacell_149 as native
    require_host(); folder = Path(folder)
    metadata, folds, _ = canonical
    query = set(folds.loc[folds.outer_fold_id.eq(fold), 'donor_id'].astype(str))
    source = set(folds.donor_id.astype(str)) - query
    raw = settings.scratch_root/settings.cancer_id.upper()
    assignments = align_frozen_membership(metadata, pd.read_parquet(folder/'METACELL_MEMBERSHIP.parquet'))
    validate_frozen_metacell_metadata(metadata, assignments, pd.read_parquet(folder/'METACELL_METADATA.parquet'))
    library = np.load(raw/'ALL_GENE_LIBRARY.npy', allow_pickle=False)
    coding, groups, policy = native.aggregate_measurements(
        sparse.load_npz(raw/'CODING_MEASURED.npz'), assignments,
        scale=mapped['measurement_scale'], total_library=library)
    lncrna, lnc_groups, lnc_policy = native.aggregate_measurements(
        sparse.load_npz(raw/'LNC_MEASURED.npz'), assignments,
        scale=mapped['measurement_scale'], total_library=library)
    if not np.array_equal(groups, lnc_groups) or policy != lnc_policy:
        raise ValueError('Outer reconstruction aggregation order/policy differs')
    first = np.asarray([np.flatnonzero(assignments == group)[0] for group in groups])
    expected = metadata.iloc[first][['dataset_id','cancer_id','donor_id','cell_type','state_id']].reset_index(drop=True)
    expected['metacell_id'] = groups
    expected['n_cells'] = np.bincount(assignments[assignments >= 0])[groups]
    membership = metadata[['cell_id','donor_id','cell_type','state_id']].copy()
    membership['metacell_id'] = assignments
    _table(folder/'METACELL_MEMBERSHIP.parquet', membership)
    _table(folder/'METACELL_METADATA.parquet', expected)
    activity = native._ucell_from_metacells(coding, sparse.load_npz(raw/'EXACT_SIGNATURE_MATRIX.npz'),
        np.load(raw/'SIGNATURE_GENE_COUNTS.npy',allow_pickle=False),
        np.load(raw/'MISSING_GENE_COUNTS.npy',allow_pickle=False), settings.deadline)
    ids = read(raw/'IDS.json')
    bulk = native._bulk_candidates(settings.candidates, settings.cancer_id.upper(), ids['lncrna_ids'], ids['pathway_ids'])
    train, query_rows, universe, contexts = native._context_targets(
        lncrna, activity, expected, ids, bulk, source, query, mapped,
        settings.cancer_id.upper(), fold, settings)
    for name, frame in (('TRAIN_ROWS.parquet',train), ('QUERY_ROWS.parquet',query_rows), ('UNIVERSE.parquet',universe)):
        _table(folder/name, frame)
    if read(folder/'CONTEXT_COVERAGE.json') != contexts:
        raise ValueError('Outer reconstruction context coverage differs')
    fields = {'source_donor_hash':native.canonical_hash(sorted(source)),
        'query_donor_hash':native.canonical_hash(sorted(query)), 'limited_replication':bool(folds.limited_replication.any()),
        'metacells':len(expected),'unassigned_cells':int((assignments < 0).sum()),
        'train_rows':len(train),'query_rows':len(query_rows),'universe_rows':len(universe),'contexts':len(contexts),
        'target_policy':native.TARGET_POLICY,'aggregation_policy':policy,
        'query_donor_used_for_geometry_fit':False,'query_donor_used_for_candidate_selection':False,
        'signed_rho_preserved':True}
    if any(receipt.get(key) != value for key,value in fields.items()):
        raise ValueError('Outer reconstruction summary differs')


def verify_completed_fold(settings, canonical, fold, folder, receipt):
    """Memoize only after actual current input/output hashes have been checked.

The caller validates canonical preparation/configuration and donor identities.
All proof inputs and output bytes are checked again before caching a success.
"""
    require_host(); folder = Path(folder)
    if set(receipt.get('files',{})) != FILES:
        raise ValueError('Outer reconstruction output inventory differs')
    raw = settings.scratch_root/settings.cancer_id.upper()
    mapped = read(raw/'MAPPED_SUCCESS.json')

    def inputs():
        pins = [record(raw/'MAPPED_SUCCESS.json'), record(settings.candidates), *[record(ROOT/name) for name in CODE]]
        for name,pin in sorted(mapped['files'].items()):
            path = verified_path(pin,raw)
            if path != (raw/name).resolve(): raise ValueError('Outer reconstruction mapped input location differs')
            pins.append(record(path))
        for label,pin in canonical[2].items():
            path = verified_path(pin,settings.output_root); pins.append(record(path))
            if label == 'native_donor_preparation':
                for item in read(path)['files'].values(): pins.append(record(verified_path(item,path.parent)))
        return pins

    def outputs():
        pins = [record(folder/'SUCCESS.json')]
        if read(folder/'SUCCESS.json') != receipt: raise ValueError('Outer reconstruction receipt changed')
        for name,pin in sorted(receipt['files'].items()):
            path = verified_path(pin,folder)
            if path != (folder/name).resolve(): raise ValueError('Outer reconstruction output location differs')
            pins.append(record(path))
        return pins

    signature = {'inputs':inputs(),'outputs':outputs(),'fold':fold,
                 'settings':{k:str(v) for k,v in vars(settings).items() if k!='deadline'}}
    key = str(folder.resolve())
    if _CACHE.get(key) == signature: return
    verify_fold(settings,canonical,fold,folder,receipt,mapped)
    if signature['inputs'] != inputs() or signature['outputs'] != outputs():
        raise ValueError('Outer reconstruction source or output changed during verification')
    _CACHE[key] = signature
