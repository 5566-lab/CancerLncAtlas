"""Rebuild nested statistics from measured cells and supplied frozen membership.

Geometry fitting and LASSO are never called. The supplied grouping itself is
not independently validated as the result of a particular PCA/radius fit.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .cnv_input_sources import require_host
from .formal_expression_sources import read, record
from .current_core_export import verified_path
from .single_cell_graph_assets import align_frozen_membership, aggregate_frozen_measurements, validate_frozen_metacell_metadata
from .single_cell_nested_episodes import donor_roles, lasso_observations, source_node_features, validate_nested_isolation

ROOT = Path(__file__).resolve().parents[2]
CODE = ('cc_hhgt/v32/single_cell_nested_numeric_replay.py',
        'cc_hhgt/v32/single_cell_geometry_replay.py',
        'scripts/materialize_single_cell_nested_hhgt_episodes_149.py',
        'scripts/materialize_single_cell_metacell_149.py',
        'cc_hhgt/v32/single_cell_nested_episodes.py', 'cc_hhgt/v32/single_cell_metacell_screen.py',
        'cc_hhgt/v32/single_cell_metacells.py', 'cc_hhgt/v32/single_cell_graph_assets.py',
        'cc_hhgt/v32/single_cell_cell_level.py')
_CACHE = {}


def implementations():
    return [record(ROOT/name) for name in CODE]


def _table(path, expected):
    try: pd.testing.assert_frame_equal(pd.read_parquet(path), expected, check_exact=True)
    except AssertionError as error: raise ValueError('Nested reconstruction differs: '+path.name) from error


def _array(observed, expected, name):
    if (observed.dtype != expected.dtype or observed.shape != expected.shape
            or not np.array_equal(observed, expected, equal_nan=True)):
        raise ValueError('Nested reconstruction differs: '+name)


def _matrix(path, expected):
    actual = sparse.load_npz(path)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or (actual != expected).nnz:
        raise ValueError('Nested reconstruction differs: '+path.name)


def verify_plan(settings, plan, mapped, matrices, ids, bulk, family_membership,
                static_root, matrix_identity_receipt, folder, receipt):
    """Read and compare a complete original plan without fitting or writing."""
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    from scripts import materialize_single_cell_metacell_149 as base
    require_host(); folder = Path(folder)
    partition = {'format':native.NESTED_ASSET_FORMAT, 'target_host':'149', **plan.as_dict(),
                 'donor_partition_sha256':native.canonical_hash(plan.as_dict())}
    if read(folder/'DONOR_PARTITION.json') != partition:
        raise ValueError('Nested reconstruction donor partition differs')
    fields = {'cancer_id':plan.cancer_id,'outer_fold_id':plan.outer_fold_id,'role':plan.role,
        'inner_target_fold_id':plan.inner_target_fold_id,'inner_validation_fold_id':plan.inner_validation_fold_id,
        'source_donors':list(plan.source_donors),'target_donors':list(plan.inner_target_donors),
        'validation_donors':list(plan.inner_validation_donors),'outer_query_donors':list(plan.outer_query_donors),
        'excluded_donors':list(plan.excluded_donors),'geometry_isolation':plan.geometry_isolation,
        'validation_provider':plan.validation_provider,'support_limited_fit':plan.support_limited_fit,
        'minimum_source_donors':plan.minimum_source_donors}
    if any(receipt.get(k) != v for k,v in fields.items()): raise ValueError('Nested reconstruction plan metadata differs')
    if not plan.available:
        expected = {**partition,'state':'NESTED_EPISODE_UNAVAILABLE',
                    'reason':f'INSUFFICIENT_SOURCE_DONORS_LT{plan.minimum_source_donors}'}
        if (read(folder/'UNAVAILABLE.json') != expected or receipt.get('state') != expected['state']
                or receipt.get('unavailable_reason') != expected['reason']):
            raise ValueError('Nested reconstruction unavailable state differs')
        return
    metadata = matrices['metadata']
    keep = metadata.donor_id.isin(set(plan.source_donors)|set(plan.inner_target_donors)).to_numpy()
    retained = metadata.loc[keep].reset_index(drop=True)
    membership = pd.read_parquet(folder/'METACELL_MEMBERSHIP.parquet')
    assignments = align_frozen_membership(retained,membership)
    from .single_cell_geometry_replay import policies, verify
    seed = settings.seed + plan.outer_fold_id*1000 + plan.inner_target_fold_id*100
    fit, grouping = policies(settings, seed=seed, radius_seed=seed+1)
    verify(folder, matrices['coding_log'][keep], retained, plan.source_donors, fit, grouping,
           settings.pca_components, assignments, read(folder/'GEOMETRY_AUDIT.json'))
    saved_meta = pd.read_parquet(folder/'METACELL_METADATA.parquet')
    validate_frozen_metacell_metadata(retained,assignments,saved_meta)
    coding,lncrna,groups,policy = aggregate_frozen_measurements(
        matrices['coding'][keep],matrices['lncrna'][keep],assignments,
        measurement_scale=mapped['measurement_scale'],all_gene_library=matrices['library'][keep])
    first = [np.flatnonzero(assignments==group)[0] for group in groups]
    keys = [key for key in ('dataset_id','cancer_id','donor_id','cell_type','state_id') if key in retained]
    expected_meta = retained.iloc[first][keys].reset_index(drop=True)
    expected_meta['metacell_id'] = groups
    sizes = np.bincount(assignments[assignments>=0]); expected_meta['n_cells'] = sizes[groups]
    roles = donor_roles(plan)
    expected_meta['observation_role'] = expected_meta.donor_id.map(roles)
    expected_meta['lasso_input_allowed'] = expected_meta.observation_role.eq('source')
    expected_membership = retained[['cell_id','donor_id','cell_type','state_id']].copy()
    expected_membership['metacell_id'] = assignments
    expected_membership['observation_role'] = expected_membership.donor_id.map(roles)
    expected_membership['geometry_fit_allowed'] = expected_membership.observation_role.eq('source')
    _table(folder/'METACELL_MEMBERSHIP.parquet',expected_membership)
    _table(folder/'METACELL_METADATA.parquet',expected_meta)
    _matrix(folder/'METACELL_CODING.npz',coding); _matrix(folder/'METACELL_LNCRNA.npz',lncrna)
    activity = base._ucell_from_metacells(coding,matrices['signature'],matrices['signature_counts'],
                                        matrices['missing_counts'],settings.deadline).astype(np.float32,copy=False)
    _array(np.load(folder/'METACELL_PATHWAY_ACTIVITY.npy',allow_pickle=False),activity,'METACELL_PATHWAY_ACTIVITY.npy')
    metacell = {'coding':coding,'lncrna':lncrna,'metadata':expected_meta}
    observations = lasso_observations(expected_meta,plan)
    _table(folder/'LASSO_OBSERVATIONS.parquet',observations)
    _table(folder/'LASSO_SOURCE_ROWS.parquet',observations.loc[observations.lasso_input_allowed].reset_index(drop=True))
    universe,targets,coverage = native._context_assets(metacell,activity,ids,bulk,plan,settings,
        dataset_id=str(mapped['dataset_id']),measurement_scale=str(mapped['measurement_scale']))
    _table(folder/'CANDIDATE_UNIVERSE.parquet',universe)
    _table(folder/'SIGNED_TARGET_ROWS.parquet',targets)
    features,contexts = source_node_features(coding,lncrna,activity,expected_meta,plan,family_membership,target_rows=targets)
    _table(folder/'NODE_FEATURE_CONTEXTS.parquet',contexts)
    with np.load(folder/'NODE_FEATURES.npz',allow_pickle=False) as stored:
        if set(stored.files) != set(features): raise ValueError('Nested reconstruction node feature schema differs')
        for name,array in features.items(): _array(stored[name],array,'NODE_FEATURES.npz/'+name)
    if read(folder/'CONTEXT_COVERAGE.json') != {'format':native.NESTED_ASSET_FORMAT,
                                             'geometry_isolation':plan.geometry_isolation,'contexts':coverage}:
        raise ValueError('Nested reconstruction context coverage differs')
    validate_nested_isolation(plan,read(folder/'GEOMETRY_AUDIT.json'),observations,universe,targets)
    state = 'NESTED_EPISODE_ASSETS_READY' if len(targets) and len(contexts) else 'NESTED_EPISODE_NO_ESTIMABLE_CONTEXTS'
    expected = {'state':state,'unavailable_reason':None if state=='NESTED_EPISODE_ASSETS_READY' else state,
        'measurement_scale':mapped['measurement_scale'],'aggregation_policy':policy,'metacells':len(expected_meta),
        'assigned_cells':int((assignments>=0).sum()),'unassigned_cells':int((assignments<0).sum()),
        'lasso_source_observations':int(observations.lasso_input_allowed.sum()),
        'lasso_excluded_observations':int((~observations.lasso_input_allowed).sum()),
        'candidate_rows':len(universe),'signed_target_rows':len(targets),'feature_contexts':len(contexts),
        'graph_asset_static_success_sha256':native.sha256(static_root/'SUCCESS.json'),
        'matrix_identity_sha256':native.sha256(matrix_identity_receipt)}
    if any(receipt.get(k) != v for k,v in expected.items()): raise ValueError('Nested reconstruction summary differs')
    if read(folder/'STATIC_BINDING.json') != {'format':native.NESTED_ASSET_FORMAT,
            'graph_asset_static_success':native.file_record(static_root/'SUCCESS.json'),
            'matrix_identity':native.file_record(matrix_identity_receipt)}:
        raise ValueError('Nested reconstruction static binding differs')


def verify_completed(value, donor_path, settings, metadata, plans):
    """Cache proofs per plan only after checking current source/output bytes."""
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    from scripts import materialize_single_cell_metacell_149 as base
    require_host(); mapped_path = Path(value['mapped_success']['path']); mapped = read(mapped_path)

    def inputs():
        native._verify_record({**value['mapped_success'],'bytes':mapped_path.stat().st_size},mapped_path,'NESTED_RAW_MAPPING')
        pins = [record(mapped_path),record(donor_path),*implementations()]
        for pin in [value['candidates'],value['donor_crosswalk'],value['graph_preflight'],*value['static_sources']]:
            pins.append(record(verified_path(pin,donor_path.parent)))
        for name,pin in mapped['files'].items():
            native._verify_record(pin,mapped_path.parent/name,'NESTED_RAW_INPUT')
            pins.append(record(mapped_path.parent/name))
        return pins

    def outputs(folder, receipt):
        pins = [record(folder/'SUCCESS.json')]
        for name,pin in receipt['files'].items():
            native._verify_record(pin,folder/name,'NESTED_NUMERIC_OUTPUT');pins.append(record(folder/name))
        return pins

    before = inputs(); matrices = None; pending = []
    for plan,folder,receipt in plans:
        signature = {'inputs':before,'value':value,'plan':plan.as_dict(),'outputs':outputs(folder,receipt)}
        if _CACHE.get(str(folder)) == signature: continue
        if matrices is None:
            raw = mapped_path.parent
            matrices = {'metadata':metadata,'coding':sparse.load_npz(raw/'CODING_MEASURED.npz').tocsr(),
                'coding_log':sparse.load_npz(raw/'CODING_LOG_FOR_GROUPING.npz').tocsr(),
                'lncrna':sparse.load_npz(raw/'LNC_MEASURED.npz').tocsr(),
                'library':np.load(raw/'ALL_GENE_LIBRARY.npy',allow_pickle=False),
                'signature':sparse.load_npz(raw/'EXACT_SIGNATURE_MATRIX.npz').tocsr(),
                'signature_counts':np.load(raw/'SIGNATURE_GENE_COUNTS.npy',allow_pickle=False),
                'missing_counts':np.load(raw/'MISSING_GENE_COUNTS.npy',allow_pickle=False)}
            ids = {k:list(map(str,v)) for k,v in read(raw/'IDS.json').items()}
            static_root,_,family,_ = native._validate_static(settings,value['cancer_id'],ids)
            bulk = base._bulk_candidates(settings.candidates,value['cancer_id'],ids['lncrna_ids'],ids['pathway_ids'])
            identity = donor_path.parent/'nested/cancers'/value['cancer_id']/'MATRIX_IDENTITY.json'
        verify_plan(settings,plan,mapped,matrices,ids,bulk,family,static_root,identity,folder,receipt)
        if signature['outputs'] != outputs(folder,receipt): raise ValueError('Nested outputs changed during reconstruction')
        pending.append((str(folder),signature))
    if before != inputs(): raise ValueError('Nested inputs changed during reconstruction')
    _CACHE.update(pending)
