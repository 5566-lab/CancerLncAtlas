"""Read-only native episode source checks; never dispatch geometry or LASSO."""
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

from .cnv_input_sources import require_host
from .formal_expression_sources import read, record
from . import native_single_cell_graph_workflow as graph
from . import native_single_cell_donors_workflow as donors
from . import single_cell_nested_numeric_replay as replay
from .single_cell_nested_episodes import nested_fold_plans, lasso_observations, validate_nested_isolation, donor_roles
from .single_cell_graph_assets import align_frozen_membership, validate_frozen_metacell_metadata


def nested_inventory(value, donor_path):
    """Check canonical partitions and rebuild statistics with frozen grouping."""
    from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
    output = donor_path.parent / 'nested'
    settings, _ = donors.build_settings(value, donor_path.parent, time.time()+3600)
    native.validate_preflight(settings)
    metadata, folds, _ = donors.derived(value)
    cancer = value['cancer_id']; root = output / 'cancers' / cancer
    pins = {}; numeric_plans = []

    def checked(path, pin=None):
        path = Path(path)
        if pin is not None: native._verify_record(pin, path, 'NATIVE_EPISODE_SOURCE')
        pins[str(path)] = record(path)
        return read(path)

    success = checked(root / 'SUCCESS.json')
    if (success.get('state') != 'CANCER_NESTED_ASSETS_COMPLETE' or success.get('target_host') != '149'
            or success.get('cancer_id') != cancer or success.get('source_kind') != 'NATIVE_CANONICAL_DONOR_PREPARATION'
            or success.get('source_cancer_success_sha256') != record(donor_path)['sha256']
            or success.get('source_dataset_id') != value['dataset_id']):
        raise ValueError('Native nested completion source differs')
    active = sorted(map(int, folds.outer_fold_id.unique()))
    paths = [root / f'outer_fold={fold}' / 'SUCCESS.json' for fold in active]
    if success.get('outer_fold_receipts') != [native.file_record(p) for p in paths]:
        raise ValueError('Native nested outer-fold coverage differs')
    identity = checked(root / 'MATRIX_IDENTITY.json', success['matrix_identity'])
    identity_path = root / 'MATRIX_IDENTITY.parquet'
    native._verify_record(identity['matrix_identity'], identity_path, 'NATIVE_MATRIX_IDENTITY')
    pins[str(identity_path)] = record(identity_path)
    ids = read(Path(value['mapped_success']['path']).parent / 'IDS.json')
    static_root, _, _, families = native._validate_static(settings, cancer, ids)
    if (identity.get('mapped_success') != native.file_record(Path(value['mapped_success']['path']))
            or identity.get('static_success') != native.file_record(static_root / 'SUCCESS.json')):
        raise ValueError('Native nested matrix source differs')
    ordered = {'gene': ids['protein_ids'], 'lncRNA': ids['lncrna_ids'],
               'pathway': ids['pathway_ids'], 'pathway_family': families}
    expected = pd.concat([pd.DataFrame({'node_type':kind, 'matrix_column':np.arange(len(names), dtype=np.int64),
                                       'canonical_id':names}) for kind,names in ordered.items()], ignore_index=True)
    pd.testing.assert_frame_equal(pd.read_parquet(identity_path), expected, check_exact=True)
    if identity.get('id_order_sha256') != native.canonical_hash(ordered):
        raise ValueError('Native nested node order differs')
    ready_files = {'DONOR_PARTITION.json', 'METACELL_MEMBERSHIP.parquet', 'METACELL_METADATA.parquet',
        'METACELL_CODING.npz', 'METACELL_LNCRNA.npz', 'METACELL_PATHWAY_ACTIVITY.npy',
        'LASSO_OBSERVATIONS.parquet', 'LASSO_SOURCE_ROWS.parquet', 'CANDIDATE_UNIVERSE.parquet',
        'SIGNED_TARGET_ROWS.parquet', 'NODE_FEATURE_CONTEXTS.parquet', 'NODE_FEATURES.npz',
        'GEOMETRY_AUDIT.json', 'CONTEXT_COVERAGE.json', 'STATIC_BINDING.json'}
    for outer, path in zip(active, paths):
        receipt = checked(path); plans = nested_fold_plans(folds, cancer_id=cancer, outer_fold_id=outer)
        if (receipt.get('state') != 'OUTER_FOLD_NESTED_ASSETS_COMPLETE' or receipt.get('target_host') != '149'
                or receipt.get('cancer_id') != cancer or receipt.get('outer_fold_id') != outer):
            raise ValueError('Native nested fold identity differs')
        plan_path = path.parent / 'NESTED_PLAN.parquet'; pins[str(plan_path)] = record(plan_path)
        pd.testing.assert_frame_equal(pd.read_parquet(plan_path), native._plan_rows(plans), check_exact=True)
        episode_paths = [path.parent / f'role={p.role}/inner_target_fold={p.inner_target_fold_id}/SUCCESS.json' for p in plans]
        if receipt.get('episode_receipts') != [native.file_record(p) for p in episode_paths]:
            raise ValueError('Native nested episode coverage differs')
        for plan, episode_path in zip(plans, episode_paths):
            current = checked(episode_path); folder = episode_path.parent
            numeric_plans.append((plan, folder, current))
            if native._verify_complete(folder, plan) is None:
                raise ValueError('Completed native nested episode is missing; fitting is forbidden')
            state = current.get('state')
            if state not in {'NESTED_EPISODE_ASSETS_READY', 'NESTED_EPISODE_NO_ESTIMABLE_CONTEXTS', 'NESTED_EPISODE_UNAVAILABLE'}:
                raise ValueError('Native nested episode state differs')
            from .single_cell_geometry_replay import extra_files
            expected_files = (ready_files | extra_files(folder, read(folder/'GEOMETRY_AUDIT.json'))
                              if plan.available else {'DONOR_PARTITION.json', 'UNAVAILABLE.json'})
            if set(current['files']) != expected_files or (state == 'NESTED_EPISODE_UNAVAILABLE') != (not plan.available):
                raise ValueError('Native nested episode inventory or availability differs')
            for name, pin in current['files'].items():
                native._verify_record(pin, folder/name, 'NATIVE_NESTED_FILE'); pins[str(folder/name)] = record(folder/name)
            partition = read(folder/'DONOR_PARTITION.json')
            if partition != {'format':native.NESTED_ASSET_FORMAT, 'target_host':'149', **plan.as_dict(),
                             'donor_partition_sha256':native.canonical_hash(plan.as_dict())}:
                raise ValueError('Native nested donor partition differs')
            fields = {'cancer_id':cancer, 'outer_fold_id':outer, 'inner_target_fold_id':plan.inner_target_fold_id,
                'inner_validation_fold_id':plan.inner_validation_fold_id, 'role':plan.role,
                'source_donors':list(plan.source_donors), 'target_donors':list(plan.inner_target_donors),
                'excluded_donors':list(plan.excluded_donors), 'outer_query_donors':list(plan.outer_query_donors),
                'validation_donors':list(plan.inner_validation_donors)}
            if any(current.get(k) != v for k,v in fields.items()): raise ValueError('Native nested receipt donors differ')
            if not plan.available: continue
            binding = read(folder/'STATIC_BINDING.json')
            if binding != {'format':native.NESTED_ASSET_FORMAT,
                           'graph_asset_static_success':native.file_record(static_root/'SUCCESS.json'),
                           'matrix_identity':native.file_record(root/'MATRIX_IDENTITY.json')}:
                raise ValueError('Native nested static binding differs')
            retained = metadata.loc[metadata.donor_id.isin(set(plan.source_donors)|set(plan.inner_target_donors))].reset_index(drop=True)
            membership = pd.read_parquet(folder/'METACELL_MEMBERSHIP.parquet')
            assignments = align_frozen_membership(retained, membership)
            frozen = pd.read_parquet(folder/'METACELL_METADATA.parquet')
            validate_frozen_metacell_metadata(retained, assignments, frozen)
            roles = donor_roles(plan)
            for frame, flag in ((frozen, 'lasso_input_allowed'), (membership, 'geometry_fit_allowed')):
                expected_roles = frame.donor_id.map(roles)
                if (not pd.api.types.is_bool_dtype(frame[flag]) or not frame.observation_role.equals(expected_roles)
                        or not frame[flag].equals(expected_roles.eq('source'))):
                    raise ValueError('Native nested member roles differ')
            observations = pd.read_parquet(folder/'LASSO_OBSERVATIONS.parquet')
            pd.testing.assert_frame_equal(observations, lasso_observations(frozen, plan), check_exact=True)
            pd.testing.assert_frame_equal(pd.read_parquet(folder/'LASSO_SOURCE_ROWS.parquet'),
                                          observations.loc[observations.lasso_input_allowed].reset_index(drop=True), check_exact=True)
            targets = pd.read_parquet(folder/'SIGNED_TARGET_ROWS.parquet')
            universe = pd.read_parquet(folder/'CANDIDATE_UNIVERSE.parquet')
            validate_nested_isolation(plan, read(folder/'GEOMETRY_AUDIT.json'), observations, universe, targets)
            if (len(targets) != current.get('signed_target_rows') or len(universe) != current.get('candidate_rows')
                    or len(frozen) != current.get('metacells')):
                raise ValueError('Native nested row counts differ')
            if len(targets) and (not targets.cancer_id.eq(cancer).all() or not targets.outer_fold_id.eq(outer).all()
                    or not targets.inner_target_fold_id.eq(plan.inner_target_fold_id).all()
                    or not targets.episode_role.eq(plan.role).all() or not targets.dataset_id.eq(value['dataset_id']).all()):
                raise ValueError('Native nested target identities differ')
    replay.verify_completed(value, donor_path, settings, metadata, numeric_plans)
    return sorted(pins.values(), key=lambda p:p['path']), ordered


def preflight_sources(settings):
    require_host()
    path = Path(settings.native_graph_binding).resolve()
    _, bound = graph.validate(record(path), path.parent)
    value = read(bound['preflight']['path']); donor_path = Path(value['donor_preparation']['path'])
    donor_value = read(read(donor_path)['preflight']['path'])
    if (settings.graph_asset_root.resolve() != path.parent
            or settings.source_result_root.resolve() != donor_path.parent/'metacells'
            or settings.nested_asset_root.resolve() != donor_path.parent/'nested'
            or (settings.cancer_id is not None and settings.cancer_id != value['cancer_id'])):
        raise ValueError('Native episode source roots or cancer differ')
    output = settings.output_root.resolve()
    for source in (path.parent, donor_path.parent):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Native episode output overlaps its inputs')
    if not math.isfinite(settings.deadline) or settings.deadline <= 0:
        raise ValueError('Native episode deadline must be finite and positive')
    if not math.isfinite(settings.lasso_stability_threshold) or not 0 <= settings.lasso_stability_threshold <= 1:
        raise ValueError('Invalid native episode LASSO stability threshold')
    pins, ordered = nested_inventory(donor_value, donor_path)
    nodes = pd.read_parquet(path.parent/'cancers'/value['cancer_id']/'STATIC/NODE_IDENTIFIERS.parquet')
    for kind, names in ordered.items():
        actual = nodes.loc[nodes.node_type.eq(kind)].sort_values('global_index').canonical_id.tolist()
        if actual != names: raise ValueError('Native nested and query graph node order differs')
    return {'native_graph_binding':record(path), 'native_donor_preparation':record(donor_path),
        'nested_preflight':record(settings.nested_asset_root/'LAUNCH_PREFLIGHT.json'),
        'nested_sources':pins, 'authorized_cancers':[value['cancer_id']],
        'source_ancestry_complete':False, 'nested_fitted_numerics_replayed':False,
        'nested_frozen_statistics_replayed':True, 'nested_numeric_implementations':replay.implementations(),
        'source_validator':record(Path(__file__))}


def cache_binding(settings):
    from scripts.materialize_single_cell_hhgt_training_episodes_149 import file_record
    return {'native_episode_preflight':file_record(settings.output_root/'LAUNCH_PREFLIGHT.json')}
