"""Attach source-specific WikiPathways measurements to a fixed pathway Y.

Equivalent definitions share a primary identity. Composition context remains a
source-level target, accessible without silently becoming a backbone label.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def merge_primary_observations(old_values, old_mask, old_observations, old_links,
                               source_values, source_mask, source_observations,
                               source_links, exact_routes, targets):
    from .measured_y_registry import verify_matrix
    patients = old_values.index.tolist()
    verify_matrix(old_values, old_mask, patients)
    verify_matrix(source_values, source_mask, patients)
    for frame in (old_observations, source_observations):
        if not frame.observation_id.is_unique:
            raise ValueError('WP_DUPLICATE_OBSERVATION')
    molecular_keys = ['kind', 'layer', 'feature_id']
    if set(molecular_keys) <= set(old_observations) and set(molecular_keys) <= set(source_observations):
        left = old_observations[old_observations.kind.eq('member_measurement')]
        right = source_observations[source_observations.kind.eq('member_measurement')]
        if left.duplicated(molecular_keys).any() or right.duplicated(molecular_keys).any():
            raise ValueError('WP_DUPLICATE_MOLECULAR_KEY')
        shared = left[molecular_keys + ['observation_id']].merge(right[molecular_keys + ['observation_id']], on=molecular_keys)
        if not shared.observation_id_x.eq(shared.observation_id_y).all():
            raise ValueError('WP_SHARED_MOLECULE_ID_CHANGED')
    if set(old_values) != set(old_observations.observation_id):
        raise ValueError('WP_OLD_METADATA_AXIS')
    if set(source_values) != set(source_observations.observation_id):
        raise ValueError('WP_SOURCE_METADATA_AXIS')
    if not set(source_links.observation_id) <= set(source_values):
        raise ValueError('WP_SOURCE_LINK_AXIS')
    if exact_routes.duplicated(['source_target_id', 'canonical_id']).any():
        raise ValueError('WP_DUPLICATE_EXACT_ROUTE')
    if not set(exact_routes.canonical_id) <= set(targets.canonical_id):
        raise ValueError('WP_UNKNOWN_PRIMARY')
    if 'route_type' in exact_routes and not exact_routes.route_type.eq('equivalent_definition').all():
        raise ValueError('WP_CONTEXT_CANNOT_BECOME_PRIMARY_LABEL')
    common = sorted(set(old_values) & set(source_values))
    for column in common:
        if not np.array_equal(old_values[column].to_numpy(), source_values[column].to_numpy(), equal_nan=True):
            raise ValueError('WP_SHARED_MEASUREMENT_DIFFERS: ' + column)
        if not old_mask[column].equals(source_mask[column]):
            raise ValueError('WP_SHARED_MASK_DIFFERS: ' + column)
    route = source_links.merge(exact_routes[['source_target_id', 'canonical_id']], on='source_target_id', how='inner')
    if 'evidence_role' in route:
        route = route[~route.evidence_role.eq('ambiguous_group_context')].copy()
    if 'is_source_observation' in route:
        route = route[route.is_source_observation.astype(str).str.lower().isin(['true', '1'])].copy()
    smeta = source_observations.set_index('observation_id')
    route['evidence_role'] = route.observation_id.map(smeta.kind).map({
        'member_measurement': 'member_measurement',
        'source_readout': 'equivalent_pathway_readout'})
    if route.evidence_role.isna().any():
        raise ValueError('WP_UNKNOWN_OBSERVATION_KIND')
    conditions = targets.set_index('canonical_id').condition_required
    route['condition_required'] = route.canonical_id.map(conditions)
    route['scope_relation'] = 'equivalent_source_definition'
    route['is_target_observation'] = True
    route['independent_loss_enabled'] = False
    route['definition_source'] = 'WikiPathways'
    selected_ids = set(route.observation_id)
    new_ids = [x for x in source_values if x in selected_ids and x not in old_values]
    values = pd.concat([old_values, source_values[new_ids]], axis=1)
    mask = pd.concat([old_mask, source_mask[new_ids]], axis=1)
    values.columns.name = old_values.columns.name
    values.index.name = old_values.index.name
    mask.columns.name = old_mask.columns.name
    mask.index.name = old_mask.index.name
    metadata = pd.concat([old_observations, source_observations[source_observations.observation_id.isin(new_ids)]], ignore_index=True)
    metadata = metadata.set_index('observation_id').loc[list(values)].reset_index()
    existing_pairs = set(zip(old_links.canonical_id, old_links.observation_id))
    additions = route.loc[[pair not in existing_pairs for pair in zip(route.canonical_id, route.observation_id)]]
    additions = additions.drop_duplicates(['canonical_id', 'observation_id'])
    links = pd.concat([old_links, additions], ignore_index=True)
    verify_matrix(values, mask, patients)
    return values, mask, metadata, links, route


def load_wikipathways_source(root, case, source_target_id, *, layer=None,
                             kind=None, for_supervision=False):
    """Read a WP definition's own observations, without asserting equivalence.

root is a fixed Y bundle containing sources/WikiPathways. The same measurement
ID may be linked to several definitions; a caller should count it once.
"""
    folder = Path(root) / 'sources' / 'WikiPathways' / 'cohorts' / case
    receipt = json.loads((folder / 'COMPLETE.json').read_text())
    names = ['OBSERVATIONS.tsv.gz', 'SOURCE_OBSERVATIONS.tsv.gz', 'Y_VALUES.parquet', 'Y_OBSERVED.parquet']
    for name in names:
        expected = receipt['files'][name]
        expected = expected.get('sha256') if isinstance(expected, dict) else expected
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('WP_SOURCE_FILE_DRIFT')
    links = pd.read_csv(folder / 'SOURCE_OBSERVATIONS.tsv.gz', sep='\t', keep_default_na=False)
    obs = pd.read_csv(folder / 'OBSERVATIONS.tsv.gz', sep='\t', keep_default_na=False)
    links = links[links.source_target_id.eq(source_target_id)].copy()
    if 'is_source_observation' in links:
        links = links[links.is_source_observation.astype(str).str.lower().isin(['true', '1'])]
    if 'evidence_role' in links:
        links = links[~links.evidence_role.eq('ambiguous_group_context')]
    ids = set(links.observation_id)
    if not ids:
        definitions = pd.read_csv(folder.parent.parent / 'SOURCE_TARGETS.tsv', sep='\t', keep_default_na=False)
        if source_target_id not in set(definitions.source_target_id):
            raise KeyError(source_target_id)
    obs = obs[obs.observation_id.isin(ids)]
    if layer is not None:
        obs = obs[obs.layer.eq(layer)]
    if kind is not None:
        obs = obs[obs.kind.eq(kind)]
    columns = obs.observation_id.tolist()
    values = pd.read_parquet(folder / 'Y_VALUES.parquet', columns=columns)
    mask = pd.read_parquet(folder / 'Y_OBSERVED.parquet', columns=columns)
    if for_supervision:
        mask = mask.mul(obs.set_index('observation_id').supervision_eligible.astype(bool), axis=1).astype(bool)
    return values, mask, obs
