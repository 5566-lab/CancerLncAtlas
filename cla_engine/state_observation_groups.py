"""Organize measured State components under biological collection entries."""
from __future__ import annotations

import hashlib
import pandas as pd

GROUPS = {
    'cell_composition': ('Cellular composition', '免疫、基质与组织细胞组成'),
    'sbs_spectrum': ('SBS mutational signature spectrum', '突变特征谱（SBS）'),
}


def group_id(family):
    if family not in GROUPS:
        raise ValueError('UNKNOWN_STATE_GROUP')
    return 'CLA-S-' + hashlib.sha256(('CLA_STATE_GROUP_V1:' + family).encode()).hexdigest()[:16].upper()


def state_membership(states, definitions):
    if not states.canonical_id.is_unique:
        raise ValueError('DUPLICATE_STATE_IDENTITY')
    selected = definitions[definitions.canonical_id.isin(states.canonical_id)]
    selected = selected[selected.state_family.isin(GROUPS)][['canonical_id', 'state_family']].drop_duplicates()
    if selected.canonical_id.duplicated().any():
        raise ValueError('MULTIPLE_STATE_PARENTS')
    selected = selected.rename(columns={'canonical_id': 'component_id'})
    selected['parent_id'] = selected.state_family.map(group_id)
    selected['parent_name'] = selected.state_family.map(lambda x: GROUPS[x][0])
    selected['parent_name_zh'] = selected.state_family.map(lambda x: GROUPS[x][1])
    selected['component_name'] = selected.component_id.map(states.set_index('canonical_id').display_name)
    selected['relation'] = 'component_of'
    return selected.sort_values(['parent_id', 'component_id']).reset_index(drop=True)


def group_observation_links(links, membership):
    if not membership.component_id.is_unique:
        raise ValueError('MULTIPLE_STATE_PARENTS')
    parent = membership.set_index('component_id').parent_id.to_dict()
    out = links.copy()
    selected = out.canonical_id.isin(parent)
    out['state_component_ids'] = ''
    untouched = out.loc[~selected].copy()
    out = out.loc[selected].copy()
    selected = out.index
    out.loc[selected, 'state_component_ids'] = out.loc[selected, 'canonical_id']
    out.loc[selected, 'canonical_id'] = out.loc[selected, 'canonical_id'].map(parent)
    out.loc[selected, 'scope_relation'] = 'state_component_of'
    out.loc[selected, 'routing_basis'] = 'biological_state_collection'
    out.loc[selected, 'evidence_role'] = 'state_component_readout'
    keys = ['canonical_id', 'observation_id', 'evidence_role']
    def union(values):
        return '|'.join(sorted({s for value in values for s in str(value).split('|') if s}))
    aggcols = [c for c in ['state_component_ids', 'source_canonical_ids', 'molecular_components'] if c in out]
    values = out.groupby(keys, sort=False)[aggcols].agg(union).reset_index()
    grouped = out.drop_duplicates(keys).drop(columns=aggcols).merge(values, on=keys, validate='one_to_one')
    return pd.concat([untouched, grouped], ignore_index=True)
