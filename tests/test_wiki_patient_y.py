import numpy as np
import pandas as pd
import pytest
from cla_engine.wiki_patient_y import merge_primary_observations


def fixture():
    v = pd.DataFrame({'shared': [0., np.nan]}, index=['p1', 'p2'])
    v.columns.name = 'observation_id'
    s = pd.DataFrame({'shared': [0., np.nan], 'wp_score': [1., 2.], 'context_score': [3., 4.]}, index=v.index)
    meta = pd.DataFrame({'observation_id': list(s), 'kind': ['member_measurement', 'source_readout', 'source_readout']})
    links = pd.DataFrame({'canonical_id': ['main'], 'observation_id': ['shared'], 'evidence_role': ['member_measurement']})
    source_links = pd.DataFrame({'source_target_id': ['WP1', 'WP1', 'WP2'], 'observation_id': list(s)})
    routes = pd.DataFrame({'source_target_id': ['WP1'], 'canonical_id': ['main'], 'route_type': ['equivalent_definition']})
    targets = pd.DataFrame({'canonical_id': ['main'], 'condition_required': [False]})
    return [v, v.notna(), meta.iloc[:1], links, s, s.notna(), meta, source_links, routes, targets]


def test_unique_measurements_context_not_promoted_and_missing_preserved():
    values, mask, metadata, links, provenance = merge_primary_observations(*fixture())
    assert list(values) == ['shared', 'wp_score']
    assert values.loc['p1', 'shared'] == 0 and mask.loc['p1', 'shared']
    assert not mask.loc['p2', 'shared']
    assert len(links) == 2 and len(metadata) == 2
    assert len(provenance) == 2
    assert values.columns.name == mask.columns.name == 'observation_id'


def test_conflicting_shared_molecular_value_rejected():
    args = fixture()
    args[4].loc['p1', 'shared'] = 9
    with pytest.raises(ValueError, match='SHARED_MEASUREMENT_DIFFERS'):
        merge_primary_observations(*args)


def test_context_route_cannot_be_declared_equivalent_by_accident():
    args = fixture()
    args[8]['route_type'] = 'composition_context'
    with pytest.raises(ValueError, match='CONTEXT_CANNOT_BECOME_PRIMARY_LABEL'):
        merge_primary_observations(*args)


def test_patient_partition_order_must_match():
    args = fixture()
    args[4] = args[4].iloc[::-1]
    args[5] = args[5].iloc[::-1]
    with pytest.raises(ValueError, match='Y_PATIENT_ORDER'):
        merge_primary_observations(*args)


def test_ambiguous_assay_does_not_become_primary_supervision():
    args = fixture()
    args[7]['is_source_observation'] = [True, False, True]
    values, mask, metadata, links, provenance = merge_primary_observations(*args)
    assert list(values) == ['shared']
    assert set(provenance.observation_id) == {'shared'}
