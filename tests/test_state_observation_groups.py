import pandas as pd
import pytest
from cla_engine.state_observation_groups import group_id, state_membership, group_observation_links


def test_only_requested_families_become_children():
    states = pd.DataFrame({'canonical_id':['a','b','c'], 'display_name':['CD8','SBS1','Hypoxia']})
    definitions = pd.DataFrame({'canonical_id':['a','a','b','c'], 'state_family':['cell_composition','cell_composition','sbs_spectrum','hypoxia']})
    m = state_membership(states, definitions)
    assert set(m.component_id) == {'a','b'}
    assert set(m.relation) == {'component_of'}
    assert len(set(m.parent_id)) == 2


def test_ambiguous_component_assignment_rejected():
    states = pd.DataFrame({'canonical_id':['a'], 'display_name':['a']})
    defs = pd.DataFrame({'canonical_id':['a','a'], 'state_family':['cell_composition','sbs_spectrum']})
    with pytest.raises(ValueError, match='MULTIPLE_STATE_PARENTS'):
        state_membership(states, defs)


def test_shared_observations_kept_once_with_all_component_origins():
    links = pd.DataFrame(dict(canonical_id=['a','b','c'],observation_id=['x','x','y'],evidence_role=['same_state_readout']*3,scope_relation=['same_identity']*3,routing_basis=['source']*3,source_canonical_ids=['a','b','c'],molecular_components=['G1','G2','G9|G3|G4'],is_target_observation=[True]*3))
    original = links.copy(deep=True)
    members = pd.DataFrame(dict(component_id=['a','b'],parent_id=['p','p']))
    out = group_observation_links(links, members)
    row = out.loc[out.canonical_id.eq('p')].iloc[0]
    assert len(out) == 2 and row.state_component_ids == 'a|b'
    assert row.source_canonical_ids == 'a|b' and row.molecular_components == 'G1|G2'
    assert row.evidence_role == 'state_component_readout'
    assert out.loc[out.canonical_id.eq('c'),'evidence_role'].iloc[0] == 'same_state_readout'
    assert out.loc[out.canonical_id.eq('c'),'molecular_components'].iloc[0] == 'G9|G3|G4'
    pd.testing.assert_frame_equal(links, original)


def test_unknown_group_rejected():
    with pytest.raises(ValueError, match='UNKNOWN_STATE_GROUP'):
        group_id('other')
