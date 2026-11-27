import numpy as np
import pandas as pd
import pytest
from cla_engine.measured_y_registry import (
    observation_id, verify_matrix, source_categories, normalize_links, condition_for_target,
)


def test_measurement_zero_constant_and_missing_are_distinct():
    values = pd.DataFrame({'zero':[0.,0.], 'missing':[np.nan,2.]}, index=['p1','p2'])
    verify_matrix(values, values.notna(), ['p1','p2'])
    bad = values.notna(); bad.loc['p1','zero'] = False
    with pytest.raises(ValueError, match='MASK_NOT_MEASURED'):
        verify_matrix(values, bad, ['p1','p2'])
    with pytest.raises(ValueError, match='PATIENT_ORDER'):
        verify_matrix(values, values.notna(), ['p2','p1'])


def test_source_category_does_not_inherit_state_or_enable_loss():
    readouts = pd.DataFrame({'source_target_id':['PROGENY:Hypoxia','Buffa'],
                             'display_category':['Pathway|State']*2})
    roles = pd.DataFrame({'source_target_id':['PROGENY:Hypoxia','Buffa'],
                         'display_category':['Pathway','State']})
    assert source_categories(readouts, roles).tolist() == ['Pathway','State']
    with pytest.raises(ValueError, match='UNKNOWN_SOURCE_ROLE'):
        source_categories(readouts, roles.iloc[:1])


def test_shared_measurement_has_one_identity_and_typed_membership():
    rp = observation_id('TCGA','member','PHOSPHO','AKT-pS473')
    ms = observation_id('CPTAC','member','PHOSPHO','AKT-pS473')
    assert rp != ms
    obs = pd.DataFrame({'observation_id':[rp]})
    targets = pd.DataFrame({'canonical_id':['k1','k2']})
    links = pd.DataFrame({'canonical_id':['k1','k1','k2'], 'observation_id':[rp]*3,
                          'evidence_role':['member_measurement']*2+['ambiguous_group_context'],
                          'condition_required':[False,False,True]})
    out = normalize_links(links,obs,targets)
    assert len(out)==2 and out.is_target_observation.tolist()==[True,False]
    assert not out.independent_loss_enabled.any()
    links.loc[0,'canonical_id']='wrong'
    with pytest.raises(ValueError, match='UNKNOWN_TARGET'):
        normalize_links(links,obs,targets)


def test_missing_condition_axis_cannot_enable_a_primary_mechanism():
    target = dict(canonical_id='variant', analysis_role='primary_pathway', condition_required=True)
    frame = pd.DataFrame({'variant': [1., 0., np.nan]}, index=['p1','p2','p3'])
    result = condition_for_target(frame,target,['p1','p2','p3'])
    assert result.eq(1).tolist() == [True,False,False]
    assert np.isnan(result.iloc[2])
    with pytest.raises(ValueError, match='MISSING_PRIMARY_CONDITION_AXIS'):
        condition_for_target(frame.drop(columns='variant'),target,list(frame.index))
    bad = frame.copy(); bad.iloc[0,0] = 2.
    with pytest.raises(ValueError, match='INVALID_CONDITION_VALUE'):
        condition_for_target(bad,target,list(frame.index))
    state = dict(canonical_id='state', analysis_role='state_auxiliary', condition_required=False)
    assert condition_for_target(frame,state,list(frame.index)).eq(1).all()
