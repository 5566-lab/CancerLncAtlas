import numpy as np
import pandas as pd
import pytest
from cla_engine.measured_members import annotation_features,bind_members,align_patients,coverage,condition_qualified_mask


def fixtures():
    members=pd.DataFrame({'canonical_id':['p1','p1','p2'],'ensembl_gene_id':['g1','g2','g1']})
    conditions=pd.DataFrame({'canonical_id':['p1','p2','p3'],'condition_required':[False,True,False],'condition_kind':['none','activating_mutation','none']})
    annotations=pd.DataFrame({'feature_id':['pan','pan','siteA','siteB'],'gene_id':['g1','g2','g1','g1'],'site':['','','S10','S20']})
    return members,conditions,annotations


def test_group_measurement_is_not_duplicated_into_individual_proteins():
    members,conditions,annotations=fixtures()
    features=annotation_features(annotations,['pan','siteA','siteB'])
    bindings=bind_members(members,features,conditions)
    pan=bindings[bindings.feature_id.eq('pan')].set_index('canonical_id')
    assert len(features)==3 and len(pan)==2
    assert pan.loc['p1','measurement_role']=='member_measurement'
    assert pan.loc['p2','measurement_role']=='ambiguous_group_context'
    assert not pan.gene_specific.any()
    assert not bindings.binding_establishes_mechanism_condition.any()


def test_site_identity_missing_patients_and_zero_measurements_survive():
    members,conditions,annotations=fixtures()
    features=annotation_features(annotations,['pan','siteA','siteB'])
    bindings=bind_members(members,features,conditions)
    original=pd.DataFrame({'pan':[0.,0.],'siteA':[1.,np.nan],'siteB':[np.nan,2.]},index=['a','b'])
    values,mask=align_patients(original,['b','missing','a'])
    assert values.columns.tolist()==['pan','siteA','siteB']
    assert mask.loc[['a','b'],'pan'].all() and not mask.loc['missing'].any()
    pd.testing.assert_frame_equal(values.loc[['a','b']].rename_axis(None),original)
    result=coverage(values,mask,bindings,conditions.canonical_id).set_index('canonical_id')
    assert result.loc['p1','measured_member_features']==3
    assert result.loc['p2','measured_member_features']==2
    assert result.loc['p3','patients_with_member_measurement']==0


def test_duplicate_axes_infinite_values_and_untyped_condition_are_rejected():
    with pytest.raises(ValueError,match='DUPLICATE'):
        align_patients(pd.DataFrame({'g':[1.]},index=['a']),['a','a'])
    with pytest.raises(ValueError,match='INFINITE'):
        align_patients(pd.DataFrame({'g':[np.inf]},index=['a']),['a'])
    members,conditions,annotations=fixtures()
    conditions['condition_required']=['False','True','False']
    with pytest.raises(ValueError,match='REQUIRES_BOOLEAN'):
        bind_members(members,annotation_features(annotations,['pan']),conditions)


def test_measured_value_does_not_establish_an_unknown_mechanism_condition():
    values=pd.DataFrame({'protein':[0.,5.,2.,np.nan]},index=['present','absent','unknown','unmeasured'])
    observed=values.notna()
    applicability=pd.Series([1.,0.,np.nan,1.],index=values.index)
    before=values.copy()
    eligible=condition_qualified_mask(observed,applicability)
    assert eligible.protein.tolist()==[True,False,False,False]
    pd.testing.assert_frame_equal(values,before)
    assert observed.loc['unknown','protein'] and observed.loc['absent','protein']
    with pytest.raises(ValueError,match='PATIENT_ORDER'):
        condition_qualified_mask(observed,applicability.iloc[::-1])
