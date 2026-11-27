import numpy as np
import pytest
from cla_engine.relation_baseline import target_balanced_weights,predict_numeric_head,resolve_state_families
from cla_engine.m0_relation import _strength_metrics


def test_balancing_prevents_large_targets_and_large_families_dominating():
    targets=['A','B','C'];families={'A':'one','B':'one','C':'two'}
    target=np.array([0,0,0,1,2,2,2,2]);observed=np.ones(8,bool);observed[-1]=False
    w=target_balanced_weights(target,observed,targets,families)
    assert w[-1]==0
    assert w[target==0].sum()==pytest.approx(.5)
    assert w[target==1].sum()==pytest.approx(.5)
    assert w[target==2].sum()==pytest.approx(1.)
    identities=[('BRCA',str(i),t) for i,t in enumerate(['A','A','B','B','C','C'])]
    _,mse=_strength_metrics(np.zeros(6),np.array([1,1,1,1,0,0]),np.ones(6,bool),identities,targets,families)
    assert mse==pytest.approx(.5)


def test_saved_numeric_coefficients_reproduce_prediction():
    head={'coef':np.array([[2.,-1.]]),'intercept':np.array([.1]),
          'numeric_mean':np.array([2.,10.]),'numeric_scale':np.array([2.,5.])}
    np.testing.assert_allclose(predict_numeric_head(head,np.array([[4.,15.],[2.,10.]])),[1.1,.1])


def test_progeny_method_family_does_not_reclassify_hypoxia_state():
    rows=[{'canonical_id':'H','state_id':'signifinder::Hypoxia_Buffa','state_family':'hypoxia'},
          {'canonical_id':'H','state_id':'PROGENY:Hypoxia','state_family':'pathway_footprint'}]
    assert resolve_state_families(rows,['H'])=={'H':'hypoxia'}
    with pytest.raises(ValueError,match='AMBIGUOUS_OR_MISSING'):
        resolve_state_families(rows[1:],['H'])
