
import numpy as np
from scripts.score_tcpa_cptac_paired_protein_pilot import standardize_train, observed_panel_mean, calibration_fit

def test_heldout_values_do_not_change_reference_or_calibration():
    raw=np.arange(30,dtype=float).reshape(10,3)
    train=np.arange(10)<7
    z,m,s,valid=standardize_train(raw,train)
    changed=raw.copy();changed[~train]=100000
    z2,m2,s2,valid2=standardize_train(changed,train)
    np.testing.assert_array_equal(m,m2);np.testing.assert_array_equal(s,s2)
    np.testing.assert_array_equal(valid,valid2)
    x,_=observed_panel_mean(z);x2,_=observed_panel_mean(z2)
    y=2+3*x
    assert calibration_fit(x,y,train)==calibration_fit(x2,y,train)
    a,b,n=calibration_fit(x,y,train)
    assert abs(a-2)<1e-12 and abs(b-3)<1e-12 and n==7

def test_missing_scores_remain_nan_and_constants_are_not_usable():
    values=np.array([[np.nan,np.nan],[2,np.nan],[2,4.]])
    mean,n=observed_panel_mean(values)
    assert np.isnan(mean[0]) and n[0]==0
    np.testing.assert_allclose(mean[1:],[2,3])
    z,_,_,valid=standardize_train(np.ones((6,2)),np.array([True]*4+[False]*2))
    assert not valid.any() and np.isnan(z).all()
